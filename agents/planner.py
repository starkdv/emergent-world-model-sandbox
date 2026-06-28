"""
Latent rollout planner — model-based action selection.

With a latent dynamics head, the brain can *imagine*: from the current hidden
state, simulate candidate action sequences entirely in latent space (no world
access, no rendering back to observations):

    repeat depth times:
        ẑ', r̂ = dynamics(h, a)        imagine the consequence
        h     = GRU(ẑ', h)            advance imagined memory
    score = Σ_k γ^k·r̂_k + γ^D·V(ẑ_D, h_D)   (value bootstrap at horizon)

Pick the first action of the best-scoring rollout, execute it, and (usually)
replan next tick — receding-horizon / model-predictive control.

Strategies (``brain.world_model.planner.strategy``):
  * ``shooting``        — original: first action uniform over the valid set,
                          continuation actions uniform-random (high variance).
  * ``policy_shooting`` — P1: continuation actions sampled from the brain's OWN
                          policy at the imagined state (Dreamer-style, lower
                          variance, in-distribution). Keep ``first_action:
                          uniform`` early so exploration is preserved.
  * ``cem``             — P2: categorical cross-entropy method. Maintain a
                          per-step action distribution; each iteration sample a
                          population of sequences, keep the top-scoring elites,
                          and refit the distribution toward them. Concentrates
                          the search budget on promising sequences.

Estimation:
  * ``lam`` (λ)    — P2: TD(λ) return over the imagined trajectory instead of a
                     single end-of-horizon bootstrap (λ=1 → the original
                     reward-sum + terminal bootstrap; λ<1 mixes intermediate
                     values, lower variance/bias trade-off).
  * ``normalize``  — z-score imagined rewards and values so per-step and horizon
                     terms are commensurable.
  * ``commit``     — control horizon: execute the best plan's first ``commit``
                     actions before replanning.

Off by default (``planner.enabled``); legacy ``shooting`` with the default knobs
reproduces the original controller exactly. NOT yet done (see
docs/PLANNING_PROPOSAL.md P2): model-error discipline via a dynamics ensemble,
which needs a genome change.

Author: Karan Vasa
Date: June 2026
"""

from typing import TYPE_CHECKING, Optional, Tuple, List, Callable

import numpy as np

if TYPE_CHECKING:
    from agents.brain import Brain


class _RunningStat:
    """Welford running mean/variance for cheap online normalisation."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def z(self, x: float, warmup: int = 20) -> float:
        """Z-score x once enough samples have been seen; else pass through."""
        if self.n < warmup:
            return x
        var = self.m2 / max(1, self.n - 1)
        std = var**0.5
        return (x - self.mean) / std if std > 1e-6 else (x - self.mean)


class LatentPlanner:
    """Random-shooting / policy-shooting / CEM planner over latent rollouts."""

    def __init__(
        self,
        depth: int = 3,
        samples: int = 16,
        gamma: float = 0.95,
        strategy: str = "shooting",
        first_action: str = "uniform",
        topk: int = 3,
        normalize: bool = False,
        commit: int = 1,
        lam: float = 1.0,
        cem_iters: int = 3,
        cem_elite_frac: float = 0.3,
    ):
        """
        Args:
            depth: Imagination horizon (steps).
            samples: Candidate sequences per decision (population size for CEM).
            gamma: Discount applied to imagined rewards.
            strategy: ``shooting`` | ``policy_shooting`` | ``cem``.
            first_action: ``uniform`` | ``policy`` | ``policy_topk`` (shooting /
                policy_shooting only).
            topk: top-k actions kept when ``first_action == policy_topk``.
            normalize: z-score rewards and values before summing.
            commit: control horizon — execute this many actions of the best plan
                before replanning (>=1; 1 = pure MPC).
            lam: TD(λ) parameter for the rollout return (1.0 = reward-sum +
                terminal bootstrap, the original; <1 mixes intermediate values).
            cem_iters: CEM refinement iterations (``cem`` only).
            cem_elite_frac: fraction of the population kept as elites per CEM
                iteration.
        """
        self.depth = max(1, int(depth))
        self.samples = max(1, int(samples))
        self.gamma = float(gamma)
        self.strategy = strategy
        self.first_action = first_action
        self.topk = max(1, int(topk))
        self.normalize = bool(normalize)
        self.commit = max(1, int(commit))
        self.lam = float(lam)
        self.cem_iters = max(1, int(cem_iters))
        self.cem_elite_frac = float(cem_elite_frac)
        self._policy_rollout = strategy == "policy_shooting"
        self._rstat = _RunningStat()
        self._vstat = _RunningStat()
        self._queue: List[int] = []

    @classmethod
    def from_config(cls, config: dict) -> "LatentPlanner":
        """Build from a ``brain.world_model.planner`` config dict."""
        config = config or {}
        return cls(
            depth=config.get("depth", 3),
            samples=config.get("samples", 16),
            gamma=config.get("gamma", 0.95),
            strategy=config.get("strategy", "shooting"),
            first_action=config.get("first_action", "uniform"),
            topk=config.get("topk", 3),
            normalize=config.get("normalize", False),
            commit=config.get("commit", 1),
            lam=config.get("lam", 1.0),
            cem_iters=config.get("cem_iters", 3),
            cem_elite_frac=config.get("cem_elite_frac", 0.3),
        )

    def plan(
        self,
        brain: "Brain",
        h: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Choose the next action (serving a committed plan if one is queued)."""
        if self._queue:
            return self._queue.pop(0)
        if self.strategy == "cem":
            seq = self._search_cem(brain, h, action_mask)
        else:
            seq, _ = self._search(brain, h, action_mask)
        if self.commit > 1 and len(seq) > 1:
            self._queue = [int(a) for a in seq[1 : self.commit]]
        return int(seq[0])

    # -- scoring -------------------------------------------------------------

    def _score(self, rewards: List[float], values: List[float]) -> float:
        """Discounted return from imagined rewards (+ value bootstrap / TD(λ))."""
        if self.lam >= 1.0:
            score = 0.0
            disc = 1.0
            for r in rewards:
                score += disc * r
                disc *= self.gamma
            return score + disc * values[-1]  # terminal bootstrap
        # forward-view TD(λ): G = r_k + γ[(1-λ)V_{k+1} + λ G]
        g = values[-1]
        for k in range(len(rewards) - 1, -1, -1):
            g = rewards[k] + self.gamma * ((1.0 - self.lam) * values[k] + self.lam * g)
        return g

    def _rollout_core(
        self,
        brain: "Brain",
        h: np.ndarray,
        action_provider: Callable[[int, np.ndarray], int],
    ) -> Tuple[List[int], float]:
        """Roll the dynamics ``depth`` steps; actions come from ``action_provider``."""
        h_sim = h
        z_sim = None
        rewards: List[float] = []
        values: List[float] = []
        seq: List[int] = []
        need_values = self.lam < 1.0
        for k in range(self.depth):
            a = int(action_provider(k, h_sim))
            seq.append(a)
            z_sim, r_pred = brain.predict_next_latent(h_sim, a)
            if self.normalize:
                self._rstat.update(r_pred)
                r_pred = self._rstat.z(r_pred)
            rewards.append(r_pred)
            h_sim = brain._gru_step(z_sim, h_sim)
            if need_values:
                v = brain._value(z_sim, h_sim)
                if self.normalize:
                    self._vstat.update(v)
                    v = self._vstat.z(v)
                values.append(v)
        if not need_values:  # only the terminal bootstrap is required
            vf = brain._value(z_sim, h_sim)
            if self.normalize:
                self._vstat.update(vf)
                vf = self._vstat.z(vf)
            values.append(vf)
        return seq, self._score(rewards, values)

    # -- shooting / policy_shooting -----------------------------------------

    def _first_actions(self, brain, h, action_mask, valid_first) -> np.ndarray:
        if self.first_action == "uniform" or not self._policy_rollout:
            return np.array(
                [int(np.random.choice(valid_first)) for _ in range(self.samples)]
            )
        probs = brain.policy_from_hidden(h, action_mask)
        if self.first_action == "policy_topk":
            idx = np.argsort(probs)[::-1][: self.topk]
            p = probs[idx]
            s = p.sum()
            p = p / s if s > 0 else np.ones(len(idx)) / len(idx)
            return np.array(
                [int(np.random.choice(idx, p=p)) for _ in range(self.samples)]
            )
        return np.array(
            [int(np.random.choice(len(probs), p=probs)) for _ in range(self.samples)]
        )

    def _search(
        self, brain: "Brain", h: np.ndarray, action_mask: Optional[np.ndarray]
    ) -> Tuple[List[int], float]:
        n_actions = brain.output_size
        valid_first = (
            np.flatnonzero(action_mask)
            if action_mask is not None
            else np.arange(n_actions)
        )
        if len(valid_first) == 0:
            valid_first = np.arange(n_actions)
        first_actions = self._first_actions(brain, h, action_mask, valid_first)

        best_seq: List[int] = [int(first_actions[0])]
        best_score = -np.inf
        for a0 in first_actions:
            a0 = int(a0)

            def provider(k, h_sim, _a0=a0):
                if k == 0:
                    return _a0
                if self._policy_rollout:
                    probs = brain.policy_from_hidden(h_sim)
                    return int(np.random.choice(len(probs), p=probs))
                return int(np.random.randint(n_actions))

            seq, score = self._rollout_core(brain, h, provider)
            if score > best_score:
                best_score = score
                best_seq = seq
        return best_seq, best_score

    # -- CEM -----------------------------------------------------------------

    def _search_cem(
        self, brain: "Brain", h: np.ndarray, action_mask: Optional[np.ndarray]
    ) -> List[int]:
        n = brain.output_size
        # per-step categorical distribution, init uniform
        dist = np.ones((self.depth, n), dtype=np.float64) / n
        if action_mask is not None and action_mask.sum() > 0:
            m = action_mask.astype(np.float64)
            dist[0] = m / m.sum()  # first step honours the valid-action mask
        n_elite = max(1, int(round(self.samples * self.cem_elite_frac)))

        best_seq = None
        for _ in range(self.cem_iters):
            seqs = np.empty((self.samples, self.depth), dtype=int)
            for d in range(self.depth):
                seqs[:, d] = np.random.choice(n, size=self.samples, p=dist[d])
            scored = []
            for i in range(self.samples):
                actions = seqs[i]

                def provider(k, _h, _acts=actions):
                    return int(_acts[k])

                _, score = self._rollout_core(brain, h, provider)
                scored.append((score, i))
            scored.sort(key=lambda si: si[0], reverse=True)
            elite_idx = [i for _, i in scored[:n_elite]]
            best_seq = [int(a) for a in seqs[elite_idx[0]]]
            # refit: smoothed elite action frequencies per step
            new = np.full((self.depth, n), 1e-3)
            for i in elite_idx:
                for d in range(self.depth):
                    new[d, seqs[i, d]] += 1.0
            if action_mask is not None and action_mask.sum() > 0:
                new[0] *= action_mask.astype(np.float64) + 1e-9
            dist = new / new.sum(axis=1, keepdims=True)
        return best_seq if best_seq is not None else [0]
