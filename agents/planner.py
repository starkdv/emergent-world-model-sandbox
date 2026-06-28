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

Two rollout strategies (``brain.world_model.planner.strategy``):
  * ``shooting``        — the original: first action drawn uniformly from the
                          valid set, continuation actions uniform-random. The
                          continuation is a random walk, so a rollout's score is
                          a high-variance estimate of its first action.
  * ``policy_shooting`` — Dreamer-style imagination (Planning proposal P1):
                          continuation actions are sampled from the brain's OWN
                          policy at the imagined hidden state, and the first
                          action is drawn from the policy (optionally top-k).
                          In-distribution, far lower variance.

Refinements (all default to the legacy behaviour so ``shooting`` is unchanged):
  * ``normalize``  — z-score imagined rewards and the value bootstrap by running
                     stats so the per-step and horizon terms are commensurable.
  * ``commit``     — control horizon: execute the best rollout's first ``commit``
                     actions before replanning (cheaper, more coherent).
  * ``first_action``/``topk`` — how the candidate first actions are chosen.

Off by default (``planner.enabled``): rollouts cost ``samples × depth`` extra
forward passes per decision, and a planner is only as good as its learned model.

See docs/PLANNING_PROPOSAL.md for the rationale and the staged plan.

Author: Karan Vasa
Date: June 2026
"""

from typing import TYPE_CHECKING, Optional, Tuple, List

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
    """Random-shooting / policy-shooting planner over imagined latent rollouts."""

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
    ):
        """
        Args:
            depth: Imagination horizon (steps).
            samples: Candidate action sequences per decision.
            gamma: Discount applied to imagined rewards.
            strategy: ``shooting`` (uniform continuations) or ``policy_shooting``
                (policy-sampled continuations).
            first_action: ``uniform`` | ``policy`` | ``policy_topk`` — how the
                candidate first actions are drawn.
            topk: top-k actions kept when ``first_action == policy_topk``.
            normalize: z-score rewards and value before summing.
            commit: control horizon — execute this many actions of the best plan
                before replanning (>=1; 1 = pure MPC, the default).
        """
        self.depth = max(1, int(depth))
        self.samples = max(1, int(samples))
        self.gamma = float(gamma)
        self.strategy = strategy
        self.first_action = first_action
        self.topk = max(1, int(topk))
        self.normalize = bool(normalize)
        self.commit = max(1, int(commit))
        # policy-guided continuations require a policy that ``shooting`` ignores
        self._policy_rollout = strategy == "policy_shooting"
        self._rstat = _RunningStat()
        self._vstat = _RunningStat()
        self._queue: List[int] = []  # committed actions awaiting execution

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

        seq, _ = self._search(brain, h, action_mask)
        if self.commit > 1 and len(seq) > 1:
            # cache the next (commit-1) imagined actions to replay before replanning
            self._queue = [int(a) for a in seq[1 : self.commit]]
        return int(seq[0])

    # -- internals -----------------------------------------------------------

    def _first_actions(self, brain, h, action_mask, valid_first) -> np.ndarray:
        """Sample the candidate first actions per the configured policy."""
        if self.first_action == "uniform" or not self._policy_rollout:
            # legacy: uniform over the valid set
            return np.array(
                [int(np.random.choice(valid_first)) for _ in range(self.samples)]
            )
        probs = brain.policy_from_hidden(h, action_mask)
        if self.first_action == "policy_topk":
            # keep only the top-k actions (renormalised), then sample
            idx = np.argsort(probs)[::-1][: self.topk]
            p = probs[idx]
            s = p.sum()
            p = p / s if s > 0 else np.ones(len(idx)) / len(idx)
            return np.array(
                [int(np.random.choice(idx, p=p)) for _ in range(self.samples)]
            )
        # "policy": sample first actions straight from the policy
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
            seq, score = self._rollout(brain, h, int(a0))
            if score > best_score:
                best_score = score
                best_seq = seq
        return best_seq, best_score

    def _rollout(
        self, brain: "Brain", h: np.ndarray, first_action: int
    ) -> Tuple[List[int], float]:
        """Imagine one action sequence; return (actions, discounted score)."""
        h_sim = h
        z_sim = None
        score = 0.0
        discount = 1.0
        action = first_action
        seq: List[int] = []
        for _ in range(self.depth):
            seq.append(int(action))
            z_sim, r_pred = brain.predict_next_latent(h_sim, action)
            if self.normalize:
                self._rstat.update(r_pred)
                score += discount * self._rstat.z(r_pred)
            else:
                score += discount * r_pred
            discount *= self.gamma
            h_sim = brain._gru_step(z_sim, h_sim)
            # next imagined action
            if self._policy_rollout:
                probs = brain.policy_from_hidden(h_sim)
                action = int(np.random.choice(len(probs), p=probs))
            else:
                action = int(np.random.randint(brain.output_size))

        v = brain._value(z_sim, h_sim)
        if self.normalize:
            self._vstat.update(v)
            score += discount * self._vstat.z(v)
        else:
            score += discount * v
        return seq, score
