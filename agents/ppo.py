"""
PPO sequence learner — full-network lifetime learning (Brain v3 Phase 3b).

Upgrades over the legacy AgentLearner (A2C, heads-only):

1. **Full-network backprop.** A persistent torch mirror of the brain's
   parameters is trained end-to-end with autograd — encoder/attention,
   GRU, and both heads all receive gradients. (The legacy learner only
   updated the output heads; the recurrent core could only evolve.)
2. **Sequence replay.** Experience is stored as ordered chunks of
   ``seq_len`` steps with the hidden state captured at chunk start, and
   the GRU is re-run over each chunk during learning. This replaces
   random single-transition replay, whose stored hidden states go stale
   and which cannot carry gradients through time.
3. **GAE(λ).** Generalised Advantage Estimation replaces raw TD(0)
   advantage: A_t = Σ_k (γλ)^k δ_{t+k}, trading a little bias for a
   large variance reduction.
4. **PPO clipping.** The clipped surrogate objective bounds each update:
   L = -E[min(r_t·Â_t, clip(r_t, 1-ε, 1+ε)·Â_t)], where
   r_t = π_new(a|s)/π_behaviour(a|s). Replayed data is slightly
   off-policy (and young agents act partly on instinct biases), so the
   ratio can drift from 1 — clipping prevents destructive updates.

Lamarckian inheritance is preserved: after every update the torch
parameters are written back through the brain's ParamSpec into the
genome, so offspring inherit learned weights.

Known approximations (documented trade-offs, standard in practice):
- Chunk-start hidden states are stored from acting time and go slightly
  stale as weights change ("stored state" strategy, cf. R2D2).
- The behaviour policy includes fading instinct biases and temperature;
  the learner's π_new is the raw masked network. The PPO clip bounds the
  resulting importance-ratio mismatch, which vanishes as instincts fade.

Requires torch; ``Agent.enable_learning(algorithm="ppo")`` falls back to
the legacy A2C learner when torch is unavailable.

Author: Karan Vasa
Date: June 2026
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment without torch
    torch = None
    TORCH_AVAILABLE = False

from utils.agents import RewardShaper

if TYPE_CHECKING:
    from agents.brain import Brain


# ---------------------------------------------------------------------------
# Sequence storage
# ---------------------------------------------------------------------------


@dataclass
class SequenceChunk:
    """
    A fixed-length, time-ordered slice of one agent's experience.

    Attributes:
        obs: (L, obs_dim) observations at decision time
        h0: (H,) GRU hidden state BEFORE the first step (from acting time)
        actions: (L,) action indices
        rewards: (L,) shaped rewards
        dones: (L,) 1.0 where the episode ended at that step
        logprobs: (L,) behaviour-policy log π(a_t|s_t) at acting time
        masks: (L, A) action-validity masks at decision time
        valid: (L,) 1.0 for real steps, 0.0 for padding (short final chunks)
        bootstrap_obs: (obs_dim,) observation after the last valid step,
            used to bootstrap V(s_L) in GAE (ignored when the last step
            is terminal)
    """

    obs: np.ndarray
    h0: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    logprobs: np.ndarray
    masks: np.ndarray
    valid: np.ndarray
    bootstrap_obs: np.ndarray


class _ChunkBuffer:
    """FIFO buffer of SequenceChunks with random sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._chunks: list[SequenceChunk] = []

    def add(self, chunk: SequenceChunk) -> None:
        self._chunks.append(chunk)
        if len(self._chunks) > self.capacity:
            self._chunks.pop(0)

    def sample(self, n: int) -> list[SequenceChunk]:
        idx = np.random.choice(
            len(self._chunks), size=min(n, len(self._chunks)), replace=False
        )
        return [self._chunks[i] for i in idx]

    def __len__(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_value: float,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation over one sequence.

    δ_t = r_t + γ·V(s_{t+1})·(1-done_t) - V(s_t)
    A_t = δ_t + γλ·(1-done_t)·A_{t+1}          (computed backwards)
    R_t = A_t + V(s_t)                          (value targets)

    Args:
        rewards: (L,) rewards
        values: (L,) V(s_t) estimates
        bootstrap_value: V(s_L) estimate for the state after the last step
        dones: (L,) terminal flags
        gamma: Discount factor
        lam: GAE λ (1.0 → Monte-Carlo-like, 0.0 → TD(0))

    Returns:
        (advantages (L,), value_targets (L,))
    """
    length = len(rewards)
    advantages = np.zeros(length, dtype=np.float64)
    next_value = bootstrap_value
    next_advantage = 0.0
    for t in reversed(range(length)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        next_advantage = delta + gamma * lam * not_done * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages, advantages + values


# ---------------------------------------------------------------------------
# Torch parameter mirror with version-aware functional forward
# ---------------------------------------------------------------------------


class TorchBrainMirror:
    """
    Persistent torch copy of a brain's parameters.

    Holds one ``torch.nn.Parameter`` per ParamSpec entry plus an Adam
    optimizer whose state survives across learn() calls. ``sync_to_brain``
    writes trained values back into the brain's numpy views (and thus,
    via ParamSpec.pack, into the genome).
    """

    def __init__(self, brain: "Brain", lr: float, device: str = "cpu"):
        self.device = torch.device(device)
        self.version = brain.spec.version
        self.params = {
            name: torch.nn.Parameter(
                torch.as_tensor(
                    np.asarray(arr, dtype=np.float32), device=self.device
                ).clone()
            )
            for name, arr in brain.named_params.items()
        }
        self.optimizer = torch.optim.Adam(self.params.values(), lr=lr)

        # Static structure info needed by the functional forward
        if self.version == 3:
            self.obs_spec = brain.obs_spec
            self.embed_dim = brain.embed_dim
            self.pos_enc = torch.as_tensor(brain.pos_enc, device=self.device)
        else:
            self.encoder_count = len(brain.encoder_layers)

    def matches(self, brain: "Brain") -> bool:
        """True if this mirror still corresponds to the brain's spec."""
        return (
            self.version == brain.spec.version
            and set(self.params.keys()) == set(brain.named_params.keys())
            and all(
                tuple(self.params[k].shape) == brain.named_params[k].shape
                for k in self.params
            )
        )

    def sync_to_brain(self, brain: "Brain") -> None:
        """Write torch parameters back into the brain's numpy views."""
        for name, p in self.params.items():
            brain.named_params[name][...] = p.detach().cpu().numpy()

    # -- functional forward ------------------------------------------------

    def _encode(self, obs: "torch.Tensor") -> "torch.Tensor":
        """Batched encode: (B, obs_dim) → latent z (B, Z)."""
        p = self.params
        if self.version == 3:
            so = self.obs_spec
            # +EXTRA slice (empty under v1, 6-wide under v2 / Brain v3.5)
            state_feats = torch.cat(
                [
                    obs[:, so.agent_state],
                    obs[:, so.stimulus],
                    obs[:, so.inventory],
                    obs[:, so.extra],
                ],
                dim=1,
            )
            s = torch.tanh(state_feats @ p["state_enc.W"] + p["state_enc.b"])

            rows, cols, feats = so.vision_shape
            n_tiles = rows * cols
            tiles = obs[:, so.vision].reshape(-1, n_tiles, feats)
            pos = self.pos_enc.unsqueeze(0).expand(tiles.shape[0], -1, -1)
            tokens = torch.cat([tiles, pos], dim=2)
            t = torch.tanh(tokens @ p["tile_embed.W"] + p["tile_embed.b"])

            q = s @ p["attn.Wq"]
            k = t @ p["attn.Wk"]
            v = t @ p["attn.Wv"]
            scores = torch.einsum("bte,be->bt", k, q) / np.sqrt(self.embed_dim)
            attn = torch.softmax(scores, dim=1)
            pooled = torch.einsum("bt,bte->be", attn, v)
            return torch.cat([s, pooled], dim=1)

        x = obs
        for i in range(self.encoder_count):
            x = torch.tanh(x @ p[f"encoder.{i}.W"] + p[f"encoder.{i}.b"])
        return x

    def _gru(self, z: "torch.Tensor", h: "torch.Tensor") -> "torch.Tensor":
        p = self.params
        r = torch.sigmoid(
            torch.clamp(
                z @ p["gru.Wr_input"] + h @ p["gru.Wr_hidden"] + p["gru.br"], -20, 20
            )
        )
        u = torch.sigmoid(
            torch.clamp(
                z @ p["gru.Wz_input"] + h @ p["gru.Wz_hidden"] + p["gru.bz"], -20, 20
            )
        )
        h_tilde = torch.tanh(
            z @ p["gru.Wh_input"] + (r * h) @ p["gru.Wh_hidden"] + p["gru.bh"]
        )
        return (1 - u) * h + u * h_tilde

    def _value(self, z: "torch.Tensor", h: "torch.Tensor") -> "torch.Tensor":
        p = self.params
        if self.version == 3:
            zh = torch.cat([z, h], dim=1)
            hidden = torch.tanh(zh @ p["value.W1"] + p["value.b1"])
            return (hidden @ p["value.W2"] + p["value.b2"]).squeeze(-1)
        return (h @ p["value.W"] + p["value.b"]).squeeze(-1)

    @property
    def has_world_model(self) -> bool:
        """True when the mirrored brain includes a dynamics head."""
        return "dyn.W1" in self.params

    def _dynamics(
        self, h: "torch.Tensor", actions_onehot: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Batched dynamics head: (h, onehot a) → (ẑ', r̂)."""
        p = self.params
        d = torch.tanh(
            torch.cat([h, actions_onehot], dim=1) @ p["dyn.W1"] + p["dyn.b1"]
        )
        z_pred = d @ p["dyn.Wz"] + p["dyn.bz"]
        r_pred = (d @ p["dyn.Wr"] + p["dyn.br"]).squeeze(-1)
        return z_pred, r_pred

    def forward_sequence(
        self,
        obs_seq: "torch.Tensor",
        h0: "torch.Tensor",
        bootstrap_obs: "torch.Tensor",
    ) -> tuple[
        "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"
    ]:
        """
        Run the recurrent network over time-ordered sequences.

        Args:
            obs_seq: (B, L, obs_dim) observations
            h0: (B, H) hidden state before the first step
            bootstrap_obs: (B, obs_dim) observation after the last step

        Returns:
            (logits (B, L, A), values (B, L), bootstrap_values (B,),
            latents zs (B, L+1, Z) including the bootstrap latent,
            hiddens hs (B, L, H)) — the extra tensors feed the
            world-model auxiliary loss.
        """
        p = self.params
        batch, length, _ = obs_seq.shape
        h = h0
        logits_steps = []
        value_steps = []
        z_steps = []
        h_steps = []
        for t in range(length):
            z = self._encode(obs_seq[:, t, :])
            h = self._gru(z, h)
            logits_steps.append(h @ p["policy.W"] + p["policy.b"])
            value_steps.append(self._value(z, h))
            z_steps.append(z)
            h_steps.append(h)

        # Bootstrap value of the state after the final step
        z_boot = self._encode(bootstrap_obs)
        h_boot = self._gru(z_boot, h)
        v_boot = self._value(z_boot, h_boot)
        z_steps.append(z_boot)

        return (
            torch.stack(logits_steps, dim=1),
            torch.stack(value_steps, dim=1),
            v_boot,
            torch.stack(z_steps, dim=1),
            torch.stack(h_steps, dim=1),
        )

    def multistep_errors(
        self,
        hs: "torch.Tensor",
        zs: "torch.Tensor",
        actions: "torch.Tensor",
        rewards: "torch.Tensor",
        valid: "torch.Tensor",
        dones: "torch.Tensor",
        k: int,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Open-loop k-step rollout errors of the dynamics head vs real data.

        From every real hidden state h_t, roll the dynamics forward k steps
        feeding its own predictions back (ẑ → GRU → ĥ), always taking the
        REAL logged actions, and compare each predicted latent / reward with
        the real encoded latent / observed reward j steps ahead. This is
        exactly the compounding-error regime the planner operates in, so it
        is both the model-quality diagnostic (under no_grad) and the
        multi-step training loss (with grad).

        Args:
            hs: (B, L, H) real hidden states
            zs: (B, L+1, Z) real encoded latents (incl. bootstrap)
            actions: (B, L) logged action indices
            rewards: (B, L) observed rewards
            valid: (B, L) 1 where the step is real (not padding)
            dones: (B, L) 1 where the episode ended at that step
            k: rollout horizon (clamped to L)

        Returns:
            (latent_mse, reward_mse) — tensors of shape (k,), entry j the
            mean error at horizon j+1 over all windows that stay inside the
            chunk and cross no terminal/padding boundary.
        """
        batch, length, h_size = hs.shape
        k = max(1, min(int(k), length))
        n_actions = int(self.params["policy.b"].shape[0])
        n_win = length - k + 1  # windows start at t = 0..L-k
        cur = hs[:, :n_win, :].reshape(batch * n_win, h_size)
        ok = valid * (1.0 - dones)  # step is real and successor in-episode
        win_ok = torch.ones(batch * n_win, device=hs.device)
        lat_errs, rew_errs = [], []
        for j in range(k):
            act = actions[:, j : j + n_win].reshape(-1)
            onehot = torch.nn.functional.one_hot(act, n_actions).float()
            z_pred, r_pred = self._dynamics(cur, onehot)
            win_ok = win_ok * ok[:, j : j + n_win].reshape(-1)
            n = torch.clamp(win_ok.sum(), min=1.0)
            z_tgt = zs[:, j + 1 : j + 1 + n_win, :].reshape(batch * n_win, -1).detach()
            r_tgt = rewards[:, j : j + n_win].reshape(-1)
            lat_errs.append((((z_pred - z_tgt) ** 2).mean(dim=1) * win_ok).sum() / n)
            rew_errs.append(((r_pred - r_tgt) ** 2 * win_ok).sum() / n)
            cur = self._gru(z_pred, cur)
        return torch.stack(lat_errs), torch.stack(rew_errs)

    def imagine_loss(
        self,
        h0: "torch.Tensor",
        horizon: int,
        gamma: float,
        lam: float,
        entropy_coef: float,
    ) -> "torch.Tensor":
        """
        Dreamer-style actor-critic in imagination (Planning proposal P3).

        From a batch of (detached) start hidden states, roll the *actor* forward
        ``horizon`` steps entirely in the latent world model, score the imagined
        trajectory with TD(λ) returns from the critic, and return a loss that
        trains the actor (REINFORCE with a value baseline) and the critic
        (regression to the imagined returns). The dynamics/encoder get no
        gradient here (returns are detached) — they are trained by the
        world-model auxiliary loss; this only distils planning into the policy.

        Requires the brain to have a world model (``has_world_model``).
        """
        p = self.params
        n_actions = p["policy.b"].shape[0]
        h = h0
        logps, ents, vals, rews = [], [], [], []
        for _ in range(horizon):
            logits = h @ p["policy.W"] + p["policy.b"]
            logp_all = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            a = torch.multinomial(probs, 1).squeeze(-1)  # sample (no grad thru sample)
            logp = logp_all.gather(1, a.unsqueeze(-1)).squeeze(-1)
            ent = -(probs * torch.clamp(logp_all, min=-30.0)).sum(-1)
            onehot = torch.nn.functional.one_hot(a, n_actions).float()
            z, r = self._dynamics(h, onehot)
            h = self._gru(z, h)
            v = self._value(z, h)
            logps.append(logp)
            ents.append(ent)
            rews.append(r)
            vals.append(v)
        vals_t = torch.stack(vals, dim=1)  # (M, H)
        rews_t = torch.stack(rews, dim=1)
        logp_t = torch.stack(logps, dim=1)
        ent_t = torch.stack(ents, dim=1)
        # forward-view TD(λ) returns (detached — used as targets only)
        with torch.no_grad():
            ret = torch.zeros_like(vals_t)
            g = vals_t[:, -1]
            for t in range(horizon - 1, -1, -1):
                g = rews_t[:, t] + gamma * ((1.0 - lam) * vals_t[:, t] + lam * g)
                ret[:, t] = g
        adv = (ret - vals_t).detach()
        actor_loss = -(logp_t * adv).mean()
        critic_loss = 0.5 * ((vals_t - ret) ** 2).mean()
        ent_bonus = ent_t.mean()
        return actor_loss + critic_loss - entropy_coef * ent_bonus


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


class PPOSequenceLearner:
    """
    PPO learner over experience sequences, with full-network backprop.

    Exposes the same surface the Agent/World code relies on
    (``store_experience``-style storage, ``learn(brain)``,
    ``replay_buffer``/``batch_size`` for the world's training scheduler,
    ``reward_shaper`` for reward computation).
    """

    # Marker the Agent checks to route decide_with_logprob + step storage
    wants_sequences = True
    algorithm = "ppo"

    def __init__(
        self,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.95,
        batch_size: int = 8,  # chunks per update
        seq_len: int = 8,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 2,
        grad_clip: float = 0.5,
        chunk_capacity: int = 64,
        compute_device: str = "cpu",
        world_model_coef: float = 1.0,
        imagination: Optional[dict] = None,
        world_model_multistep: Optional[dict] = None,
        rollout_metric_k: int = 3,
    ):
        """
        Initialize the PPO learner.

        Args:
            learning_rate: Adam learning rate
            discount_factor: Discount factor γ
            batch_size: Number of chunks sampled per learn() call
            seq_len: Steps per stored sequence chunk (L)
            gae_lambda: GAE λ
            clip_epsilon: PPO clip range ε
            value_coef: Weight of the value loss
            entropy_coef: Weight of the entropy bonus
            epochs: Optimisation passes over the sampled batch
            grad_clip: Max gradient norm
            chunk_capacity: Max chunks kept in the buffer
            compute_device: torch device ("cpu", "cuda", "mps")
            world_model_coef: Weight of the dynamics-head auxiliary loss
                (only used when the brain has a world model)
            world_model_multistep: Optional ``{k, coef}`` — train the dynamics
                head on k-step open-loop rollouts (horizons 2..k) in addition
                to the 1-step loss; off when k < 2 (legacy behaviour)
            rollout_metric_k: Horizon of the k-step rollout-error diagnostic
                computed every learn() (0 disables); exposed as
                ``wm_rollout_error`` (per-horizon) / ``wm_rollout_error_ema``
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PPOSequenceLearner requires torch; use algorithm='a2c' instead"
            )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.world_model_coef = world_model_coef
        # Dreamer-style imagination actor-critic (Planning proposal P3). Off by
        # default; trains the actor/critic on rollouts imagined in the latent
        # world model, so a good policy emerges without per-tick planning.
        im = imagination or {}
        self.imag_enabled = bool(im.get("enabled", False))
        self.imag_horizon = max(1, int(im.get("horizon", 5)))
        self.imag_weight = float(im.get("weight", 0.5))
        self.imag_batch = max(1, int(im.get("batch", 256)))
        self.imag_lambda = float(im.get("lambda", 0.95))
        self.imag_entropy = float(im.get("entropy", 0.001))
        # Warmup: don't imagine until the world model has trained for this many
        # world ticks (the agent sets ``current_tick`` before learn()).
        self.imag_warmup = max(0, int(im.get("warmup_ticks", 0)))
        # Readiness gate: additionally require the measured k-step rollout
        # error EMA to be below this threshold (0 = tick gate only).
        self.imag_warmup_error = float(im.get("warmup_error", 0.0))
        self.current_tick = 0
        # Multi-step world-model training loss: train the dynamics head on
        # k-step open-loop rollouts (feeding predictions back), not just
        # 1-step prediction — the regime the planner actually uses it in.
        # Off by default (k < 2) to preserve legacy training exactly.
        ms = world_model_multistep or {}
        self.ms_k = max(0, int(ms.get("k", 0)))
        self.ms_coef = float(ms.get("coef", 0.5))
        # k-step rollout-error diagnostic (no_grad, logged every learn()):
        # per-horizon latent MSE + an EMA at horizon k for readiness gating.
        self.rollout_metric_k = max(0, int(rollout_metric_k))
        self.wm_rollout_error: Optional[list] = None
        self.wm_rollout_error_ema: Optional[float] = None
        self._wm_ema_beta = 0.9
        self.compute_backend = "torch"
        self.compute_device = compute_device

        self.replay_buffer = _ChunkBuffer(chunk_capacity)
        self.reward_shaper = RewardShaper()

        self._mirror: Optional[TorchBrainMirror] = None
        self._steps: list[dict] = []  # current, not-yet-finalized chunk
        self._chunk_h0: Optional[np.ndarray] = None

    def imagination_active(self) -> bool:
        """True when the imagination loss should run now: enabled, past the
        tick warmup, and (when a readiness threshold is set) the measured
        k-step rollout error has dropped below it."""
        if not (self.imag_enabled and self.current_tick >= self.imag_warmup):
            return False
        if self.imag_warmup_error > 0.0:
            ema = self.wm_rollout_error_ema
            return ema is not None and ema <= self.imag_warmup_error
        return True

    # -- storage -----------------------------------------------------------

    def store_step(
        self,
        observation: np.ndarray,
        hidden_before: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        logprob: float,
        action_mask: np.ndarray,
    ) -> None:
        """
        Append one time-ordered step; finalizes a chunk every seq_len
        steps (or immediately on terminal steps).

        Args:
            observation: Observation at decision time
            hidden_before: GRU hidden state before this step
            action: Action index taken
            reward: Shaped reward received
            next_observation: Observation after the action
            done: Whether the episode ended at this step
            logprob: Behaviour-policy log π(a|s) at acting time
            action_mask: Action-validity mask at decision time
        """
        if not self._steps:
            self._chunk_h0 = np.asarray(hidden_before, dtype=np.float32).copy()

        self._steps.append(
            {
                "obs": np.asarray(observation, dtype=np.float32),
                "action": int(action),
                "reward": float(reward),
                "next_obs": np.asarray(next_observation, dtype=np.float32),
                "done": bool(done),
                "logprob": float(logprob),
                "mask": np.asarray(action_mask, dtype=np.float32),
            }
        )

        if len(self._steps) >= self.seq_len or done:
            self._finalize_chunk()

    def mark_done(self, terminal_observation: np.ndarray) -> None:
        """
        Flag the most recent stored step as terminal (agent died) and
        finalize the in-progress chunk.

        Args:
            terminal_observation: Final observation at death
        """
        if not self._steps:
            return
        self._steps[-1]["done"] = True
        self._steps[-1]["next_obs"] = np.asarray(terminal_observation, dtype=np.float32)
        self._finalize_chunk()

    def _finalize_chunk(self) -> None:
        """Pad the current steps to seq_len and push a SequenceChunk."""
        steps = self._steps
        if not steps:
            return
        length = self.seq_len
        obs_dim = steps[0]["obs"].shape[0]
        n_actions = steps[0]["mask"].shape[0]

        chunk = SequenceChunk(
            obs=np.zeros((length, obs_dim), dtype=np.float32),
            h0=self._chunk_h0,
            actions=np.zeros(length, dtype=np.int64),
            rewards=np.zeros(length, dtype=np.float32),
            dones=np.ones(length, dtype=np.float32),  # padding counts as done
            logprobs=np.zeros(length, dtype=np.float32),
            masks=np.ones((length, n_actions), dtype=np.float32),
            valid=np.zeros(length, dtype=np.float32),
            bootstrap_obs=steps[-1]["next_obs"],
        )
        for i, s in enumerate(steps[:length]):
            chunk.obs[i] = s["obs"]
            chunk.actions[i] = s["action"]
            chunk.rewards[i] = s["reward"]
            chunk.dones[i] = 1.0 if s["done"] else 0.0
            chunk.logprobs[i] = s["logprob"]
            chunk.masks[i] = s["mask"]
            chunk.valid[i] = 1.0

        self.replay_buffer.add(chunk)
        self._steps = []
        self._chunk_h0 = None

    # -- learning ----------------------------------------------------------

    def learn(self, brain: "Brain") -> float:
        """
        Run PPO updates on a sampled batch of sequence chunks.

        Full-network backprop: gradients flow through the policy and
        value heads, the GRU (through time within each chunk), and the
        encoder/attention layers. Finishes with a Lamarckian sync of the
        trained weights back into the genome.

        Args:
            brain: Agent's brain to update

        Returns:
            Mean total loss over the final epoch (0.0 if not enough data)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        if self._mirror is None or not self._mirror.matches(brain):
            self._mirror = TorchBrainMirror(
                brain, lr=self.learning_rate, device=self.compute_device
            )

        chunks = self.replay_buffer.sample(self.batch_size)
        device = self._mirror.device

        obs = torch.as_tensor(np.stack([c.obs for c in chunks]), device=device)
        h0 = torch.as_tensor(np.stack([c.h0 for c in chunks]), device=device)
        boot_obs = torch.as_tensor(
            np.stack([c.bootstrap_obs for c in chunks]), device=device
        )
        actions = torch.as_tensor(np.stack([c.actions for c in chunks]), device=device)
        old_logprobs = torch.as_tensor(
            np.stack([c.logprobs for c in chunks]), device=device
        )
        masks = torch.as_tensor(np.stack([c.masks for c in chunks]), device=device)
        valid = torch.as_tensor(np.stack([c.valid for c in chunks]), device=device)
        dones = torch.as_tensor(np.stack([c.dones for c in chunks]), device=device)
        rewards_t = torch.as_tensor(
            np.stack([c.rewards for c in chunks]), device=device
        )
        n_valid = torch.clamp(valid.sum(), min=1.0)

        # Advantages/targets from the CURRENT network (recomputed once,
        # before the optimisation epochs — standard PPO practice).
        with torch.no_grad():
            _, values_now, v_boot, zs_ng, hs_ng = self._mirror.forward_sequence(
                obs, h0, boot_obs
            )
            # Model-quality diagnostic: k-step open-loop rollout error on the
            # sampled real sequences — the quantity the planner's usefulness
            # actually depends on. Cheap (no_grad) and observable via the
            # metrics CSV / readiness gating.
            if self._mirror.has_world_model and self.rollout_metric_k > 0:
                lat_err, _ = self._mirror.multistep_errors(
                    hs_ng, zs_ng, actions, rewards_t, valid, dones,
                    self.rollout_metric_k,
                )
                errs = [float(e) for e in lat_err]
                self.wm_rollout_error = errs
                ema = self.wm_rollout_error_ema
                self.wm_rollout_error_ema = (
                    errs[-1]
                    if ema is None
                    else self._wm_ema_beta * ema
                    + (1.0 - self._wm_ema_beta) * errs[-1]
                )
        advantages_np = np.zeros((len(chunks), self.seq_len), dtype=np.float32)
        targets_np = np.zeros((len(chunks), self.seq_len), dtype=np.float32)
        values_now_np = values_now.cpu().numpy()
        v_boot_np = v_boot.cpu().numpy()
        for i, c in enumerate(chunks):
            last_done = (
                bool(c.dones[c.valid.astype(bool)][-1]) if c.valid.any() else True
            )
            boot = 0.0 if last_done else float(v_boot_np[i])
            adv, tgt = compute_gae(
                c.rewards,
                values_now_np[i],
                boot,
                c.dones,
                self.discount_factor,
                self.gae_lambda,
            )
            advantages_np[i] = adv
            targets_np[i] = tgt

        advantages = torch.as_tensor(advantages_np, device=device)
        targets = torch.as_tensor(targets_np, device=device)
        # Normalise advantages over valid steps (variance reduction)
        adv_mean = (advantages * valid).sum() / n_valid
        adv_std = torch.sqrt(
            ((advantages - adv_mean) ** 2 * valid).sum() / n_valid + 1e-8
        )
        advantages = (advantages - adv_mean) / adv_std

        total_loss = 0.0
        for _ in range(self.epochs):
            logits, values, _, zs, hs = self._mirror.forward_sequence(obs, h0, boot_obs)
            logits = logits.masked_fill(masks <= 0, -1e9)
            log_probs_all = torch.log_softmax(logits, dim=-1)
            new_logprobs = torch.gather(
                log_probs_all, 2, actions.unsqueeze(-1)
            ).squeeze(-1)

            # PPO clipped surrogate
            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * advantages
            )
            policy_loss = -(torch.min(surr1, surr2) * valid).sum() / n_valid

            value_loss = (0.5 * (values - targets) ** 2 * valid).sum() / n_valid

            probs = torch.softmax(logits, dim=-1)
            entropy = (
                -(probs * torch.clamp(log_probs_all, min=-30.0)).sum(dim=-1) * valid
            ).sum() / n_valid

            loss = (
                policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            )

            # World-model auxiliary loss: predict the NEXT latent and the
            # reward from (h_t, a_t). Targets are detached (stop-gradient)
            # so the encoder cannot collapse the latent space to make
            # prediction trivially easy — the policy/value losses anchor
            # the representation; the dynamics head chases it.
            if self._mirror.has_world_model:
                batch_n, length, h_size = hs.shape
                onehot = torch.nn.functional.one_hot(
                    actions.reshape(-1), num_classes=logits.shape[-1]
                ).float()
                z_pred, r_pred = self._mirror._dynamics(hs.reshape(-1, h_size), onehot)
                z_target = zs[:, 1:, :].reshape(batch_n * length, -1).detach()
                # Steps whose successor crosses a terminal/padding boundary
                # are excluded via the valid/done masks
                wm_mask = (valid * (1.0 - dones)).reshape(-1)
                wm_n = torch.clamp(wm_mask.sum(), min=1.0)
                wm_latent = (
                    ((z_pred - z_target) ** 2).mean(dim=1) * wm_mask
                ).sum() / wm_n
                wm_reward = (
                    (r_pred - rewards_t.reshape(-1)) ** 2 * wm_mask
                ).sum() / wm_n
                loss = loss + self.world_model_coef * (wm_latent + wm_reward)

                # Multi-step consistency loss (opt-in): the planner rolls the
                # dynamics open-loop for several steps, but the 1-step loss
                # above never trains that regime — compounding error is
                # unconstrained. Horizon 1 is already covered above, so only
                # horizons 2..k contribute here.
                if self.ms_k >= 2:
                    ms_lat, ms_rew = self._mirror.multistep_errors(
                        hs, zs, actions, rewards_t, valid, dones, self.ms_k
                    )
                    loss = loss + self.ms_coef * (
                        ms_lat[1:].mean() + ms_rew[1:].mean()
                    )

                # Dreamer-style imagination actor-critic (P3, opt-in). Start from
                # detached valid hidden states so it distils planning into the
                # policy without perturbing representation learning.
                if self.imagination_active():
                    flat_h = hs.reshape(-1, hs.shape[-1])
                    keep = valid.reshape(-1) > 0
                    flat_h = flat_h[keep]
                    if flat_h.shape[0] > 0:
                        if flat_h.shape[0] > self.imag_batch:
                            sel = torch.randperm(flat_h.shape[0])[: self.imag_batch]
                            flat_h = flat_h[sel]
                        imag_loss = self._mirror.imagine_loss(
                            flat_h.detach(),
                            self.imag_horizon,
                            self.discount_factor,
                            self.imag_lambda,
                            self.imag_entropy,
                        )
                        loss = loss + self.imag_weight * imag_loss

            self._mirror.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._mirror.params.values(), self.grad_clip)
            self._mirror.optimizer.step()
            total_loss = float(loss.detach().cpu().item())

        # Write trained weights back to the brain, then Lamarckian-sync
        # them into the genome so offspring inherit them.
        self._mirror.sync_to_brain(brain)
        brain.genome.weights = brain.spec.pack(brain.named_params)

        return total_loss
