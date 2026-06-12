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
            state_feats = torch.cat(
                [obs[:, so.agent_state], obs[:, so.stimulus], obs[:, so.inventory]],
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
        self.compute_backend = "torch"
        self.compute_device = compute_device

        self.replay_buffer = _ChunkBuffer(chunk_capacity)
        self.reward_shaper = RewardShaper()

        self._mirror: Optional[TorchBrainMirror] = None
        self._steps: list[dict] = []  # current, not-yet-finalized chunk
        self._chunk_h0: Optional[np.ndarray] = None

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
            _, values_now, v_boot, _, _ = self._mirror.forward_sequence(
                obs, h0, boot_obs
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
