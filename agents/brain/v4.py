"""
Brain v4 — multi-timescale memory, episodic place memory, dual-discount critic.

v4 keeps every part of v3.5 that works (the shared tile embedding, the
state-conditioned attention pool, the GRU, the [z, h] critic, the latent
dynamics head) and adds the four things the v3.5 baseline campaign showed
missing (docs/BRAIN_V4_PROPOSAL.md §4, docs/sample_v35_emergence_baseline):

1. **A slow recurrent core with evolved time constants.** The fast GRU is
   joined by ``H_s`` leaky units whose per-unit leak rate lives in the genome:

       alpha = sigmoid(rho),  tau = 1 / alpha
       u_t   = tanh( W_f h^f_t + W_s h^s_{t-1} + b_s )
       h^s_t = (1 - alpha) (.) h^s_{t-1} + alpha (.) u_t

   The gradient along that recurrence decays like ``exp(-k / tau)`` instead of
   the GRU's ``~2^-k``, so a unit at ``tau = 300`` still carries 0.59 of the
   gradient across the ~160 ticks between planting a seed and eating its
   fruit. ``rho`` mutates and is selected, so the population evolves its own
   memory horizons.

2. **An episodic place memory.** ``M`` slots, each holding a latent snapshot,
   an egocentric displacement kept exact by integer path integration, a
   salience and an age. Read by a second attention pool whose query comes
   from ``[s || h^s]``. This is what makes "go back to the patch" and "go back
   to where I planted" *representable* — a 5x5 window never can.

3. **A dual-discount critic.** One value MLP, two outputs: gamma 0.95 for
   local control and gamma 0.999 for the 1000-tick horizon that cultivation
   and the seasons live on. The learner mixes their advantages.

4. **Heads for the society layer.** A continuous ``C``-channel communication
   vector, the evolved drive weights (agents/scoring.py and the reward
   shaper read these), and a visible identity tag.

The recurrent *state* is packed into one flat vector so every caller that
treats ``h`` as an opaque array — the agent, the parallel pipeline, the PPO
chunk buffer, the checkpointer — keeps working:

    h = [ h_fast (H_f) | h_slow (H_s) | slots (M x SLOT_WIDTH) ]

Slot layout is ``[ z_pre (S+E) | d_right | d_ahead | salience | age ]``; a
salience of exactly 0 marks an empty slot, which the read masks out.

Author: Karan Vasa
Date: September 2026
"""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from agents.actions import Action
from agents.brain import modules
from agents.brain.instincts import InstinctModule
from agents.brain.v3 import BrainV3, make_positional_encoding
from agents.brain.spec import (
    COMM_CHANNELS,
    DRIVE_BETA_INDEX,
    MEMORY_SLOT_EXTRA,
    ObservationSpec,
    OBSERVATION_SPEC_V4,
    build_brain_v4_param_spec,
    build_nested_params_v4,
)

if TYPE_CHECKING:
    from agents.genome import Genome


# Per-tick multiplicative decay of a stored slot's salience, so an old strong
# memory can eventually be displaced by a new one instead of locking a slot
# forever.
SALIENCE_DECAY = 0.999

# A new memory must beat the weakest stored one by this factor to claim its
# slot. Without a margin the slots churn every tick: a homeostatic reward is
# small but never exactly zero, so "better than the weakest" is satisfied by
# the ordinary metabolic trickle and the memory only ever holds the last M
# ticks — which is what the GRU is for. With it, the slots settle after the
# initial fill and only a genuinely more salient event displaces one.
WRITE_MARGIN = 1.5

# Timescale of the recency feature in the slot token: recency = exp(-age / T).
RECENCY_SCALE = 200.0

# Drive weights are mutated without bound; they are clipped at read time so a
# random walk cannot turn the reward function into a divergence.
DRIVE_CLIP = 4.0


class BrainV4(BrainV3):
    """
    Multi-timescale attention brain (Brain v4).

    Inherits perception (tile embedding + state-conditioned attention) from
    BrainV3 and replaces the memory core, the critic and the heads.

    Attributes (in addition to BrainV3's):
        slow_hidden_size (int): Size of the slow leaky core (H_s); 0 disables
        memory_slots (int): Episodic slots (M); 0 disables the place memory
        comm_channels (int): Width of the communication vector (C)
        slot_width (int): Floats per slot in the packed state
    """

    VERSION = 4

    def __init__(
        self,
        genome: "Genome",
        embed_dim: int = 8,
        state_dim: int = 40,
        gru_hidden_size: int = 48,
        slow_hidden_size: int = 24,
        value_hidden: int = 16,
        output_size: int = 9,
        memory_slots: int = 8,
        comm_channels: int = COMM_CHANNELS,
        instincts: Optional[InstinctModule] = None,
        obs_spec: Optional[ObservationSpec] = None,
        world_model_hidden: Optional[int] = None,
    ):
        """
        Initialize Brain v4 from genome.

        Args:
            genome: Genome containing the network weights
            embed_dim: Per-tile and per-slot embedding size (E)
            state_dim: State encoder output size (S)
            gru_hidden_size: Fast GRU hidden size (H_f)
            slow_hidden_size: Slow leaky core size (H_s); 0 = no slow core
            value_hidden: Hidden size of the value MLP
            output_size: Number of discrete actions
            memory_slots: Episodic memory slots (M); 0 = no place memory
            comm_channels: Communication vector width (C)
            instincts: Instinct module (default: standard InstinctModule)
            obs_spec: Observation layout (default: the active v4 spec)
            world_model_hidden: Dynamics-head hidden width (None = none)
        """
        self.genome = genome
        self.obs_spec = obs_spec if obs_spec is not None else OBSERVATION_SPEC_V4
        self.input_size = self.obs_spec.size
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.gru_hidden_size = gru_hidden_size
        self.slow_hidden_size = max(0, int(slow_hidden_size))
        self.value_hidden = value_hidden
        self.output_size = output_size
        self.memory_slots = max(0, int(memory_slots))
        self.comm_channels = comm_channels
        self.world_model_hidden = world_model_hidden
        self.instincts = instincts if instincts is not None else InstinctModule()

        spec_o = self.obs_spec
        self.state_inputs = (
            (spec_o.agent_state.stop - spec_o.agent_state.start)
            + (spec_o.stimulus.stop - spec_o.stimulus.start)
            + (spec_o.inventory.stop - spec_o.inventory.start)
            + (spec_o.extra.stop - spec_o.extra.start)
        )
        self.pos_enc = make_positional_encoding(spec_o.vision_shape)

        # Derived widths used everywhere below.
        self.latent_pre = state_dim + embed_dim  # [s || attended vision]
        self.latent_size = self.latent_pre + embed_dim  # + memory read
        self.core_size = gru_hidden_size + self.slow_hidden_size
        self.slot_width = self.latent_pre + MEMORY_SLOT_EXTRA
        self.state_size = self.core_size + self.memory_slots * self.slot_width

        self.spec = build_brain_v4_param_spec(
            state_inputs=self.state_inputs,
            embed_dim=embed_dim,
            state_dim=state_dim,
            gru_hidden_size=gru_hidden_size,
            slow_hidden_size=self.slow_hidden_size,
            value_hidden=value_hidden,
            output_size=output_size,
            world_model_hidden=world_model_hidden,
            comm_channels=comm_channels,
        )
        self.named_params = self.spec.unpack(genome.weights)
        self.params = self._build_nested(self.named_params)

    def _build_nested(self, named: dict) -> dict:
        """Build the v4 nested params structure."""
        return build_nested_params_v4(named)

    # -- packed recurrent state -------------------------------------------

    def initial_state(self) -> np.ndarray:
        """Zeroed packed state: fast core, slow core, and empty slots."""
        return np.zeros(self.state_size, dtype=np.float32)

    def split_state(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unpack the recurrent state.

        Args:
            h: Packed state of length ``state_size``

        Returns:
            (h_fast, h_slow, slots) where slots is (M, slot_width); the slot
            array is a view, so in-place edits land back in ``h``.
        """
        h = np.asarray(h, dtype=np.float32)
        hf = h[: self.gru_hidden_size]
        hs = h[self.gru_hidden_size : self.core_size]
        slots = h[self.core_size :].reshape(self.memory_slots, self.slot_width)
        return hf, hs, slots

    def pack_state(
        self, h_fast: np.ndarray, h_slow: np.ndarray, slots: np.ndarray
    ) -> np.ndarray:
        """Pack fast core, slow core and slots back into one flat state."""
        return np.concatenate(
            [
                np.asarray(h_fast, dtype=np.float32).ravel(),
                np.asarray(h_slow, dtype=np.float32).ravel(),
                np.asarray(slots, dtype=np.float32).ravel(),
            ]
        ).astype(np.float32)

    # -- perception --------------------------------------------------------

    def _encode_pre(self, observation: np.ndarray) -> np.ndarray:
        """
        The v3 latent: ``[state || attention-pooled vision]``.

        Identical to BrainV3._encode; kept separate because v4 needs it both
        as the GRU input prefix and as what a memory slot stores.
        """
        return BrainV3._encode(self, observation)

    def _read_memory(self, slots: np.ndarray, s: np.ndarray, h_slow: np.ndarray):
        """
        Attention read over the episodic slots.

        Empty slots (salience exactly 0) are masked out of the softmax; when
        every slot is empty the read is all zeros, which is exactly what a
        migrated v3.5 genome sees before it has written anything.

        Args:
            slots: (M, slot_width) slot array
            s: (S,) state-encoder output — half of the query
            h_slow: (H_s,) slow core — the other half

        Returns:
            (E,) memory read vector
        """
        e = self.embed_dim
        if self.memory_slots == 0:
            return np.zeros(e, dtype=np.float32)
        occupied = slots[:, self.latent_pre + 2] != 0.0
        if not np.any(occupied):
            return np.zeros(e, dtype=np.float32)

        tokens = self._slot_tokens(slots)
        p = self.params["memory"]
        t = np.tanh(tokens @ p["Wtok"] + p["btok"])
        query_in = np.concatenate([s, h_slow])
        q = query_in @ p["Wq"]
        k = t @ p["Wk"]
        v = t @ p["Wv"]
        scores = (k @ q) / np.sqrt(e)
        scores = np.where(occupied, scores, -1e9)
        scores = scores - np.max(scores)
        weights = np.exp(scores)
        weights = weights / np.sum(weights)
        return (weights @ v).astype(np.float32)

    def _slot_tokens(self, slots: np.ndarray) -> np.ndarray:
        """
        Turn raw slots into attention tokens.

        The displacement is bounded by ``d / (1 + ||d||)`` so it sits in the
        same range as every other feature while keeping its direction exactly,
        and the age becomes a recency in (0, 1].
        """
        n = self.latent_pre
        tokens = np.array(slots, dtype=np.float32, copy=True)
        d = tokens[:, n : n + 2]
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        tokens[:, n : n + 2] = d / (1.0 + norm)
        tokens[:, n + 3] = np.exp(-tokens[:, n + 3] / RECENCY_SCALE)
        return tokens

    # -- memory dynamics ---------------------------------------------------

    def update_memory(
        self,
        h: np.ndarray,
        observation: np.ndarray,
        action: int,
        reward: float,
        moved: bool,
    ) -> np.ndarray:
        """
        Advance the episodic memory by one tick: path-integrate, then write.

        Path integration is exact integer arithmetic on a 4-heading grid — no
        parameters, no drift. Displacements are stored as ``(right, ahead)``
        in the agent's current frame:

            move forward (successfully):  ahead -= 1
            turn left:                    (right, ahead) <- ( ahead, -right)
            turn right:                   (right, ahead) <- (-ahead,  right)

        The write comes *after* the move, so the recorded place is where the
        agent actually ended up — the place the reward was realised at. A slot
        is claimed when this step's salience ``|reward|`` beats the weakest
        stored salience, which for an empty slot is 0.

        The stored latent is a **stop-gradient snapshot**: memory is a record
        of what happened, not a second path for the encoder's gradient.

        Args:
            h: Packed state returned by ``forward``
            observation: The decision-time observation of this step
            action: Action index taken
            reward: Reward received this step (its magnitude is the salience)
            moved: Whether a MOVE_FORWARD actually displaced the agent

        Returns:
            A new packed state with the memory advanced
        """
        if self.memory_slots == 0:
            return np.asarray(h, dtype=np.float32)

        h = np.array(h, dtype=np.float32, copy=True)
        hf, hs, slots = self.split_state(h)
        n = self.latent_pre

        # 1. Age and fade every stored memory.
        slots[:, n + 2] *= SALIENCE_DECAY
        slots[:, n + 3] += 1.0

        # 2. Path integration for the action just taken.
        d_right = slots[:, n].copy()
        d_ahead = slots[:, n + 1].copy()
        if action == Action.TURN_LEFT.value:
            slots[:, n] = d_ahead
            slots[:, n + 1] = -d_right
        elif action == Action.TURN_RIGHT.value:
            slots[:, n] = -d_ahead
            slots[:, n + 1] = d_right
        elif action == Action.MOVE_FORWARD.value and moved:
            slots[:, n + 1] = d_ahead - 1.0

        # 3. Write, if this step was more salient than the weakest slot.
        salience = float(abs(reward))
        if salience > 0.0:
            weakest = int(np.argmin(slots[:, n + 2]))
            if salience > WRITE_MARGIN * slots[weakest, n + 2]:
                slots[weakest, :n] = self._encode_pre(observation)
                slots[weakest, n] = 0.0
                slots[weakest, n + 1] = 0.0
                slots[weakest, n + 2] = salience
                slots[weakest, n + 3] = 0.0

        return self.pack_state(hf, hs, slots)

    # -- forward -----------------------------------------------------------

    def core_step(
        self, observation: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        One step of perception + both recurrent cores.

        Args:
            observation: Observation vector
            h: Packed recurrent state

        Returns:
            (z, h_next_packed, core, s) — the full latent, the advanced packed
            state, the combined core readout ``[h_fast || h_slow]`` the heads
            use, and the state-encoder output.
        """
        hf, hs, slots = self.split_state(h)
        z_pre = self._encode_pre(observation)
        s = z_pre[: self.state_dim]
        mem = self._read_memory(slots, s, hs)
        z = np.concatenate([z_pre, mem])

        hf_next = modules.gru_step(z, hf, self.params["gru"])
        hs_next = self._slow_step(hf_next, hs)
        core = np.concatenate([hf_next, hs_next])
        return z, self.pack_state(hf_next, hs_next, slots), core, s

    def _slow_step(self, h_fast: np.ndarray, h_slow: np.ndarray) -> np.ndarray:
        """
        Leaky-integrator update with per-unit, genome-encoded time constants.

            alpha = sigmoid(rho)
            u     = tanh( h_fast W_f + h_slow W_s + b )
            h_s'  = (1 - alpha) * h_slow + alpha * u
        """
        if self.slow_hidden_size == 0:
            return h_slow
        p = self.params["slow"]
        alpha = modules.sigmoid(p["rho"])
        u = np.tanh(h_fast @ p["Wf"] + h_slow @ p["Ws"] + p["b"])
        return ((1.0 - alpha) * h_slow + alpha * u).astype(np.float32)

    def forward(
        self,
        observation: np.ndarray,
        h: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        instinct_strength: float = 1.0,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Forward pass: (action probabilities, short-horizon value, next state).

        Args:
            observation: Observation vector
            h: Packed recurrent state
            action_mask: Optional binary mask (1 = valid)
            temperature: Sampling temperature
            instinct_strength: Scale factor for instinct biases

        Returns:
            (action_probs, value, next_packed_state)
        """
        z, h_next, core, _ = self.core_step(observation, h)

        logits = (
            core @ self.params["policy_head"]["W"] + self.params["policy_head"]["b"]
        )
        if action_mask is not None:
            logits = np.where(action_mask > 0, logits, -1e9)
            if self.instincts is not None:
                logits = self.instincts.apply(
                    logits, observation, action_mask, strength=instinct_strength
                )
        probs = modules.softmax(logits / temperature)
        return probs, self.values(z, core)[0], h_next

    def value_input(self, z: np.ndarray, core: np.ndarray) -> np.ndarray:
        """
        Critic input, in migration-safe order.

        ``[z_pre || h_fast || memory_read || h_slow]`` rather than the natural
        ``[z || core]``: the genome migration is a top-left copy, so every
        block that grew since v3.5 has to sit at the END of the concatenation
        or a migrated genome's rows land on the wrong inputs. The first
        ``(S + E) + H_f`` entries are exactly v3.5's ``[z || h]``.
        """
        n = self.latent_pre
        hf = core[: self.gru_hidden_size]
        return np.concatenate([z[:n], hf, z[n:], core[self.gru_hidden_size :]])

    def values(self, z: np.ndarray, core: np.ndarray) -> np.ndarray:
        """
        Both critic outputs for a latent/core pair.

        Returns:
            (2,) array: [V_short (gamma_s), V_long (gamma_l)]
        """
        vm = self.params["value_mlp"]
        hidden = np.tanh(self.value_input(z, core) @ vm["W1"] + vm["b1"])
        return (hidden @ vm["W2"] + vm["b2"]).astype(np.float32)

    def _value(self, z: np.ndarray, core: np.ndarray) -> float:
        """Short-horizon value (the one the planner and logging use)."""
        return float(self.values(z, core)[0])

    # -- heads the rest of the simulation reads ----------------------------

    def policy_from_hidden(
        self, h: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Masked policy from a packed state or a bare core readout."""
        core = self._core_of(h)
        logits = (
            core @ self.params["policy_head"]["W"] + self.params["policy_head"]["b"]
        )
        if action_mask is not None:
            logits = np.where(action_mask > 0, logits, -1e9)
        return modules.softmax(logits)

    def predict_next_latent(
        self, h: np.ndarray, action_idx: int
    ) -> Tuple[np.ndarray, float]:
        """Dynamics head over the combined core (see Brain.predict_next_latent)."""
        if not self.has_world_model:
            raise RuntimeError("Brain has no world model (dynamics head)")
        dyn = self.params["dynamics"]
        onehot = np.zeros(self.output_size, dtype=np.float32)
        onehot[action_idx] = 1.0
        d = np.tanh(
            self.dynamics_input(self._core_of(h), onehot) @ dyn["W1"] + dyn["b1"]
        )
        return d @ dyn["Wz"] + dyn["bz"], float((d @ dyn["Wr"] + dyn["br"]).item())

    def dynamics_input(self, core: np.ndarray, onehot: np.ndarray) -> np.ndarray:
        """
        Dynamics-head input, in migration-safe order:
        ``[h_fast || onehot(a) || h_slow]`` (see value_input for why).
        """
        hf = core[: self.gru_hidden_size]
        hs = core[self.gru_hidden_size :]
        return np.concatenate([hf, onehot, hs])

    def action_latent_spread(self, h: np.ndarray) -> float:
        """
        One-step empowerment proxy: how much this state's choice matters.

            zbar' = (1/|A|) sum_a g_z(h, a)
            E     = (1/|A|) sum_a || g_z(h, a) - zbar' ||^2 / Z

        i.e. the variance across actions of the predicted next latent. Under a
        fixed-noise Gaussian channel ``Z' = mu(a) + eps`` the true one-step
        empowerment ``max_p(a) I(A; Z')`` is ``~ 0.5 log(1 + Var_a[mu]/sigma^2)``,
        monotone increasing in exactly this quantity — so maximising the proxy
        maximises a monotone transform of the real thing, at the cost of |A|
        evaluations of a two-layer MLP (docs/BRAIN_V4_PROPOSAL.md §4.5b).

        Args:
            h: Packed state or bare core readout

        Returns:
            Mean per-dimension variance of the predicted next latent, or 0.0
            when the brain has no dynamics head.
        """
        if not self.has_world_model:
            return 0.0
        dyn = self.params["dynamics"]
        core = self._core_of(h)
        onehots = np.eye(self.output_size, dtype=np.float32)
        heads = np.concatenate(
            [
                np.repeat(core[: self.gru_hidden_size][None, :], self.output_size, 0),
                onehots,
                np.repeat(core[self.gru_hidden_size :][None, :], self.output_size, 0),
            ],
            axis=1,
        )
        d = np.tanh(heads @ dyn["W1"] + dyn["b1"])
        z_pred = d @ dyn["Wz"] + dyn["bz"]
        return float(np.mean(np.var(z_pred, axis=0)))

    def comm_vector(self, h: np.ndarray) -> np.ndarray:
        """
        The continuous symbol this agent would emit right now.

        ``u = tanh( [h_fast || h_slow] W_u + b_u )`` in [-1, 1]^C. It carries
        no built-in meaning; emitting it costs energy proportional to its
        amplitude (see world.emit_signal), which is the precondition for
        honest signalling to be stable.
        """
        p = self.params["comm"]
        return np.tanh(self._core_of(h) @ p["W"] + p["b"]).astype(np.float32)

    @property
    def drive_weights(self) -> np.ndarray:
        """
        Evolved reward weights (homeostasis, empowerment, curiosity, social).

        Clipped to a sane range at read time: mutation is unbounded, and an
        agent whose curiosity weight drifted to 1e3 would not be exploring,
        it would be diverging.
        """
        return np.clip(
            self.params["drive"]["lam"][:DRIVE_BETA_INDEX], -DRIVE_CLIP, DRIVE_CLIP
        )

    @property
    def advantage_mix(self) -> float:
        """
        Evolved dual-discount mix ``beta = sigmoid(beta_raw)`` in (0, 1).

        ``A = (1 - beta) A_short + beta A_long``: the population evolves its
        own planning horizon. The prior starts it near 0, i.e. at v3.5's
        behaviour (docs/BRAIN_V4_PROPOSAL.md §4.3).
        """
        return float(modules.sigmoid(self.params["drive"]["lam"][DRIVE_BETA_INDEX]))

    @property
    def identity_tag(self) -> np.ndarray:
        """Visible identity tag, bounded to [0, 1] for the observation vector."""
        return (0.5 * (1.0 + np.tanh(self.params["tag"]["g"]))).astype(np.float32)

    def _core_of(self, h: np.ndarray) -> np.ndarray:
        """Accept either a packed state or an already-extracted core readout."""
        h = np.asarray(h, dtype=np.float32)
        return h[: self.core_size] if h.shape[0] != self.core_size else h

    @staticmethod
    def calculate_v4_weight_count(
        state_inputs: int = 41,
        embed_dim: int = 8,
        state_dim: int = 40,
        gru_hidden_size: int = 48,
        slow_hidden_size: int = 24,
        value_hidden: int = 16,
        output_size: int = 9,
        world_model_hidden: Optional[int] = None,
        comm_channels: int = COMM_CHANNELS,
    ) -> int:
        """Total genome length for the v4 network (derived from the ParamSpec)."""
        return build_brain_v4_param_spec(
            state_inputs=state_inputs,
            embed_dim=embed_dim,
            state_dim=state_dim,
            gru_hidden_size=gru_hidden_size,
            slow_hidden_size=slow_hidden_size,
            value_hidden=value_hidden,
            output_size=output_size,
            world_model_hidden=world_model_hidden,
            comm_channels=comm_channels,
        ).count()
