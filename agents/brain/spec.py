"""
Single sources of truth for brain parameter and observation layouts.

ParamSpec
    A declarative, ordered list of named parameter tensors. Weight
    counting, genome unpacking (zero-copy views), and genome packing
    are all derived from the one spec. This replaces the three
    hand-maintained layout definitions that previously lived in
    brain_utils.calculate_weight_count, brain_utils.unpack_weights,
    and AgentLearner._sync_genome_weights.

ObservationSpec
    Named layout of the observation vector (group slices + stimulus
    field indices), replacing the magic indices previously scattered
    across the brain and instinct code.

Author: Karan Vasa
Date: June 2026
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Parameter specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """
    Ordered specification of named parameter tensors.

    Attributes:
        entries: Tuple of (name, shape) pairs in genome order
        version: Layout version tag (used to detect/migrate old genomes)
    """

    entries: tuple[tuple[str, tuple[int, ...]], ...]
    version: int = 2
    # Structured priors. Most tensors are fine starting at zero (after a
    # migration) or at random (in a fresh genome), but a few are not: a leak
    # rate of 0 makes the slow core useless, and a drive weight of 0 makes the
    # reward identically zero. Entries listed here get their value from the
    # callable instead. See build_brain_v4_param_spec.
    inits: tuple[tuple[str, Callable[[tuple[int, ...]], np.ndarray]], ...] = ()

    def init_map(self) -> dict[str, Callable[[tuple[int, ...]], np.ndarray]]:
        """Name → initialiser, for the entries that declare one."""
        return dict(self.inits)

    def apply_inits(self, flat: np.ndarray) -> np.ndarray:
        """
        Overwrite the structured-prior entries of a flat genome in place.

        Used for freshly randomised genomes, where ``np.random.randn`` would
        otherwise destroy the priors (a random ``slow.rho`` collapses every
        time constant to ~2 ticks).

        Args:
            flat: Flat weight vector laid out by this spec

        Returns:
            The same array, for chaining
        """
        inits = self.init_map()
        if not inits:
            return flat
        named = self.unpack(flat)
        for name, make in inits.items():
            if name in named:
                named[name][...] = make(named[name].shape)
        return flat

    def count(self) -> int:
        """Total number of scalar weights (including biases)."""
        return sum(int(np.prod(shape)) for _, shape in self.entries)

    def names(self) -> list[str]:
        """Parameter names in genome order."""
        return [name for name, _ in self.entries]

    def unpack(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """
        Unpack a flat weight vector into named tensors.

        The returned arrays are views into ``flat`` (zero-copy), so
        in-place updates to them are reflected in the flat vector.

        Args:
            flat: Flat weight vector of length ``count()``

        Returns:
            Dict mapping parameter name to ndarray view

        Raises:
            ValueError: If ``flat`` has the wrong length
        """
        flat = np.asarray(flat)
        if flat.shape != (self.count(),):
            raise ValueError(
                f"Expected flat weights of shape ({self.count()},), "
                f"got {flat.shape} (spec version {self.version})"
            )

        named: dict[str, np.ndarray] = {}
        idx = 0
        for name, shape in self.entries:
            size = int(np.prod(shape))
            named[name] = flat[idx : idx + size].reshape(shape)
            idx += size
        return named

    def pack(
        self, named: dict[str, np.ndarray], dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """
        Pack named tensors back into a single flat vector.

        Args:
            named: Dict mapping parameter name to ndarray
            dtype: Output dtype (float32 keeps genomes compact)

        Returns:
            Flat weight vector in spec order
        """
        parts = [np.asarray(named[name]).ravel() for name, _ in self.entries]
        return np.concatenate(parts).astype(dtype)


def migrate_genome(
    old_flat: np.ndarray,
    old_spec: ParamSpec,
    new_spec: ParamSpec,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Migrate a flat genome from ``old_spec`` to ``new_spec`` losslessly.

    This is the mechanism behind the W4 "single batched genome break" (and
    any future append-only growth). Because every spec extension is
    *append-only* — new observation features become extra **rows** at the end
    of the first weight matrix, a new action becomes an extra **column** at
    the end of the policy head, the world-model head is appended last — each
    new parameter tensor contains the old one in its **top-left corner**.
    So the migration is: for every entry in the new spec, allocate zeros and
    copy the overlapping top-left block from the old genome (when that entry
    existed). New rows/columns stay zero, which means:

      * new observation features contribute exactly 0 to the encoder, so a
        migrated brain's encoder/GRU/value outputs and its logits for the
        original actions are **bit-identical** to the old brain's, and
      * a new action's policy column is 0 (a neutral logit) — it only starts
        being used once mutation/learning fills it in.

    Args:
        old_flat: Flat genome laid out by ``old_spec``
        old_spec: The genome's current layout
        new_spec: The target layout (must be append-only-compatible)
        dtype: Output dtype

    Returns:
        Flat genome laid out by ``new_spec``
    """
    old_named = old_spec.unpack(old_flat)
    inits = new_spec.init_map()
    new_named: dict[str, np.ndarray] = {}
    for name, shape in new_spec.entries:
        old = old_named.get(name)
        if old is None and name in inits:
            # A genuinely new tensor with a structured prior (a leak-rate
            # ladder, a drive-weight default). Zero would be wrong, not
            # merely neutral, so the prior wins.
            new_named[name] = np.asarray(inits[name](shape), dtype=dtype)
            continue
        arr = np.zeros(shape, dtype=dtype)
        if old is not None:
            region = tuple(slice(0, min(o, n)) for o, n in zip(old.shape, shape))
            arr[region] = old[region]
        new_named[name] = arr
    return new_spec.pack(new_named, dtype=dtype)


def _dynamics_entries(
    latent_size: int, gru_hidden_size: int, output_size: int, hidden: int
) -> list[tuple[str, tuple[int, ...]]]:
    """
    Genome entries for the latent dynamics head (learned world model).

    The head predicts the NEXT latent ẑ_{t+1} and reward r̂_t from the
    post-decision hidden state h_{t+1} and a one-hot action:

        d  = tanh([h ‖ onehot(a)]·W1 + b1)
        ẑ' = d·Wz + bz          (next-latent prediction, linear)
        r̂  = d·Wr + br          (reward prediction, scalar)

    Appended at the END of the spec so enabling the world model only
    extends existing genome layouts (prefix stays valid for migration).

    Args:
        latent_size: Size of the latent z the head must predict
        gru_hidden_size: Size of the GRU hidden state (head input)
        output_size: Number of actions (one-hot size)
        hidden: Hidden layer width of the dynamics MLP

    Returns:
        List of (name, shape) entries
    """
    d_in = gru_hidden_size + output_size
    return [
        ("dyn.W1", (d_in, hidden)),
        ("dyn.b1", (hidden,)),
        ("dyn.Wz", (hidden, latent_size)),
        ("dyn.bz", (latent_size,)),
        ("dyn.Wr", (hidden, 1)),
        ("dyn.br", (1,)),
    ]


def build_brain_param_spec(
    input_size: int = 72,
    encoder_layers: Optional[list[int]] = None,
    gru_hidden_size: int = 32,
    output_size: int = 8,
    world_model_hidden: Optional[int] = None,
) -> ParamSpec:
    """
    Build the ParamSpec for the recurrent Actor-Critic brain.

    The entry order exactly matches the historical (v2) flat genome
    layout, so existing genomes remain valid:
      encoder (W, b per layer) → GRU (r, z, h gates: W_in, W_hid, b)
      → policy head (W, b) → value head (W, b)
      [→ dynamics head, only when world_model_hidden is set]

    Args:
        input_size: Size of observation vector
        encoder_layers: Sizes of encoder hidden layers (default: [32])
        gru_hidden_size: Size of GRU hidden state
        output_size: Number of actions
        world_model_hidden: Hidden width of the latent dynamics head
            (None = no world model; genome layout unchanged)

    Returns:
        ParamSpec describing the brain's genome layout
    """
    if encoder_layers is None:
        encoder_layers = [32]

    entries: list[tuple[str, tuple[int, ...]]] = []

    # 1. Encoder MLP
    sizes = [input_size] + list(encoder_layers)
    for i in range(len(sizes) - 1):
        entries.append((f"encoder.{i}.W", (sizes[i], sizes[i + 1])))
        entries.append((f"encoder.{i}.b", (sizes[i + 1],)))

    # 2. GRU (3 gates: reset, update, candidate)
    enc_out = encoder_layers[-1]
    h = gru_hidden_size
    for gate in ("r", "z", "h"):
        entries.append((f"gru.W{gate}_input", (enc_out, h)))
        entries.append((f"gru.W{gate}_hidden", (h, h)))
        entries.append((f"gru.b{gate}", (h,)))

    # 3. Policy head
    entries.append(("policy.W", (h, output_size)))
    entries.append(("policy.b", (output_size,)))

    # 4. Value head
    entries.append(("value.W", (h, 1)))
    entries.append(("value.b", (1,)))

    # 5. Optional latent dynamics head (world model)
    if world_model_hidden is not None:
        entries.extend(_dynamics_entries(enc_out, h, output_size, world_model_hidden))

    return ParamSpec(entries=tuple(entries), version=2)


def build_nested_params(named: dict[str, np.ndarray], num_encoder_layers: int) -> dict:
    """
    Arrange named parameter views into the nested structure used by
    Brain.forward and AgentLearner (encoder_weights / gru / heads).

    The nested dict shares memory with ``named`` — both expose views
    of the same underlying flat vector.

    Args:
        named: Output of ParamSpec.unpack
        num_encoder_layers: Number of encoder layers in the spec

    Returns:
        Nested parameter dictionary
    """
    nested = {
        "encoder_weights": [named[f"encoder.{i}.W"] for i in range(num_encoder_layers)],
        "encoder_biases": [named[f"encoder.{i}.b"] for i in range(num_encoder_layers)],
        "gru": {
            "Wr_input": named["gru.Wr_input"],
            "Wr_hidden": named["gru.Wr_hidden"],
            "br": named["gru.br"],
            "Wz_input": named["gru.Wz_input"],
            "Wz_hidden": named["gru.Wz_hidden"],
            "bz": named["gru.bz"],
            "Wh_input": named["gru.Wh_input"],
            "Wh_hidden": named["gru.Wh_hidden"],
            "bh": named["gru.bh"],
        },
        "policy_head": {"W": named["policy.W"], "b": named["policy.b"]},
        "value_head": {"W": named["value.W"], "b": named["value.b"]},
    }
    if "dyn.W1" in named:
        nested["dynamics"] = {
            key.split(".", 1)[1]: named[key] for key in named if key.startswith("dyn.")
        }
    return nested


def build_brain_v3_param_spec(
    state_inputs: int = 22,
    embed_dim: int = 8,
    state_dim: int = 40,
    gru_hidden_size: int = 48,
    value_hidden: int = 16,
    output_size: int = 8,
    world_model_hidden: Optional[int] = None,
) -> ParamSpec:
    """
    Build the ParamSpec for the Brain v3 architecture.

    v3 layout (see agents/brain/v3.py for the forward pass):
      state encoder (22 non-vision features → S)
      tile embedding (2 tile features + 2 positional → E, shared by all tiles)
      attention (query from state, keys/values from tile embeddings)
      GRU over the latent z = [state S | attended vision E]
      policy head (H → actions)
      value MLP ([z, h] → value_hidden → 1)
      [dynamics head, only when world_model_hidden is set]

    Args:
        state_inputs: Non-vision feature count (agent_state+stimulus+inventory)
        embed_dim: Per-tile embedding size (E)
        state_dim: State encoder output size (S); GRU input is S + E
        gru_hidden_size: GRU hidden state size (H)
        value_hidden: Hidden size of the value MLP
        output_size: Number of actions
        world_model_hidden: Hidden width of the latent dynamics head
            (None = no world model; genome layout unchanged)

    Returns:
        ParamSpec (version=3) describing the v3 genome layout
    """
    e = embed_dim
    s = state_dim
    h = gru_hidden_size
    z = s + e  # latent fed to the GRU and (with h) to the value MLP

    entries: list[tuple[str, tuple[int, ...]]] = [
        # 1. State encoder (agent_state + stimulus + inventory → S)
        ("state_enc.W", (state_inputs, s)),
        ("state_enc.b", (s,)),
        # 2. Shared tile embedding ([type, value, pos_row, pos_col] → E).
        #    One small matrix shared by every tile — position-equivariant,
        #    unlike v2's dense vision layer that memorises tile positions.
        ("tile_embed.W", (4, e)),
        ("tile_embed.b", (e,)),
        # 3. Single-head attention pool over tile tokens
        ("attn.Wq", (s, e)),
        ("attn.Wk", (e, e)),
        ("attn.Wv", (e, e)),
    ]

    # 4. GRU over the latent z (3 gates: reset, update, candidate)
    for gate in ("r", "z", "h"):
        entries.append((f"gru.W{gate}_input", (z, h)))
        entries.append((f"gru.W{gate}_hidden", (h, h)))
        entries.append((f"gru.b{gate}", (h,)))

    # 5. Policy head
    entries.append(("policy.W", (h, output_size)))
    entries.append(("policy.b", (output_size,)))

    # 6. Value MLP reads [z, h]: the critic gets a direct view of the
    #    current state instead of only what the GRU chose to remember.
    entries.append(("value.W1", (z + h, value_hidden)))
    entries.append(("value.b1", (value_hidden,)))
    entries.append(("value.W2", (value_hidden, 1)))
    entries.append(("value.b2", (1,)))

    # 7. Optional latent dynamics head (world model)
    if world_model_hidden is not None:
        entries.extend(_dynamics_entries(z, h, output_size, world_model_hidden))

    return ParamSpec(entries=tuple(entries), version=3)


# ---------------------------------------------------------------------------
# Brain v4 — multi-timescale memory core, episodic place memory, dual-discount
# critic, vector communication head, evolved drive weights.
# See docs/BRAIN_V4_PROPOSAL.md §4.
# ---------------------------------------------------------------------------

# Fixed sizes of the v4 observation tail, so the layout is a version, not a
# config. Features can be zeroed by config (ablation) without a genome change.
COMM_CHANNELS = 4
IDENTITY_TAG_DIM = 4

# Episodic slot token: [ z_pre (S+E) | delta (2) | salience (1) | recency (1) ]
MEMORY_SLOT_EXTRA = 4

# Evolved scalars in the genome (docs/BRAIN_V4_PROPOSAL.md §4.5, §4.3):
#   0 homeostasis, 1 empowerment, 2 curiosity, 3 social — the drive weights
#   4 beta_raw — the dual-discount advantage mix, beta = sigmoid(beta_raw)
DRIVE_COUNT = 5
DRIVE_BETA_INDEX = 4

# Leak-rate ladder bounds (ticks). The world's own timescales are plant
# maturation ~160, the day 200, the season 2000.
SLOW_TAU_MIN = 2.0
SLOW_TAU_MAX = 512.0


def slow_rho_ladder(shape: tuple[int, ...]) -> np.ndarray:
    """
    Geometric ladder of leak rates for the slow recurrent core.

    ``alpha_i = sigmoid(rho_i)`` and ``tau_i = 1 / alpha_i``, so

        tau_i = TAU_MIN * (TAU_MAX / TAU_MIN) ** (i / (n - 1))
        rho_i = logit(1 / tau_i) = -log(tau_i - 1)

    spreads the units geometrically from 2 to 512 ticks. A zero-initialised
    ``rho`` would give every unit ``alpha = 0.5``, i.e. ``tau = 2`` — the one
    setting that makes the slow core pointless — which is why this is a
    structured prior rather than a default of zeros.

    Args:
        shape: (n,) — the number of slow units

    Returns:
        (n,) array of rho values
    """
    n = int(np.prod(shape))
    if n <= 0:
        return np.zeros(shape, dtype=np.float32)
    if n == 1:
        taus = np.array([SLOW_TAU_MAX], dtype=np.float64)
    else:
        taus = SLOW_TAU_MIN * (SLOW_TAU_MAX / SLOW_TAU_MIN) ** (
            np.arange(n, dtype=np.float64) / (n - 1)
        )
    rho = -np.log(np.maximum(taus - 1.0, 1e-6))
    return rho.astype(np.float32).reshape(shape)


def drive_lambda_prior(shape: tuple[int, ...]) -> np.ndarray:
    """
    Default evolved scalars: homeostasis only, short discount only.

    A migrated or freshly randomised genome must not start with a reward that
    is identically zero, so the drive weights start at (1, 0, 0, 0) — pure
    drive reduction — and evolution moves them from there. ``beta_raw``
    starts at -4, i.e. ``beta = sigmoid(-4) ~ 0.018``, so a migrated v3.5
    genome behaves as it always did until selection finds a use for the
    long-horizon head.

    Args:
        shape: (DRIVE_COUNT,)

    Returns:
        Prior array
    """
    arr = np.zeros(shape, dtype=np.float32)
    flat = arr.reshape(-1)
    if flat.size:
        flat[0] = 1.0
    if flat.size > DRIVE_BETA_INDEX:
        flat[DRIVE_BETA_INDEX] = -4.0
    return arr


def build_brain_v4_param_spec(
    state_inputs: int = 41,
    embed_dim: int = 8,
    state_dim: int = 40,
    gru_hidden_size: int = 48,
    slow_hidden_size: int = 24,
    value_hidden: int = 16,
    output_size: int = 9,
    world_model_hidden: Optional[int] = None,
    comm_channels: int = COMM_CHANNELS,
) -> ParamSpec:
    """
    Build the ParamSpec for the Brain v4 architecture.

    v4 = v3.5 plus four things, all append-only over the v3 entry order so a
    v3/v3.5 genome migrates by the shipped top-left copy:

      * an episodic place memory read (``mem.*``), concatenated into the
        latent, so ``gru.W*_input`` grows by ``embed_dim`` rows;
      * a slow leaky recurrent core (``slow.*``) whose per-unit leak rates
        live in the genome, so the policy/value/dynamics heads read
        ``[h_fast || h_slow]`` and grow by ``slow_hidden_size`` rows;
      * a second value output (``value.W2`` gains a column) for the long
        discount;
      * a continuous communication head (``comm.*``), the drive weights
        (``drive.lam``) and a visible identity tag (``tag.g``).

    Args:
        state_inputs: Non-vision feature count (41 under Observation v4)
        embed_dim: Per-tile and per-slot embedding size (E)
        state_dim: State encoder output size (S)
        gru_hidden_size: Fast GRU hidden size (H_f)
        slow_hidden_size: Slow leaky core size (H_s); 0 disables the core
        value_hidden: Hidden size of the value MLP
        output_size: Number of discrete actions
        world_model_hidden: Dynamics-head hidden width (None = no world model)
        comm_channels: Width of the communication vector (C)

    Returns:
        ParamSpec (version=4) describing the v4 genome layout
    """
    e = embed_dim
    s = state_dim
    hf = gru_hidden_size
    hs = slow_hidden_size
    hc = hf + hs  # what the heads read
    z_pre = s + e  # state ‖ attended vision — what a memory slot stores
    z = z_pre + e  # ‖ memory read — what the GRU and the critic see
    slot_token = z_pre + MEMORY_SLOT_EXTRA

    entries: list[tuple[str, tuple[int, ...]]] = [
        # 1. State encoder (agent_state + stimulus + inventory + EXTRA → S)
        ("state_enc.W", (state_inputs, s)),
        ("state_enc.b", (s,)),
        # 2. Shared tile embedding — unchanged from v3
        ("tile_embed.W", (4, e)),
        ("tile_embed.b", (e,)),
        # 3. Vision attention — unchanged from v3
        ("attn.Wq", (s, e)),
        ("attn.Wk", (e, e)),
        ("attn.Wv", (e, e)),
    ]

    # 4. Fast GRU over the latent z (3 gates). Entry order matches v3 so the
    #    migration's top-left copy lines up.
    for gate in ("r", "z", "h"):
        entries.append((f"gru.W{gate}_input", (z, hf)))
        entries.append((f"gru.W{gate}_hidden", (hf, hf)))
        entries.append((f"gru.b{gate}", (hf,)))

    # 5. Policy head over [h_fast ‖ h_slow]
    entries.append(("policy.W", (hc, output_size)))
    entries.append(("policy.b", (output_size,)))

    # 6. Value MLP with TWO outputs: the short discount (column 0, migrates
    #    from v3's single column) and the long one (column 1, starts at zero).
    #    Input order is [z_pre ‖ h_fast ‖ memory_read ‖ h_slow] for the same
    #    append-only reason as the dynamics head above: the first (z_pre + H_f)
    #    rows are exactly v3.5's [z ‖ h], and the two new blocks follow.
    entries.append(("value.W1", (z + hc, value_hidden)))
    entries.append(("value.b1", (value_hidden,)))
    entries.append(("value.W2", (value_hidden, 2)))
    entries.append(("value.b2", (2,)))

    # 7. Optional latent dynamics head (world model).
    #    Input order is [h_fast ‖ onehot(a) ‖ h_slow], NOT the natural
    #    [core ‖ onehot]: migration is a top-left copy, so every block that
    #    grows must grow at the END of the concatenation or a v3.5 genome's
    #    rows land on the wrong inputs. h_slow is the new block, so it goes
    #    last and the first (H_f + A) rows are exactly v3.5's layout.
    if world_model_hidden is not None:
        d_in = hf + output_size + hs
        entries.extend(
            [
                ("dyn.W1", (d_in, world_model_hidden)),
                ("dyn.b1", (world_model_hidden,)),
                ("dyn.Wz", (world_model_hidden, z)),
                ("dyn.bz", (z,)),
                ("dyn.Wr", (world_model_hidden, 1)),
                ("dyn.br", (1,)),
            ]
        )

    # 8. Episodic place memory (§4.4): token embedding + its own attention.
    entries.append(("mem.Wtok", (slot_token, e)))
    entries.append(("mem.btok", (e,)))
    entries.append(("mem.Wq", (s + hs, e)))
    entries.append(("mem.Wk", (e, e)))
    entries.append(("mem.Wv", (e, e)))

    # 9. Slow leaky core (§4.2). `rho` carries the structured prior.
    entries.append(("slow.Wf", (hf, hs)))
    entries.append(("slow.Ws", (hs, hs)))
    entries.append(("slow.b", (hs,)))
    entries.append(("slow.rho", (hs,)))

    # 10. Communication head (§4.6), drive weights (§4.5), identity tag (§4.7)
    entries.append(("comm.W", (hc, comm_channels)))
    entries.append(("comm.b", (comm_channels,)))
    entries.append(("drive.lam", (DRIVE_COUNT,)))
    entries.append(("tag.g", (IDENTITY_TAG_DIM,)))

    return ParamSpec(
        entries=tuple(entries),
        version=4,
        inits=(
            ("slow.rho", slow_rho_ladder),
            ("drive.lam", drive_lambda_prior),
        ),
    )


def build_nested_params_v4(named: dict[str, np.ndarray]) -> dict:
    """
    Arrange v4 named parameter views into the nested structure BrainV4 and
    the learner use. Shares memory with ``named``.

    Args:
        named: Output of ParamSpec.unpack for a version-4 spec

    Returns:
        Nested parameter dictionary
    """
    nested = build_nested_params_v3(named)
    nested["memory"] = {
        "Wtok": named["mem.Wtok"],
        "btok": named["mem.btok"],
        "Wq": named["mem.Wq"],
        "Wk": named["mem.Wk"],
        "Wv": named["mem.Wv"],
    }
    nested["slow"] = {
        "Wf": named["slow.Wf"],
        "Ws": named["slow.Ws"],
        "b": named["slow.b"],
        "rho": named["slow.rho"],
    }
    nested["comm"] = {"W": named["comm.W"], "b": named["comm.b"]}
    nested["drive"] = {"lam": named["drive.lam"]}
    nested["tag"] = {"g": named["tag.g"]}
    return nested


def build_nested_params_v3(named: dict[str, np.ndarray]) -> dict:
    """
    Arrange v3 named parameter views into the nested structure used by
    BrainV3 and the learner. Shares memory with ``named``.

    The "gru" and "policy_head" sub-dicts use the same keys as v2 so
    shared code (GRU step, policy update, instincts) works unchanged.

    Args:
        named: Output of ParamSpec.unpack for a version-3 spec

    Returns:
        Nested parameter dictionary
    """
    nested = {
        "state_enc": {"W": named["state_enc.W"], "b": named["state_enc.b"]},
        "tile_embed": {"W": named["tile_embed.W"], "b": named["tile_embed.b"]},
        "attn": {
            "Wq": named["attn.Wq"],
            "Wk": named["attn.Wk"],
            "Wv": named["attn.Wv"],
        },
        "gru": {
            "Wr_input": named["gru.Wr_input"],
            "Wr_hidden": named["gru.Wr_hidden"],
            "br": named["gru.br"],
            "Wz_input": named["gru.Wz_input"],
            "Wz_hidden": named["gru.Wz_hidden"],
            "bz": named["gru.bz"],
            "Wh_input": named["gru.Wh_input"],
            "Wh_hidden": named["gru.Wh_hidden"],
            "bh": named["gru.bh"],
        },
        "policy_head": {"W": named["policy.W"], "b": named["policy.b"]},
        "value_mlp": {
            "W1": named["value.W1"],
            "b1": named["value.b1"],
            "W2": named["value.W2"],
            "b2": named["value.b2"],
        },
    }
    if "dyn.W1" in named:
        nested["dynamics"] = {
            key.split(".", 1)[1]: named[key] for key in named if key.startswith("dyn.")
        }
    return nested


# ---------------------------------------------------------------------------
# Observation specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationSpec:
    """
    Named layout of the observation vector.

    Group slices:
        agent_state — energy, age, direction one-hot, inventory space,
                      metabolism
        vision      — egocentric grid, (side × side × 2) flattened
        stimulus    — pre-processed survival signals
        inventory   — inventory summary
        extra       — Observation-v2 social/climate block (empty in v1)

    Stimulus (and v2 extra) fields are exposed as absolute indices.
    """

    agent_state: slice
    vision: slice
    stimulus: slice
    inventory: slice
    vision_shape: tuple[int, int, int]
    size: int

    # Absolute indices of stimulus fields
    food_on_tile: int
    seed_on_tile: int
    food_ahead: int
    resource_ahead: int
    nearest_food_prox: int
    food_dir_match: int
    energy_urgency: int
    can_interact: int

    # Observation version (1 = legacy 72-dim; 2 = +EXTRA block, Brain v3.5;
    # 4 = +kin, comm channels and neighbour identity tag, Brain v4)
    version: int = 1
    # EXTRA block (empty slice in v1). Absolute field indices are -1 in v1.
    # default_factory: slice is unhashable, so dataclass rejects a bare default.
    extra: slice = field(default_factory=lambda: slice(0, 0))
    time_of_day_sin: int = -1
    time_of_day_cos: int = -1
    tile_temperature: int = -1
    nearest_agent_proximity: int = -1
    nearest_agent_signal: int = -1
    on_hazard: int = -1
    # Observation-v4 tail (indices are -1 / empty slices below version 4)
    nearest_agent_kin: int = -1
    comm_mean: slice = field(default_factory=lambda: slice(0, 0))
    comm_max: slice = field(default_factory=lambda: slice(0, 0))
    neighbour_tag: slice = field(default_factory=lambda: slice(0, 0))

    def vision_grid(self, observation: np.ndarray) -> np.ndarray:
        """
        Return the vision portion of an observation as a
        (rows, cols, features) grid view.

        Rows index the agent-relative dy (row 0 = furthest ahead),
        cols index dx (col < center = agent's left).
        """
        return np.asarray(observation)[self.vision].reshape(self.vision_shape)


def build_observation_spec(vision_radius: int = 2, version: int = 1) -> ObservationSpec:
    """
    Build the ObservationSpec matching utils/agents/perception.py.

    Args:
        vision_radius: Vision grid radius (2 → 5×5 grid)
        version: 1 = legacy 72-dim layout; 2 = append the 6-feature EXTRA
            social/climate block (Brain v3.5 / World phase W4). The 0–71
            prefix is identical between versions (append-only).

    Returns:
        ObservationSpec with derived slices and field indices
    """
    side = 2 * vision_radius + 1

    agent_state = slice(0, 8)
    vision = slice(agent_state.stop, agent_state.stop + side * side * 2)
    stimulus = slice(vision.stop, vision.stop + 8)
    inventory = slice(stimulus.stop, stimulus.stop + 6)

    s = stimulus.start
    if version >= 2:
        # v2 EXTRA block (6). Under v4 the same block is extended in place
        # with the kin scalar, the C-channel communication readings and the
        # nearest agent's identity tag — append-only, so 0..77 is untouched.
        v4_width = 1 + 2 * COMM_CHANNELS + IDENTITY_TAG_DIM if version >= 4 else 0
        extra = slice(inventory.stop, inventory.stop + 6 + v4_width)
        e = extra.start
        extra_idx = dict(
            extra=extra,
            time_of_day_sin=e + 0,
            time_of_day_cos=e + 1,
            tile_temperature=e + 2,
            nearest_agent_proximity=e + 3,
            nearest_agent_signal=e + 4,
            on_hazard=e + 5,
        )
        if version >= 4:
            kin = e + 6
            cm = kin + 1
            cx = cm + COMM_CHANNELS
            tag = cx + COMM_CHANNELS
            extra_idx.update(
                nearest_agent_kin=kin,
                comm_mean=slice(cm, cm + COMM_CHANNELS),
                comm_max=slice(cx, cx + COMM_CHANNELS),
                neighbour_tag=slice(tag, tag + IDENTITY_TAG_DIM),
            )
        size = extra.stop
    else:
        extra_idx = dict(extra=slice(inventory.stop, inventory.stop))
        size = inventory.stop

    return ObservationSpec(
        agent_state=agent_state,
        vision=vision,
        stimulus=stimulus,
        inventory=inventory,
        vision_shape=(side, side, 2),
        size=size,
        food_on_tile=s + 0,
        seed_on_tile=s + 1,
        food_ahead=s + 2,
        resource_ahead=s + 3,
        nearest_food_prox=s + 4,
        food_dir_match=s + 5,
        energy_urgency=s + 6,
        can_interact=s + 7,
        version=version,
        **extra_idx,
    )


# Default spec for the standard 72-feature observation (5×5 vision)
DEFAULT_OBSERVATION_SPEC = build_observation_spec(vision_radius=2, version=1)

# Observation-v2 spec (78-feature, Brain v3.5). Built once for reuse.
OBSERVATION_SPEC_V2 = build_observation_spec(vision_radius=2, version=2)

# Observation-v4 spec (91-feature, Brain v4): the v2 layout plus
# nearest_agent_kin (78), comm mean/max over 4 channels (79..86) and the
# nearest agent's identity tag (87..90).
OBSERVATION_SPEC_V4 = build_observation_spec(vision_radius=2, version=4)

# ---------------------------------------------------------------------------
# Active observation spec — the single switch perception and the brain both
# read so they always agree. main.py sets it from the brain version at
# startup; it defaults to the legacy v1 layout so existing runs are unchanged.
# ---------------------------------------------------------------------------

_ACTIVE_OBSERVATION_SPEC = DEFAULT_OBSERVATION_SPEC


def get_active_observation_spec() -> ObservationSpec:
    """Return the observation spec the simulation is currently using."""
    return _ACTIVE_OBSERVATION_SPEC


def set_active_observation_spec(spec: ObservationSpec) -> None:
    """
    Set the active observation spec (call once at startup, before agents are
    created). Perception, the brain encoder, and genome length all derive
    from this, so it must be set consistently with the brain version.
    """
    global _ACTIVE_OBSERVATION_SPEC
    _ACTIVE_OBSERVATION_SPEC = spec


def set_observation_version(version: int) -> None:
    """Convenience: activate the v1, v2 or v4 observation layout by number."""
    if version >= 4:
        set_active_observation_spec(OBSERVATION_SPEC_V4)
    elif version >= 2:
        set_active_observation_spec(OBSERVATION_SPEC_V2)
    else:
        set_active_observation_spec(DEFAULT_OBSERVATION_SPEC)


# ---------------------------------------------------------------------------
# Active genome layout — set alongside the observation spec so a freshly
# randomised genome gets the v4 structured priors (slow.rho's leak ladder,
# drive.lam's homeostasis default) without every Genome.random caller having
# to know about them.
# ---------------------------------------------------------------------------

_ACTIVE_PARAM_SPEC: Optional[ParamSpec] = None


def get_active_param_spec() -> Optional[ParamSpec]:
    """Return the genome layout the simulation is currently using, if set."""
    return _ACTIVE_PARAM_SPEC


def set_active_param_spec(spec: Optional[ParamSpec]) -> None:
    """Set the active genome layout (call once at startup)."""
    global _ACTIVE_PARAM_SPEC
    _ACTIVE_PARAM_SPEC = spec
