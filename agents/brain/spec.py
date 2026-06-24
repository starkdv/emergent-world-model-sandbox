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
from typing import Optional

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
    new_named: dict[str, np.ndarray] = {}
    for name, shape in new_spec.entries:
        arr = np.zeros(shape, dtype=dtype)
        old = old_named.get(name)
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

    # Observation version (1 = legacy 72-dim; 2 = +EXTRA block, Brain v3.5)
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
        extra = slice(inventory.stop, inventory.stop + 6)
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
    """Convenience: activate the v1 or v2 observation layout by number."""
    set_active_observation_spec(
        OBSERVATION_SPEC_V2 if version >= 2 else DEFAULT_OBSERVATION_SPEC
    )
