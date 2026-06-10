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

from dataclasses import dataclass
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


def build_brain_param_spec(
    input_size: int = 72,
    encoder_layers: Optional[list[int]] = None,
    gru_hidden_size: int = 32,
    output_size: int = 8,
) -> ParamSpec:
    """
    Build the ParamSpec for the recurrent Actor-Critic brain.

    The entry order exactly matches the historical (v2) flat genome
    layout, so existing genomes remain valid:
      encoder (W, b per layer) → GRU (r, z, h gates: W_in, W_hid, b)
      → policy head (W, b) → value head (W, b)

    Args:
        input_size: Size of observation vector
        encoder_layers: Sizes of encoder hidden layers (default: [32])
        gru_hidden_size: Size of GRU hidden state
        output_size: Number of actions

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
    return {
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


def build_brain_v3_param_spec(
    state_inputs: int = 22,
    embed_dim: int = 8,
    state_dim: int = 40,
    gru_hidden_size: int = 48,
    value_hidden: int = 16,
    output_size: int = 8,
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

    Args:
        state_inputs: Non-vision feature count (agent_state+stimulus+inventory)
        embed_dim: Per-tile embedding size (E)
        state_dim: State encoder output size (S); GRU input is S + E
        gru_hidden_size: GRU hidden state size (H)
        value_hidden: Hidden size of the value MLP
        output_size: Number of actions

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
    return {
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

    Stimulus fields are exposed as absolute indices into the vector.
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

    def vision_grid(self, observation: np.ndarray) -> np.ndarray:
        """
        Return the vision portion of an observation as a
        (rows, cols, features) grid view.

        Rows index the agent-relative dy (row 0 = furthest ahead),
        cols index dx (col < center = agent's left).
        """
        return np.asarray(observation)[self.vision].reshape(self.vision_shape)


def build_observation_spec(vision_radius: int = 2) -> ObservationSpec:
    """
    Build the ObservationSpec matching utils/agents/perception.py.

    Args:
        vision_radius: Vision grid radius (2 → 5×5 grid)

    Returns:
        ObservationSpec with derived slices and field indices
    """
    side = 2 * vision_radius + 1

    agent_state = slice(0, 8)
    vision = slice(agent_state.stop, agent_state.stop + side * side * 2)
    stimulus = slice(vision.stop, vision.stop + 8)
    inventory = slice(stimulus.stop, stimulus.stop + 6)

    s = stimulus.start
    return ObservationSpec(
        agent_state=agent_state,
        vision=vision,
        stimulus=stimulus,
        inventory=inventory,
        vision_shape=(side, side, 2),
        size=inventory.stop,
        food_on_tile=s + 0,
        seed_on_tile=s + 1,
        food_ahead=s + 2,
        resource_ahead=s + 3,
        nearest_food_prox=s + 4,
        food_dir_match=s + 5,
        energy_urgency=s + 6,
        can_interact=s + 7,
    )


# Default spec for the standard 72-feature observation (5×5 vision)
DEFAULT_OBSERVATION_SPEC = build_observation_spec(vision_radius=2)
