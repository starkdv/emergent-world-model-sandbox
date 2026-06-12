"""
Brain v3 — attention perception, larger GRU, [z, h] value head.

Architecture (config: ``brain.version: 3``):

    vision 5×5×2 ──→ 25 tile tokens (+ fixed positional encoding)
                     → shared embedding (4→E) → keys/values
    agent state +
    stimulus +      ──→ state encoder (22→S) ──→ attention query
    inventory                │
                             ▼
        latent z = [state S | attention-pooled vision E]
                             │
                  GRU (H, input = S+E)
                             │ h
        policy head (H → actions, masked)  ·  value MLP ([z, h] → V → 1)

Why this shape (see docs/BRAIN_V3_PROPOSAL.md §3 for the full rationale):

- The tile embedding is ONE small matrix shared by all 25 tiles, so
  perception is position-equivariant and scales to larger vision radii
  with the SAME weights — unlike v2's dense vision layer, which spends
  ~1,600 parameters memorising tile positions.
- A single attention query derived from the agent's internal state pools
  the tiles, letting the network focus on relevant tiles (e.g. food when
  hungry) instead of treating all 25 cells equally.
- The value head reads [z, h] — the critic gets a direct view of the
  current state instead of only what the GRU chose to remember.

The public API is identical to Brain (initial_state / forward / decide /
rebind), so agents, evolution, and logging work unchanged. Default sizes
(E=8, S=40, H=48, V=16) give ≈17k parameters — still compact enough for
thousands of concurrent agents and genome-based evolution.

Author: Karan Vasa
Date: June 2026
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from agents.brain import Brain
from agents.brain.instincts import InstinctModule
from agents.brain.spec import (
    DEFAULT_OBSERVATION_SPEC,
    ObservationSpec,
    build_brain_v3_param_spec,
    build_nested_params_v3,
)

if TYPE_CHECKING:
    from agents.genome import Genome


def make_positional_encoding(vision_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Fixed 2-D positional encoding for the vision grid tokens.

    Each tile token gets (row, col) normalised to [-1, 1] appended to its
    two observation features. The encoding is constant (not learned).

    Args:
        vision_shape: (rows, cols, features) of the vision grid

    Returns:
        (rows*cols, 2) array of positional features
    """
    rows, cols, _ = vision_shape
    r = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
    c = np.linspace(-1.0, 1.0, cols, dtype=np.float32)
    grid_r, grid_c = np.meshgrid(r, c, indexing="ij")
    return np.stack([grid_r.ravel(), grid_c.ravel()], axis=1)


class BrainV3(Brain):
    """
    Attention-perception Actor-Critic brain (Brain v3).

    Inherits forward/decide/instinct handling from Brain and overrides
    the observation encoder, value head, and genome layout.

    Attributes (in addition to Brain's):
        embed_dim (int): Per-tile embedding size (E)
        state_dim (int): State encoder output size (S); GRU input = S+E
        value_hidden (int): Hidden size of the value MLP
        obs_spec (ObservationSpec): Named observation layout
        pos_enc (np.ndarray): Fixed (tiles, 2) positional encoding
    """

    VERSION = 3

    def __init__(
        self,
        genome: "Genome",
        embed_dim: int = 8,
        state_dim: int = 40,
        gru_hidden_size: int = 48,
        value_hidden: int = 16,
        output_size: int = 8,
        instincts: Optional[InstinctModule] = None,
        obs_spec: ObservationSpec = DEFAULT_OBSERVATION_SPEC,
        world_model_hidden: Optional[int] = None,
    ):
        """
        Initialize Brain v3 from genome.

        Args:
            genome: Genome containing neural network weights
            embed_dim: Per-tile embedding size (E)
            state_dim: State encoder output size (S)
            gru_hidden_size: GRU hidden state size (H)
            value_hidden: Hidden size of the value MLP
            output_size: Number of possible actions
            instincts: Instinct module (default: standard InstinctModule)
            obs_spec: Observation layout specification
            world_model_hidden: Hidden width of the latent dynamics head
                (None = no world model)
        """
        # Deliberately not calling Brain.__init__ — v3 has its own
        # architecture parameters and spec; shared behaviour
        # (forward/decide/rebind/gru) is inherited as methods.
        self.genome = genome
        self.obs_spec = obs_spec
        self.input_size = obs_spec.size
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.gru_hidden_size = gru_hidden_size
        self.value_hidden = value_hidden
        self.output_size = output_size
        self.world_model_hidden = world_model_hidden
        self.instincts = instincts if instincts is not None else InstinctModule()

        # Non-vision features: agent_state + stimulus + inventory
        self.state_inputs = (
            (obs_spec.agent_state.stop - obs_spec.agent_state.start)
            + (obs_spec.stimulus.stop - obs_spec.stimulus.start)
            + (obs_spec.inventory.stop - obs_spec.inventory.start)
        )

        self.pos_enc = make_positional_encoding(obs_spec.vision_shape)

        self.spec = build_brain_v3_param_spec(
            state_inputs=self.state_inputs,
            embed_dim=embed_dim,
            state_dim=state_dim,
            gru_hidden_size=gru_hidden_size,
            value_hidden=value_hidden,
            output_size=output_size,
            world_model_hidden=world_model_hidden,
        )
        self.named_params = self.spec.unpack(genome.weights)
        self.params = self._build_nested(self.named_params)

    def _build_nested(self, named: dict) -> dict:
        """Build the v3 nested params structure."""
        return build_nested_params_v3(named)

    def _encode(self, observation: np.ndarray) -> np.ndarray:
        """
        Encode an observation into the latent z = [state | attended vision].

        1. State path: non-vision features → tanh linear → s (S)
        2. Vision path: 25 tile tokens (2 features + 2 positional)
           → shared embedding → tanh → t (25×E)
        3. Attention: query from s; softmax(k·q/√E) pools the values
        4. z = concat(s, pooled)  — fed to the GRU and the value MLP
        """
        obs = observation.flatten()
        spec_o = self.obs_spec
        p = self.params

        # 1. State path
        state_feats = np.concatenate(
            [obs[spec_o.agent_state], obs[spec_o.stimulus], obs[spec_o.inventory]]
        )
        s = np.tanh(state_feats @ p["state_enc"]["W"] + p["state_enc"]["b"])

        # 2. Vision path: tokens with positional encoding
        tiles = obs[spec_o.vision].reshape(-1, spec_o.vision_shape[2])
        tokens = np.concatenate([tiles, self.pos_enc], axis=1)  # (25, 4)
        t = np.tanh(tokens @ p["tile_embed"]["W"] + p["tile_embed"]["b"])  # (25, E)

        # 3. Single-head attention pool (query from agent state)
        q = s @ p["attn"]["Wq"]  # (E,)
        k = t @ p["attn"]["Wk"]  # (25, E)
        v = t @ p["attn"]["Wv"]  # (25, E)
        scores = (k @ q) / np.sqrt(self.embed_dim)
        scores = scores - np.max(scores)  # numerical stability
        weights = np.exp(scores)
        weights = weights / np.sum(weights)
        pooled = weights @ v  # (E,)

        # 4. Latent
        return np.concatenate([s, pooled])

    def _value(self, z: np.ndarray, h_next: np.ndarray) -> float:
        """
        Value MLP over [z, h]: tanh hidden layer, then linear scalar.
        """
        vm = self.params["value_mlp"]
        zh = np.concatenate([z, h_next])
        hidden = np.tanh(zh @ vm["W1"] + vm["b1"])
        return float((hidden @ vm["W2"] + vm["b2"]).item())

    @staticmethod
    def calculate_v3_weight_count(
        state_inputs: int = 22,
        embed_dim: int = 8,
        state_dim: int = 40,
        gru_hidden_size: int = 48,
        value_hidden: int = 16,
        output_size: int = 8,
        world_model_hidden: Optional[int] = None,
    ) -> int:
        """
        Calculate total number of weights for the v3 network.
        Derived from the declarative ParamSpec.

        Args:
            state_inputs: Non-vision feature count
            embed_dim: Per-tile embedding size (E)
            state_dim: State encoder output size (S)
            gru_hidden_size: GRU hidden state size (H)
            value_hidden: Hidden size of the value MLP
            output_size: Number of actions
            world_model_hidden: Dynamics-head hidden width (None = none)

        Returns:
            Total number of weights (including biases)
        """
        return build_brain_v3_param_spec(
            state_inputs,
            embed_dim,
            state_dim,
            gru_hidden_size,
            value_hidden,
            output_size,
            world_model_hidden=world_model_hidden,
        ).count()
