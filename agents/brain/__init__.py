"""
Neural network brain for agent decision making — GRU + Actor-Critic.

The brain is a recurrent Actor-Critic architecture that:
- Maintains memory via GRU hidden state (enables long-horizon behaviors)
- Predicts both policy and value (stable temporal credit assignment)
- Supports action masking (prevents wasting probability on invalid actions)
- Uses policy sampling instead of epsilon-greedy (state-dependent exploration)

Architecture:
        Input (72) → Encoder MLP → GRU → Policy Head (8 actions) + Value Head (1 scalar)
Weights are encoded in the agent's genome as a flat vector for evolution;
the genome layout is defined once, declaratively, in agents/brain/spec.py
(ParamSpec). Bootstrap instinct biases live in agents/brain/instincts.py
and are applied to the logits — the network itself is a pure function of
(observation, hidden state, parameters).

Author: Karan Vasa
Date: February 11, 2026
Updated: June 2026 — spec-driven genome layout, instincts extracted
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional

from agents.actions import Action
from agents.brain import modules
from agents.brain.instincts import InstinctModule
from agents.brain.spec import (
    OBSERVATION_SPEC_V2,
    build_brain_param_spec,
    build_brain_v3_param_spec,
    build_nested_params,
    migrate_genome,
)


def _is_v35(version) -> bool:
    """True if the brain version selects the v3.5 (social) attention brain."""
    return version == 3.5 or str(version) in ("3.5", "3_5")


def _v35_state_inputs() -> int:
    """Non-vision input count for the v3.5 observation layout (= 28)."""
    s = OBSERVATION_SPEC_V2
    return (
        (s.agent_state.stop - s.agent_state.start)
        + (s.stimulus.stop - s.stimulus.start)
        + (s.inventory.stop - s.inventory.start)
        + (s.extra.stop - s.extra.start)
    )


if TYPE_CHECKING:
    from agents.genome import Genome


class Brain:
    """
    Recurrent Actor-Critic neural network policy for agent decision making.

    Architecture:
        - Encoder: MLP that processes observations
        - GRU: Recurrent layer for memory (hidden state)
        - Policy Head: Outputs action logits
        - Value Head: Outputs state value estimate

    Attributes:
        genome (Genome): Source of neural network weights
        input_size (int): Size of observation vector
        encoder_layers (List[int]): Sizes of encoder hidden layers
        gru_hidden_size (int): Size of GRU hidden state
        output_size (int): Number of actions
        spec (ParamSpec): Declarative genome layout
        named_params (dict): Flat name → tensor views into the genome
        params (dict): Nested parameters (same memory as named_params)
        instincts (InstinctModule): Bootstrap biases (None = pure network)
    """

    def __init__(
        self,
        genome: "Genome",
        input_size: int = 72,
        encoder_layers: Optional[list[int]] = None,
        gru_hidden_size: int = 32,
        output_size: int = 8,
        instincts: Optional[InstinctModule] = None,
        world_model_hidden: Optional[int] = None,
    ):
        """
        Initialize brain from genome.

        Args:
            genome: Genome containing neural network weights
            input_size: Size of observation vector
            encoder_layers: Sizes of encoder hidden layers (default: [32])
            gru_hidden_size: Size of GRU hidden state (default: 32)
            output_size: Number of possible actions
            instincts: Instinct module (default: standard InstinctModule;
                       pass InstinctModule(enabled=False) for a pure network)
            world_model_hidden: Hidden width of the latent dynamics head
                (None = no world model)
        """
        self.genome = genome
        self.input_size = input_size
        self.encoder_layers = encoder_layers if encoder_layers else [32]
        self.gru_hidden_size = gru_hidden_size
        self.output_size = output_size
        self.world_model_hidden = world_model_hidden
        self.instincts = instincts if instincts is not None else InstinctModule()

        # Declarative genome layout — single source of truth for
        # weight counting, unpacking, and packing.
        self.spec = build_brain_param_spec(
            self.input_size,
            self.encoder_layers,
            self.gru_hidden_size,
            self.output_size,
            world_model_hidden=world_model_hidden,
        )

        # Views into genome.weights (zero-copy), in two addressings:
        # flat named dict and the nested structure used by forward/learner.
        self.named_params = self.spec.unpack(genome.weights)
        self.params = self._build_nested(self.named_params)

    @property
    def has_world_model(self) -> bool:
        """True when the genome includes the latent dynamics head."""
        return "dynamics" in self.params

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """
        Public access to the perception latent z (used by curiosity and
        the planner to compare predictions against reality).
        """
        return self._encode(observation)

    def predict_next_latent(
        self, h: np.ndarray, action_idx: int
    ) -> Tuple[np.ndarray, float]:
        """
        World-model prediction: next latent and reward from the
        post-decision hidden state and a chosen action.

            d  = tanh([h ‖ onehot(a)]·W1 + b1)
            ẑ' = d·Wz + bz
            r̂  = d·Wr + br

        Args:
            h: GRU hidden state AFTER processing the current observation
            action_idx: Index of the action to imagine taking

        Returns:
            (predicted_next_latent, predicted_reward)

        Raises:
            RuntimeError: If the brain has no dynamics head
        """
        if not self.has_world_model:
            raise RuntimeError("Brain has no world model (dynamics head)")
        dyn = self.params["dynamics"]
        onehot = np.zeros(self.output_size, dtype=np.float32)
        onehot[action_idx] = 1.0
        d = np.tanh(np.concatenate([h, onehot]) @ dyn["W1"] + dyn["b1"])
        z_pred = d @ dyn["Wz"] + dyn["bz"]
        r_pred = float((d @ dyn["Wr"] + dyn["br"]).item())
        return z_pred, r_pred

    def policy_from_hidden(
        self, h: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Action probabilities directly from a (possibly imagined) GRU hidden
        state — the same masked softmax ``forward`` computes from ``h``, but
        without re-encoding an observation or applying instincts.

        Used by the latent planner to sample *policy-guided* rollout
        continuations (Dreamer-style imagination) instead of uniform-random
        actions, which keeps imagined trajectories in-distribution.

        Args:
            h: GRU hidden state.
            action_mask: Optional binary mask (1 = valid). Imagined steps past
                the first usually pass None (validity depends on world state
                the model does not expose).

        Returns:
            Probability vector over the ``output_size`` actions.
        """
        logits = h @ self.params["policy_head"]["W"] + self.params["policy_head"]["b"]
        if action_mask is not None:
            logits = np.where(action_mask > 0, logits, -1e9)
        return modules.softmax(logits)

    def initial_state(self) -> np.ndarray:
        """
        Get initial GRU hidden state (zeros).

        Returns:
            Zero-initialized hidden state vector
        """
        return np.zeros(self.gru_hidden_size, dtype=np.float32)

    def forward(
        self,
        observation: np.ndarray,
        h: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        instinct_strength: float = 1.0,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Forward pass through the network.

        Args:
            observation: Input observation vector
            h: Current GRU hidden state
            action_mask: Optional binary mask for valid actions (1=valid, 0=invalid)
            temperature: Sampling temperature (default: 1.0)
            instinct_strength: Scale factor for instinct biases (default: 1.0)

        Returns:
            Tuple of (action_probs, value, next_hidden_state)
        """
        # 1. Encode observation into the latent fed to the GRU
        z = self._encode(observation)

        # 2. GRU: Update hidden state with memory
        h_next = self._gru_step(z, h)

        # 3. Policy head: Compute action logits
        logits = (
            h_next @ self.params["policy_head"]["W"] + self.params["policy_head"]["b"]
        )

        if action_mask is not None:
            # Set logits of invalid actions to very negative value
            logits = np.where(action_mask > 0, logits, -1e9)

            # Bootstrap instinct biases (see agents/brain/instincts.py)
            if self.instincts is not None:
                logits = self.instincts.apply(
                    logits, observation, action_mask, strength=instinct_strength
                )

        # Apply temperature scaling
        logits = logits / temperature

        # Softmax to get probabilities
        probs = modules.softmax(logits)

        # 4. Value head: Estimate state value
        value = self._value(z, h_next)

        return probs, value, h_next

    def _encode(self, observation: np.ndarray) -> np.ndarray:
        """
        Encode an observation into the latent fed to the GRU.

        v2: plain MLP over the full observation vector.
        (Overridden by BrainV3 with attention-pooled perception.)
        """
        x = observation.flatten()
        for i in range(len(self.params["encoder_weights"])):
            x = np.tanh(
                x @ self.params["encoder_weights"][i] + self.params["encoder_biases"][i]
            )
        return x

    def _value(self, z: np.ndarray, h_next: np.ndarray) -> float:
        """
        Compute the state-value estimate.

        v2: linear head on the GRU hidden state only (``z`` unused).
        (Overridden by BrainV3, whose value MLP reads [z, h].)
        """
        return float(
            (
                h_next @ self.params["value_head"]["W"] + self.params["value_head"]["b"]
            ).item()
        )

    def rebind(self, genome: "Genome") -> None:
        """
        Re-bind this brain's parameter views to (possibly new) genome
        weights, keeping the architecture and instinct configuration.

        Use after replacing ``genome.weights`` with a new array (e.g.
        mutation that reallocates, inherited weights, loaded weights).

        Args:
            genome: Genome whose weights to bind to
        """
        self.genome = genome
        self.named_params = self.spec.unpack(genome.weights)
        self.params = self._build_nested(self.named_params)

    def _build_nested(self, named: dict) -> dict:
        """Build the nested params structure (overridden by BrainV3)."""
        return build_nested_params(named, len(self.encoder_layers))

    def _gru_step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Single GRU step - delegates to modules.gru_step.

        Args:
            x: Current input (encoder output)
            h: Previous hidden state

        Returns:
            New hidden state
        """
        return modules.gru_step(x, h, self.params["gru"])

    def decide(
        self,
        observation: np.ndarray,
        h: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        instinct_strength: float = 1.0,
    ) -> Tuple[Action, np.ndarray, float]:
        """
        Decide which action to take based on observation and hidden state.

        Samples from the policy distribution (no epsilon-greedy).

        Args:
            observation: Normalized observation vector
            h: Current GRU hidden state
            action_mask: Optional binary mask for valid actions
            temperature: Sampling temperature (higher = more exploration)
            instinct_strength: Scale factor for instinct biases

        Returns:
            Tuple of (selected_action, next_hidden_state, value_estimate)
        """
        # Forward pass
        action_probs, value, h_next = self.forward(
            observation, h, action_mask, temperature, instinct_strength
        )

        # Sample action from distribution
        action_idx = np.random.choice(len(action_probs), p=action_probs)

        return Action(action_idx), h_next, value

    def decide_with_logprob(
        self,
        observation: np.ndarray,
        h: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        instinct_strength: float = 1.0,
    ) -> Tuple[Action, np.ndarray, float, float]:
        """
        Like decide(), but also returns the log-probability of the
        sampled action under the behaviour policy (the full acting
        distribution: network + mask + instincts + temperature).

        Needed by the PPO learner, whose clipped importance ratio is
        π_new(a|s) / π_behaviour(a|s).

        Returns:
            Tuple of (selected_action, next_hidden_state, value_estimate,
            log_prob_of_selected_action)
        """
        action_probs, value, h_next = self.forward(
            observation, h, action_mask, temperature, instinct_strength
        )
        action_idx = np.random.choice(len(action_probs), p=action_probs)
        log_prob = float(np.log(max(action_probs[action_idx], 1e-8)))
        return Action(action_idx), h_next, value, log_prob

    @staticmethod
    def calculate_weight_count(
        input_size: int = 72,
        encoder_layers: Optional[list[int]] = None,
        gru_hidden_size: int = 32,
        output_size: int = 8,
        world_model_hidden: Optional[int] = None,
    ) -> int:
        """
        Calculate total number of weights needed for the network.
        Derived from the declarative ParamSpec.

        Args:
            input_size: Size of observation vector
            encoder_layers: Sizes of encoder hidden layers (default: [32])
            gru_hidden_size: Size of GRU hidden state
            output_size: Number of actions
            world_model_hidden: Dynamics-head hidden width (None = none)

        Returns:
            Total number of weights (including biases)
        """
        return build_brain_param_spec(
            input_size,
            encoder_layers,
            gru_hidden_size,
            output_size,
            world_model_hidden=world_model_hidden,
        ).count()

    def get_action_preferences(
        self, observation: np.ndarray, h: np.ndarray
    ) -> dict[str, float]:
        """
        Get action probabilities as a dictionary.

        Useful for debugging and visualization.

        Args:
            observation: Normalized observation vector
            h: Current GRU hidden state

        Returns:
            Dictionary mapping action names to probabilities
        """
        probs, _, _ = self.forward(observation, h)

        # Only actions this brain can actually emit (SIGNAL is absent for the
        # 8-wide v2/v3 policy heads).
        return {
            action.name: float(probs[action.value])
            for action in Action
            if action.value < len(probs)
        }


# ---------------------------------------------------------------------------
# Factory — single place that maps the YAML ``brain`` section to a Brain.
# Version 2 (default) is the legacy GRU-MLP; version 3 adds attention
# perception and a [z, h] value head (see agents/brain/v3.py).
# ---------------------------------------------------------------------------


def _world_model_hidden(brain_config: dict) -> Optional[int]:
    """Read the ``brain.world_model`` block: hidden width or None."""
    wm = brain_config.get("world_model", {}) or {}
    if wm.get("enabled", False):
        return wm.get("hidden", 32)
    return None


def _v3_kwargs(brain_config: dict) -> dict:
    """Extract BrainV3 size kwargs from a ``brain`` config dict."""
    v3 = brain_config.get("v3", {}) or {}
    return {
        "embed_dim": v3.get("embed_dim", 8),
        "state_dim": v3.get("state_dim", 40),
        "gru_hidden_size": v3.get("gru_hidden_size", 48),
        "value_hidden": v3.get("value_hidden", 16),
        "output_size": brain_config.get("output_size", 8),
        "world_model_hidden": _world_model_hidden(brain_config),
    }


def create_brain(
    genome: "Genome",
    brain_config: Optional[dict] = None,
    instincts: Optional[InstinctModule] = None,
) -> Brain:
    """
    Build a Brain matching the ``brain`` config section.

    Args:
        genome: Genome containing the network weights
        brain_config: ``brain`` config dict (None → v2 defaults)
        instincts: Instinct module to attach

    Returns:
        Brain (version 2) or BrainV3 (version 3)
    """
    cfg = brain_config or {}
    version = cfg.get("version", 2)
    if _is_v35(version):
        # Brain v3.5 = v3 attention brain with the Observation-v2 input block
        # (78-dim, state encoder 28→S) and the SIGNAL action (output 9).
        from agents.brain.v3 import BrainV3

        kwargs = _v3_kwargs(cfg)
        # SIGNAL is definitional to v3.5 → always 9 actions (the legacy
        # top-level ``output_size: 8`` is a v2 setting and is ignored here;
        # SIGNAL availability is controlled by signal.enabled via masking).
        kwargs["output_size"] = 9
        return BrainV3(
            genome, instincts=instincts, obs_spec=OBSERVATION_SPEC_V2, **kwargs
        )
    if version == 3:
        # Imported lazily: v3.py imports this module
        from agents.brain.v3 import BrainV3

        return BrainV3(genome, instincts=instincts, **_v3_kwargs(cfg))

    return Brain(
        genome,
        input_size=cfg.get("input_size", 72),
        encoder_layers=cfg.get("encoder_layers"),
        gru_hidden_size=cfg.get("gru_hidden_size", 32),
        output_size=cfg.get("output_size", 8),
        instincts=instincts,
        world_model_hidden=_world_model_hidden(cfg),
    )


def calculate_weight_count_for_config(brain_config: Optional[dict] = None) -> int:
    """
    Genome length required by the ``brain`` config section.

    Args:
        brain_config: ``brain`` config dict (None → v2 defaults)

    Returns:
        Total number of weights (including biases)
    """
    cfg = brain_config or {}
    version = cfg.get("version", 2)
    if _is_v35(version) or version == 3:
        from agents.brain.v3 import BrainV3

        kwargs = _v3_kwargs(cfg)
        if _is_v35(version):
            # v3.5: 28 non-vision inputs (Observation v2) + SIGNAL action (9).
            # output_size is fixed at 9 (see create_brain).
            state_inputs = _v35_state_inputs()
            output_size = 9
        else:
            state_inputs = 22
            output_size = kwargs["output_size"]
        return BrainV3.calculate_v3_weight_count(
            state_inputs=state_inputs,
            embed_dim=kwargs["embed_dim"],
            state_dim=kwargs["state_dim"],
            gru_hidden_size=kwargs["gru_hidden_size"],
            value_hidden=kwargs["value_hidden"],
            output_size=output_size,
            world_model_hidden=kwargs["world_model_hidden"],
        )

    return Brain.calculate_weight_count(
        input_size=cfg.get("input_size", 72),
        encoder_layers=cfg.get("encoder_layers"),
        gru_hidden_size=cfg.get("gru_hidden_size", 32),
        output_size=cfg.get("output_size", 8),
        world_model_hidden=_world_model_hidden(cfg),
    )


def _v3_spec_for(cfg: dict, state_inputs: int, output_size: int):
    """Build a v3-family ParamSpec for the given cfg, inputs, and actions."""
    k = _v3_kwargs(cfg)
    return build_brain_v3_param_spec(
        state_inputs=state_inputs,
        embed_dim=k["embed_dim"],
        state_dim=k["state_dim"],
        gru_hidden_size=k["gru_hidden_size"],
        value_hidden=k["value_hidden"],
        output_size=output_size,
        world_model_hidden=k["world_model_hidden"],
    )


def adapt_loaded_genome(flat, brain_config: Optional[dict] = None):
    """
    Make a loaded flat genome fit the configured brain, migrating if needed.

    - If the length already matches the config, returns it unchanged.
    - If the config is **v3.5** and the genome is a **v3** layout (the prior
      version), it is migrated via ``migrate_genome`` (the EXTRA observation
      rows and the SIGNAL policy column are zero-filled, so the loaded brain
      behaves bit-identically on the original actions until those weights are
      trained/evolved).
    - Otherwise returns ``None`` so the caller can warn and fall back.

    Args:
        flat: Loaded flat weight vector
        brain_config: Target ``brain`` config

    Returns:
        A flat genome of the configured length, or None if it can't be adapted
    """
    import numpy as np

    cfg = brain_config or {}
    flat = np.asarray(flat)
    expected = calculate_weight_count_for_config(cfg)
    if flat.shape == (expected,):
        return flat

    if _is_v35(cfg.get("version", 2)):
        old_spec = _v3_spec_for(cfg, state_inputs=22, output_size=8)
        if flat.shape == (old_spec.count(),):
            new_spec = _v3_spec_for(
                cfg, state_inputs=_v35_state_inputs(), output_size=9
            )
            return migrate_genome(flat, old_spec, new_spec)

    return None
