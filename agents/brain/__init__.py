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
from agents.brain.spec import build_brain_param_spec, build_nested_params

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
        """
        self.genome = genome
        self.input_size = input_size
        self.encoder_layers = encoder_layers if encoder_layers else [32]
        self.gru_hidden_size = gru_hidden_size
        self.output_size = output_size
        self.instincts = instincts if instincts is not None else InstinctModule()

        # Declarative genome layout — single source of truth for
        # weight counting, unpacking, and packing.
        self.spec = build_brain_param_spec(
            self.input_size,
            self.encoder_layers,
            self.gru_hidden_size,
            self.output_size,
        )

        # Views into genome.weights (zero-copy), in two addressings:
        # flat named dict and the nested structure used by forward/learner.
        self.named_params = self.spec.unpack(genome.weights)
        self.params = build_nested_params(self.named_params, len(self.encoder_layers))

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
        # Ensure input is 1D
        x = observation.flatten()

        # 1. Encoder: Process observation
        for i in range(len(self.params["encoder_weights"])):
            x = np.tanh(
                x @ self.params["encoder_weights"][i] + self.params["encoder_biases"][i]
            )

        # 2. GRU: Update hidden state with memory
        h_next = self._gru_step(x, h)

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
        value = float(
            (
                h_next @ self.params["value_head"]["W"] + self.params["value_head"]["b"]
            ).item()
        )

        return probs, value, h_next

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

    @staticmethod
    def calculate_weight_count(
        input_size: int = 72,
        encoder_layers: Optional[list[int]] = None,
        gru_hidden_size: int = 32,
        output_size: int = 8,
    ) -> int:
        """
        Calculate total number of weights needed for the network.
        Derived from the declarative ParamSpec.

        Args:
            input_size: Size of observation vector
            encoder_layers: Sizes of encoder hidden layers (default: [32])
            gru_hidden_size: Size of GRU hidden state
            output_size: Number of actions

        Returns:
            Total number of weights (including biases)
        """
        return build_brain_param_spec(
            input_size, encoder_layers, gru_hidden_size, output_size
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

        return {action.name: float(probs[action.value]) for action in Action}
