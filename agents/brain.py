"""
Neural network brain for agent decision making - v2 with GRU + Actor-Critic.

The brain is now a recurrent Actor-Critic architecture that:
- Maintains memory via GRU hidden state (enables long-horizon behaviors)
- Predicts both policy and value (stable temporal credit assignment)
- Supports action masking (prevents wasting probability on invalid actions)
- Uses policy sampling instead of epsilon-greedy (state-dependent exploration)

Architecture:
        Input (72) → Encoder MLP → GRU → Policy Head (8 actions) + Value Head (1 scalar)
Weights are still encoded in the agent's genome as a flat vector for evolution.

Author: Karan Vasa
Date: February 11, 2026
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional

from agents.actions import Action
import utils.agents.brain_utils as brain_utils

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
    
    Key improvements over v1:
        - Memory (GRU) enables long-horizon strategies
        - Actor-Critic enables stable learning with temporal credit
        - Action masking prevents invalid actions
        - Policy sampling (no epsilon-greedy)
    
    Attributes:
        genome (Genome): Source of neural network weights
        input_size (int): Size of observation vector
        encoder_layers (List[int]): Sizes of encoder hidden layers
        gru_hidden_size (int): Size of GRU hidden state
        output_size (int): Number of actions
        params (dict): Unpacked parameters for each component
    """
    
    def __init__(
        self,
        genome: 'Genome',
        input_size: int = 72,
        encoder_layers: list[int] = None,
        gru_hidden_size: int = 32,
        output_size: int = 8,
    ):
        """
        Initialize brain from genome.
        
        Args:
            genome: Genome containing neural network weights
            input_size: Size of observation vector
            encoder_layers: Sizes of encoder hidden layers (default: [32])
            gru_hidden_size: Size of GRU hidden state (default: 32)
            output_size: Number of possible actions
        """
        self.genome = genome
        self.input_size = input_size
        self.encoder_layers = encoder_layers if encoder_layers else [32]
        self.gru_hidden_size = gru_hidden_size
        self.output_size = output_size
        
        # Unpack weights from genome into structured parameters
        self.params = brain_utils.unpack_weights(
            genome.weights,
            self.input_size,
            self.encoder_layers,
            self.gru_hidden_size,
            self.output_size
        )
    

    
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
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Forward pass through the network.
        
        Args:
            observation: Input observation vector
            h: Current GRU hidden state
            action_mask: Optional binary mask for valid actions (1=valid, 0=invalid)
            temperature: Sampling temperature (default: 1.0)
            
        Returns:
            Tuple of (action_probs, value, next_hidden_state)
        """
        # Ensure input is 1D
        x = observation.flatten()
        
        # 1. Encoder: Process observation
        for i in range(len(self.params['encoder_weights'])):
            x = np.tanh(x @ self.params['encoder_weights'][i] + self.params['encoder_biases'][i])
        
        # 2. GRU: Update hidden state with memory
        h_next = self._gru_step(x, h)
        
        # 3. Policy head: Compute action logits
        logits = h_next @ self.params['policy_head']['W'] + self.params['policy_head']['b']
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set logits of invalid actions to very negative value
            logits = np.where(action_mask > 0, logits, -1e9)

            # ----------------------------------------------------------
            # Contextual instinct biases
            #
            # Unlike the removed unconditional MOVE_FORWARD bias, these
            # only fire when the action is contextually appropriate
            # (action mask says it's valid).  They bootstrap survival
            # behaviour so agents with random/early-evolution weights
            # actually interact with the environment.  As the brain's
            # own weights strengthen through learning, the logits it
            # produces will dominate over these small additive biases.
            # ----------------------------------------------------------
            # PICK_UP (index 3) — strong instinct to pick things up
            if action_mask[Action.PICK_UP.value] > 0:
                logits[Action.PICK_UP.value] += 1.5

            # EAT (index 5) — instinct to eat when food in inventory
            if action_mask[Action.EAT.value] > 0:
                logits[Action.EAT.value] += 1.0

            # USE / plant (index 6) — instinct to plant seeds
            if action_mask[Action.USE.value] > 0:
                logits[Action.USE.value] += 0.5

            # ----------------------------------------------------------
            # DIRECTION-AWARE TURN TOWARD FOOD instinct
            #
            # When food is nearby but NOT in the agent's facing direction,
            # bias the CORRECT turn using the egocentric vision grid.
            # The 5×5 vision (indices 8-57, 2 features per tile) is
            # already rotated so dx<0 = agent's left, dx>0 = right.
            # We sum food-like type encodings on each side to decide
            # which turn actually points toward the food.
            #
            # Observation stimulus layout (indices 58-65):
            #   [60] food_ahead        (1.0 if food within 3 tiles ahead)
            #   [62] nearest_food_prox (1.0 = on food, 0 = far/none)
            #   [63] food_dir_match    (1.0 = facing food, 0 = facing away)
            # ----------------------------------------------------------
            if len(observation) > 63:
                food_prox = observation[62]       # nearest_food_prox
                food_dir  = observation[63]        # food_dir_match
                food_ahead_sig = observation[60]   # food_ahead

                # Food is nearby but agent isn't facing it — encourage turning
                if food_prox > 0.2 and food_dir < 0.6 and food_ahead_sig < 0.5:
                    base_bias = 0.8 * food_prox

                    # Scan egocentric vision to decide left vs right.
                    # Vision grid: 5 rows (dy -2..+2) × 5 cols (dx -2..+2),
                    # 2 features each, starting at obs index 8.
                    left_score = 0.0
                    right_score = 0.0
                    for row in range(5):          # dy offset
                        for col in range(5):      # dx offset
                            idx = 8 + (row * 5 + col) * 2  # type_enc
                            if idx < len(observation):
                                t_enc = observation[idx]
                                if t_enc >= 0.5:  # food / plant / seed
                                    dx_ego = col - 2
                                    if dx_ego < 0:
                                        left_score += t_enc
                                    elif dx_ego > 0:
                                        right_score += t_enc

                    # Bias the turn that faces the food side
                    if left_score > right_score + 0.1:
                        if action_mask[Action.TURN_LEFT.value] > 0:
                            logits[Action.TURN_LEFT.value] += base_bias
                        if action_mask[Action.TURN_RIGHT.value] > 0:
                            logits[Action.TURN_RIGHT.value] += base_bias * 0.2
                    elif right_score > left_score + 0.1:
                        if action_mask[Action.TURN_RIGHT.value] > 0:
                            logits[Action.TURN_RIGHT.value] += base_bias
                        if action_mask[Action.TURN_LEFT.value] > 0:
                            logits[Action.TURN_LEFT.value] += base_bias * 0.2
                    else:
                        # Food roughly centred / behind — boost both equally
                        if action_mask[Action.TURN_LEFT.value] > 0:
                            logits[Action.TURN_LEFT.value] += base_bias
                        if action_mask[Action.TURN_RIGHT.value] > 0:
                            logits[Action.TURN_RIGHT.value] += base_bias
        
        # Apply temperature scaling
        logits = logits / temperature
        
        # Softmax to get probabilities
        probs = brain_utils.softmax(logits)
        
        # 4. Value head: Estimate state value
        value = float(h_next @ self.params['value_head']['W'] + self.params['value_head']['b'])
        
        return probs, value, h_next
    
    def _gru_step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Single GRU step - delegates to brain_utils.
        
        Args:
            x: Current input (encoder output)
            h: Previous hidden state
            
        Returns:
            New hidden state
        """
        return brain_utils.gru_step(x, h, self.params['gru'])
    
    def decide(
        self,
        observation: np.ndarray,
        h: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> Tuple[Action, np.ndarray, float]:
        """
        Decide which action to take based on observation and hidden state.
        
        Samples from the policy distribution (no epsilon-greedy).
        
        Args:
            observation: Normalized observation vector
            h: Current GRU hidden state
            action_mask: Optional binary mask for valid actions
            temperature: Sampling temperature (higher = more exploration)
            
        Returns:
            Tuple of (selected_action, next_hidden_state, value_estimate)
        """
        # Forward pass
        action_probs, value, h_next = self.forward(observation, h, action_mask, temperature)
        
        # Sample action from distribution
        action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        return Action(action_idx), h_next, value
    
    @staticmethod
    def calculate_weight_count(
        input_size: int = 72,
        encoder_layers: list[int] = None,
        gru_hidden_size: int = 32,
        output_size: int = 8
    ) -> int:
        """
        Calculate total number of weights needed for the network.
        Delegates to brain_utils.
        
        Args:
            input_size: Size of observation vector
            encoder_layers: Sizes of encoder hidden layers (default: [32])
            gru_hidden_size: Size of GRU hidden state
            output_size: Number of actions
            
        Returns:
            Total number of weights (including biases)
        """
        return brain_utils.calculate_weight_count(
            input_size, encoder_layers, gru_hidden_size, output_size
        )
    
    def get_action_preferences(self, observation: np.ndarray, h: np.ndarray) -> dict[str, float]:
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
        
        return {
            action.name: float(probs[action.value])
            for action in Action
        }
