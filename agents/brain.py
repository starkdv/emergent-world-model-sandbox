"""
Neural network brain for agent decision making - v2 with GRU + Actor-Critic.

The brain is now a recurrent Actor-Critic architecture that:
- Maintains memory via GRU hidden state (enables long-horizon behaviors)
- Predicts both policy and value (stable temporal credit assignment)
- Supports action masking (prevents wasting probability on invalid actions)
- Uses policy sampling instead of epsilon-greedy (state-dependent exploration)

Architecture:
    Input (64) → Encoder MLP → GRU → Policy Head (8 actions) + Value Head (1 scalar)

Weights are still encoded in the agent's genome as a flat vector for evolution.

Author: Karan Vasa
Date: February 11, 2026
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional

from agents.actions import Action

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
        input_size: int = 64,
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
        self.params = self._unpack_weights(genome.weights)
    
    def _unpack_weights(self, flat_weights: np.ndarray) -> dict:
        """
        Unpack flattened weight vector into structured parameters.
        
        Creates separate dictionaries for:
        - Encoder (feedforward layers)
        - GRU (recurrent update gates)
        - Policy head (action logits)
        - Value head (state value)
        
        Args:
            flat_weights: Flattened weight vector from genome
            
        Returns:
            Dictionary of parameter matrices and biases
        """
        params = {
            'encoder_weights': [],
            'encoder_biases': [],
            'gru': {},
            'policy_head': {},
            'value_head': {}
        }
        
        idx = 0
        
        # 1. Encoder MLP
        encoder_sizes = [self.input_size] + self.encoder_layers
        for i in range(len(encoder_sizes) - 1):
            in_size = encoder_sizes[i]
            out_size = encoder_sizes[i + 1]
            
            # Weight matrix
            w_size = in_size * out_size
            w = flat_weights[idx:idx + w_size].reshape(in_size, out_size)
            params['encoder_weights'].append(w)
            idx += w_size
            
            # Bias vector
            b = flat_weights[idx:idx + out_size]
            params['encoder_biases'].append(b)
            idx += out_size
        
        # Encoder output size
        encoder_out = self.encoder_layers[-1]
        
        # 2. GRU parameters
        # GRU has 3 gates: reset (r), update (z), candidate (h_tilde)
        # Each gate: W_input @ x + W_hidden @ h + bias
        
        # Reset gate
        params['gru']['Wr_input'] = flat_weights[idx:idx + encoder_out * self.gru_hidden_size].reshape(encoder_out, self.gru_hidden_size)
        idx += encoder_out * self.gru_hidden_size
        params['gru']['Wr_hidden'] = flat_weights[idx:idx + self.gru_hidden_size * self.gru_hidden_size].reshape(self.gru_hidden_size, self.gru_hidden_size)
        idx += self.gru_hidden_size * self.gru_hidden_size
        params['gru']['br'] = flat_weights[idx:idx + self.gru_hidden_size]
        idx += self.gru_hidden_size
        
        # Update gate
        params['gru']['Wz_input'] = flat_weights[idx:idx + encoder_out * self.gru_hidden_size].reshape(encoder_out, self.gru_hidden_size)
        idx += encoder_out * self.gru_hidden_size
        params['gru']['Wz_hidden'] = flat_weights[idx:idx + self.gru_hidden_size * self.gru_hidden_size].reshape(self.gru_hidden_size, self.gru_hidden_size)
        idx += self.gru_hidden_size * self.gru_hidden_size
        params['gru']['bz'] = flat_weights[idx:idx + self.gru_hidden_size]
        idx += self.gru_hidden_size
        
        # Candidate hidden state
        params['gru']['Wh_input'] = flat_weights[idx:idx + encoder_out * self.gru_hidden_size].reshape(encoder_out, self.gru_hidden_size)
        idx += encoder_out * self.gru_hidden_size
        params['gru']['Wh_hidden'] = flat_weights[idx:idx + self.gru_hidden_size * self.gru_hidden_size].reshape(self.gru_hidden_size, self.gru_hidden_size)
        idx += self.gru_hidden_size * self.gru_hidden_size
        params['gru']['bh'] = flat_weights[idx:idx + self.gru_hidden_size]
        idx += self.gru_hidden_size
        
        # 3. Policy head (GRU hidden → action logits)
        params['policy_head']['W'] = flat_weights[idx:idx + self.gru_hidden_size * self.output_size].reshape(self.gru_hidden_size, self.output_size)
        idx += self.gru_hidden_size * self.output_size
        params['policy_head']['b'] = flat_weights[idx:idx + self.output_size]
        idx += self.output_size
        
        # 4. Value head (GRU hidden → scalar value)
        params['value_head']['W'] = flat_weights[idx:idx + self.gru_hidden_size].reshape(self.gru_hidden_size, 1)
        idx += self.gru_hidden_size
        params['value_head']['b'] = flat_weights[idx:idx + 1]
        idx += 1
        
        return params
    
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
        
        # Apply temperature scaling
        logits = logits / temperature
        
        # Softmax to get probabilities
        probs = self._softmax(logits)
        
        # 4. Value head: Estimate state value
        value = float(h_next @ self.params['value_head']['W'] + self.params['value_head']['b'])
        
        return probs, value, h_next
    
    def _gru_step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Single GRU step.
        
        GRU equations:
            r = sigmoid(x @ Wr_input + h @ Wr_hidden + br)   # reset gate
            z = sigmoid(x @ Wz_input + h @ Wz_hidden + bz)   # update gate
            h_tilde = tanh(x @ Wh_input + (r * h) @ Wh_hidden + bh)  # candidate
            h_next = (1 - z) * h + z * h_tilde               # new hidden state
        
        Args:
            x: Current input (encoder output)
            h: Previous hidden state
            
        Returns:
            New hidden state
        """
        gru = self.params['gru']
        
        # Reset gate
        r = self._sigmoid(x @ gru['Wr_input'] + h @ gru['Wr_hidden'] + gru['br'])
        
        # Update gate
        z = self._sigmoid(x @ gru['Wz_input'] + h @ gru['Wz_hidden'] + gru['bz'])
        
        # Candidate hidden state
        h_tilde = np.tanh(x @ gru['Wh_input'] + (r * h) @ gru['Wh_hidden'] + gru['bh'])
        
        # New hidden state
        h_next = (1 - z) * h + z * h_tilde
        
        return h_next
    
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
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """
        Compute softmax probabilities.
        
        Args:
            x: Input logits
            
        Returns:
            Probability distribution
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    @staticmethod
    def calculate_weight_count(
        input_size: int = 64,
        encoder_layers: list[int] = None,
        gru_hidden_size: int = 32,
        output_size: int = 8
    ) -> int:
        """
        Calculate total number of weights needed for the network.
        
        Args:
            input_size: Size of observation vector
            encoder_layers: Sizes of encoder hidden layers (default: [32])
            gru_hidden_size: Size of GRU hidden state
            output_size: Number of actions
            
        Returns:
            Total number of weights (including biases)
        """
        if encoder_layers is None:
            encoder_layers = [32]
        
        total = 0
        
        # 1. Encoder weights
        encoder_sizes = [input_size] + encoder_layers
        for i in range(len(encoder_sizes) - 1):
            in_size = encoder_sizes[i]
            out_size = encoder_sizes[i + 1]
            total += in_size * out_size + out_size  # weights + biases
        
        encoder_out = encoder_layers[-1]
        
        # 2. GRU weights (3 gates: reset, update, candidate)
        # Each gate: input_weights + hidden_weights + bias
        for _ in range(3):
            total += encoder_out * gru_hidden_size  # input weights
            total += gru_hidden_size * gru_hidden_size  # hidden weights
            total += gru_hidden_size  # bias
        
        # 3. Policy head
        total += gru_hidden_size * output_size + output_size  # weights + bias
        
        # 4. Value head
        total += gru_hidden_size + 1  # weight + bias (scalar output)
        
        return total
    
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
