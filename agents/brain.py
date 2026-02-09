"""
Neural network brain for agent decision making.

The brain is a simple Multi-Layer Perceptron (MLP) that maps
observations to action probabilities. Weights are encoded in
the agent's genome and evolved over generations.

Author: Karan Vasa
Date: November 14, 2025
"""

import numpy as np
from typing import TYPE_CHECKING

from agents.actions import Action

if TYPE_CHECKING:
    from agents.genome import Genome


class Brain:
    """
    Neural network policy for agent decision making.
    
    Architecture:
        - Input layer: observation vector (normalized sensory data)
        - Hidden layers: configurable size (default: [32, 16])
        - Output layer: action probabilities (8 actions)
        - Activation: tanh for hidden layers, softmax for output
    
    The brain's weights come from the agent's genome and are not
    trained during the agent's lifetime (pure genetic evolution).
    
    Attributes:
        genome (Genome): Source of neural network weights
        input_size (int): Size of observation vector
        hidden_sizes (List[int]): Sizes of hidden layers
        output_size (int): Number of actions
        weights (List[np.ndarray]): Layer weight matrices
        biases (List[np.ndarray]): Layer bias vectors
    """
    
    def __init__(
        self,
        genome: 'Genome',
        input_size: int = 64,
        hidden_sizes: list[int] = None,
        output_size: int = 8,
    ):
        """
        Initialize brain from genome.
        
        Args:
            genome: Genome containing neural network weights
            input_size: Size of observation vector
            hidden_sizes: Sizes of hidden layers (default: [32, 16])
            output_size: Number of possible actions
        """
        self.genome = genome
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if hidden_sizes else [32, 16]
        self.output_size = output_size
        
        # Unpack weights from genome
        self.weights, self.biases = self._unpack_weights(genome.weights)
    
    def _unpack_weights(
        self,
        flat_weights: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Unpack flattened weight vector into layer matrices.
        
        Args:
            flat_weights: Flattened weight vector from genome
            
        Returns:
            Tuple of (weight_matrices, bias_vectors)
        """
        weights = []
        biases = []
        
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        idx = 0
        
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            # Extract weight matrix
            w_size = in_size * out_size
            w = flat_weights[idx:idx + w_size].reshape(in_size, out_size)
            weights.append(w)
            idx += w_size
            
            # Extract bias vector
            b = flat_weights[idx:idx + out_size]
            biases.append(b)
            idx += out_size
        
        return weights, biases
    def decide(self, observation: np.ndarray, epsilon: float = 0.1) -> Action:
        """
        Decide which action to take based on observation.
        
        Uses epsilon-greedy exploration: with probability epsilon,
        choose a random action; otherwise sample from policy.
        
        Args:
            observation: Normalized observation vector
            epsilon: Exploration probability (default: 0.1 = 10% random)
            
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            # Random exploration
            action_idx = np.random.randint(0, self.output_size)
        else:
            # Forward pass through network
            action_probs = self.forward(observation)
            
            # Sample action from probability distribution
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        return Action(action_idx)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through neural network.
        
        Args:
            x: Input observation vector
            
        Returns:
            Action probability distribution
        """
        # Ensure input is 1D
        x = x.flatten()
        
        # Pass through hidden layers with tanh activation
        for i in range(len(self.weights) - 1):
            x = np.tanh(x @ self.weights[i] + self.biases[i])
        
        # Output layer with softmax
        logits = x @ self.weights[-1] + self.biases[-1]
        probs = self._softmax(logits)
        
        return probs
    
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
        hidden_sizes: list[int] = None,
        output_size: int = 8
    ) -> int:
        """
        Calculate total number of weights needed for the network.
        
        Args:
            input_size: Size of observation vector
            hidden_sizes: Sizes of hidden layers
            output_size: Number of actions
            
        Returns:
            Total number of weights (including biases)
        """
        if hidden_sizes is None:
            hidden_sizes = [32, 16]
        
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        total = 0
        
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            # Weights + biases
            total += in_size * out_size + out_size
        
        return total
    
    def get_action_preferences(self, observation: np.ndarray) -> dict[str, float]:
        """
        Get action probabilities as a dictionary.
        
        Useful for debugging and visualization.
        
        Args:
            observation: Normalized observation vector
            
        Returns:
            Dictionary mapping action names to probabilities
        """
        probs = self.forward(observation)
        
        return {
            action.name: float(probs[action.value])
            for action in Action
        }
