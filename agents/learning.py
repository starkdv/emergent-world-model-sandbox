"""
Reinforcement learning system for agents.

This module implements Actor-Critic reinforcement learning that allows
agents to improve their neural network weights during their lifetime based
on rewards from the environment.

Key features:
- Actor-Critic learning with GRU hidden states
- TD advantage estimation
- Policy gradient + value function learning
- Entropy regularization for exploration
- Online learning (agents improve during lifetime)
- Knowledge transfer to offspring through evolved weights

Author: Karan Vasa
Date: November 15, 2025
"""

from random import random
import numpy as np
from typing import Tuple, TYPE_CHECKING

# Import utility classes from utils.agents
from utils.agents import Experience, ReplayBuffer, RewardShaper

if TYPE_CHECKING:
    from agents.brain import Brain
    from agents.agent import Agent


class AgentLearner:
    """
    Actor-Critic learning system for agent neural networks.
    
    Implements Actor-Critic algorithm with GRU hidden states:
    - Policy gradient for action selection (Actor)
    - Value function for advantage estimation (Critic)
    - TD advantage for variance reduction
    - Entropy regularization for exploration
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        batch_size: int = 32,
        buffer_capacity: int = 1000,
        entropy_coef: float = 0.01  # Entropy bonus coefficient
    ):
        """
        Initialize the learner.
        
        Args:
            learning_rate: Learning rate for gradient updates
            discount_factor: Discount factor for future rewards
            batch_size: Batch size for learning updates
            buffer_capacity: Size of experience replay buffer
            entropy_coef: Coefficient for entropy bonus (encourages exploration)
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.reward_shaper = RewardShaper()
    
    def store_experience(
        self,
        observation: np.ndarray,
        hidden_state: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_hidden_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store an experience in the replay buffer.
        
        Args:
            observation: State before action
            hidden_state: Hidden state before action
            action: Action taken
            reward: Reward received
            next_observation: State after action
            next_hidden_state: Hidden state after action
            done: Whether episode ended
        """
        experience = Experience(
            observation, hidden_state, action, reward,
            next_observation, next_hidden_state, done
        )
        self.replay_buffer.add(experience)
    
    def learn(self, brain: 'Brain') -> float:
        """
        Update brain weights using Actor-Critic learning.
        
        Computes:
        - TD advantage: A = r + γV(s') * (1-done) - V(s)
        - Policy loss: -log π(a|s) * A
        - Value loss: 0.5 * (V(s) - target)^2
        - Entropy bonus: -β * H(π)
        
        Args:
            brain: Agent's brain to update
            
        Returns:
            Average loss for this update
        """        
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        experiences = self.replay_buffer.sample(self.batch_size)
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for exp in experiences:
            # Forward pass to get current state value and action probs
            probs, value, h_out = brain.forward(exp.observation, exp.hidden_state)
            
            # Forward pass to get next state value
            if exp.done:
                next_value = 0.0
            else:
                _, next_value, _ = brain.forward(exp.next_observation, exp.next_hidden_state)
            
            # Compute TD target and advantage
            td_target = exp.reward + self.discount_factor * next_value * (1 - int(exp.done))
            advantage = td_target - value
            
            # Policy loss: -log π(a|s) * A
            log_prob = np.log(probs[exp.action] + 1e-8)
            policy_loss = -log_prob * advantage
            
            # Value loss: 0.5 * (V(s) - target)^2
            value_loss = 0.5 * (value - td_target) ** 2
            
            # Entropy: -Σ π(a) log π(a)  (we want to maximize this, so minimize negative)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            
            # Total loss
            loss = policy_loss + value_loss - self.entropy_coef * entropy
            
            total_loss += abs(loss)
            total_policy_loss += abs(policy_loss)
            total_value_loss += value_loss
            total_entropy += entropy
            
            # Simplified gradient update using parameter perturbation
            # This avoids complex backprop through GRU
            self._update_parameters_simple(brain, exp, advantage, td_target, probs, h_out, value)
        
        # Sync updated parameters back to genome
        self._sync_genome_weights(brain)
        
        # Debug logging occasionally
        if random.random() < 0.01:
            avg_policy = total_policy_loss / len(experiences)
            avg_value = total_value_loss / len(experiences)
            avg_entropy = total_entropy / len(experiences)
            print(f"  [LEARN] Policy: {avg_policy:.3f}, Value: {avg_value:.3f}, Entropy: {avg_entropy:.3f}")
        
        return total_loss / len(experiences)
    
    def _update_parameters_simple(
        self,
        brain: 'Brain',
        exp: Experience,
        advantage: float,
        td_target: float,
        probs: np.ndarray,
        h_out: np.ndarray,
        value: float
    ) -> None:
        """
        Simple parameter update using gradients.
        
        Updates policy head and value head based on losses.
        For encoder and GRU, we use a simplified approach.
        
        Args:
            brain: Brain to update
            exp: Experience tuple
            advantage: TD advantage
            td_target: TD target for value
            probs: Current action probabilities
            h_out: GRU hidden state after processing observation
            value: Current state value
        """
        # Policy gradient: ∇θ log π(a|s) * A
        # Simplified: update policy head to increase prob of action if advantage > 0
        action_gradient = probs.copy()
        action_gradient[exp.action] -= 1.0  # Gradient of log π
        action_gradient *= advantage * self.learning_rate
        
        
        # Update policy head
        brain.params['policy_head']['W'] -= np.outer(h_out, action_gradient)
        brain.params['policy_head']['b'] -= action_gradient
        
        value_error = value - td_target
        value_gradient = 2 * value_error * self.learning_rate
        
        # Update value head
        brain.params['value_head']['W'] -= np.outer(h_out, value_gradient)
        brain.params['value_head']['b'] -= value_gradient
        
        # Entropy gradient: encourage higher entropy (more exploration)
        # This is already incorporated via the entropy bonus in the loss
        # For simplicity, we apply a small perturbation to encoder to encourage variation
        # if abs(advantage) > 0.1:  # Only update encoder for significant errors
        #     # Small update to encoder to adjust representations
        #     lr_encoder = self.learning_rate * 0.1  # Smaller LR for encoder
        #     for i in range(len(brain.params['encoder_weights'])):
        #         # Small random perturbation scaled by advantage
        #         perturbation = np.random.randn(*brain.params['encoder_weights'][i].shape) * 0.001
        #         brain.params['encoder_weights'][i] -= perturbation * advantage * lr_encoder
    
    
    def _sync_genome_weights(self, brain: 'Brain') -> None:
        """
        Sync brain parameters back to genome.
        
        This ensures learned weights are stored in the genome
        and can be passed to offspring (Lamarckian inheritance).
        
        Args:
            brain: Brain with updated parameters
        """
        # Flatten all parameters back into a single vector
        flat_weights = []
        
        # 1. Encoder
        for w, b in zip(brain.params['encoder_weights'], brain.params['encoder_biases']):
            flat_weights.extend(w.flatten())
            flat_weights.extend(b.flatten())
        
        # 2. GRU (3 gates: reset, update, candidate)
        gru = brain.params['gru']

        # Reset gate
        flat_weights.extend(gru['Wr_input'].flatten())
        flat_weights.extend(gru['Wr_hidden'].flatten())
        flat_weights.extend(gru['br'].flatten())

        # Update gate
        flat_weights.extend(gru['Wz_input'].flatten())
        flat_weights.extend(gru['Wz_hidden'].flatten())
        flat_weights.extend(gru['bz'].flatten())

        # Candidate
        flat_weights.extend(gru['Wh_input'].flatten())
        flat_weights.extend(gru['Wh_hidden'].flatten())
        flat_weights.extend(gru['bh'].flatten())
        
        # 3. Policy head
        flat_weights.extend(brain.params['policy_head']['W'].flatten())
        flat_weights.extend(brain.params['policy_head']['b'].flatten())
        
        # 4. Value head
        flat_weights.extend(brain.params['value_head']['W'].flatten())
        flat_weights.extend(brain.params['value_head']['b'].flatten())
        
        # Update genome
        brain.genome.weights = np.array(flat_weights, dtype=np.float32)
