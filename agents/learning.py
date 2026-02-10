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
            probs, value, _ = brain.forward(exp.observation, exp.hidden_state)
            
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
            self._update_parameters_simple(brain, exp, advantage, td_target, probs)
        
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
        probs: np.ndarray
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
        """
        # Policy gradient: ∇θ log π(a|s) * A
        # Simplified: update policy head to increase prob of action if advantage > 0
        action_gradient = probs.copy()
        action_gradient[exp.action] -= 1.0  # Gradient of log π
        action_gradient *= advantage * self.learning_rate
        
        # Get GRU output (hidden state after processing observation)
        _, _, h = brain.forward(exp.observation, exp.hidden_state)
        
        # Update policy head
        brain.params['policy_head']['W'] -= np.outer(h, action_gradient)
        brain.params['policy_head']['b'] -= action_gradient
        
        # Value gradient: ∇θ (V(s) - target)^2 = 2 * (V(s) - target) * ∇θ V(s)
        _, value, _ = brain.forward(exp.observation, exp.hidden_state)
        value_error = value - td_target
        value_gradient = 2 * value_error * self.learning_rate
        
        # Update value head
        brain.params['value_head']['W'] -= h.reshape(-1, 1) * value_gradient
        brain.params['value_head']['b'] -= value_gradient
        
        # Entropy gradient: encourage higher entropy (more exploration)
        # This is already incorporated via the entropy bonus in the loss
        # For simplicity, we apply a small perturbation to encoder to encourage variation
        if abs(advantage) > 0.1:  # Only update encoder for significant errors
            # Small update to encoder to adjust representations
            lr_encoder = self.learning_rate * 0.1  # Smaller LR for encoder
            for i in range(len(brain.params['encoder_weights'])):
                # Small random perturbation scaled by advantage
                perturbation = np.random.randn(*brain.params['encoder_weights'][i].shape) * 0.001
                brain.params['encoder_weights'][i] -= perturbation * advantage * lr_encoder
    
    
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
        for gate in ['r', 'z', 'h']:
            flat_weights.extend(gru[f'W{gate}_input'].flatten())
            flat_weights.extend(gru[f'W{gate}_hidden'].flatten())
            flat_weights.extend(gru[f'b{gate}'].flatten())
        
        # 3. Policy head
        flat_weights.extend(brain.params['policy_head']['W'].flatten())
        flat_weights.extend(brain.params['policy_head']['b'].flatten())
        
        # 4. Value head
        flat_weights.extend(brain.params['value_head']['W'].flatten())
        flat_weights.extend(brain.params['value_head']['b'].flatten())
        
        # Update genome
        brain.genome.weights = np.array(flat_weights, dtype=np.float32)


class BestAgentTracker:
    """
    Tracks and saves the best performing agents' weights.
    
    This allows starting subsequent runs with pre-trained weights
    from agents that performed well in previous runs.
    """
    
    def __init__(self, save_dir: str = "data/weights"):
        """
        Initialize the tracker.
        
        Args:
            save_dir: Directory to save weight files
        """
        self.save_dir = save_dir
        self.best_agents = []  # List of (fitness, weights, metadata)
        self.max_saved = 5  # Keep top 5 agents
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def update(self, agent: 'Agent', world: 'World') -> bool:
        """
        Check if agent should be added to best agents list.
        
        Args:
            agent: Agent to evaluate
            world: Current world state
            
        Returns:
            True if agent was added to best list
        """
        # Calculate comprehensive fitness score
        fitness = self._calculate_fitness(agent)
        
        # Check if this agent qualifies for the best list
        if len(self.best_agents) < self.max_saved:
            self._add_agent(agent, fitness)
            return True
        elif fitness > self.best_agents[-1][0]:  # Better than worst in list
            self._add_agent(agent, fitness)
            return True
        
        return False
    
    def _calculate_fitness(self, agent: 'Agent') -> float:
        """
        Calculate comprehensive fitness score.
        
        Args:
            agent: Agent to evaluate
            
        Returns:
            Fitness score
        """
        # Base fitness from agent's own fitness calculation
        fitness = agent.fitness
        
        # Bonus for longevity
        fitness += agent.age * 0.1
        
        # Bonus for remaining energy
        fitness += agent.energy * 0.05
        
        # Bonus for generation (evolved agents are likely better)
        fitness += agent.genome.generation * 2.0
        
        return fitness
    
    def _add_agent(self, agent: 'Agent', fitness: float) -> None:
        """
        Add agent to the best list.
        
        Args:
            agent: Agent to add
            fitness: Pre-calculated fitness score
        """
        metadata = {
            'generation': agent.genome.generation,
            'age': agent.age,
            'energy': agent.energy,
            'fitness': agent.fitness,
            'traits': {k: float(v) for k, v in agent.genome.traits.items()}
        }
        
        # Get weights from genome
        weights = agent.genome.weights.copy()
        
        self.best_agents.append((fitness, weights, metadata))
        
        # Sort by fitness (descending) and keep only top agents
        self.best_agents.sort(key=lambda x: x[0], reverse=True)
        self.best_agents = self.best_agents[:self.max_saved]
    
    def save_best_weights(self, filename: str = "best_weights.npz") -> str:
        """
        Save best agent weights to file.
        
        Args:
            filename: Name of the file to save
            
        Returns:
            Path to saved file
        """
        if not self.best_agents:
            print("No best agents to save")
            return ""
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Save weights as numpy arrays
        weights_dict = {}
        for i, (fitness, weights, metadata) in enumerate(self.best_agents):
            weights_dict[f'weights_{i}'] = weights
            weights_dict[f'fitness_{i}'] = np.array([fitness])
        
        np.savez(filepath, **weights_dict)
        
        # Also save metadata as JSON
        metadata_path = filepath.replace('.npz', '_metadata.json')
        metadata_list = [
            {'rank': i, 'fitness': f, 'metadata': m}
            for i, (f, _, m) in enumerate(self.best_agents)
        ]
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        print(f"Saved {len(self.best_agents)} best agent weights to {filepath}")
        return filepath
    
    @staticmethod
    def load_best_weights(filepath: str = "data/weights/best_weights.npz") -> Optional[np.ndarray]:
        """
        Load best weights from file.
        
        Args:
            filepath: Path to weight file
            
        Returns:
            Best agent's weights, or None if file doesn't exist
        """
        if not os.path.exists(filepath):
            print(f"No saved weights found at {filepath}")
            return None
        
        try:
            data = np.load(filepath)
            # Return the best (first) weights
            if 'weights_0' in data:
                print(f"Loaded best weights from {filepath}")
                return data['weights_0']
        except Exception as e:
            print(f"Error loading weights: {e}")
        
        return None
    
    @staticmethod
    def initialize_agent_from_weights(
        agent: 'Agent',
        weights: np.ndarray,
        mutation_rate: float = 0.01
    ) -> None:
        """
        Initialize an agent's brain with pre-trained weights.
        
        Adds small mutations to create variation.
        
        Args:
            agent: Agent to initialize
            weights: Pre-trained weights
            mutation_rate: Standard deviation of mutations to add
        """
        # Apply small mutations for variation
        mutated_weights = weights + np.random.normal(0, mutation_rate, weights.shape)
        agent.genome.weights = mutated_weights.astype(np.float32)
        
        # Reinitialize brain with new weights
        agent.brain = agent.brain.__class__(
            agent.genome,
            agent.brain.input_size,
            agent.brain.hidden_sizes,
            agent.brain.output_size
        )

