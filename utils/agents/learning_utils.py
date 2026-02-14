"""
Utility classes for reinforcement learning.

Contains helper classes that support the core learning system:
- Experience: Single experience tuple for RL
- ReplayBuffer: Experience replay buffer
- RewardShaper: Reward shaping for survival behaviors  
- BestAgentTracker: Tracks and saves best performing agents

Author: Karan Vasa
Date: November 15, 2025
"""

import numpy as np
from typing import List, Optional, TYPE_CHECKING
from collections import deque
import os
import json

if TYPE_CHECKING:
    from agents.agent import Agent
    from world.world import World
    from agents.actions import Action, ActionResult


class Experience:
    """
    A single experience tuple for reinforcement learning with GRU hidden states.
    
    Stores the observation, hidden state, action, reward, and next observation/hidden state
    for learning from past decisions in a recurrent policy.
    """
    
    def __init__(
        self,
        observation: np.ndarray,
        hidden_state: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_hidden_state: np.ndarray,
        done: bool
    ):
        self.observation = observation
        self.hidden_state = hidden_state
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.next_hidden_state = next_hidden_state
        self.done = done


class ReplayBuffer:
    """
    Experience replay buffer for stable learning.
    
    Stores recent experiences and allows sampling for learning updates.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def clear(self) -> None:
        """Clear all experiences."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)


class RewardShaper:
    """
    Shapes rewards to encourage survival behaviors.
    
    Provides intrinsic rewards for:
    - Finding and eating food (with energy-level bonuses)
    - Active exploration and movement
    - Moving toward food (dense reward)
    - Penalizing excessive waiting
    - Planting seeds (future benefit)
    - Heavy penalty for EAT spamming when no food present
    """
    
    def __init__(self):
        """Initialize reward shaper with tracking for distance rewards."""
        self.last_food_distance = None
        self.last_actions = []  # Track recent actions to detect spinning
        self.last_position = None  # Track position to detect movement
        self.consecutive_waits = 0  # Track consecutive wait actions
        self.positions_visited = set()  # Track visited positions for exploration
        self.steps_without_movement = 0  # Track steps without position change
        self.consecutive_same_action = 0  # Track repeated same actions
        self.last_action = None  # Track last action for repetition detection
        self.consecutive_failed_eats = 0  # Track failed EAT attempts for spam detection
    
    def calculate_reward(
        self,
        action: 'Action',
        action_result: 'ActionResult',
        energy_before: float,
        energy_after: float,
        agent: 'Agent',
        world: 'World'
    ) -> float:
        """
        Calculate shaped reward for an action.
        
        Uses DENSE REWARDS with EXPLORATION INCENTIVES:
        - Base survival reward per step
        - STRONG exploration bonus for movement
        - Energy-aware eating rewards (critical energy = big reward)
        - Distance-based reward (moving toward food)
        - Heavy penalty for excessive waiting
        
        Args:
            action: The action taken
            action_result: Result of the action
            energy_before: Energy before action
            energy_after: Energy after action
            agent: The agent
            world: The world
            
        Returns:
            Shaped reward value
        """
        from agents.actions import Action
        
        reward = 0.0
        
        # ===== DENSE REWARD #1: Base survival =====
        reward += 0.02  # Keep survival signal, but avoid rewarding idling too strongly
        
        # ===== ANTI-SPAM: Penalize repeating the same action =====
        if action == self.last_action:
            self.consecutive_same_action += 1
            if self.consecutive_same_action > 3:
                # Progressive penalty for spamming same action
                reward -= 0.15 * min(self.consecutive_same_action - 3, 10)
        else:
            self.consecutive_same_action = 0
        self.last_action = action
          # ===== EXPLORATION BONUS (REDUCED for balance) =====
        current_position = (agent.x, agent.y)
        
        # Track if agent moved to a new position
        if self.last_position is not None:
            if current_position != self.last_position:
                # Agent moved! Reward active behavior
                self.steps_without_movement = 0
                reward += 0.18
                
                # Extra bonus for visiting new tiles
                if current_position not in self.positions_visited:
                    reward += 0.25
                    self.positions_visited.add(current_position)
                    
                    # Keep visited set from growing too large
                    if len(self.positions_visited) > 100:
                        self.positions_visited = set(list(self.positions_visited)[-50:])
            else:
                # Agent didn't move
                self.steps_without_movement += 1
                
                # Mild penalty for staying in place (only after several steps)
                if self.steps_without_movement > 5:
                    reward -= 0.05 * min(self.steps_without_movement - 5, 10)
        
        self.last_position = current_position
        
        nearest_food_dist = self._find_nearest_food_distance(agent, world)

        # ===== WAIT: Discourage passive policies =====
        if action == Action.WAIT:
            self.consecutive_waits += 1
            
            # Always apply a small immediate wait cost, then escalate for repeats
            reward -= 0.06

            if self.consecutive_waits > 2:
                reward -= 0.12 * min(self.consecutive_waits - 2, 10)
            
            # Extra penalty if energy is critically low and waiting
            if agent.energy < agent.max_energy * 0.3:
                reward -= 0.5

            # If food is visible nearby, waiting should be discouraged (should go pick it up / move onto it)
            if nearest_food_dist is not None and nearest_food_dist <= 3.0:
                reward -= 0.35
        else:
            self.consecutive_waits = 0
        
        # ===== DENSE REWARD #2: Moving toward food =====
        nearest_food_dist = self._find_nearest_food_distance(agent, world)
        
        if nearest_food_dist is not None:
            # Reward for getting closer to food
            if self.last_food_distance is not None:
                distance_change = self.last_food_distance - nearest_food_dist
                if distance_change > 0:
                    # Got closer - reward scales with urgency
                    urgency_multiplier = 1.0
                    if agent.energy < agent.max_energy * 0.3:
                        urgency_multiplier = 3.0  # Critical energy = 3x reward
                    elif agent.energy < agent.max_energy * 0.5:
                        urgency_multiplier = 2.0  # Low energy = 2x reward
                    
                    reward += 2.0 * distance_change * urgency_multiplier
                elif distance_change < 0:
                    # Got farther - penalty
                    reward -= 0.3 * abs(distance_change)
            
            # Proximity bonus (being close to food)
            if nearest_food_dist < 3.0:
                reward += 1.5 / (nearest_food_dist + 0.5)
            
            # On food tile bonus
            if nearest_food_dist < 1.0:
                reward += 3.0  # Strong signal: you're on food!
            
            self.last_food_distance = nearest_food_dist
          # ===== ENERGY-AWARE EATING REWARDS =====
        energy_gain = energy_after - energy_before
        
        if action == Action.EAT:
            if action_result.success:
                # Reset failed eat counter on success
                self.consecutive_failed_eats = 0
                
                # Base eating reward
                reward += 5.0
                
                # ENERGY-LEVEL MULTIPLIER for eating
                energy_ratio = energy_before / agent.max_energy
                
                if energy_ratio < 0.2:  # Critical (red) - HUGE reward
                    reward += 20.0
                elif energy_ratio < 0.4:  # Low (orange/red)
                    reward += 15.0
                elif energy_ratio < 0.6:  # Medium (yellow)
                    reward += 10.0
                elif energy_ratio < 0.8:  # Good (green)
                    reward += 5.0
                else:  # Full - still reward but less
                    reward += 2.0
                
                # Additional reward proportional to energy gained
                reward += energy_gain * 0.2
            else:
                # FAILED EAT - Track and heavily penalize spam
                self.consecutive_failed_eats += 1
                
                # Escalating penalty for EAT spam
                # 1st fail: -2.5, 2nd: -3.5, 3rd: -4.5, etc.
                base_penalty = -2.5
                spam_multiplier = min(self.consecutive_failed_eats, 10)
                reward += base_penalty - (0.5 * (spam_multiplier - 1))
                
                # Extra harsh penalty if agent is just standing and spamming EAT
                if self.steps_without_movement > 2:
                    reward -= 1.0  # Should move to find food, not spam EAT
        else:
            # Reset failed eat counter when doing other actions
            self.consecutive_failed_eats = 0
        
        # Energy loss penalty (metabolism)
        if energy_gain < 0 and action != Action.EAT:
            reward -= abs(energy_gain) * 0.01  # Slight penalty for metabolism
          # ===== SUCCESS BONUSES =====
        if action_result.success:
            if "Picked up" in action_result.message:
                reward += 1.0  # Found and picked up food/item!
            elif "Planted" in action_result.message:
                reward += 2.0  # Good for ecosystem
            elif action == Action.MOVE_FORWARD:
                reward += 0.25  # Small bonus for successful movement        
        else:
            # Penalty for failed actions (EAT handled separately above)
            if action == Action.PICK_UP:
                reward -= 0.8  # Increased penalty for failed pickup (was 0.5)
            elif action == Action.MOVE_FORWARD:
                reward -= 0.05  # Tiny penalty for bumping into walls
            elif action == Action.DROP:
                reward -= 0.3
            else:
                reward -= 0.05  # Small penalty for other failures        # ===== Inventory and hunger interaction =====
        from world.objects import EdibleComponent
        has_food_in_inventory = False
        food_count = 0
        
        if agent.inventory:
            for obj_id in agent.inventory:
                obj = world.objects.get(obj_id)
                if obj is not None and obj.has_component(EdibleComponent):
                    has_food_in_inventory = True
                    food_count += 1
        
        # Small bonus for having food (insurance)
        reward += food_count * 0.1
        
        # ===== CRITICAL: Encourage eating from inventory when hungry =====
        energy_ratio = agent.energy / agent.max_energy
        
        if has_food_in_inventory and energy_ratio < 0.5:
            # Agent has food and is hungry!
            if action == Action.EAT:
                # HUGE bonus for eating when you have food and need it
                if action_result.success:
                    if energy_ratio < 0.2:
                        reward += 15.0  # Critical: massive reward
                    elif energy_ratio < 0.35:
                        reward += 10.0  # Low: big reward
                    else:
                        reward += 5.0   # Medium: good reward
            else:
                # Penalty for NOT eating when you have food and need energy
                if energy_ratio < 0.2:
                    reward -= 2.0  # Critical: should be eating!
                elif energy_ratio < 0.35:
                    reward -= 1.0  # Low: really should eat
                else:
                    reward -= 0.3  # Medium: consider eating
        
        # ===== Anti-spinning penalty =====
        self.last_actions.append(action)
        if len(self.last_actions) > 8:
            self.last_actions.pop(0)
        
        if len(self.last_actions) >= 8:
            recent = self.last_actions[-8:]
            turn_count = sum(1 for a in recent if a in [Action.TURN_LEFT, Action.TURN_RIGHT])
            move_count = sum(1 for a in recent if a == Action.MOVE_FORWARD)

            # If turning dominates recent actions, penalize harder.
            if turn_count >= 4:
                reward -= 2.0

            # If mostly turning and almost no movement, heavy penalty
            if turn_count >= 5 and move_count <= 1:
                reward -= 4.0
                # Extra penalty if it's literally not changing position
                if self.steps_without_movement > 2:
                    reward -= 3.0

            # Also penalize excessive turning in place regardless
            if self.steps_without_movement > 2 and turn_count >= 4:
                reward -= 3.0
        
        # ===== Critical energy state penalties/urgency =====
        if agent.energy < agent.max_energy * 0.2:
            # Critical energy - bonus for moving (finding food)
            if action == Action.MOVE_FORWARD and action_result.success:
                reward += 0.05 * min(3, self.consecutive_same_action)
            # Penalty for waiting when critical
            if action == Action.WAIT:
                reward -= 1.0
        
        # Death penalty
        if not agent.alive:
            reward -= 10.0

        if action in [Action.TURN_LEFT, Action.TURN_RIGHT]:
            reward -= 0.12
            if self.steps_without_movement >= 2:
                reward -= 0.25

        
        return reward
    
    def _find_nearest_food_distance(self, agent: 'Agent', world: 'World') -> Optional[float]:
        """
        Find distance to nearest food.
        
        Args:
            agent: The agent
            world: The world
            
        Returns:
            Distance to nearest food, or None if no food exists        """
        from world.objects import EdibleComponent
        import math
        
        min_dist = None
        for obj in world.objects.values():  # Iterate over values, not keys
            edible = obj.get_component(EdibleComponent)
            if edible is not None:
                dist = math.sqrt((obj.x - agent.x)**2 + (obj.y - agent.y)**2)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
        
        return min_dist
    
    def reset(self):
        """Reset tracking for new episode."""
        self.last_food_distance = None
        self.last_actions = []
        self.last_position = None
        self.consecutive_waits = 0
        self.positions_visited.clear()
        self.steps_without_movement = 0
        self.consecutive_same_action = 0
        self.last_action = None
        self.consecutive_failed_eats = 0


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
            agent.brain.encoder_layers,
            agent.brain.gru_hidden_size,
            agent.brain.output_size
        )
        # Reset hidden state
        agent.h = agent.brain.initial_state()
