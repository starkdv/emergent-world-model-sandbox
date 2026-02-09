"""
Agent class for the emergent world simulation.

Agents are autonomous entities that:
- Perceive their environment through observations
- Make decisions using neural network brains
- Execute primitive actions to survive and reproduce
- Evolve over generations through genetic algorithms
- Learn during their lifetime through reinforcement learning

Author: Karan Vasa
Date: November 14, 2025
"""

from typing import TYPE_CHECKING, Optional, List, ClassVar
import numpy as np

from agents.actions import Action, ActionResult, DIRECTIONS
from agents.brain import Brain
from agents.genome import Genome

if TYPE_CHECKING:
    from world.world import World
    from utils.data.agent_logger import AgentLogger, WorldModelLogger
    from agents.learning import AgentLearner


class Agent:
    """
    An autonomous agent in the simulation.
    
    Agents must:
    - Maintain energy by consuming food
    - Navigate the world using primitive actions
    - Learn behaviors through evolutionary processes
    - Survive and reproduce to pass on genes
    
    Attributes:
        id (int): Unique identifier
        x (int): X coordinate in world
        y (int): Y coordinate in world
        direction (tuple[int, int]): Facing direction vector
        energy (float): Current energy level
        max_energy (float): Maximum energy capacity
        age (int): Age in ticks
        max_age (int): Maximum lifespan
        alive (bool): Whether agent is alive
        inventory (List[int]): Object IDs held by agent
        inventory_size (int): Maximum inventory capacity
        genome (Genome): Genetic information
        brain (Brain): Neural network policy
        traits (dict[str, float]): Phenotypic traits
        fitness (float): Fitness score for evolution
        metabolism_rate (float): Energy consumed per tick
        
    Class Attributes:
        logger (AgentLogger): Optional logger for tracking actions/states
        world_model_logger: Optional logger for world model training data
    """
    
    _next_id = 0
    logger: ClassVar[Optional['AgentLogger']] = None
    world_model_logger: ClassVar[Optional['WorldModelLogger']] = None
    
    def __init__(
        self,
        x: int,
        y: int,
        genome: Genome,
        max_energy: float = 200.0,
        max_age: int = 1000,
        inventory_size: int = 5,
        metabolism_rate: float = 0.5,
    ):
        """
        Initialize a new agent.
        
        Args:
            x: Starting x coordinate
            y: Starting y coordinate
            genome: Genetic information
            max_energy: Maximum energy capacity
            max_age: Maximum lifespan in ticks
            inventory_size: Maximum inventory slots
            metabolism_rate: Base energy consumption per tick
        """
        self.id = Agent._next_id
        Agent._next_id += 1
        
        self.x = x
        self.y = y
        self.direction = (0, -1)  # Start facing north
        
        self.energy = max_energy  # Start with full energy
        self.max_energy = max_energy
        self.age = 0
        self.max_age = max_age
        self.alive = True
        
        self.inventory: List[int] = []
        self.inventory_size = inventory_size
          # Evolutionary components
        self.genome = genome
        self.brain = Brain(genome)
        self.traits = genome.traits.copy()
        self.fitness = 0.0        # Learning components
        self.learner: Optional['AgentLearner'] = None
        self.last_observation: Optional[np.ndarray] = None
        self.learning_enabled = True  # Can be disabled for pure evolution
        self.epsilon = 0.20  # Exploration rate (20% random actions) - balanced with anti-spin penalties
          # Apply trait-based modifications
        self.metabolism_rate = metabolism_rate * self.traits.get('metabolism_rate', 1.0)
        self.vision_radius = int(self.traits.get('vision_radius', 5.0))
    
    def update(self, world: 'World') -> None:
        """
        Update agent state each tick.
        
        Args:
            world: The world the agent exists in
        """
        if not self.alive:
            return
        
        # Age the agent
        self.age += 1
        
        # Consume energy for metabolism
        energy_before = self.energy
        self.energy -= self.metabolism_rate
        
        # Check for death conditions
        if self.energy <= 0 or self.age >= self.max_age:
            death_reason = "starvation" if self.energy <= 0 else "old_age"
            
            # Log terminal transition for world model
            if Agent.world_model_logger is not None and self.last_observation is not None:
                terminal_obs = self.observe(world)
                Agent.world_model_logger.log_transition(
                    tick=world.tick,
                    agent=self,
                    action=Action.WAIT,  # Placeholder
                    result=ActionResult(False, 0.0, f"Died: {death_reason}"),
                    reward=-1.0,
                    obs_before=self.last_observation,
                    obs_after=terminal_obs,
                    world=world,
                    x_before=self.x,
                    y_before=self.y,
                    energy_before=energy_before,
                    done=True,
                    death_reason=death_reason
                )
            
            self.die(world)
            
            # Store terminal experience if learning
            if self.learning_enabled and self.learner and self.last_observation is not None:
                terminal_obs = self.observe(world)
                self.learner.store_experience(
                    self.last_observation,
                    0,  # Doesn't matter
                    -1.0,  # Death penalty
                    terminal_obs,
                    True  # Episode done
                )
            return# Get observation and decide action
        observation = self.observe(world)
        
        # AUTO-EAT: If energy is low and agent has food, eat automatically
        # This ensures agents don't starve while holding food
        from world.objects import EdibleComponent
        
        # Check if agent has food in inventory
        has_food = False
        if self.inventory:
            for obj_id in self.inventory:
                obj = world.objects.get(obj_id)
                if obj is not None and obj.has_component(EdibleComponent):
                    has_food = True
                    break
        
        # Force EAT when hungry with food (50% threshold for safety)
        if self.energy < self.max_energy * 0.5 and has_food:
            action = Action.EAT
        else:
            action = self.brain.decide(observation, epsilon=self.epsilon)
        
        # Store observation before action for logging
        obs_before = observation.copy()
        x_before_action = self.x
        y_before_action = self.y
        
        # Execute action
        result = self.execute_action(action, world)
        
        # Get observation after action
        obs_after = self.observe(world)
        
        # Debug: Log food-related actions
        if "Picked up" in result.message or "Ate" in result.message:
            print(f"[FOOD ACTION] Agent {self.id} (age {self.age}): {action.name} -> {result.message}")
        
        # Calculate reward (needed for both learning and logging)
        reward = 0.0
        if self.learning_enabled and self.learner:
            reward = self.learner.reward_shaper.calculate_reward(
                action, result, energy_before, self.energy, self, world
            )
        
        # World model logging (captures full transitions for training)
        if Agent.world_model_logger is not None:
            Agent.world_model_logger.log_transition(
                tick=world.tick,
                agent=self,
                action=action,
                result=result,
                reward=reward,
                obs_before=obs_before,
                obs_after=obs_after,
                world=world,
                x_before=x_before_action,
                y_before=y_before_action,
                energy_before=energy_before,
                done=False,
                death_reason=""
            )
        
        # Learning step
        if self.learning_enabled and self.learner:
            # Store experience (if we have previous observation)
            if self.last_observation is not None:
                self.learner.store_experience(
                    self.last_observation,
                    action.value,
                    reward,
                    obs_after,
                    False  # Not done yet
                )
            
            # Learn periodically (every 3 ticks for faster adaptation)
            if self.age % 3 == 0 and len(self.learner.replay_buffer) >= self.learner.batch_size:
                loss = self.learner.learn(self.brain)
                if self.age % 100 == 0:  # Log every 100 ticks
                    print(f"  Agent {self.id} trained at age {self.age}: buffer={len(self.learner.replay_buffer)}, loss={loss:.4f}")
            
            # Store current observation for next step
            self.last_observation = obs_after.copy()

        
    def observe(self, world: 'World') -> np.ndarray:
        """
        Build observation vector from world state.
        
        Args:
            world: The world to observe
            
        Returns:
            Normalized observation vector        """
        from agents.observation import build_observation
        return build_observation(self, world)
    
    def execute_action(self, action: Action, world: 'World') -> ActionResult:
        """
        Execute a primitive action in the world.
        
        Args:
            action: The action to execute
            world: The world to act in
            
        Returns:
            Result of the action execution
        """
        if not self.alive:
            return ActionResult(False, 0.0, "Agent is dead")
        
        # Store state before action for logging
        x_before = self.x
        y_before = self.y
        energy_before = self.energy
        
        result = ActionResult(True, 0.0)
        
        if action == Action.MOVE_FORWARD:
            result = self._move_forward(world)
        elif action == Action.TURN_LEFT:
            result = self._turn_left()
        elif action == Action.TURN_RIGHT:
            result = self._turn_right()
        elif action == Action.PICK_UP:
            result = self._pick_up(world)
        elif action == Action.DROP:
            result = self._drop(world)
        elif action == Action.EAT:
            result = self._eat(world)
        elif action == Action.USE:
            result = self._use(world)
        elif action == Action.WAIT:
            result = self._wait()
          # Deduct energy cost
        self.energy -= result.energy_cost
        
        # Update fitness based on action outcomes
        if result.success:
            self.fitness += 0.1  # Small reward for successful action
        else:
            self.fitness -= 0.05  # Small penalty for failed action
        
        # Log action if logger is enabled
        if Agent.logger is not None:
            Agent.logger.log_action(
                world.tick, self, action, result,
                x_before, y_before, energy_before
            )
        
        return result
    
    def _move_forward(self, world: 'World') -> ActionResult:
        """Move one tile in current direction."""
        new_x = self.x + self.direction[0]
        new_y = self.y + self.direction[1]
        
        # Check bounds
        if not (0 <= new_x < world.width and 0 <= new_y < world.height):
            return ActionResult(False, 0.2, "Out of bounds")  # Reduced from 1.0
        
        # Check if tile is passable
        tile = world.tiles[new_y][new_x]
        if not tile.is_passable():
            return ActionResult(False, 0.2, "Tile blocked")  # Reduced from 1.0
        
        # Move agent
        self.x = new_x
        self.y = new_y
        return ActionResult(True, 0.3, "Moved forward")  # Reduced from 2.0
    
    def _turn_left(self) -> ActionResult:
        """Rotate direction 90° counter-clockwise."""
        dx, dy = self.direction
        self.direction = (dy, -dx)  # Rotate left
        return ActionResult(True, 0.1, "Turned left")  # Reduced from 0.5
    
    def _turn_right(self) -> ActionResult:
        """Rotate direction 90° clockwise."""
        dx, dy = self.direction
        self.direction = (-dy, dx)  # Rotate right
        return ActionResult(True, 0.1, "Turned right")  # Reduced from 0.5
    def _pick_up(self, world: 'World') -> ActionResult:
        """Pick up object from current tile, prioritizing food."""
        from world.objects import EdibleComponent, SeedComponent
        
        if len(self.inventory) >= self.inventory_size:
            return ActionResult(False, 0.1, "Inventory full")  # Reduced from 1.0
        
        tile = world.tiles[self.y][self.x]
        if not tile.object_ids:
            return ActionResult(False, 0.1, "No objects here")  # Reduced from 1.0
        
        # Prioritize edible items first
        obj_id_to_pick = None
        for obj_id in tile.object_ids:
            obj = world.objects.get(obj_id)
            if obj and obj.has_component(EdibleComponent):
                obj_id_to_pick = obj_id
                break
        
        # If no food, pick up seeds (useful for planting)
        if obj_id_to_pick is None:
            for obj_id in tile.object_ids:
                obj = world.objects.get(obj_id)
                if obj and obj.has_component(SeedComponent):
                    obj_id_to_pick = obj_id
                    break
        
        # If still nothing, pick first object
        if obj_id_to_pick is None:
            obj_id_to_pick = tile.object_ids[0]
        
        obj = world.objects.get(obj_id_to_pick)
        if obj is None:
            return ActionResult(False, 0.1, "Object not found")  # Reduced from 1.0
        
        # Add to inventory and remove from world tile
        self.inventory.append(obj_id_to_pick)
        tile.object_ids.remove(obj_id_to_pick)
        
        obj_type = "food" if obj.has_component(EdibleComponent) else "object"
        return ActionResult(True, 0.2, f"Picked up {obj_type} {obj_id_to_pick}")  # Reduced from 1.0
    def _drop(self, world: 'World') -> ActionResult:
        """Drop held object onto current tile or nearby if occupied."""
        if not self.inventory:
            return ActionResult(False, 0.05, "Inventory empty")  # Reduced from 0.5
        
        # Drop last object
        obj_id = self.inventory.pop()
        obj = world.objects.get(obj_id)
        
        if obj is None:
            return ActionResult(False, 0.05, "Object not found")  # Reduced from 0.5
        
        # Check stacking configuration
        tile = world.tiles[self.y][self.x]
        
        if world.allow_stacking or not tile.object_ids:
            # Stacking allowed OR tile is empty - drop here
            tile.object_ids.append(obj_id)
            obj.x = self.x
            obj.y = self.y
            return ActionResult(True, 0.1, f"Dropped object {obj_id}")  # Reduced from 1.0
        
        # Stacking disabled and tile occupied - try nearby tiles
        nearby_positions = [
            (self.x + dx, self.y + dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if (dx != 0 or dy != 0)
        ]
        
        for nx, ny in nearby_positions:
            if world.is_valid_position(nx, ny):
                nearby_tile = world.get_tile(nx, ny)
                if nearby_tile and not nearby_tile.object_ids:
                    # Found empty spot
                    nearby_tile.object_ids.append(obj_id)
                    obj.x = nx
                    obj.y = ny
                    return ActionResult(True, 0.1, f"Dropped object {obj_id} nearby")  # Reduced from 1.0
        
        # No empty spots - put back in inventory
        self.inventory.append(obj_id)
        return ActionResult(False, 0.05, "No space to drop (tile occupied)")  # Reduced from 0.5
    
    def _eat(self, world: 'World') -> ActionResult:
        """Consume edible object from inventory."""
        from world.objects import EdibleComponent
        
        if not self.inventory:
            return ActionResult(False, 0.05, "Nothing to eat")  # Reduced from 0.5
        
        # Find edible item
        for obj_id in self.inventory:
            obj = world.objects.get(obj_id)
            if obj is None:
                continue
            
            edible = obj.get_component(EdibleComponent)
            if edible is not None:
                # Consume the food
                energy_gained = edible.calories * edible.freshness
                self.energy = min(self.max_energy, self.energy + energy_gained)
                
                # Remove from inventory and world
                self.inventory.remove(obj_id)
                world.remove_object(obj_id)
                  # Fitness reward for eating
                self.fitness += energy_gained * 0.1
                
                return ActionResult(True, 0.1, f"Ate food, gained {energy_gained:.1f} energy")  # Reduced from 1.0
        
        return ActionResult(False, 0.05, "No edible items")  # Reduced from 0.5
    def _use(self, world: 'World') -> ActionResult:
        """Use/plant object (e.g., plant seed, apply fertilizer)."""
        from world.objects import SeedComponent, FertilizerComponent
        import random
        
        if not self.inventory:
            return ActionResult(False, 0.05, "Nothing to use")  # Reduced from 0.5
        
        tile = world.tiles[self.y][self.x]
        
        # Try to use first item
        obj_id = self.inventory[0]
        obj = world.objects.get(obj_id)
        
        if obj is None:
            return ActionResult(False, 0.05, "Object not found")  # Reduced from 0.5
        
        # Check if it's a seed
        seed = obj.get_component(SeedComponent)
        if seed is not None:
            # Plant the seed
            if tile.can_support_plant():
                # Check stacking configuration
                if world.allow_stacking or not tile.object_ids:
                    # Stacking allowed OR tile is empty - plant here
                    self.inventory.remove(obj_id)
                    tile.object_ids.append(obj_id)
                    obj.x = self.x
                    obj.y = self.y
                    self.fitness += 1.0
                    return ActionResult(True, 0.5, "Planted seed")  # Reduced from 2.0
                else:
                    # Stacking disabled and tile occupied - try nearby tiles
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                                (-1, -1), (-1, 1), (1, -1), (1, 1)]
                    nearby_positions = [
                        (self.x + dx, self.y + dy) 
                        for dx, dy in directions
                    ]
                    random.shuffle(nearby_positions)
                    
                    for nx, ny in nearby_positions:
                        if 0 <= nx < world.width and 0 <= ny < world.height:
                            nearby_tile = world.tiles[ny][nx]
                            if nearby_tile.can_support_plant() and not nearby_tile.object_ids:
                                # Found empty plantable spot
                                self.inventory.remove(obj_id)
                                nearby_tile.object_ids.append(obj_id)
                                obj.x = nx
                                obj.y = ny
                                self.fitness += 1.0
                                return ActionResult(True, 0.5, f"Planted seed nearby at ({nx}, {ny})")  # Reduced from 2.0
                    
                    # No empty tiles nearby - keep in inventory
                    return ActionResult(False, 0.1, "Cannot plant - tile occupied and no space nearby")  # Reduced from 0.5
            else:
                return ActionResult(False, 0.1, "Cannot plant here")  # Reduced from 1.0
        
        # Check if it's fertilizer
        fertilizer = obj.get_component(FertilizerComponent)
        if fertilizer is not None:
            # Apply fertilizer to tile
            tile.fertility = min(1.0, tile.fertility + fertilizer.fertility_boost)
            
            # Remove from inventory and world
            self.inventory.remove(obj_id)
            world.remove_object(obj_id)
            
            return ActionResult(True, 0.5, "Applied fertilizer")  # Reduced from 2.0
        
        return ActionResult(False, 0.1, "Cannot use this object")  # Reduced from 1.0
    
    def _wait(self) -> ActionResult:
        """Do nothing (conserve energy)."""
        return ActionResult(True, 0.0, "Waiting")  # FREE - was 0.1
    def die(self, world: 'World') -> None:
        """
        Handle agent death.
        
        Args:
            world: The world the agent exists in
        """
        import random
        
        self.alive = False
        
        # Death penalty to fitness (proportional to how early the death was)
        # Dying young = big penalty, dying old = small penalty
        age_ratio = self.age / self.max_age
        death_penalty = 10.0 * (1.0 - age_ratio)  # Max -10 for instant death, 0 for old age
        self.fitness -= death_penalty
        
        # Extra penalty for starvation (should have eaten!)
        if self.energy <= 0:
            self.fitness -= 5.0  # Starvation penalty
        
        # Drop all inventory items with stacking configuration check
        for obj_id in self.inventory:
            obj = world.objects.get(obj_id)
            if obj is not None:
                tile = world.tiles[self.y][self.x]
                
                # Check stacking configuration
                if world.allow_stacking or not tile.object_ids:
                    # Stacking allowed OR tile is empty - drop here
                    tile.object_ids.append(obj_id)
                    obj.x = self.x
                    obj.y = self.y
                else:
                    # Stacking disabled and tile occupied - try nearby tiles
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                                (-1, -1), (-1, 1), (1, -1), (1, 1)]
                    nearby_positions = [
                        (self.x + dx, self.y + dy) 
                        for dx, dy in directions
                    ]
                    random.shuffle(nearby_positions)
                    
                    placed = False
                    for nx, ny in nearby_positions:
                        if 0 <= nx < world.width and 0 <= ny < world.height:
                            nearby_tile = world.tiles[ny][nx]
                            if not nearby_tile.object_ids:
                                # Found empty spot
                                nearby_tile.object_ids.append(obj_id)
                                obj.x = nx
                                obj.y = ny
                                placed = True
                                break
                    
                    # If no space found, remove object from world
                    if not placed:
                        world.remove_object(obj_id)
        
        self.inventory.clear()
    def can_reproduce(self, config: dict = None) -> bool:        
        """
        Check if agent has sufficient energy to reproduce.
        
        Reproduction requires:
        - Agent must be alive
        - Energy >= threshold (from config)
        - Age >= minimum (from config)
        
        Args:
            config: Optional reproduction config dict
        
        Returns:
            True if agent can reproduce, False otherwise
        """
        if not self.alive:
            return False
        
        # Default values if no config provided
        if config is None:
            energy_threshold_pct = 0.6
            min_age = 100
        else:
            energy_threshold_pct = config.get('energy_threshold', 0.6)
            min_age = config.get('min_age', 100)
        
        energy_threshold = self.max_energy * energy_threshold_pct
        
        return self.energy >= energy_threshold and self.age >= min_age
    def reproduce(self, world: 'World', config: dict = None) -> Optional['Agent']:
        """
        Reproduce via fission, creating an offspring.
        
        Uses config for reproduction parameters or defaults if not provided.
        
        Args:
            world: The world to spawn offspring in
            config: Optional reproduction config dict
            
        Returns:
            Offspring agent if successful, None if reproduction failed
        """
        if not self.can_reproduce(config):
            return None
        
        # Get config values or use defaults
        if config is None:
            energy_split = 0.6
            mutation_std = 0.02
        else:
            energy_split = config.get('energy_split', 0.6)
            mutation_std = config.get('mutation_std', 0.02)
        
        # Import here to avoid circular dependency
        from agents.evolution import clone_agent
          # Create offspring using evolution system
        offspring = clone_agent(
            parent=self,
            mutate=True,
            mutation_std=mutation_std
        )
        
        # Give offspring FULL energy for best survival chance
        # Parent loses energy based on split ratio
        energy_cost = self.energy * energy_split
        self.energy -= energy_cost
        offspring.energy = offspring.max_energy  # Start with FULL energy!
        
        # Find nearby empty position for offspring
        spawn_positions = [
            (self.x + dx, self.y + dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if (dx != 0 or dy != 0)  # Not same position
        ]
          # Try positions in random order
        import random
        random.shuffle(spawn_positions)
        
        for x, y in spawn_positions:
            if world.is_valid_position(x, y):
                # Check if position is empty (no other agents)
                occupied = any(
                    agent.x == x and agent.y == y
                    for agent in world.agents.values()
                    if agent.alive
                )
                
                if not occupied:                    
                    offspring.x = x
                    offspring.y = y
                      # Enable learning if parent has it
                    if self.learner:
                        offspring.enable_learning(
                            learning_rate=self.learner.learning_rate,
                            discount_factor=self.learner.discount_factor,
                            batch_size=self.learner.batch_size,
                            buffer_capacity=1000  # Use default capacity
                        )
                    
                    # Inherit parent's exploration rate
                    offspring.epsilon = self.epsilon
                    
                    return offspring
        
        # No valid position found - reproduction fails
        # Refund energy
        self.energy += energy_cost
        return None
    
    def enable_learning(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        batch_size: int = 32,
        buffer_capacity: int = 1000
    ) -> None:
        """
        Enable reinforcement learning for this agent.
        
        Args:
            learning_rate: Learning rate for gradient updates
            discount_factor: Discount factor for future rewards
            batch_size: Batch size for learning updates
            buffer_capacity: Size of experience replay buffer
        """
        from agents.learning import AgentLearner
        
        self.learner = AgentLearner(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity
        )
        self.learning_enabled = True
        self.last_observation = None
    
    def disable_learning(self) -> None:
        """Disable learning for this agent (use pure evolution)."""
        self.learning_enabled = False
        self.learner = None
        self.last_observation = None
    
    def get_learned_knowledge(self) -> Optional[np.ndarray]:
        """
        Extract learned weights from brain.
        
        Returns:
            Flattened weight array or None if no learning
        """
        if not self.learning_enabled or self.learner is None:
            return None
        
        # Get current brain weights (already synced by learner)
        return self.genome.weights.copy()
    
    def inherit_knowledge(self, parent_knowledge: np.ndarray) -> None:
        """
        Initialize brain with knowledge from parent.
        
        This allows offspring to start with learned behaviors
        from their parents, combining evolution with learning.
        
        Args:
            parent_knowledge: Flattened weight array from parent
        """
        # Update genome with parent's learned weights
        self.genome.weights = parent_knowledge.copy()
        
        # Rebuild brain with new weights
        self.brain = Brain(self.genome)
    
    def __repr__(self) -> str:
        return (
            f"Agent(id={self.id}, pos=({self.x},{self.y}), "
            f"energy={self.energy:.1f}, age={self.age}, alive={self.alive})"
        )
