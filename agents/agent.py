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
from agents.brain.instincts import InstinctModule
from agents.genome import Genome
import utils.agents.agent_utils as agent_utils

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
        instinct_config: Optional ``brain.instincts`` config dict applied
            to every newly created agent (set once from YAML in main.py)
    """

    _next_id = 0
    logger: ClassVar[Optional["AgentLogger"]] = None
    world_model_logger: ClassVar[Optional["WorldModelLogger"]] = None
    instinct_config: ClassVar[Optional[dict]] = None

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
        # Randomize initial facing direction to prevent population-wide
        # turn bias from correlated starting orientations.
        _cardinal = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.direction = _cardinal[np.random.randint(4)]

        self.energy = max_energy  # Start with full energy
        self.max_energy = max_energy
        self.age = 0
        self.max_age = max_age
        self.alive = True

        self.inventory: List[int] = []
        self.inventory_size = inventory_size
        # Evolutionary components
        self.genome = genome
        # Instincts are configured once (class-level) and fade with age —
        # see agents/brain/instincts.py for the rationale.
        self.brain = Brain(
            genome, instincts=InstinctModule.from_config(Agent.instinct_config)
        )
        self.traits = genome.traits.copy()
        self.fitness = 0.0

        # GRU hidden state (memory)
        self.h = self.brain.initial_state()

        # Learning components
        self.learner: Optional["AgentLearner"] = None
        self.last_observation: Optional[np.ndarray] = None
        self.last_hidden_state: Optional[np.ndarray] = None
        self.learning_enabled = True  # Can be disabled for pure evolution
        self.temperature = 1.0  # Sampling temperature for exploration

        # Action-pattern tracking for energy shaping
        self._previous_action: Optional[Action] = None
        self._consecutive_turns = 0
        self._consecutive_waits = 0
        # Apply trait-based modifications
        self.metabolism_rate = metabolism_rate * self.traits.get("metabolism_rate", 1.0)
        self.vision_radius = int(self.traits.get("vision_radius", 5.0))

    def update(self, world: "World") -> None:
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

            # Compute terminal observation ONCE for both loggers
            terminal_obs = None
            if (
                Agent.world_model_logger is not None
                and self.last_observation is not None
            ) or (
                self.learning_enabled
                and self.learner
                and self.last_observation is not None
            ):
                terminal_obs = self.observe(world)

            # Log terminal transition for world model
            if (
                Agent.world_model_logger is not None
                and self.last_observation is not None
                and terminal_obs is not None
            ):
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
                    death_reason=death_reason,
                )

            self.die(world)

            # Store terminal experience if learning
            if (
                self.learning_enabled
                and self.learner
                and self.last_observation is not None
                and self.last_hidden_state is not None
            ):
                if terminal_obs is None:
                    terminal_obs = self.observe(world)
                terminal_h = self.brain.initial_state()  # Dead state
                self.learner.store_experience(
                    self.last_observation,
                    self.last_hidden_state,
                    0,  # Action doesn't matter
                    -1.0,  # Death penalty
                    terminal_obs,
                    terminal_h,
                    True,  # Episode done
                )
            return  # Get observation and decide action
        observation = self.observe(world)

        # Instinct strength fades with age (1.0 at birth → 0.0 at fade_age),
        # so adults act purely on learned weights. The old hardcoded
        # auto-eat override was replaced by a hunger-scaled EAT instinct
        # inside the InstinctModule — a strong prior, never a forced action.
        action_mask = self.get_action_mask(world)
        instinct_strength = self.brain.instincts.strength_at(self.age)

        # Use brain to decide action (samples from policy)
        action, self.h, _ = self.brain.decide(
            observation,
            self.h,
            action_mask=action_mask,
            temperature=self.temperature,
            instinct_strength=instinct_strength,
        )

        # Store observation before action for logging
        obs_before = observation  # no copy needed — not modified before use
        x_before_action = self.x
        y_before_action = self.y

        # Execute action
        result = self.execute_action(action, world)

        # Get observation after action
        obs_after = self.observe(world)

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
                death_reason="",
            )

        # Learning step
        if self.learning_enabled and self.learner:
            # Store experience (if we have previous observation and hidden state)
            if self.last_observation is not None and self.last_hidden_state is not None:
                # Note: self.h has already been updated to h_next by brain.decide above
                self.learner.store_experience(
                    self.last_observation,
                    self.last_hidden_state,
                    action.value,
                    reward,
                    obs_after,
                    self.h,  # Current (next) hidden state
                    False,  # Not done yet
                )

            # Learn when world scheduler grants a training slot (staggered + capped)
            has_enough_experience = (
                len(self.learner.replay_buffer) >= self.learner.batch_size
            )
            can_train_now = False

            if has_enough_experience:
                if hasattr(world, "try_acquire_learning_slot"):
                    can_train_now = world.try_acquire_learning_slot(self.id, self.age)
                else:
                    can_train_now = self.age % 3 == 0

            if can_train_now:
                self.learner.learn(self.brain)

            # Store current observation and hidden state for next step
            self.last_observation = obs_after.copy()
            self.last_hidden_state = self.h.copy()

    def get_action_mask(self, world: "World") -> np.ndarray:
        """
        Binary mask over Action enum (1 = valid, 0 = invalid).
        Delegates to agent_utils.
        """
        return agent_utils.get_action_mask(self, world)

    def observe(self, world: "World") -> np.ndarray:
        """
        Build observation vector from world state.

        Args:
            world: The world to observe

        Returns:
            Normalized observation vector"""
        from utils.agents import build_observation

        return build_observation(self, world)

    def execute_action(self, action: Action, world: "World") -> ActionResult:
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
            result = agent_utils.execute_move_forward(self, world)
        elif action == Action.TURN_LEFT:
            result = agent_utils.execute_turn_left(self)
        elif action == Action.TURN_RIGHT:
            result = agent_utils.execute_turn_right(self)
        elif action == Action.PICK_UP:
            result = agent_utils.execute_pick_up(self, world)
        elif action == Action.DROP:
            result = agent_utils.execute_drop(self, world)
        elif action == Action.EAT:
            result = agent_utils.execute_eat(self, world)
        elif action == Action.USE:
            result = agent_utils.execute_use(self, world)
        elif action == Action.WAIT:
            result = agent_utils.execute_wait(self)

        # Dynamic energy shaping (behavior economics, not reward shaping)
        effective_energy_cost = result.energy_cost

        if action in [Action.TURN_LEFT, Action.TURN_RIGHT]:
            self._consecutive_turns += 1
            self._consecutive_waits = 0
            # Mild escalating turn cost — only punishes extended spin loops.
            # IMPORTANT: don't penalize the first couple of turns; otherwise
            # agents learn "always MOVE_FORWARD until masked".
            extra_turn_penalty = max(0, self._consecutive_turns - 2)
            effective_energy_cost += min(0.04 * extra_turn_penalty, 0.20)
        elif action == Action.WAIT:
            self._consecutive_waits += 1
            self._consecutive_turns = 0
            # Very gentle escalating wait cost — WAIT should remain affordable
            extra_wait_penalty = max(0, self._consecutive_waits - 2)
            effective_energy_cost += min(0.02 * extra_wait_penalty, 0.10)
        elif action == Action.MOVE_FORWARD:
            # Reward turn->move transition with a tiny cost discount
            if result.success and self._previous_action in [
                Action.TURN_LEFT,
                Action.TURN_RIGHT,
            ]:
                effective_energy_cost = max(0.10, effective_energy_cost - 0.02)
            self._consecutive_turns = 0
            self._consecutive_waits = 0
        else:
            self._consecutive_turns = 0
            self._consecutive_waits = 0

        # Use effective cost for state update and downstream logging
        result = result._replace(energy_cost=round(effective_energy_cost, 3))

        # Deduct energy cost
        self.energy -= result.energy_cost

        # Track action for next-step energy shaping
        self._previous_action = action

        # Update fitness based on action outcomes.
        # Successful turns should not be penalized; otherwise the policy
        # is structurally biased toward MOVE_FORWARD.
        if result.success:
            if action == Action.WAIT:
                # WAIT is neutral from a fitness perspective.
                self.fitness += 0.0
            else:
                self.fitness += 0.1  # Small reward for successful action
        else:
            self.fitness -= 0.05  # Small penalty for failed action

        # Log action if logger is enabled
        if Agent.logger is not None:
            Agent.logger.log_action(
                world.tick, self, action, result, x_before, y_before, energy_before
            )

        return result

    def die(self, world: "World") -> None:
        """
        Handle agent death.

        Args:
            world: The world the agent exists in
        """
        import random

        self.alive = False

        # Reset hidden state (agent's memory is lost on death)
        self.h = self.brain.initial_state()

        # Death penalty to fitness (proportional to how early the death was)
        # Dying young = big penalty, dying old = small penalty
        age_ratio = self.age / self.max_age
        death_penalty = 10.0 * (
            1.0 - age_ratio
        )  # Max -10 for instant death, 0 for old age
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
                    tile.object_ids.add(obj_id)
                    obj.x = self.x
                    obj.y = self.y
                else:
                    # Stacking disabled and tile occupied - try nearby tiles
                    directions = [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                    ]
                    nearby_positions = [
                        (self.x + dx, self.y + dy) for dx, dy in directions
                    ]
                    random.shuffle(nearby_positions)

                    placed = False
                    for nx, ny in nearby_positions:
                        if 0 <= nx < world.width and 0 <= ny < world.height:
                            nearby_tile = world.tiles[ny][nx]
                            if not nearby_tile.object_ids:
                                # Found empty spot
                                nearby_tile.object_ids.add(obj_id)
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
            energy_threshold_pct = config.get("energy_threshold", 0.6)
            min_age = config.get("min_age", 100)

        energy_threshold = self.max_energy * energy_threshold_pct

        return self.energy >= energy_threshold and self.age >= min_age

    def reproduce(self, world: "World", config: dict = None) -> Optional["Agent"]:
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
            energy_split = config.get("energy_split", 0.6)
            mutation_std = config.get("mutation_std", 0.02)

        # Import here to avoid circular dependency
        from agents.evolution import clone_agent

        # Create offspring using evolution system
        offspring = clone_agent(parent=self, mutate=True, mutation_std=mutation_std)

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

                    # Offspring starts with fresh memory (no inherited hidden state)
                    offspring.h = offspring.brain.initial_state()

                    # Enable learning if parent has it
                    if self.learner:
                        offspring.enable_learning(
                            learning_rate=self.learner.learning_rate,
                            discount_factor=self.learner.discount_factor,
                            batch_size=self.learner.batch_size,
                            buffer_capacity=1000,  # Use default capacity
                            compute_backend=self.learner.compute_backend,
                            compute_device=self.learner.compute_device,
                        )

                    # Inherit parent's temperature
                    offspring.temperature = self.temperature

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
        buffer_capacity: int = 1000,
        compute_backend: str = "auto",
        compute_device: str = "auto",
    ) -> None:
        """
        Enable reinforcement learning for this agent.

        Args:
            learning_rate: Learning rate for gradient updates
            discount_factor: Discount factor for future rewards
            batch_size: Batch size for learning updates
            buffer_capacity: Size of experience replay buffer
            compute_backend: 'auto', 'numpy', or 'torch'
            compute_device: 'auto', 'cpu', 'cuda', or 'mps'
        """
        from agents.learning import AgentLearner

        self.learner = AgentLearner(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            compute_backend=compute_backend,
            compute_device=compute_device,
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

        # Rebuild brain with new weights (keep the configured instincts)
        self.brain = Brain(self.genome, instincts=self.brain.instincts)

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.id}, pos=({self.x},{self.y}), "
            f"energy={self.energy:.1f}, age={self.age}, alive={self.alive})"
        )
