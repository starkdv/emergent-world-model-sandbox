"""
Main World class for the simulation environment.

This module implements the world grid, object management, and the main
simulation update loop.

Author: Karan Vasa
"""

import random
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from world.tiles import Tile, TerrainType
from world.objects import WorldObject
from world.systems import WorldSystemManager

if TYPE_CHECKING:
    from agents.agent import Agent


class World:
    """
    Main world simulation class.
    
    Manages the 2D grid world, tiles, objects, agents, and simulation state.
    
    Attributes:
        width: Width of the world grid
        height: Height of the world grid
        tiles: 2D list of Tile objects
        objects: Dictionary mapping object ID to WorldObject
        agents: Dictionary mapping agent ID to Agent (populated by simulation)
        tick: Current simulation tick counter
        seed: Random seed for reproducibility
      Author: Karan Vasa
    """
    def __init__(
        self,
        width: int,
        height: int,
        seed: Optional[int] = None,
        soil_ratio: float = 0.7,
        rock_ratio: float = 0.2,
        water_ratio: float = 0.1,
        fertility_range: Tuple[float, float] = (0.3, 1.0),
        moisture_range: Tuple[float, float] = (0.2, 0.8),
        # System configuration parameters
        plant_mature_age: int = 100,
        plant_max_age: int = 500,
        decay_rate: float = 0.01,
        seed_drop_chance: float = 0.7,
        germination_success_rate: float = 0.75,
        fertility_consumption: float = 0.001,
        moisture_consumption: float = 0.0005,
        fertility_recovery_rate: float = 0.0005,
        moisture_evaporation_rate: float = 0.0002,
        moisture_recovery_rate: float = 0.0008,
        fertility_return_on_death: float = 0.15,
        berry_calories: float = 20.0,
        safety_spawn_rate: float = 0.01,
        min_resources: int = 10,
        seed_max_age: int = 200,
        allow_stacking: bool = False,  # NEW: Controls object stacking
        learning_train_interval_ticks: int = 3,
        learning_max_updates_per_tick: int = 16,
        learning_enable_stagger: bool = True,
        learning_adaptive_budget: bool = True,
        learning_min_updates_per_tick: int = 2,
        learning_max_budget_updates_per_tick: int = 24,
        learning_budget_adjust_step: int = 1,
        learning_budget_high_frame_factor: float = 1.10,
        learning_budget_low_frame_factor: float = 0.80,
    ):
        """
        Initialize a new world with generated terrain.
        
        Args:
            width: Width of the world grid
            height: Height of the world grid
            seed: Random seed for reproducibility (None for random)
            soil_ratio: Proportion of tiles that are soil
            rock_ratio: Proportion of tiles that are rock
            water_ratio: Proportion of tiles that are water
            fertility_range: Min and max fertility for soil tiles
            moisture_range: Min and max moisture for tiles
            plant_mature_age: Age at which plants mature
            plant_max_age: Maximum plant lifespan
            decay_rate: Freshness decay rate per tick
            seed_drop_chance: Chance for decomposed berries to drop seeds
            germination_success_rate: Probability of seed germination success
            fertility_consumption: Fertility consumed by plants per tick
            moisture_consumption: Moisture consumed by plants per tick
            fertility_recovery_rate: Fertility recovery for empty soil per tick
            moisture_evaporation_rate: Natural moisture loss per tick
            moisture_recovery_rate: Moisture recovery from rain/groundwater per tick
            fertility_return_on_death: Fertility returned on decomposition
            berry_calories: Calorie value for berries
            safety_spawn_rate: Safety net spawn probability
            min_resources: Minimum resources before safety spawning
            seed_max_age: Maximum age before seed rots (in ticks)
            learning_train_interval_ticks: Minimum ticks between training attempts per agent
            learning_max_updates_per_tick: Max agent training updates allowed per world tick
            learning_enable_stagger: Whether to stagger training by agent id to avoid synchronization spikes
            learning_adaptive_budget: Enable adaptive update-budget tuning based on frame time
            learning_min_updates_per_tick: Lower bound for adaptive training budget
            learning_max_budget_updates_per_tick: Upper bound for adaptive training budget
            learning_budget_adjust_step: Step size when adjusting adaptive training budget
            learning_budget_high_frame_factor: Decrease budget when frame_ms > target_ms * this factor
            learning_budget_low_frame_factor: Increase budget when frame_ms < target_ms * this factor
            
        Raises:
            ValueError: If dimensions are invalid or ratios don't sum to 1.0
        """        
        if width <= 0 or height <= 0:
            raise ValueError(f"World dimensions must be positive, got {width}x{height}")
        
        ratio_sum = soil_ratio + rock_ratio + water_ratio
        if not abs(ratio_sum - 1.0) < 0.01:
            raise ValueError(f"Terrain ratios must sum to 1.0, got {ratio_sum}")
        
        self.width = width
        self.height = height
        self.tick = 0
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.allow_stacking = allow_stacking  # NEW: Store stacking configuration
        
        # Set random seed for reproducibility
        random.seed(self.seed)
          # Initialize data structures
        self.tiles: List[List[Tile]] = []
        self.objects: Dict[int, WorldObject] = {}
        self.agents: Dict[int, 'Agent'] = {}  # Forward reference for Agent
        
        # Reproduction config (can be set after init)
        self.reproduction_config: Optional[dict] = None
        
        # Calamity config (can be set after init)
        self.calamity_config: Optional[dict] = None
        self.last_calamity_tick: int = 0

        # Learning scheduler config
        self.learning_train_interval_ticks = max(1, int(learning_train_interval_ticks))
        self.learning_max_updates_per_tick = max(0, int(learning_max_updates_per_tick))
        self.learning_enable_stagger = bool(learning_enable_stagger)
        self._learning_updates_this_tick = 0

        # Adaptive learning budget config
        self.learning_adaptive_budget = bool(learning_adaptive_budget)
        self.learning_min_updates_per_tick = max(0, int(learning_min_updates_per_tick))
        self.learning_max_budget_updates_per_tick = max(
            self.learning_min_updates_per_tick,
            int(learning_max_budget_updates_per_tick)
        )
        self.learning_budget_adjust_step = max(1, int(learning_budget_adjust_step))
        self.learning_budget_high_frame_factor = max(1.0, float(learning_budget_high_frame_factor))
        self.learning_budget_low_frame_factor = max(0.0, min(1.0, float(learning_budget_low_frame_factor)))

        # Clamp initial budget into adaptive bounds
        self.learning_max_updates_per_tick = min(
            self.learning_max_budget_updates_per_tick,
            max(self.learning_min_updates_per_tick, self.learning_max_updates_per_tick)
        )
        
        # Initialize world systems with configuration
        self.systems = WorldSystemManager(
            plant_mature_age=plant_mature_age,
            plant_max_age=plant_max_age,
            decay_rate=decay_rate,
            seed_drop_chance=seed_drop_chance,
            germination_success_rate=germination_success_rate,
            fertility_consumption=fertility_consumption,
            moisture_consumption=moisture_consumption,
            fertility_recovery_rate=fertility_recovery_rate,
            moisture_evaporation_rate=moisture_evaporation_rate,
            moisture_recovery_rate=moisture_recovery_rate,
            fertility_return_on_death=fertility_return_on_death,
            berry_calories=berry_calories,
            safety_spawn_rate=safety_spawn_rate,
            min_resources=min_resources,
            seed_max_age=seed_max_age
        )
        
        # Generate terrain
        self._generate_terrain(
            soil_ratio, rock_ratio, water_ratio,
            fertility_range, moisture_range
        )
    
    def _generate_terrain(
        self,
        soil_ratio: float,
        rock_ratio: float,
        water_ratio: float,
        fertility_range: Tuple[float, float],
        moisture_range: Tuple[float, float]
    ) -> None:
        """
        Generate the world terrain with specified ratios.
        
        Args:
            soil_ratio: Proportion of soil tiles
            rock_ratio: Proportion of rock tiles
            water_ratio: Proportion of water tiles
            fertility_range: Min and max fertility values
            moisture_range: Min and max moisture values
        """
        # Create terrain type distribution
        terrain_types = (
            [TerrainType.SOIL] * int(self.width * self.height * soil_ratio) +
            [TerrainType.ROCK] * int(self.width * self.height * rock_ratio) +
            [TerrainType.WATER] * int(self.width * self.height * water_ratio)
        )
        
        # Fill remaining with soil
        while len(terrain_types) < self.width * self.height:
            terrain_types.append(TerrainType.SOIL)
        
        # Shuffle for random distribution
        random.shuffle(terrain_types)
        
        # Create tiles
        idx = 0
        for y in range(self.height):
            row = []
            for x in range(self.width):
                terrain_type = terrain_types[idx]
                idx += 1
                
                # Generate random fertility and moisture
                fertility = random.uniform(fertility_range[0], fertility_range[1])
                moisture = random.uniform(moisture_range[0], moisture_range[1])
                
                # Rock tiles have no fertility
                if terrain_type == TerrainType.ROCK:
                    fertility = 0.0
                
                # Water tiles have high moisture
                if terrain_type == TerrainType.WATER:
                    moisture = 1.0
                
                tile = Tile(x, y, terrain_type, fertility, moisture)
                row.append(tile)
            
            self.tiles.append(row)
    
    def get_tile(self, x: int, y: int) -> Optional[Tile]:
        """
        Get tile at the given coordinates.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            Tile at position or None if out of bounds
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[y][x]
        return None
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """
        Check if coordinates are within world bounds.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            True if position is valid, False otherwise
        """
        return 0 <= x < self.width and 0 <= y < self.height      
    def add_object(self, obj: WorldObject) -> bool:
        """
        Add a world object to the simulation.
        
        Enforces one-object-per-tile rule if allow_stacking is False.
        
        Args:
            obj: WorldObject to add
            
        Returns:
            True if added successfully, False if position invalid or tile occupied
        """
        if not self.is_valid_position(obj.x, obj.y):
            return False
        
        # Check stacking configuration
        if not self.allow_stacking:
            # Enforce one-per-tile: Check if tile already has an object
            tile = self.get_tile(obj.x, obj.y)
            if tile and tile.object_ids:
                # Tile is occupied - try to find nearby empty tile
                nearby_positions = [
                    (obj.x + dx, obj.y + dy)
                    for dx in [-1, 0, 1]
                    for dy in [-1, 0, 1]
                    if (dx != 0 or dy != 0)  # Not same position
                ]
                
                # Shuffle for randomness
                random.shuffle(nearby_positions)
                
                # Try to place in nearby empty tile
                for nx, ny in nearby_positions:
                    if self.is_valid_position(nx, ny):
                        nearby_tile = self.get_tile(nx, ny)
                        if nearby_tile and not nearby_tile.object_ids:
                            # Found empty spot - move object there
                            obj.x = nx
                            obj.y = ny
                            self.objects[obj.id] = obj
                            nearby_tile.add_object(obj.id)
                            return True
                
                # No empty nearby tiles - don't add object
                return False
        
        # Stacking allowed OR tile is empty - add normally
        tile = self.get_tile(obj.x, obj.y)
        self.objects[obj.id] = obj
        if tile:
            tile.add_object(obj.id)
        return True
    
    def remove_object(self, object_id: int) -> bool:
        """
        Remove a world object from the simulation.
        
        Args:
            object_id: ID of object to remove
            
        Returns:
            True if removed successfully, False if not found
        """
        if object_id not in self.objects:
            return False
        
        obj = self.objects[object_id]
        tile = self.get_tile(obj.x, obj.y)
        if tile:
            tile.remove_object(object_id)
        
        del self.objects[object_id]
        return True
    
    def move_object(self, object_id: int, new_x: int, new_y: int) -> bool:
        """
        Move an object to a new position.
        
        Args:
            object_id: ID of object to move
            new_x: New X-coordinate
            new_y: New Y-coordinate
            
        Returns:
            True if moved successfully, False otherwise
        """
        if object_id not in self.objects:
            return False
        
        if not self.is_valid_position(new_x, new_y):
            return False
        
        obj = self.objects[object_id]
        
        # Remove from old tile
        old_tile = self.get_tile(obj.x, obj.y)
        if old_tile:
            old_tile.remove_object(object_id)
        
        # Update position
        obj.x = new_x
        obj.y = new_y
        
        # Add to new tile
        new_tile = self.get_tile(new_x, new_y)
        if new_tile:
            new_tile.add_object(object_id)
        
        return True
    
    def get_objects_at(self, x: int, y: int) -> List[WorldObject]:
        """
        Get all objects at a specific position.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            List of WorldObjects at that position
        """
        tile = self.get_tile(x, y)
        if not tile:
            return []
        
        return [self.objects[obj_id] for obj_id in tile.object_ids if obj_id in self.objects]
    
    def get_neighbors(self, x: int, y: int, radius: int = 1) -> List[Tile]:
        """
        Get all tiles within a radius of a position.
        
        Args:
            x: Center X-coordinate            y: Center Y-coordinate
            radius: Radius to search (default 1 for immediate neighbors)
        
        Returns:
            List of Tiles within radius
        """
        neighbors = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                tile = self.get_tile(nx, ny)
                if tile:                neighbors.append(tile)
        
        return neighbors
    def update(self) -> None:
        """
        Update world state for one simulation tick.
        
        This method advances the simulation by one tick, applying all
        world systems (plant growth, seed germination, decay, fertilizer
        effects, and resource spawning) and updating all agents.
        
        Author: Karan Vasa
        """
        self.tick += 1
        self._learning_updates_this_tick = 0
        
        # Check for calamity event
        self._check_calamity()
        
        # Update agents first (they act in the world)
        self._update_agents()
        
        # Then update world systems (physics, growth, decay)
        self.systems.update(self)
          # Clean up dead agents
        self._cleanup_dead_agents()
          # Log agent states if logger is enabled
        from agents.agent import Agent
        if Agent.logger is not None:
            Agent.logger.log_all_states(self.tick, self.agents)

    def try_acquire_learning_slot(self, agent_id: int, agent_age: int) -> bool:
        """
        Try to reserve one learning update slot for an agent this tick.

        Applies interval-based training and a global per-tick budget to
        prevent synchronized training spikes from tanking GUI FPS.

        Args:
            agent_id: Unique agent id
            agent_age: Current age (ticks)

        Returns:
            True if the agent may run a learning update now, False otherwise
        """
        if self.learning_max_updates_per_tick <= 0:
            return False

        if self.learning_enable_stagger:
            should_train_this_tick = ((agent_age + agent_id) % self.learning_train_interval_ticks) == 0
        else:
            should_train_this_tick = (agent_age % self.learning_train_interval_ticks) == 0

        if not should_train_this_tick:
            return False

        if self._learning_updates_this_tick >= self.learning_max_updates_per_tick:
            return False

        self._learning_updates_this_tick += 1
        return True

    def adapt_learning_budget(self, frame_time_ms: float, target_fps: int) -> None:
        """
        Adapt learning budget based on measured frame time.

        This keeps simulation responsive by lowering training load when frame
        time spikes, and increasing it again when there's headroom.

        Args:
            frame_time_ms: Measured frame time in milliseconds
            target_fps: Renderer target FPS
        """
        if not self.learning_adaptive_budget:
            return

        if target_fps <= 0:
            return

        target_frame_ms = 1000.0 / float(target_fps)
        high_threshold = target_frame_ms * self.learning_budget_high_frame_factor
        low_threshold = target_frame_ms * self.learning_budget_low_frame_factor

        if frame_time_ms > high_threshold:
            self.learning_max_updates_per_tick = max(
                self.learning_min_updates_per_tick,
                self.learning_max_updates_per_tick - self.learning_budget_adjust_step
            )
        elif frame_time_ms < low_threshold:
            self.learning_max_updates_per_tick = min(
                self.learning_max_budget_updates_per_tick,
                self.learning_max_updates_per_tick + self.learning_budget_adjust_step
            )
    
    def _update_agents(self) -> None:
        """Update all agents in the world."""
        new_offspring = []
        
        # Get max population from config (default: unlimited)
        max_population = None
        if self.reproduction_config:
            max_population = self.reproduction_config.get('max_population', None)
        
        for agent in list(self.agents.values()):
            if agent.alive:
                agent.update(self)
                  # Check for reproduction after update (pass config)
                if agent.can_reproduce(self.reproduction_config):
                    # Check if population limit reached
                    if max_population is not None:
                        current_population = len(self.agents) + len(new_offspring)
                        if current_population >= max_population:
                            continue  # Skip reproduction, population limit reached
                    
                    offspring = agent.reproduce(self, self.reproduction_config)
                    if offspring is not None:
                        new_offspring.append(offspring)
                        print(f"[REPRODUCTION] Agent {agent.id} -> Agent {offspring.id} (parent age: {agent.age}, parent energy: {agent.energy:.1f}, pop: {len(self.agents) + len(new_offspring)}/{max_population or 'unlimited'})")
        
        # Add all offspring to world
        for offspring in new_offspring:
            self.add_agent(offspring)
    def _cleanup_dead_agents(self) -> None:
        """Remove dead agents from the world."""
        dead_agent_ids = [
            agent_id for agent_id, agent in self.agents.items()
            if not agent.alive
        ]
        for agent_id in dead_agent_ids:
            del self.agents[agent_id]
    
    def _check_calamity(self) -> None:
        """Check if a calamity should occur and trigger it if needed."""
        if not self.calamity_config or not self.calamity_config.get('enabled', False):
            return
        
        interval = self.calamity_config.get('interval', 500)
        
        # Check if it's time for a calamity
        if self.tick - self.last_calamity_tick >= interval:
            self._trigger_calamity()
            self.last_calamity_tick = self.tick
    
    def _trigger_calamity(self) -> None:
        """Trigger a calamity event that destroys resources."""
        from world.objects import EdibleComponent, PlantComponent, SeedComponent
        
        destruction_rate = self.calamity_config.get('destruction_rate', 0.3)
        affect_plants = self.calamity_config.get('affect_plants', True)
        affect_food = self.calamity_config.get('affect_food', True)
        affect_seeds = self.calamity_config.get('affect_seeds', False)
        
        objects_to_destroy = []
        plants_destroyed = 0
        food_destroyed = 0
        seeds_destroyed = 0
        
        # Collect objects to destroy
        for obj_id, obj in list(self.objects.items()):
            should_destroy = False
            
            # Check if should affect this object type
            if affect_plants and obj.has_component(PlantComponent):
                if random.random() < destruction_rate:
                    should_destroy = True
                    plants_destroyed += 1
            elif affect_food and obj.has_component(EdibleComponent):
                if random.random() < destruction_rate:
                    should_destroy = True
                    food_destroyed += 1
            elif affect_seeds and obj.has_component(SeedComponent):
                if random.random() < destruction_rate:
                    should_destroy = True
                    seeds_destroyed += 1
            
            if should_destroy:
                objects_to_destroy.append(obj_id)
        
        # Destroy the selected objects
        for obj_id in objects_to_destroy:
            self.remove_object(obj_id)
        
        # Print calamity report
        total_destroyed = plants_destroyed + food_destroyed + seeds_destroyed
        print(f"\n⚠️  [CALAMITY] Tick {self.tick}: Environmental disaster!")
        print(f"   Destroyed {total_destroyed} objects ({destruction_rate*100:.0f}% rate)")
        print(f"   Plants: {plants_destroyed}, Food: {food_destroyed}, Seeds: {seeds_destroyed}")
        print(f"   Remaining objects: {len(self.objects)}")
    
    def add_agent(self, agent: 'Agent') -> None:
        """
        Add an agent to the world.
        
        Args:
            agent: The agent to add
        """
        self.agents[agent.id] = agent
    
    def remove_agent(self, agent_id: int) -> None:
        """
        Remove an agent from the world.
        
        Args:
            agent_id: ID of agent to remove
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def __repr__(self) -> str:
        """String representation of the world."""
        return (f"World(size={self.width}x{self.height}, tick={self.tick}, "
                f"objects={len(self.objects)}, agents={len(self.agents)})")
