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
    from utils.parallel import update_agents_parallel


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
        soil_ratio: float = 0.65,
        rock_ratio: float = 0.2,
        water_ratio: float = 0.1,
        sand_ratio: float = 0.05,
        fertility_range: Tuple[float, float] = (0.3, 1.0),
        moisture_range: Tuple[float, float] = (0.2, 0.8),
        terrain_generator: str = "legacy",
        heightmap_config: dict = None,
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
        max_neighbor_plants: int = 3,
        neighbor_radius: int = 2,
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
        parallel: bool = True,
        environment_config: dict = None,
        fire_config: dict = None,
        agents_visible: bool = False,
        agent_collision: bool = False,
        signal_config: dict = None,
        social_config: dict = None,
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
            sand_ratio: Proportion of tiles that are sand
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

        ratio_sum = soil_ratio + rock_ratio + water_ratio + sand_ratio
        if not abs(ratio_sum - 1.0) < 0.01:
            raise ValueError(f"Terrain ratios must sum to 1.0, got {ratio_sum}")

        self.width = width
        self.height = height
        self.tick = 0
        # W4: agents perceive each other in vision / contest tiles (opt-in)
        self.agents_visible = bool(agents_visible)
        self.agent_collision = bool(agent_collision)

        # W4 / Brain v3.5: SIGNAL action + decaying pheromone field (opt-in).
        sig = signal_config or {}
        self.signal_enabled = bool(sig.get("enabled", False))
        self.signal_strength = float(sig.get("strength", 1.0))
        self.signal_decay = float(sig.get("decay", 0.9))
        self.signal_diffuse = float(sig.get("diffuse", 0.0))
        # The field is only allocated when signalling is on (None = no field,
        # which perception reads as "no signal anywhere").
        self.pheromones = None
        if self.signal_enabled:
            import numpy as _np

            self.pheromones = _np.zeros((height, width), dtype=_np.float32)

        # W5: social capabilities (opt-in). When transfer_enabled, the USE
        # action transfers the first inventory item to a living agent on the
        # tile directly in front of the actor (recipient must have space).
        # Carries NO built-in reward — any trade protocol must emerge.
        soc = social_config or {}
        self.transfer_enabled = bool(soc.get("transfer_enabled", False))

        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.allow_stacking = allow_stacking  # NEW: Store stacking configuration
        self.parallel = parallel  # Enable parallel agent updates

        # Environment engine (day/night, seasons, weather — W1).
        # Disabled by default: every multiplier is then exactly 1.0 and
        # soil dynamics keep their legacy arithmetic.
        from world.environment import EnvironmentSystem

        self.environment = EnvironmentSystem(environment_config)
        self._water_adjacent_cache = None

        # Set random seed for reproducibility
        random.seed(self.seed)
        # Initialize data structures
        self.tiles: List[List[Tile]] = []
        self.objects: Dict[int, WorldObject] = {}
        self.agents: Dict[int, "Agent"] = {}  # Forward reference for Agent

        # Index: object IDs that have a TileEffectSpec (e.g., sand).
        # This prevents per-tick O(total_objects) scans in TileEffectSystem.
        self._tile_effect_object_ids: set[int] = set()

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

        # Cached world counts (lazily computed, invalidated each tick)
        self._cached_counts: dict | None = None
        self._cached_soil_stats: tuple | None = None

        # Adaptive learning budget config
        self.learning_adaptive_budget = bool(learning_adaptive_budget)
        self.learning_min_updates_per_tick = max(0, int(learning_min_updates_per_tick))
        self.learning_max_budget_updates_per_tick = max(
            self.learning_min_updates_per_tick,
            int(learning_max_budget_updates_per_tick),
        )
        self.learning_budget_adjust_step = max(1, int(learning_budget_adjust_step))
        self.learning_budget_high_frame_factor = max(
            1.0, float(learning_budget_high_frame_factor)
        )
        self.learning_budget_low_frame_factor = max(
            0.0, min(1.0, float(learning_budget_low_frame_factor))
        )

        # Clamp initial budget into adaptive bounds
        self.learning_max_updates_per_tick = min(
            self.learning_max_budget_updates_per_tick,
            max(self.learning_min_updates_per_tick, self.learning_max_updates_per_tick),
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
            seed_max_age=seed_max_age,
            max_neighbor_plants=max_neighbor_plants,
            neighbor_radius=neighbor_radius,
            fire_config=fire_config,
        )

        # Generate terrain (legacy uniform shuffle, or W2 heightmap)
        self.terrain_generator = terrain_generator
        self.heightmap_config = heightmap_config or {}
        if str(terrain_generator).lower() in ("heightmap", "biomes"):
            self._generate_terrain_heightmap(
                rock_ratio,
                water_ratio,
                sand_ratio,
                fertility_range,
                moisture_range,
            )
        else:
            self._generate_terrain(
                soil_ratio,
                rock_ratio,
                water_ratio,
                sand_ratio,
                fertility_range,
                moisture_range,
            )

    def _generate_terrain(
        self,
        soil_ratio: float,
        rock_ratio: float,
        water_ratio: float,
        sand_ratio: float,
        fertility_range: Tuple[float, float],
        moisture_range: Tuple[float, float],
    ) -> None:
        """
        Generate the world terrain with specified ratios.

        Args:
            soil_ratio: Proportion of soil tiles
            rock_ratio: Proportion of rock tiles
            water_ratio: Proportion of water tiles
            sand_ratio: Proportion of sand tiles
            fertility_range: Min and max fertility values
            moisture_range: Min and max moisture values
        """
        total_tiles = self.width * self.height
        # Create terrain type distribution
        terrain_types = (
            [TerrainType.SOIL] * int(total_tiles * soil_ratio)
            + [TerrainType.ROCK] * int(total_tiles * rock_ratio)
            + [TerrainType.WATER] * int(total_tiles * water_ratio)
            + [TerrainType.SAND] * int(total_tiles * sand_ratio)
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

                # Sand tiles have very low fertility and moisture
                if terrain_type == TerrainType.SAND:
                    fertility = random.uniform(0.0, 0.05)
                    moisture = random.uniform(0.0, 0.05)

                tile = Tile(x, y, terrain_type, fertility, moisture)
                row.append(tile)

            self.tiles.append(row)

        self._spawn_sand_objects()

    def _spawn_sand_objects(self) -> None:
        """
        Spawn sand objects on SAND terrain tiles so TileEffectSystem can
        track them (and they appear in the renderer/observation). Shared by
        the legacy and heightmap generators.
        """
        from world.object_registry import ObjectRegistry

        if ObjectRegistry.get("sand") is None:
            return
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[y][x]
                if tile.terrain_type == TerrainType.SAND:
                    sand_obj = ObjectRegistry.create("sand", x, y)
                    self.add_object(sand_obj)

    def _generate_terrain_heightmap(
        self,
        rock_ratio: float,
        water_ratio: float,
        sand_ratio: float,
        fertility_range: Tuple[float, float],
        moisture_range: Tuple[float, float],
    ) -> None:
        """
        Generate coherent terrain from an elevation heightmap (W2).

        Produces mountains (high elevation → rock), rivers that flow downhill
        into basins, and biomes (soil/sand) derived from a geography-driven
        moisture field — with fertile river corridors. Elevation is stored on
        every tile. See ``world/terrain_generation.py``.

        Args:
            rock_ratio: Fraction of highest tiles that become mountain rock
            water_ratio: Total water fraction (lakes + rivers)
            sand_ratio: Fraction of the driest remaining land → desert sand
            fertility_range: Min/max fertility for generated land
            moisture_range: Min/max moisture for generated land
        """
        from world.terrain_generation import HeightmapConfig, generate_terrain

        hc = self.heightmap_config
        cfg = HeightmapConfig(
            feature_scale=int(hc.get("feature_scale", 12)),
            octaves=int(hc.get("octaves", 4)),
            persistence=float(hc.get("persistence", 0.5)),
            rock_ratio=rock_ratio,
            water_ratio=water_ratio,
            sand_ratio=sand_ratio,
            river_sources=int(hc.get("river_sources", 6)),
            fertility_range=tuple(fertility_range),
            moisture_range=tuple(moisture_range),
        )
        result = generate_terrain(self.width, self.height, self.seed, cfg)
        self.terrain_stats = result.stats

        for y in range(self.height):
            row = []
            for x in range(self.width):
                terrain_type = result.terrain[y][x]
                fertility = float(result.fertility[y, x])
                moisture = float(result.moisture[y, x])
                elevation = float(result.elevation[y, x])
                if terrain_type == TerrainType.ROCK:
                    fertility = 0.0
                elif terrain_type == TerrainType.WATER:
                    moisture = 1.0
                row.append(
                    Tile(
                        x,
                        y,
                        terrain_type,
                        max(0.0, min(1.0, fertility)),
                        max(0.0, min(1.0, moisture)),
                        max(0.0, min(1.0, elevation)),
                    )
                )
            self.tiles.append(row)

        self._spawn_sand_objects()

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
            # Enforce one-per-tile: Check if tile already has a *real* object
            # (terrain-layer objects like sand are transparent to stacking)
            tile = self.get_tile(obj.x, obj.y)
            if tile and tile.object_ids:
                from world.object_registry import ObjectRegistry

                real_objects = [
                    oid
                    for oid in tile.object_ids
                    if not ObjectRegistry.is_terrain_layer(self.objects.get(oid))
                ]
                if real_objects:
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
                            if nearby_tile:
                                nearby_real = [
                                    oid
                                    for oid in nearby_tile.object_ids
                                    if not ObjectRegistry.is_terrain_layer(
                                        self.objects.get(oid)
                                    )
                                ]
                                if not nearby_real:
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

        # Maintain tile-effect index
        try:
            from world.object_registry import ObjectRegistry

            if ObjectRegistry.get_tile_effect(obj) is not None:
                self._tile_effect_object_ids.add(obj.id)
        except Exception:
            pass
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

        # Maintain tile-effect index
        if object_id in self._tile_effect_object_ids:
            self._tile_effect_object_ids.discard(object_id)

        del self.objects[object_id]
        return True

    @property
    def tile_effect_object_ids(self) -> set[int]:
        """Set of object IDs that have tile effects (read-only view)."""
        return self._tile_effect_object_ids

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

        return [
            self.objects[obj_id] for obj_id in tile.object_ids if obj_id in self.objects
        ]

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
                if tile:
                    neighbors.append(tile)

        return neighbors

    def update(self) -> None:
        """
        Update world state for one simulation tick.

        Advances the simulation by one tick, applying all world systems
        (plant growth, seed germination, decay, fertilizer, soil dynamics,
        tile effects, and resource spawning) and updating all agents.
        State logging is dispatched to a background thread when parallel
        mode is enabled.
        """
        self.tick += 1
        self._learning_updates_this_tick = 0

        # Environment first: every system this tick consumes its multipliers
        self.environment.update(self)

        # Pheromone field decays (and optionally diffuses) each tick (W4)
        self._update_pheromones()

        # Invalidate cached world counts (lazily recomputed on first use)
        self._cached_counts = None
        self._cached_soil_stats = None

        # Check for calamity event
        self._check_calamity()

        # Update agents first (they act in the world)
        self._update_agents()

        # Then update world systems (physics, growth, decay)
        self.systems.update(self)

        # Clean up dead agents
        self._cleanup_dead_agents()

        # Log agent states — offload to thread when parallel
        from agents.agent import Agent

        if Agent.logger is not None:
            if self.parallel:
                from utils.parallel import get_io_pool

                # Snapshot in main thread to avoid iterating a mutating dict
                agent_snapshot = list(self.agents.values())
                # I/O pool is single-threaded so file writes never overlap
                self._log_future = get_io_pool().submit(
                    Agent.logger.log_all_states, self.tick, agent_snapshot
                )
            else:
                Agent.logger.log_all_states(self.tick, self.agents)

    def _update_pheromones(self) -> None:
        """
        Decay (and optionally diffuse) the SIGNAL pheromone field each tick.

        No-op when signalling is disabled. Decay is geometric
        (``signal_decay``); a small ``signal_diffuse`` fraction spreads to the
        4-neighbourhood so trails blur over time (a stigmergic medium).
        """
        if self.pheromones is None:
            return
        if self.signal_diffuse > 0.0:
            f = self.pheromones
            blurred = f.copy()
            blurred[1:, :] += self.signal_diffuse * f[:-1, :]
            blurred[:-1, :] += self.signal_diffuse * f[1:, :]
            blurred[:, 1:] += self.signal_diffuse * f[:, :-1]
            blurred[:, :-1] += self.signal_diffuse * f[:, 1:]
            blurred /= 1.0 + 4.0 * self.signal_diffuse
            self.pheromones = blurred
        self.pheromones *= self.signal_decay
        # Drop negligible residue so the field doesn't carry float dust forever
        self.pheromones[self.pheromones < 1e-3] = 0.0

    def emit_signal(self, x: int, y: int, strength: float = None) -> bool:
        """
        Deposit a signal at (x, y) on the pheromone field (W4 SIGNAL action).

        Returns False when signalling is disabled or the field is absent, so
        the caller can treat SIGNAL as a no-op in that case.
        """
        if self.pheromones is None:
            return False
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        s = self.signal_strength if strength is None else strength
        self.pheromones[y, x] = min(1.0, self.pheromones[y, x] + s)
        return True

    @property
    def water_adjacent(self) -> set:
        """
        Tiles orthogonally adjacent to WATER (computed once — water tiles
        are never created or destroyed). Used by the environment-enabled
        soil dynamics: these tiles recover moisture even without rain.
        """
        if self._water_adjacent_cache is None:
            adjacent = set()
            for y in range(self.height):
                for x in range(self.width):
                    if self.tiles[y][x].terrain_type == TerrainType.WATER:
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            adjacent.add((x + dx, y + dy))
            self._water_adjacent_cache = adjacent
        return self._water_adjacent_cache

    def get_cached_object_counts(self) -> dict:
        """
        Return cached counts of food, plants, seeds, and alive agents.

        Recomputed at most once per tick (invalidated at start of update()).
        This avoids O(agents * objects) scanning when every agent's logger
        calls sum(...) independently.

        Returns:
            dict with keys: total_food, total_plants, total_seeds, alive_agents
        """
        if self._cached_counts is not None:
            return self._cached_counts

        from world.objects import EdibleComponent, PlantComponent, SeedComponent

        food = 0
        plants = 0
        seeds = 0
        for obj in self.objects.values():
            if obj.has_component(EdibleComponent):
                food += 1
            if obj.has_component(PlantComponent):
                plants += 1
            if obj.has_component(SeedComponent):
                seeds += 1
        alive = sum(1 for a in self.agents.values() if a.alive)
        self._cached_counts = {
            "total_food": food,
            "total_plants": plants,
            "total_seeds": seeds,
            "alive_agents": alive,
        }
        return self._cached_counts

    def get_cached_soil_stats(self) -> tuple:
        """
        Return cached avg fertility and avg moisture for soil tiles.

        Recomputed at most once per tick.

        Returns:
            (avg_fertility, avg_moisture)
        """
        if self._cached_soil_stats is not None:
            return self._cached_soil_stats

        total_fertility = 0.0
        total_moisture = 0.0
        tile_count = 0
        for row in self.tiles:
            for tile in row:
                if tile.terrain_type.value == "soil":
                    total_fertility += tile.fertility
                    total_moisture += tile.moisture
                    tile_count += 1
        if tile_count > 0:
            self._cached_soil_stats = (
                total_fertility / tile_count,
                total_moisture / tile_count,
            )
        else:
            self._cached_soil_stats = (0.0, 0.0)
        return self._cached_soil_stats

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
            should_train_this_tick = (
                (agent_age + agent_id) % self.learning_train_interval_ticks
            ) == 0
        else:
            should_train_this_tick = (
                agent_age % self.learning_train_interval_ticks
            ) == 0

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
                self.learning_max_updates_per_tick - self.learning_budget_adjust_step,
            )
        elif frame_time_ms < low_threshold:
            self.learning_max_updates_per_tick = min(
                self.learning_max_budget_updates_per_tick,
                self.learning_max_updates_per_tick + self.learning_budget_adjust_step,
            )

    def _update_agents(self) -> None:
        """Update all agents — uses parallel pipeline when enabled."""
        if self.parallel:
            from utils.parallel import update_agents_parallel

            update_agents_parallel(self)
        else:
            self._update_agents_serial()

    def _update_agents_serial(self) -> None:
        """Original serial agent update loop (fallback)."""
        new_offspring = []

        max_population = None
        if self.reproduction_config:
            max_population = self.reproduction_config.get("max_population", None)

        for agent in list(self.agents.values()):
            if agent.alive:
                agent.update(self)
                if agent.can_reproduce(self.reproduction_config):
                    if max_population is not None:
                        current_population = len(self.agents) + len(new_offspring)
                        if current_population >= max_population:
                            continue
                    offspring = agent.reproduce(self, self.reproduction_config)
                    if offspring is not None:
                        new_offspring.append(offspring)

        for offspring in new_offspring:
            self.add_agent(offspring)

    def _cleanup_dead_agents(self) -> None:
        """Remove dead agents from the world."""
        dead_agent_ids = [
            agent_id for agent_id, agent in self.agents.items() if not agent.alive
        ]
        for agent_id in dead_agent_ids:
            del self.agents[agent_id]

    def _check_calamity(self) -> None:
        """Check if a calamity should occur and trigger it if needed."""
        if not self.calamity_config or not self.calamity_config.get("enabled", False):
            return

        interval = self.calamity_config.get("interval", 500)

        # Check if it's time for a calamity
        if self.tick - self.last_calamity_tick >= interval:
            self._trigger_calamity()
            self.last_calamity_tick = self.tick

    def _trigger_calamity(self) -> None:
        """Trigger a calamity event that destroys resources."""
        from world.objects import EdibleComponent, PlantComponent, SeedComponent

        destruction_rate = self.calamity_config.get("destruction_rate", 0.3)
        affect_plants = self.calamity_config.get("affect_plants", True)
        affect_food = self.calamity_config.get("affect_food", True)
        affect_seeds = self.calamity_config.get("affect_seeds", False)

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
        print(
            f"   Destroyed {total_destroyed} objects ({destruction_rate*100:.0f}% rate)"
        )
        print(
            f"   Plants: {plants_destroyed}, Food: {food_destroyed}, Seeds: {seeds_destroyed}"
        )
        print(f"   Remaining objects: {len(self.objects)}")

    def add_agent(self, agent: "Agent") -> None:
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
        return (
            f"World(size={self.width}x{self.height}, tick={self.tick}, "
            f"objects={len(self.objects)}, agents={len(self.agents)})"
        )
