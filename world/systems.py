"""
World update systems for the simulation.

This module implements all the systems that update the world state each tick:
- Plant growth and aging
- Seed germination
- Object decay (freshness reduction)
- Fertilizer effects
- Resource spawning (safety net)

Author: Karan Vasa
"""

import random
from typing import Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from world.world import World

from world.objects import (
    WorldObject,
    PlantComponent,
    SeedComponent,
    EdibleComponent,
    FertilizerComponent
)
from world.object_registry import ObjectRegistry


class PlantGrowthSystem:
    """
    System for updating plant age and lifecycle.
    Handles plant aging, maturity tracking, and death from old age.
    Returns nutrients to soil when plants die.
    
    Author: Karan Vasa
    """
    
    def __init__(self, soil_system: 'SoilDynamicsSystem' = None):
        """
        Initialize plant growth system.
        
        Args:
            soil_system: Reference to soil dynamics system for nutrient return
            
        Author: Karan Vasa
        """
        self.soil_system = soil_system
    
    def update(self, world: 'World') -> None:
        """
        Update all plants in the world.
        
        Applies growth_multiplier from any TileEffectSpec objects on the
        same tile (e.g. sand slows growth).
        
        Args:
            world: World instance to update
            
        Author: Karan Vasa
        """
        objects_to_remove = []
        dead_plant_positions = []
        
        for obj_id, obj in world.objects.items():
            if not obj.has_component(PlantComponent):
                continue
            
            plant = obj.get_component(PlantComponent)
            
            # Apply growth multiplier from tile effects
            growth_mult = _get_tile_growth_multiplier(world, obj.x, obj.y)
            plant.age += growth_mult
            
            # Check if plant died of old age
            if not plant.is_alive():
                objects_to_remove.append(obj_id)
                dead_plant_positions.append((obj.x, obj.y))
        
        # Remove dead plants
        for obj_id in objects_to_remove:
            world.remove_object(obj_id)
        
        # Return nutrients to soil from dead plants
        if self.soil_system:
            for x, y in dead_plant_positions:
                self.soil_system.return_nutrients_to_soil(world, x, y)


class SeedGerminationSystem:
    """
    System for seed germination into plants.
    
    Checks if seeds meet growth requirements and converts them to plants.
    Seeds have a probabilistic success rate to simulate natural germination failure.
    
    Author: Karan Vasa
    """
    
    def __init__(
        self, 
        plant_mature_age: int = 100, 
        plant_max_age: int = 500,
        germination_success_rate: float = 0.75
    ):
        """
        Initialize germination system.
        
        Args:
            plant_mature_age: Age at which new plants mature
            plant_max_age: Maximum age for new plants
            germination_success_rate: Probability that a seed successfully germinates
            
        Author: Karan Vasa
        """
        self.plant_mature_age = plant_mature_age
        self.plant_max_age = plant_max_age
        self.germination_success_rate = germination_success_rate
    
    def update(self, world: 'World') -> None:
        """
        Update all seeds and germinate when ready.
        
        Seeds must meet soil requirements and pass a probability check
        to successfully germinate into plants.
          Args:
            world: World instance to update
            
        Author: Karan Vasa
        """
        seeds_to_germinate = []
        seeds_to_fail = []  # Seeds that fail to germinate
        seeds_to_rot = []  # Seeds that are too old and rot away
        
        for obj_id, obj in world.objects.items():
            if not obj.has_component(SeedComponent):
                continue
            
            seed = obj.get_component(SeedComponent)
            tile = world.get_tile(obj.x, obj.y)
            
            if not tile:
                continue
            
            # Increment time in soil (seed ages)
            seed.time_in_soil += 1
            
            # Check if seed has rotted from old age
            if seed.time_in_soil >= seed.max_age:
                seeds_to_rot.append(obj_id)
                continue
            
            # Check if seed is on suitable soil
            if not tile.is_plantable():
                continue
            
            # Check fertility and moisture requirements
            if tile.fertility < seed.required_fertility:
                continue
            if tile.moisture < seed.required_moisture:
                continue
            
            # Check if ready to attempt germination
            if seed.time_in_soil >= seed.grow_time:
                # Apply germination multiplier from tile effects
                germ_mult = _get_tile_germination_multiplier(world, obj.x, obj.y)
                effective_rate = self.germination_success_rate * germ_mult
                
                # Also check if any object on tile blocks growth
                if _tile_blocks_growth(world, obj.x, obj.y):
                    # Growth is blocked — seed stays alive and keeps waiting
                    continue
                elif random.random() < effective_rate:
                    seeds_to_germinate.append((obj_id, obj.x, obj.y, seed.plant_type))
                else:
                    # Seed failed to germinate - remove it
                    seeds_to_fail.append(obj_id)
        
        # Remove rotted seeds
        for obj_id in seeds_to_rot:
            world.remove_object(obj_id)
        
        # Remove failed seeds
        for obj_id in seeds_to_fail:
            world.remove_object(obj_id)
        
        # Germinate successful seeds (convert to plants)
        for obj_id, x, y, plant_type in seeds_to_germinate:
            # Remove seed
            world.remove_object(obj_id)
            
            # Look up what this seed grows into via registry
            # plant_type comes from SeedComponent.plant_type (= SeedSpec.grows_into)
            defn = ObjectRegistry.get(plant_type)
            if defn is not None:
                plant_obj = ObjectRegistry.create(
                    plant_type, x, y,
                    mature_age=self.plant_mature_age,
                    plant_max_age=self.plant_max_age,
                )
            else:
                # Fallback for unregistered plant types
                plant_obj = WorldObject(x, y)
                plant_obj.add_component(PlantComponent(
                    mature_age=self.plant_mature_age,
                    max_age=self.plant_max_age,
                    spawn_resource_type="berry",
                    spawn_rate=0.1
                ))
            world.add_object(plant_obj)


class DecaySystem:
    """
    System for object decay (freshness reduction).
    
    Reduces freshness of edible objects over time and removes spoiled items.
    When berries fully decay, they have a chance to drop seeds and return nutrients to soil.
    
    Author: Karan Vasa
    """
    
    def __init__(self, decay_rate: float = 0.01, seed_drop_chance: float = 0.7, soil_system: 'SoilDynamicsSystem' = None, seed_max_age: int = 200):
        """
        Initialize decay system.
        
        Args:
            decay_rate: Rate at which freshness decreases per tick
            seed_drop_chance: Probability that decomposed fruit drops a seed
            soil_system: Reference to soil dynamics system for nutrient return
            seed_max_age: Maximum age for spawned seeds before they rot
            
        Author: Karan Vasa
        """
        self.decay_rate = decay_rate
        self.seed_drop_chance = seed_drop_chance
        self.soil_system = soil_system
        self.seed_max_age = seed_max_age
    
    def update(self, world: 'World') -> None:
        """
        Update freshness of all edible objects and handle decomposition.
        
        When berries fully decay, they have a chance to drop seeds at their location,
        creating a natural reproduction cycle. Decomposed berries also return nutrients to soil.
        
        Args:
            world: World instance to update
            
        Author: Karan Vasa
        """
        objects_to_remove = []
        seeds_to_spawn = []  # (x, y, decompose_into_type_id)
        decomposed_positions = []  # (x, y, nutrient_return)
        
        for obj_id, obj in world.objects.items():
            if not obj.has_component(EdibleComponent):
                continue
            
            edible = obj.get_component(EdibleComponent)
            
            # Per-type decay rate from registry, fallback to global
            physics = ObjectRegistry.get_physics(obj)
            rate = physics.decay_rate if (physics and physics.decay_rate > 0) else self.decay_rate
            
            # Reduce freshness
            edible.freshness -= rate
            
            # Handle completely spoiled items
            if edible.freshness <= 0.0:
                objects_to_remove.append(obj_id)
                
                # Per-type decomposition chain from registry
                if physics and physics.decompose_into:
                    chance = physics.decompose_chance
                    nutrient = physics.nutrient_return
                    if random.random() < chance:
                        seeds_to_spawn.append((obj.x, obj.y, physics.decompose_into))
                    decomposed_positions.append((obj.x, obj.y, nutrient))
                else:
                    # Legacy fallback: global seed_drop_chance
                    if random.random() < self.seed_drop_chance:
                        seeds_to_spawn.append((obj.x, obj.y, "berry_seed"))
                    decomposed_positions.append((obj.x, obj.y, 0.0))
        
        # Remove spoiled objects
        for obj_id in objects_to_remove:
            world.remove_object(obj_id)
        
        # Return nutrients to soil from decomposed items
        if self.soil_system:
            for x, y, nutrient in decomposed_positions:
                if nutrient > 0:
                    # Per-type nutrient return
                    tile = world.get_tile(x, y)
                    if tile and tile.is_plantable():
                        tile.fertility = min(1.0, tile.fertility + nutrient)
                else:
                    # Fallback to global nutrient return
                    self.soil_system.return_nutrients_to_soil(world, x, y)
        
        # Spawn decomposition products (seeds, etc.)
        for x, y, spawn_type_id in seeds_to_spawn:
            if world.is_valid_position(x, y):
                defn = ObjectRegistry.get(spawn_type_id)
                if defn is not None:
                    spawn_obj = ObjectRegistry.create(
                        spawn_type_id, x, y,
                        seed_max_age=self.seed_max_age,
                    )
                else:
                    # Fallback for unregistered types
                    spawn_obj = WorldObject(x, y)
                    spawn_obj.add_component(SeedComponent(
                        plant_type="berry_plant",
                        grow_time=50,
                        required_fertility=0.3,
                        required_moisture=0.2,
                        max_age=self.seed_max_age
                    ))
                world.add_object(spawn_obj)


class FertilizerSystem:
    """
    System for applying fertilizer effects to nearby tiles.
    
    Increases fertility of tiles within fertilizer radius and manages
    fertilizer duration.
    
    Author: Karan Vasa
    """
    
    def update(self, world: 'World') -> None:
        """
        Update all fertilizers and apply effects.
        
        Args:
            world: World instance to update
            
        Author: Karan Vasa
        """
        fertilizers_to_remove = []
        fertilizers_to_apply: List[tuple] = []
        
        # First pass: collect fertilizer effects
        for obj_id, obj in world.objects.items():
            if not obj.has_component(FertilizerComponent):
                continue
            
            fert = obj.get_component(FertilizerComponent)
            
            # Decrease duration
            fert.duration -= 1
            
            # Mark expired fertilizers for removal
            if fert.duration <= 0:
                fertilizers_to_remove.append(obj_id)
                continue
            
            # Collect tiles to affect
            for dy in range(-fert.radius, fert.radius + 1):
                for dx in range(-fert.radius, fert.radius + 1):
                    # Manhattan distance check
                    if abs(dx) + abs(dy) <= fert.radius:
                        target_x = obj.x + dx
                        target_y = obj.y + dy
                        fertilizers_to_apply.append((target_x, target_y, fert.fertility_boost))
        
        # Second pass: apply fertility boosts
        tile_boosts: Dict[tuple, float] = {}
        for x, y, boost in fertilizers_to_apply:
            tile = world.get_tile(x, y)
            if tile and tile.terrain_type.value == "soil":
                key = (x, y)
                tile_boosts[key] = tile_boosts.get(key, 0.0) + boost
        
        # Apply accumulated boosts (capped at 1.0)
        for (x, y), boost in tile_boosts.items():
            tile = world.get_tile(x, y)
            if tile:
                tile.fertility = min(1.0, tile.fertility + boost / 10.0)  # Gradual increase
        
        # Remove expired fertilizers
        for obj_id in fertilizers_to_remove:
            world.remove_object(obj_id)


class SoilDynamicsSystem:
    """
    System for soil resource dynamics.
    
    Manages:
    - Fertility depletion from growing plants
    - Moisture consumption by plants
    - Natural fertility recovery on empty soil
    - Natural moisture evaporation
    - Natural moisture recovery (rain/groundwater)
    - Nutrient return when plants/berries decompose
    
    Author: Karan Vasa
    """
    
    def __init__(
        self,
        fertility_consumption: float = 0.001,
        moisture_consumption: float = 0.0005,
        fertility_recovery_rate: float = 0.0005,
        moisture_evaporation_rate: float = 0.0002,
        moisture_recovery_rate: float = 0.0008,
        fertility_return_on_death: float = 0.15
    ):
        """
        Initialize soil dynamics system.
        
        Args:
            fertility_consumption: Fertility consumed per plant per tick
            moisture_consumption: Moisture consumed per plant per tick
            fertility_recovery_rate: Fertility recovery for empty soil per tick
            moisture_evaporation_rate: Natural moisture loss per tick
            moisture_recovery_rate: Moisture recovery from rain/groundwater per tick
            fertility_return_on_death: Fertility returned when organic matter decomposes
            
        Author: Karan Vasa
        """
        self.fertility_consumption = fertility_consumption
        self.moisture_consumption = moisture_consumption
        self.fertility_recovery_rate = fertility_recovery_rate
        self.moisture_evaporation_rate = moisture_evaporation_rate
        self.moisture_recovery_rate = moisture_recovery_rate
        self.fertility_return_on_death = fertility_return_on_death
    
    def update(self, world: 'World') -> None:
        """
        Update soil dynamics across the world.

        Pass 1 (serial): identify plant-occupied tiles + apply consumption.
        Pass 2 (row-parallel when world.parallel=True): evaporation,
                 rain, fertility recovery across all tiles.

        Args:
            world: World instance to update
        """
        # Track which tiles have plants
        occupied_tiles: Set[tuple] = set()

        # First pass: identify tiles with plants and apply consumption
        for obj_id, obj in world.objects.items():
            if obj.has_component(PlantComponent):
                occupied_tiles.add((obj.x, obj.y))
                tile = world.tiles[obj.y][obj.x]

                if tile.is_plantable():
                    tile.fertility = max(0.0, tile.fertility - self.fertility_consumption)
                    tile.moisture = max(0.0, tile.moisture - self.moisture_consumption)

        # Second pass: per-tile dynamics (parallelisable by row)
        if getattr(world, 'parallel', False) and world.height >= 8:
            from utils.parallel import get_pool
            pool = get_pool()

            # Chunk rows into ~4 batches
            n_workers = 4
            chunk = max(1, world.height // n_workers)
            futures = []
            for start_y in range(0, world.height, chunk):
                end_y = min(start_y + chunk, world.height)
                futures.append(
                    pool.submit(
                        self._update_tile_rows, world, start_y, end_y, occupied_tiles
                    )
                )
            for f in futures:
                f.result()
        else:
            self._update_tile_rows(world, 0, world.height, occupied_tiles)

    def _update_tile_rows(
        self, world: 'World', start_y: int, end_y: int, occupied_tiles: Set[tuple]
    ) -> None:
        """Process a contiguous range of tile rows."""
        evap = self.moisture_evaporation_rate
        rain = self.moisture_recovery_rate
        fert_rec = self.fertility_recovery_rate

        for y in range(start_y, end_y):
            row = world.tiles[y]
            for x in range(len(row)):
                tile = row[x]
                if not tile.is_plantable():
                    continue
                tile.moisture = max(0.0, tile.moisture - evap)
                tile.moisture = min(1.0, tile.moisture + rain)
                if (x, y) not in occupied_tiles:
                    tile.fertility = min(1.0, tile.fertility + fert_rec)
    
    def return_nutrients_to_soil(self, world: 'World', x: int, y: int) -> None:
        """
        Return nutrients to soil when organic matter decomposes.
        
        Called when plants die or berries rot away.
        
        Args:
            world: World instance
            x: X coordinate
            y: Y coordinate
            
        Author: Karan Vasa
        """
        tile = world.get_tile(x, y)
        if tile and tile.is_plantable():
            tile.fertility = min(1.0, tile.fertility + self.fertility_return_on_death)


class ResourceSpawnSystem:
    """
    System for spawning resources from mature plants.
    
    Mature plants have a chance to spawn berries or seeds each tick.
    Also provides a safety net by spawning random resources if the world
    becomes too depleted.
    
    Author: Karan Vasa
    """
    
    def __init__(
        self,
        berry_calories: float = 20.0,
        safety_spawn_rate: float = 0.01,
        min_resources: int = 10,
        seed_max_age: int = 200
    ):
        """
        Initialize resource spawn system.
        
        Args:
            berry_calories: Calorie value for spawned berries
            safety_spawn_rate: Probability of safety spawn per tick
            min_resources: Minimum edible resources before safety spawning
            seed_max_age: Maximum age for spawned seeds before they rot
            
        Author: Karan Vasa
        """
        self.berry_calories = berry_calories
        self.safety_spawn_rate = safety_spawn_rate
        self.min_resources = min_resources
        self.seed_max_age = seed_max_age
    
    def update(self, world: 'World') -> None:
        """
        Update resource spawning from plants and safety net.
        
        Args:
            world: World instance to update
            
        Author: Karan Vasa
        """
        # Plant-based resource spawning
        # Create a list copy to avoid modifying dict during iteration
        for obj_id, obj in list(world.objects.items()):
            if not obj.has_component(PlantComponent):
                continue
            
            plant = obj.get_component(PlantComponent)
            
            # Only mature plants spawn resources
            if not plant.is_mature():
                continue
            
            # Chance to spawn resource (modulated by tile effects)
            spawn_mult = _get_tile_spawn_rate_multiplier(world, obj.x, obj.y)
            effective_rate = plant.spawn_rate * spawn_mult
            if random.random() < effective_rate:
                self._spawn_resource_near(world, obj.x, obj.y, plant.spawn_resource_type)
        
        # Safety net: spawn resources if world is depleted (use cached counts)
        edible_count = world.get_cached_object_counts()['total_food']
        
        if edible_count < self.min_resources and random.random() < self.safety_spawn_rate:
            # Spawn resource at random location
            x = random.randint(0, world.width - 1)
            y = random.randint(0, world.height - 1)
            self._spawn_resource_near(world, x, y, "berry")
    
    def _spawn_resource_near(
        self,
        world: 'World',
        center_x: int,
        center_y: int,
        resource_type: str
    ) -> bool:
        """
        Spawn a resource near the given position.
        
        Args:
            world: World instance
            center_x: Center X coordinate
            center_y: Center Y coordinate
            resource_type: Type of resource to spawn
            
        Returns:
            True if spawned successfully, False otherwise
            
        Author: Karan Vasa
        """
        # Try to find empty adjacent tile
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
        random.shuffle(offsets)
        for dx, dy in offsets:
            x = center_x + dx
            y = center_y + dy
            
            if not world.is_valid_position(x, y):
                continue
            
            # Check if tile has space (respects allow_stacking configuration)
            objects_here = world.get_objects_at(x, y)
            if world.allow_stacking:
                if len(objects_here) >= 3:  # Max 3 objects per tile in stacking mode
                    continue
            else:
                if len(objects_here) >= 1:  # Max 1 object per tile in no-stacking mode
                    continue
            
            # Spawn resource via registry (with legacy fallback)
            defn = ObjectRegistry.get(resource_type)
            if defn is not None:
                new_obj = ObjectRegistry.create(
                    resource_type, x, y,
                    calories=self.berry_calories,
                    seed_max_age=self.seed_max_age,
                )
                world.add_object(new_obj)
                return True
            
            # Legacy fallback for unregistered types
            if resource_type == "berry":
                berry = WorldObject(x, y)
                berry.add_component(EdibleComponent(
                    calories=self.berry_calories,
                    toxicity=0.0,
                    freshness=1.0
                ))
                world.add_object(berry)
                return True
            elif resource_type == "seed":
                seed = WorldObject(x, y)
                seed.add_component(SeedComponent(
                    plant_type="berry_plant",
                    grow_time=50,
                    required_fertility=0.3,
                    required_moisture=0.2,
                    max_age=self.seed_max_age
                ))
                world.add_object(seed)
                return True
        
        return False


class WorldSystemManager:
    """
    Manager for all world update systems.
    
    Coordinates the execution of all systems in the correct order.
    
    Author: Karan Vasa
    """
    
    def __init__(
        self,
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
        seed_max_age: int = 200
    ):
        """
        Initialize system manager with all systems.
        
        Args:
            plant_mature_age: Age at which plants mature
            plant_max_age: Maximum plant lifespan
            decay_rate: Freshness decay rate per tick
            seed_drop_chance: Chance for decomposed berries to drop seeds
            germination_success_rate: Probability that seeds successfully germinate
            fertility_consumption: Fertility consumed by plants per tick
            moisture_consumption: Moisture consumed by plants per tick
            fertility_recovery_rate: Fertility recovery for empty soil per tick
            moisture_evaporation_rate: Natural moisture loss per tick
            moisture_recovery_rate: Moisture recovery from rain/groundwater per tick
            fertility_return_on_death: Fertility returned when organic matter decomposes
            berry_calories: Calorie value for berries
            safety_spawn_rate: Safety net spawn probability
            min_resources: Minimum resources before safety spawning
            seed_max_age: Maximum age before seed rots (in ticks)
            
        Author: Karan Vasa
        """
        # Store seed_max_age for use by systems
        self.seed_max_age = seed_max_age
        
        # Create soil dynamics system first (needed by other systems)
        self.soil_dynamics = SoilDynamicsSystem(
            fertility_consumption,
            moisture_consumption,
            fertility_recovery_rate,
            moisture_evaporation_rate,
            moisture_recovery_rate,
            fertility_return_on_death
        )
        
        # Create systems with references to soil dynamics
        self.plant_growth = PlantGrowthSystem(self.soil_dynamics)
        self.seed_germination = SeedGerminationSystem(
            plant_mature_age, 
            plant_max_age,
            germination_success_rate
        )
        self.decay = DecaySystem(decay_rate, seed_drop_chance, self.soil_dynamics, seed_max_age)
        self.fertilizer = FertilizerSystem()
        self.tile_effect = TileEffectSystem()
        self.resource_spawn = ResourceSpawnSystem(
            berry_calories,
            safety_spawn_rate,
            min_resources,
            seed_max_age
        )
    
    def update(self, world: 'World') -> None:
        """
                Run all systems.

                Parallel mode is intentionally conservative: only stages that do not
                mutate shared dictionaries concurrently are parallelised.  Currently
                the heavy tile-scan in SoilDynamics can run row-parallel internally.

                Execution pipeline:
                    Serial: PlantGrowth (mutates world.objects)
                    Serial: Decay (mutates world.objects)
                    Serial: SeedGermination (mutates world.objects)
                    Serial: Fertilizer (writes tile.fertility)
                    Serial: SoilDynamics (writes tile.fertility + moisture, row-parallel internally)
                    Serial: TileEffect (writes tile values, adds/removes objects)
                    Serial: ResourceSpawn (adds objects, reads cached counts)

        Args:
            world: World instance to update
        """
        # NOTE: PlantGrowth/Decay both mutate world.objects (add/remove).
        # Running them concurrently is not thread-safe.
        self.plant_growth.update(world)
        self.decay.update(world)

        # Sequential stages (dependency chain)
        self.seed_germination.update(world)
        self.fertilizer.update(world)
        self.soil_dynamics.update(world)
        self.tile_effect.update(world)
        self.resource_spawn.update(world)


# ---------------------------------------------------------------------------
# Tile-effect helpers (used by growth/germination/spawn systems)
# ---------------------------------------------------------------------------

def _get_tile_effect_objects(world: 'World', x: int, y: int):
    """Yield TileEffectSpec for every object at (x, y) that has one."""
    tile = world.get_tile(x, y)
    if not tile:
        return
    for obj_id in tile.object_ids:
        obj = world.objects.get(obj_id)
        if obj is None:
            continue
        te = ObjectRegistry.get_tile_effect(obj)
        if te is not None:
            yield te


def _get_tile_growth_multiplier(world: 'World', x: int, y: int) -> float:
    """Return the combined growth multiplier at a tile (multiplicative)."""
    mult = 1.0
    for te in _get_tile_effect_objects(world, x, y):
        mult *= te.growth_multiplier
    return mult


def _get_tile_germination_multiplier(world: 'World', x: int, y: int) -> float:
    """Return the combined germination multiplier at a tile."""
    mult = 1.0
    for te in _get_tile_effect_objects(world, x, y):
        mult *= te.germination_multiplier
    return mult


def _get_tile_spawn_rate_multiplier(world: 'World', x: int, y: int) -> float:
    """Return the combined spawn-rate multiplier at a tile."""
    mult = 1.0
    for te in _get_tile_effect_objects(world, x, y):
        mult *= te.spawn_rate_multiplier
    return mult


def _tile_blocks_growth(world: 'World', x: int, y: int) -> bool:
    """Return True if any object at (x, y) has blocks_growth=True."""
    tile = world.get_tile(x, y)
    if not tile:
        return False
    for obj_id in tile.object_ids:
        obj = world.objects.get(obj_id)
        if obj is None:
            continue
        interaction = ObjectRegistry.get_interaction(obj)
        if interaction.blocks_growth:
            return True
    return False


# ---------------------------------------------------------------------------
# TileEffectSystem – spreading, fertility/moisture clamping
# ---------------------------------------------------------------------------

class TileEffectSystem:
    """
    System that handles tile-effect objects (sand, etc.).

    Per tick this system:
    1. Clamps fertility / moisture on tiles where effect objects sit.
    2. Tracks how long neighbouring *soil* tiles have gone without a
       blocking object (plant) nearby.
    3. When the spread interval is exceeded and the random check passes,
       converts the neighbour tile and spawns a new tile-effect object.

    Author: Karan Vasa
    """

    def __init__(self):
        """Initialise the system with an empty exposure tracker."""
        # Maps (x, y, source_type_id) -> ticks without a blocker
        self._exposure_ticks: Dict[tuple, int] = {}
        # Maps (x, y, type_id) -> ticks a blocker has been present (for reclaim)
        self._reclaim_ticks: Dict[tuple, int] = {}

    def update(self, world: 'World') -> None:
        """
        Run the tile-effect system for one tick.

        Args:
            world: World instance to update.
        """
        from world.tiles import TerrainType

        objects_to_spawn: List[tuple] = []   # (type_id, x, y)
        tiles_to_convert: List[tuple] = []   # (x, y, terrain_str)
        objects_to_remove: List[int] = []    # obj_ids of reclaimed effect objects
        tiles_to_reclaim: List[tuple] = []   # (x, y, terrain_str, fertility_restore)

        # Collect all tile-effect sources
        effect_sources: List[tuple] = []  # (obj, TileEffectSpec)
        stale_ids: List[int] = []
        ids = getattr(world, 'tile_effect_object_ids', None)
        if ids is None:
            # Fallback for older World implementations
            for obj_id, obj in world.objects.items():
                te = ObjectRegistry.get_tile_effect(obj)
                if te is not None:
                    effect_sources.append((obj, te))
        else:
            # Iterate only tile-effect objects (fast when world has many non-effect objects)
            for obj_id in tuple(ids):
                obj = world.objects.get(obj_id)
                if obj is None:
                    stale_ids.append(obj_id)
                    continue
                te = ObjectRegistry.get_tile_effect(obj)
                if te is None:
                    stale_ids.append(obj_id)
                    continue
                effect_sources.append((obj, te))

            # Prune stale index entries
            for obj_id in stale_ids:
                try:
                    ids.discard(obj_id)
                except Exception:
                    pass

        # 1) Clamp fertility / moisture on tiles with effect objects
        for obj, te in effect_sources:
            tile = world.get_tile(obj.x, obj.y)
            if tile:
                if te.fertility_override >= 0:
                    tile.fertility = min(tile.fertility, te.fertility_override)
                if te.moisture_override >= 0:
                    tile.moisture = min(tile.moisture, te.moisture_override)

        # 2) Evaluate reclamation: if a blocker sits on the effect tile
        #    long enough, convert back and remove the effect object.
        active_reclaim_keys: Set[tuple] = set()

        for obj, te in effect_sources:
            if not te.reclaim_terrain or te.reclaim_interval <= 0:
                continue
            key = (obj.x, obj.y, obj.type_id)
            active_reclaim_keys.add(key)

            has_blocker = self._tile_has_blocker(
                world, obj.x, obj.y, te.spread_blocked_by
            )
            if has_blocker:
                current = self._reclaim_ticks.get(key, 0) + 1
                self._reclaim_ticks[key] = current
                if current >= te.reclaim_interval:
                    objects_to_remove.append(obj.id)
                    tiles_to_reclaim.append((
                        obj.x, obj.y, te.reclaim_terrain, 0.3
                    ))
                    self._reclaim_ticks[key] = 0
            else:
                self._reclaim_ticks[key] = 0

        # Prune stale reclaim entries
        stale_reclaim = [k for k in self._reclaim_ticks if k not in active_reclaim_keys]
        for k in stale_reclaim:
            del self._reclaim_ticks[k]

        # 3) Evaluate spread for each effect source
        active_spread_keys: Set[tuple] = set()

        for obj, te in effect_sources:
            if not te.spread_type_id:
                continue

            # Check each tile within spread_radius
            for dy in range(-te.spread_radius, te.spread_radius + 1):
                for dx in range(-te.spread_radius, te.spread_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    # Manhattan distance
                    if abs(dx) + abs(dy) > te.spread_radius:
                        continue

                    nx, ny = obj.x + dx, obj.y + dy
                    tile = world.get_tile(nx, ny)
                    if tile is None:
                        continue

                    # Only spread to soil tiles
                    if tile.terrain_type != TerrainType.SOIL:
                        continue

                    # Check if tile already has the effect object
                    already_has_effect = False
                    for oid in tile.object_ids:
                        nearby_obj = world.objects.get(oid)
                        if nearby_obj and getattr(nearby_obj, 'type_id', '') == te.spread_type_id:
                            already_has_effect = True
                            break
                    if already_has_effect:
                        continue

                    key = (nx, ny, te.spread_type_id)
                    active_spread_keys.add(key)

                    # Check if a blocking object is present on the neighbour tile
                    has_blocker = self._tile_has_blocker(world, nx, ny, te.spread_blocked_by)
                    if has_blocker:
                        # Reset exposure counter
                        self._exposure_ticks[key] = 0
                        continue

                    # Increment exposure timer
                    current = self._exposure_ticks.get(key, 0) + 1
                    self._exposure_ticks[key] = current

                    # Check if spread should trigger
                    if current >= te.spread_interval:
                        if random.random() < te.spread_chance:
                            objects_to_spawn.append((te.spread_type_id, nx, ny))
                            if te.converts_terrain:
                                tiles_to_convert.append((nx, ny, te.converts_terrain))
                            # Reset counter after successful spread
                            self._exposure_ticks[key] = 0

        # Prune exposure entries that are no longer near an effect source
        stale_keys = [k for k in self._exposure_ticks if k not in active_spread_keys]
        for k in stale_keys:
            del self._exposure_ticks[k]

        # Apply terrain conversions
        for x, y, terrain_str in tiles_to_convert:
            tile = world.get_tile(x, y)
            if tile:
                try:
                    tile.terrain_type = TerrainType(terrain_str)
                except ValueError:
                    pass  # Unknown terrain string, skip silently

        # Spawn new tile-effect objects
        for type_id, x, y in objects_to_spawn:
            defn = ObjectRegistry.get(type_id)
            if defn is not None:
                new_obj = ObjectRegistry.create(type_id, x, y)
                world.add_object(new_obj)

        # Apply reclamation: remove effect objects and convert terrain back
        for obj_id in objects_to_remove:
            world.remove_object(obj_id)

        for x, y, terrain_str, fertility_restore in tiles_to_reclaim:
            tile = world.get_tile(x, y)
            if tile:
                try:
                    tile.terrain_type = TerrainType(terrain_str)
                    # Give the reclaimed tile a small fertility boost
                    tile.fertility = max(tile.fertility, fertility_restore)
                except ValueError:
                    pass

    @staticmethod
    def _tile_has_blocker(world: 'World', x: int, y: int, blocked_by: list) -> bool:
        """
        Check whether any object on the tile has a category in blocked_by.

        Args:
            world: World instance.
            x: Tile X.
            y: Tile Y.
            blocked_by: List of category strings that block spreading.

        Returns:
            True if a blocker is present.
        """
        if not blocked_by:
            return False
        tile = world.get_tile(x, y)
        if not tile:
            return False
        for obj_id in tile.object_ids:
            obj = world.objects.get(obj_id)
            if obj is None:
                continue
            cat = ObjectRegistry.get_category(obj)
            if cat in blocked_by:
                return True
        return False