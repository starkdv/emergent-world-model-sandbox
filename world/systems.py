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
            plant.age += 1
            
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
                # Probability check for germination success
                if random.random() < self.germination_success_rate:
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
            
            # Create plant
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
        seeds_to_spawn = []
        decomposed_positions = []
        
        for obj_id, obj in world.objects.items():
            if not obj.has_component(EdibleComponent):
                continue
            
            edible = obj.get_component(EdibleComponent)
            
            # Reduce freshness
            edible.freshness -= self.decay_rate
            
            # Handle completely spoiled items
            if edible.freshness <= 0.0:
                objects_to_remove.append(obj_id)
                decomposed_positions.append((obj.x, obj.y))
                
                # Chance to drop seed when decomposing
                if random.random() < self.seed_drop_chance:
                    seeds_to_spawn.append((obj.x, obj.y))
        
        # Remove spoiled objects
        for obj_id in objects_to_remove:
            world.remove_object(obj_id)
        
        # Return nutrients to soil from decomposed berries
        if self.soil_system:
            for x, y in decomposed_positions:
                self.soil_system.return_nutrients_to_soil(world, x, y)
        
        # Spawn seeds from decomposed fruit
        for x, y in seeds_to_spawn:
            if world.is_valid_position(x, y):
                # Create seed at the decomposition location
                seed = WorldObject(x, y)
                seed.add_component(SeedComponent(
                    plant_type="berry_plant",
                    grow_time=50,
                    required_fertility=0.3,
                    required_moisture=0.2,
                    max_age=self.seed_max_age
                ))
                world.add_object(seed)


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
        
        Args:
            world: World instance to update
            
        Author: Karan Vasa
        """
        # Track which tiles have plants
        occupied_tiles: Set[tuple] = set()
        
        # First pass: identify tiles with plants and apply consumption
        for obj_id, obj in world.objects.items():
            if obj.has_component(PlantComponent):
                occupied_tiles.add((obj.x, obj.y))
                tile = world.get_tile(obj.x, obj.y)
                
                if tile and tile.is_plantable():
                    # Plants consume soil resources
                    tile.fertility = max(0.0, tile.fertility - self.fertility_consumption)
                    tile.moisture = max(0.0, tile.moisture - self.moisture_consumption)
        
        # Second pass: process all soil tiles
        for y in range(world.height):
            for x in range(world.width):
                tile = world.get_tile(x, y)
                
                if not tile or not tile.is_plantable():
                    continue
                
                # Natural moisture dynamics
                tile.moisture = max(0.0, tile.moisture - self.moisture_evaporation_rate)  # Evaporation
                tile.moisture = min(1.0, tile.moisture + self.moisture_recovery_rate)  # Rain/groundwater
                
                # Empty soil recovers fertility naturally
                if (x, y) not in occupied_tiles:
                    tile.fertility = min(1.0, tile.fertility + self.fertility_recovery_rate)
    
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
            
            # Chance to spawn resource
            if random.random() < plant.spawn_rate:
                self._spawn_resource_near(world, obj.x, obj.y, plant.spawn_resource_type)
        
        # Safety net: spawn resources if world is depleted
        edible_count = sum(
            1 for obj in world.objects.values()
            if obj.has_component(EdibleComponent)
        )
        
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
            
            # Spawn resource based on type
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
        self.resource_spawn = ResourceSpawnSystem(
            berry_calories,
            safety_spawn_rate,
            min_resources,
            seed_max_age
        )
    
    def update(self, world: 'World') -> None:
        """
        Run all systems in order.
        
        Execution order:
        1. Plant growth (aging)
        2. Seed germination (creates new plants)
        3. Decay (removes spoiled items)
        4. Fertilizer effects (boosts soil)
        5. Soil dynamics (manages soil resources)
        6. Resource spawning (creates new resources)
        
        Args:
            world: World instance to update
            
        Author: Karan Vasa
        """
        self.plant_growth.update(world)
        self.seed_germination.update(world)
        self.decay.update(world)
        self.fertilizer.update(world)
        self.soil_dynamics.update(world)
        self.resource_spawn.update(world)
