"""
Unit tests for world systems.

Tests for plant growth, seed germination, decay, fertilizer effects,
and resource spawning systems.

Author: Karan Vasa
"""

import pytest
from world.world import World
from world.objects import (
    WorldObject,
    PlantComponent,
    SeedComponent,
    EdibleComponent,
    FertilizerComponent,
)
from world.systems import (
    PlantGrowthSystem,
    SeedGerminationSystem,
    DecaySystem,
    FertilizerSystem,
    ResourceSpawnSystem,
    WorldSystemManager,
    _count_plants_in_radius,
)


class TestPlantGrowthSystem:
    """Tests for plant growth system."""

    def test_plant_ages_each_tick(self):
        """Test that plants age by 1 each tick."""
        world = World(10, 10, seed=42)
        system = PlantGrowthSystem()

        # Add a plant
        plant = WorldObject(5, 5)
        plant.add_component(PlantComponent(mature_age=10, max_age=100))
        world.add_object(plant)

        # Age should be 0 initially
        plant_comp = plant.get_component(PlantComponent)
        assert plant_comp.age == 0

        # Update system
        system.update(world)
        assert plant_comp.age == 1

        # Update again
        system.update(world)
        assert plant_comp.age == 2

    def test_old_plants_die(self):
        """Test that plants are removed when they reach max age."""
        world = World(10, 10, seed=42)
        system = PlantGrowthSystem()

        # Add a plant that's almost dead
        plant = WorldObject(5, 5)
        plant_comp = PlantComponent(mature_age=10, max_age=50)
        plant_comp.age = 49  # One tick from death
        plant.add_component(plant_comp)
        world.add_object(plant)

        plant_id = plant.id
        assert plant_id in world.objects

        # Update - plant should die
        system.update(world)
        assert plant_id not in world.objects

    def test_multiple_plants(self):
        """Test system handles multiple plants correctly."""
        world = World(10, 10, seed=42)
        system = PlantGrowthSystem()

        # Add three plants at different ages
        plants = []
        for i in range(3):
            plant = WorldObject(i, i)
            plant_comp = PlantComponent(mature_age=10, max_age=100)
            plant_comp.age = i * 10
            plant.add_component(plant_comp)
            world.add_object(plant)
            plants.append(plant)

        # Update
        system.update(world)

        # All should age by 1
        assert plants[0].get_component(PlantComponent).age == 1
        assert plants[1].get_component(PlantComponent).age == 11
        assert plants[2].get_component(PlantComponent).age == 21


class TestSeedGerminationSystem:
    """Tests for seed germination system."""

    def test_seed_germinates_on_suitable_soil(self):
        """Test that seeds germinate when conditions are met."""
        world = World(10, 10, seed=42)
        # Use 100% germination rate for deterministic testing
        system = SeedGerminationSystem(
            plant_mature_age=100, plant_max_age=500, germination_success_rate=1.0
        )

        # Find a soil tile
        soil_x, soil_y = None, None
        for y in range(world.height):
            for x in range(world.width):
                tile = world.get_tile(x, y)
                if (
                    tile.is_plantable()
                    and tile.fertility >= 0.3
                    and tile.moisture >= 0.2
                ):
                    soil_x, soil_y = x, y
                    break
            if soil_x is not None:
                break

        assert soil_x is not None, "No suitable soil found"

        # Add seed
        seed = WorldObject(soil_x, soil_y)
        seed_comp = SeedComponent(plant_type="test_plant", grow_time=5)
        seed_comp.time_in_soil = 4  # Almost ready
        seed.add_component(seed_comp)
        world.add_object(seed)

        seed_id = seed.id
        initial_count = len(world.objects)

        # Update - seed should germinate
        system.update(world)

        # Seed should be gone
        assert seed_id not in world.objects

        # Plant should exist
        objects_at_pos = world.get_objects_at(soil_x, soil_y)
        assert len(objects_at_pos) == 1
        assert objects_at_pos[0].has_component(PlantComponent)

    def test_seed_needs_time_in_soil(self):
        """Test that seeds don't germinate immediately."""
        world = World(10, 10, seed=42)
        system = SeedGerminationSystem()

        # Find suitable soil
        soil_x, soil_y = 5, 5
        tile = world.get_tile(soil_x, soil_y)
        tile.fertility = 0.8
        tile.moisture = 0.6

        # Add new seed
        seed = WorldObject(soil_x, soil_y)
        seed.add_component(SeedComponent(plant_type="test", grow_time=50))
        world.add_object(seed)

        seed_id = seed.id

        # Update once - shouldn't germinate yet
        system.update(world)
        assert seed_id in world.objects
        assert seed.get_component(SeedComponent).time_in_soil == 1


class TestGerminationCarryingCapacity:
    """
    Tests for the plant carrying-capacity fix (runaway accumulation bug).

    A seed must not germinate where the surrounding neighbourhood already
    holds ``max_neighbor_plants`` plants — competition for space/light.
    """

    def _prep_soil(self, world):
        """Force the whole world to fertile, plantable soil."""
        from world.tiles import TerrainType

        for y in range(world.height):
            for x in range(world.width):
                tile = world.get_tile(x, y)
                tile.terrain_type = TerrainType.SOIL
                tile.fertility = 0.9
                tile.moisture = 0.7

    def _add_plant(self, world, x, y):
        plant = WorldObject(x, y)
        plant.add_component(
            PlantComponent(
                mature_age=100,
                max_age=500,
                spawn_resource_type="berry",
                spawn_rate=0.1,
            )
        )
        world.add_object(plant)
        return plant

    def _add_ready_seed(self, world, x, y):
        seed = WorldObject(x, y)
        comp = SeedComponent(plant_type="berry_plant", grow_time=1)
        comp.time_in_soil = 5  # ready
        seed.add_component(comp)
        world.add_object(seed)
        return seed

    def test_count_plants_in_radius(self):
        world = World(10, 10, seed=1)
        self._prep_soil(world)
        self._add_plant(world, 5, 5)
        self._add_plant(world, 6, 5)
        self._add_plant(world, 8, 8)  # outside radius-1 of (5,5)

        assert _count_plants_in_radius(world, 5, 5, radius=1) == 2
        assert _count_plants_in_radius(world, 5, 5, radius=0) == 1
        assert _count_plants_in_radius(world, 5, 5, radius=2) == 2
        assert _count_plants_in_radius(world, 0, 0, radius=1) == 0

    def test_seed_blocked_when_neighborhood_full(self):
        """A ready seed should NOT germinate in a saturated neighbourhood."""
        world = World(10, 10, seed=1)
        self._prep_soil(world)
        # Cap of 2 within a 3x3 window; place 2 plants near the seed tile
        self._add_plant(world, 4, 5)
        self._add_plant(world, 6, 5)
        seed = self._add_ready_seed(world, 5, 5)
        seed_id = seed.id

        system = SeedGerminationSystem(
            germination_success_rate=1.0, max_neighbor_plants=2, neighbor_radius=1
        )
        system.update(world)

        # Seed survives (waits) and did not become a plant
        assert seed_id in world.objects
        assert not world.get_objects_at(5, 5)[0].has_component(PlantComponent) or (
            world.get_objects_at(5, 5)[0].id == seed_id
        )

    def test_seed_germinates_when_neighborhood_has_room(self):
        """With one fewer neighbour than the cap, germination proceeds."""
        world = World(10, 10, seed=1)
        self._prep_soil(world)
        self._add_plant(world, 4, 5)  # only 1 neighbour, cap is 2
        seed = self._add_ready_seed(world, 5, 5)
        seed_id = seed.id

        system = SeedGerminationSystem(
            germination_success_rate=1.0, max_neighbor_plants=2, neighbor_radius=1
        )
        system.update(world)

        assert seed_id not in world.objects  # seed consumed → plant
        objs = world.get_objects_at(5, 5)
        assert any(o.has_component(PlantComponent) for o in objs)

    def test_cap_zero_disables_carrying_capacity(self):
        """max_neighbor_plants=0 restores legacy unbounded behaviour."""
        world = World(10, 10, seed=1)
        self._prep_soil(world)
        # Crowd the neighbourhood heavily
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1)]:
            self._add_plant(world, 5 + dx, 5 + dy)
        seed = self._add_ready_seed(world, 5, 5)
        seed_id = seed.id

        system = SeedGerminationSystem(
            germination_success_rate=1.0, max_neighbor_plants=0, neighbor_radius=2
        )
        system.update(world)

        # With the cap disabled, the seed germinates despite the crowd
        assert seed_id not in world.objects
        objs = world.get_objects_at(5, 5)
        assert any(o.has_component(PlantComponent) for o in objs)

    def test_population_stays_bounded_over_long_run(self):
        """
        Regression for the accumulation bug: with the carrying cap on, the
        plant population must NOT keep climbing toward world saturation.
        """
        world = World(24, 24, seed=3, parallel=False)
        self._prep_soil(world)
        plantable = world.width * world.height
        # Seed a few starter plants
        for i in range(6):
            self._add_plant(world, 2 + i * 3, 2)

        peak = 0
        for t in range(4000):
            world.update()
            if (t + 1) % 1000 == 0:
                plants = sum(
                    1 for o in world.objects.values() if o.has_component(PlantComponent)
                )
                peak = max(peak, plants)
        # Must stay well below tiling the world (legacy reaches ~65%)
        assert peak < 0.45 * plantable, f"plants peaked at {peak}/{plantable}"


class TestDecaySystem:
    """Tests for decay system."""

    def test_freshness_decreases(self):
        """Test that edible objects lose freshness over time."""
        world = World(10, 10, seed=42)
        system = DecaySystem(decay_rate=0.1)

        # Add berry
        berry = WorldObject(5, 5)
        berry.add_component(EdibleComponent(calories=20.0, freshness=1.0))
        world.add_object(berry)

        edible = berry.get_component(EdibleComponent)
        assert edible.freshness == 1.0

        # Update
        system.update(world)
        assert edible.freshness == pytest.approx(0.9)

        # Update again
        system.update(world)
        assert edible.freshness == pytest.approx(0.8)

    def test_spoiled_objects_removed(self):
        """Test that completely spoiled objects are removed."""
        world = World(10, 10, seed=42)
        system = DecaySystem(decay_rate=0.1, seed_drop_chance=0.0)  # No seed dropping

        # Add berry with low freshness
        berry = WorldObject(5, 5)
        berry.add_component(EdibleComponent(calories=20.0, freshness=0.05))
        world.add_object(berry)

        berry_id = berry.id
        assert berry_id in world.objects

        # Update - berry should spoil
        system.update(world)
        assert berry_id not in world.objects

    def test_spoiled_berries_drop_seeds(self):
        """Test that decomposed berries have a chance to drop seeds."""
        world = World(10, 10, seed=42)
        system = DecaySystem(decay_rate=0.1, seed_drop_chance=1.0)  # Always drop seeds

        # Add berry with low freshness at position (5, 5)
        berry = WorldObject(5, 5)
        berry.add_component(EdibleComponent(calories=20.0, freshness=0.05))
        world.add_object(berry)

        berry_id = berry.id
        initial_object_count = len(world.objects)

        # Count initial seeds
        initial_seeds = sum(
            1 for obj in world.objects.values() if obj.has_component(SeedComponent)
        )

        # Update - berry should spoil and drop a seed
        system.update(world)

        # Berry should be removed
        assert berry_id not in world.objects

        # A seed should have been spawned
        final_seeds = sum(
            1 for obj in world.objects.values() if obj.has_component(SeedComponent)
        )
        assert final_seeds == initial_seeds + 1

        # Find the newly spawned seed
        seed = None
        for obj in world.objects.values():
            if obj.has_component(SeedComponent) and obj.x == 5 and obj.y == 5:
                seed = obj
                break

        assert seed is not None, "Seed should be spawned at berry location"

        # Verify seed has correct properties
        seed_comp = seed.get_component(SeedComponent)
        assert seed_comp.plant_type == "berry_plant"
        assert seed_comp.grow_time == 50
        assert seed_comp.required_fertility == 0.3
        assert seed_comp.required_moisture == 0.2


class TestFertilizerSystem:
    """Tests for fertilizer system."""

    def test_fertilizer_boosts_nearby_tiles(self):
        """Test that fertilizer increases fertility of nearby soil."""
        world = World(10, 10, seed=42)
        system = FertilizerSystem()

        # Get a soil tile and record its fertility
        center_x, center_y = 5, 5
        center_tile = world.get_tile(center_x, center_y)
        initial_fertility = center_tile.fertility

        # Add fertilizer
        fert = WorldObject(center_x, center_y)
        fert.add_component(
            FertilizerComponent(fertility_boost=0.5, duration=10, radius=1)
        )
        world.add_object(fert)

        # Update several times
        for _ in range(5):
            system.update(world)

        # Fertility should have increased (gradually)
        assert center_tile.fertility > initial_fertility

    def test_fertilizer_expires(self):
        """Test that fertilizer is removed after duration expires."""
        world = World(10, 10, seed=42)
        system = FertilizerSystem()

        # Add fertilizer with short duration
        fert = WorldObject(5, 5)
        fert.add_component(FertilizerComponent(duration=2))
        world.add_object(fert)

        fert_id = fert.id

        # Update once - should still exist
        system.update(world)
        assert fert_id in world.objects

        # Update again - should be removed
        system.update(world)
        assert fert_id not in world.objects


class TestResourceSpawnSystem:
    """Tests for resource spawning system."""

    def test_mature_plants_spawn_berries(self):
        """Test that mature plants can spawn berries."""
        world = World(10, 10, seed=42)
        system = ResourceSpawnSystem(berry_calories=20.0)

        # Add mature plant
        plant = WorldObject(5, 5)
        plant_comp = PlantComponent(
            mature_age=10,
            max_age=500,
            spawn_resource_type="berry",
            spawn_rate=1.0,  # 100% spawn rate for testing
        )
        plant_comp.age = 100  # Mature
        plant.add_component(plant_comp)
        world.add_object(plant)

        initial_count = len(world.objects)

        # Update - should spawn berry
        system.update(world)

        # Should have more objects
        assert len(world.objects) >= initial_count

        # Check for berry nearby
        has_berry = False
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                objects = world.get_objects_at(5 + dx, 5 + dy)
                for obj in objects:
                    if obj.has_component(EdibleComponent):
                        has_berry = True
                        break

        assert has_berry

    def test_safety_spawn_when_depleted(self):
        """Test that safety net spawns resources when world is depleted."""
        world = World(10, 10, seed=42)
        system = ResourceSpawnSystem(
            safety_spawn_rate=1.0, min_resources=5  # Always spawn for testing
        )

        # Remove all edible objects
        to_remove = [
            obj_id
            for obj_id, obj in world.objects.items()
            if obj.has_component(EdibleComponent)
        ]
        for obj_id in to_remove:
            world.remove_object(obj_id)

        initial_count = len(world.objects)

        # Update - should trigger safety spawn
        system.update(world)

        # Should have spawned something
        assert len(world.objects) > initial_count


class TestWorldSystemManager:
    """Tests for integrated system manager."""

    def test_world_update_runs_all_systems(self):
        """Test that world.update() runs all systems."""
        world = World(10, 10, seed=42)
        # Add various objects
        plant = WorldObject(5, 5)
        plant.add_component(PlantComponent())
        world.add_object(plant)

        berry = WorldObject(6, 6)
        berry.add_component(EdibleComponent(calories=20.0, freshness=1.0))
        world.add_object(berry)

        initial_tick = world.tick
        initial_plant_age = plant.get_component(PlantComponent).age
        initial_freshness = berry.get_component(EdibleComponent).freshness

        # Update world
        world.update()

        # Tick should increment
        assert world.tick == initial_tick + 1

        # Plant should age
        assert plant.get_component(PlantComponent).age == initial_plant_age + 1

        # Berry should decay
        assert berry.get_component(EdibleComponent).freshness < initial_freshness

    def test_systems_run_in_order(self):
        """Test that systems execute in correct order."""
        world = World(10, 10, seed=42)

        # Add a seed ready to germinate
        # Note: With probabilistic germination (75% success rate),
        # we need to test with 100% rate or add multiple seeds
        soil_x, soil_y = 5, 5
        tile = world.get_tile(soil_x, soil_y)
        tile.fertility = 0.8
        tile.moisture = 0.6

        # Add multiple seeds to ensure at least one germinates
        seed_ids = []
        for i in range(5):
            seed = WorldObject(soil_x, soil_y + i if i > 0 else soil_y)
            seed_comp = SeedComponent(plant_type="berry_plant", grow_time=1)
            seed_comp.time_in_soil = 0
            seed.add_component(seed_comp)
            world.add_object(seed)
            seed_ids.append(seed.id)

        # Update world once
        world.update()

        # With 5 seeds at 75% success rate, at least one should germinate (99.9% probability)
        # Check that at least one seed germinated into a plant
        plant_found = False
        for y_offset in range(5):
            objects_at_pos = world.get_objects_at(soil_x, soil_y + y_offset)
            if any(obj.has_component(PlantComponent) for obj in objects_at_pos):
                plant_found = True
                break

        assert plant_found, "At least one seed should have germinated into a plant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
