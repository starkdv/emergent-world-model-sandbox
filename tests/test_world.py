"""
Unit tests for the World system.

Tests tile creation, world generation, and object management.

Author: Karan Vasa
"""

import pytest
from world.tiles import Tile, TerrainType
from world.objects import WorldObject, EdibleComponent, SeedComponent, PlantComponent
from world.world import World


class TestTile:
    """Tests for Tile class."""

    def test_tile_creation(self):
        """Test basic tile creation."""
        tile = Tile(5, 10, TerrainType.SOIL, fertility=0.8, moisture=0.6)
        assert tile.x == 5
        assert tile.y == 10
        assert tile.terrain_type == TerrainType.SOIL
        assert tile.fertility == 0.8
        assert tile.moisture == 0.6
        assert len(tile.object_ids) == 0

    def test_tile_fertility_validation(self):
        """Test that fertility is validated."""
        with pytest.raises(ValueError):
            Tile(0, 0, TerrainType.SOIL, fertility=1.5)

        with pytest.raises(ValueError):
            Tile(0, 0, TerrainType.SOIL, fertility=-0.1)

    def test_tile_passable(self):
        """Test passability based on terrain type."""
        soil_tile = Tile(0, 0, TerrainType.SOIL)
        rock_tile = Tile(0, 0, TerrainType.ROCK)
        water_tile = Tile(0, 0, TerrainType.WATER)

        assert soil_tile.is_passable()
        assert not rock_tile.is_passable()
        assert (
            not water_tile.is_passable()
        )  # water blocks movement (agents stay on land)
        sand_tile = Tile(0, 0, TerrainType.SAND)
        assert sand_tile.is_passable()

    def test_tile_plantable(self):
        """Test plantability based on terrain type."""
        soil_tile = Tile(0, 0, TerrainType.SOIL)
        rock_tile = Tile(0, 0, TerrainType.ROCK)
        water_tile = Tile(0, 0, TerrainType.WATER)

        assert soil_tile.is_plantable()
        assert not rock_tile.is_plantable()
        assert not water_tile.is_plantable()

    def test_tile_object_management(self):
        """Test adding and removing objects from tiles."""
        tile = Tile(0, 0, TerrainType.SOIL)

        tile.add_object(1)
        tile.add_object(2)
        assert 1 in tile.object_ids
        assert 2 in tile.object_ids
        assert len(tile.object_ids) == 2

        # Adding same object twice should not duplicate
        tile.add_object(1)
        assert len(tile.object_ids) == 2

        # Remove objects
        assert tile.remove_object(1)
        assert 1 not in tile.object_ids
        assert not tile.remove_object(999)  # Non-existent


class TestWorldObject:
    """Tests for WorldObject and components."""

    def test_object_creation(self):
        """Test basic object creation."""
        obj = WorldObject(5, 10)
        assert obj.x == 5
        assert obj.y == 10
        assert len(obj.components) == 0

    def test_add_components(self):
        """Test adding components to objects."""
        obj = WorldObject(0, 0)

        edible = EdibleComponent(calories=20.0)
        obj.add_component(edible)

        assert obj.has_component(EdibleComponent)
        retrieved = obj.get_component(EdibleComponent)
        assert retrieved is edible
        assert retrieved.calories == 20.0

    def test_multiple_components(self):
        """Test object with multiple components."""
        obj = WorldObject(0, 0)

        obj.add_component(EdibleComponent(calories=10.0))
        obj.add_component(SeedComponent(plant_type="berry"))

        assert obj.has_component(EdibleComponent)
        assert obj.has_component(SeedComponent)
        assert not obj.has_component(PlantComponent)

    def test_remove_component(self):
        """Test removing components."""
        obj = WorldObject(0, 0)
        obj.add_component(EdibleComponent(calories=10.0))

        assert obj.has_component(EdibleComponent)
        assert obj.remove_component(EdibleComponent)
        assert not obj.has_component(EdibleComponent)
        assert not obj.remove_component(EdibleComponent)  # Already removed


class TestWorld:
    """Tests for World class."""

    def test_world_creation(self):
        """Test basic world creation."""
        world = World(width=10, height=10, seed=42)
        assert world.width == 10
        assert world.height == 10
        assert world.tick == 0
        assert world.seed == 42
        assert len(world.tiles) == 10
        assert len(world.tiles[0]) == 10

    def test_world_invalid_dimensions(self):
        """Test that invalid dimensions raise errors."""
        with pytest.raises(ValueError):
            World(width=0, height=10)

        with pytest.raises(ValueError):
            World(width=10, height=-5)

    def test_get_tile(self):
        """Test getting tiles by coordinates."""
        world = World(width=5, height=5, seed=42)

        tile = world.get_tile(2, 3)
        assert tile is not None
        assert tile.x == 2
        assert tile.y == 3

        # Out of bounds
        assert world.get_tile(-1, 0) is None
        assert world.get_tile(0, 10) is None
        assert world.get_tile(10, 0) is None

    def test_is_valid_position(self):
        """Test position validation."""
        world = World(width=10, height=10)

        assert world.is_valid_position(0, 0)
        assert world.is_valid_position(9, 9)
        assert world.is_valid_position(5, 5)

        assert not world.is_valid_position(-1, 0)
        assert not world.is_valid_position(0, -1)
        assert not world.is_valid_position(10, 0)
        assert not world.is_valid_position(0, 10)

    def test_add_object(self):
        """Test adding objects to world."""
        world = World(width=10, height=10)
        obj = WorldObject(5, 5)

        assert world.add_object(obj)
        assert obj.id in world.objects

        tile = world.get_tile(5, 5)
        assert obj.id in tile.object_ids

    def test_add_object_invalid_position(self):
        """Test that objects at invalid positions are rejected."""
        world = World(width=10, height=10)
        obj = WorldObject(20, 20)  # Out of bounds

        assert not world.add_object(obj)
        assert obj.id not in world.objects

    def test_remove_object(self):
        """Test removing objects from world."""
        world = World(width=10, height=10)
        obj = WorldObject(5, 5)
        world.add_object(obj)

        assert world.remove_object(obj.id)
        assert obj.id not in world.objects

        tile = world.get_tile(5, 5)
        assert obj.id not in tile.object_ids

    def test_move_object(self):
        """Test moving objects."""
        world = World(width=10, height=10)
        obj = WorldObject(2, 2)
        world.add_object(obj)

        assert world.move_object(obj.id, 5, 5)
        assert obj.x == 5
        assert obj.y == 5

        # Check old tile
        old_tile = world.get_tile(2, 2)
        assert obj.id not in old_tile.object_ids

        # Check new tile
        new_tile = world.get_tile(5, 5)
        assert obj.id in new_tile.object_ids

    def test_get_objects_at(self):
        """Test getting objects at a position."""
        # Enable stacking mode for this test
        world = World(width=10, height=10, allow_stacking=True)

        obj1 = WorldObject(3, 3)
        obj2 = WorldObject(3, 3)
        obj3 = WorldObject(5, 5)

        world.add_object(obj1)
        world.add_object(obj2)
        world.add_object(obj3)

        objects_at_3_3 = world.get_objects_at(3, 3)
        assert len(objects_at_3_3) == 2
        assert obj1 in objects_at_3_3
        assert obj2 in objects_at_3_3

        objects_at_5_5 = world.get_objects_at(5, 5)
        assert len(objects_at_5_5) == 1
        assert obj3 in objects_at_5_5

    def test_get_neighbors(self):
        """Test getting neighboring tiles."""
        world = World(width=10, height=10)

        # Center tile (5, 5) should have 8 neighbors
        neighbors = world.get_neighbors(5, 5, radius=1)
        assert len(neighbors) == 8

        # Corner tile (0, 0) should have 3 neighbors
        neighbors = world.get_neighbors(0, 0, radius=1)
        assert len(neighbors) == 3

        # Edge tile should have 5 neighbors
        neighbors = world.get_neighbors(0, 5, radius=1)
        assert len(neighbors) == 5

    def test_world_update(self):
        """Test world tick update."""
        world = World(width=10, height=10)
        assert world.tick == 0

        world.update()
        assert world.tick == 1

        world.update()
        assert world.tick == 2

    def test_terrain_generation_reproducible(self):
        """Test that same seed produces same terrain."""
        world1 = World(width=10, height=10, seed=42)
        world2 = World(width=10, height=10, seed=42)

        for y in range(10):
            for x in range(10):
                tile1 = world1.get_tile(x, y)
                tile2 = world2.get_tile(x, y)
                assert tile1.terrain_type == tile2.terrain_type
                assert abs(tile1.fertility - tile2.fertility) < 0.0001
                assert abs(tile1.moisture - tile2.moisture) < 0.0001
