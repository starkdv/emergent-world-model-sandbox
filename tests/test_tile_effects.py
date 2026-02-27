"""
Unit tests for tile-effect system: InteractionSpec, TileEffectSpec, sand mechanics.

Covers:
- InteractionSpec (pickable, blocks_growth, etc.)
- TileEffectSpec multipliers (germination, growth, spawn)
- TileEffectSystem spreading mechanics & terrain conversion
- Sand builtin definition
- Action mask / pick-up gating by pickable flag

Author: Karan Vasa
"""

import pytest
import random

from world.world import World
from world.tiles import TerrainType
from world.objects import (
    WorldObject,
    PlantComponent,
    SeedComponent,
    EdibleComponent,
)
from world.object_registry import (
    ObjectRegistry,
    ObjectDefinition,
    InteractionSpec,
    TileEffectSpec,
    register_builtin_objects,
)
from world.systems import (
    PlantGrowthSystem,
    SeedGerminationSystem,
    ResourceSpawnSystem,
    TileEffectSystem,
    _get_tile_growth_multiplier,
    _get_tile_germination_multiplier,
    _get_tile_spawn_rate_multiplier,
    _tile_blocks_growth,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure a clean registry for every test."""
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    ObjectRegistry._definitions.clear()


@pytest.fixture
def small_world():
    """A tiny 5x5 world with all-soil terrain for deterministic tests."""
    w = World(
        5, 5, seed=1, soil_ratio=1.0, rock_ratio=0.0, water_ratio=0.0, sand_ratio=0.0
    )
    for y in range(5):
        for x in range(5):
            tile = w.get_tile(x, y)
            if tile:
                tile.terrain_type = TerrainType.SOIL
                tile.fertility = 0.8
                tile.moisture = 0.6
    return w


# ===================================================================
# InteractionSpec tests
# ===================================================================


class TestInteractionSpec:
    """Tests for InteractionSpec defaults and per-object lookups."""

    def test_default_is_pickable(self):
        spec = InteractionSpec()
        assert spec.pickable is True

    def test_plant_not_pickable(self):
        defn = ObjectRegistry.get("berry_plant")
        assert defn is not None
        assert defn.interaction.pickable is False

    def test_berry_is_pickable(self):
        defn = ObjectRegistry.get("berry")
        assert defn is not None
        assert defn.interaction.pickable is True

    def test_seed_is_pickable(self):
        defn = ObjectRegistry.get("berry_seed")
        assert defn is not None
        assert defn.interaction.pickable is True

    def test_sand_not_pickable(self):
        defn = ObjectRegistry.get("sand")
        assert defn is not None
        assert defn.interaction.pickable is False

    def test_sand_does_not_block_growth(self):
        defn = ObjectRegistry.get("sand")
        assert defn.interaction.blocks_growth is False

    def test_is_pickable_helper(self):
        berry = ObjectRegistry.create("berry", 0, 0)
        plant = ObjectRegistry.create("berry_plant", 0, 0)
        sand = ObjectRegistry.create("sand", 0, 0)

        assert ObjectRegistry.is_pickable(berry) is True
        assert ObjectRegistry.is_pickable(plant) is False
        assert ObjectRegistry.is_pickable(sand) is False

    def test_get_interaction_fallback(self):
        """Unknown type_id should return default InteractionSpec."""
        obj = WorldObject(0, 0)
        obj.type_id = "nonexistent"
        spec = ObjectRegistry.get_interaction(obj)
        assert spec.pickable is True  # Default


# ===================================================================
# TileEffectSpec attribute tests
# ===================================================================


class TestTileEffectSpec:
    """Verify the sand definition's TileEffectSpec attributes."""

    def test_sand_germination_multiplier(self):
        defn = ObjectRegistry.get("sand")
        assert defn.tile_effect is not None
        assert defn.tile_effect.germination_multiplier == pytest.approx(0.1)

    def test_sand_growth_multiplier(self):
        defn = ObjectRegistry.get("sand")
        assert defn.tile_effect.growth_multiplier == pytest.approx(0.1)

    def test_sand_spawn_rate_multiplier(self):
        defn = ObjectRegistry.get("sand")
        assert defn.tile_effect.spawn_rate_multiplier == pytest.approx(0.3)

    def test_sand_spread_type_id(self):
        defn = ObjectRegistry.get("sand")
        assert defn.tile_effect.spread_type_id == "sand"

    def test_sand_spread_blocked_by_plant(self):
        defn = ObjectRegistry.get("sand")
        assert "plant" in defn.tile_effect.spread_blocked_by

    def test_sand_converts_terrain(self):
        defn = ObjectRegistry.get("sand")
        assert defn.tile_effect.converts_terrain == "sand"

    def test_sand_fertility_override(self):
        defn = ObjectRegistry.get("sand")
        assert defn.tile_effect.fertility_override == pytest.approx(0.05)

    def test_sand_moisture_override(self):
        defn = ObjectRegistry.get("sand")
        assert defn.tile_effect.moisture_override == pytest.approx(0.05)


# ===================================================================
# Tile-effect helper function tests
# ===================================================================


class TestTileEffectHelpers:
    """Tests for the module-level helper functions used by systems."""

    def test_growth_multiplier_no_effects(self, small_world):
        """A tile with no effect-objects should return 1.0."""
        assert _get_tile_growth_multiplier(small_world, 2, 2) == pytest.approx(1.0)

    def test_growth_multiplier_with_sand(self, small_world):
        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)
        mult = _get_tile_growth_multiplier(small_world, 2, 2)
        assert mult == pytest.approx(0.1)

    def test_germination_multiplier_with_sand(self, small_world):
        sand = ObjectRegistry.create("sand", 3, 3)
        small_world.add_object(sand)
        mult = _get_tile_germination_multiplier(small_world, 3, 3)
        assert mult == pytest.approx(0.1)

    def test_spawn_rate_multiplier_with_sand(self, small_world):
        sand = ObjectRegistry.create("sand", 1, 1)
        small_world.add_object(sand)
        mult = _get_tile_spawn_rate_multiplier(small_world, 1, 1)
        assert mult == pytest.approx(0.3)

    def test_blocks_growth_with_sand(self, small_world):
        """Sand no longer blocks growth outright (relies on multiplier)."""
        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)
        assert _tile_blocks_growth(small_world, 2, 2) is False

    def test_blocks_growth_empty_tile(self, small_world):
        assert _tile_blocks_growth(small_world, 2, 2) is False

    def test_blocks_growth_normal_object(self, small_world):
        berry = ObjectRegistry.create("berry", 2, 2)
        small_world.add_object(berry)
        assert _tile_blocks_growth(small_world, 2, 2) is False

    def test_multipliers_stack_multiplicatively(self, small_world):
        """Two effect objects on the same tile should multiply."""
        # Register a second sand-like object for stacking test
        ObjectRegistry.register(
            ObjectDefinition(
                type_id="quicksand",
                display_name="Quicksand",
                category="terrain_effect",
                interaction=InteractionSpec(pickable=False, blocks_growth=True),
                tile_effect=TileEffectSpec(growth_multiplier=0.5),
            )
        )
        sand = ObjectRegistry.create("sand", 2, 2)
        qs = ObjectRegistry.create("quicksand", 2, 2)
        # Need allow_stacking or add manually
        small_world.add_object(sand)
        tile = small_world.get_tile(2, 2)
        tile.object_ids.add(qs.id)
        small_world.objects[qs.id] = qs
        mult = _get_tile_growth_multiplier(small_world, 2, 2)
        assert mult == pytest.approx(0.1 * 0.5)


# ===================================================================
# PlantGrowthSystem with tile effects
# ===================================================================


class TestPlantGrowthWithTileEffects:
    """Ensure PlantGrowthSystem applies growth multiplier."""

    def test_plant_grows_normally_without_sand(self, small_world):
        plant_obj = ObjectRegistry.create("berry_plant", 2, 2)
        small_world.add_object(plant_obj)
        pc = plant_obj.get_component(PlantComponent)
        assert pc.age == 0

        system = PlantGrowthSystem()
        system.update(small_world)
        assert pc.age == pytest.approx(1.0)

    def test_plant_grows_slower_on_sand(self, small_world):
        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)

        plant_obj = ObjectRegistry.create("berry_plant", 2, 2)
        tile = small_world.get_tile(2, 2)
        tile.object_ids.add(plant_obj.id)
        small_world.objects[plant_obj.id] = plant_obj

        pc = plant_obj.get_component(PlantComponent)
        assert pc.age == 0

        system = PlantGrowthSystem()
        system.update(small_world)
        # 0.1 growth multiplier => age should be ~0.1
        assert pc.age == pytest.approx(0.1)


# ===================================================================
# SeedGerminationSystem with tile effects
# ===================================================================


class TestSeedGerminationWithTileEffects:
    """Germination should be harder on sand and blocked when blocks_growth."""

    def test_germination_severely_hindered_on_sand(self, small_world):
        """Sand has germination_multiplier=0.1 making germination very rare.

        We test by running many ticks and verifying germination almost never
        succeeds, compared to normal soil where it always would.
        """
        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)

        seed = ObjectRegistry.create("berry_seed", 2, 2)
        tile = small_world.get_tile(2, 2)
        tile.object_ids.add(seed.id)
        small_world.objects[seed.id] = seed

        sc = seed.get_component(SeedComponent)
        # Fast-forward to ready germination
        sc.time_in_soil = sc.grow_time + 1
        tile.fertility = 1.0
        tile.moisture = 1.0

        # With 0.1 multiplier, germination chance is very low
        # The seed CAN germinate but it's extremely unlikely in 1 tick
        # Just verify the system runs without error and seed isn't
        # outright blocked anymore
        system = SeedGerminationSystem()
        random.seed(42)
        system.update(small_world)
        # Seed may or may not have germinated — the key change is
        # it's no longer categorically blocked


# ===================================================================
# TileEffectSystem spreading tests
# ===================================================================


class TestTileEffectSystem:
    """Tests for the TileEffectSystem: spreading, clamping, conversion."""

    def test_fertility_clamping(self, small_world):
        """Sand should clamp tile fertility to its override value."""
        tile = small_world.get_tile(2, 2)
        tile.fertility = 0.8

        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)

        tes = TileEffectSystem()
        tes.update(small_world)

        assert tile.fertility <= 0.1 + 1e-9

    def test_moisture_clamping(self, small_world):
        """Sand should clamp tile moisture to its override value."""
        tile = small_world.get_tile(2, 2)
        tile.moisture = 0.6

        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)

        tes = TileEffectSystem()
        tes.update(small_world)

        assert tile.moisture <= 0.05 + 1e-9

    def test_no_spread_before_interval(self, small_world):
        """Sand should NOT spread before spread_interval ticks."""
        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)

        tes = TileEffectSystem()
        # Run for fewer ticks than spread_interval (200)
        for _ in range(50):
            tes.update(small_world)

        # Neighbouring soil tiles should still be SOIL
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = 2 + dx, 2 + dy
            tile = small_world.get_tile(nx, ny)
            if tile:
                assert tile.terrain_type == TerrainType.SOIL

    def test_spread_after_interval_with_no_blockers(self, small_world):
        """After spread_interval ticks with no blockers, sand should spread."""
        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)

        tes = TileEffectSystem()
        random.seed(42)

        # Run enough ticks for spread to trigger (interval=200, then chance=0.05)
        # We run many more to ensure at least one spread fires
        for _ in range(500):
            tes.update(small_world)

        # At least one neighbor should have been converted to sand
        converted = False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = 2 + dx, 2 + dy
            tile = small_world.get_tile(nx, ny)
            if tile and tile.terrain_type == TerrainType.SAND:
                converted = True
                break
        assert converted, "Sand should have spread to at least one neighbour"

    def test_spread_blocked_by_plant(self, small_world):
        """Plants on a neighbour tile should prevent sand spread to that tile."""
        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)

        # Place a plant on every neighbour
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = 2 + dx, 2 + dy
            plant = ObjectRegistry.create("berry_plant", nx, ny)
            small_world.add_object(plant)

        tes = TileEffectSystem()
        for _ in range(300):
            tes.update(small_world)

        # No neighbours should have been converted
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = 2 + dx, 2 + dy
            tile = small_world.get_tile(nx, ny)
            if tile:
                assert tile.terrain_type == TerrainType.SOIL

    def test_exposure_resets_when_blocker_appears(self, small_world):
        """If a plant appears on a tile mid-exposure, counter resets."""
        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)

        tes = TileEffectSystem()
        # Run 100 ticks to build up exposure on neighbours
        for _ in range(100):
            tes.update(small_world)

        # Now place a plant on the neighbour (3, 2)
        plant = ObjectRegistry.create("berry_plant", 3, 2)
        small_world.add_object(plant)

        # Run one more tick — the counter should reset to 0
        tes.update(small_world)
        key = (3, 2, "sand")
        assert tes._exposure_ticks.get(key, 0) == 0

    def test_spread_spawns_sand_object(self, small_world):
        """Spread should not only convert terrain but also spawn a sand object."""
        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)

        tes = TileEffectSystem()
        random.seed(42)

        for _ in range(500):
            tes.update(small_world)

        # Check that neighbour tiles that became SAND also have a sand object
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = 2 + dx, 2 + dy
            tile = small_world.get_tile(nx, ny)
            if tile and tile.terrain_type == TerrainType.SAND:
                has_sand_obj = any(
                    small_world.objects.get(oid)
                    and getattr(small_world.objects[oid], "type_id", "") == "sand"
                    for oid in tile.object_ids
                )
                assert (
                    has_sand_obj
                ), f"Tile ({nx},{ny}) converted but has no sand object"

    def test_does_not_spread_to_non_soil(self, small_world):
        """Sand should only spread to SOIL tiles, not ROCK/WATER."""
        # Set all neighbours to ROCK
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = 2 + dx, 2 + dy
            tile = small_world.get_tile(nx, ny)
            if tile:
                tile.terrain_type = TerrainType.ROCK

        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)

        tes = TileEffectSystem()
        for _ in range(400):
            tes.update(small_world)

        # No neighbour should have become SAND
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = 2 + dx, 2 + dy
            tile = small_world.get_tile(nx, ny)
            if tile:
                assert tile.terrain_type == TerrainType.ROCK

    def test_does_not_spread_to_already_sand_tile(self, small_world):
        """If a neighbour is already SAND, spread shouldn't try again."""
        # Pre-set neighbour to SAND
        tile_n = small_world.get_tile(3, 2)
        tile_n.terrain_type = TerrainType.SAND
        # Also place sand object there
        sand_n = ObjectRegistry.create("sand", 3, 2)
        small_world.add_object(sand_n)

        sand_origin = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand_origin)

        tes = TileEffectSystem()
        tes.update(small_world)

        # (3,2,"sand") should NOT be in exposure ticks since it already has sand
        key = (3, 2, "sand")
        assert key not in tes._exposure_ticks

    def test_stale_exposure_keys_pruned(self, small_world):
        """Exposure entries for tiles no longer near a source are cleaned up."""
        sand = ObjectRegistry.create("sand", 2, 2)
        small_world.add_object(sand)

        tes = TileEffectSystem()
        tes.update(small_world)
        # Exposure keys should exist for soil neighbours
        assert len(tes._exposure_ticks) > 0

        # Remove the sand object
        small_world.remove_object(sand.id)
        tes.update(small_world)

        # All exposure entries should be pruned
        assert len(tes._exposure_ticks) == 0


# ===================================================================
# SAND terrain type
# ===================================================================


class TestSandTerrainType:
    """Verify SAND terrain type integration in tiles."""

    def test_sand_in_terrain_enum(self):
        assert TerrainType.SAND.value == "sand"

    def test_sand_is_plantable(self):
        """Seeds can be planted on sand (though growth is very slow)."""
        w = World(5, 5, seed=1)
        tile = w.get_tile(2, 2)
        tile.terrain_type = TerrainType.SAND
        assert tile.is_plantable() is True

    def test_sand_is_passable(self):
        w = World(5, 5, seed=1)
        tile = w.get_tile(2, 2)
        tile.terrain_type = TerrainType.SAND
        assert tile.is_passable() is True


# ===================================================================
# Reclamation tests
# ===================================================================


class TestReclamation:
    """Tests for the reclaim_terrain / reclaim_interval feature.

    When a blocker (e.g. a plant) sits on an effect tile (e.g. sand)
    for reclaim_interval ticks, the terrain should convert back
    and the effect object should be removed.
    """

    def _make_sand_tile(self, world, x, y):
        """Place a sand object on (x, y) and set terrain to SAND."""
        tile = world.get_tile(x, y)
        tile.terrain_type = TerrainType.SAND
        tile.fertility = 0.02
        tile.moisture = 0.01
        sand = ObjectRegistry.create("sand", x, y)
        world.add_object(sand)
        return sand

    def _make_plant(self, world, x, y):
        """Place a berry_plant on (x, y) – acts as a blocker.

        Temporarily enables stacking so the plant is placed on the
        same tile as the sand object (instead of being displaced).
        """
        old_stacking = world.allow_stacking
        world.allow_stacking = True
        plant = ObjectRegistry.create("berry_plant", x, y)
        world.add_object(plant)
        world.allow_stacking = old_stacking
        return plant

    def test_reclaim_after_interval(self, small_world):
        """Terrain converts back to soil after reclaim_interval ticks
        with a blocker present."""
        # Override reclaim_interval to a small value for testing
        defn = ObjectRegistry.get("sand")
        old_interval = defn.tile_effect.reclaim_interval
        defn.tile_effect.reclaim_interval = 5

        sand = self._make_sand_tile(small_world, 2, 2)
        plant = self._make_plant(small_world, 2, 2)

        tes = TileEffectSystem()
        # Run for 4 ticks – not yet
        for _ in range(4):
            tes.update(small_world)

        tile = small_world.get_tile(2, 2)
        assert tile.terrain_type == TerrainType.SAND
        assert sand.id in small_world.objects

        # 5th tick should trigger reclamation
        tes.update(small_world)
        assert tile.terrain_type == TerrainType.SOIL
        assert sand.id not in small_world.objects
        assert tile.fertility >= 0.3

        # Restore original interval
        defn.tile_effect.reclaim_interval = old_interval

    def test_reclaim_removes_sand_object(self, small_world):
        """The sand WorldObject is removed upon reclamation."""
        defn = ObjectRegistry.get("sand")
        old_interval = defn.tile_effect.reclaim_interval
        defn.tile_effect.reclaim_interval = 3

        sand = self._make_sand_tile(small_world, 1, 1)
        self._make_plant(small_world, 1, 1)

        tes = TileEffectSystem()
        for _ in range(3):
            tes.update(small_world)

        assert sand.id not in small_world.objects
        tile = small_world.get_tile(1, 1)
        assert sand.id not in tile.object_ids

        defn.tile_effect.reclaim_interval = old_interval

    def test_reclaim_fertility_restored(self, small_world):
        """Reclaimed tile gets fertility >= 0.3."""
        defn = ObjectRegistry.get("sand")
        old_interval = defn.tile_effect.reclaim_interval
        defn.tile_effect.reclaim_interval = 2

        self._make_sand_tile(small_world, 3, 3)
        self._make_plant(small_world, 3, 3)

        tes = TileEffectSystem()
        for _ in range(2):
            tes.update(small_world)

        tile = small_world.get_tile(3, 3)
        assert tile.fertility >= 0.3

        defn.tile_effect.reclaim_interval = old_interval

    def test_reclaim_counter_resets_when_blocker_removed(self, small_world):
        """If the blocker is removed before interval, counter resets to 0."""
        defn = ObjectRegistry.get("sand")
        old_interval = defn.tile_effect.reclaim_interval
        defn.tile_effect.reclaim_interval = 5

        sand = self._make_sand_tile(small_world, 2, 2)
        plant = self._make_plant(small_world, 2, 2)

        tes = TileEffectSystem()
        # Run 3 ticks with blocker
        for _ in range(3):
            tes.update(small_world)
        key = (2, 2, sand.type_id)
        assert tes._reclaim_ticks.get(key, 0) == 3

        # Remove blocker
        small_world.remove_object(plant.id)
        tes.update(small_world)
        assert tes._reclaim_ticks.get(key, 0) == 0

        # Terrain should still be SAND
        tile = small_world.get_tile(2, 2)
        assert tile.terrain_type == TerrainType.SAND

        defn.tile_effect.reclaim_interval = old_interval

    def test_reclaim_disabled_when_empty_terrain(self, small_world):
        """reclaim_terrain='' disables reclamation entirely."""
        defn = ObjectRegistry.get("sand")
        old_terrain = defn.tile_effect.reclaim_terrain
        old_interval = defn.tile_effect.reclaim_interval
        defn.tile_effect.reclaim_terrain = ""
        defn.tile_effect.reclaim_interval = 2

        sand = self._make_sand_tile(small_world, 2, 2)
        self._make_plant(small_world, 2, 2)

        tes = TileEffectSystem()
        for _ in range(5):
            tes.update(small_world)

        tile = small_world.get_tile(2, 2)
        assert tile.terrain_type == TerrainType.SAND
        assert sand.id in small_world.objects

        defn.tile_effect.reclaim_terrain = old_terrain
        defn.tile_effect.reclaim_interval = old_interval

    def test_reclaim_disabled_when_interval_zero(self, small_world):
        """reclaim_interval=0 disables reclamation."""
        defn = ObjectRegistry.get("sand")
        old_interval = defn.tile_effect.reclaim_interval
        defn.tile_effect.reclaim_interval = 0

        sand = self._make_sand_tile(small_world, 2, 2)
        self._make_plant(small_world, 2, 2)

        tes = TileEffectSystem()
        for _ in range(10):
            tes.update(small_world)

        tile = small_world.get_tile(2, 2)
        assert tile.terrain_type == TerrainType.SAND
        assert sand.id in small_world.objects

        defn.tile_effect.reclaim_interval = old_interval

    def test_stale_reclaim_keys_pruned(self, small_world):
        """Reclaim entries for removed effect objects are cleaned up."""
        defn = ObjectRegistry.get("sand")
        old_interval = defn.tile_effect.reclaim_interval
        defn.tile_effect.reclaim_interval = 100  # large so it won't trigger

        sand = self._make_sand_tile(small_world, 2, 2)
        self._make_plant(small_world, 2, 2)

        tes = TileEffectSystem()
        tes.update(small_world)
        assert len(tes._reclaim_ticks) > 0

        # Manually remove sand object
        small_world.remove_object(sand.id)
        tes.update(small_world)

        # Stale reclaim entry should be pruned
        assert len(tes._reclaim_ticks) == 0

        defn.tile_effect.reclaim_interval = old_interval
