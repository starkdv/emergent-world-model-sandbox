"""
Tests for the heightmap terrain generator (World upgrade W2).

Covers:
- elevation field: range, shape, determinism
- mountains at high elevation, water at low elevation
- rivers carved downhill (steepest descent), within the water budget
- spatial coherence (biomes cluster — not a uniform shuffle)
- fertile river corridors (water-adjacent land is more fertile)
- World integration: heightmap build, sand objects, elevation on tiles
- legacy generator remains flat (elevation 0.0) and selectable
- slope movement cost (uphill costs more; flat unchanged)

Author: Karan Vasa
"""

import numpy as np
import pytest

from world.terrain_generation import (
    HeightmapConfig,
    generate_elevation,
    generate_terrain,
)
from world.tiles import TerrainType
from world.world import World
from world.object_registry import ObjectRegistry, register_builtin_objects


@pytest.fixture(autouse=True)
def _reset_registry():
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    ObjectRegistry._definitions.clear()


def _cfg(**kw):
    base = dict(rock_ratio=0.20, water_ratio=0.10, sand_ratio=0.05, river_sources=6)
    base.update(kw)
    return HeightmapConfig(**base)


# ===================================================================
# Elevation field
# ===================================================================


class TestElevationField:
    def test_range_and_shape(self):
        rng = np.random.default_rng(0)
        elev = generate_elevation(40, 30, rng, _cfg())
        assert elev.shape == (30, 40)
        assert elev.min() >= 0.0 and elev.max() <= 1.0
        # A real surface spans most of the range, not a constant
        assert elev.max() - elev.min() > 0.5

    def test_smoothness(self):
        """Neighbouring elevations should be close (it's smoothed noise)."""
        rng = np.random.default_rng(1)
        elev = generate_elevation(60, 60, rng, _cfg())
        horiz = np.abs(np.diff(elev, axis=1)).mean()
        # Smoothed noise has small average neighbour-to-neighbour deltas
        assert horiz < 0.05


# ===================================================================
# Terrain generation
# ===================================================================


class TestGenerateTerrain:
    def test_deterministic(self):
        a = generate_terrain(50, 40, seed=7, cfg=_cfg())
        b = generate_terrain(50, 40, seed=7, cfg=_cfg())
        assert np.array_equal(a.elevation, b.elevation)
        assert a.terrain == b.terrain
        c = generate_terrain(50, 40, seed=8, cfg=_cfg())
        assert not np.array_equal(a.elevation, c.elevation)

    def test_ratios_approximately_honored(self):
        res = generate_terrain(80, 80, seed=3, cfg=_cfg())
        total = 80 * 80
        # Rock is set by a hard quantile → close to target
        assert abs(res.stats["rock"] / total - 0.20) < 0.03
        # Water never exceeds its budget (lakes + rivers), and is non-trivial
        assert 0.03 < res.stats["water"] / total <= 0.11

    def test_mountains_are_high_water_is_low(self):
        res = generate_terrain(80, 80, seed=5, cfg=_cfg())
        elev = res.elevation
        rock = np.array([[t == TerrainType.ROCK for t in row] for row in res.terrain])
        water = np.array([[t == TerrainType.WATER for t in row] for row in res.terrain])
        soil = np.array([[t == TerrainType.SOIL for t in row] for row in res.terrain])
        assert elev[rock].mean() > elev[soil].mean()
        assert elev[water].mean() < elev[soil].mean()

    def test_rivers_are_carved_downhill(self):
        res = generate_terrain(80, 80, seed=11, cfg=_cfg(river_sources=8))
        assert res.stats["rivers_carved"] > 0
        # Rivers reach up into higher ground than lakes alone would: the
        # spread of water elevations is meaningful (not all at the bottom)
        elev = res.elevation
        water = np.array([[t == TerrainType.WATER for t in row] for row in res.terrain])
        assert water.sum() > 0
        assert elev[water].max() - elev[water].min() > 0.2

    def test_no_rivers_when_sources_zero(self):
        res = generate_terrain(60, 60, seed=2, cfg=_cfg(river_sources=0))
        assert res.stats["rivers_carved"] == 0

    def test_spatial_coherence(self):
        """Biomes cluster: orthogonal neighbours share a type far more often
        than the ~uniform-shuffle baseline would predict."""
        res = generate_terrain(80, 80, seed=4, cfg=_cfg())
        terr = res.terrain
        same = 0
        compared = 0
        for y in range(80):
            for x in range(79):
                compared += 1
                if terr[y][x] == terr[y][x + 1]:
                    same += 1
        coherence = same / compared
        # A uniform shuffle of these ratios would share ~0.55; coherent
        # terrain is much higher
        assert coherence > 0.80

    def test_river_corridors_are_fertile(self):
        """Land next to water should be more fertile than land far from it."""
        res = generate_terrain(80, 80, seed=6, cfg=_cfg())
        h, w = res.elevation.shape
        water = np.array([[t == TerrainType.WATER for t in row] for row in res.terrain])
        soil = np.array([[t == TerrainType.SOIL for t in row] for row in res.terrain])
        # Dilate water by one tile to find adjacency
        adj = np.zeros_like(water)
        adj[:-1, :] |= water[1:, :]
        adj[1:, :] |= water[:-1, :]
        adj[:, :-1] |= water[:, 1:]
        adj[:, 1:] |= water[:, :-1]
        near = soil & adj
        far = soil & ~adj
        if near.sum() > 5 and far.sum() > 5:
            assert res.fertility[near].mean() > res.fertility[far].mean()


# ===================================================================
# World integration
# ===================================================================


class TestWorldIntegration:
    def test_heightmap_world_builds_with_elevation(self):
        world = World(
            40,
            40,
            seed=42,
            terrain_generator="heightmap",
            parallel=False,
        )
        # Every tile carries a valid elevation in [0, 1]
        elevs = [world.get_tile(x, y).elevation for y in range(40) for x in range(40)]
        assert all(0.0 <= e <= 1.0 for e in elevs)
        # Not flat — the surface actually varies
        assert max(elevs) - min(elevs) > 0.5
        # Stats recorded
        assert "rivers_carved" in world.terrain_stats

    def test_heightmap_spawns_sand_objects(self):
        world = World(
            50,
            50,
            seed=1,
            terrain_generator="heightmap",
            soil_ratio=0.62,
            rock_ratio=0.20,
            water_ratio=0.10,
            sand_ratio=0.08,
            parallel=False,
        )
        sand_tiles = sum(
            1
            for y in range(50)
            for x in range(50)
            if world.get_tile(x, y).terrain_type == TerrainType.SAND
        )
        sand_objs = sum(
            1 for o in world.objects.values() if getattr(o, "type_id", "") == "sand"
        )
        assert sand_tiles > 0
        assert sand_objs == sand_tiles

    def test_biomes_alias_selects_heightmap(self):
        world = World(30, 30, seed=2, terrain_generator="biomes", parallel=False)
        elevs = [world.get_tile(x, y).elevation for y in range(30) for x in range(30)]
        assert max(elevs) - min(elevs) > 0.4

    def test_legacy_generator_is_flat(self):
        """Legacy generator (default) keeps elevation 0.0 — bit-compatible."""
        world = World(30, 30, seed=42, parallel=False)
        assert world.terrain_generator == "legacy"
        elevs = [world.get_tile(x, y).elevation for y in range(30) for x in range(30)]
        assert all(e == 0.0 for e in elevs)


# ===================================================================
# Slope movement cost
# ===================================================================


class TestSlopeMovementCost:
    def _agent_at(self, world, x, y, direction):
        # execute_move_forward only reads x, y, direction — a tiny stub avoids
        # constructing a full genome/brain
        from types import SimpleNamespace

        return SimpleNamespace(x=x, y=y, direction=direction)

    def test_uphill_costs_more_than_flat(self):
        import utils.agents.agent_utils as au
        from world.tiles import TerrainType

        world = World(
            10,
            10,
            seed=1,
            soil_ratio=1.0,
            rock_ratio=0.0,
            water_ratio=0.0,
            sand_ratio=0.0,
            parallel=False,
        )
        for y in range(10):
            for x in range(10):
                t = world.get_tile(x, y)
                t.terrain_type = TerrainType.SOIL

        # Flat move
        world.get_tile(5, 5).elevation = 0.5
        world.get_tile(6, 5).elevation = 0.5
        agent = self._agent_at(world, 5, 5, (1, 0))
        flat = au.execute_move_forward(agent, world)

        # Uphill move
        world.get_tile(2, 2).elevation = 0.2
        world.get_tile(3, 2).elevation = 0.9
        agent2 = self._agent_at(world, 2, 2, (1, 0))
        uphill = au.execute_move_forward(agent2, world)

        assert flat.success and uphill.success
        assert uphill.energy_cost > flat.energy_cost
        assert flat.energy_cost == pytest.approx(0.20)

    def test_downhill_not_cheaper_than_base(self):
        import utils.agents.agent_utils as au
        from world.tiles import TerrainType

        world = World(
            10,
            10,
            seed=1,
            soil_ratio=1.0,
            rock_ratio=0.0,
            water_ratio=0.0,
            sand_ratio=0.0,
            parallel=False,
        )
        for y in range(10):
            for x in range(10):
                world.get_tile(x, y).terrain_type = TerrainType.SOIL
        world.get_tile(4, 4).elevation = 0.9
        world.get_tile(5, 4).elevation = 0.1
        agent = self._agent_at(world, 4, 4, (1, 0))
        res = au.execute_move_forward(agent, world)
        assert res.energy_cost == pytest.approx(0.20)
