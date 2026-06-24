"""
Tests for the W6a spatial index and World.nearest_edible.

The index is an acceleration structure for the nearest-food scans in
perception and the reward shaper. Its defining contract is that it changes
*speed, not results* — so the core tests assert that nearest_edible with the
index ON matches the legacy tile scan (index OFF) across random worlds, and
that the index stays consistent as berries are added, moved, eaten, picked up
and dropped.

Author: Karan Vasa
"""

import random
from types import SimpleNamespace

import pytest

from world.spatial_index import SpatialIndex
from world.objects import WorldObject, EdibleComponent
from world.tiles import TerrainType
from world.world import World
from world.object_registry import ObjectRegistry, register_builtin_objects
import utils.agents.agent_utils as au


@pytest.fixture(autouse=True)
def _registry():
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    ObjectRegistry._definitions.clear()


# ===========================================================================
# SpatialIndex unit behaviour
# ===========================================================================


class TestSpatialIndex:
    def test_add_query_remove(self):
        idx = SpatialIndex(50, 50, cell_size=8)
        idx.add(1, 10, 10)
        idx.add(2, 11, 12)
        idx.add(3, 40, 40)
        got = set(idx.query_box(8, 8, 16, 16))
        assert got == {1, 2}
        assert 3 not in got
        idx.remove(2)
        assert set(idx.query_box(8, 8, 16, 16)) == {1}
        assert 2 not in idx and len(idx) == 2

    def test_move_across_cells(self):
        idx = SpatialIndex(50, 50, cell_size=8)
        idx.add(1, 1, 1)
        assert set(idx.query_box(0, 0, 7, 7)) == {1}
        idx.move(1, 40, 40)
        assert set(idx.query_box(0, 0, 7, 7)) == set()
        assert set(idx.query_box(40, 40, 41, 41)) == {1}
        assert idx.position_of(1) == (40, 40)

    def test_readd_is_idempotent(self):
        idx = SpatialIndex(50, 50)
        idx.add(1, 5, 5)
        idx.add(1, 5, 5)
        assert len(idx) == 1
        assert list(idx.query_box(0, 0, 10, 10)).count(1) == 1


# ===========================================================================
# World.nearest_edible — index ON must equal index OFF (the contract)
# ===========================================================================


def _blank_world(n=30, spatial_index=True, seed=1):
    w = World(
        n,
        n,
        seed=seed,
        soil_ratio=1.0,
        rock_ratio=0.0,
        water_ratio=0.0,
        sand_ratio=0.0,
        parallel=False,
        allow_stacking=True,  # let us scatter freely for the equality test
        performance_config={"spatial_index": spatial_index},
    )
    for y in range(n):
        for x in range(n):
            t = w.get_tile(x, y)
            t.terrain_type = TerrainType.SOIL
            t.fertility = 0.6
            t.moisture = 0.6
    return w


def _scatter_berries(world, coords):
    for x, y in coords:
        obj = WorldObject(x=x, y=y)
        obj.type_id = "berry"
        obj.add_component(EdibleComponent(calories=20.0, freshness=1.0))
        world.add_object(obj)


def test_nearest_edible_matches_tile_scan_random():
    rng = random.Random(42)
    for trial in range(20):
        coords = {
            (rng.randint(0, 29), rng.randint(0, 29)) for _ in range(rng.randint(0, 25))
        }
        w_on = _blank_world(spatial_index=True, seed=trial)
        w_off = _blank_world(spatial_index=False, seed=trial)
        _scatter_berries(w_on, coords)
        _scatter_berries(w_off, coords)
        assert w_on.food_index is not None
        assert w_off.food_index is None
        for _ in range(15):
            ax, ay = rng.randint(0, 29), rng.randint(0, 29)
            for scan_r in (5, 10):
                a = w_on.nearest_edible(ax, ay, scan_r)
                b = w_off.nearest_edible(ax, ay, scan_r)
                assert a == b, f"trial={trial} ({ax},{ay}) r={scan_r}: {a} != {b}"


def test_nearest_edible_tiebreak_is_row_major():
    # Two berries equidistant (Manhattan 2) from (5,5): (3,5) and (5,3).
    # Row-major tie-break (min y, then x) → (5,3) wins (y=3 < y=5).
    w = _blank_world(seed=7)
    _scatter_berries(w, [(3, 5), (5, 3)])
    dist, fx, fy = w.nearest_edible(5, 5, 5)
    assert (dist, fx, fy) == (2.0, 5, 3)


def test_nearest_edible_none_when_empty():
    w = _blank_world(seed=3)
    assert w.nearest_edible(15, 15, 5) is None


# ===========================================================================
# Index maintenance through the real object lifecycle
# ===========================================================================


class TestIndexMaintenance:
    def _agent(self, x, y, direction=(0, -1), inv=None):
        return SimpleNamespace(
            id=1,
            x=x,
            y=y,
            direction=direction,
            alive=True,
            inventory=list(inv or []),
            inventory_size=5,
            energy=100.0,
            max_energy=200.0,
            fitness=0.0,
        )

    def test_add_remove_object_tracked(self):
        w = _blank_world(seed=2)
        obj = WorldObject(x=10, y=10)
        obj.type_id = "berry"
        obj.add_component(EdibleComponent(calories=20.0, freshness=1.0))
        w.add_object(obj)
        assert obj.id in w.food_index
        w.remove_object(obj.id)
        assert obj.id not in w.food_index
        assert w.nearest_edible(10, 10, 5) is None

    def test_non_edible_not_indexed(self):
        from world.objects import SeedComponent

        w = _blank_world(seed=2)
        seed_obj = WorldObject(x=10, y=10)
        seed_obj.type_id = "plant_seed"
        seed_obj.add_component(SeedComponent(plant_type="plant"))
        w.add_object(seed_obj)
        assert seed_obj.id not in w.food_index

    def test_pickup_removes_from_index_then_drop_readds(self):
        w = _blank_world(seed=5)
        # Berry on the agent's tile
        obj = WorldObject(x=10, y=10)
        obj.type_id = "berry"
        obj.add_component(EdibleComponent(calories=20.0, freshness=1.0))
        w.add_object(obj)
        a = self._agent(10, 10)
        w.agents = {a.id: a}
        assert obj.id in w.food_index

        res = au.execute_pick_up(a, w)
        assert res.success and obj.id in a.inventory
        assert obj.id not in w.food_index  # left the tile → out of the index
        assert w.nearest_edible(10, 10, 5) is None

        # Drop it back onto a tile → re-enters the index
        res = au.execute_drop(a, w)
        assert res.success
        assert obj.id in w.food_index
        assert w.nearest_edible(obj.x, obj.y, 5) is not None

    def test_move_object_updates_index(self):
        w = _blank_world(seed=9)
        obj = WorldObject(x=5, y=5)
        obj.type_id = "berry"
        obj.add_component(EdibleComponent(calories=20.0, freshness=1.0))
        w.add_object(obj)
        w.move_object(obj.id, 20, 20)
        assert w.food_index.position_of(obj.id) == (20, 20)
        assert w.nearest_edible(5, 5, 5) is None
        assert w.nearest_edible(20, 20, 5)[0] == 0.0


# ===========================================================================
# End-to-end: a simulated world runs identically with index on/off
# ===========================================================================


def _seeded_world_run(spatial_index, ticks=15):
    from agents.agent import Agent
    from agents.genome import Genome
    from agents.brain import calculate_weight_count_for_config

    cfg = {"version": 2}
    Agent.brain_config = cfg
    try:
        w = World(
            20,
            20,
            seed=123,
            soil_ratio=1.0,
            rock_ratio=0.0,
            water_ratio=0.0,
            sand_ratio=0.0,
            parallel=False,
            performance_config={"spatial_index": spatial_index},
        )
        for y in range(20):
            for x in range(20):
                t = w.get_tile(x, y)
                t.terrain_type = TerrainType.SOIL
                t.fertility = 0.6
                t.moisture = 0.6
        _scatter_berries(w, [(3, 3), (7, 9), (12, 5), (15, 15), (2, 18)])
        rng = random.Random(0)
        for i in range(4):
            g = Genome.random(calculate_weight_count_for_config(cfg), {})
            ag = Agent(x=rng.randint(0, 19), y=rng.randint(0, 19), genome=g)
            w.add_agent(ag)
        for _ in range(ticks):
            w.update()
        return w.tick, len(w.objects)
    finally:
        Agent.brain_config = None


def test_world_runs_with_and_without_index():
    on = _seeded_world_run(True)
    off = _seeded_world_run(False)
    assert on[0] == off[0] == 15
