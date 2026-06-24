"""
Tests for the F0 world→render state bridge.

The bridge is read-only and JSON-focused, so the tests assert: a snapshot
round-trips the terrain grids and lists every entity; the delta tracker reports
only what changed (moves, spawns, removals, signals); and the bridge never
mutates the world.

Author: Karan Vasa
"""

import json

import numpy as np
import pytest

from render.state_bridge import (
    BRIDGE_VERSION,
    StateTracker,
    decode_grid,
    world_snapshot,
)
from world.world import World
from world.tiles import TerrainType
from world.objects import WorldObject, EdibleComponent
from world.object_registry import ObjectRegistry, register_builtin_objects


@pytest.fixture(autouse=True)
def _registry():
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    ObjectRegistry._definitions.clear()


def _world(n=8, **kw):
    w = World(
        n,
        n,
        seed=1,
        soil_ratio=1.0,
        rock_ratio=0.0,
        water_ratio=0.0,
        sand_ratio=0.0,
        parallel=False,
        **kw,
    )
    for y in range(n):
        for x in range(n):
            t = w.get_tile(x, y)
            t.terrain_type = TerrainType.SOIL
            t.fertility = 0.6
            t.moisture = 0.4
            t.elevation = (x + y) / (2 * (n - 1))  # a simple ramp
    return w


def _berry(world, x, y):
    o = WorldObject(x=x, y=y)
    o.type_id = "berry"
    o.add_component(EdibleComponent(calories=20.0, freshness=1.0))
    world.add_object(o)
    return o.id


class TestSnapshot:
    def test_snapshot_is_json_serializable(self):
        w = _world()
        snap = world_snapshot(w)
        # Must serialize cleanly (the wire format)
        json.dumps(snap)
        assert snap["type"] == "snapshot"
        assert snap["version"] == BRIDGE_VERSION
        assert snap["terrain"]["width"] == 8
        assert snap["terrain"]["height"] == 8

    def test_terrain_grids_round_trip(self):
        w = _world()
        snap = world_snapshot(w)
        elev = decode_grid(snap["terrain"]["elevation"])
        assert elev.shape == (8, 8)
        # Corner (0,0) elevation 0 → 0; corner (7,7) elevation 1.0 → 255
        assert elev[0, 0] == 0
        assert elev[7, 7] == 255
        terrain = decode_grid(snap["terrain"]["terrain"])
        assert (terrain == 0).all()  # all soil → code 0

    def test_objects_and_agents_listed(self):
        from types import SimpleNamespace

        w = _world()
        _berry(w, 2, 3)
        w.agents = {
            7: SimpleNamespace(
                id=7,
                x=1,
                y=1,
                direction=(0, -1),
                energy=100.0,
                max_energy=200.0,
                alive=True,
                genome=SimpleNamespace(lineage_id=3, generation=2),
            )
        }
        snap = world_snapshot(w)
        assert len(snap["objects"]) == 1
        assert snap["objects"][0]["type_id"] == "berry"
        assert snap["objects"][0]["x"] == 2 and snap["objects"][0]["y"] == 3
        assert len(snap["agents"]) == 1
        a = snap["agents"][0]
        assert a["id"] == 7
        assert a["energy"] == pytest.approx(0.5)  # 100/200
        assert a["lineage"] == 3 and a["generation"] == 2

    def test_snapshot_does_not_mutate_world(self):
        w = _world()
        _berry(w, 2, 3)
        before_objs = dict(w.objects)
        before_tick = w.tick
        world_snapshot(w)
        assert w.objects == before_objs
        assert w.tick == before_tick


class TestDelta:
    def test_first_delta_reports_everything(self):
        from types import SimpleNamespace

        w = _world()
        _berry(w, 2, 3)
        w.agents = {
            1: SimpleNamespace(
                id=1,
                x=0,
                y=0,
                direction=(1, 0),
                energy=50.0,
                max_energy=100.0,
                alive=True,
                genome=None,
            )
        }
        tr = StateTracker()
        d = tr.delta(w)
        assert d["type"] == "delta"
        assert len(d["objects"]) == 1  # the berry, first-seen
        assert len(d["agents"]) == 1

    def test_delta_reports_only_changes(self):
        from types import SimpleNamespace

        w = _world()
        bid = _berry(w, 2, 3)
        agent = SimpleNamespace(
            id=1,
            x=0,
            y=0,
            direction=(1, 0),
            energy=50.0,
            max_energy=100.0,
            alive=True,
            genome=None,
        )
        w.agents = {1: agent}
        tr = StateTracker()
        tr.delta(w)  # prime

        # Nothing changed → empty upserts
        d = tr.delta(w)
        assert d["objects"] == []
        assert d["agents"] == []
        assert d["removed_objects"] == []
        assert d["removed_agents"] == []

        # Move the agent → only the agent is reported
        agent.x = 1
        d = tr.delta(w)
        assert len(d["agents"]) == 1 and d["agents"][0]["x"] == 1
        assert d["objects"] == []

        # Remove the berry → reported as removed
        w.remove_object(bid)
        d = tr.delta(w)
        assert d["removed_objects"] == [bid]

    def test_delta_reports_signals(self):
        w = _world(signal_config={"enabled": True, "strength": 1.0, "decay": 0.9})
        w.emit_signal(3, 4)
        tr = StateTracker()
        d = tr.delta(w)
        assert [3, 4, pytest.approx(1.0, abs=1e-3)] in [
            [s[0], s[1], pytest.approx(s[2], abs=1e-3)] for s in d["signals"]
        ]

    def test_burning_ids_reported_when_fire_on(self):
        # FireSystem off by default → empty; turning it on and marking a plant
        # burning surfaces its id read-only in snapshot + delta.
        w = _world(fire_config={"enabled": True})
        o = _berry(w, 1, 1)  # any object id works for the burning set
        assert world_snapshot(w)["burning"] == []
        w.systems.fire.burning[o] = 3  # read-only from the bridge's view
        assert world_snapshot(w)["burning"] == [o]
        tr = StateTracker()
        assert tr.delta(w)["burning"] == [o]

    def test_dead_agent_removed(self):
        from types import SimpleNamespace

        w = _world()
        agent = SimpleNamespace(
            id=1,
            x=0,
            y=0,
            direction=(1, 0),
            energy=50.0,
            max_energy=100.0,
            alive=True,
            genome=None,
        )
        w.agents = {1: agent}
        tr = StateTracker()
        tr.delta(w)
        del w.agents[1]
        d = tr.delta(w)
        assert d["removed_agents"] == [1]
