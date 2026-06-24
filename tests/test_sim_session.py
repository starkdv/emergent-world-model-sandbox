"""
Tests for the F3a live session core (render/sim_session.py).

SimSession is the transport-agnostic streaming core: snapshot() then a delta
per step(). These tests exercise it without any sockets; the SSE server is a
thin consumer smoke-tested separately.

Author: Karan Vasa
"""

import pytest

from render.sim_session import SimSession, build_demo_world, session_from_checkpoint
from world.object_registry import ObjectRegistry, register_builtin_objects


@pytest.fixture(autouse=True)
def _registry():
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    from agents.agent import Agent

    Agent.brain_config = None
    ObjectRegistry._definitions.clear()


def test_build_demo_world_has_terrain_and_agents():
    s = build_demo_world(width=32, height=32, n_agents=6, seed=3)
    snap = s.snapshot()
    assert snap["type"] == "snapshot"
    assert snap["terrain"]["width"] == 32
    assert len(snap["agents"]) == 6
    # heightmap terrain → at least some elevation variation
    from render.state_bridge import decode_grid

    elev = decode_grid(snap["terrain"]["elevation"])
    assert int(elev.max()) > int(elev.min())


def test_step_advances_tick_and_returns_delta():
    s = build_demo_world(width=24, height=24, n_agents=5, seed=1)
    s.snapshot()
    t0 = s.tick
    d = s.step()
    assert d["type"] == "delta"
    assert s.tick == t0 + 1
    assert d["tick"] == s.tick


def test_snapshot_reprimes_tracker():
    # After a fresh snapshot, the very next step reports only post-snapshot
    # changes (not the whole world again).
    s = build_demo_world(width=24, height=24, n_agents=4, seed=2)
    s.snapshot()
    d1 = s.step()
    # agents move/metabolise each tick, so the delta is non-trivial but bounded
    assert len(d1["agents"]) <= 4


def test_multi_step():
    s = build_demo_world(width=20, height=20, n_agents=3, seed=5)
    s.snapshot()
    d = s.step(n=5)
    assert s.tick == 5
    assert d["tick"] == 5


def test_session_from_checkpoint(tmp_path):
    import os
    from world.checkpoint import save_state

    s = build_demo_world(width=20, height=20, n_agents=4, seed=8)
    s.snapshot()
    s.step(n=3)
    path = os.path.join(str(tmp_path), "ck.pkl")
    save_state(s.world, path, config={"brain": {"version": 2}})

    s2 = session_from_checkpoint(path, config={"brain": {"version": 2}})
    assert isinstance(s2, SimSession)
    assert s2.tick == s.tick
    snap = s2.snapshot()
    assert snap["tick"] == s.tick
