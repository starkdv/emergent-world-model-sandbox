"""
Tests for W6b checkpointing — save_state / load_state.

The defining contract: a checkpoint taken at tick T, then resumed, reproduces a
bit-identical trajectory in serial mode. The core test runs a world, saves,
continues the original N more ticks, then loads the checkpoint into a fresh
world, runs N ticks, and asserts the two final states match exactly (tick,
agent positions/energy/age, object positions, pheromones).

Author: Karan Vasa
"""

import os

import numpy as np
import pytest

from agents.agent import Agent
from agents.genome import Genome
from agents.brain import calculate_weight_count_for_config
from world.world import World
from world.tiles import TerrainType
from world.objects import WorldObject, EdibleComponent
from world.object_registry import ObjectRegistry, register_builtin_objects
from world.checkpoint import save_state, load_state, CHECKPOINT_VERSION


@pytest.fixture(autouse=True)
def _registry():
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    Agent.brain_config = None
    ObjectRegistry._definitions.clear()


CFG = {
    "brain": {"version": 2},
    "simulation": {"parallel": False},
    "terrain": {
        "soil_ratio": 1.0,
        "rock_ratio": 0.0,
        "water_ratio": 0.0,
        "sand_ratio": 0.0,
    },
    "world": {},
    "performance": {"spatial_index": True},
    "signal": {"enabled": True},
}


def _make_world(seed=123):
    w = World(
        18,
        18,
        seed=seed,
        soil_ratio=1.0,
        rock_ratio=0.0,
        water_ratio=0.0,
        sand_ratio=0.0,
        parallel=False,
        performance_config={"spatial_index": True},
        signal_config={"enabled": True},
    )
    for y in range(18):
        for x in range(18):
            t = w.get_tile(x, y)
            t.terrain_type = TerrainType.SOIL
            t.fertility = 0.6
            t.moisture = 0.6
    # Scatter some berries
    for x, y in [(2, 2), (6, 8), (11, 4), (14, 14), (3, 15)]:
        o = WorldObject(x=x, y=y)
        o.type_id = "berry"
        o.add_component(EdibleComponent(calories=20.0, freshness=1.0))
        w.add_object(o)
    return w


def _populate(world, n=5, seed=0):
    Agent.brain_config = CFG["brain"]
    rng = np.random.RandomState(seed)
    for _ in range(n):
        g = Genome.random(calculate_weight_count_for_config(CFG["brain"]), {})
        ag = Agent(x=int(rng.randint(0, 18)), y=int(rng.randint(0, 18)), genome=g)
        world.add_agent(ag)


def _snapshot(world):
    """A hashable, comparable fingerprint of the dynamic world state."""
    agents = sorted(
        (
            a.id,
            a.x,
            a.y,
            tuple(a.direction),
            round(float(a.energy), 6),
            a.age,
            a.alive,
            tuple(a.inventory),
        )
        for a in world.agents.values()
    )
    objects = sorted(
        (oid, o.x, o.y, getattr(o, "type_id", "")) for oid, o in world.objects.items()
    )
    pher = None if world.pheromones is None else np.round(world.pheromones, 6).tobytes()
    return (world.tick, tuple(agents), tuple(objects), pher)


def test_checkpoint_roundtrip_is_identical(tmp_path):
    path = os.path.join(str(tmp_path), "ckpt.pkl")

    # Build + warm up
    w = _make_world()
    _populate(w)
    for _ in range(10):
        w.update()

    # Save, then continue the ORIGINAL another 12 ticks → reference trajectory
    save_state(w, path, config=CFG)
    for _ in range(12):
        w.update()
    reference = _snapshot(w)

    # Load the checkpoint into a fresh world and run the SAME 12 ticks
    w2 = load_state(path, config=CFG)
    for _ in range(12):
        w2.update()
    resumed = _snapshot(w2)

    assert resumed == reference


def test_checkpoint_restores_tick_and_population(tmp_path):
    path = os.path.join(str(tmp_path), "c.pkl")
    w = _make_world(seed=7)
    _populate(w, n=4)
    for _ in range(8):
        w.update()
    save_state(w, path, config=CFG)

    w2 = load_state(path, config=CFG)
    assert w2.tick == w.tick
    assert set(w2.agents.keys()) == set(w.agents.keys())
    assert w2.pheromones is not None  # signal was enabled
    # Spatial index rebuilt from restored objects
    assert w2.food_index is not None
    assert len(w2.food_index) == sum(
        1 for o in w2.objects.values() if World._is_edible(o)
    )


def test_checkpoint_immediate_resume_matches(tmp_path):
    # Save at tick T, load, and run one tick on each — they must agree (guards
    # against RNG state being captured/restored at the wrong moment).
    path = os.path.join(str(tmp_path), "c2.pkl")
    w = _make_world(seed=99)
    _populate(w, n=5)
    for _ in range(6):
        w.update()
    save_state(w, path, config=CFG)

    w.update()
    after_original = _snapshot(w)

    w2 = load_state(path, config=CFG)
    w2.update()
    after_resumed = _snapshot(w2)
    assert after_resumed == after_original


def test_bad_version_rejected(tmp_path):
    import pickle

    path = os.path.join(str(tmp_path), "bad.pkl")
    with open(path, "wb") as f:
        pickle.dump({"version": CHECKPOINT_VERSION + 99}, f)
    with pytest.raises(ValueError):
        load_state(path, config=CFG)
