"""
Tests for the Three.js web UI serialisation layer (``utils.ui.web_serialize``).

These verify that the live ``World`` is translated into JSON-serialisable
payloads the browser client expects:

- ``build_meta``    – static metadata with a card for every registered object
- ``build_state``   – per-frame snapshot (tick, counts, objects, agents)
- ``build_terrain`` – flat terrain-type grid + change signature
- ``inspect_*``     – full detail for tiles / objects / agents

The suite is dependency-light (no browser, no pygame/torch) and JSON-round-trips
every payload to guarantee serialisability.

Author: Karan Vasa
"""

import json

import pytest

from world.world import World
from world.object_registry import ObjectRegistry, register_builtin_objects
from agents.agent import Agent
from agents.brain import Brain
from agents.genome import Genome, create_default_trait_config
from utils.ui import web_serialize as ws


@pytest.fixture
def world():
    """A small, deterministic populated world for serialisation tests."""
    register_builtin_objects()
    w = World(width=12, height=12, seed=123, parallel=False)
    # A few objects of different categories.
    w.add_object(ObjectRegistry.create("berry", 1, 1))
    w.add_object(ObjectRegistry.create("berry_seed", 2, 2))
    w.add_object(ObjectRegistry.create("berry_plant", 3, 3))
    w.add_object(ObjectRegistry.create("fertilizer", 4, 4))
    # A couple of agents on passable tiles.
    trait_cfg = create_default_trait_config()
    weight_count = Brain.calculate_weight_count(
        input_size=72, encoder_layers=[32], gru_hidden_size=32, output_size=8
    )
    for x, y in [(5, 5), (6, 6)]:
        tile = w.get_tile(x, y)
        if tile and tile.is_passable():
            genome = Genome.random(weight_count, trait_cfg)
            w.add_agent(Agent(x=x, y=y, genome=genome))
    return w


def _assert_json(obj):
    """Round-trip through JSON to guarantee serialisability."""
    return json.loads(json.dumps(obj))


def test_build_meta_has_every_object_type(world):
    meta = _assert_json(ws.build_meta(world, None))
    types = meta["object_types"]
    # Every registered type appears with the UI fields the client needs.
    for tid in ObjectRegistry.type_ids():
        assert tid in types
        card = types[tid]
        assert card["display_name"]
        assert card["category"]
        assert len(card["color"]) == 3
        assert "components" in card
    # World dimensions + palette + actions are present.
    assert meta["world"]["width"] == 12
    assert meta["world"]["height"] == 12
    assert "soil" in meta["terrain_palette"]
    assert "MOVE_FORWARD" in meta["actions"]


def test_build_meta_object_cards_include_component_specs(world):
    meta = ws.build_meta(world, None)
    berry = meta["object_types"]["berry"]
    assert "edible" in berry["components"]
    assert berry["edible"]["calories"] > 0
    sand = meta["object_types"]["sand"]
    assert "tile_effect" in sand["components"]
    assert "tile_effect" in sand


def test_build_state_shape(world):
    state = _assert_json(ws.build_state(world, paused=True, speed=2.0, sim_tps=15.0))
    expected_alive = sum(1 for a in world.agents.values() if a.alive)
    assert state["tick"] == 0
    assert state["paused"] is True
    assert state["speed"] == 2.0
    assert state["agent_count"] == expected_alive >= 1
    assert {"total_food", "total_plants", "total_seeds"} <= set(state["counts"])
    # Every object record carries id, type, position and category.
    assert len(state["objects"]) >= 4
    for rec in state["objects"]:
        assert {"id", "t", "x", "y", "cat"} <= set(rec)
    for a in state["agents"]:
        assert {"id", "x", "y", "dx", "dy", "e", "me", "age", "gen"} <= set(a)


def test_state_excludes_inventory_objects(world):
    # Put an object into an agent's inventory; it must not render on the ground.
    agent = next(iter(world.agents.values()))
    berry = ObjectRegistry.create("berry", agent.x, agent.y)
    world.objects[berry.id] = berry
    agent.inventory.append(berry.id)

    state = ws.build_state(world, paused=True, speed=1.0, sim_tps=0.0)
    ids = {rec["id"] for rec in state["objects"]}
    assert berry.id not in ids


def test_build_terrain_and_signature(world):
    terrain = _assert_json(ws.build_terrain(world))
    assert terrain["width"] == 12 and terrain["height"] == 12
    assert len(terrain["types"]) == 12 * 12
    # Signature is stable until terrain types actually change.
    sig1 = ws.terrain_signature(world)
    sig2 = ws.terrain_signature(world)
    assert sig1 == sig2
    # Changing a tile's terrain changes the signature.
    from world.tiles import TerrainType

    world.tiles[0][0].terrain_type = (
        TerrainType.ROCK
        if world.tiles[0][0].terrain_type != TerrainType.ROCK
        else TerrainType.SOIL
    )
    assert ws.terrain_signature(world) != sig1


def test_inspect_tile(world):
    detail = _assert_json(ws.inspect_tile(world, 1, 1))
    assert detail["x"] == 1 and detail["y"] == 1
    assert "terrain" in detail
    assert any(o["type_id"] == "berry" for o in detail["objects"])
    # Out-of-bounds returns None.
    assert ws.inspect_tile(world, -1, -1) is None


def test_inspect_object(world):
    berry_id = next(
        oid for oid, o in world.objects.items() if getattr(o, "type_id", "") == "berry"
    )
    detail = _assert_json(ws.inspect_object(world, berry_id))
    assert detail["type_id"] == "berry"
    assert "edible" in detail["components"]
    assert ws.inspect_object(world, 10**9) is None


def test_inspect_agent(world):
    agent_id = next(iter(world.agents))
    detail = _assert_json(ws.inspect_agent(world, agent_id))
    assert detail["id"] == agent_id
    assert 0 <= detail["energy_pct"] <= 100
    assert "traits" in detail
    assert detail["facing"] in {"NORTH", "EAST", "SOUTH", "WEST"}
    assert ws.inspect_agent(world, 10**9) is None
