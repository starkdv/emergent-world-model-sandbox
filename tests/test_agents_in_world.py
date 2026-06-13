"""
Tests for W4 increment 1: genome migration, agents-in-vision, tile collision.

Covers:
- migrate_genome: top-left copy across an append-only spec change, with
  bit-identical brain behaviour for the original actions + value
- agents visible in vision (opt-in), self excluded, energy in the value slot
- tile exclusivity (agent_collision) blocks moving onto an occupied tile

Author: Karan Vasa
"""

from types import SimpleNamespace

import numpy as np
import pytest

import utils.agents.agent_utils as au
from utils.agents.perception import AGENT_VISION_ENCODING, build_observation
from agents.brain import Brain, create_brain
from agents.brain.spec import (
    build_brain_param_spec,
    build_brain_v3_param_spec,
    migrate_genome,
)
from agents.genome import Genome
from world.object_registry import ObjectRegistry, register_builtin_objects
from world.tiles import TerrainType
from world.world import World


@pytest.fixture(autouse=True)
def _reset_registry():
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    ObjectRegistry._definitions.clear()


# ===================================================================
# Genome migration
# ===================================================================


class TestGenomeMigration:
    def test_top_left_copy_v2_obs_growth(self):
        """Growing the observation adds zero encoder rows; everything else
        is preserved exactly."""
        old = build_brain_param_spec(input_size=72, output_size=8)
        new = build_brain_param_spec(input_size=78, output_size=8)
        flat = np.arange(old.count(), dtype=np.float32) + 1.0  # nonzero, distinct
        migrated = migrate_genome(flat, old, new)

        assert migrated.shape == (new.count(),)
        o = old.unpack(flat)
        n = new.unpack(migrated)
        # Encoder weight: old 72 rows copied, 6 new rows zero
        assert np.array_equal(n["encoder.0.W"][:72, :], o["encoder.0.W"])
        assert np.all(n["encoder.0.W"][72:, :] == 0.0)
        # Everything after the encoder is unchanged
        for name in ("gru.Wr_input", "policy.W", "policy.b", "value.W", "value.b"):
            assert np.array_equal(n[name], o[name])

    def test_top_left_copy_v2_action_growth(self):
        """Adding an action appends a zero policy column."""
        old = build_brain_param_spec(input_size=72, output_size=8)
        new = build_brain_param_spec(input_size=72, output_size=9)
        flat = np.arange(old.count(), dtype=np.float32) + 1.0
        migrated = migrate_genome(flat, old, new)
        o = old.unpack(flat)
        n = new.unpack(migrated)
        assert np.array_equal(n["policy.W"][:, :8], o["policy.W"])
        assert np.all(n["policy.W"][:, 8] == 0.0)
        assert np.array_equal(n["policy.b"][:8], o["policy.b"])
        assert n["policy.b"][8] == 0.0

    def test_v3_state_and_action_growth(self):
        old = build_brain_v3_param_spec(state_inputs=22, output_size=8)
        new = build_brain_v3_param_spec(state_inputs=28, output_size=9)
        flat = np.arange(old.count(), dtype=np.float32) + 1.0
        migrated = migrate_genome(flat, old, new)
        o = old.unpack(flat)
        n = new.unpack(migrated)
        assert np.array_equal(n["state_enc.W"][:22, :], o["state_enc.W"])
        assert np.all(n["state_enc.W"][22:, :] == 0.0)
        assert np.array_equal(n["policy.W"][:, :8], o["policy.W"])
        assert np.all(n["policy.W"][:, 8] == 0.0)

    def test_migrated_brain_is_behaviourally_identical(self):
        """A v2 genome migrated to a larger observation must produce
        bit-identical logits for the original actions and the same value,
        given the same base observation (new features → zero rows)."""
        old = build_brain_param_spec(input_size=72, output_size=8)
        rng = np.random.default_rng(0)
        flat = rng.standard_normal(old.count()).astype(np.float32)

        g_old = Genome(weights=flat.copy(), traits={})
        brain_old = Brain(g_old, input_size=72, output_size=8)

        new = build_brain_param_spec(input_size=78, output_size=8)
        g_new = Genome(weights=migrate_genome(flat, old, new), traits={})
        brain_new = Brain(g_new, input_size=78, output_size=8)

        obs72 = rng.standard_normal(72).astype(np.float32)
        obs78 = np.concatenate([obs72, rng.standard_normal(6).astype(np.float32)])

        # forward returns (action_probs, value, next_h)
        probs_old, value_old, _ = brain_old.forward(obs72, brain_old.initial_state())
        probs_new, value_new, _ = brain_new.forward(obs78, brain_new.initial_state())

        assert np.allclose(probs_old, probs_new, atol=1e-6)
        assert abs(value_old - value_new) < 1e-6
        # New features genuinely do nothing for the migrated genome:
        obs78_b = np.concatenate([obs72, rng.standard_normal(6).astype(np.float32)])
        probs_b, _, _ = brain_new.forward(obs78_b, brain_new.initial_state())
        assert np.allclose(probs_new, probs_b, atol=1e-6)


# ===================================================================
# Agents visible in vision
# ===================================================================


def _world(n=9, **kw):
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
            t.moisture = 0.6
    return w


def _make_agent(world, x, y, energy=100.0, direction=(0, -1)):
    from agents.genome import Genome as G

    brain_cfg = {"version": 2}
    from agents.brain import calculate_weight_count_for_config

    n = calculate_weight_count_for_config(brain_cfg)
    g = G.random(n, {})
    from agents.agent import Agent

    a = Agent(x=x, y=y, genome=g)
    a.energy = energy
    a.direction = direction
    return a


class TestAgentsVisible:
    def test_other_agent_appears_in_vision_when_enabled(self):
        w = _world(agents_visible=True)
        a = _make_agent(w, 4, 4, direction=(0, -1))
        b = _make_agent(w, 4, 3, energy=50.0)  # directly ahead (north)
        w.agents = {a.id: a, b.id: b}

        obs = build_observation(a, w)
        spec = __import__(
            "agents.brain.spec", fromlist=["DEFAULT_OBSERVATION_SPEC"]
        ).DEFAULT_OBSERVATION_SPEC
        grid = spec.vision_grid(obs)  # (5,5,2)
        # The tile one step ahead is row=center-1 (rows index dy, 0=furthest)
        center = 2
        ahead_type = grid[center - 1, center, 0]
        ahead_val = grid[center - 1, center, 1]
        assert ahead_type == pytest.approx(AGENT_VISION_ENCODING)
        assert ahead_val == pytest.approx(0.25, abs=0.01)  # 50 / 200 max energy

    def test_disabled_by_default(self):
        w = _world(agents_visible=False)
        a = _make_agent(w, 4, 4, direction=(0, -1))
        b = _make_agent(w, 4, 3)
        w.agents = {a.id: a, b.id: b}
        obs = build_observation(a, w)
        assert not np.any(np.isclose(obs, AGENT_VISION_ENCODING))

    def test_self_not_visible(self):
        w = _world(agents_visible=True)
        a = _make_agent(w, 4, 4)
        w.agents = {a.id: a}
        obs = build_observation(a, w)
        # The center tile (the agent itself) must not read as an agent
        assert not np.any(np.isclose(obs, AGENT_VISION_ENCODING))


# ===================================================================
# Tile exclusivity
# ===================================================================


class TestAgentCollision:
    def test_collision_blocks_move(self):
        w = _world(agent_collision=True)
        a = _make_agent(w, 4, 4, direction=(1, 0))
        b = _make_agent(w, 5, 4)
        w.agents = {a.id: a, b.id: b}
        res = au.execute_move_forward(a, w)
        assert not res.success
        assert a.x == 4 and a.y == 4  # did not move

    def test_no_collision_when_disabled(self):
        w = _world(agent_collision=False)
        a = _make_agent(w, 4, 4, direction=(1, 0))
        b = _make_agent(w, 5, 4)
        w.agents = {a.id: a, b.id: b}
        res = au.execute_move_forward(a, w)
        assert res.success
        assert a.x == 5 and a.y == 4  # moved onto b's tile

    def test_can_move_into_empty_tile_with_collision_on(self):
        w = _world(agent_collision=True)
        a = _make_agent(w, 4, 4, direction=(1, 0))
        w.agents = {a.id: a}
        res = au.execute_move_forward(a, w)
        assert res.success
        assert a.x == 5
