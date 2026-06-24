"""
Tests for Brain v3.5 — the W4 Observation-v2 + SIGNAL genome break.

Covers:
- ObservationSpec v2 layout (size 78, EXTRA slice + named indices)
- active-spec switch; perception emits 72 under v1, 78 under v2
- the six EXTRA features (clock, temperature, agent proximity, signal, hazard)
- v3.5 weight count (= v3 + 289) and brain construction (in 78, out 9, state 28)
- migration bit-identity v3 -> v3.5 (original-action probs + value, to float tol)
- SIGNAL: masked when signalling off, available + deposits/decays when on
- get_action_mask widths follow the brain's output_size

Author: Karan Vasa
"""

from types import SimpleNamespace

import numpy as np
import pytest

import utils.agents.agent_utils as au
from agents.actions import Action
from agents.brain import (
    adapt_loaded_genome,
    calculate_weight_count_for_config,
    create_brain,
)
from agents.brain.spec import (
    DEFAULT_OBSERVATION_SPEC,
    OBSERVATION_SPEC_V2,
    build_observation_spec,
    get_active_observation_spec,
    set_observation_version,
)
from agents.genome import Genome
from utils.agents.perception import build_observation
from world.object_registry import ObjectRegistry, register_builtin_objects
from world.tiles import TerrainType
from world.world import World

CFG_V3 = {"version": 3}
CFG_V35 = {"version": 3.5}


@pytest.fixture(autouse=True)
def _reset_env():
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    set_observation_version(1)  # always start from the legacy layout
    yield
    set_observation_version(1)  # and never leak v2 into other tests
    ObjectRegistry._definitions.clear()


# ===================================================================
# ObservationSpec v2
# ===================================================================


class TestObservationSpecV2:
    def test_v1_is_72_with_empty_extra(self):
        s = build_observation_spec(version=1)
        assert s.size == 72
        assert s.extra == slice(72, 72)
        assert s.version == 1
        assert s.time_of_day_sin == -1

    def test_v2_is_78_with_named_extra(self):
        s = OBSERVATION_SPEC_V2
        assert s.size == 78
        assert s.extra == slice(72, 78)
        assert s.version == 2
        assert s.time_of_day_sin == 72
        assert s.time_of_day_cos == 73
        assert s.tile_temperature == 74
        assert s.nearest_agent_proximity == 75
        assert s.nearest_agent_signal == 76
        assert s.on_hazard == 77

    def test_prefix_is_identical(self):
        v1, v2 = build_observation_spec(version=1), build_observation_spec(version=2)
        assert v1.agent_state == v2.agent_state
        assert v1.vision == v2.vision
        assert v1.stimulus == v2.stimulus
        assert v1.inventory == v2.inventory

    def test_active_spec_switch(self):
        assert get_active_observation_spec().size == 72
        set_observation_version(2)
        assert get_active_observation_spec().size == 78
        set_observation_version(1)
        assert get_active_observation_spec().size == 72


# ===================================================================
# Perception under v2
# ===================================================================


def _soil_world(n=9, **kw):
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


def _agent(x, y, energy=100.0, direction=(0, -1)):
    return SimpleNamespace(
        x=x,
        y=y,
        energy=energy,
        max_energy=200.0,
        age=0,
        max_age=1000,
        direction=direction,
        inventory=[],
        inventory_size=5,
        metabolism_rate=0.5,
        alive=True,
        id=id(object()),
    )


class TestPerceptionV2:
    def test_observation_length_follows_version(self):
        w = _soil_world()
        a = _agent(4, 4)
        assert len(build_observation(a, w)) == 72
        set_observation_version(2)
        assert len(build_observation(a, w)) == 78

    def test_extra_features_present_and_bounded(self):
        set_observation_version(2)
        w = _soil_world(
            environment_config={
                "enabled": True,
                "base_temperature": 0.6,
                "weather": {"rain_start_chance": 0.0, "drought_start_chance": 0.0},
            },
            signal_config={"enabled": True},
        )
        w.environment.update(w)  # advance the clock once
        a = _agent(4, 4)
        b = _agent(4, 3, energy=80.0)  # one tile ahead/adjacent
        w.agents = {a.id: a, b.id: b}
        obs = build_observation(a, w)
        spec = OBSERVATION_SPEC_V2
        extra = obs[spec.extra]
        assert len(extra) == 6
        assert np.all(extra >= 0.0) and np.all(extra <= 1.0)
        # temperature feature reflects the environment
        assert obs[spec.tile_temperature] == pytest.approx(w.environment.temperature)
        # an adjacent agent gives high proximity
        assert obs[spec.nearest_agent_proximity] > 0.7

    def test_on_hazard_feature(self):
        set_observation_version(2)
        w = _soil_world()
        thorns = ObjectRegistry.create("thorns", 4, 4)
        w.add_object(thorns)
        a = _agent(4, 4)
        w.agents = {a.id: a}
        obs = build_observation(a, w)
        assert obs[OBSERVATION_SPEC_V2.on_hazard] == pytest.approx(1.0)

    def test_signal_sensed_from_field(self):
        set_observation_version(2)
        w = _soil_world(signal_config={"enabled": True, "strength": 1.0})
        a = _agent(4, 4)
        w.agents = {a.id: a}
        w.emit_signal(4, 4)
        obs = build_observation(a, w)
        assert obs[OBSERVATION_SPEC_V2.nearest_agent_signal] > 0.5


# ===================================================================
# Brain construction & weight count
# ===================================================================


class TestV35Brain:
    def test_weight_count_delta(self):
        n3 = calculate_weight_count_for_config(CFG_V3)
        n35 = calculate_weight_count_for_config(CFG_V35)
        assert n35 - n3 == 289  # +6*40 state_enc, +48 policy.W, +1 policy.b

    def test_brain_shapes(self):
        set_observation_version(2)
        g = Genome.random(calculate_weight_count_for_config(CFG_V35), {})
        b = create_brain(g, CFG_V35)
        assert b.output_size == 9
        assert b.input_size == 78
        assert b.state_inputs == 28
        probs, _, _ = b.forward(
            np.random.randn(78).astype(np.float32), b.initial_state()
        )
        assert len(probs) == 9
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)


# ===================================================================
# Migration bit-identity (the W4 acceptance criterion)
# ===================================================================


class TestMigrationBitIdentity:
    def test_v3_genome_migrates_to_v35_identically(self):
        # Build a v3 brain + its decision on a 72-obs
        set_observation_version(1)
        g3 = Genome.random(calculate_weight_count_for_config(CFG_V3), {})
        b3 = create_brain(g3, CFG_V3)
        obs72 = np.random.default_rng(0).standard_normal(72).astype(np.float32)
        mask8 = np.ones(8, dtype=np.float32)
        p3, v3val, _ = b3.forward(obs72, b3.initial_state(), action_mask=mask8)

        # Migrate the SAME genome into v3.5 and decide on the extended obs
        set_observation_version(2)
        flat = adapt_loaded_genome(g3.weights, CFG_V35)
        assert flat is not None
        b35 = create_brain(Genome(weights=flat, traits={}), CFG_V35)
        obs78 = np.concatenate([obs72, np.random.randn(6).astype(np.float32)])
        mask9 = np.ones(9, dtype=np.float32)
        mask9[Action.SIGNAL.value] = 0.0  # signalling off
        p35, v35val, _ = b35.forward(obs78, b35.initial_state(), action_mask=mask9)

        # Original-action probabilities and value match to float tolerance;
        # SIGNAL is suppressed.
        assert np.allclose(p3, p35[:8], atol=1e-6)
        assert v3val == pytest.approx(v35val, abs=1e-6)
        assert p35[Action.SIGNAL.value] < 1e-9

    def test_adapt_returns_none_on_unmigratable(self):
        # A v2-length genome can't be auto-migrated into v3.5
        set_observation_version(2)
        assert adapt_loaded_genome(np.zeros(8873, dtype=np.float32), CFG_V35) is None

    def test_adapt_passthrough_when_matching(self):
        set_observation_version(2)
        n = calculate_weight_count_for_config(CFG_V35)
        flat = np.arange(n, dtype=np.float32)
        out = adapt_loaded_genome(flat, CFG_V35)
        assert np.array_equal(out, flat)


# ===================================================================
# SIGNAL action + action mask + pheromone field
# ===================================================================


class TestSignalAction:
    def _agent_with_brain(self, world, x, y, cfg):
        g = Genome.random(calculate_weight_count_for_config(cfg), {})
        brain = create_brain(g, cfg)
        return SimpleNamespace(
            x=x,
            y=y,
            energy=100.0,
            max_energy=200.0,
            direction=(1, 0),
            inventory=[],
            inventory_size=5,
            brain=brain,
            id=1,
            alive=True,
        )

    def test_mask_width_follows_output_size(self):
        set_observation_version(2)
        w = _soil_world(signal_config={"enabled": True})
        a = self._agent_with_brain(w, 4, 4, CFG_V35)
        w.agents = {a.id: a}
        mask = au.get_action_mask(a, w)
        assert len(mask) == 9
        assert mask[Action.SIGNAL.value] == 1.0  # available when enabled

    def test_signal_masked_when_disabled(self):
        set_observation_version(2)
        w = _soil_world(signal_config={"enabled": False})
        a = self._agent_with_brain(w, 4, 4, CFG_V35)
        w.agents = {a.id: a}
        mask = au.get_action_mask(a, w)
        assert len(mask) == 9
        assert mask[Action.SIGNAL.value] == 0.0

    def test_v3_mask_is_8_wide(self):
        set_observation_version(1)
        w = _soil_world()
        a = self._agent_with_brain(w, 4, 4, CFG_V3)
        w.agents = {a.id: a}
        assert len(au.get_action_mask(a, w)) == 8

    def test_execute_signal_deposits_and_decays(self):
        w = _soil_world(signal_config={"enabled": True, "strength": 1.0, "decay": 0.5})
        a = _agent(4, 4)
        res = au.execute_signal(a, w)
        assert res.success and res.interaction_kind == "signal"
        assert w.pheromones[4, 4] == pytest.approx(1.0)
        w._update_pheromones()  # one tick of decay
        assert w.pheromones[4, 4] == pytest.approx(0.5)

    def test_signal_noop_when_field_absent(self):
        w = _soil_world(signal_config={"enabled": False})
        a = _agent(4, 4)
        res = au.execute_signal(a, w)
        assert res.success  # graceful no-op
        assert w.pheromones is None


# ===================================================================
# End-to-end
# ===================================================================


class TestEndToEnd:
    def test_v35_world_runs(self):
        set_observation_version(2)
        w = _soil_world(
            signal_config={"enabled": True},
            environment_config={"enabled": True},
        )
        # A couple of agents with v3.5 brains
        from agents.agent import Agent

        Agent.brain_config = CFG_V35
        try:
            for i in range(3):
                g = Genome.random(calculate_weight_count_for_config(CFG_V35), {})
                ag = Agent(x=2 + i, y=2, genome=g)
                w.add_agent(ag)
            for _ in range(20):
                w.update()
            assert w.tick == 20
            # pheromone field stays bounded
            assert float(w.pheromones.max()) <= 1.0 + 1e-6
        finally:
            Agent.brain_config = None
