"""
V4.0 scoring integrity — flat action costs and reproduction fitness.

These lock the two invariants the phase exists for:
  1. no always-legal action is cheaper than WAIT under the flat cost model,
     and an action's cost stops depending on what the agent did last tick;
  2. fitness counts descendants, not actions.

The legacy models remain the default, so every pre-V4.0 test still passes
unchanged — that is asserted here too.
"""

import numpy as np
import pytest

from agents import Agent, Brain, Genome, create_default_trait_config
from agents.actions import Action
from agents.scoring import (
    ALWAYS_VALID_ACTIONS,
    FLAT_ACTION_ENERGY_COST,
    ScoringConfig,
    get_active_scoring_config,
    set_active_scoring_config,
)
from world.world import World


@pytest.fixture(autouse=True)
def _restore_scoring():
    """Every test restores the process-wide scoring config."""
    previous = get_active_scoring_config()
    yield
    set_active_scoring_config(previous)


def _agent_on_open_tile(world: World) -> Agent:
    """Spawn an agent on a tile whose four neighbours are passable."""
    genome = Genome.random(
        weight_count=Brain.calculate_weight_count(),
        trait_config=create_default_trait_config(),
    )
    for y in range(2, world.height - 1):
        for x in range(1, world.width - 1):
            tiles = [
                world.get_tile(x, y),
                world.get_tile(x, y - 1),
                world.get_tile(x - 1, y),
                world.get_tile(x + 1, y),
            ]
            if all(t is not None and t.is_passable() for t in tiles):
                agent = Agent(x=x, y=y, genome=genome)
                world.add_agent(agent)
                return agent
    raise AssertionError("no open spawn tile found")


def test_flat_table_respects_the_wait_floor():
    """The invariant the phase exists for, asserted on the table itself."""
    wait = FLAT_ACTION_ENERGY_COST[Action.WAIT]
    for action in ALWAYS_VALID_ACTIONS:
        assert FLAT_ACTION_ENERGY_COST[action] >= wait, action


def test_flat_costs_do_not_depend_on_history():
    """Under `flat`, repeating an action never makes it more expensive."""
    set_active_scoring_config(ScoringConfig(action_cost_model="flat"))
    world = World(width=20, height=20, seed=42)
    agent = _agent_on_open_tile(world)

    turns = [agent.execute_action(Action.TURN_RIGHT, world) for _ in range(5)]
    assert all(r.success for r in turns)
    assert len({r.energy_cost for r in turns}) == 1
    assert turns[0].energy_cost == FLAT_ACTION_ENERGY_COST[Action.TURN_RIGHT]

    waits = [agent.execute_action(Action.WAIT, world) for _ in range(5)]
    assert len({r.energy_cost for r in waits}) == 1
    assert waits[0].energy_cost == FLAT_ACTION_ENERGY_COST[Action.WAIT]


def test_flat_costs_have_no_turn_to_move_discount():
    """The legacy turn->move discount is a behaviour policy; it is gone."""
    set_active_scoring_config(ScoringConfig(action_cost_model="flat"))
    world = World(width=20, height=20, seed=42)
    agent = _agent_on_open_tile(world)

    agent.execute_action(Action.TURN_LEFT, world)
    move = agent.execute_action(Action.MOVE_FORWARD, world)
    if move.success:
        # >= because the world may add a slope/hazard surcharge on top
        assert move.energy_cost >= FLAT_ACTION_ENERGY_COST[Action.MOVE_FORWARD]


def test_signal_is_not_cheaper_than_waiting_in_a_live_world():
    """`signal.energy_cost` is what actually bills the SIGNAL action."""
    world = World(
        width=20,
        height=20,
        seed=42,
        signal_config={"enabled": True, "energy_cost": 0.18},
    )
    set_active_scoring_config(ScoringConfig(action_cost_model="flat"))
    agent = _agent_on_open_tile(world)

    signal = agent.execute_action(Action.SIGNAL, world)
    wait = agent.execute_action(Action.WAIT, world)
    assert signal.success
    assert signal.energy_cost >= wait.energy_cost


def test_legacy_cost_model_is_still_the_default_and_still_escalates():
    """Nothing pre-V4.0 changes unless a config asks for it."""
    assert get_active_scoring_config().action_cost_model == "legacy"
    world = World(width=20, height=20, seed=42)
    agent = _agent_on_open_tile(world)

    r1 = agent.execute_action(Action.TURN_RIGHT, world)
    r2 = agent.execute_action(Action.TURN_RIGHT, world)
    r3 = agent.execute_action(Action.TURN_RIGHT, world)
    assert r1.energy_cost == r2.energy_cost < r3.energy_cost


def test_reproduction_fitness_ignores_actions():
    """Acting must not move fitness under the reproduction model."""
    set_active_scoring_config(ScoringConfig(fitness_model="reproduction"))
    world = World(width=20, height=20, seed=42)
    agent = _agent_on_open_tile(world)
    agent.fitness = 0.0

    for _ in range(20):
        agent.execute_action(Action.TURN_RIGHT, world)
    assert agent.fitness == 0.0

    for _ in range(20):
        agent.execute_action(Action.EAT, world)  # fails: nothing to eat
    assert agent.fitness == 0.0


def test_reproduction_fitness_counts_matured_offspring():
    """A child credits its parent exactly once, when it comes of age."""
    set_active_scoring_config(
        ScoringConfig(
            fitness_model="reproduction",
            fitness_lifespan_coef=0.0,
            offspring_maturity_ticks=5,
        )
    )
    world = World(width=20, height=20, seed=42)
    parent = _agent_on_open_tile(world)
    child = _agent_on_open_tile(world)
    child.parent_agent_id = parent.id
    child.age = 0

    assert parent.fitness == 0.0
    for _ in range(4):
        child.update(world)
    assert parent.offspring_matured == 0

    child.update(world)  # age now 5 == maturity
    assert parent.offspring_matured == 1
    assert parent.fitness == pytest.approx(1.0)

    for _ in range(10):
        child.update(world)
    assert parent.offspring_matured == 1, "a child must credit its parent once"


def test_reproduction_fitness_lifespan_tiebreak():
    """Lifespan breaks ties but cannot outweigh a single descendant."""
    cfg = ScoringConfig(fitness_model="reproduction", fitness_lifespan_coef=0.001)
    set_active_scoring_config(cfg)
    world = World(width=20, height=20, seed=42)
    agent = _agent_on_open_tile(world)
    agent.age = 999
    agent.offspring_matured = 0
    agent._refresh_fitness(world)
    assert agent.fitness < 1.0


def test_death_penalty_only_applies_to_the_legacy_model():
    """Dying young already costs descendants; no extra hand-written penalty."""
    set_active_scoring_config(
        ScoringConfig(fitness_model="reproduction", fitness_lifespan_coef=0.0)
    )
    world = World(width=20, height=20, seed=42)
    agent = _agent_on_open_tile(world)
    agent.age = 1
    agent.die(world)
    assert agent.fitness == 0.0


def test_scoring_config_from_yaml_shape():
    """The config surface the v4 configs use."""
    cfg = ScoringConfig.from_config(
        {
            "reward": {"action_cost_model": "flat"},
            "evolution": {"fitness_model": "reproduction"},
            "reproduction": {"min_age": 50},
        }
    )
    assert cfg.flat_costs and cfg.reproduction_fitness
    assert cfg.offspring_maturity_ticks == 50
    # Unknown values fall back to legacy rather than exploding
    bad = ScoringConfig.from_config(
        {"reward": {"action_cost_model": "nonsense"}, "evolution": {}}
    )
    assert not bad.flat_costs and not bad.reproduction_fitness


def test_weight_manager_fitness_is_unshaped_under_reproduction():
    """The best-agent ranking must not reintroduce hand-written bonuses."""
    from utils.agents.learning_utils import BestAgentTracker

    set_active_scoring_config(
        ScoringConfig(fitness_model="reproduction", fitness_lifespan_coef=0.0)
    )
    world = World(width=20, height=20, seed=42)
    agent = _agent_on_open_tile(world)
    agent.offspring_matured = 3
    agent.age = 500
    agent.energy = 123.0
    agent._refresh_fitness(world)

    manager = BestAgentTracker(save_dir="data/weights")
    assert manager._calculate_fitness(agent) == pytest.approx(3.0)


def test_observation_is_untouched_by_v40():
    """V4.0 changes scoring only — the observation vector is byte-identical."""
    world = World(width=20, height=20, seed=42)
    agent = _agent_on_open_tile(world)
    before = agent.observe(world)
    set_active_scoring_config(
        ScoringConfig(action_cost_model="flat", fitness_model="reproduction")
    )
    after = agent.observe(world)
    assert np.array_equal(before, after)
