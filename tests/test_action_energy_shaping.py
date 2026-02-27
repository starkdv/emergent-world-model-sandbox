"""Tests for dynamic action energy shaping in Agent.execute_action."""

from agents import Agent, Brain, Genome, create_default_trait_config
from agents.actions import Action
from world.world import World


def _create_agent_on_passable_tile(world: World) -> Agent:
    trait_config = create_default_trait_config()
    weight_count = Brain.calculate_weight_count()
    genome = Genome.random(weight_count=weight_count, trait_config=trait_config)

    for y in range(2, world.height - 1):
        for x in range(1, world.width - 1):
            tile = world.get_tile(x, y)
            up = world.get_tile(x, y - 1)
            left = world.get_tile(x - 1, y)
            right = world.get_tile(x + 1, y)
            if (
                tile
                and up
                and left
                and right
                and tile.is_passable()
                and up.is_passable()
                and left.is_passable()
                and right.is_passable()
            ):
                agent = Agent(x=x, y=y, genome=genome)
                world.add_agent(agent)
                return agent

    raise AssertionError(
        "Could not find passable spawn tile with passable forward tile"
    )


def test_turn_streak_energy_cost_increases():
    world = World(width=20, height=20, seed=42)
    agent = _create_agent_on_passable_tile(world)

    r1 = agent.execute_action(Action.TURN_RIGHT, world)
    r2 = agent.execute_action(Action.TURN_RIGHT, world)
    r3 = agent.execute_action(Action.TURN_RIGHT, world)

    assert r1.success and r2.success and r3.success
    # Occasional turns should be affordable; only extended spin loops
    # should get progressively more expensive.
    assert r1.energy_cost == r2.energy_cost
    assert r3.energy_cost > r2.energy_cost


def test_move_after_turn_gets_energy_discount():
    world = World(width=20, height=20, seed=42)
    agent = _create_agent_on_passable_tile(world)

    # Initial direction is north; ensure moving forward is possible by helper setup
    turn_result = agent.execute_action(Action.TURN_LEFT, world)
    move_result = agent.execute_action(Action.MOVE_FORWARD, world)

    assert turn_result.success
    assert move_result.success

    # Base MOVE_FORWARD success cost in agent_utils is 0.20; after turn it should be discounted
    assert move_result.energy_cost < 0.20


def test_wait_streak_energy_cost_increases():
    """Consecutive WAITs should become progressively more expensive."""
    world = World(width=20, height=20, seed=42)
    agent = _create_agent_on_passable_tile(world)

    r1 = agent.execute_action(Action.WAIT, world)
    r2 = agent.execute_action(Action.WAIT, world)
    r3 = agent.execute_action(Action.WAIT, world)

    assert r1.success and r2.success and r3.success
    # Occasional WAIT should be affordable; only extended idling streaks
    # should get progressively more expensive.
    assert r1.energy_cost == r2.energy_cost
    assert r3.energy_cost > r2.energy_cost


def test_non_wait_resets_wait_streak():
    """A non-WAIT action should reset the consecutive wait counter."""
    world = World(width=20, height=20, seed=42)
    agent = _create_agent_on_passable_tile(world)

    # Build up wait streak
    agent.execute_action(Action.WAIT, world)
    agent.execute_action(Action.WAIT, world)
    agent.execute_action(Action.WAIT, world)

    # Break the streak with a turn
    agent.execute_action(Action.TURN_RIGHT, world)

    # Next WAIT should be at base cost again (streak = 1)
    r = agent.execute_action(Action.WAIT, world)
    base_wait_cost = 0.18  # from agent_utils.execute_wait
    assert r.energy_cost == round(base_wait_cost, 3)


def test_successful_turn_does_not_reduce_fitness():
    """Successful turning should not incur a fitness penalty."""
    world = World(width=20, height=20, seed=42)
    agent = _create_agent_on_passable_tile(world)

    start_fitness = agent.fitness
    result = agent.execute_action(Action.TURN_LEFT, world)

    assert result.success
    assert agent.fitness >= start_fitness


def test_brain_no_move_forward_bias():
    """Brain should NOT apply a hardcoded bias to MOVE_FORWARD.

    With zero weights and all-valid mask, MOVE_FORWARD should have no
    special advantage.  However, contextual instinct biases for
    PICK_UP (+1.5), EAT (+1.0) and USE (+0.5) are intentional and
    expected to shift probability mass **away** from movement actions.

    We verify:
      1. MOVE_FORWARD is NOT boosted above other non-biased actions.
      2. PICK_UP > EAT > USE > MOVE_FORWARD (instinct ordering).
    """
    import numpy as np
    from agents import Brain, Genome, create_default_trait_config

    trait_config = create_default_trait_config()
    weight_count = Brain.calculate_weight_count()
    # Use zero weights so all base logits are equal (from bias terms only)
    genome = Genome.random(weight_count=weight_count, trait_config=trait_config)
    genome.weights = np.zeros(weight_count, dtype=np.float32)
    brain = Brain(genome)

    obs = np.zeros(brain.input_size, dtype=np.float32)
    h = brain.initial_state()

    # All actions valid
    mask_all = np.ones(brain.output_size, dtype=np.float32)
    probs_all, _, _ = brain.forward(obs, h, action_mask=mask_all)

    move_prob = probs_all[Action.MOVE_FORWARD.value]
    turn_l_prob = probs_all[Action.TURN_LEFT.value]
    pick_prob = probs_all[Action.PICK_UP.value]
    eat_prob = probs_all[Action.EAT.value]
    use_prob = probs_all[Action.USE.value]

    # 1. MOVE_FORWARD should equal other non-biased actions (TURN_LEFT, TURN_RIGHT, DROP, WAIT)
    assert (
        abs(move_prob - turn_l_prob) < 0.001
    ), f"MOVE_FORWARD ({move_prob:.4f}) should match TURN_LEFT ({turn_l_prob:.4f})"

    # 2. Contextual instinct ordering: PICK_UP > EAT > USE > MOVE_FORWARD
    assert pick_prob > eat_prob > use_prob > move_prob, (
        f"Instinct ordering wrong: PICK={pick_prob:.3f} EAT={eat_prob:.3f} "
        f"USE={use_prob:.3f} MOVE={move_prob:.3f}"
    )
