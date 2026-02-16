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
                tile and up and left and right
                and tile.is_passable()
                and up.is_passable()
                and left.is_passable()
                and right.is_passable()
            ):
                agent = Agent(x=x, y=y, genome=genome)
                world.add_agent(agent)
                return agent

    raise AssertionError("Could not find passable spawn tile with passable forward tile")


def test_turn_streak_energy_cost_increases():
    world = World(width=20, height=20, seed=42)
    agent = _create_agent_on_passable_tile(world)

    r1 = agent.execute_action(Action.TURN_RIGHT, world)
    r2 = agent.execute_action(Action.TURN_RIGHT, world)
    r3 = agent.execute_action(Action.TURN_RIGHT, world)

    assert r1.success and r2.success and r3.success
    assert r1.energy_cost < r2.energy_cost < r3.energy_cost


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
    assert r1.energy_cost < r2.energy_cost < r3.energy_cost


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
    first_surcharge = 0.03 * 1  # streak = 1
    expected = round(base_wait_cost + first_surcharge, 3)
    assert r.energy_cost == expected


def test_brain_move_forward_logit_bias():
    """Brain should add a +0.5 logit bonus to MOVE_FORWARD when it's valid."""
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

    # With zero weights, all base logits are 0.  The +0.5 bias on MOVE_FORWARD
    # means its logit is 0.5, giving it e^0.5 / (7*e^0 + e^0.5) ≈ 0.19 share.
    move_prob = probs_all[Action.MOVE_FORWARD.value]
    assert move_prob > 0.15, f"MOVE_FORWARD prob {move_prob:.4f} unexpectedly low"

    # MOVE_FORWARD should be the most probable action under zero weights + bias
    assert move_prob == probs_all.max(), "MOVE_FORWARD should be the top action with +0.5 bias"
