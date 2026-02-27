"""Sanity checks for observation encoding and egocentric perception."""

from agents import Agent, Brain, Genome, create_default_trait_config
from utils.agents import build_observation
from world.object_registry import ObjectRegistry, register_builtin_objects
from world.world import World


def _make_world_and_agent(seed: int = 11):
    register_builtin_objects()
    world = World(width=20, height=20, seed=seed)

    weight_count = Brain.calculate_weight_count()
    genome = Genome.random(
        weight_count=weight_count, trait_config=create_default_trait_config()
    )
    agent = Agent(x=10, y=10, genome=genome)
    world.add_agent(agent)
    return world, agent


def _stimulus(obs):
    # [58..65]
    return obs[58:66]


def _berry_cells(obs):
    # Vision block [8..57] => 25 cells x 2 features
    vis = obs[8:58]
    cells = []
    idx = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            type_enc = float(vis[idx])
            val_enc = float(vis[idx + 1])
            idx += 2
            if type_enc >= 0.95:  # berry/object encoding
                cells.append((dx, dy, type_enc, val_enc))
    return cells


def test_stimulus_food_signals_change_with_facing_and_position():
    world, agent = _make_world_and_agent()

    # Food 2 tiles north of agent
    berry = ObjectRegistry.create("berry", 10, 8)
    world.add_object(berry)

    agent.direction = (0, -1)  # north
    obs_north = build_observation(agent, world)
    s_north = _stimulus(obs_north)

    assert len(obs_north) == 72
    assert s_north[2] == 1.0  # food_ahead
    assert s_north[3] == 1.0  # resource_ahead
    assert s_north[4] > 0.0  # nearest_food_prox

    # Turn east; same food should no longer be ahead
    agent.direction = (1, 0)
    obs_east = build_observation(agent, world)
    s_east = _stimulus(obs_east)

    assert s_east[2] == 0.0  # food_ahead off when not in front
    assert s_east[4] > 0.0  # still detectable as nearby food


def test_vision_grid_is_egocentric_rotating_with_agent_direction():
    world, agent = _make_world_and_agent(seed=13)

    # Fixed world-space berry to the north of agent
    berry = ObjectRegistry.create("berry", 10, 8)
    world.add_object(berry)

    agent.direction = (0, -1)  # north
    obs_n = build_observation(agent, world)
    cells_n = _berry_cells(obs_n)
    assert any((dx, dy) == (0, -2) for dx, dy, _, _ in cells_n)

    agent.direction = (1, 0)  # east
    obs_e = build_observation(agent, world)
    cells_e = _berry_cells(obs_e)
    assert any((dx, dy) == (-2, 0) for dx, dy, _, _ in cells_e)
