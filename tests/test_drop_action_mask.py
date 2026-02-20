"""Tests for DROP action masking.

When allow_stacking=False, DROP should only be valid if the current tile
is empty OR a nearby empty tile exists (matching execute_drop behavior).
"""

from agents import Agent, Brain, Genome, create_default_trait_config
from agents.actions import Action
from world.world import World
from world.objects import WorldObject, EdibleComponent


def _create_agent_on_passable_tile(world: World) -> Agent:
    trait_config = create_default_trait_config()
    weight_count = Brain.calculate_weight_count()
    genome = Genome.random(weight_count=weight_count, trait_config=trait_config)

    for y in range(2, world.height - 2):
        for x in range(2, world.width - 2):
            tile = world.get_tile(x, y)
            if tile and tile.is_passable():
                agent = Agent(x=x, y=y, genome=genome)
                world.add_agent(agent)
                return agent

    raise AssertionError("Could not find passable spawn tile")


def _add_dummy_object(world: World, x: int, y: int) -> int:
    obj = WorldObject(x, y)
    obj.add_component(EdibleComponent(calories=1.0))
    assert world.add_object(obj)
    return obj.id


def _give_agent_inventory_item(world: World, agent: Agent) -> None:
    # Create an object in the world, then remove it from its tile and put it
    # into the agent inventory (object remains in world.objects).
    oid = _add_dummy_object(world, 0, 0)
    world.tiles[0][0].object_ids.discard(oid)
    agent.inventory.append(oid)


def test_drop_mask_off_when_no_space_strict_mode():
    world = World(width=10, height=10, seed=42, allow_stacking=False)
    agent = _create_agent_on_passable_tile(world)
    _give_agent_inventory_item(world, agent)

    # Occupy the agent tile and all 8 neighbors.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            _add_dummy_object(world, agent.x + dx, agent.y + dy)

    mask = agent.get_action_mask(world)
    assert mask[Action.DROP.value] == 0.0


def test_drop_mask_on_when_neighbor_empty_strict_mode():
    world = World(width=10, height=10, seed=42, allow_stacking=False)
    agent = _create_agent_on_passable_tile(world)
    _give_agent_inventory_item(world, agent)

    # Occupy agent tile and 7 neighbors; leave (agent.x+1, agent.y) empty.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 1 and dy == 0:
                continue
            _add_dummy_object(world, agent.x + dx, agent.y + dy)

    mask = agent.get_action_mask(world)
    assert mask[Action.DROP.value] == 1.0
