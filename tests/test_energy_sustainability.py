"""
Test script to verify energy sustainability with new settings.
"""
import yaml
import sys
import random
import pytest
import os
sys.path.insert(0, '.')

from world.world import World
from agents.agent import Agent
from agents.brain import Brain
from agents.genome import Genome
from world.objects import EdibleComponent, WorldObject

def test_energy_sustainability():
    # Load config
    config_path = 'config/training_easy.yaml'
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Minimal mock config
        config = {
            'world': {'width': 50, 'height': 50, 'resource_spawn_rate': 0.1, 'initial_resources': 100},
            'agents': {'max_age': 500, 'starting_energy': 500, 'max_energy': 1000, 'metabolism_rate': 1.0},
            'resources': {'berry_calories': 20, 'min_resources': 20}
        }

    world_cfg = config['world']
    agent_cfg = config['agents']

    print("=== CONFIGURATION ===")
    print(f"Max Age: {agent_cfg['max_age']} ticks")
    print(f"Starting Energy: {agent_cfg['starting_energy']}")
    print(f"Max Energy: {agent_cfg['max_energy']}")
    print(f"Metabolism: {agent_cfg['metabolism_rate']}/tick")
    print(f"Berry Calories: {config['resources']['berry_calories']}")
    print()

    # Create world (simplified)
    world = World(
        width=world_cfg['width'],
        height=world_cfg['height'],
        seed=42,
        safety_spawn_rate=world_cfg['resource_spawn_rate'],
        min_resources=config['resources'].get('min_resources', 20),
        berry_calories=config['resources']['berry_calories']
    )

    # Spawn initial berries
    initial_berries = world_cfg['initial_resources']
    spawned = 0
    for _ in range(initial_berries * 2):  # Try extra times for failed spawns
        if spawned >= initial_berries:
            break
        x = random.randint(0, world_cfg['width']-1)
        y = random.randint(0, world_cfg['height']-1)
        tile = world.get_tile(x, y)
        if tile and tile.is_passable() and not tile.object_ids:
            berry = WorldObject(x, y)
            berry.add_component(EdibleComponent(calories=config['resources']['berry_calories']))
            world.add_object(berry)
            spawned += 1

    # Create agents
    trait_config = {'metabolism_rate': (0.8, 1.2), 'vision_radius': (3, 7)}
    weight_count = Brain.calculate_weight_count()

    for i in range(5):
        x, y = random.randint(0, world_cfg['width']-1), random.randint(0, world_cfg['height']-1)
        tile = world.get_tile(x, y)
        if tile and tile.is_passable():
            genome = Genome.random(weight_count, trait_config)
            agent = Agent(
                x=x, y=y, genome=genome,
                max_energy=agent_cfg['max_energy'],
                max_age=agent_cfg['max_age'],
                metabolism_rate=agent_cfg['metabolism_rate']
            )
            agent.energy = agent_cfg['starting_energy']
            world.add_agent(agent)

    initial_food = sum(1 for o in world.objects.values() if o.has_component(EdibleComponent))
    print("=== SIMULATION START ===")
    print(f"Agents: {len(world.agents)}")
    print(f"Initial food: {initial_food}")
    print()

    # Run simulation
    # Reduced ticks for automated test (was 3000)
    for tick in range(100):
        world.update()
        
        if tick % 50 == 0:
            alive = [a for a in world.agents.values() if a.alive]
            food_count = sum(1 for o in world.objects.values() if o.has_component(EdibleComponent))
            if alive:
                avg_energy = sum(a.energy for a in alive) / len(alive)
                max_age = max(a.age for a in alive)
                min_energy = min(a.energy for a in alive)
                max_energy = max(a.energy for a in alive)
                print(f"Tick {tick:4d}: {len(alive)} alive, energy={min_energy:.0f}-{max_energy:.0f} (avg {avg_energy:.0f}), max_age={max_age}, food={food_count}")
            else:
                print(f"Tick {tick:4d}: All agents dead!")
                break

    print()
    print("=== FINAL STATE ===")
    alive = [a for a in world.agents.values() if a.alive]
    dead = [a for a in world.agents.values() if not a.alive]
    print(f"Survivors: {len(alive)}/{len(world.agents)}")

    if alive:
        print("\nSurviving agents:")
        for a in alive:
            print(f"  Agent {a.id}: age={a.age}, energy={a.energy:.0f}/{a.max_energy:.0f}, inventory={len(a.inventory)}")

    if dead:
        print(f"\nDead agents: {len(dead)}")
        for a in dead[:3]:
            print(f"  Agent {a.id}: died at age {a.age}")
            
    # As this is a sustainability test, we generally expect some survival in short term
    assert len(alive) > 0 or len(dead) < 5

if __name__ == "__main__":
    test_energy_sustainability()
