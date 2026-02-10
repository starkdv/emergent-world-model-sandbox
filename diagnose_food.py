"""
Diagnostic script to understand why agents can't find/eat food.
"""

import yaml
import random
from world.world import World
from world.objects import EdibleComponent, WorldObject, PlantComponent
from agents.agent import Agent
from agents.genome import Genome
from utils.agents import build_observation

# Load config
with open('config/training_easy.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create world
world = World(
    width=config['world']['width'],
    height=config['world']['height'],
    seed=42,  # Fixed seed for reproducibility
    **config['terrain']
)

# Populate with resources (like in main.py)
print("=" * 60)
print("FOOD SYSTEM DIAGNOSTIC")
print("=" * 60)

initial_resources = config['world']['initial_resources']
print(f"\nAdding {initial_resources} initial resources...")

resources_added = 0
# Add plants
for _ in range(initial_resources // 2):
    x = random.randint(0, world.width - 1)
    y = random.randint(0, world.height - 1)
    tile = world.get_tile(x, y)
    
    if tile and tile.is_plantable():
        plant = WorldObject(x, y)
        plant_cfg = config['plants']
        plant.add_component(PlantComponent(
            mature_age=plant_cfg['mature_age'],
            max_age=plant_cfg['max_age'],
            spawn_rate=plant_cfg['seed_spawn_rate']
        ))
        world.add_object(plant)
        resources_added += 1

# Add berries
for _ in range(initial_resources // 2):
    x = random.randint(0, world.width - 1)
    y = random.randint(0, world.height - 1)
    tile = world.get_tile(x, y)
    
    if tile:
        berry = WorldObject(x, y)
        berry.add_component(EdibleComponent(
            calories=config['resources']['berry_calories']
        ))
        world.add_object(berry)
        resources_added += 1

print(f"Resources added: {resources_added}")

# Check how many food objects exist
edible_objects = []
for obj_id, obj in world.objects.items():
    edible = obj.get_component(EdibleComponent)
    if edible:
        edible_objects.append(obj)

print(f"\nTotal objects in world: {len(world.objects)}")
print(f"Edible objects: {len(edible_objects)}")

if edible_objects:
    print("\nSample edible objects:")
    for obj in edible_objects[:5]:
        edible = obj.get_component(EdibleComponent)
        print(f"  - {obj.object_type} at ({obj.x}, {obj.y}), calories={edible.calories}")

# Create a test agent
print("\n" + "=" * 60)
print("AGENT OBSERVATION TEST")
print("=" * 60)

trait_config = create_default_trait_config(
    metabolism_rate=config['agents']['metabolism_rate'],
    vision_radius=config['agents']['vision_radius']
)

genome = Genome.create_random(
    brain_config=config['brain'],
    trait_config=trait_config
)

agent = Agent(
    genome=genome,
    x=25,
    y=25,
    energy=config['agents']['starting_energy'],
    max_energy=config['agents']['max_energy'],
    max_age=config['agents']['max_age'],
    inventory_size=config['agents']['inventory_size']
)

# Place agent in world
world.agents[agent.id] = agent

# Place food near agent
if len(edible_objects) > 0:
    # Move first food object near agent
    food = edible_objects[0]
    food.x = 25
    food.y = 26  # Right next to agent
    print(f"\nPlaced {food.object_type} at ({food.x}, {food.y})")
    print(f"Agent at ({agent.x}, {agent.y})")
    print(f"Agent vision radius: {agent.vision_radius}")

# Get agent's observation
observation = build_observation(agent, world)

print("\n" + "=" * 60)
print("AGENT'S OBSERVATION")
print("=" * 60)
print(f"Observation vector shape: {observation.shape}")
print(f"Observation values (first 20): {observation[:20]}")

# Check if agent sees food
food_in_vision = False
for obj_id, obj in world.objects.items():
    edible = obj.get_component(EdibleComponent)
    if edible:
        dx = obj.x - agent.x
        dy = obj.y - agent.y
        distance = (dx**2 + dy**2) ** 0.5
        if distance <= agent.vision_radius:
            food_in_vision = True
            print(f"\nFood in vision: {obj.object_type} at ({obj.x}, {obj.y}), distance={distance:.2f}")

if not food_in_vision:
    print("\n⚠️ NO FOOD IN AGENT'S VISION!")

# Test EAT action
print("\n" + "=" * 60)
print("TESTING EAT ACTION")
print("=" * 60)

from agents.actions import ActionType, perform_action

# Get objects on agent's tile
tile = world.get_tile(agent.x, agent.y)
objects_on_tile = world.get_objects_at(agent.x, agent.y)
print(f"\nObjects on agent's tile: {len(objects_on_tile)}")
for obj in objects_on_tile:
    print(f"  - {obj.object_type}")

# Try to eat
energy_before = agent.energy
result = perform_action(agent, ActionType.EAT, world)
energy_after = agent.energy

print(f"\nEAT action result:")
print(f"  Success: {result.success}")
print(f"  Message: {result.message}")
print(f"  Energy before: {energy_before:.2f}")
print(f"  Energy after: {energy_after:.2f}")
print(f"  Energy gain: {energy_after - energy_before:.2f}")

# Now move agent to food location and try again
if len(edible_objects) > 0:
    food = edible_objects[0]
    agent.x = food.x
    agent.y = food.y
    
    print("\n" + "=" * 60)
    print("TESTING EAT WHILE ON FOOD TILE")
    print("=" * 60)
    
    objects_on_tile = world.get_objects_at(agent.x, agent.y)
    print(f"\nAgent moved to ({agent.x}, {agent.y})")
    print(f"Objects on tile: {len(objects_on_tile)}")
    for obj in objects_on_tile:
        edible = obj.get_component(EdibleComponent)
        print(f"  - {obj.object_type}, edible={edible is not None}")
    
    energy_before = agent.energy
    result = perform_action(agent, ActionType.EAT, world)
    energy_after = agent.energy
    
    print(f"\nEAT action result:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Energy before: {energy_before:.2f}")
    print(f"  Energy after: {energy_after:.2f}")
    print(f"  Energy gain: {energy_after - energy_before:.2f}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
