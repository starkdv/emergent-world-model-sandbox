"""
Test the WorldModelLogger for training data collection.
"""
import sys
import os
sys.path.insert(0, '.')

from utils.data.agent_logger import WorldModelLogger
from world.world import World
from agents.agent import Agent
from agents.genome import Genome
from world.objects import WorldObject, EdibleComponent
import numpy as np
import random

print("=== Testing WorldModelLogger ===\n")

# Create output directory
os.makedirs("data/test_logs", exist_ok=True)

# Initialize logger
logger = WorldModelLogger("data/test_logs", log_every_n_ticks=1)

# Create a simple world
world = World(width=20, height=20, seed=42)

# Add some food
for i in range(10):
    x, y = random.randint(0, 19), random.randint(0, 19)
    tile = world.get_tile(x, y)
    if tile and tile.is_passable() and not tile.object_ids:
        berry = WorldObject(x, y)
        berry.add_component(EdibleComponent(calories=35.0))
        world.add_object(berry)

# Create an agent
trait_config = {'metabolism_rate': (0.8, 1.2), 'vision_radius': (3, 7)}
weight_count = 64 * 32 + 32 + 32 * 16 + 16 + 16 * 8 + 8

genome = Genome.random(weight_count, trait_config)
agent = Agent(x=10, y=10, genome=genome, max_energy=1000, max_age=5000, metabolism_rate=0.015)
agent.energy = 700
world.add_agent(agent)

# Set up logging
Agent.world_model_logger = logger

# Run a few ticks manually to generate transitions
from agents.actions import Action, ActionResult
from agents.observation import build_observation

print("Running 20 test ticks...\n")

for tick in range(20):
    world.tick = tick
    
    # Get observation before
    obs_before = build_observation(agent, world)
    x_before = agent.x
    y_before = agent.y
    energy_before = agent.energy
    
    # Take a random action
    action = Action(random.randint(0, 7))
    result = agent.execute_action(action, world)
    
    # Get observation after
    obs_after = build_observation(agent, world)
    
    # Simulate reward
    reward = 0.1 if result.success else -0.1
    
    # Log transition
    logger.log_transition(
        tick=tick,
        agent=agent,
        action=action,
        result=result,
        reward=reward,
        obs_before=obs_before,
        obs_after=obs_after,
        world=world,
        x_before=x_before,
        y_before=y_before,
        energy_before=energy_before,
        done=False,
        death_reason=""
    )
    
    # Log world state
    logger.log_world_state(tick, world)
    
    if tick % 5 == 0:
        print(f"Tick {tick}: action={action.name}, success={result.success}, energy={agent.energy:.1f}")

# Close logger
logger.close()

# Verify files created
print("\n=== Files Created ===")
import os
for f in os.listdir("data/test_logs"):
    fpath = os.path.join("data/test_logs", f)
    size = os.path.getsize(fpath)
    print(f"  {f}: {size:,} bytes")

# Check transitions file
import csv
print("\n=== Transitions Sample ===")
with open(logger.transitions_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f"Columns: {len(header)}")
    print(f"First 10 columns: {header[:10]}")
    print(f"Obs columns: obs_0 to obs_63 present: {'obs_0' in header and 'obs_63' in header}")
    print(f"Obs_next columns: obs_next_0 to obs_next_63 present: {'obs_next_0' in header and 'obs_next_63' in header}")
    
    # Count rows
    rows = list(reader)
    print(f"Data rows: {len(rows)}")

print("\n=== World States Sample ===")
with open(logger.world_states_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f"Columns: {header}")
    rows = list(reader)
    print(f"Data rows: {len(rows)}")

print("\n✅ WorldModelLogger test complete!")
