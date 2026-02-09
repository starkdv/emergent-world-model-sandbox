"""
Test the WorldModelLogger for training data collection.
"""
import pytest
import os
import csv
from pathlib import Path
import random

from utils.data.agent_logger import WorldModelLogger
from world.world import World
from agents.agent import Agent
from agents.genome import Genome
from world.objects import WorldObject, EdibleComponent
from agents.actions import Action

def test_world_model_logger(tmp_path):
    """Test the WorldModelLogger functionality."""
    
    # Initialize logger with temporary directory
    log_dir = tmp_path / "test_logs"
    logger = WorldModelLogger(str(log_dir), log_every_n_ticks=1)
    
    try:
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
        # weights: 64*32 + 32 + 32*16 + 16 + 16*8 + 8 = 2744
        weight_count = 2744
        
        genome = Genome.random(weight_count, trait_config)
        agent = Agent(x=10, y=10, genome=genome)
        agent.energy = 700
        world.add_agent(agent)
        
        # Set up logging
        Agent.world_model_logger = logger
        
        # Run a few ticks manually to generate transitions
        from agents.observation import build_observation
        
        for tick in range(5):
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
        
        # Verify files created
        assert Path(logger.transitions_file).exists()
        assert Path(logger.world_states_file).exists()
        
        # Check transitions file content
        with open(logger.transitions_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            assert 'obs_0' in header
            assert 'obs_63' in header
            assert 'obs_next_0' in header
            
            rows = list(reader)
            assert len(rows) == 5
            
        # Check world states file content
        with open(logger.world_states_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
            assert len(rows) == 5

    finally:
        logger.close()
        Agent.world_model_logger = None
