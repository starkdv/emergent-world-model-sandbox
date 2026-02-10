"""
Test the AsyncWorldModelLogger for performance and correctness.

Author: Karan Vasa
Date: February 10, 2026
"""

import pytest
import time
import os
import csv
import numpy as np
from pathlib import Path
from utils.data.async_logger import AsyncWorldModelLogger
from world.world import World
from world.objects import WorldObject, EdibleComponent
from agents import Agent, Genome, create_default_trait_config
from agents.actions import Action, ActionResult


def test_async_logger_basic(tmp_path):
    """Test basic async logger functionality."""
    log_dir = tmp_path / "logs"
    logger = AsyncWorldModelLogger(str(log_dir), log_every_n_ticks=1, batch_size=10)
    
    # Create minimal world
    world = World(width=10, height=10)
    
    # Create agent
    trait_config = create_default_trait_config()
    genome = Genome.random(weight_count=2744, trait_config=trait_config)
    agent = Agent(x=5, y=5, genome=genome)
    world.add_agent(agent)
    
    # Create some transitions
    Agent.world_model_logger = logger
    
    for i in range(5):
        obs_before = agent.observe(world)
        action = Action.MOVE_FORWARD
        result = ActionResult(success=True, energy_cost=1.0)
        obs_after = agent.observe(world)
        
        logger.log_transition(
            tick=i,
            agent=agent,
            action=action,
            result=result,
            reward=0.1,
            obs_before=obs_before,
            obs_after=obs_after,
            world=world,
            x_before=agent.x,
            y_before=agent.y,
            energy_before=agent.energy + 1.0,
            done=False
        )
    
    # Get stats before closing
    stats = logger.get_stats()
    assert stats['transitions_logged'] == 5
    
    # Close and flush
    logger.close()
    
    # Verify file exists and has content
    transitions_file = list(Path(log_dir).glob("transitions_*.csv"))[0]
    assert transitions_file.exists()
    
    # Read and verify
    with open(transitions_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 6  # Header + 5 data rows
        assert rows[0][0] == 'tick'  # Header check
        assert rows[1][0] == '0'  # First tick


def test_async_logger_performance(tmp_path):
    """Test that async logger is non-blocking and fast."""
    log_dir = tmp_path / "logs"
    logger = AsyncWorldModelLogger(
        str(log_dir),
        log_every_n_ticks=1,
        batch_size=100,
        flush_interval=1.0
    )
    
    # Create world and agent
    world = World(width=10, height=10)
    trait_config = create_default_trait_config()
    genome = Genome.random(weight_count=2744, trait_config=trait_config)
    agent = Agent(x=5, y=5, genome=genome)
    world.add_agent(agent)
    
    Agent.world_model_logger = logger
    
    # Time 1000 log operations
    start_time = time.perf_counter()
    
    for i in range(1000):
        obs = agent.observe(world)
        action = Action.MOVE_FORWARD
        result = ActionResult(success=True, energy_cost=1.0)
        
        logger.log_transition(
            tick=i,
            agent=agent,
            action=action,
            result=result,
            reward=0.1,
            obs_before=obs,
            obs_after=obs,
            world=world,
            x_before=agent.x,
            y_before=agent.y,
            energy_before=agent.energy + 1.0,
            done=False
        )
    
    elapsed = time.perf_counter() - start_time
    
    # Should be very fast (non-blocking)
    # 1000 transitions should take < 200ms (vs 2000ms+ with blocking I/O)
    assert elapsed < 0.2, f"Logging 1000 transitions took {elapsed:.3f}s (too slow!)"
    
    # Average time per log should be < 0.2ms
    avg_time_ms = (elapsed / 1000) * 1000
    print(f"Average time per log: {avg_time_ms:.3f}ms")
    assert avg_time_ms < 0.2
    
    # Close and verify all written
    logger.close()
    
    stats = logger.get_stats()
    assert stats['transitions_logged'] == 1000
    
    # Verify file
    transitions_file = list(Path(log_dir).glob("transitions_*.csv"))[0]
    with open(transitions_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 1001  # Header + 1000 data rows


def test_async_logger_batching(tmp_path):
    """Test that batching works correctly."""
    log_dir = tmp_path / "logs"
    logger = AsyncWorldModelLogger(
        str(log_dir),
        log_every_n_ticks=1,
        batch_size=10,  # Small batch for testing
        flush_interval=10.0  # Long interval to test batching
    )
    
    # Create world and agent
    world = World(width=10, height=10)
    trait_config = create_default_trait_config()
    genome = Genome.random(weight_count=2744, trait_config=trait_config)
    agent = Agent(x=5, y=5, genome=genome)
    world.add_agent(agent)
    
    Agent.world_model_logger = logger
    
    # Log 25 transitions (should trigger 2 batches of 10, leaving 5 buffered)
    for i in range(25):
        obs = agent.observe(world)
        action = Action.MOVE_FORWARD
        result = ActionResult(success=True, energy_cost=1.0)
        
        logger.log_transition(
            tick=i,
            agent=agent,
            action=action,
            result=result,
            reward=0.1,
            obs_before=obs,
            obs_after=obs,
            world=world,
            x_before=agent.x,
            y_before=agent.y,
            energy_before=agent.energy + 1.0,
            done=False
        )
    
    # Wait a bit for background thread to process
    time.sleep(0.5)
    
    stats = logger.get_stats()
    assert stats['transitions_logged'] == 25
    assert stats['batches_written'] >= 2  # At least 2 full batches
    
    # Close triggers final flush
    logger.close()
    
    # Verify all written
    transitions_file = list(Path(log_dir).glob("transitions_*.csv"))[0]
    with open(transitions_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 26  # Header + 25 data rows


def test_async_logger_world_state(tmp_path):
    """Test world state logging."""
    log_dir = tmp_path / "logs"
    logger = AsyncWorldModelLogger(str(log_dir), log_every_n_ticks=5, batch_size=5)
    
    # Create world with objects
    world = World(width=10, height=10)
    berry = WorldObject(5, 5)
    berry.add_component(EdibleComponent(calories=25.0))
    world.add_object(berry)
    
    # Create agent
    trait_config = create_default_trait_config()
    genome = Genome.random(weight_count=2744, trait_config=trait_config)
    agent = Agent(x=3, y=3, genome=genome)
    world.add_agent(agent)
    
    # Log world states
    for tick in range(20):
        logger.log_world_state(tick, world)
    
    # Should only log every 5 ticks: ticks 0, 5, 10, 15
    logger.close()
    
    stats = logger.get_stats()
    assert stats['world_states_logged'] == 4
    
    # Verify file
    world_states_file = list(Path(log_dir).glob("world_states_*.csv"))[0]
    with open(world_states_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 5  # Header + 4 data rows


def test_async_logger_episode_end(tmp_path):
    """Test episode end logging."""
    log_dir = tmp_path / "logs"
    logger = AsyncWorldModelLogger(str(log_dir), log_every_n_ticks=1, batch_size=5)
    
    # Create world and agent
    world = World(width=10, height=10)
    trait_config = create_default_trait_config()
    genome = Genome.random(weight_count=2744, trait_config=trait_config)
    agent = Agent(x=5, y=5, genome=genome)
    world.add_agent(agent)
    
    Agent.world_model_logger = logger
    
    # Log some transitions, ending with done=True
    for i in range(5):
        obs = agent.observe(world)
        action = Action.MOVE_FORWARD
        result = ActionResult(success=True, energy_cost=1.0)
        
        is_done = (i == 4)  # Last one is done
        
        logger.log_transition(
            tick=i,
            agent=agent,
            action=action,
            result=result,
            reward=0.1,
            obs_before=obs,
            obs_after=obs,
            world=world,
            x_before=agent.x,
            y_before=agent.y,
            energy_before=agent.energy + 1.0,
            done=is_done,
            death_reason="test_death" if is_done else ""
        )
    
    logger.close()
    
    stats = logger.get_stats()
    assert stats['episodes_logged'] == 1
    
    # Verify episode file
    episodes_file = list(Path(log_dir).glob("episodes_*.csv"))[0]
    with open(episodes_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 2  # Header + 1 episode
        assert rows[1][-6] == 'test_death'  # death_reason column


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
