"""
Tests for CSV agent logging functionality.
"""

import pytest
from pathlib import Path
from world.world import World
from agents.agent import Agent
from agents.brain import Brain
from agents.genome import Genome, create_default_trait_config
from utils.data.agent_logger import AgentLogger


def test_agent_logging(tmp_path):
    """Test the agent logging system with a simple simulation."""

    # Initialize logger with temporary directory
    log_dir = tmp_path / "logs"
    logger = AgentLogger(output_dir=str(log_dir), log_every_n_ticks=1)
    Agent.logger = logger

    try:
        # Create world
        world = World(width=20, height=20, seed=42)

        # Create agents
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count()

        agents_created = 0
        for i in range(10):
            x = (i * 3) % world.width
            y = (i * 2) % world.height
            tile = world.get_tile(x, y)

            if tile and tile.is_passable():
                genome = Genome.random(weight_count, trait_config)
                agent = Agent(x=x, y=y, genome=genome)
                world.add_agent(agent)
                agents_created += 1
                if agents_created >= 2:
                    break

        assert agents_created > 0, "Failed to create any agents"

        # Run simulation
        for _ in range(5):
            world.update()

        # Verify logger files exist
        assert Path(logger.action_file).exists()
        assert Path(logger.state_file).exists()

        # Verify content
        with open(logger.state_file, "r") as f:
            header = f.readline()
            assert "tick" in header
            assert "agent_id" in header

            # Check for data
            data = f.readlines()
            assert len(data) > 0

    finally:
        logger.close()
        Agent.logger = None
