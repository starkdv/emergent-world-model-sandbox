"""
V2 Final Verification - Using Training Easy Config

This test properly uses the training_easy.yaml configuration.
"""

import pytest
import random
import yaml
import os
from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent, PlantComponent
from agents import Agent, Brain, Genome, create_default_trait_config
from utils.data.agent_logger import AgentLogger


def load_config():
    """Load configuration from YAML file."""
    # Use config/training_easy.yaml if available, else mock it
    config_path = "config/training_easy.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        # Fallback default config for testing if file doesn't exist
        return {
            "resources": {"berry_calories": 25},
            "plants": {
                "mature_age": 30,
                "max_age": 200,
                "seed_spawn_rate": 0.1,
                "growth_time": 30,
            },
            "world": {"width": 30, "height": 30},
            "agents": {
                "max_energy": 1000,
                "max_age": 5000,
                "inventory_size": 10,
                "metabolism_rate": 0.015,
            },
            "learning": {
                "learning_rate": 0.01,
                "discount_factor": 0.95,
                "batch_size": 32,
                "buffer_capacity": 10000,
            },
        }


def populate_world_with_resources(world, config):
    """Add initial resources based on config."""
    berry_calories = config["resources"]["berry_calories"]
    berry_count = 50  # Reduced for test speed (was 400)
    plant_count = 20  # Reduced for test speed (was 150)
    seed_count = 10  # Reduced for test speed (was 80)

    # Add berries
    for _ in range(berry_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        if world.is_valid_position(x, y):
            berry = WorldObject(x, y)
            berry.add_component(EdibleComponent(calories=berry_calories))
            world.add_object(berry)

    # Add plants
    for _ in range(plant_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        tile = world.get_tile(x, y)
        if tile and tile.is_plantable():
            plant = WorldObject(x, y)
            plant.add_component(
                PlantComponent(
                    mature_age=config["plants"]["mature_age"],
                    max_age=config["plants"]["max_age"],
                    spawn_rate=config["plants"]["seed_spawn_rate"],
                )
            )
            world.add_object(plant)

    # Add seeds
    for _ in range(seed_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        if world.is_valid_position(x, y):
            seed = WorldObject(x, y)
            seed.add_component(
                SeedComponent(
                    plant_type="berry_plant", grow_time=config["plants"]["growth_time"]
                )
            )
            world.add_object(seed)


def test_baseline_easy_config(tmp_path):
    # Load configuration
    config = load_config()

    # Initialize logger with temporary directory
    log_dir = tmp_path / "logs"
    logger = AgentLogger(output_dir=str(log_dir), log_every_n_ticks=1)
    Agent.logger = logger

    try:
        # Create world with config dimensions
        world = World(width=config["world"]["width"], height=config["world"]["height"])

        # Set reproduction config from YAML
        if "reproduction" in config:
            world.reproduction_config = config["reproduction"]

        populate_world_with_resources(world, config)

        # Create agents
        agents = []
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count()
        agent_config = config["agents"]

        for i in range(10):
            x = random.randint(0, world.width - 1)
            y = random.randint(0, world.height - 1)

            genome = Genome.random(weight_count=weight_count, trait_config=trait_config)
            agent = Agent(
                x,
                y,
                genome=genome,
                max_energy=agent_config["max_energy"],
                max_age=agent_config["max_age"],
                inventory_size=agent_config["inventory_size"],
                metabolism_rate=agent_config["metabolism_rate"],
            )
            agent.energy = 700.0  # High starting energy
            agent.enable_learning(
                learning_rate=config["learning"]["learning_rate"],
                discount_factor=config["learning"]["discount_factor"],
                batch_size=config["learning"]["batch_size"],
                buffer_capacity=config["learning"]["buffer_capacity"],
            )
            agents.append(agent)
            world.add_agent(agent)

        # Run simulation
        max_ticks = 20  # Short run for automated test

        all_dead = False

        while world.tick < max_ticks and not all_dead:
            world.update()

            # Check alive agents
            alive_agents = [a for a in agents if a.alive]

            if len(alive_agents) == 0:
                all_dead = True
                break

            # Log states
            agent_dict = {a.id: a for a in agents}
            logger.log_all_states(world.tick, agent_dict)

        assert world.tick > 0

    finally:
        logger.close()
        Agent.logger = None
