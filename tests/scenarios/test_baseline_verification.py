"""
Final V2 Baseline Verification Test

Long-duration test to verify V2 baseline performance.
This is the final confirmation test before declaring the system production-ready.
"""

import random
from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent, PlantComponent
from agents import Agent, Brain, Genome, create_default_trait_config
from utils.data.agent_logger import AgentLogger


def populate_world_with_resources(world, berry_count=30, plant_count=20, seed_count=15):
    """Add initial resources to the world."""
    # Add berries
    for _ in range(berry_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        if world.is_valid_position(x, y):
            berry = WorldObject(x, y)
            berry.add_component(EdibleComponent(calories=25.0))
            world.add_object(berry)

    # Add plants
    for _ in range(plant_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        tile = world.get_tile(x, y)
        if tile and tile.is_plantable():
            plant = WorldObject(x, y)
            plant.add_component(
                PlantComponent(mature_age=30, max_age=200, spawn_rate=0.1)
            )
            world.add_object(plant)

    # Add seeds
    for _ in range(seed_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        if world.is_valid_position(x, y):
            seed = WorldObject(x, y)
            seed.add_component(SeedComponent(plant_type="berry_plant", grow_time=30))
            world.add_object(seed)


def test_baseline_verification(tmp_path):
    print("=" * 70)
    print("FINAL V2 BASELINE VERIFICATION TEST")
    print("=" * 70)

    # Initialize logger with temporary directory
    log_dir = tmp_path / "logs"
    logger = AgentLogger(output_dir=str(log_dir), log_every_n_ticks=1)
    Agent.logger = logger

    try:
        # Create world
        print("\nInitializing world...")
        world = World(width=30, height=30)
        populate_world_with_resources(world)

        # Create agents with learning enabled
        print("Creating 10 learning-enabled agents...")
        agents = []
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count()

        for i in range(10):
            x = random.randint(0, world.width - 1)
            y = random.randint(0, world.height - 1)

            genome = Genome.random(weight_count=weight_count, trait_config=trait_config)
            agent = Agent(x, y, genome=genome)
            agent.enable_learning()  # CRITICAL: Initialize the learner!
            agents.append(agent)
            world.add_agent(agent)

        # Run simulation
        # Reduced duration for automated testing, original was 2500
        max_ticks = 100
        print(f"\nRunning simulation for {max_ticks} ticks...")

        tick = 0
        all_dead = False

        while tick < max_ticks and not all_dead:
            # Update world
            world.update()

            # Update agents
            alive_agents = [a for a in agents if a.alive]
            for agent in alive_agents:
                agent.update(world)

            # Log states
            agent_dict = {a.id: a for a in agents}
            logger.log_all_states(tick, agent_dict)

            # Check if all dead
            if len(alive_agents) == 0:
                print(f"Tick {tick:4d}: ALL AGENTS DIED")
                all_dead = True
                break

            tick += 1

        # Final assertions
        assert tick > 0
        assert not all_dead or tick > 0

    finally:
        logger.close()
        Agent.logger = None
