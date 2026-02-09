"""
Test V3.1 - Tuned exploration bonus (+0.15)

This script runs a full simulation with logging enabled to analyze V3.1 performance.
"""

import random
from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent, PlantComponent
from agents import Agent, Genome, create_default_trait_config
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
            plant.add_component(PlantComponent(
                mature_age=30,
                max_age=200,
                spawn_rate=0.1
            ))
            world.add_object(plant)
    
    # Add seeds
    for _ in range(seed_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        if world.is_valid_position(x, y):
            seed = WorldObject(x, y)
            seed.add_component(SeedComponent(
                plant_type="berry_plant",
                grow_time=30
            ))
            world.add_object(seed)


def main():
    print("=" * 70)
    print("V3.1 TEST - Tuned Exploration Bonus (+0.15)")
    print("=" * 70)
    
    # Initialize logger
    logger = AgentLogger(output_dir="data/logs", log_every_n_ticks=1)
    
    # Set logger as class variable so agents can use it
    Agent.logger = logger
    
    # Create world
    print("\nInitializing 30x30 world...")
    world = World(width=30, height=30)
    
    # Populate with resources
    populate_world_with_resources(world)
    
    # Create agents with learning enabled
    print("Creating 10 learning-enabled agents...")
    agents = []
    trait_config = create_default_trait_config()
    
    for i in range(10):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        
        # Create random genome (brain needs 2744 weights)
        genome = Genome.random(weight_count=2744, trait_config=trait_config)
        agent = Agent(x, y, genome=genome)
        agent.learning_enabled = True  # Enable learning
        agents.append(agent)
        world.add_agent(agent)
        print(f"  Agent {i}: pos=({x},{y}), learning=ON")
    
    # Run simulation
    max_ticks = 2500
    print(f"\nRunning simulation for {max_ticks} ticks...")
    print("=" * 70)
    
    tick = 0
    all_dead = False
    
    while tick < max_ticks and not all_dead:
        # Update world
        world.update()
        
        # Update agents
        alive_agents = [a for a in agents if a.alive]
        for agent in alive_agents:
            agent.update(world)
        
        # Log states - convert list to dict with agent IDs as keys
        agent_dict = {a.id: a for a in agents}
        logger.log_all_states(tick, agent_dict)
        
        # Check if all dead
        if len(alive_agents) == 0:
            print(f"Tick {tick:4d}: ALL AGENTS DIED")
            all_dead = True
            break
        
        # Status update every 50 ticks
        if tick % 50 == 0:
            avg_energy = sum(a.energy for a in alive_agents) / len(alive_agents)
            avg_fitness = sum(a.fitness for a in alive_agents) / len(alive_agents)
            avg_age = sum(a.age for a in alive_agents) / len(alive_agents)
            avg_exp = sum(len(a.learner.replay_buffer) for a in alive_agents if a.learner) / len(alive_agents)
            
            print(f"Tick {tick:4d}: {len(alive_agents):2d} alive | "
                  f"Avg Energy: {avg_energy:5.1f} | "
                  f"Avg Fitness: {avg_fitness:6.1f} | "
                  f"Avg Age: {avg_age:5.1f} | "
                  f"Avg Exp: {avg_exp:5.1f}")
        
        tick += 1
    
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    
    # Final statistics
    alive_agents = [a for a in agents if a.alive]
    dead_agents = [a for a in agents if not a.alive]
    
    print(f"\nFinal Statistics:")
    print(f"  Total ticks: {tick}")
    print(f"  Survivors: {len(alive_agents)}/{len(agents)}")
    print(f"  Deaths: {len(dead_agents)}")
    
    if alive_agents:
        print(f"  Survivor average age: {sum(a.age for a in alive_agents) / len(alive_agents):.1f}")
        print(f"  Survivor average energy: {sum(a.energy for a in alive_agents) / len(alive_agents):.1f}")
    
    if dead_agents:
        print(f"  Dead agents average age: {sum(a.age for a in dead_agents) / len(dead_agents):.1f}")
    
    print(f"\n✅ Logs saved:")
    print(f"  - {logger.action_file}")
    print(f"  - {logger.state_file}")
    
    print(f"\n📊 To analyze results, run:")
    print(f"  python analyze_food_from_logs.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
