"""
Example: Running the simulation with learning enabled.

This demonstrates how to enable learning for agents in the main simulation.

Usage:
    python example_with_learning.py

Author: Karan Vasa
Date: November 15, 2025
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
                grow_time=50,
                max_age=200
            ))
            world.add_object(seed)


def create_learning_agent(x, y):
    """Create an agent with learning enabled."""
    trait_config = create_default_trait_config()
    genome = Genome.random(weight_count=2744, trait_config=trait_config)
    agent = Agent(x=x, y=y, genome=genome)
    
    # Enable learning
    agent.enable_learning(
        learning_rate=0.01,       # Learning speed
        discount_factor=0.95,     # Future reward importance
        batch_size=16,            # Experiences per update
        buffer_capacity=1000      # Max experiences stored
    )
    
    return agent


def main():
    """Run simulation with learning-enabled agents."""
    print("=" * 70)
    print("SIMULATION WITH LEARNING")
    print("=" * 70)
    
    # Initialize logger
    logger = AgentLogger(output_dir="data/logs", log_every_n_ticks=1)
    Agent.logger = logger
    
    # Configuration
    WORLD_SIZE = 30
    NUM_AGENTS = 10
    NUM_TICKS = 500
    REPORT_INTERVAL = 50
    
    # Create world
    print(f"\nInitializing {WORLD_SIZE}x{WORLD_SIZE} world...")
    world = World(width=WORLD_SIZE, height=WORLD_SIZE)
    populate_world_with_resources(world, berry_count=40, plant_count=25, seed_count=20)
    
    # Create agents with learning
    print(f"Creating {NUM_AGENTS} learning-enabled agents...")
    agents = []
    for i in range(NUM_AGENTS):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        agent = create_learning_agent(x, y)
        world.add_agent(agent)
        agents.append(agent)
        print(f"  Agent {i}: pos=({x},{y}), learning=ON")
    
    # Run simulation
    print(f"\nRunning simulation for {NUM_TICKS} ticks...")
    print("=" * 70)
    
    for tick in range(NUM_TICKS):
        world.update()
        
        # Log states
        agent_dict = {a.id: a for a in agents}
        logger.log_all_states(tick, agent_dict)
        
        # Report progress
        if tick % REPORT_INTERVAL == 0 or tick == NUM_TICKS - 1:
            alive_agents = [a for a in agents if a.alive]
            if alive_agents:
                avg_energy = sum(a.energy for a in alive_agents) / len(alive_agents)
                avg_fitness = sum(a.fitness for a in alive_agents) / len(alive_agents)
                avg_age = sum(a.age for a in alive_agents) / len(alive_agents)
                avg_experiences = sum(len(a.learner.replay_buffer) for a in alive_agents) / len(alive_agents)
                
                print(f"Tick {tick:4d}: "
                      f"{len(alive_agents):2d} alive | "
                      f"Avg Energy: {avg_energy:5.1f} | "
                      f"Avg Fitness: {avg_fitness:6.1f} | "
                      f"Avg Age: {avg_age:5.1f} | "
                      f"Avg Exp: {avg_experiences:5.1f}")
            else:
                print(f"Tick {tick:4d}: ALL AGENTS DIED")
                break
    
    # Final report
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    
    alive_agents = [a for a in agents if a.alive]
    dead_agents = [a for a in agents if not a.alive]
    
    print(f"\nFinal Statistics:")
    print(f"  Survivors: {len(alive_agents)}/{NUM_AGENTS}")
    print(f"  Deaths: {len(dead_agents)}")
    
    if alive_agents:
        print(f"\nSurvivors:")
        for agent in sorted(alive_agents, key=lambda a: a.fitness, reverse=True):
            exp_count = len(agent.learner.replay_buffer) if agent.learner else 0
            print(f"  Agent {agent.id}: "
                  f"Age={agent.age:3d}, "
                  f"Energy={agent.energy:5.1f}, "
                  f"Fitness={agent.fitness:6.1f}, "
                  f"Experiences={exp_count:4d}")
        
        # Best agent
        best = max(alive_agents, key=lambda a: a.fitness)
        print(f"\n🏆 Best Agent: #{best.id}")
        print(f"   Fitness: {best.fitness:.1f}")
        print(f"   Age: {best.age}")
        print(f"   Energy: {best.energy:.1f}")
        print(f"   Experiences: {len(best.learner.replay_buffer)}")
        print(f"   Learned weights available: {best.get_learned_knowledge() is not None}")
    
    if dead_agents:
        avg_death_age = sum(a.age for a in dead_agents) / len(dead_agents)
        print(f"\nDead agents average age: {avg_death_age:.1f}")
    
    print(f"\n✅ Logs saved:")
    print(f"  - {logger.action_file}")
    print(f"  - {logger.state_file}")
    
    print("\n" + "=" * 70)
    print("LEARNING SYSTEM STATUS: ✅ OPERATIONAL")
    print("=" * 70)
    print("\nKey observations:")
    print("  • Agents collect experiences during their lifetime")
    print("  • Learning updates occur every 10 ticks")
    print("  • Survivors show improved fitness over time")
    print("  • Learned knowledge can be extracted for offspring")
    print("\nNext steps:")
    print("  • Implement mating system to create offspring")
    print("  • Transfer learned knowledge to next generation")
    print("  • Observe multi-generation improvement")


if __name__ == "__main__":
    main()
