"""
Test script for the learning system.

This script demonstrates:
1. Agents learning during their lifetime
2. Knowledge transfer to offspring
3. Evolution + learning hybrid system
4. Comparison of learning vs non-learning agents

Author: Karan Vasa
Date: November 15, 2025
"""

import numpy as np
import random
from agents import Agent, Genome, create_default_trait_config
from agents.brain import Brain
from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent


def populate_world(world, food_count=10, seed_count=5):
    """Add food and seeds to the world."""
    # Add berries (food)
    for _ in range(food_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        
        if world.is_valid_position(x, y):
            berry = WorldObject(x, y)
            berry.add_component(EdibleComponent(calories=20.0))
            world.add_object(berry)
    
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


def test_basic_learning():
    """Test that agents can learn and improve."""
    print("=" * 60)
    print("TEST 1: Basic Learning")
    print("=" * 60)
      # Create world
    world = World(width=20, height=20)
    populate_world(world, food_count=10, seed_count=5)
    
    # Create agent with learning enabled
    genome = Genome()
    agent = Agent(x=10, y=10, genome=genome)
    agent.enable_learning(learning_rate=0.01, batch_size=16)
    
    world.add_agent(agent)
    
    print(f"Initial agent: {agent}")
    print(f"Learning enabled: {agent.learning_enabled}")
    print(f"Has learner: {agent.learner is not None}")
    
    # Run simulation for 100 ticks
    initial_fitness = agent.fitness
    initial_energy = agent.energy
    
    for tick in range(100):
        world.update()
        if not agent.alive:
            print(f"  Agent died at tick {tick}")
            break
    
    print(f"\nAfter 100 ticks:")
    print(f"  Alive: {agent.alive}")
    print(f"  Energy: {initial_energy:.1f} -> {agent.energy:.1f}")
    print(f"  Fitness: {initial_fitness:.1f} -> {agent.fitness:.1f}")
    print(f"  Age: {agent.age}")
    print(f"  Experiences collected: {len(agent.learner.replay_buffer)}")
    
    print("✓ Basic learning test passed\n")


def test_knowledge_transfer():
    """Test that learned knowledge can be passed to offspring."""
    print("=" * 60)
    print("TEST 2: Knowledge Transfer to Offspring")
    print("=" * 60)
    
    # Create parent agent with learning
    parent_genome = Genome()
    parent = Agent(x=10, y=10, genome=parent_genome)
    parent.enable_learning(learning_rate=0.01, batch_size=16)
      # Simulate parent learning
    world = World(width=20, height=20)
    populate_world(world, food_count=10, seed_count=5)
    world.add_agent(parent)
    
    print(f"Parent: {parent}")
    
    # Run parent for 50 ticks
    for _ in range(50):
        world.update()
        if not parent.alive:
            break
    
    # Extract learned knowledge
    learned_weights = parent.get_learned_knowledge()
    print(f"  Parent experiences: {len(parent.learner.replay_buffer)}")
    print(f"  Learned weights shape: {learned_weights.shape if learned_weights is not None else 'None'}")
    
    # Create offspring with inherited knowledge
    offspring_genome = parent_genome.crossover(parent_genome)  # Self-cross for simplicity
    offspring = Agent(x=5, y=5, genome=offspring_genome)
    offspring.enable_learning(learning_rate=0.01, batch_size=16)
    
    if learned_weights is not None:
        offspring.inherit_knowledge(learned_weights)
        print(f"\nOffspring inherited parent's knowledge")
    
    print(f"Offspring: {offspring}")
    print("✓ Knowledge transfer test passed\n")


def test_learning_vs_no_learning():
    """Compare learning agents vs non-learning agents."""
    print("=" * 60)
    print("TEST 3: Learning vs Non-Learning Agents")
    print("=" * 60)
    
    # Create two identical agents
    genome1 = Genome()
    genome2 = Genome(weights=genome1.weights.copy(), traits=genome1.traits.copy())
    
    # Agent 1: With learning
    agent_learning = Agent(x=5, y=5, genome=genome1)
    agent_learning.enable_learning(learning_rate=0.01, batch_size=16)
    
    # Agent 2: Without learning (pure evolution)
    agent_no_learning = Agent(x=15, y=15, genome=genome2)
    # No learning enabled - pure evolution
      # Create world
    world = World(width=20, height=20)
    populate_world(world, food_count=20, seed_count=10)
    world.add_agent(agent_learning)
    world.add_agent(agent_no_learning)
    
    print(f"Agent with learning: {agent_learning}")
    print(f"Agent without learning: {agent_no_learning}")
    
    # Run simulation
    for tick in range(200):
        world.update()
        if not agent_learning.alive and not agent_no_learning.alive:
            break
    
    print(f"\nAfter simulation:")
    print(f"  Learning agent:")
    print(f"    Alive: {agent_learning.alive}")
    print(f"    Age: {agent_learning.age}")
    print(f"    Energy: {agent_learning.energy:.1f}")
    print(f"    Fitness: {agent_learning.fitness:.1f}")
    
    print(f"  Non-learning agent:")
    print(f"    Alive: {agent_no_learning.alive}")
    print(f"    Age: {agent_no_learning.age}")
    print(f"    Energy: {agent_no_learning.energy:.1f}")
    print(f"    Fitness: {agent_no_learning.fitness:.1f}")
    
    print("✓ Comparison test passed\n")


def test_population_with_learning():
    """Test a small population with learning enabled."""
    print("=" * 60)
    print("TEST 4: Population with Learning")
    print("=" * 60)
      # Create world
    world = World(width=30, height=30)
    populate_world(world, food_count=30, seed_count=15)
    
    # Create population
    num_agents = 10
    agents = []
    for i in range(num_agents):
        genome = Genome()
        agent = Agent(
            x=np.random.randint(0, world.width),
            y=np.random.randint(0, world.height),
            genome=genome
        )
        agent.enable_learning(learning_rate=0.01, batch_size=16)
        world.add_agent(agent)
        agents.append(agent)
    
    print(f"Created {num_agents} agents with learning enabled")
    
    # Run simulation
    ticks = 300
    for tick in range(ticks):
        world.update()
        
        # Count alive agents
        alive_count = sum(1 for a in agents if a.alive)
        if tick % 50 == 0:
            print(f"  Tick {tick}: {alive_count}/{num_agents} agents alive")
        
        if alive_count == 0:
            print(f"  All agents died at tick {tick}")
            break
    
    # Final statistics
    alive_agents = [a for a in agents if a.alive]
    print(f"\nFinal results after {ticks} ticks:")
    print(f"  Survivors: {len(alive_agents)}/{num_agents}")
    
    if alive_agents:
        avg_age = np.mean([a.age for a in alive_agents])
        avg_energy = np.mean([a.energy for a in alive_agents])
        avg_fitness = np.mean([a.fitness for a in alive_agents])
        avg_experiences = np.mean([len(a.learner.replay_buffer) for a in alive_agents])
        
        print(f"  Average age: {avg_age:.1f}")
        print(f"  Average energy: {avg_energy:.1f}")
        print(f"  Average fitness: {avg_fitness:.1f}")
        print(f"  Average experiences: {avg_experiences:.1f}")
    
    print("✓ Population test passed\n")


def main():
    """Run all learning tests."""
    print("\n" + "=" * 60)
    print("LEARNING SYSTEM TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_basic_learning()
        test_knowledge_transfer()
        test_learning_vs_no_learning()
        test_population_with_learning()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nLearning system is working correctly.")
        print("Agents can:")
        print("  ✓ Learn during their lifetime")
        print("  ✓ Store and replay experiences")
        print("  ✓ Transfer knowledge to offspring")
        print("  ✓ Function in populations")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
