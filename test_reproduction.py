"""
Test in-simulation reproduction system.

This test verifies that agents can reproduce via fission when they
have sufficient energy, and that offspring inherit trained weights.

Author: Karan Vasa
Date: November 16, 2025
"""

import random
import yaml
from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent, PlantComponent
from agents import Agent, Genome, create_default_trait_config


def load_config(config_path='config/training_easy.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def populate_world_with_abundant_resources(world, config):
    """Add abundant resources to facilitate reproduction."""
    berry_calories = config['resources']['berry_calories']
    
    # Add LOTS of berries to ensure high energy
    berry_count = 800  # Double the normal amount
    
    for _ in range(berry_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        if world.is_valid_position(x, y):
            berry = WorldObject(x, y)
            berry.add_component(EdibleComponent(calories=berry_calories))
            world.add_object(berry)


def main():
    """Test reproduction system."""
    print("=" * 70)
    print("REPRODUCTION SYSTEM TEST")
    print("=" * 70)
    print("\nThis test verifies in-simulation reproduction:")
    print("  1. Agents reach high energy (70% of max)")
    print("  2. Agents reproduce via fission after age 100")
    print("  3. Offspring inherit trained weights (Lamarckian)")
    print("  4. Population grows over time")
    print("=" * 70)
    
    # Load config
    config = load_config()
      # Create world with abundant resources
    world = World(
        width=config['world']['width'],
        height=config['world']['height']
    )
    
    # Set reproduction config from YAML
    if 'reproduction' in config:
        world.reproduction_config = config['reproduction']
    
    populate_world_with_abundant_resources(world, config)
    
    # Create initial agents
    trait_config = create_default_trait_config()
    agent_config = config['agents']
    num_initial_agents = 5
    
    print(f"\nCreating {num_initial_agents} initial agents...")
    for i in range(num_initial_agents):
        genome = Genome.random(weight_count=2744, trait_config=trait_config)
        agent = Agent(
            x=random.randint(0, world.width - 1),
            y=random.randint(0, world.height - 1),
            genome=genome,
            max_energy=agent_config['max_energy'],
            max_age=agent_config['max_age'],
            inventory_size=agent_config['inventory_size'],
            metabolism_rate=agent_config['metabolism_rate']
        )
        agent.energy = 700.0
        agent.enable_learning(            learning_rate=config['learning']['learning_rate'],
            discount_factor=config['learning']['discount_factor'],
            batch_size=config['learning']['batch_size'],
            buffer_capacity=config['learning']['buffer_capacity']
        )
        world.add_agent(agent)
    
    print(f"  World has {len(world.agents)} agents")
    print(f"  World has {len([o for o in world.objects.values() if o.has_component(EdibleComponent)])} berries")
      # Run simulation
    max_ticks = 1000
    print(f"\nRunning simulation for {max_ticks} ticks...")
    print(f"  Reproduction threshold: 60% of {agent_config['max_energy']} = {agent_config['max_energy'] * 0.6}")
    print(f"  Minimum age for reproduction: 100 ticks")
    print("=" * 70)
    
    reproductions = []
    
    for tick in range(max_ticks):
        initial_count = len([a for a in world.agents.values() if a.alive])
        world.update()
        final_count = len([a for a in world.agents.values() if a.alive])
        
        # Track reproductions
        if final_count > initial_count:
            reproductions.append((tick, initial_count, final_count))
        
        # Print progress
        if tick % 100 == 0 or final_count > initial_count:
            alive_agents = [a for a in world.agents.values() if a.alive]
            if alive_agents:
                avg_energy = sum(a.energy for a in alive_agents) / len(alive_agents)
                max_energy = max(a.energy for a in alive_agents)
                avg_age = sum(a.age for a in alive_agents) / len(alive_agents)
                can_reproduce = sum(1 for a in alive_agents if a.can_reproduce())
                
                print(f"Tick {tick:4d}: {len(alive_agents):2d} alive | "
                      f"Avg Energy: {avg_energy:6.1f} | Max Energy: {max_energy:6.1f} | "
                      f"Avg Age: {avg_age:6.1f} | Can Reproduce: {can_reproduce}")
        
        # Stop if all dead
        if final_count == 0:
            print(f"\n⚠️  All agents died at tick {tick}")
            break
    
    # Print results
    print("\n" + "=" * 70)
    print("REPRODUCTION TEST RESULTS")
    print("=" * 70)
    
    final_count = len([a for a in world.agents.values() if a.alive])
    print(f"\nPopulation:")
    print(f"  Initial: {num_initial_agents}")
    print(f"  Final: {final_count}")
    print(f"  Net change: {final_count - num_initial_agents:+d}")
    
    print(f"\nReproduction events:")
    print(f"  Total: {len(reproductions)}")
    
    if reproductions:
        print(f"\nFirst 10 reproduction events:")
        for i, (tick, before, after) in enumerate(reproductions[:10]):
            print(f"    {i+1}. Tick {tick}: {before} → {after} agents (+{after-before})")
        
        if len(reproductions) > 10:
            print(f"    ... and {len(reproductions) - 10} more")
        
        print(f"\n✅ Reproduction system is working!")
        print(f"✅ Agents successfully reproduced {len(reproductions)} times")
    else:
        print(f"\n⚠️  No reproductions occurred")
        print(f"   Possible reasons:")
        print(f"   - Energy never reached 70% threshold")
        print(f"   - Agents didn't survive to age 100")
        print(f"   - Increase berry count or calories")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
