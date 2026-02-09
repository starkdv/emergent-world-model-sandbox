"""
V2 Final Verification - Using Training Easy Config

This test properly uses the training_easy.yaml configuration that was
designed for learning experiments.
"""

import random
import yaml
from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent, PlantComponent
from agents import Agent, Genome, create_default_trait_config
from utils.data.agent_logger import AgentLogger


def load_config(config_path='config/training_easy.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def populate_world_with_resources(world, config):
    """Add initial resources based on config."""
    berry_calories = config['resources']['berry_calories']
    berry_count = 400  # MASSIVELY INCREASED - need abundant food!
    plant_count = 150  # INCREASED - more renewable food sources
    seed_count = 80    # INCREASED - more planting options
    
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
            plant.add_component(PlantComponent(
                mature_age=config['plants']['mature_age'],
                max_age=config['plants']['max_age'],
                spawn_rate=config['plants']['seed_spawn_rate']
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
                grow_time=config['plants']['growth_time']
            ))
            world.add_object(seed)


def main():
    print("=" * 70)
    print("FINAL V2 BASELINE VERIFICATION TEST")
    print("=" * 70)
    print("\nConfiguration:")
    print("  Version: V2 Baseline (No exploration bonus)")
    print("  Duration: 2500 ticks")
    print("  Agents: 10 learning-enabled")
    print("  World: 30x30")
    print("  Resources: Config-based")
    print("\nExpected Performance:")
    print("  - Survival: ~2000 ticks")
    print("  - EAT success: ~6%")
    print("  - WAIT actions: ~40%")
    print("  - Movement success: ~79%")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    
    # Initialize logger
    logger = AgentLogger(output_dir="data/logs", log_every_n_ticks=1)
    Agent.logger = logger    # Create world with config dimensions
    print("\nInitializing world...")
    world = World(
        width=config['world']['width'],
        height=config['world']['height']
    )
    
    # Set reproduction config from YAML
    if 'reproduction' in config:
        world.reproduction_config = config['reproduction']
    
    populate_world_with_resources(world, config)
    
    # Create agents with learning enabled and config settings
    print("Creating 10 learning-enabled agents with training_easy config...")
    agents = []
    trait_config = create_default_trait_config()
    agent_config = config['agents']
    
    for i in range(10):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        
        genome = Genome.random(weight_count=2744, trait_config=trait_config)
        agent = Agent(
            x, y, 
            genome=genome,
            max_energy=agent_config['max_energy'],
            max_age=agent_config['max_age'],
            inventory_size=agent_config['inventory_size'],
            metabolism_rate=agent_config['metabolism_rate']
        )
        agent.energy = 700.0  # High starting energy
        agent.enable_learning(
            learning_rate=config['learning']['learning_rate'],
            discount_factor=config['learning']['discount_factor'],
            batch_size=config['learning']['batch_size'],
            buffer_capacity=config['learning']['buffer_capacity']
        )
        agents.append(agent)
        world.add_agent(agent)
        print(f"  Agent {i}: pos=({x},{y}), energy={agent.energy:.0f}, metabolism={agent.metabolism_rate:.3f}")
    
    # Run simulation
    max_ticks = 2500    
    print(f"\nRunning simulation for {max_ticks} ticks...")
    print("This may take several minutes...\n")
    print("=" * 70)
    
    all_dead = False
    
    while world.tick < max_ticks and not all_dead:
        # Update world (this internally updates all agents AND increments world.tick)
        world.update()
        
        # Check alive agents
        alive_agents = [a for a in agents if a.alive]
        
        # Check if all dead
        if len(alive_agents) == 0:
            print(f"Tick {world.tick:4d}: ALL AGENTS DIED")
            all_dead = True
            break
        
        # Status update every 100 ticks
        if world.tick % 100 == 0:
            avg_energy = sum(a.energy for a in alive_agents) / len(alive_agents)
            avg_fitness = sum(a.fitness for a in alive_agents) / len(alive_agents)
            avg_age = sum(a.age for a in alive_agents) / len(alive_agents)
            avg_exp = sum(len(a.learner.replay_buffer) for a in alive_agents if a.learner) / len(alive_agents)
            
            print(f"Tick {world.tick:4d}: {len(alive_agents):2d} alive | "
                  f"Avg Energy: {avg_energy:5.1f} | "
                  f"Avg Fitness: {avg_fitness:6.1f} | "
                  f"Avg Age: {avg_age:5.1f} | "
                  f"Avg Exp: {avg_exp:5.1f}")
    
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    
    # Final statistics
    alive_agents = [a for a in agents if a.alive]
    dead_agents = [a for a in agents if not a.alive]
    
    print(f"\n📊 FINAL STATISTICS:")
    print(f"  Total ticks: {world.tick}")
    print(f"  Survivors: {len(alive_agents)}/{len(agents)}")
    print(f"  Deaths: {len(dead_agents)}")
    
    if alive_agents:
        print(f"\n✅ SURVIVORS:")
        for agent in sorted(alive_agents, key=lambda a: a.fitness, reverse=True):
            exp_count = len(agent.learner.replay_buffer) if agent.learner else 0
            print(f"  Agent {agent.id}: Age={agent.age:4d}, Energy={agent.energy:5.1f}, "
                  f"Fitness={agent.fitness:6.1f}, Exp={exp_count:4d}")
        
        print(f"\n  Average survivor age: {sum(a.age for a in alive_agents) / len(alive_agents):.1f}")
        print(f"  Average survivor energy: {sum(a.energy for a in alive_agents) / len(alive_agents):.1f}")
        print(f"  Average survivor fitness: {sum(a.fitness for a in alive_agents) / len(alive_agents):.1f}")
    
    if dead_agents:
        avg_death_age = sum(a.age for a in dead_agents) / len(dead_agents)
        print(f"\n💀 DEAD AGENTS:")
        print(f"  Average death age: {avg_death_age:.1f}")
    
    print(f"\n✅ Logs saved:")
    print(f"  - {logger.action_file}")
    print(f"  - {logger.state_file}")
    
    # Performance assessment
    print("\n" + "=" * 70)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 70)
    
    survival_time = world.tick
    expected_survival = 2000
    
    print(f"\n🎯 Survival Time:")
    print(f"  Actual: {survival_time} ticks")
    print(f"  Expected: ~{expected_survival} ticks")
    if survival_time >= expected_survival * 0.8:
        print(f"  Status: ✅ EXCELLENT (>80% of target)")
    elif survival_time >= expected_survival * 0.6:
        print(f"  Status: ⚠️  ACCEPTABLE (>60% of target)")
    else:
        print(f"  Status: ❌ BELOW EXPECTED")
    
    print(f"\n💡 Next Steps:")
    print(f"  1. Run: python analyze_v3_1.py")
    print(f"  2. Compare WAIT %, EAT success, and movement patterns")
    print(f"  3. If results match expectations, system is production-ready!")
    
    print("\n" + "=" * 70)
    print("📊 To analyze results, run:")
    print("  python analyze_v3_1.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
