"""
Test multi-generation evolution with Lamarckian learning.

This test runs agents over multiple generations, allowing them to:
1. Learn during their lifetime (RL)
2. Pass learned weights to offspring (Lamarckian evolution)
3. Evolve via mutation and selection

Author: Karan Vasa
Date: November 16, 2025
"""

import random
import yaml
from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent, PlantComponent
from agents import Agent, Brain, Genome, create_default_trait_config
from agents.evolution import EvolutionConfig, next_generation
from utils.agents import calculate_fitness, EvolutionStats


def load_config(config_path="config/training_easy.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def populate_world_with_resources(world, config):
    """Add initial resources based on config."""
    berry_calories = config["resources"]["berry_calories"]
    berry_count = 400
    plant_count = 150
    seed_count = 80

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


def create_initial_population(config, population_size=10):
    """Create initial population with random genomes."""
    trait_config = create_default_trait_config()
    weight_count = Brain.calculate_weight_count()
    agent_config = config["agents"]

    population = []
    for i in range(population_size):
        genome = Genome.random(weight_count=weight_count, trait_config=trait_config)
        agent = Agent(
            x=0,
            y=0,  # Will be repositioned
            genome=genome,
            max_energy=agent_config["max_energy"],
            max_age=agent_config["max_age"],
            inventory_size=agent_config["inventory_size"],
            metabolism_rate=agent_config["metabolism_rate"],
        )
        agent.energy = 700.0
        agent.enable_learning(
            learning_rate=config["learning"]["learning_rate"],
            discount_factor=config["learning"]["discount_factor"],
            batch_size=config["learning"]["batch_size"],
            buffer_capacity=config["learning"]["buffer_capacity"],
        )
        population.append(agent)

    return population


def run_generation(population, config, max_ticks=1000):
    """
    Run one generation - all agents live out their lives.

    Args:
        population: List of agents
        config: Configuration dict
        max_ticks: Maximum ticks per generation

    Returns:
        Population with updated fitness scores
    """
    # Create fresh world for this generation
    world = World(width=config["world"]["width"], height=config["world"]["height"])
    populate_world_with_resources(world, config)

    # Position agents randomly
    for agent in population:
        agent.x = random.randint(0, world.width - 1)
        agent.y = random.randint(0, world.height - 1)
        world.add_agent(agent)

    # Run simulation
    for tick in range(max_ticks):
        world.update()

        alive_agents = [a for a in population if a.alive]
        if len(alive_agents) == 0:
            break

    return population


def main():
    """Run multi-generation evolution experiment."""
    print("=" * 70)
    print("MULTI-GENERATION EVOLUTION TEST")
    print("=" * 70)
    print("\nThis test demonstrates Lamarckian evolution:")
    print("  1. Agents learn during their lifetime (RL)")
    print("  2. Offspring inherit trained weights from parents")
    print("  3. Population evolves via selection and mutation")
    print("=" * 70)

    # Configuration
    config = load_config()
    evo_config = EvolutionConfig(
        population_size=10,
        elite_count=2,
        parent_count=3,
        mutation_rate=0.7,
        mutation_std=0.02,
    )
    stats = EvolutionStats()

    # Create initial population
    print("\nCreating initial population...")
    population = create_initial_population(config, evo_config.population_size)
    print(f"  Population size: {len(population)}")
    print(f"  Elite count: {evo_config.elite_count}")
    print(f"  Parent count: {evo_config.parent_count}")
    print(f"  Mutation rate: {evo_config.mutation_rate}")

    # Run multiple generations
    num_generations = 5
    max_ticks_per_gen = 1000

    print(f"\nRunning {num_generations} generations...")
    print(f"  Max ticks per generation: {max_ticks_per_gen}")
    print("=" * 70)

    for gen in range(num_generations):
        print(f"\n🔄 GENERATION {gen}")
        print("-" * 70)

        # Run generation
        population = run_generation(population, config, max_ticks_per_gen)

        # Calculate statistics
        fitnesses = [calculate_fitness(a) for a in population]
        ages = [a.age for a in population]
        alive_count = sum(1 for a in population if a.alive)

        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        avg_age = sum(ages) / len(ages)

        print(f"Results:")
        print(f"  Alive at end: {alive_count}/{len(population)}")
        print(f"  Best fitness: {best_fitness:.1f}")
        print(f"  Avg fitness: {avg_fitness:.1f}")
        print(f"  Avg age: {avg_age:.1f}")

        # Record stats
        stats.record_generation(population)
        # Create next generation (except on last generation)
        if gen < num_generations - 1:
            print(f"\n  Creating generation {gen+1}...")
            population = next_generation(population, evo_config, stats)
            print(f"  - {evo_config.elite_count} elites preserved")
            print(f"  - {len(population) - evo_config.elite_count} offspring created")

    # Print summary
    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    stats.print_summary()

    print("\n📊 Analysis:")
    if stats.best_fitness_history[-1] > stats.best_fitness_history[0]:
        improvement = stats.best_fitness_history[-1] - stats.best_fitness_history[0]
        print(f"  ✅ Evolution successful! Fitness improved by {improvement:.1f}")
        print(f"  ✅ Lamarckian learning is passing knowledge to offspring")
    else:
        print(f"  ⚠️  No improvement detected")
        print(f"  Consider: tuning mutation rate, running more generations")

    print("\n💡 Next steps:")
    print("  - Run more generations to see continued improvement")
    print("  - Adjust mutation_std for faster/slower evolution")
    print("  - Implement curriculum learning for harder challenges")
    print("  - Add crossover mating between two parents")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
