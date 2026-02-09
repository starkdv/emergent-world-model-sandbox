"""
Evolution module for agents.

Implements fission-based reproduction, mutation, and selection
for Lamarckian evolution combined with reinforcement learning.

Based on agent_training_world_design.md Section 6.

Author: Karan Vasa
Date: November 16, 2025
"""

import random
import copy
from typing import List, Tuple, Optional
import numpy as np

if __name__ != "__main__":
    from agents.agent import Agent
    from agents.genome import Genome


class EvolutionConfig:
    """Configuration for evolution parameters."""
    
    def __init__(
        self,
        population_size: int = 10,
        elite_count: int = 2,
        parent_count: int = 3,
        mutation_rate: float = 0.7,  # Probability of mutation
        mutation_std: float = 0.02,  # Gaussian noise std dev
        selection_method: str = "fitness"  # "fitness" or "tournament"
    ):
        """
        Initialize evolution config.
        
        Args:
            population_size: Number of agents per generation
            elite_count: Number of top agents to preserve unchanged
            parent_count: Number of top agents to use as parents
            mutation_rate: Probability of mutating an offspring
            mutation_std: Standard deviation of Gaussian mutation noise
            selection_method: How to select parents
        """
        self.population_size = population_size
        self.elite_count = elite_count
        self.parent_count = parent_count
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.selection_method = selection_method
        
        # Validate
        assert elite_count < parent_count <= population_size
        assert 0.0 <= mutation_rate <= 1.0
        assert mutation_std > 0.0


def calculate_fitness(agent: Agent) -> float:
    """
    Calculate agent fitness for selection.
    
    Fitness formula from design guide:
        fitness = steps_survived + 5.0 * food_eaten
    
    Args:
        agent: The agent to evaluate
        
    Returns:
        Fitness score (higher is better)
    """
    # Use agent's existing fitness property, which tracks:
    # - 0.1 per step survived
    # - rewards for eating, planting, etc.
    
    # Alternative explicit calculation:
    # food_eaten = getattr(agent, 'food_eaten_count', 0)
    # return agent.age + 5.0 * food_eaten
    
    return agent.fitness


def select_parents(
    population: List[Agent],
    config: EvolutionConfig
) -> List[Agent]:
    """
    Select top-performing agents as parents.
    
    Args:
        population: Current generation of agents
        config: Evolution configuration
        
    Returns:
        List of parent agents sorted by fitness (descending)
    """
    # Sort by fitness
    sorted_agents = sorted(
        population,
        key=calculate_fitness,
        reverse=True
    )
    
    # Return top K parents
    return sorted_agents[:config.parent_count]


def clone_agent(
    parent: Agent,
    mutate: bool = False,
    mutation_std: float = 0.02
) -> Agent:
    """
    Create offspring by cloning parent agent.
    
    Implements Lamarckian evolution: offspring inherit
    the trained weights from parent's end-of-life state.
    
    Args:
        parent: Parent agent to clone
        mutate: Whether to apply mutation
        mutation_std: Standard deviation of mutation noise
        
    Returns:
        New agent with cloned/mutated weights
    """
    # Create a copy of the parent genome with incremented generation
    child_genome = copy.deepcopy(parent.genome)
    child_genome.generation = parent.genome.generation + 1  # INCREMENT GENERATION!
    child_genome.parent_ids = (parent.genome.lineage_id, parent.genome.lineage_id)  # Track parent lineage (asexual)
    
    # Create new agent with the new genome
    child = Agent(
        x=0,  # Will be repositioned
        y=0,
        genome=child_genome,
        max_energy=parent.max_energy,
        max_age=parent.max_age,
        inventory_size=parent.inventory_size,
        metabolism_rate=parent.metabolism_rate
    )
    
    # LAMARCKIAN INHERITANCE: Copy trained weights from parent
    # This is the key to passing learned knowledge to offspring
    child.brain.weights = [w.copy() for w in parent.brain.weights]
    child.brain.biases = [b.copy() for b in parent.brain.biases]
    
    # Apply mutation if requested
    if mutate:
        mutate_weights(child.brain, mutation_std)
    
    # Reset life-specific state
    child.age = 0
    child.energy = parent.max_energy * 0.7  # Start with 70% energy
    child.alive = True
    child.fitness = 0.0
    
    return child


def mutate_weights(brain, std: float = 0.02):
    """
    Apply Gaussian noise mutation to brain weights.
    
    Mutates weights and biases in-place.
    
    Args:
        brain: The Brain object to mutate
        std: Standard deviation of Gaussian noise
    """
    # Mutate all weight matrices
    for i in range(len(brain.weights)):
        noise = np.random.normal(0.0, std, size=brain.weights[i].shape)
        brain.weights[i] += noise
    
    # Mutate all bias vectors
    for i in range(len(brain.biases)):
        noise = np.random.normal(0.0, std, size=brain.biases[i].shape)
        brain.biases[i] += noise


def next_generation(
    population: List[Agent],
    config: EvolutionConfig,
    stats: 'EvolutionStats'
) -> List[Agent]:
    """
    Create next generation from current population.
    
    Uses fission-based reproduction:
    1. Select top-K parents by fitness
    2. Keep top-E as elites (no mutation)
    3. Fill remaining slots with mutated offspring
    
    Args:
        population: Current generation
        config: Evolution configuration
        stats: Evolution statistics tracker
        
    Returns:
        New population for next generation
    """
    # Select parents
    parents = select_parents(population, config)
    
    # Create new population
    new_population = []
    
    # Step 1: Add elites (best agents, no mutation)
    elites = parents[:config.elite_count]
    for elite in elites:
        child = clone_agent(elite, mutate=False)
        new_population.append(child)
    
    # Step 2: Fill rest with offspring from parents
    while len(new_population) < config.population_size:
        # Randomly select a parent
        parent = random.choice(parents)
        
        # Decide if this child should be mutated
        should_mutate = random.random() < config.mutation_rate
        
        # Adjust mutation std based on diversity
        mutation_std = adaptive_mutation_std(stats, base_std=config.mutation_std)
        
        # Create offspring
        child = clone_agent(
            parent,
            mutate=should_mutate,
            mutation_std=mutation_std
        )
        
        new_population.append(child)
    
    return new_population


class EvolutionStats:
    """Track evolution statistics across generations."""
    
    def __init__(self):
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.avg_age_history = []
        self.survival_rate_history = []
        self.diversity_history = []  # Track genetic diversity
        
    def calculate_diversity(self, population: List) -> float:
        """
        Calculate genetic diversity as average pairwise distance.
        
        Higher values indicate more diverse population.
        Lower values indicate convergence (similar genomes).
        """
        if len(population) < 2:
            return 0.0
        
        # Sample weights from first layer for efficiency
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):                # Calculate L2 distance between weight matrices
                w1 = population[i].brain.weights[0].flatten()
                w2 = population[j].brain.weights[0].flatten()
                dist = np.linalg.norm(w1 - w2)
                distances.append(dist)
        
        return float(np.mean(distances)) if distances else 0.0
    
    def record_generation(self, population: List):
        """Record statistics from a generation."""
        fitnesses = [calculate_fitness(agent) for agent in population]
        ages = [agent.age for agent in population]
        alive = sum(1 for agent in population if agent.alive)
        diversity = self.calculate_diversity(population)
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        self.avg_age_history.append(sum(ages) / len(ages))
        self.survival_rate_history.append(alive / len(population))
        self.diversity_history.append(diversity)
        self.generation += 1
    
    def print_summary(self):
        """Print evolution summary."""
        print("\n" + "=" * 70)
        print("EVOLUTION SUMMARY")
        print("=" * 70)
        print(f"Generations completed: {self.generation}")
        
        if self.best_fitness_history:
            print(f"\nBest fitness:")
            print(f"  Generation 0: {self.best_fitness_history[0]:.1f}")
            print(f"  Generation {self.generation-1}: {self.best_fitness_history[-1]:.1f}")
            improvement = self.best_fitness_history[-1] - self.best_fitness_history[0]
            print(f"  Improvement: {improvement:+.1f} ({improvement/self.best_fitness_history[0]*100:+.1f}%)")
        
        if self.avg_fitness_history:
            print(f"\nAverage fitness:")
            print(f"  Generation 0: {self.avg_fitness_history[0]:.1f}")
            print(f"  Generation {self.generation-1}: {self.avg_fitness_history[-1]:.1f}")
            improvement = self.avg_fitness_history[-1] - self.avg_fitness_history[0]
            print(f"  Improvement: {improvement:+.1f}")
        
        if self.diversity_history:
            print(f"\nGenetic diversity:")
            print(f"  Generation 0: {self.diversity_history[0]:.2f}")
            print(f"  Generation {self.generation-1}: {self.diversity_history[-1]:.2f}")
            change = self.diversity_history[-1] - self.diversity_history[0]
            print(f"  Change: {change:+.2f}")
            if self.diversity_history[-1] < 2.0:
                print(f"  ⚠️  Low diversity - population converging (consider increasing mutation)")
            elif self.diversity_history[-1] < 5.0:
                print(f"  ⚠️  Moderate diversity - monitor for convergence")
            else:
                print(f"  ✅  Healthy diversity maintained")
        
        if self.survival_rate_history:
            avg_survival = sum(self.survival_rate_history) / len(self.survival_rate_history)
            print(f"\nAverage survival rate: {avg_survival*100:.1f}%")
        
        print("=" * 70)


def adaptive_mutation_std(
    stats: 'EvolutionStats',
    base_std: float = 0.02,
    min_std: float = 0.01,
    max_std: float = 0.05
) -> float:
    """
    Adjust mutation standard deviation based on population diversity.
    
    When diversity is low, increase mutation to prevent convergence.
    When diversity is high, decrease mutation to preserve good solutions.
    
    Args:
        stats: Evolution statistics tracker
        base_std: Base mutation standard deviation
        min_std: Minimum mutation std
        max_std: Maximum mutation std
        
    Returns:
        Adjusted mutation standard deviation
    """
    if not stats.diversity_history:
        return base_std
    
    current_diversity = stats.diversity_history[-1]
    
    # Low diversity threshold (adjust based on your observations)
    low_diversity_threshold = 2.0
    high_diversity_threshold = 20.0
    
    if current_diversity < low_diversity_threshold:
        # Increase mutation when diversity is low
        return max_std
    elif current_diversity > high_diversity_threshold:
        # Decrease mutation when diversity is high
        return min_std
    else:
        # Scale linearly between thresholds
        ratio = (current_diversity - low_diversity_threshold) / (high_diversity_threshold - low_diversity_threshold)
        return max_std - (max_std - min_std) * ratio


# Example usage
if __name__ == "__main__":
    print("Evolution module loaded successfully!")
    print("\nKey functions:")
    print("  - calculate_fitness(agent) -> float")
    print("  - select_parents(population, config) -> List[Agent]")
    print("  - clone_agent(parent, mutate=False) -> Agent")
    print("  - mutate_weights(brain, std=0.02)")
    print("  - next_generation(population, config) -> List[Agent]")
    print("\nUsage example:")
    print("  config = EvolutionConfig()")
    print("  population = run_generation(population, world, max_ticks)")
    print("  population = next_generation(population, config, stats)")
