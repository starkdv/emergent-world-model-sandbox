"""
Utility functions for evolution system.

Contains helper functions and statistics tracking for the evolution module.

Author: Karan Vasa
Date: November 16, 2025
"""

from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from agents.agent import Agent


def calculate_fitness(agent: 'Agent') -> float:
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
        
        # Calculate pairwise distances between genome weights
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Calculate L2 distance between genome weight arrays
                w1 = population[i].genome.weights
                w2 = population[j].genome.weights
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
