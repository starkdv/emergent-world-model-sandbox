"""
Evolution module for agents.

Implements fission-based reproduction, mutation, and selection
for Lamarckian evolution combined with reinforcement learning.

Core classes:
- EvolutionConfig: Evolution parameters
- Clone and mutation functions

Utility functions moved to utils.agents.evolution_utils:
- EvolutionStats: Statistics tracking
- calculate_fitness: Fitness calculation
- adaptive_mutation_std: Adaptive mutation

Author: Karan Vasa
Date: November 16, 2025
"""

import random
import copy
from typing import List, Tuple, Optional
import numpy as np

# Import utility functions
from utils.agents import calculate_fitness, adaptive_mutation_std
from utils.agents.evolution_utils import EvolutionStats

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
        selection_method: str = "fitness",  # "fitness" or "tournament"
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


def select_parents(population: List[Agent], config: EvolutionConfig) -> List[Agent]:
    """
    Select top-performing agents as parents.

    Args:
        population: Current generation of agents
        config: Evolution configuration

    Returns:
        List of parent agents sorted by fitness (descending)
    """
    # Sort by fitness
    sorted_agents = sorted(population, key=calculate_fitness, reverse=True)

    # Return top K parents
    return sorted_agents[: config.parent_count]


def clone_agent(
    parent: Agent, mutate: bool = False, mutation_std: float = 0.02
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
    child_genome.parent_ids = (
        parent.genome.lineage_id,
        parent.genome.lineage_id,
    )  # Track parent lineage (asexual)

    # Create new agent with the new genome
    child = Agent(
        x=0,  # Will be repositioned
        y=0,
        genome=child_genome,
        max_energy=parent.max_energy,
        max_age=parent.max_age,
        inventory_size=parent.inventory_size,
        metabolism_rate=parent.metabolism_rate,
    )

    # LAMARCKIAN INHERITANCE: The genome already contains the trained weights
    # from the parent's end-of-life state (updated via AgentLearner._sync_genome_weights)
    # No need to copy brain.weights/biases anymore - they're automatically loaded from genome

    # Apply mutation if requested
    if mutate:
        mutate_genome_weights(child.genome, mutation_std)
        # Recreate brain with mutated genome to apply changes
        child.brain = child.brain.__class__(
            child.genome,
            input_size=child.brain.input_size,
            encoder_layers=child.brain.encoder_layers,
            gru_hidden_size=child.brain.gru_hidden_size,
            output_size=child.brain.output_size,
        )
        # Reset hidden state
        child.h = child.brain.initial_state()

    # Reset life-specific state
    child.age = 0
    child.energy = parent.max_energy * 0.7  # Start with 70% energy
    child.alive = True
    child.fitness = 0.0

    return child


def mutate_genome_weights(genome: Genome, std: float = 0.02):
    """
    Apply Gaussian noise mutation to genome weights.

    Mutates the flat weight array in-place.
    Works with Brain v2 architecture where all parameters are stored
    in genome.weights as a flat array.

    Args:
        genome: The Genome object containing weights to mutate
        std: Standard deviation of Gaussian noise
    """
    # Apply Gaussian noise to all weights
    noise = np.random.normal(0.0, std, size=genome.weights.shape)
    genome.weights += noise


def next_generation(
    population: List[Agent], config: EvolutionConfig, stats: "EvolutionStats"
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
    elites = parents[: config.elite_count]
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
        child = clone_agent(parent, mutate=should_mutate, mutation_std=mutation_std)

        new_population.append(child)

    return new_population
