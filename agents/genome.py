"""
Genome representation and genetic operations.

The genome encodes:
- Neural network weights (brain structure)
- Phenotypic traits (metabolism, vision, etc.)
- Lineage information for tracking evolution

Author: Karan Vasa
Date: November 14, 2025
"""

import numpy as np
from typing import Optional


class Genome:
    """
    Genetic information for an agent.

    The genome contains:
    - weights: Flattened neural network parameters
    - traits: Phenotypic traits (metabolism_rate, vision_radius, etc.)
    - lineage_id: Identifier for evolutionary tracking

    Attributes:
        weights (np.ndarray): Neural network weights (flattened)
        traits (dict[str, float]): Phenotypic trait values
        lineage_id (int): Lineage identifier
        generation (int): Generation number
        parent_ids (tuple[int, int]): IDs of parent genomes
    """

    _next_lineage_id = 0

    def __init__(
        self,
        weights: np.ndarray,
        traits: dict[str, float],
        lineage_id: Optional[int] = None,
        generation: int = 0,
        parent_ids: tuple[int, int] = (-1, -1),
    ):
        """
        Initialize genome.

        Args:
            weights: Flattened neural network weights
            traits: Dictionary of phenotypic traits
            lineage_id: Lineage identifier (auto-assigned if None)
            generation: Generation number
            parent_ids: IDs of parent genomes
        """
        self.weights = weights.copy()
        self.traits = traits.copy()

        if lineage_id is None:
            self.lineage_id = Genome._next_lineage_id
            Genome._next_lineage_id += 1
        else:
            self.lineage_id = lineage_id

        self.generation = generation
        self.parent_ids = parent_ids

    @staticmethod
    def random(
        weight_count: int,
        trait_config: dict[str, tuple[float, float]],
        weight_init_std: float = 0.5,
    ) -> "Genome":
        """
        Create a random genome.

        Args:
            weight_count: Number of neural network weights
            trait_config: Dict mapping trait names to (min, max) ranges
            weight_init_std: Standard deviation for weight initialization

        Returns:
            New random genome
        """
        # Random weights (small values near zero)
        weights = np.random.randn(weight_count) * weight_init_std

        # Random traits within ranges
        traits = {}
        for trait_name, (min_val, max_val) in trait_config.items():
            traits[trait_name] = np.random.uniform(min_val, max_val)

        return Genome(weights, traits)

    @staticmethod
    def mate(
        parent_a: "Genome",
        parent_b: "Genome",
        crossover_method: str = "uniform",
        mutation_rate: float = 0.01,
        mutation_std: float = 0.1,
        trait_mutation_std: float = 0.05,
    ) -> "Genome":
        """
        Create offspring genome from two parents.

        Args:
            parent_a: First parent genome
            parent_b: Second parent genome
            crossover_method: 'uniform', 'one_point', or 'blend'
            mutation_rate: Probability of mutating each weight
            mutation_std: Standard deviation of weight mutations
            trait_mutation_std: Standard deviation of trait mutations

        Returns:
            New child genome
        """
        # Crossover weights
        if crossover_method == "uniform":
            child_weights = Genome._uniform_crossover(
                parent_a.weights, parent_b.weights
            )
        elif crossover_method == "one_point":
            child_weights = Genome._one_point_crossover(
                parent_a.weights, parent_b.weights
            )
        elif crossover_method == "blend":
            child_weights = Genome._blend_crossover(parent_a.weights, parent_b.weights)
        else:
            raise ValueError(f"Unknown crossover method: {crossover_method}")

        # Mutate weights
        child_weights = Genome._mutate_weights(
            child_weights, mutation_rate, mutation_std
        )

        # Crossover and mutate traits
        child_traits = Genome._crossover_traits(
            parent_a.traits, parent_b.traits, trait_mutation_std
        )

        # Create child genome
        return Genome(
            weights=child_weights,
            traits=child_traits,
            generation=max(parent_a.generation, parent_b.generation) + 1,
            parent_ids=(parent_a.lineage_id, parent_b.lineage_id),
        )

    @staticmethod
    def _uniform_crossover(weights_a: np.ndarray, weights_b: np.ndarray) -> np.ndarray:
        """
        Uniform crossover: each gene randomly from either parent.

        Args:
            weights_a: Parent A weights
            weights_b: Parent B weights

        Returns:
            Child weights
        """
        mask = np.random.rand(len(weights_a)) < 0.5
        return np.where(mask, weights_a, weights_b)

    @staticmethod
    def _one_point_crossover(
        weights_a: np.ndarray, weights_b: np.ndarray
    ) -> np.ndarray:
        """
        One-point crossover: split at random point.

        Args:
            weights_a: Parent A weights
            weights_b: Parent B weights

        Returns:
            Child weights
        """
        point = np.random.randint(0, len(weights_a))
        child = weights_a.copy()
        child[point:] = weights_b[point:]
        return child

    @staticmethod
    def _blend_crossover(
        weights_a: np.ndarray, weights_b: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Blend crossover: weighted average of parents.

        Args:
            weights_a: Parent A weights
            weights_b: Parent B weights
            alpha: Blend factor (0.5 = average)

        Returns:
            Child weights
        """
        return alpha * weights_a + (1 - alpha) * weights_b

    @staticmethod
    def _mutate_weights(
        weights: np.ndarray, mutation_rate: float, mutation_std: float
    ) -> np.ndarray:
        """
        Mutate weights with Gaussian noise.

        Args:
            weights: Weight array to mutate
            mutation_rate: Probability of mutating each weight
            mutation_std: Standard deviation of mutations

        Returns:
            Mutated weights
        """
        mutation_mask = np.random.rand(len(weights)) < mutation_rate
        mutations = np.random.randn(len(weights)) * mutation_std
        return weights + mutation_mask * mutations

    @staticmethod
    def _crossover_traits(
        traits_a: dict[str, float], traits_b: dict[str, float], mutation_std: float
    ) -> dict[str, float]:
        """
        Crossover and mutate traits.

        Args:
            traits_a: Parent A traits
            traits_b: Parent B traits
            mutation_std: Standard deviation of trait mutations

        Returns:
            Child traits
        """
        child_traits = {}

        for trait_name in traits_a.keys():
            # Average of parents
            avg_value = (traits_a[trait_name] + traits_b[trait_name]) / 2

            # Add mutation
            mutation = np.random.randn() * mutation_std
            child_traits[trait_name] = avg_value + mutation

        # Clamp traits to reasonable ranges
        child_traits["metabolism_rate"] = np.clip(
            child_traits.get("metabolism_rate", 1.0), 0.5, 2.0
        )
        child_traits["vision_radius"] = np.clip(
            child_traits.get("vision_radius", 5.0), 2.0, 10.0
        )

        return child_traits

    def copy(self) -> "Genome":
        """
        Create a copy of this genome.

        Returns:
            Genome copy
        """
        return Genome(
            weights=self.weights.copy(),
            traits=self.traits.copy(),
            lineage_id=self.lineage_id,
            generation=self.generation,
            parent_ids=self.parent_ids,
        )

    def __repr__(self) -> str:
        return (
            f"Genome(lineage={self.lineage_id}, gen={self.generation}, "
            f"weights={len(self.weights)}, traits={list(self.traits.keys())})"
        )


def create_default_trait_config() -> dict[str, tuple[float, float]]:
    """
    Get default trait configuration.

    Returns:
        Dictionary mapping trait names to (min, max) ranges
    """
    return {
        "metabolism_rate": (0.5, 2.0),  # Energy consumption multiplier
        "vision_radius": (2.0, 10.0),  # How far agent can see
        "movement_speed": (0.5, 1.5),  # Movement speed multiplier
    }
