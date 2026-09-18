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
        # Birth-time fingerprint (Brain v4 §4.7). Computed once here; the
        # Lamarckian write-back changes `weights` during a lifetime but the
        # fingerprint deliberately does NOT follow it — relatedness is about
        # inheritance, not about what an individual learned.
        self.fingerprint = genome_fingerprint(self.weights)

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

        # A few tensors carry structured priors that random initialisation
        # would destroy — a random slow.rho collapses every v4 time constant
        # to ~2 ticks, a random drive.lam makes the reward arbitrary. The
        # active genome layout (set at startup alongside the observation
        # spec) restores them; specs without priors are a no-op.
        from agents.brain.spec import get_active_param_spec

        active = get_active_param_spec()
        if active is not None and active.count() == weight_count:
            active.apply_inits(weights)

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
        # Niche traits (body_size, visual_acuity, diet) when the ecology is on
        from agents import ecology

        ecology.clamp_traits(child_traits)

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


# ---------------------------------------------------------------------------
# Genetic fingerprint — cheap graded relatedness (Brain v3.6 §9.4, shipped
# as part of Brain v4 §4.7)
# ---------------------------------------------------------------------------

# Dimensionality of the fingerprint. Small enough that comparing two agents
# is one k-dim dot product per tick, against the ONE nearest neighbour that
# perception has already located for the proximity feature.
FINGERPRINT_DIM = 8

# Cache of fixed projections, one per genome length. Deterministic (seeded),
# so a fingerprint is stable across a run and across runs.
_PROJECTIONS: dict[int, np.ndarray] = {}


def _projection(weight_count: int) -> np.ndarray:
    """Fixed (k, W) random projection for genomes of this length."""
    proj = _PROJECTIONS.get(weight_count)
    if proj is None:
        rng = np.random.default_rng(20260918)
        proj = rng.standard_normal((FINGERPRINT_DIM, weight_count)).astype(np.float32)
        _PROJECTIONS[weight_count] = proj
    return proj


def genome_fingerprint(weights: np.ndarray) -> np.ndarray:
    """
    Birth-time genetic fingerprint: ``f = normalize(P w)``.

    Comparing two full genomes per agent per tick is far too expensive
    (thousands of weights x N^2 neighbours). A fixed random projection to
    ``FINGERPRINT_DIM`` preserves cosine similarity in expectation
    (Johnson-Lindenstrauss), so relatedness costs one k-dim dot product.

    Args:
        weights: Flat genome weight vector

    Returns:
        Unit-norm (FINGERPRINT_DIM,) fingerprint
    """
    w = np.asarray(weights, dtype=np.float32).ravel()
    f = _projection(w.shape[0]) @ w
    norm = float(np.linalg.norm(f))
    if norm < 1e-8:
        return np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    return (f / norm).astype(np.float32)


def kin_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Graded relatedness in [0, 1] from two fingerprints.

    ``kin = (cos + 1) / 2``: clones score ~1, unrelated lineages project to
    near-orthogonal fingerprints and score ~0.5. It is a smooth genetic
    distance, not a same-lineage bit — which is what kin-selection theory
    actually wants.

    Args:
        a: Fingerprint of one agent
        b: Fingerprint of the other

    Returns:
        Similarity in [0, 1] (0.5 when either fingerprint is missing)
    """
    if a is None or b is None or a.shape != b.shape:
        return 0.5
    return float(np.clip(0.5 * (1.0 + float(np.dot(a, b))), 0.0, 1.0))


def create_default_trait_config() -> dict[str, tuple[float, float]]:
    """
    Get default trait configuration.

    Returns:
        Dictionary mapping trait names to (min, max) ranges
    """
    from agents import ecology

    config = {
        "metabolism_rate": (0.5, 2.0),  # Energy consumption multiplier
        "vision_radius": (2.0, 10.0),  # How far agent can see
        "movement_speed": (0.5, 1.5),  # Movement speed multiplier
    }
    # When the ecology is enabled, add the traits that actually cost something:
    # body_size, visual_acuity and diet (see agents/ecology.py). The three
    # above are legacy — movement_speed is read only by the renderer's debug
    # text, vision_radius is never read by perception at all, and
    # metabolism_rate is a free lunch with no downside anywhere.
    return ecology.extend_trait_config(config)
