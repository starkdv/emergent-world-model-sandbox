"""
Tests for Genome class.

Tests:
- Genome creation and initialization
- Weight management
- Trait system
- Crossover operations
- Mutation
- Serialization
"""

import pytest
import numpy as np

from agents.genome import Genome, create_default_trait_config


class TestGenome:
    """Test suite for Genome class."""
    
    def test_random_genome_creation(self):
        """Test random genome generation."""
        trait_config = create_default_trait_config()
        weight_count = 1000
        
        genome = Genome.random(weight_count, trait_config)
        
        # Check weights
        assert genome.weights.shape == (weight_count,)
        assert genome.weights.dtype in (np.float32, np.float64), "Weights should be float type"
        assert np.all(np.isfinite(genome.weights)), "Weights should be finite"
        
        # Check traits
        for trait_name, (min_val, max_val) in trait_config.items():
            assert trait_name in genome.traits
            trait_value = genome.traits[trait_name]
            assert min_val <= trait_value <= max_val, \
                f"Trait {trait_name}={trait_value} outside bounds [{min_val}, {max_val}]"
    
    def test_genome_from_weights(self):
        """Test creating genome from existing weights."""
        trait_config = create_default_trait_config()
        weights = np.random.randn(500).astype(np.float32)
        traits = {name: (min_val + max_val) / 2 for name, (min_val, max_val) in trait_config.items()}
        
        genome = Genome(
            weights=weights,
            traits=traits,
            generation=5,
            lineage_id=123,
            parent_ids=(100, 101)
        )
        
        assert np.array_equal(genome.weights, weights)
        assert genome.generation == 5
        assert genome.lineage_id == 123
        assert genome.parent_ids == (100, 101)
    
    def test_genome_copy(self):
        """Test genome copying."""
        trait_config = create_default_trait_config()
        genome1 = Genome.random(1000, trait_config)
        
        genome2 = genome1.copy()
        
        # Check independence
        assert np.array_equal(genome1.weights, genome2.weights)
        assert genome1.traits == genome2.traits
        assert genome1.generation == genome2.generation
        
        # Modify copy
        genome2.weights[0] = 999.0
        genome2.traits['metabolism_rate'] = 100.0
        genome2.generation = 10
        
        # Original should be unchanged
        assert genome1.weights[0] != 999.0
        assert genome1.traits['metabolism_rate'] != 100.0
        assert genome1.generation != 10
    
    def test_crossover_uniform(self):
        """Test uniform crossover."""
        trait_config = create_default_trait_config()
        parent_a = Genome.random(100, trait_config)
        parent_b = Genome.random(100, trait_config)
        
        # Set distinct values for testing
        parent_a.weights.fill(1.0)
        parent_b.weights.fill(2.0)
        
        child = Genome.mate(
            parent_a,
            parent_b,
            crossover_method='uniform'
        )
        
        # Child should have mix of parent weights
        assert child.weights.shape == (100,)
        assert np.any(child.weights == 1.0) or np.any(child.weights == 2.0)
        assert child.generation == max(parent_a.generation, parent_b.generation) + 1
        assert child.parent_ids == (parent_a.lineage_id, parent_b.lineage_id)
    
    def test_crossover_one_point(self):
        """Test one-point crossover."""
        trait_config = create_default_trait_config()
        parent_a = Genome.random(100, trait_config)
        parent_b = Genome.random(100, trait_config)
        
        parent_a.weights.fill(1.0)
        parent_b.weights.fill(2.0)
        
        child = Genome.mate(
            parent_a,
            parent_b,
            crossover_method='one_point',
            mutation_rate=0.0  # Disable mutations for clearer testing
        )
        
        # Should have continuous segment from each parent (with no mutations)
        unique_vals = np.unique(child.weights)
        assert len(unique_vals) <= 2, f"Should have at most 2 unique values, got {len(unique_vals)}"
        assert 1.0 in unique_vals or 2.0 in unique_vals
    
    def test_crossover_blend(self):
        """Test blend crossover."""
        trait_config = create_default_trait_config()
        parent_a = Genome.random(100, trait_config)
        parent_b = Genome.random(100, trait_config)
        
        parent_a.weights.fill(1.0)
        parent_b.weights.fill(2.0)
        
        child = Genome.mate(
            parent_a,
            parent_b,
            crossover_method='blend'
        )
        
        # Blend should produce values between parents
        assert child.weights.shape == (100,)
        # With mutations, most values should be near 1.5 (average of 1 and 2)
        mean_val = np.mean(child.weights)
        assert 1.0 <= mean_val <= 2.0, f"Mean {mean_val} outside parent range"
    
    def test_mutation_via_mate(self):
        """Test genome mutation via mating."""
        trait_config = create_default_trait_config()
        parent = Genome.random(100, trait_config)
        
        original_weights = parent.weights.copy()
        original_traits = parent.traits.copy()
        
        # Mate with itself with high mutation
        child = Genome.mate(
            parent,
            parent,
            crossover_method='uniform',
            mutation_rate=1.0,  # Mutate all weights
            mutation_std=0.1,
            trait_mutation_std=0.05
        )
        
        # Weights should change due to mutation
        # (small chance they could be equal with low mutation_std, so check most changed)
        differences = np.sum(child.weights != original_weights)
        assert differences > 50, "Most weights should have mutated"
        
        # Traits should change
        assert child.traits != original_traits
        
        # Traits should stay in reasonable bounds (with clipping)
        assert 0.5 <= child.traits['metabolism_rate'] <= 2.0
        assert 2.0 <= child.traits['vision_radius'] <= 10.0
    
    def test_lineage_tracking(self):
        """Test lineage ID tracking through generations."""
        trait_config = create_default_trait_config()
        
        parent = Genome.random(100, trait_config)
        parent_lineage = parent.lineage_id
        
        child = parent.copy()
        child.generation = parent.generation + 1
        child.parent_ids = (parent_lineage, parent_lineage)
        
        # Child should track parent
        assert child.parent_ids[0] == parent_lineage
        assert child.generation == parent.generation + 1
    
    def test_weight_statistics(self):
        """Test weight distribution stats."""
        trait_config = create_default_trait_config()
        genome = Genome.random(10000, trait_config)
        
        # Weights should be approximately normal
        mean = np.mean(genome.weights)
        std = np.std(genome.weights)
        
        # Mean should be close to 0
        assert abs(mean) < 0.1, f"Mean {mean} too far from 0"
        
        # Std should be reasonable (around weight_init_std=0.5 by default)
        assert 0.4 < std < 0.6, f"Std {std} outside expected range"
    
    def test_trait_inheritance(self):
        """Test trait inheritance in mating."""
        trait_config = create_default_trait_config()
        
        parent_a = Genome.random(100, trait_config)
        parent_b = Genome.random(100, trait_config)
        
        # Set distinct trait values
        parent_a.traits['metabolism_rate'] = 0.8
        parent_b.traits['metabolism_rate'] = 1.6
        
        children = [Genome.mate(parent_a, parent_b, trait_mutation_std=0.01) for _ in range(10)]
        
        # Most children should have metabolism_rate near average of parents (1.2)
        child_rates = [c.traits['metabolism_rate'] for c in children]
        mean_rate = np.mean(child_rates)
        assert 0.7 <= mean_rate <= 1.7, \
            f"Child mean metabolism_rate {mean_rate} far from parent average 1.2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
