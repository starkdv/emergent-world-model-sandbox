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
        assert genome.weights.dtype == np.float32
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
        
        genome = Genome(
            weights=weights,
            trait_config=trait_config,
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
        genome2.traits['max_energy'] = 1000.0
        genome2.generation = 10
        
        # Original should be unchanged
        assert genome1.weights[0] != 999.0
        assert genome1.traits['max_energy'] != 1000.0
        assert genome1.generation != 10
    
    def test_crossover_uniform(self):
        """Test uniform crossover."""
        trait_config = create_default_trait_config()
        parent_a = Genome.random(100, trait_config)
        parent_b = Genome.random(100, trait_config)
        
        # Set distinct values for testing
        parent_a.weights.fill(1.0)
        parent_b.weights.fill(2.0)
        
        child = Genome.crossover(
            parent_a,
            parent_b,
            method='uniform',
            trait_config=trait_config
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
        
        child = Genome.crossover(
            parent_a,
            parent_b,
            method='one_point',
            trait_config=trait_config
        )
        
        # Should have continuous segment from each parent
        unique_vals = np.unique(child.weights)
        assert len(unique_vals) <= 2, "Should have at most 2 unique values"
        assert 1.0 in unique_vals or 2.0 in unique_vals
    
    def test_crossover_two_point(self):
        """Test two-point crossover."""
        trait_config = create_default_trait_config()
        parent_a = Genome.random(100, trait_config)
        parent_b = Genome.random(100, trait_config)
        
        parent_a.weights.fill(1.0)
        parent_b.weights.fill(2.0)
        
        child = Genome.crossover(
            parent_a,
            parent_b,
            method='two_point',
            trait_config=trait_config
        )
        
        # Should have segments from both parents
        assert child.weights.shape == (100,)
        unique_vals = np.unique(child.weights)
        assert len(unique_vals) <= 2
    
    def test_mutate(self):
        """Test genome mutation."""
        trait_config = create_default_trait_config()
        genome = Genome.random(100, trait_config)
        
        original_weights = genome.weights.copy()
        original_traits = genome.traits.copy()
        
        genome.mutate(
            weight_mutation_rate=1.0,  # Mutate all weights
            weight_mutation_std=0.1,
            trait_mutation_rate=1.0,  # Mutate all traits
            trait_mutation_std=0.05
        )
        
        # Weights should change
        assert not np.array_equal(genome.weights, original_weights)
        
        # Traits should change
        assert genome.traits != original_traits
        
        # Traits should stay in bounds
        for trait_name, (min_val, max_val) in trait_config.items():
            trait_value = genome.traits[trait_name]
            assert min_val <= trait_value <= max_val, \
                f"Trait {trait_name}={trait_value} outside bounds after mutation"
    
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
        
        # Std should be reasonable
        assert 0.5 < std < 2.0, f"Std {std} outside expected range"
    
    def test_trait_inheritance(self):
        """Test trait inheritance in crossover."""
        trait_config = create_default_trait_config()
        
        parent_a = Genome.random(100, trait_config)
        parent_b = Genome.random(100, trait_config)
        
        # Set distinct trait values
        parent_a.traits['max_energy'] = 100.0
        parent_b.traits['max_energy'] = 200.0
        
        child = Genome.crossover(parent_a, parent_b, trait_config=trait_config)
        
        # Child trait should be between parents
        child_energy = child.traits['max_energy']
        assert 100.0 <= child_energy <= 200.0, \
            f"Child max_energy {child_energy} outside parent range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
