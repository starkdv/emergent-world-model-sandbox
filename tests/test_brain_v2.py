"""
Tests for Brain v2 with GRU + Actor-Critic architecture.

Tests:
- Weight count calculation
- Brain initialization from genome
- Forward pass with GRU hidden states
- Action selection (decide method)
- Policy and value outputs
- Weight packing/unpacking
"""

import pytest
import numpy as np

from agents.brain import Brain
from agents.genome import Genome, create_default_trait_config
from agents.actions import Action


class TestBrainV2:
    """Test suite for Brain v2 with GRU + Actor-Critic."""

    def test_weight_count_calculation(self):
        """Test weight count calculation for different architectures."""
        # Default architecture: 72 -> [32] -> GRU(32) -> 8 actions
        weight_count = Brain.calculate_weight_count(
            input_size=72, encoder_layers=[32], gru_hidden_size=32, output_size=8
        )

        # Expected:
        # Encoder: (72*32 + 32) = 2336
        # GRU: 3 gates * (32*32 + 32*32 + 32) = 3 * 2080 = 6240
        # Policy head: (32*8 + 8) = 264
        # Value head: (32*1 + 1) = 33
        # Total: 2336 + 6240 + 264 + 33 = 8873
        assert weight_count == 8873, f"Expected 8873, got {weight_count}"

    def test_brain_initialization(self):
        """Test brain initialization from genome."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count(
            input_size=72, encoder_layers=[32], gru_hidden_size=32, output_size=8
        )
        genome = Genome.random(weight_count, trait_config)

        brain = Brain(
            genome,
            input_size=72,
            encoder_layers=[32],
            gru_hidden_size=32,
            output_size=8,
        )

        # Check structure
        assert hasattr(brain, "params")
        assert "encoder_weights" in brain.params
        assert "gru" in brain.params
        assert "policy_head" in brain.params
        assert "value_head" in brain.params

        # Check GRU structure
        gru = brain.params["gru"]
        for gate in ["r", "z", "h"]:
            assert f"Wr_input" in gru or f"W{gate}_input" in gru
            assert f"Wr_hidden" in gru or f"W{gate}_hidden" in gru
            assert f"br" in gru or f"b{gate}" in gru

    def test_initial_state(self):
        """Test GRU hidden state initialization."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count(
            input_size=72, encoder_layers=[32], gru_hidden_size=32, output_size=8
        )
        genome = Genome.random(weight_count, trait_config)

        brain = Brain(
            genome,
            input_size=72,
            encoder_layers=[32],
            gru_hidden_size=32,
            output_size=8,
        )

        h = brain.initial_state()

        # Check shape and initialization
        assert h.shape == (32,)
        assert np.all(h == 0), "Initial hidden state should be zeros"

    def test_forward_pass(self):
        """Test forward pass through network."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count(
            input_size=72, encoder_layers=[32], gru_hidden_size=32, output_size=8
        )
        genome = Genome.random(weight_count, trait_config)

        brain = Brain(
            genome,
            input_size=72,
            encoder_layers=[32],
            gru_hidden_size=32,
            output_size=8,
        )

        obs = np.random.randn(72)
        h = brain.initial_state()

        # Forward pass
        probs, value, h_next = brain.forward(obs, h)

        # Check outputs
        assert probs.shape == (8,), f"Expected shape (8,), got {probs.shape}"
        assert np.isclose(np.sum(probs), 1.0), "Probabilities should sum to 1"
        assert np.all(probs >= 0) and np.all(probs <= 1), "Invalid probabilities"
        assert isinstance(value, (float, np.floating)), "Value should be scalar"
        assert h_next.shape == h.shape, "Hidden state shape should be preserved"
        assert not np.array_equal(h, h_next), "Hidden state should update"

    def test_decide_action(self):
        """Test action selection."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count(
            input_size=72, encoder_layers=[32], gru_hidden_size=32, output_size=8
        )
        genome = Genome.random(weight_count, trait_config)

        brain = Brain(
            genome,
            input_size=72,
            encoder_layers=[32],
            gru_hidden_size=32,
            output_size=8,
        )

        obs = np.random.randn(72)
        h = brain.initial_state()

        # Decide action
        action, h_next, value = brain.decide(obs, h)

        # Check outputs
        assert isinstance(action, Action), "Should return Action enum"
        assert action.value in range(8), "Action should be valid"
        assert h_next.shape == h.shape, "Hidden state shape preserved"
        assert isinstance(value, (float, np.floating)), "Value should be scalar"

    def test_action_masking(self):
        """Test action masking in decide()."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count(
            input_size=72, encoder_layers=[32], gru_hidden_size=32, output_size=8
        )
        genome = Genome.random(weight_count, trait_config)

        brain = Brain(
            genome,
            input_size=72,
            encoder_layers=[32],
            gru_hidden_size=32,
            output_size=8,
        )

        obs = np.random.randn(72)
        h = brain.initial_state()

        # Mask all actions except WAIT (index 7)
        action_mask = np.array([False, False, False, False, False, False, False, True])

        # Sample many times to verify masking works
        actions = []
        for _ in range(100):
            action, h_next, value = brain.decide(obs, h, action_mask=action_mask)
            actions.append(action)

        # All actions should be WAIT
        assert all(a == Action.WAIT for a in actions), "Action mask not working"

    def test_temperature_sampling(self):
        """Test temperature parameter for exploration."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count(
            input_size=72, encoder_layers=[32], gru_hidden_size=32, output_size=8
        )
        genome = Genome.random(weight_count, trait_config)

        brain = Brain(
            genome,
            input_size=72,
            encoder_layers=[32],
            gru_hidden_size=32,
            output_size=8,
        )

        obs = np.random.randn(72)
        h = brain.initial_state()

        # High temperature (more exploration)
        actions_high_temp = []
        for _ in range(100):
            action, _, _ = brain.decide(obs, h, temperature=2.0)
            actions_high_temp.append(action.value)

        # Low temperature (more exploitation)
        actions_low_temp = []
        for _ in range(100):
            action, _, _ = brain.decide(obs, h, temperature=0.1)
            actions_low_temp.append(action.value)

        # High temperature should have more diversity
        unique_high = len(set(actions_high_temp))
        unique_low = len(set(actions_low_temp))

        assert unique_high >= unique_low, "High temperature should explore more"

    def test_hidden_state_sequence(self):
        """Test that hidden state evolves over sequence of observations."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count(
            input_size=72, encoder_layers=[32], gru_hidden_size=32, output_size=8
        )
        genome = Genome.random(weight_count, trait_config)

        brain = Brain(
            genome,
            input_size=72,
            encoder_layers=[32],
            gru_hidden_size=32,
            output_size=8,
        )

        # Create sequence of observations
        obs_sequence = [np.random.randn(72) for _ in range(5)]

        h = brain.initial_state()
        h_states = [h.copy()]

        # Process sequence
        for obs in obs_sequence:
            _, _, h = brain.forward(obs, h)
            h_states.append(h.copy())

        # Verify states evolve
        for i in range(len(h_states) - 1):
            assert not np.array_equal(
                h_states[i], h_states[i + 1]
            ), f"Hidden state should change at step {i}"

    def test_weight_packing_roundtrip(self):
        """Test that weights can be packed to genome and unpacked correctly."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count(
            input_size=72, encoder_layers=[32], gru_hidden_size=32, output_size=8
        )
        genome = Genome.random(weight_count, trait_config)

        brain1 = Brain(
            genome,
            input_size=72,
            encoder_layers=[32],
            gru_hidden_size=32,
            output_size=8,
        )

        # Get original genome weights
        original_weights = genome.weights.copy()

        # Create new brain from same genome
        brain2 = Brain(
            genome,
            input_size=72,
            encoder_layers=[32],
            gru_hidden_size=32,
            output_size=8,
        )

        # Test forward pass produces same results
        obs = np.random.randn(72)
        h = brain1.initial_state()

        probs1, val1, h1 = brain1.forward(obs, h)
        probs2, val2, h2 = brain2.forward(obs, h)

        assert np.allclose(probs1, probs2), "Probabilities should match"
        assert np.isclose(val1, val2), "Values should match"
        assert np.allclose(h1, h2), "Hidden states should match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
