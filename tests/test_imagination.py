"""
Tests for the Dreamer-style imagination actor-critic (Planning proposal P3).

Exercises ``TorchBrainMirror.imagine_loss`` end-to-end on a real Brain v3 with a
latent world-model head, and the PPO learner's imagination config plumbing.
Skipped when torch is unavailable.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agents.brain import create_brain, calculate_weight_count_for_config  # noqa: E402
from agents.genome import Genome  # noqa: E402
from agents.ppo import TorchBrainMirror, PPOSequenceLearner  # noqa: E402


def _wm_brain():
    cfg = {
        "version": 3,
        "v3": {
            "embed_dim": 6,
            "state_dim": 12,
            "gru_hidden_size": 10,
            "value_hidden": 8,
        },
        "world_model": {"enabled": True, "hidden": 12},
    }
    genome = Genome.random(calculate_weight_count_for_config(cfg), {})
    return create_brain(genome, cfg), cfg


def test_mirror_has_world_model():
    brain, _ = _wm_brain()
    assert brain.has_world_model
    m = TorchBrainMirror(brain, lr=1e-3)
    assert m.has_world_model


def test_imagine_loss_is_finite_scalar_with_grad():
    torch.manual_seed(0)
    brain, _ = _wm_brain()
    m = TorchBrainMirror(brain, lr=1e-3)
    h0 = torch.zeros(32, brain.gru_hidden_size, dtype=torch.float32)
    loss = m.imagine_loss(h0, horizon=5, gamma=0.95, lam=0.95, entropy_coef=0.001)
    assert loss.shape == ()  # scalar
    assert torch.isfinite(loss)
    # backprop reaches the policy + value heads
    loss.backward()
    assert m.params["policy.W"].grad is not None
    assert torch.isfinite(m.params["policy.W"].grad).all()


def test_imagine_loss_horizon_one():
    brain, _ = _wm_brain()
    m = TorchBrainMirror(brain, lr=1e-3)
    h0 = torch.zeros(8, brain.gru_hidden_size, dtype=torch.float32)
    loss = m.imagine_loss(h0, horizon=1, gamma=0.9, lam=0.9, entropy_coef=0.0)
    assert torch.isfinite(loss)


def test_learner_imagination_config_off_by_default():
    lrn = PPOSequenceLearner()
    assert lrn.imag_enabled is False


def test_learner_imagination_config_enabled():
    lrn = PPOSequenceLearner(
        imagination={
            "enabled": True,
            "horizon": 7,
            "weight": 0.3,
            "batch": 64,
            "lambda": 0.9,
            "entropy": 0.002,
        }
    )
    assert lrn.imag_enabled is True
    assert lrn.imag_horizon == 7
    assert lrn.imag_weight == 0.3
    assert lrn.imag_batch == 64
    assert lrn.imag_lambda == 0.9
    assert lrn.imag_entropy == 0.002


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
