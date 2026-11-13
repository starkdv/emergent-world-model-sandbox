"""
Tests for population-shared world-model training
(``learning.ppo.world_model_shared``): finished chunks go into a
process-wide pool and every update adds a world-model loss on pooled
chunks encoded by the learning agent's own brain.
"""

import numpy as np
import pytest

from agents.brain import calculate_weight_count_for_config, create_brain
from agents.genome import Genome

torch = pytest.importorskip("torch")

from agents import ppo  # noqa: E402
from agents.ppo import PPOSequenceLearner  # noqa: E402


def _brain(seed=0):
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
    np.random.seed(seed)
    return create_brain(Genome.random(calculate_weight_count_for_config(cfg), {}), cfg)


def _learner(shared):
    return PPOSequenceLearner(
        batch_size=2,
        seq_len=4,
        epochs=1,
        rollout_metric_k=2,
        world_model_multistep={"k": 2, "coef": 0.5},
        world_model_shared=shared,
    )


def _feed(lrn, brain, steps, seed):
    g = np.random.default_rng(seed)
    for _ in range(steps):
        lrn.store_step(
            g.normal(size=brain.input_size).astype(np.float32),
            np.zeros(brain.gru_hidden_size, dtype=np.float32),
            int(g.integers(brain.output_size)),
            float(g.normal()),
            g.normal(size=brain.input_size).astype(np.float32),
            False,
            -1.0,
            np.ones(brain.output_size, dtype=np.float32),
        )


def setup_function():
    ppo.reset_shared_wm_pool()


def test_off_by_default_pool_stays_empty():
    brain = _brain()
    lrn = _learner(None)
    _feed(lrn, brain, 16, 0)
    assert not lrn.shared_wm_enabled
    assert len(ppo.SHARED_WM_POOL) == 0


def test_pool_collects_chunks_from_all_learners():
    shared = {"enabled": True, "batch": 2, "pool": 8}
    a, b = _learner(shared), _learner(shared)
    brain = _brain()
    _feed(a, brain, 8, 0)
    _feed(b, brain, 12, 1)
    assert len(ppo.SHARED_WM_POOL) == 5
    assert ppo.SHARED_WM_POOL.capacity == 8


def test_shared_loss_changes_the_update():
    shared = {"enabled": True, "batch": 2, "coef": 1.0}
    weights = []
    for use in (False, True):
        ppo.reset_shared_wm_pool()
        torch.manual_seed(0)
        brain = _brain()
        other = _learner(shared)
        _feed(other, _brain(1), 16, 5)  # another agent's experience
        lrn = _learner(shared if use else None)
        _feed(lrn, brain, 16, 0)
        np.random.seed(3)
        lrn.learn(brain)
        weights.append(brain.genome.weights.copy())
        assert np.isfinite(lrn.wm_rollout_error_ema)
    assert not np.allclose(weights[0], weights[1])
