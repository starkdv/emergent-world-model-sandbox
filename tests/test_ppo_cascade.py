"""
Tests for the multi-scale BPTT cascade (``learning.ppo.cascade``).

With ``seq_len`` L the learner always trains on L-step chunks; the cascade
additionally queues the most recent L*2^j steps every 2^j chunks (up to
``max_len``) and trains on each queued window after the regular update.
"""

import numpy as np
import pytest

from agents.brain import calculate_weight_count_for_config, create_brain
from agents.genome import Genome

torch = pytest.importorskip("torch")

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


def _learner(cascade=None, seq_len=4):
    return PPOSequenceLearner(
        batch_size=1,
        seq_len=seq_len,
        epochs=1,
        chunk_capacity=4,
        rollout_metric_k=2,
        cascade=cascade,
    )


def _step(lrn, brain, g, t, done=False):
    h = np.full(brain.gru_hidden_size, float(t), dtype=np.float32)
    lrn.store_step(
        g.normal(size=brain.input_size).astype(np.float32),
        h,
        int(g.integers(brain.output_size)),
        float(t),
        g.normal(size=brain.input_size).astype(np.float32),
        done,
        -1.0,
        np.ones(brain.output_size, dtype=np.float32),
    )


def test_cascade_off_by_default_keeps_no_history():
    brain = _brain()
    lrn = _learner()
    g = np.random.default_rng(0)
    for t in range(40):
        _step(lrn, brain, g, t)
    assert not lrn.cascade_enabled
    assert len(lrn._history) == 0 and not lrn._pending_long


def test_binary_counter_schedule_and_window_contents():
    brain = _brain()
    lrn = _learner({"enabled": True, "max_len": 16})
    g = np.random.default_rng(0)
    queued = []
    for t in range(32):
        _step(lrn, brain, g, t)
        if lrn._pending_long:
            queued.append((t + 1, sorted(lrn._pending_long)))
            lrn._pending_long.clear()
    # seq_len 4: 8-step windows every 2nd chunk, 16-step every 4th.
    assert queued == [
        (8, [8]),
        (16, [8, 16]),
        (24, [8]),
        (32, [8, 16]),
    ]
    win = lrn._window_chunk(16)
    assert win.obs.shape[0] == 16 and win.valid.all() and not win.dones.any()
    np.testing.assert_allclose(win.rewards, np.arange(16, 32))
    # h0 is the hidden state before the window's first step (t = 16).
    np.testing.assert_allclose(win.h0, 16.0)


def test_death_clears_history_and_pending():
    brain = _brain()
    lrn = _learner({"enabled": True, "max_len": 16})
    g = np.random.default_rng(0)
    for t in range(8):
        _step(lrn, brain, g, t)
    assert lrn._pending_long
    _step(lrn, brain, g, 8, done=True)
    assert len(lrn._history) == 0 and not lrn._pending_long


def test_learn_trains_long_windows_and_syncs_genome():
    brain = _brain()
    lrn = _learner({"enabled": True, "max_len": 16})
    g = np.random.default_rng(0)
    for t in range(16):
        _step(lrn, brain, g, t)
    before = brain.genome.weights.copy()
    lrn.learn(brain)
    assert lrn.update_count == 1
    assert lrn.cascade_updates == 2  # the 8- and 16-step windows
    assert not lrn._pending_long
    assert not np.allclose(before, brain.genome.weights)
    assert np.isfinite(lrn.wm_rollout_error_ema)


def test_learn_without_cascade_matches_legacy_path():
    # Same seed, same data: cascade off must give the pre-change update.
    out = []
    for cascade in (None, {"enabled": False}):
        torch.manual_seed(0)
        np.random.seed(0)
        brain = _brain()
        lrn = _learner(cascade)
        g = np.random.default_rng(0)
        for t in range(16):
            _step(lrn, brain, g, t)
        np.random.seed(1)
        lrn.learn(brain)
        out.append(brain.genome.weights.copy())
    np.testing.assert_array_equal(out[0], out[1])
