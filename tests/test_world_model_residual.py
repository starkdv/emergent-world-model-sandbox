"""
Tests for the residual dynamics head and the readable world-model metrics.

- ``brain.world_model.residual``: the head reads the current latent z and
  predicts its change, ẑ' = z + Δ([h ‖ onehot(a) ‖ z]). Off by default, so
  legacy genomes, counts and predictions are untouched.
- ``TorchBrainMirror.baseline_errors`` + learner ``wm_rel_persistence_ema`` /
  ``wm_rel_variance_ema``: the k-step rollout error divided by the do-nothing
  (ẑ = z) error and by the latent variance.
"""

import numpy as np
import pytest

from agents.brain import (
    adapt_loaded_genome,
    calculate_weight_count_for_config,
    create_brain,
)
from agents.brain.spec import migrate_genome
from agents.genome import Genome
from agents.planner import LatentPlanner


def _cfg(version=3, residual=True, hidden=12):
    cfg = {
        "version": version,
        "v3": {
            "embed_dim": 6,
            "state_dim": 12,
            "gru_hidden_size": 10,
            "value_hidden": 8,
        },
        "world_model": {"enabled": True, "hidden": hidden},
    }
    if residual:
        cfg["world_model"]["residual"] = True
    return cfg


def _brain(cfg, seed=0):
    np.random.seed(seed)
    genome = Genome.random(calculate_weight_count_for_config(cfg), {})
    return create_brain(genome, cfg)


# --- layout -----------------------------------------------------------------


@pytest.mark.parametrize("version", [2, 3, 3.5])
def test_residual_adds_exactly_z_rows_to_w1(version):
    legacy = _brain(_cfg(version, residual=False))
    res = _brain(_cfg(version, residual=True))
    z = legacy.params["dynamics"]["Wz"].shape[1]
    hid = legacy.params["dynamics"]["W1"].shape[1]
    assert res.world_model_residual and not legacy.world_model_residual
    assert res.params["dynamics"]["W1"].shape == (
        legacy.params["dynamics"]["W1"].shape[0] + z,
        hid,
    )
    assert res.spec.count() == legacy.spec.count() + z * hid


def test_residual_flag_ignored_without_world_model():
    cfg = _cfg(residual=True)
    cfg["world_model"]["enabled"] = False
    plain = _cfg(residual=False)
    plain["world_model"]["enabled"] = False
    assert calculate_weight_count_for_config(cfg) == calculate_weight_count_for_config(
        plain
    )


def test_v4_rejects_residual():
    cfg = {"version": 4, "world_model": {"enabled": True, "residual": True}}
    with pytest.raises(ValueError):
        create_brain(Genome.random(100, {}), cfg)


def test_legacy_head_migrates_into_residual_layout():
    legacy = _brain(_cfg(residual=False))
    res_cfg = _cfg(residual=True)
    res = _brain(res_cfg, seed=1)
    flat = migrate_genome(legacy.genome.weights, legacy.spec, res.spec)
    assert flat.shape == (res.spec.count(),)
    migrated = create_brain(Genome(flat, {}), res_cfg)
    w_old = legacy.params["dynamics"]["W1"]
    w_new = migrated.params["dynamics"]["W1"]
    np.testing.assert_allclose(w_new[: w_old.shape[0]], w_old, rtol=1e-6)
    assert np.all(w_new[w_old.shape[0] :] == 0.0)
    assert adapt_loaded_genome(flat, res_cfg) is flat


# --- numpy forward ----------------------------------------------------------


def test_residual_prediction_is_z_plus_delta():
    brain = _brain(_cfg())
    obs = np.random.rand(brain.input_size).astype(np.float32)
    _, _, h = brain.forward(obs, brain.initial_state())
    z = brain.encode(obs)
    np.testing.assert_allclose(brain.last_latent, z)
    dyn = brain.params["dynamics"]
    # Zero the output layer: Δ = 0, so the prediction must be exactly z.
    dyn["Wz"][...] = 0.0
    dyn["bz"][...] = 0.0
    z_pred, _ = brain.predict_next_latent(h, 3)
    np.testing.assert_allclose(z_pred, z)
    # Explicit z overrides the cached latent.
    other = np.ones_like(z)
    z_pred2, _ = brain.predict_next_latent(h, 3, z=other)
    np.testing.assert_allclose(z_pred2, other)


def test_legacy_prediction_unchanged_by_z_argument():
    brain = _brain(_cfg(residual=False))
    h = np.random.rand(brain.gru_hidden_size).astype(np.float32)
    a, _ = brain.predict_next_latent(h, 2)
    b, _ = brain.predict_next_latent(h, 2, z=np.ones(a.shape, dtype=np.float32))
    np.testing.assert_array_equal(a, b)


def test_planner_runs_with_residual_head():
    brain = _brain(_cfg())
    obs = np.random.rand(brain.input_size).astype(np.float32)
    _, _, h = brain.forward(obs, brain.initial_state())
    mask = np.ones(brain.output_size, dtype=np.float32)
    for strategy in ("shooting", "policy_shooting", "cem"):
        planner = LatentPlanner(depth=3, samples=4, strategy=strategy)
        a = planner.plan(brain, h, mask)
        assert 0 <= a < brain.output_size


# --- torch mirror + learner ---------------------------------------------------

torch = pytest.importorskip("torch")

from agents.ppo import PPOSequenceLearner, TorchBrainMirror  # noqa: E402


def test_mirror_residual_matches_numpy():
    brain = _brain(_cfg())
    m = TorchBrainMirror(brain, lr=1e-3)
    assert m.wm_residual
    h = np.random.rand(brain.gru_hidden_size).astype(np.float32)
    z = np.random.rand(brain.params["dynamics"]["Wz"].shape[1]).astype(np.float32)
    zp_np, rp_np = brain.predict_next_latent(h, 4, z=z)
    onehot = torch.nn.functional.one_hot(torch.tensor([4]), brain.output_size)
    with torch.no_grad():
        zp_t, rp_t = m._dynamics(
            torch.as_tensor(h)[None], onehot.float(), torch.as_tensor(z)[None]
        )
    np.testing.assert_allclose(zp_t[0].numpy(), zp_np, atol=1e-5)
    assert abs(float(rp_t[0]) - rp_np) < 1e-5


def test_mirror_residual_requires_z():
    brain = _brain(_cfg())
    m = TorchBrainMirror(brain, lr=1e-3)
    h = torch.zeros(1, brain.gru_hidden_size)
    onehot = torch.zeros(1, brain.output_size)
    with pytest.raises(ValueError):
        m._dynamics(h, onehot)


def _batch(mirror, brain, batch=3, length=6, seed=0):
    g = np.random.default_rng(seed)
    obs = torch.as_tensor(
        g.normal(size=(batch, length, brain.input_size)).astype(np.float32)
    )
    h0 = torch.zeros(batch, brain.gru_hidden_size)
    boot = torch.as_tensor(g.normal(size=(batch, brain.input_size)).astype(np.float32))
    actions = torch.as_tensor(
        g.integers(0, brain.output_size, size=(batch, length)).astype(np.int64)
    )
    rewards = torch.as_tensor(g.normal(size=(batch, length)).astype(np.float32))
    valid = torch.ones(batch, length)
    dones = torch.zeros(batch, length)
    with torch.no_grad():
        _, _, _, zs, hs = mirror.forward_sequence(obs, h0, boot)
    return hs, zs, actions, rewards, valid, dones


def test_baseline_errors_match_hand_computation():
    brain = _brain(_cfg(residual=False))
    m = TorchBrainMirror(brain, lr=1e-3)
    hs, zs, actions, rewards, valid, dones = _batch(m, brain)
    pers, var = m.baseline_errors(zs, valid, dones, k=2)
    z = zs.numpy()
    n_win = 6 - 2 + 1
    expect1 = ((z[:, :n_win] - z[:, 1 : 1 + n_win]) ** 2).mean()
    expect2 = ((z[:, :n_win] - z[:, 2 : 2 + n_win]) ** 2).mean()
    np.testing.assert_allclose(pers.numpy(), [expect1, expect2], rtol=1e-5)
    real = z[:, 1:].reshape(-1, z.shape[-1])
    np.testing.assert_allclose(float(var), real.var(axis=0, ddof=1).mean(), rtol=1e-4)


def test_residual_head_with_zero_delta_equals_persistence():
    brain = _brain(_cfg())
    brain.params["dynamics"]["Wz"][...] = 0.0
    brain.params["dynamics"]["bz"][...] = 0.0
    m = TorchBrainMirror(brain, lr=1e-3)
    hs, zs, actions, rewards, valid, dones = _batch(m, brain)
    with torch.no_grad():
        lat, _ = m.multistep_errors(hs, zs, actions, rewards, valid, dones, k=3)
        pers, _ = m.baseline_errors(zs, valid, dones, k=3)
    np.testing.assert_allclose(lat.numpy(), pers.numpy(), rtol=1e-5)


def _feed(lrn, brain, steps=24, seed=1):
    g = np.random.default_rng(seed)
    h = np.zeros(brain.gru_hidden_size, dtype=np.float32)
    mask = np.ones(brain.output_size, dtype=np.float32)
    for _ in range(steps):
        lrn.store_step(
            g.normal(size=brain.input_size).astype(np.float32),
            h,
            int(g.integers(brain.output_size)),
            float(g.normal()),
            g.normal(size=brain.input_size).astype(np.float32),
            False,
            -1.0,
            mask,
        )


@pytest.mark.parametrize("residual", [False, True])
def test_learner_trains_and_reports_relative_errors(residual):
    brain = _brain(_cfg(residual=residual))
    lrn = PPOSequenceLearner(
        batch_size=2,
        seq_len=4,
        epochs=1,
        rollout_metric_k=3,
        world_model_multistep={"k": 3, "coef": 0.5},
        imagination={"enabled": True, "horizon": 2, "batch": 4},
    )
    _feed(lrn, brain)
    before = brain.params["dynamics"]["W1"].copy()
    lrn.learn(brain)
    assert not np.allclose(before, brain.params["dynamics"]["W1"])
    assert len(lrn.wm_persistence_error) == 3
    assert lrn.wm_latent_var > 0
    assert np.isfinite(lrn.wm_rel_persistence_ema)
    assert np.isfinite(lrn.wm_rel_variance_ema)
    ratio = lrn.wm_rollout_error[-1] / lrn.wm_persistence_error[-1]
    assert abs(lrn.wm_rel_persistence_ema - ratio) < 1e-6


# --- zero-initialised dynamics output -----------------------------------------


@pytest.mark.parametrize("version", [2, 3, 3.5])
@pytest.mark.parametrize("residual", [False, True])
def test_zero_init_prior_applies_to_fresh_genomes(version, residual):
    from agents.brain import activate_brain_layout
    from agents.brain.spec import set_active_param_spec

    cfg = _cfg(version, residual=residual)
    cfg["world_model"]["zero_init"] = True
    try:
        activate_brain_layout(cfg)
        brain = _brain(cfg)
        dyn = brain.params["dynamics"]
        assert np.all(dyn["Wz"] == 0.0) and np.all(dyn["bz"] == 0.0)
        assert np.any(dyn["W1"] != 0.0)  # hidden layer still random
        if residual:
            obs = np.random.rand(brain.input_size).astype(np.float32)
            _, _, h = brain.forward(obs, brain.initial_state())
            z_pred, _ = brain.predict_next_latent(h, 1)
            np.testing.assert_allclose(z_pred, brain.encode(obs))
    finally:
        activate_brain_layout(None)
        set_active_param_spec(None)


def test_zero_init_off_by_default():
    from agents.brain import activate_brain_layout
    from agents.brain.spec import get_active_param_spec

    activate_brain_layout(_cfg(3, residual=True))
    assert get_active_param_spec() is None
    activate_brain_layout(None)
