"""
Tests for the world-model quality upgrades (M1–M3):

- M1: k-step open-loop rollout-error diagnostic
  (``TorchBrainMirror.multistep_errors`` + learner ``wm_rollout_error``)
- M2: readiness-based planner gating (``warmup_error_threshold``) and the
  imagination ``warmup_error`` gate
- M3: multi-step world-model training loss (``world_model_multistep``)

Skipped when torch is unavailable (planner gating tests are pure numpy and
run regardless — they live at the bottom, above the torch guard imports).
"""

import numpy as np
import pytest

from agents.planner import LatentPlanner

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


def _batch(mirror, brain, batch=3, length=6, seed=0):
    """Random sequences + the real hiddens/latents the mirror produces."""
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


# --- M1: multistep_errors -------------------------------------------------


def test_multistep_errors_shape_and_finite():
    brain, _ = _wm_brain()
    m = TorchBrainMirror(brain, lr=1e-3)
    hs, zs, actions, rewards, valid, dones = _batch(m, brain)
    lat, rew = m.multistep_errors(hs, zs, actions, rewards, valid, dones, k=3)
    assert lat.shape == (3,) and rew.shape == (3,)
    assert torch.isfinite(lat).all() and torch.isfinite(rew).all()
    assert (lat >= 0).all() and (rew >= 0).all()


def test_multistep_horizon_one_matches_onestep_loss():
    """At horizon 1 the rollout error IS the standard 1-step WM loss."""
    brain, _ = _wm_brain()
    m = TorchBrainMirror(brain, lr=1e-3)
    hs, zs, actions, rewards, valid, dones = _batch(m, brain)
    lat, rew = m.multistep_errors(hs, zs, actions, rewards, valid, dones, k=1)

    batch, length, h_size = hs.shape
    onehot = torch.nn.functional.one_hot(
        actions.reshape(-1), num_classes=brain.output_size
    ).float()
    z_pred, r_pred = m._dynamics(hs.reshape(-1, h_size), onehot)
    z_tgt = zs[:, 1:, :].reshape(batch * length, -1)
    mask = (valid * (1.0 - dones)).reshape(-1)
    n = torch.clamp(mask.sum(), min=1.0)
    ref_lat = (((z_pred - z_tgt) ** 2).mean(dim=1) * mask).sum() / n
    ref_rew = ((r_pred - rewards.reshape(-1)) ** 2 * mask).sum() / n
    assert torch.allclose(lat[0], ref_lat, atol=1e-6)
    assert torch.allclose(rew[0], ref_rew, atol=1e-6)


def test_multistep_errors_respect_done_boundaries():
    """A done inside the window removes it from deeper horizons."""
    brain, _ = _wm_brain()
    m = TorchBrainMirror(brain, lr=1e-3)
    hs, zs, actions, rewards, valid, dones = _batch(m, brain, batch=1, length=4)
    dones_cut = dones.clone()
    dones_cut[0, 1] = 1.0  # episode ends at step 1
    lat_full, _ = m.multistep_errors(hs, zs, actions, rewards, valid, dones, k=3)
    lat_cut, _ = m.multistep_errors(hs, zs, actions, rewards, valid, dones_cut, k=3)
    # errors stay finite; the masked version is computed over fewer windows
    assert torch.isfinite(lat_cut).all()
    assert not torch.equal(lat_full, lat_cut)


def test_multistep_errors_grad_flows():
    brain, _ = _wm_brain()
    m = TorchBrainMirror(brain, lr=1e-3)
    hs, zs, actions, rewards, valid, dones = _batch(m, brain)
    lat, rew = m.multistep_errors(hs, zs, actions, rewards, valid, dones, k=3)
    (lat[1:].mean() + rew[1:].mean()).backward()
    assert m.params["dyn.W1"].grad is not None
    assert torch.isfinite(m.params["dyn.W1"].grad).all()


def test_multistep_k_clamped_to_seq_len():
    brain, _ = _wm_brain()
    m = TorchBrainMirror(brain, lr=1e-3)
    hs, zs, actions, rewards, valid, dones = _batch(m, brain, length=4)
    lat, _ = m.multistep_errors(hs, zs, actions, rewards, valid, dones, k=99)
    assert lat.shape == (4,)  # clamped to L


# --- M1/M3: learner plumbing ------------------------------------------------


def _feed_learner(lrn, brain, steps=24, seed=1):
    g = np.random.default_rng(seed)
    h = np.zeros(brain.gru_hidden_size, dtype=np.float32)
    mask = np.ones(brain.output_size, dtype=np.float32)
    for _ in range(steps):
        obs = g.normal(size=brain.input_size).astype(np.float32)
        nxt = g.normal(size=brain.input_size).astype(np.float32)
        lrn.store_step(
            obs,
            h,
            int(g.integers(brain.output_size)),
            float(g.normal()),
            nxt,
            False,
            -1.0,
            mask,
        )


def test_learner_rollout_diagnostic_populated():
    brain, _ = _wm_brain()
    lrn = PPOSequenceLearner(batch_size=2, seq_len=4, epochs=1, rollout_metric_k=3)
    _feed_learner(lrn, brain)
    assert lrn.wm_rollout_error is None
    lrn.learn(brain)
    assert isinstance(lrn.wm_rollout_error, list)
    assert len(lrn.wm_rollout_error) == 3
    assert all(np.isfinite(e) and e >= 0 for e in lrn.wm_rollout_error)
    assert lrn.wm_rollout_error_ema is not None
    ema1 = lrn.wm_rollout_error_ema
    lrn.learn(brain)  # EMA updates on subsequent calls
    assert np.isfinite(lrn.wm_rollout_error_ema)
    assert lrn.wm_rollout_error_ema != ema1 or lrn.wm_rollout_error[-1] == ema1


def test_learner_diagnostic_disabled():
    brain, _ = _wm_brain()
    lrn = PPOSequenceLearner(batch_size=2, seq_len=4, epochs=1, rollout_metric_k=0)
    _feed_learner(lrn, brain)
    lrn.learn(brain)
    assert lrn.wm_rollout_error is None
    assert lrn.wm_rollout_error_ema is None


def test_learner_multistep_loss_config_and_runs():
    lrn = PPOSequenceLearner(world_model_multistep={"k": 3, "coef": 0.25})
    assert lrn.ms_k == 3 and lrn.ms_coef == 0.25
    off = PPOSequenceLearner()
    assert off.ms_k == 0  # legacy default: off

    brain, _ = _wm_brain()
    lrn = PPOSequenceLearner(
        batch_size=2, seq_len=4, epochs=1, world_model_multistep={"k": 3}
    )
    _feed_learner(lrn, brain)
    loss = lrn.learn(brain)
    assert np.isfinite(loss)


def test_imagination_error_readiness_gate():
    lrn = PPOSequenceLearner(
        imagination={"enabled": True, "warmup_ticks": 100, "warmup_error": 0.5}
    )
    lrn.current_tick = 200  # past the tick warmup
    assert lrn.imagination_active() is False  # no measurement yet
    lrn.wm_rollout_error_ema = 0.9
    assert lrn.imagination_active() is False  # error too high
    lrn.wm_rollout_error_ema = 0.4
    assert lrn.imagination_active() is True  # measured ready


# --- M2: planner readiness gating (pure numpy) -------------------------------


def test_planner_error_gate_switches_and_latches():
    p = LatentPlanner(
        strategy="cem",
        warmup_strategy="policy_shooting",
        warmup_error_threshold=0.5,
    )
    assert p.effective_strategy(0, model_error=None) == "policy_shooting"
    assert p.effective_strategy(100, model_error=0.9) == "policy_shooting"
    assert p.effective_strategy(200, model_error=0.4) == "cem"  # ready
    # latched: a later noisy error (or missing measurement) does not revert
    assert p.effective_strategy(300, model_error=0.9) == "cem"
    assert p.effective_strategy(400, model_error=None) == "cem"


def test_planner_error_gate_tick_deadline():
    p = LatentPlanner(strategy="cem", warmup_ticks=1000, warmup_error_threshold=0.5)
    # error never drops below the threshold → warmup_ticks is the deadline
    assert p.effective_strategy(999, model_error=2.0) == "policy_shooting"
    assert p.effective_strategy(1000, model_error=2.0) == "cem"


def test_planner_tick_mode_unchanged_without_threshold():
    p = LatentPlanner(strategy="cem", warmup_ticks=1000)
    assert p.effective_strategy(999, model_error=0.0) == "policy_shooting"
    assert p.effective_strategy(1000) == "cem"


def test_planner_from_config_reads_error_threshold():
    p = LatentPlanner.from_config({"strategy": "cem", "warmup_error_threshold": 0.25})
    assert p.warmup_error_threshold == 0.25
    d = LatentPlanner.from_config({})
    assert d.warmup_error_threshold == 0.0  # legacy default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
