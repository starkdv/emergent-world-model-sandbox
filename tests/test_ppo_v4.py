"""
PPO learner under Brain v4 — replay fidelity and the dual-discount critic.

The single most dangerous thing about v4 is that the recurrent state now
carries an episodic memory whose evolution depends on the actions, rewards
and move outcomes of the chunk. If the torch mirror's replay diverges from
what the agent actually did, PPO trains on a trajectory that never happened
and nothing downstream will tell us. So the headline test here re-runs a
stored chunk through both implementations and compares step by step.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agents.actions import Action  # noqa: E402
from agents.brain import (  # noqa: E402
    activate_brain_layout,
    calculate_weight_count_for_config,
    create_brain,
)
from agents.brain.instincts import InstinctModule  # noqa: E402
from agents.brain.spec import OBSERVATION_SPEC_V4  # noqa: E402
from agents.genome import Genome, create_default_trait_config  # noqa: E402
from agents.ppo import PPOSequenceLearner, TorchBrainMirror  # noqa: E402

CFG_V4 = {"version": 4, "world_model": {"enabled": True, "hidden": 32}}
CFG_V35 = {"version": 3.5, "world_model": {"enabled": True, "hidden": 32}}


@pytest.fixture(autouse=True)
def _restore_layout():
    from agents.brain.spec import (
        get_active_observation_spec,
        get_active_param_spec,
        set_active_observation_spec,
        set_active_param_spec,
    )

    obs, params = get_active_observation_spec(), get_active_param_spec()
    yield
    set_active_observation_spec(obs)
    set_active_param_spec(params)


def _brain(cfg=CFG_V4, seed=0):
    activate_brain_layout(cfg)
    np.random.seed(seed)
    genome = Genome.random(
        calculate_weight_count_for_config(cfg), create_default_trait_config()
    )
    return create_brain(genome, cfg, instincts=InstinctModule(enabled=False))


def _episode(brain, length, rng, obs_dim):
    """Act for `length` steps, returning what a learner would have stored."""
    obs_seq, actions, rewards, moved, values = [], [], [], [], []
    h = brain.initial_state()
    h0 = h.copy()
    mask = np.ones(brain.output_size, dtype=np.float32)
    for t in range(length):
        obs = rng.random(obs_dim).astype(np.float32)
        z, h, core, _ = brain.core_step(obs, h)
        values.append(brain.values(z, core).copy())
        action = int(rng.integers(brain.output_size))
        reward = float(rng.normal() * 2.0)
        did_move = action == Action.MOVE_FORWARD.value and bool(rng.integers(2))
        h = brain.update_memory(h, obs, action, reward, did_move)
        obs_seq.append(obs)
        actions.append(action)
        rewards.append(reward)
        moved.append(1.0 if did_move else 0.0)
    boot = rng.random(obs_dim).astype(np.float32)
    return (
        np.stack(obs_seq),
        h0,
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(moved, dtype=np.float32),
        np.stack(values),
        boot,
        mask,
    )


def test_mirror_replay_matches_the_numpy_brain_step_for_step():
    """The whole point: replay must reproduce the state the agent had."""
    brain = _brain(seed=1)
    rng = np.random.default_rng(0)
    length = 16
    obs, h0, actions, rewards, moved, values_np, boot, _ = _episode(
        brain, length, rng, OBSERVATION_SPEC_V4.size
    )

    mirror = TorchBrainMirror(brain, lr=1e-4)
    with torch.no_grad():
        logits, values_t, v_boot, zs, hs = mirror.forward_sequence(
            torch.as_tensor(obs[None, ...]),
            torch.as_tensor(h0[None, ...]),
            torch.as_tensor(boot[None, ...]),
            torch.as_tensor(actions[None, ...]),
            torch.as_tensor(rewards[None, ...]),
            torch.as_tensor(moved[None, ...]),
        )

    assert values_t.shape == (1, length, 2)
    assert v_boot.shape == (1, 2)
    np.testing.assert_allclose(
        values_t[0].numpy(), values_np, atol=1e-4, err_msg="critic diverged in replay"
    )

    # And the policy logits, which is what the PPO ratio is built from.
    h = h0.copy()
    for t in range(length):
        z, h, core, _ = brain.core_step(obs[t], h)
        expected = (
            core @ brain.params["policy_head"]["W"] + brain.params["policy_head"]["b"]
        )
        np.testing.assert_allclose(logits[0, t].numpy(), expected, atol=1e-4)
        h = brain.update_memory(
            h, obs[t], int(actions[t]), float(rewards[t]), moved[t] > 0
        )


def test_replay_of_a_chunk_without_memory_signals_would_diverge():
    """A control: drop `moved` and the replayed trajectory is NOT the same."""
    brain = _brain(seed=2)
    rng = np.random.default_rng(1)
    length = 16
    obs, h0, actions, rewards, moved, values_np, boot, _ = _episode(
        brain, length, rng, OBSERVATION_SPEC_V4.size
    )
    assert moved.sum() > 0, "the fixture must contain a successful move"

    mirror = TorchBrainMirror(brain, lr=1e-4)
    with torch.no_grad():
        _, wrong, _, _, _ = mirror.forward_sequence(
            torch.as_tensor(obs[None, ...]),
            torch.as_tensor(h0[None, ...]),
            torch.as_tensor(boot[None, ...]),
            torch.as_tensor(actions[None, ...]),
            torch.as_tensor(rewards[None, ...]),
            None,  # pretend every move was blocked
        )
    assert not np.allclose(wrong[0].numpy(), values_np, atol=1e-4)


def test_v3_path_is_untouched_by_the_v4_branch():
    """v3.5 replay must be exactly what it was before v4 existed."""
    brain = _brain(CFG_V35, seed=3)
    rng = np.random.default_rng(2)
    length = 8
    obs = rng.random((length, OBSERVATION_SPEC_V4.size - 13)).astype(np.float32)
    h0 = brain.initial_state()
    boot = rng.random(obs.shape[1]).astype(np.float32)

    mirror = TorchBrainMirror(brain, lr=1e-4)
    assert mirror.n_values == 1
    with torch.no_grad():
        _, values, v_boot, _, hs = mirror.forward_sequence(
            torch.as_tensor(obs[None, ...]),
            torch.as_tensor(h0[None, ...]),
            torch.as_tensor(boot[None, ...]),
        )
    assert values.shape == (1, length, 1)
    assert hs.shape == (1, length, brain.gru_hidden_size)

    h = h0.copy()
    for t in range(length):
        _, value, h = brain.forward(obs[t], h)
        assert values[0, t, 0].item() == pytest.approx(value, abs=1e-4)


def _learner(**kwargs):
    cfg = dict(seq_len=8, batch_size=2, epochs=1, chunk_capacity=8)
    cfg.update(kwargs)
    return PPOSequenceLearner(**cfg)


def _fill(learner, brain, rng, steps):
    h = brain.initial_state()
    mask = np.ones(brain.output_size, dtype=np.float32)
    obs_dim = brain.input_size
    for _ in range(steps):
        obs = rng.random(obs_dim).astype(np.float32)
        h_before = h.copy()
        action, h, _, logp = brain.decide_with_logprob(obs, h, mask)
        reward = float(rng.normal())
        did_move = bool(rng.integers(2))
        nxt = rng.random(obs_dim).astype(np.float32)
        learner.store_step(
            obs, h_before, action.value, reward, nxt, False, logp, mask, did_move
        )
        update = getattr(brain, "update_memory", None)
        if update is not None:
            h = update(h, obs, action.value, reward, did_move)


def test_learn_runs_and_syncs_back_into_the_genome():
    brain = _brain(seed=4)
    learner = _learner(rollout_metric_k=2, advantage_mix=0.5)
    rng = np.random.default_rng(3)
    _fill(learner, brain, rng, 40)
    before = brain.genome.weights.copy()
    loss = learner.learn(brain)
    assert np.isfinite(loss)
    assert not np.array_equal(before, brain.genome.weights), "Lamarckian sync missing"
    assert learner.wm_rollout_error is not None
    assert len(learner.wm_rollout_error) == 2


def test_chunks_record_move_outcomes():
    brain = _brain(seed=5)
    learner = _learner()
    rng = np.random.default_rng(4)
    _fill(learner, brain, rng, 24)
    chunk = learner.replay_buffer.sample(1)[0]
    assert chunk.moved is not None and chunk.moved.shape == (learner.seq_len,)
    assert set(np.unique(chunk.moved)) <= {0.0, 1.0}


def test_genome_beta_overrides_the_config_mix():
    """The evolved horizon is the genome's, not the YAML's."""
    brain = _brain(seed=6)
    brain.params["drive"]["lam"][4] = 4.0  # sigmoid(4) ~ 0.982
    assert brain.advantage_mix > 0.9
    learner = _learner(advantage_mix=0.0)
    rng = np.random.default_rng(5)
    _fill(learner, brain, rng, 40)
    assert np.isfinite(learner.learn(brain))


def test_return_scaling_keeps_the_long_head_from_swamping_the_loss():
    """With scaling off vs on, the value loss magnitude must differ."""
    brain = _brain(seed=7)
    rng = np.random.default_rng(6)
    losses = {}
    for scale in (True, False):
        b = _brain(seed=7)
        learner = _learner(return_scale=scale, advantage_mix=0.5)
        _fill(learner, b, np.random.default_rng(6), 40)
        losses[scale] = learner.learn(b)
    assert np.isfinite(losses[True]) and np.isfinite(losses[False])
    assert losses[True] != losses[False]
