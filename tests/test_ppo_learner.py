"""
Tests for the PPO sequence learner (Brain v3 Phase 3b).

Covers:
- GAE(λ) math against hand-computed values
- Sequence chunking: finalize on length, terminal finalize, padding/valid
- Full-network backprop: encoder/attention AND GRU weights change (v2 + v3)
- Lamarckian sync: trained weights land in the genome
- PPO clip behaviour: huge off-policy ratios don't explode the policy
- Agent integration: decide_with_logprob path, death mark_done,
  offspring inherit the algorithm
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agents.agent import Agent  # noqa: E402
from agents.brain import calculate_weight_count_for_config, create_brain  # noqa: E402
from agents.genome import Genome, create_default_trait_config  # noqa: E402
from agents.ppo import PPOSequenceLearner, compute_gae  # noqa: E402

V3_CFG = {"version": 3}


@pytest.fixture(autouse=True)
def _reset_agent_class_config():
    saved_brain, saved_instinct = Agent.brain_config, Agent.instinct_config
    Agent.brain_config = None
    Agent.instinct_config = None
    yield
    Agent.brain_config, Agent.instinct_config = saved_brain, saved_instinct


def _make_brain(version=2):
    cfg = V3_CFG if version == 3 else None
    wc = calculate_weight_count_for_config(cfg)
    genome = Genome.random(wc, create_default_trait_config())
    return create_brain(genome, cfg)


def _fill_learner(learner, brain, steps=80, seed=0):
    """Feed time-ordered random steps so chunks get finalized."""
    rng = np.random.default_rng(seed)
    h = brain.initial_state()
    for _ in range(steps):
        obs = rng.random(72).astype(np.float32)
        mask = np.ones(8, dtype=np.float32)
        action, h_next, _, logprob = brain.decide_with_logprob(obs, h, action_mask=mask)
        learner.store_step(
            observation=obs,
            hidden_before=h,
            action=action.value,
            reward=float(rng.standard_normal()),
            next_observation=rng.random(72).astype(np.float32),
            done=False,
            logprob=logprob,
            action_mask=mask,
        )
        h = h_next


class TestGAE:
    def test_matches_hand_computation(self):
        gamma, lam = 0.9, 0.8
        rewards = np.array([1.0, 0.0, 2.0])
        values = np.array([0.5, 1.0, 1.5])
        bootstrap = 2.0
        dones = np.zeros(3)

        # Hand computation, backwards:
        # δ2 = 2 + 0.9*2.0 - 1.5 = 2.3            ; A2 = 2.3
        # δ1 = 0 + 0.9*1.5 - 1.0 = 0.35           ; A1 = 0.35 + 0.72*2.3 = 2.006
        # δ0 = 1 + 0.9*1.0 - 0.5 = 1.4            ; A0 = 1.4 + 0.72*2.006 = 2.84432
        adv, targets = compute_gae(rewards, values, bootstrap, dones, gamma, lam)
        assert adv == pytest.approx([2.84432, 2.006, 2.3])
        assert targets == pytest.approx(np.array([2.84432, 2.006, 2.3]) + values)

    def test_done_breaks_bootstrap_and_recursion(self):
        gamma, lam = 0.9, 0.8
        rewards = np.array([1.0, -1.0])
        values = np.array([0.5, 0.5])
        dones = np.array([0.0, 1.0])  # episode ends at step 1
        adv, _ = compute_gae(rewards, values, 99.0, dones, gamma, lam)
        # δ1 = -1 + 0 - 0.5 = -1.5 (bootstrap killed by done)
        assert adv[1] == pytest.approx(-1.5)
        # δ0 = 1 + 0.9*0.5 - 0.5 = 0.95 ; A0 = 0.95 + 0.72*(-1.5)
        assert adv[0] == pytest.approx(0.95 + 0.72 * -1.5)

    def test_lambda_zero_is_td0(self):
        rewards = np.array([1.0, 2.0])
        values = np.array([0.3, 0.7])
        adv, _ = compute_gae(rewards, values, 1.0, np.zeros(2), 0.9, 0.0)
        assert adv[0] == pytest.approx(1.0 + 0.9 * 0.7 - 0.3)
        assert adv[1] == pytest.approx(2.0 + 0.9 * 1.0 - 0.7)


class TestChunking:
    def _learner(self, seq_len=4):
        return PPOSequenceLearner(seq_len=seq_len, batch_size=2)

    def _step_kwargs(self, h_size=32, done=False):
        return dict(
            observation=np.zeros(72, dtype=np.float32),
            hidden_before=np.zeros(h_size, dtype=np.float32),
            action=1,
            reward=0.5,
            next_observation=np.ones(72, dtype=np.float32),
            done=done,
            logprob=-1.0,
            action_mask=np.ones(8, dtype=np.float32),
        )

    def test_finalizes_every_seq_len_steps(self):
        learner = self._learner(seq_len=4)
        for _ in range(9):
            learner.store_step(**self._step_kwargs())
        assert len(learner.replay_buffer) == 2  # two full chunks, 1 leftover
        chunk = learner.replay_buffer._chunks[0]
        assert chunk.valid.sum() == 4
        assert chunk.obs.shape == (4, 72)

    def test_terminal_step_finalizes_partial_chunk_with_padding(self):
        learner = self._learner(seq_len=4)
        learner.store_step(**self._step_kwargs())
        learner.store_step(**self._step_kwargs(done=True))
        assert len(learner.replay_buffer) == 1
        chunk = learner.replay_buffer._chunks[0]
        assert chunk.valid.tolist() == [1.0, 1.0, 0.0, 0.0]
        assert chunk.dones[1] == 1.0
        assert chunk.dones[2] == 1.0  # padding marked done (no bootstrap leak)

    def test_mark_done_flags_last_step(self):
        learner = self._learner(seq_len=4)
        learner.store_step(**self._step_kwargs())
        terminal = np.full(72, 0.5, dtype=np.float32)
        learner.mark_done(terminal)
        chunk = learner.replay_buffer._chunks[0]
        assert chunk.dones[0] == 1.0
        assert np.array_equal(chunk.bootstrap_obs, terminal)


class TestFullNetworkLearning:
    @pytest.mark.parametrize("version", [2, 3])
    def test_all_parameter_groups_receive_gradients(self, version):
        """The defining upgrade over A2C: encoder/attention and GRU
        weights must CHANGE after learning, not just the heads."""
        brain = _make_brain(version)
        learner = PPOSequenceLearner(batch_size=4, seq_len=8)
        _fill_learner(learner, brain, steps=64)

        before = {n: a.copy() for n, a in brain.named_params.items()}
        loss = learner.learn(brain)
        assert np.isfinite(loss)

        changed = [
            n for n, a in brain.named_params.items() if not np.array_equal(before[n], a)
        ]
        # Recurrent core must learn
        assert any(n.startswith("gru.") for n in changed), changed
        # Perception must learn (encoder for v2, tile/attention for v3)
        if version == 2:
            assert any(n.startswith("encoder.") for n in changed), changed
        else:
            assert any(n.startswith("tile_embed.") for n in changed), changed
            assert any(n.startswith("attn.") for n in changed), changed
        # Heads still learn too
        assert any(n.startswith("policy.") for n in changed), changed
        assert any(n.startswith("value.") for n in changed), changed

    def test_lamarckian_sync_to_genome(self):
        brain = _make_brain(2)
        learner = PPOSequenceLearner(batch_size=4, seq_len=8)
        _fill_learner(learner, brain, steps=64)

        genome_before = brain.genome.weights.copy()
        learner.learn(brain)
        assert not np.array_equal(genome_before, brain.genome.weights)
        # Rebuilt brain (offspring path) reproduces the trained network
        rebuilt = create_brain(brain.genome, None)
        assert np.allclose(
            rebuilt.named_params["policy.W"],
            brain.named_params["policy.W"],
            atol=1e-6,
        )

    def test_repeated_updates_stay_finite(self):
        brain = _make_brain(2)
        learner = PPOSequenceLearner(batch_size=4, seq_len=8, epochs=2)
        _fill_learner(learner, brain, steps=64)
        for _ in range(10):
            loss = learner.learn(brain)
            assert np.isfinite(loss)
        assert all(np.all(np.isfinite(a)) for a in brain.named_params.values())

    def test_clipping_bounds_off_policy_updates(self):
        """With wildly wrong stored log-probs (ratio >> 1+ε), the clipped
        objective must keep the policy finite and close to its start —
        the guarantee that makes replayed/instinct-shaped data safe."""
        brain = _make_brain(2)
        learner = PPOSequenceLearner(batch_size=4, seq_len=8, epochs=1)
        _fill_learner(learner, brain, steps=64)
        # Corrupt behaviour log-probs: pretend actions were near-impossible
        for chunk in learner.replay_buffer._chunks:
            chunk.logprobs[:] = -15.0  # ratio = exp(new + 15) — enormous

        before = brain.named_params["policy.W"].copy()
        loss = learner.learn(brain)
        assert np.isfinite(loss)
        delta = np.abs(brain.named_params["policy.W"] - before).max()
        assert delta < 0.05  # grad-clip + ratio-clip keep the step small


class TestAgentIntegration:
    def test_agent_uses_ppo_path_and_learns(self):
        from world.world import World
        from world.object_registry import register_builtin_objects

        register_builtin_objects()
        world = World(width=12, height=12, seed=3)

        wc = calculate_weight_count_for_config(None)
        agent = Agent(5, 5, Genome.random(wc, create_default_trait_config()))
        agent.enable_learning(
            algorithm="ppo", ppo_config={"seq_len": 4, "batch_size": 2}
        )
        assert agent.learner.algorithm == "ppo"
        world.add_agent(agent)

        for _ in range(40):
            if not agent.alive:
                break
            agent.update(world)

        assert len(agent.learner.replay_buffer) > 0
        loss = agent.learner.learn(agent.brain)
        assert np.isfinite(loss)

    def test_offspring_inherits_ppo_algorithm(self):
        wc = calculate_weight_count_for_config(None)
        parent = Agent(5, 5, Genome.random(wc, create_default_trait_config()))
        parent.enable_learning(algorithm="ppo", ppo_config={"seq_len": 4})
        parent.energy = parent.max_energy
        parent.age = 200

        from world.world import World

        world = World(width=12, height=12, seed=3)
        world.add_agent(parent)
        offspring = parent.reproduce(world)
        assert offspring is not None
        assert offspring.learner.algorithm == "ppo"
        assert offspring.learner.seq_len == 4

    def test_fallback_to_a2c_without_torch(self, monkeypatch):
        import agents.ppo as ppo_module

        monkeypatch.setattr(ppo_module, "TORCH_AVAILABLE", False)
        wc = calculate_weight_count_for_config(None)
        agent = Agent(5, 5, Genome.random(wc, create_default_trait_config()))
        agent.enable_learning(algorithm="ppo")
        assert agent.learner.algorithm == "a2c"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
