"""
Tests for Brain v3 — attention perception, larger GRU, [z,h] value head.

Covers:
- v3 ParamSpec weight count and version tag
- Factory dispatch (create_brain / calculate_weight_count_for_config)
- Forward pass: shapes, probabilities, masking, hidden-state evolution
- Attention sensitivity: vision content changes the latent
- rebind() across genome replacement (clone / inherit paths)
- Learner v3 path: heads update, genome sync, finite loss
- Agent + evolution integration with brain_config version 3
"""

import numpy as np
import pytest

from agents.agent import Agent
from agents.actions import Action
from agents.brain import Brain, calculate_weight_count_for_config, create_brain
from agents.brain.instincts import InstinctModule
from agents.brain.spec import DEFAULT_OBSERVATION_SPEC, build_brain_v3_param_spec
from agents.brain.v3 import BrainV3, make_positional_encoding
from agents.genome import Genome, create_default_trait_config

V3_CFG = {"version": 3}


@pytest.fixture(autouse=True)
def _reset_agent_class_config():
    """Keep class-level config from leaking between tests."""
    saved_brain, saved_instinct = Agent.brain_config, Agent.instinct_config
    Agent.brain_config = None
    Agent.instinct_config = None
    yield
    Agent.brain_config, Agent.instinct_config = saved_brain, saved_instinct


def _make_v3_brain(instincts=None):
    wc = calculate_weight_count_for_config(V3_CFG)
    genome = Genome.random(wc, create_default_trait_config())
    return create_brain(genome, V3_CFG, instincts=instincts)


class TestV3Spec:
    def test_weight_count_formula(self):
        e, s, h, v, a = 8, 40, 48, 16, 8
        z = s + e
        expected = (
            (22 * s + s)  # state encoder
            + (4 * e + e)  # tile embedding
            + (s * e)  # attn Wq
            + (e * e) * 2  # attn Wk, Wv
            + 3 * (z * h + h * h + h)  # GRU gates
            + (h * a + a)  # policy head
            + (z + h) * v
            + v  # value W1, b1
            + v
            + 1  # value W2, b2
        )
        assert build_brain_v3_param_spec().count() == expected == 17337

    def test_spec_version_tag(self):
        assert build_brain_v3_param_spec().version == 3

    def test_factory_dispatch(self):
        brain = _make_v3_brain()
        assert isinstance(brain, BrainV3)
        assert isinstance(brain, Brain)  # shares the public API
        # None / version 2 → legacy Brain
        wc2 = calculate_weight_count_for_config(None)
        v2 = create_brain(Genome.random(wc2, create_default_trait_config()), None)
        assert type(v2) is Brain

    def test_factory_respects_custom_sizes(self):
        cfg = {"version": 3, "v3": {"embed_dim": 4, "gru_hidden_size": 24}}
        wc = calculate_weight_count_for_config(cfg)
        brain = create_brain(Genome.random(wc, create_default_trait_config()), cfg)
        assert brain.embed_dim == 4
        assert brain.gru_hidden_size == 24
        assert brain.initial_state().shape == (24,)


class TestV3Forward:
    def test_forward_outputs(self):
        brain = _make_v3_brain()
        obs = np.random.rand(72).astype(np.float32)
        h = brain.initial_state()
        probs, value, h_next = brain.forward(obs, h)

        assert probs.shape == (8,)
        assert np.isclose(np.sum(probs), 1.0)
        assert np.all(probs >= 0)
        assert isinstance(value, float)
        assert h_next.shape == (48,)
        assert not np.array_equal(h, h_next)

    def test_action_masking(self):
        brain = _make_v3_brain()
        obs = np.random.rand(72).astype(np.float32)
        h = brain.initial_state()
        mask = np.zeros(8)
        mask[Action.WAIT.value] = 1
        for _ in range(50):
            action, h, _ = brain.decide(obs, h, action_mask=mask)
            assert action == Action.WAIT

    def test_instincts_apply_in_v3(self):
        wc = calculate_weight_count_for_config(V3_CFG)
        genome = Genome.random(wc, create_default_trait_config())
        with_instincts = create_brain(genome, V3_CFG)
        without = create_brain(genome, V3_CFG, instincts=InstinctModule(enabled=False))
        obs = np.zeros(72, dtype=np.float32)
        h = with_instincts.initial_state()
        probs_on, _, _ = with_instincts.forward(obs, h, action_mask=np.ones(8))
        probs_off, _, _ = without.forward(obs, h, action_mask=np.ones(8))
        assert probs_on[Action.PICK_UP.value] > probs_off[Action.PICK_UP.value]

    def test_vision_content_changes_latent(self):
        """Attention must be sensitive to what (and where) tiles contain."""
        brain = _make_v3_brain()
        spec = DEFAULT_OBSERVATION_SPEC
        obs_a = np.zeros(72, dtype=np.float32)
        obs_b = obs_a.copy()
        grid = obs_b[spec.vision].reshape(spec.vision_shape)
        grid[0, 2, 0] = 1.0  # food-like tile straight ahead

        z_a = brain._encode(obs_a)
        z_b = brain._encode(obs_b)
        assert z_a.shape == (48,)  # S + E = 40 + 8
        assert not np.allclose(z_a, z_b)

    def test_positional_encoding_distinguishes_tiles(self):
        """Same tile content at different positions → different latents
        (v2's flat vision had this for free; attention needs pos enc)."""
        brain = _make_v3_brain()
        spec = DEFAULT_OBSERVATION_SPEC
        obs_left = np.zeros(72, dtype=np.float32)
        obs_right = np.zeros(72, dtype=np.float32)
        obs_left[spec.vision].reshape(spec.vision_shape)[2, 0, 0] = 1.0
        obs_right[spec.vision].reshape(spec.vision_shape)[2, 4, 0] = 1.0
        assert not np.allclose(brain._encode(obs_left), brain._encode(obs_right))

    def test_positional_encoding_shape_and_range(self):
        pos = make_positional_encoding((5, 5, 2))
        assert pos.shape == (25, 2)
        assert pos.min() == -1.0 and pos.max() == 1.0

    def test_rebind_reflects_new_weights(self):
        brain = _make_v3_brain()
        obs = np.random.rand(72).astype(np.float32)
        h = brain.initial_state()
        probs_before, _, _ = brain.forward(obs, h)

        brain.genome.weights = np.random.randn(brain.spec.count()).astype(np.float32)
        brain.rebind(brain.genome)
        probs_after, _, _ = brain.forward(obs, h)
        assert not np.allclose(probs_before, probs_after)


class TestV3Learning:
    def _fill_buffer(self, agent, n=40):
        rng = np.random.default_rng(0)
        h_size = agent.brain.gru_hidden_size
        for _ in range(n):
            agent.learner.store_experience(
                rng.random(72).astype(np.float32),
                rng.standard_normal(h_size).astype(np.float32) * 0.1,
                int(rng.integers(0, 8)),
                float(rng.standard_normal()),
                rng.random(72).astype(np.float32),
                rng.standard_normal(h_size).astype(np.float32) * 0.1,
                False,
            )

    def test_learn_updates_heads_and_syncs_genome(self):
        Agent.brain_config = V3_CFG
        wc = calculate_weight_count_for_config(V3_CFG)
        agent = Agent(0, 0, Genome.random(wc, create_default_trait_config()))
        agent.enable_learning(compute_backend="numpy")
        self._fill_buffer(agent)

        policy_before = agent.brain.params["policy_head"]["W"].copy()
        value_before = agent.brain.params["value_mlp"]["W2"].copy()
        loss = agent.learner.learn(agent.brain)

        assert np.isfinite(loss)
        assert not np.array_equal(policy_before, agent.brain.params["policy_head"]["W"])
        assert not np.array_equal(value_before, agent.brain.params["value_mlp"]["W2"])
        # Lamarckian sync: genome reflects the updated parameters
        assert len(agent.genome.weights) == agent.brain.spec.count()
        rebuilt = create_brain(agent.genome, V3_CFG)
        assert np.allclose(
            rebuilt.params["policy_head"]["W"],
            agent.brain.params["policy_head"]["W"],
            atol=1e-6,
        )

    def test_learn_routes_v3_even_on_torch_backend(self):
        """The torch fast path only mirrors v2 — v3 must use its own path."""
        Agent.brain_config = V3_CFG
        wc = calculate_weight_count_for_config(V3_CFG)
        agent = Agent(0, 0, Genome.random(wc, create_default_trait_config()))
        agent.enable_learning(compute_backend="torch")  # may fall back to numpy
        self._fill_buffer(agent)
        loss = agent.learner.learn(agent.brain)
        assert np.isfinite(loss)


class TestV3Evolution:
    def test_agent_and_offspring_use_v3(self):
        from agents.evolution import clone_agent

        Agent.brain_config = V3_CFG
        wc = calculate_weight_count_for_config(V3_CFG)
        parent = Agent(0, 0, Genome.random(wc, create_default_trait_config()))
        assert isinstance(parent.brain, BrainV3)

        child = clone_agent(parent, mutate=True, mutation_std=0.05)
        assert isinstance(child.brain, BrainV3)
        assert child.brain.spec.version == 3
        assert len(child.genome.weights) == wc
        # Mutated child must still produce a valid policy
        probs, value, _ = child.brain.forward(
            np.random.rand(72).astype(np.float32), child.brain.initial_state()
        )
        assert np.isclose(probs.sum(), 1.0)

    def test_inherit_knowledge_v3(self):
        Agent.brain_config = V3_CFG
        wc = calculate_weight_count_for_config(V3_CFG)
        agent = Agent(0, 0, Genome.random(wc, create_default_trait_config()))
        donor = np.random.randn(wc).astype(np.float32)
        agent.inherit_knowledge(donor)
        assert isinstance(agent.brain, BrainV3)
        assert np.array_equal(agent.genome.weights, donor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
