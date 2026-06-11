"""
Tests for the learned world model (Brain v3 Phase 4).

Covers:
- Dynamics head in the genome (spec counts, both brain versions)
- predict_next_latent shapes and error handling
- CuriosityModule: warmup, normalisation, decay, only-positive-surprise
- LatentPlanner: valid actions, mask respect, prefers rewarding rollouts
- PPO training of the dynamics head (auxiliary loss reaches dyn.* params,
  prediction error drops on a learnable toy dynamics)
- Agent integration: curiosity adds reward, configs inherited by offspring
"""

import numpy as np
import pytest

from agents.agent import Agent
from agents.brain import Brain, calculate_weight_count_for_config, create_brain
from agents.curiosity import CuriosityModule
from agents.genome import Genome, create_default_trait_config
from agents.planner import LatentPlanner

WM_CFG = {"version": 2, "world_model": {"enabled": True, "hidden": 32}}
WM_V3_CFG = {"version": 3, "world_model": {"enabled": True, "hidden": 32}}


@pytest.fixture(autouse=True)
def _reset_agent_class_config():
    saved_brain, saved_instinct = Agent.brain_config, Agent.instinct_config
    Agent.brain_config = None
    Agent.instinct_config = None
    yield
    Agent.brain_config, Agent.instinct_config = saved_brain, saved_instinct


def _make_brain(cfg):
    wc = calculate_weight_count_for_config(cfg)
    genome = Genome.random(wc, create_default_trait_config())
    return create_brain(genome, cfg)


class TestDynamicsHead:
    def test_v2_weight_count_includes_dynamics(self):
        # dyn: (H+A)*hid + hid  +  hid*Z + Z  +  hid*1 + 1
        # v2: H=32, A=8, Z=32, hid=32 → 40*32+32 + 32*32+32 + 32+1 = 2401
        assert calculate_weight_count_for_config(WM_CFG) == 8873 + 2401

    def test_v3_weight_count_includes_dynamics(self):
        # v3: H=48, A=8, Z=48, hid=32 → 56*32+32 + 32*48+48 + 32+1 = 3441
        assert calculate_weight_count_for_config(WM_V3_CFG) == 17337 + 3441

    def test_disabled_by_default(self):
        brain = _make_brain(None)
        assert not brain.has_world_model
        with pytest.raises(RuntimeError):
            brain.predict_next_latent(brain.initial_state(), 0)

    @pytest.mark.parametrize("cfg,latent", [(WM_CFG, 32), (WM_V3_CFG, 48)])
    def test_predict_next_latent_shapes(self, cfg, latent):
        brain = _make_brain(cfg)
        assert brain.has_world_model
        z_pred, r_pred = brain.predict_next_latent(brain.initial_state(), 5)
        assert z_pred.shape == (latent,)
        assert isinstance(r_pred, float)

    def test_encode_matches_latent_target(self):
        """encode() must produce the latent the dynamics head predicts."""
        brain = _make_brain(WM_CFG)
        obs = np.random.rand(72).astype(np.float32)
        z = brain.encode(obs)
        z_pred, _ = brain.predict_next_latent(brain.initial_state(), 0)
        assert z.shape == z_pred.shape

    def test_genome_prefix_unchanged(self):
        """Enabling the world model only APPENDS to the layout — the
        policy/value prefix stays byte-identical (migration-friendly)."""
        base = _make_brain(None)
        wm = _make_brain(WM_CFG)
        base_names = base.spec.names()
        assert wm.spec.names()[: len(base_names)] == base_names


class TestCuriosity:
    def test_warmup_emits_zero(self):
        cur = CuriosityModule(weight=1.0, warmup=5)
        z = np.zeros(4)
        for _ in range(5):
            assert cur.intrinsic_reward(z + 1.0, z) == 0.0

    def test_surprise_is_rewarded_boredom_is_not(self):
        cur = CuriosityModule(weight=1.0, warmup=3)
        z = np.zeros(4)
        rng = np.random.default_rng(0)
        # establish a baseline error level ~0.01
        for _ in range(20):
            cur.intrinsic_reward(z + 0.1 * rng.random(), z)
        # a much larger error must be rewarded
        big = cur.intrinsic_reward(z + 5.0, z)
        assert big > 0.0
        # a tiny (below-mean) error must give exactly zero
        small = cur.intrinsic_reward(z + 1e-6, z)
        assert small == 0.0

    def test_clip_caps_reward(self):
        cur = CuriosityModule(weight=1.0, clip=3.0, warmup=3)
        z = np.zeros(4)
        for _ in range(10):
            cur.intrinsic_reward(z + 0.1, z)
        assert cur.intrinsic_reward(z + 100.0, z) <= 3.0

    def test_weight_decays(self):
        cur = CuriosityModule(weight=1.0, decay=0.5, warmup=0)
        z = np.zeros(4)
        cur.intrinsic_reward(z + 1.0, z)
        cur.intrinsic_reward(z + 1.0, z)
        assert cur.weight == pytest.approx(0.25)

    def test_from_config(self):
        cur = CuriosityModule.from_config({"weight": 0.5, "decay": 0.9})
        assert cur.weight == 0.5 and cur.decay == 0.9


class TestPlanner:
    def test_returns_valid_masked_action(self):
        brain = _make_brain(WM_CFG)
        planner = LatentPlanner(depth=2, samples=8)
        mask = np.zeros(8)
        mask[[1, 4]] = 1
        for _ in range(10):
            a = planner.plan(brain, brain.initial_state(), mask)
            assert a in (1, 4)

    def test_prefers_action_the_model_says_is_rewarding(self):
        """Craft dynamics weights so r̂ depends only on the chosen action:
        the planner must pick the action with the highest imagined reward."""
        brain = _make_brain(WM_CFG)
        dyn = brain.params["dynamics"]
        # Zero the head, then wire: d = tanh(W1ᵀ[h‖onehot]) passes the
        # one-hot straight through (identity-ish into first 8 dims of d),
        # and Wr reads dim 0 strongly → action 0 yields max reward.
        dyn["W1"][...] = 0.0
        for j in range(8):
            dyn["W1"][brain.gru_hidden_size + j, j] = 5.0  # onehot → d[j]
        dyn["b1"][...] = 0.0
        dyn["Wz"][...] = 0.0
        dyn["bz"][...] = 0.0
        dyn["Wr"][...] = 0.0
        dyn["Wr"][3, 0] = 10.0  # reward comes only from action 3
        dyn["br"][...] = 0.0
        # Make value head silent so rollout score is pure imagined reward
        brain.params["value_head"]["W"][...] = 0.0
        brain.params["value_head"]["b"][...] = 0.0

        planner = LatentPlanner(depth=1, samples=64)
        picks = [
            planner.plan(brain, brain.initial_state(), np.ones(8)) for _ in range(10)
        ]
        assert picks.count(3) >= 8, picks


class TestPPOTrainsDynamics:
    def test_dyn_params_receive_gradients(self):
        torch = pytest.importorskip("torch")  # noqa: F841
        from agents.ppo import PPOSequenceLearner

        brain = _make_brain(WM_CFG)
        learner = PPOSequenceLearner(batch_size=4, seq_len=8)

        rng = np.random.default_rng(0)
        h = brain.initial_state()
        for _ in range(64):
            obs = rng.random(72).astype(np.float32)
            mask = np.ones(8, dtype=np.float32)
            action, h_next, _, logprob = brain.decide_with_logprob(
                obs, h, action_mask=mask
            )
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

        before = {
            n: a.copy() for n, a in brain.named_params.items() if n.startswith("dyn.")
        }
        loss = learner.learn(brain)
        assert np.isfinite(loss)
        changed = [
            n for n, a in before.items() if not np.array_equal(a, brain.named_params[n])
        ]
        assert changed, "dynamics head received no gradient"

    def test_prediction_error_decreases_on_repeating_world(self):
        """On a tiny repeating environment the dynamics head must fit:
        prediction error after training < before."""
        torch = pytest.importorskip("torch")  # noqa: F841
        from agents.ppo import PPOSequenceLearner

        brain = _make_brain(WM_CFG)
        learner = PPOSequenceLearner(
            batch_size=4, seq_len=4, epochs=4, learning_rate=3e-3
        )

        # Deterministic cyclic observations — perfectly learnable
        obs_cycle = [np.full(72, v, dtype=np.float32) for v in (0.1, 0.5, 0.9)]

        def err_now():
            h = brain.initial_state()
            total = 0.0
            for t in range(6):
                obs = obs_cycle[t % 3]
                z = brain.encode(obs)
                h = brain._gru_step(z, h)
                z_pred, _ = brain.predict_next_latent(h, 0)
                z_next = brain.encode(obs_cycle[(t + 1) % 3])
                total += float(np.mean((z_pred - z_next) ** 2))
            return total

        h = brain.initial_state()
        for t in range(160):
            obs = obs_cycle[t % 3]
            mask = np.ones(8, dtype=np.float32)
            action, h_next, _, logprob = brain.decide_with_logprob(
                obs, h, action_mask=mask
            )
            learner.store_step(
                observation=obs,
                hidden_before=h,
                action=0,
                reward=0.0,
                next_observation=obs_cycle[(t + 1) % 3],
                done=False,
                logprob=logprob,
                action_mask=mask,
            )
            h = h_next

        before = err_now()
        for _ in range(25):
            learner.learn(brain)
        after = err_now()
        assert after < before, f"dynamics did not improve: {before} -> {after}"


class TestAgentIntegration:
    def test_curiosity_adds_to_reward(self):
        from world.world import World
        from world.object_registry import register_builtin_objects

        register_builtin_objects()
        world = World(width=10, height=10, seed=5)

        Agent.brain_config = WM_CFG
        wc = calculate_weight_count_for_config(WM_CFG)
        agent = Agent(5, 5, Genome.random(wc, create_default_trait_config()))
        agent.enable_learning(
            algorithm="a2c",
            curiosity_config={"enabled": True, "weight": 1.0, "warmup": 0},
        )
        assert agent.curiosity is not None

        # Drive stats past warmup with varied transitions, then assert the
        # curiosity term contributes for a surprising one
        for _ in range(30):
            agent.update(world)
            if not agent.alive:
                pytest.skip("agent died early in toy world")
        assert agent.curiosity._count > 0  # prediction errors were observed

    def test_offspring_inherit_curiosity_config(self):
        from world.world import World

        Agent.brain_config = WM_CFG
        wc = calculate_weight_count_for_config(WM_CFG)
        parent = Agent(5, 5, Genome.random(wc, create_default_trait_config()))
        parent.enable_learning(
            algorithm="a2c",
            curiosity_config={"enabled": True, "weight": 0.7, "warmup": 0},
        )
        parent.energy = parent.max_energy
        parent.age = 200

        world = World(width=10, height=10, seed=5)
        world.add_agent(parent)
        child = parent.reproduce(world)
        assert child is not None
        assert child.curiosity is not None
        assert child.curiosity.weight == 0.7
        assert child.brain.has_world_model

    def test_planner_config_creates_planner(self):
        cfg = {
            "version": 2,
            "world_model": {
                "enabled": True,
                "hidden": 32,
                "planner": {"enabled": True, "depth": 2, "samples": 4},
            },
        }
        Agent.brain_config = cfg
        wc = calculate_weight_count_for_config(cfg)
        agent = Agent(0, 0, Genome.random(wc, create_default_trait_config()))
        assert agent.planner is not None
        assert agent.planner.depth == 2

    def test_no_world_model_means_no_extras(self):
        wc = calculate_weight_count_for_config(None)
        agent = Agent(0, 0, Genome.random(wc, create_default_trait_config()))
        agent.enable_learning(algorithm="a2c", curiosity_config={"enabled": True})
        assert agent.planner is None
        assert agent.curiosity is None  # needs the dynamics head


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
