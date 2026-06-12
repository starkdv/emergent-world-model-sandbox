"""
Tests for dream-based evolution (population world model + dream evolution).

Covers:
- Transitions CSV loader against the AsyncWorldModelLogger schema
- PopulationWorldModel: training reduces loss, learns a simple dynamics,
  predict shapes/clipping, save/load roundtrip
- evaluate_in_dream: finite fitness, rewards what the model rewards
- dream_evolution: selection improves dream fitness on a crafted model
- Fixed async-logger schema: obs column count matches the observation
"""

import csv
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agents.dream import (  # noqa: E402
    PopulationWorldModel,
    TransitionDataset,
    dream_evolution,
    evaluate_in_dream,
    load_transitions_csv,
)
from agents.actions import Action  # noqa: E402
from agents.brain import calculate_weight_count_for_config  # noqa: E402
from agents.genome import Genome, create_default_trait_config  # noqa: E402


def _toy_dataset(n=3000, obs_dim=72, seed=0):
    """Learnable toy dynamics: EAT strongly shifts obs[0], reward = obs[0]."""
    rng = np.random.default_rng(seed)
    obs = rng.random((n, obs_dim)).astype(np.float32)
    actions = rng.integers(0, 8, size=n)
    next_obs = obs.copy()
    next_obs[:, 0] = np.clip(obs[:, 0] + (actions == 5) * 0.5, 0, 1)
    rewards = obs[:, 0].astype(np.float32)
    dones = np.zeros(n, dtype=np.float32)
    return TransitionDataset(obs, actions.astype(np.int64), rewards, next_obs, dones)


class _EatRewardModel:
    """Duck-typed stand-in: EAT yields +1, everything else 0."""

    def predict(self, obs, action_idx):
        reward = 1.0 if action_idx == Action.EAT.value else 0.0
        return obs, reward, 0.0


class TestCSVLoader:
    def test_roundtrip_with_logger_schema(self, tmp_path):
        path = os.path.join(tmp_path, "transitions_test.csv")
        obs_dim = 72
        header = ["tick", "agent_id", "action", "action_value", "reward", "done"]
        header += [f"obs_{i}" for i in range(obs_dim)]
        header += [f"obs_next_{i}" for i in range(obs_dim)]

        rng = np.random.default_rng(1)
        rows = []
        for t in range(5):
            o = rng.random(obs_dim).round(5)
            o2 = rng.random(obs_dim).round(5)
            rows.append([t, 1, "EAT", 5, 0.5, "False"] + list(o) + list(o2))
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

        data = load_transitions_csv(path, obs_dim=obs_dim)
        assert len(data) == 5
        assert data.obs.shape == (5, obs_dim)
        assert data.actions.tolist() == [5] * 5
        assert data.rewards == pytest.approx([0.5] * 5)
        assert data.dones.tolist() == [0.0] * 5

    def test_async_logger_header_matches_observation_size(self, tmp_path):
        """Regression: the logger wrote 64 obs columns for 72-dim
        observations, misaligning every column after obs_63."""
        from utils.agents.perception import get_observation_size
        from utils.data.async_logger import AsyncWorldModelLogger

        logger = AsyncWorldModelLogger(output_dir=str(tmp_path))
        try:
            with open(logger.transitions_file) as f:
                header = f.readline().strip().split(",")
            n = get_observation_size()
            assert f"obs_{n - 1}" in header
            assert f"obs_next_{n - 1}" in header
            assert f"obs_{n}" not in header
        finally:
            logger.close()


class TestPopulationWorldModel:
    def test_training_reduces_loss(self):
        data = _toy_dataset()
        model = PopulationWorldModel(hidden=64, lr=3e-3)
        losses = model.fit(data, epochs=8, batch_size=256)
        assert losses[-1] < losses[0] * 0.5, losses

    def test_learns_action_conditional_dynamics(self):
        """After training, the model must know EAT raises obs[0] and
        predict reward ≈ obs[0]."""
        data = _toy_dataset()
        model = PopulationWorldModel(hidden=64, lr=3e-3)
        model.fit(data, epochs=50, batch_size=256)

        obs = np.full(72, 0.4, dtype=np.float32)
        next_eat, r_eat, _ = model.predict(obs, 5)
        next_wait, _, _ = model.predict(obs, 7)
        assert next_eat[0] - next_wait[0] > 0.15  # EAT shifts obs[0] up
        assert r_eat == pytest.approx(0.4, abs=0.15)  # reward tracks obs[0]

    def test_predict_shapes_and_clipping(self):
        model = PopulationWorldModel(hidden=32)
        obs = np.ones(72, dtype=np.float32)
        next_obs, reward, p_done = model.predict(obs, 0)
        assert next_obs.shape == (72,)
        assert np.all(next_obs <= 1.0) and np.all(next_obs >= 0.0)
        assert isinstance(reward, float) and isinstance(p_done, float)
        assert 0.0 <= p_done <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        model = PopulationWorldModel(hidden=32)
        obs = np.random.rand(72).astype(np.float32)
        before = model.predict(obs, 3)

        path = os.path.join(tmp_path, "wm.pt")
        model.save(path)
        loaded = PopulationWorldModel.load(path)
        after = loaded.predict(obs, 3)
        assert np.allclose(before[0], after[0], atol=1e-6)
        assert before[1] == pytest.approx(after[1], abs=1e-6)


class TestDreamEvaluation:
    def _genome(self):
        wc = calculate_weight_count_for_config(None)
        return Genome.random(wc, create_default_trait_config())

    def test_returns_finite_fitness(self):
        model = _EatRewardModel()
        seeds = np.random.rand(10, 72).astype(np.float32)
        fit = evaluate_in_dream(
            self._genome(),
            model,
            seeds,
            episodes=2,
            steps=10,
            rng=np.random.default_rng(0),
        )
        assert np.isfinite(fit)
        assert 0.0 <= fit <= 10.0  # at most +1 per step

    def test_eat_biased_genome_scores_higher(self):
        """A genome wired to always EAT must beat one wired to WAIT on
        a model that only rewards eating."""
        from agents.brain import create_brain

        model = _EatRewardModel()
        seeds = np.zeros((4, 72), dtype=np.float32)
        rng = np.random.default_rng(0)

        def biased_genome(action_idx):
            g = self._genome()
            brain = create_brain(g, None)
            brain.params["policy_head"]["W"][...] = 0.0
            brain.params["policy_head"]["b"][...] = -5.0
            brain.params["policy_head"]["b"][action_idx] = 5.0
            g.weights = brain.spec.pack(brain.named_params)
            return g

        fit_eat = evaluate_in_dream(
            biased_genome(Action.EAT.value), model, seeds, episodes=2, steps=20, rng=rng
        )
        fit_wait = evaluate_in_dream(
            biased_genome(Action.WAIT.value),
            model,
            seeds,
            episodes=2,
            steps=20,
            rng=rng,
        )
        assert fit_eat > fit_wait + 10  # ~+1/step vs ~0


class TestDreamEvolution:
    def test_selection_improves_dream_fitness(self):
        """On the EAT-rewarding model, dream evolution must discover
        eat-heavy policies: best fitness should rise substantially."""
        model = _EatRewardModel()
        seeds = np.zeros((4, 72), dtype=np.float32)

        scored = dream_evolution(
            model,
            seed_observations=seeds,
            population_size=16,
            generations=6,
            elite_count=3,
            mutation_std=0.3,
            episodes=2,
            steps=20,
            seed=0,
        )
        assert scored[0][0] == max(f for f, _ in scored)  # sorted best-first

        # Compare against the average fitness of unevolved random genomes
        rng = np.random.default_rng(1)
        wc = calculate_weight_count_for_config(None)
        baseline = np.mean(
            [
                evaluate_in_dream(
                    Genome.random(wc, create_default_trait_config()),
                    model,
                    seeds,
                    episodes=2,
                    steps=20,
                    rng=rng,
                )
                for _ in range(8)
            ]
        )
        assert scored[0][0] > baseline * 1.5, (scored[0][0], baseline)

    def test_population_size_and_genome_shapes(self):
        model = _EatRewardModel()
        seeds = np.zeros((2, 72), dtype=np.float32)
        scored = dream_evolution(
            model,
            seed_observations=seeds,
            population_size=6,
            generations=2,
            elite_count=2,
            episodes=1,
            steps=5,
            seed=0,
        )
        wc = calculate_weight_count_for_config(None)
        assert len(scored) == 6
        assert all(len(g.weights) == wc for _, g in scored)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
