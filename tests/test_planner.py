"""
Tests for the latent rollout planner (agents/planner.py).

Covers the legacy random-shooting controller and the P1 upgrades
(policy-guided rollouts, policy-biased first action, reward/value
normalisation, and plan commitment), plus the running-stat helper.
"""

import numpy as np
import pytest

from agents.planner import LatentPlanner, _RunningStat


class StubBrain:
    """Minimal brain exposing exactly what the planner calls.

    Reward is 1.0 iff the chosen action equals ``target`` (else 0), so the
    best first action is unambiguously ``target`` when depth == 1.
    """

    def __init__(self, n=4, target=1, hidden=3, favor_target=False):
        self.output_size = n
        self.target = target
        self.hidden = hidden
        self.favor_target = favor_target

    def predict_next_latent(self, h, a):
        z = np.zeros(self.hidden, dtype=np.float32)
        r = 1.0 if a == self.target else 0.0
        return z, r

    def _gru_step(self, z, h):
        return h  # identity memory

    def _value(self, z, h):
        return 0.0

    def policy_from_hidden(self, h, action_mask=None):
        logits = np.zeros(self.output_size, dtype=np.float64)
        if self.favor_target:
            logits[self.target] = 4.0
        if action_mask is not None:
            logits = np.where(action_mask > 0, logits, -1e9)
        e = np.exp(logits - logits.max())
        return e / e.sum()


def _h(brain):
    return np.zeros(brain.hidden, dtype=np.float32)


# --- running stat --------------------------------------------------------


def test_running_stat_passthrough_then_zscore():
    s = _RunningStat()
    # before warmup, z() is the identity
    for _ in range(5):
        s.update(10.0)
    assert s.z(10.0, warmup=20) == 10.0
    # after enough samples with spread, z-scores toward ~0 mean
    s2 = _RunningStat()
    for v in np.linspace(0, 100, 100):
        s2.update(float(v))
    z = s2.z(s2.mean, warmup=20)
    assert abs(z) < 1e-6  # the mean maps to ~0


# --- core behaviour ------------------------------------------------------


def test_shooting_picks_best_first_action():
    np.random.seed(0)
    brain = StubBrain(n=4, target=2)
    p = LatentPlanner(depth=1, samples=40, strategy="shooting")
    # with depth 1 the score is r(first)+gamma*V, so the best is `target`
    assert p.plan(brain, _h(brain)) == 2


def test_policy_shooting_picks_best_first_action():
    np.random.seed(0)
    brain = StubBrain(n=4, target=3, favor_target=True)
    p = LatentPlanner(
        depth=1, samples=40, strategy="policy_shooting", first_action="policy"
    )
    assert p.plan(brain, _h(brain)) == 3


def test_returns_valid_action_all_strategies():
    np.random.seed(1)
    brain = StubBrain(n=5, target=1, favor_target=True)
    for strat, fa in [
        ("shooting", "uniform"),
        ("policy_shooting", "policy"),
        ("policy_shooting", "policy_topk"),
    ]:
        p = LatentPlanner(depth=3, samples=8, strategy=strat, first_action=fa, topk=3)
        a = p.plan(brain, _h(brain))
        assert 0 <= a < brain.output_size


def test_action_mask_respected_for_first_action():
    np.random.seed(2)
    brain = StubBrain(n=4, target=0)  # target is masked off
    mask = np.array([0, 1, 1, 0])  # only actions 1,2 valid
    p = LatentPlanner(depth=1, samples=30, strategy="shooting")
    for _ in range(10):
        a = p.plan(brain, _h(brain), action_mask=mask)
        assert a in (1, 2)


def test_normalize_runs_and_returns_valid():
    np.random.seed(3)
    brain = StubBrain(n=4, target=2)
    p = LatentPlanner(depth=3, samples=10, strategy="policy_shooting", normalize=True)
    a = p.plan(brain, _h(brain))
    assert 0 <= a < brain.output_size
    assert p._rstat.n > 0  # stats were updated


def test_commit_queues_and_replays():
    np.random.seed(4)
    brain = StubBrain(n=4, target=1)
    p = LatentPlanner(depth=4, samples=6, strategy="policy_shooting", commit=3)
    first = p.plan(brain, _h(brain))
    # commit=3 → first action returned now, next 2 cached
    assert len(p._queue) == 2
    cached = list(p._queue)
    a2 = p.plan(brain, _h(brain))
    a3 = p.plan(brain, _h(brain))
    assert [a2, a3] == cached
    assert len(p._queue) == 0  # exhausted → next plan() re-searches
    assert isinstance(first, int)


def test_commit_one_never_queues():
    np.random.seed(5)
    brain = StubBrain(n=4, target=1)
    p = LatentPlanner(depth=3, samples=6, strategy="shooting", commit=1)
    p.plan(brain, _h(brain))
    assert p._queue == []


def test_from_config_defaults_are_legacy():
    p = LatentPlanner.from_config({})
    assert p.strategy == "shooting"
    assert p.first_action == "uniform"
    assert p.normalize is False
    assert p.commit == 1
    assert p._policy_rollout is False


def test_from_config_reads_p1_options():
    p = LatentPlanner.from_config(
        {
            "strategy": "policy_shooting",
            "first_action": "policy_topk",
            "topk": 4,
            "normalize": True,
            "commit": 2,
            "depth": 5,
            "samples": 12,
        }
    )
    assert p.strategy == "policy_shooting"
    assert p._policy_rollout is True
    assert p.first_action == "policy_topk"
    assert p.topk == 4 and p.normalize is True and p.commit == 2
    assert p.depth == 5 and p.samples == 12


# --- P2: CEM + lambda-returns ------------------------------------------------


def test_cem_picks_best_first_action():
    np.random.seed(0)
    brain = StubBrain(n=4, target=2)
    p = LatentPlanner(depth=1, samples=20, strategy="cem", cem_iters=3)
    assert p.plan(brain, _h(brain)) == 2


def test_cem_respects_mask():
    np.random.seed(1)
    brain = StubBrain(n=4, target=0)  # best action masked off
    mask = np.array([0, 1, 1, 0])
    p = LatentPlanner(depth=2, samples=16, strategy="cem", cem_iters=2)
    for _ in range(5):
        a = p.plan(brain, _h(brain), action_mask=mask)
        assert a in (1, 2)


def test_cem_commit_queues():
    np.random.seed(2)
    brain = StubBrain(n=4, target=1)
    p = LatentPlanner(depth=4, samples=12, strategy="cem", cem_iters=2, commit=3)
    p.plan(brain, _h(brain))
    assert len(p._queue) == 2  # commit-1 cached


def test_score_lambda_extremes():
    p = LatentPlanner(gamma=0.95, lam=1.0)
    # lam=1: reward-sum + terminal bootstrap
    rewards, values = [1.0, 1.0], [10.0, 20.0]
    s1 = p._score(rewards, values)
    assert abs(s1 - (1 + 0.95 * 1 + 0.95**2 * 20)) < 1e-9
    # lam=0: one-step bootstrap recursion
    p.lam = 0.0
    s0 = p._score(rewards, values)
    assert abs(s0 - (1 + 0.95 * 10)) < 1e-9
    assert s0 != s1


def test_lambda_rollout_runs():
    np.random.seed(3)
    brain = StubBrain(n=4, target=2)
    p = LatentPlanner(depth=3, samples=8, strategy="policy_shooting", lam=0.5)
    a = p.plan(brain, _h(brain))
    assert 0 <= a < brain.output_size


def test_from_config_reads_p2_options():
    p = LatentPlanner.from_config(
        {"strategy": "cem", "lam": 0.8, "cem_iters": 5, "cem_elite_frac": 0.2}
    )
    assert p.strategy == "cem"
    assert p.lam == 0.8 and p.cem_iters == 5 and p.cem_elite_frac == 0.2


# --- warmup schedule (P2/P3 kick in after the world model trains) ------------


def test_effective_strategy_warmup_switch():
    p = LatentPlanner(
        strategy="cem", warmup_ticks=1000, warmup_strategy="policy_shooting"
    )
    assert p.effective_strategy(0) == "policy_shooting"
    assert p.effective_strategy(999) == "policy_shooting"
    assert p.effective_strategy(1000) == "cem"
    assert p.effective_strategy(5000) == "cem"


def test_no_warmup_uses_configured_strategy():
    p = LatentPlanner(strategy="cem", warmup_ticks=0)
    assert p.effective_strategy(0) == "cem"


def test_plan_warmup_then_main_returns_valid():
    np.random.seed(7)
    brain = StubBrain(n=4, target=2)
    p = LatentPlanner(
        depth=2,
        samples=6,
        strategy="cem",
        warmup_ticks=100,
        warmup_strategy="policy_shooting",
    )
    a_warm = p.plan(brain, _h(brain), tick=0)  # warmup: policy_shooting
    a_main = p.plan(brain, _h(brain), tick=500)  # past warmup: cem
    assert 0 <= a_warm < brain.output_size
    assert 0 <= a_main < brain.output_size


def test_from_config_reads_warmup():
    p = LatentPlanner.from_config(
        {"strategy": "cem", "warmup_ticks": 6000, "warmup_strategy": "policy_shooting"}
    )
    assert p.warmup_ticks == 6000
    assert p.warmup_strategy == "policy_shooting"
    assert p.effective_strategy(3000) == "policy_shooting"
    assert p.effective_strategy(6000) == "cem"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
