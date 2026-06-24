"""
Tests for W6c — reward-shaping diet (legacy vs minimal) and the metrics CSV.

The contract: the ``legacy`` preset is the default and is unchanged; the
``minimal`` preset rewards eat / death / energy-delta only and nothing else.
The metrics writer appends one aggregate row per call.

Author: Karan Vasa
"""

import csv
import os
from types import SimpleNamespace

import pytest

from agents.actions import Action, ActionResult
from utils.agents.learning_utils import (
    RewardConfig,
    RewardShaper,
    get_active_reward_config,
    set_active_reward_config,
)


@pytest.fixture(autouse=True)
def _reset_reward_cfg():
    prev = get_active_reward_config()
    yield
    set_active_reward_config(prev)


def _agent(energy=100.0, max_energy=200.0, alive=True, x=5, y=5):
    return SimpleNamespace(
        x=x, y=y, energy=energy, max_energy=max_energy, age=0, alive=alive,
        direction=(0, -1), inventory=[], inventory_size=5, fitness=0.0,
    )


def _ok(msg=""):
    return ActionResult(True, 0.1, msg)


def _fail(msg=""):
    return ActionResult(False, 0.1, msg)


class _MockWorld:
    """A world with no food and no tiles — nearest_edible returns None."""

    def __init__(self):
        self.width = self.height = 10
        self.objects = {}

    def nearest_edible(self, ax, ay, scan_r):
        return None


# ===========================================================================
# RewardConfig.from_dict
# ===========================================================================


class TestRewardConfig:
    def test_default_is_legacy(self):
        c = RewardConfig.from_dict(None)
        assert c.preset == "legacy"

    def test_unknown_preset_falls_back_to_legacy(self):
        assert RewardConfig.from_dict({"preset": "wild"}).preset == "legacy"

    def test_minimal_with_overrides(self):
        c = RewardConfig.from_dict({"preset": "minimal", "eat_base": 9.0})
        assert c.preset == "minimal"
        assert c.eat_base == 9.0


# ===========================================================================
# Minimal diet: eat / death / energy-delta ONLY
# ===========================================================================


class TestMinimalDiet:
    def _shaper(self):
        return RewardShaper(RewardConfig(preset="minimal"))

    def test_successful_eat_rewarded(self):
        s = self._shaper()
        a = _agent(energy=120.0)
        r = s.calculate_reward(Action.EAT, _ok("ate"), 100.0, 120.0, a, _MockWorld())
        # eat_base 5.0 + 20 gain * 0.2 = 9.0
        assert r == pytest.approx(9.0)

    def test_eat_with_no_gain_not_rewarded(self):
        s = self._shaper()
        a = _agent(energy=100.0)
        r = s.calculate_reward(Action.EAT, _ok("ate"), 100.0, 100.0, a, _MockWorld())
        assert r == pytest.approx(0.0)

    def test_metabolism_penalty_on_non_eat_loss(self):
        s = self._shaper()
        a = _agent(energy=99.0)
        r = s.calculate_reward(Action.WAIT, _ok(), 100.0, 99.0, a, _MockWorld())
        assert r == pytest.approx(-0.01)  # 1.0 lost * 0.01

    def test_death_penalty(self):
        s = self._shaper()
        a = _agent(energy=0.0, alive=False)
        r = s.calculate_reward(Action.WAIT, _ok(), 1.0, 0.0, a, _MockWorld())
        # metabolism: -0.01, death: -10.0
        assert r == pytest.approx(-10.01)

    def test_no_exploration_or_movement_reward(self):
        # A successful MOVE_FORWARD with no energy change earns nothing under
        # minimal (legacy would add exploration + new-tile bonuses).
        s = self._shaper()
        a = _agent(energy=100.0)
        r = s.calculate_reward(
            Action.MOVE_FORWARD, _ok("moved"), 100.0, 100.0, a, _MockWorld()
        )
        assert r == pytest.approx(0.0)

    def test_failed_eat_not_penalised_under_minimal(self):
        # Legacy heavily penalises EAT spam; minimal ignores it entirely.
        s = self._shaper()
        a = _agent(energy=100.0)
        r = s.calculate_reward(Action.EAT, _fail("no food"), 100.0, 100.0, a, _MockWorld())
        assert r == pytest.approx(0.0)


# ===========================================================================
# Legacy diet stays "rich" (sanity: it is NOT the minimal path)
# ===========================================================================


def test_legacy_move_forward_rewards_more_than_minimal():
    legacy = RewardShaper(RewardConfig(preset="legacy"))
    minimal = RewardShaper(RewardConfig(preset="minimal"))
    a1 = _agent(energy=100.0)
    a2 = _agent(energy=100.0)
    w = _MockWorld()
    # First MOVE establishes last_position; do two so movement registers.
    legacy.calculate_reward(Action.MOVE_FORWARD, _ok("moved"), 100.0, 100.0, a1, w)
    a1.x += 1
    r_legacy = legacy.calculate_reward(
        Action.MOVE_FORWARD, _ok("moved"), 100.0, 100.0, a1, w
    )
    minimal.calculate_reward(Action.MOVE_FORWARD, _ok("moved"), 100.0, 100.0, a2, w)
    a2.x += 1
    r_minimal = minimal.calculate_reward(
        Action.MOVE_FORWARD, _ok("moved"), 100.0, 100.0, a2, w
    )
    assert r_legacy > r_minimal  # dense shaping is active in legacy only


# ===========================================================================
# Metrics writer
# ===========================================================================


def test_metrics_writer_writes_rows(tmp_path):
    from utils.agents.metrics import MetricsWriter, FIELDS

    path = os.path.join(str(tmp_path), "m.csv")

    class _W:
        def __init__(self):
            self.tick = 100
            self.agents = {
                1: SimpleNamespace(alive=True, energy=120.0, age=50, fitness=2.0),
                2: SimpleNamespace(alive=True, energy=80.0, age=30, fitness=1.0),
                3: SimpleNamespace(alive=False, energy=0.0, age=10, fitness=0.0),
            }

        def get_cached_object_counts(self):
            return {
                "alive_agents": 2,
                "total_food": 7,
                "total_plants": 4,
                "total_seeds": 1,
            }

        def get_cached_soil_stats(self):
            return (0.55, 0.42)

    w = _W()
    mw = MetricsWriter(path)
    row = mw.record(w, generation=1)
    w.tick = 200
    mw.record(w, generation=2)
    mw.close()

    assert row["alive_agents"] == 2
    assert row["mean_energy"] == pytest.approx(100.0)  # (120+80)/2
    assert row["max_age"] == 50

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert list(rows[0].keys()) == FIELDS
    assert rows[1]["generation"] == "2"
    assert rows[1]["tick"] == "200"
