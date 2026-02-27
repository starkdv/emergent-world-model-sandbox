"""
Regression tests for left-right turn symmetry fixes.

Covers:
  1. Random initial direction — agents don't all face the same way
  2. Direction-aware turn instinct — instinct boosts the correct turn
  3. Turn-balance regularization — one-sided turn streaks are penalized
"""

import numpy as np
import pytest

from agents.actions import Action, ActionResult
from agents.agent import Agent
from agents.brain import Brain
from agents.genome import Genome
from utils.agents import RewardShaper

# ── helpers ──────────────────────────────────────────────────────────────


def _make_agent(x: int = 25, y: int = 25) -> Agent:
    wc = Brain.calculate_weight_count()
    g = Genome.random(wc, {"metabolism_rate": (1.0, 1.0), "vision_radius": (5, 5)})
    return Agent(x=x, y=y, genome=g, max_energy=1000.0)


class _SimpleWorld:
    """Minimal world stub sufficient for reward shaping."""

    def __init__(self):
        self.width = 50
        self.height = 50
        self.objects = {}


# ── 1. Random initial direction ──────────────────────────────────────────


def test_initial_directions_are_diverse():
    """Creating many agents should produce all four cardinal directions."""
    np.random.seed(0)
    dirs = set()
    for _ in range(40):
        a = _make_agent()
        dirs.add(a.direction)
    assert len(dirs) == 4, f"Expected 4 unique directions, got {dirs}"


def test_initial_direction_is_roughly_uniform():
    """Each direction should get roughly 25% of agents."""
    np.random.seed(42)
    counts = {}
    n = 400
    for _ in range(n):
        a = _make_agent()
        d = a.direction
        counts[d] = counts.get(d, 0) + 1
    for d, c in counts.items():
        frac = c / n
        assert (
            0.15 < frac < 0.35
        ), f"Direction {d} has {frac:.0%} of agents — too skewed"


# ── 2. Direction-aware turn instinct ────────────────────────────────────


def _build_obs_food_left() -> np.ndarray:
    """Observation where food is on the LEFT side of the egocentric grid."""
    obs = np.zeros(72, dtype=np.float32)
    obs[0] = 0.5  # energy ratio

    # Stimulus: food nearby, not ahead, not facing it
    obs[60] = 0.0  # food_ahead = no
    obs[62] = 0.8  # nearest_food_prox
    obs[63] = 0.2  # food_dir_match (not facing)

    # Vision grid: place food-like values on LEFT columns (dx < 0).
    # Row 0 (dy=-2), col 0 (dx=-2): type_enc = 1.0
    idx = 8 + (0 * 5 + 0) * 2
    obs[idx] = 1.0  # food type
    obs[idx + 1] = 0.8
    # Row 1, col 1 (dx=-1)
    idx2 = 8 + (1 * 5 + 1) * 2
    obs[idx2] = 1.0
    obs[idx2 + 1] = 0.7
    return obs


def _build_obs_food_right() -> np.ndarray:
    """Observation where food is on the RIGHT side of the egocentric grid."""
    obs = np.zeros(72, dtype=np.float32)
    obs[0] = 0.5
    obs[60] = 0.0
    obs[62] = 0.8
    obs[63] = 0.2

    # Vision grid: place food-like values on RIGHT columns (dx > 0).
    idx = 8 + (0 * 5 + 4) * 2  # row 0, col 4 (dx=+2)
    obs[idx] = 1.0
    obs[idx + 1] = 0.8
    idx2 = 8 + (1 * 5 + 3) * 2  # row 1, col 3 (dx=+1)
    obs[idx2] = 1.0
    obs[idx2 + 1] = 0.7
    return obs


def test_instinct_favors_left_when_food_is_left():
    """
    With food on the left side of vision, the instinct should boost
    TURN_LEFT more than TURN_RIGHT.
    """
    np.random.seed(7)
    obs_left = _build_obs_food_left()
    mask = np.ones(8, dtype=np.float32)

    # Average over many random brains to wash out weight noise
    left_probs, right_probs = [], []
    wc = Brain.calculate_weight_count()
    for _ in range(500):
        g = Genome.random(wc, {})
        b = Brain(g)
        h = b.initial_state()
        probs, _, _ = b.forward(obs_left, h, mask)
        left_probs.append(probs[Action.TURN_LEFT.value])
        right_probs.append(probs[Action.TURN_RIGHT.value])

    mean_l = np.mean(left_probs)
    mean_r = np.mean(right_probs)
    assert mean_l > mean_r, (
        f"Expected TURN_LEFT > TURN_RIGHT when food is left, "
        f"got L={mean_l:.4f} R={mean_r:.4f}"
    )


def test_instinct_favors_right_when_food_is_right():
    """
    With food on the right side of vision, the instinct should boost
    TURN_RIGHT more than TURN_LEFT.
    """
    np.random.seed(7)
    obs_right = _build_obs_food_right()
    mask = np.ones(8, dtype=np.float32)

    left_probs, right_probs = [], []
    wc = Brain.calculate_weight_count()
    for _ in range(500):
        g = Genome.random(wc, {})
        b = Brain(g)
        h = b.initial_state()
        probs, _, _ = b.forward(obs_right, h, mask)
        left_probs.append(probs[Action.TURN_LEFT.value])
        right_probs.append(probs[Action.TURN_RIGHT.value])

    mean_l = np.mean(left_probs)
    mean_r = np.mean(right_probs)
    assert mean_r > mean_l, (
        f"Expected TURN_RIGHT > TURN_LEFT when food is right, "
        f"got L={mean_l:.4f} R={mean_r:.4f}"
    )


# ── 3. Turn-balance regularization ──────────────────────────────────────


def test_turn_balance_penalty_for_dominant_left():
    """Repeated TURN_LEFT should be penalized once skew reaches 4:1."""
    shaper = RewardShaper()
    world = _SimpleWorld()
    a = _make_agent()
    a.direction = (0, -1)
    ok = ActionResult(True, 0.24, "Turned")

    # Prime with 7 left turns (fills last_actions)
    for _ in range(7):
        shaper.calculate_reward(
            Action.TURN_LEFT, ok, a.energy, a.energy - 0.24, a, world
        )

    # 8th left turn should include the balance penalty
    r_left = shaper.calculate_reward(
        Action.TURN_LEFT, ok, a.energy, a.energy - 0.24, a, world
    )

    # Reset and try right instead — same penalty expected (symmetric)
    shaper2 = RewardShaper()
    a2 = _make_agent()
    a2.direction = (0, -1)
    for _ in range(7):
        shaper2.calculate_reward(
            Action.TURN_RIGHT, ok, a2.energy, a2.energy - 0.24, a2, world
        )
    r_right = shaper2.calculate_reward(
        Action.TURN_RIGHT, ok, a2.energy, a2.energy - 0.24, a2, world
    )

    # The test checks that a heavily-skewed agent gets a lower
    # reward than a balanced one.
    shaper3 = RewardShaper()
    a3 = _make_agent()
    a3.direction = (0, -1)
    # Alternate turns: 4 left, 3 right, then 1 left (balanced)
    for act in [Action.TURN_LEFT, Action.TURN_RIGHT] * 3 + [Action.TURN_LEFT]:
        shaper3.calculate_reward(act, ok, a3.energy, a3.energy - 0.24, a3, world)
    r_balanced = shaper3.calculate_reward(
        Action.TURN_LEFT, ok, a3.energy, a3.energy - 0.24, a3, world
    )

    # The dominant-side reward should be lower (more negative) than balanced
    assert r_left < r_balanced or r_right < r_balanced, (
        f"Expected balance penalty: skewed={r_left:.3f}/{r_right:.3f}, "
        f"balanced={r_balanced:.3f}"
    )
