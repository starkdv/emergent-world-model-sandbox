"""Regression tests for loop-breaking reward shaping."""

from types import SimpleNamespace

from agents.actions import Action, ActionResult
from utils.agents import RewardShaper


class DummyWorld:
    """Minimal world surface needed by RewardShaper in unit tests."""

    def __init__(self):
        self.width = 100
        self.height = 100
        self.objects = {}



def _make_agent(x: int = 10, y: int = 10):
    return SimpleNamespace(
        x=x,
        y=y,
        direction=(1, 0),
        energy=100.0,
        max_energy=100.0,
        age=10,
        max_age=1000,
        inventory=[],
        inventory_size=5,
        metabolism_rate=0.5,
        alive=True,
    )



def test_proactive_turn_reward_after_straight_run():
    """Turning after a straight run should be more rewarding than a cold turn."""
    world = DummyWorld()

    baseline_shaper = RewardShaper()
    baseline_agent = _make_agent()
    baseline_turn = baseline_shaper.calculate_reward(
        Action.TURN_LEFT,
        ActionResult(True, 0.5, "Turned left"),
        100.0,
        99.5,
        baseline_agent,
        world,
    )

    shaper = RewardShaper()
    agent = _make_agent()

    # Build a straight run of successful MOVE_FORWARD steps.
    for step in range(4):
        agent.x += 1
        shaper.calculate_reward(
            Action.MOVE_FORWARD,
            ActionResult(True, 0.2, "Moved forward"),
            100.0 - step,
            99.8 - step,
            agent,
            world,
        )

    turn_after_run = shaper.calculate_reward(
        Action.TURN_LEFT,
        ActionResult(True, 0.5, "Turned left"),
        96.0,
        95.5,
        agent,
        world,
    )

    assert turn_after_run > baseline_turn + 0.04



def test_backtrack_is_penalized_more_than_forward_progress():
    """A->B->A movement should receive lower reward than progressing to a new tile."""
    world = DummyWorld()
    shaper = RewardShaper()
    agent = _make_agent(x=20, y=20)

    # First movement establishes initial position history.
    agent.x = 21
    shaper.calculate_reward(
        Action.MOVE_FORWARD,
        ActionResult(True, 0.2, "Moved forward"),
        100.0,
        99.8,
        agent,
        world,
    )

    # Move to new tiles (progress) so recent_positions is primed.
    agent.x = 22
    shaper.calculate_reward(
        Action.MOVE_FORWARD,
        ActionResult(True, 0.2, "Moved forward"),
        99.8,
        99.6,
        agent,
        world,
    )

    agent.x = 23
    progress_reward = shaper.calculate_reward(
        Action.MOVE_FORWARD,
        ActionResult(True, 0.2, "Moved forward"),
        99.6,
        99.4,
        agent,
        world,
    )

    # Immediate backtrack to previous tile (loop pattern): 23 -> 22.
    agent.x = 22
    backtrack_reward = shaper.calculate_reward(
        Action.MOVE_FORWARD,
        ActionResult(True, 0.2, "Moved forward"),
        99.4,
        99.2,
        agent,
        world,
    )

    assert backtrack_reward < progress_reward - 0.05
