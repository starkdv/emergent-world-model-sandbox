"""
Scoring integrity — how an action is paid for, and what fitness means.

Two knobs that decide more of an agent's behaviour than the network does,
collected in one place so they can be read, ablated and A/B'd instead of
being spread across four files (docs/BRAIN_V4_PROPOSAL.md §4.5, phase V4.0).

``action_cost_model``
    * ``legacy`` (default) — the shipped "behaviour economics": a per-turn
      cost that escalates with a turn streak, a per-WAIT cost that escalates
      with an idling streak, and a discount on MOVE_FORWARD right after a
      turn. These are hand-written behaviour policies wearing an energy
      costume.
    * ``flat`` — an action costs a constant of the action, published in
      ``FLAT_ACTION_ENERGY_COST`` below, plus whatever the *world* charges
      (slope climbing, hazard contact). Nothing depends on what the agent
      did last tick.

``fitness_model``
    * ``legacy`` (default) — ``+0.1`` per successful action, ``-0.05`` per
      failed one, minus a death penalty. It scores *acting*, not *living*,
      and the baseline campaign (docs/sample_v35_emergence_baseline) shows
      the population finding the cheapest always-legal action and staying
      there.
    * ``reproduction`` — fitness is the number of offspring that themselves
      reached reproductive age, plus ``fitness_lifespan_coef`` per tick
      survived as a tiebreak. Nothing else.

The flat cost table obeys one rule, which is the whole point of the change:
**no always-legal action may be cheaper than WAIT.** In the shipped table
``SIGNAL`` costs 0.12 against ``WAIT``'s 0.18, so emitting a meaningless
pheromone is literally the cheapest way to pass a tick — and §2.3 of the
proposal measures the population doing exactly that, in both evolution
modes.

Author: Karan Vasa
Date: September 2026
"""

from dataclasses import dataclass
from typing import Optional

from agents.actions import Action

# Actions that are legal in essentially every world state. The invariant
# below is asserted over this set.
ALWAYS_VALID_ACTIONS = (
    Action.MOVE_FORWARD,
    Action.TURN_LEFT,
    Action.TURN_RIGHT,
    Action.WAIT,
    Action.SIGNAL,
)

# Base energy cost of each action under ``action_cost_model: flat``.
# World-derived extras (slope climb, hazard contact damage) are added on top
# by the executors; nothing here depends on action history.
FLAT_ACTION_ENERGY_COST = {
    Action.MOVE_FORWARD: 0.20,
    Action.TURN_LEFT: 0.20,
    Action.TURN_RIGHT: 0.20,
    Action.PICK_UP: 0.20,
    Action.DROP: 0.10,
    Action.EAT: 0.10,
    Action.USE: 0.12,
    Action.WAIT: 0.18,
    Action.SIGNAL: 0.18,
}

# Base success cost each executor in utils/agents/agent_utils.py charges
# before any world surcharge. Used to recover that surcharge (slope climb,
# hazard contact) when the flat table replaces the base.
LEGACY_BASE_COST = {
    Action.MOVE_FORWARD: 0.20,
    Action.TURN_LEFT: 0.24,
    Action.TURN_RIGHT: 0.24,
    Action.PICK_UP: 0.20,
    Action.DROP: 0.10,
    Action.EAT: 0.10,
    Action.USE: 0.12,
    Action.WAIT: 0.18,
    Action.SIGNAL: 0.12,
}

# The invariant: idling must be the cheapest way to idle.
_WAIT_COST = FLAT_ACTION_ENERGY_COST[Action.WAIT]
assert all(
    FLAT_ACTION_ENERGY_COST[a] >= _WAIT_COST for a in ALWAYS_VALID_ACTIONS
), "an always-legal action is cheaper than WAIT — see agents/scoring.py"


@dataclass(frozen=True)
class ScoringConfig:
    """
    Active scoring rules.

    Attributes:
        action_cost_model: ``legacy`` or ``flat``
        fitness_model: ``legacy`` or ``reproduction``
        fitness_lifespan_coef: fitness per tick survived (reproduction model)
        offspring_maturity_ticks: age at which an offspring counts toward its
            parent's fitness (normally ``reproduction.min_age``)
    """

    action_cost_model: str = "legacy"
    fitness_model: str = "legacy"
    fitness_lifespan_coef: float = 0.001
    offspring_maturity_ticks: int = 50

    @property
    def flat_costs(self) -> bool:
        """True when the flat action-cost table is active."""
        return self.action_cost_model == "flat"

    @property
    def reproduction_fitness(self) -> bool:
        """True when fitness counts descendants rather than actions."""
        return self.fitness_model == "reproduction"

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "ScoringConfig":
        """
        Build from the full YAML config dict.

        Reads ``reward.action_cost_model``, ``evolution.fitness_model``,
        ``evolution.fitness_lifespan_coef`` and ``reproduction.min_age``.
        Unknown values fall back to the legacy behaviour.

        Args:
            config: Full config dict (or None for defaults)

        Returns:
            ScoringConfig
        """
        config = config or {}
        reward = config.get("reward", {}) or {}
        evolution = config.get("evolution", {}) or {}
        reproduction = config.get("reproduction", {}) or {}

        cost_model = str(reward.get("action_cost_model", "legacy")).lower()
        if cost_model not in ("legacy", "flat"):
            cost_model = "legacy"
        fitness_model = str(evolution.get("fitness_model", "legacy")).lower()
        if fitness_model not in ("legacy", "reproduction"):
            fitness_model = "legacy"

        return cls(
            action_cost_model=cost_model,
            fitness_model=fitness_model,
            fitness_lifespan_coef=float(evolution.get("fitness_lifespan_coef", 0.001)),
            offspring_maturity_ticks=int(reproduction.get("min_age", 50)),
        )


_ACTIVE_SCORING_CONFIG = ScoringConfig()


def get_active_scoring_config() -> ScoringConfig:
    """Return the scoring rules the simulation is currently using."""
    return _ACTIVE_SCORING_CONFIG


def set_active_scoring_config(config: ScoringConfig) -> None:
    """Set the active scoring rules (call once at startup)."""
    global _ACTIVE_SCORING_CONFIG
    _ACTIVE_SCORING_CONFIG = config
