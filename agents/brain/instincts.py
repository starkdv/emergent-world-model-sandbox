"""
Contextual instinct biases — bootstrap scaffolding, outside the brain.

These additive logit biases only fire when the action is contextually
appropriate (the action mask says it's valid). They bootstrap survival
behaviour so agents with random/early-evolution weights actually
interact with the environment.

Extracted from Brain.forward so that:
- the brain stays a pure function of (observation, hidden state, params),
- observation fields are addressed by name via ObservationSpec instead
  of magic indices,
- biases can be scaled by a single ``strength`` factor, enabling
  genuine fading with agent age (Phase 2) instead of lifelong
  constant biases.

Author: Karan Vasa
Date: June 2026
"""

from typing import Optional

import numpy as np

from agents.actions import Action
from agents.brain.spec import DEFAULT_OBSERVATION_SPEC, ObservationSpec


class InstinctModule:
    """
    Applies contextual instinct biases to action logits.

    All biases are multiplied by a ``strength`` factor in [0, 1].
    With ``fade_age`` unset, strength is constant 1.0 (legacy v2
    behaviour). With ``fade_age`` set, ``strength_at(age)`` decays
    linearly to 0 so adult behaviour is produced purely by the
    learned network.

    Attributes:
        enabled: Master switch
        fade_age: Age (ticks) at which instincts reach zero (None = never fade)
        obs_spec: Observation layout used to read stimulus/vision fields
    """

    # Interaction instincts
    PICK_UP_BIAS = 1.5  # strong instinct to pick things up
    EAT_BIAS = 1.0  # instinct to eat when food in inventory
    USE_BIAS = 0.5  # instinct to plant seeds

    # Hunger-scaled EAT instinct. Replaces the old hardcoded auto-eat
    # override in Agent.update (which *forced* EAT below 50% energy).
    # This is a strong prior, not a forced action: the policy can still
    # override it, and it fades with age like every other instinct.
    HUNGER_EAT_BIAS = 3.0  # scaled by the energy_urgency stimulus

    # Direction-aware turn-toward-food instinct
    TURN_TOWARD_FOOD_BIAS = 0.8  # scaled by food proximity
    OFF_SIDE_TURN_FACTOR = 0.2  # small bias for the non-food side
    FOOD_TYPE_THRESHOLD = 0.5  # vision type encodings >= this are food-like
    SIDE_MARGIN = 0.1  # left/right score margin before picking a side

    # Gating thresholds: food nearby, not facing it, not already ahead
    PROXIMITY_GATE = 0.2
    DIRECTION_GATE = 0.6
    AHEAD_GATE = 0.5

    # Default age (ticks) at which instincts fade to zero.
    # Chosen to outlast the reproduction min_age (100 ticks) so juveniles
    # are still scaffolded, while adults act purely on learned weights.
    DEFAULT_FADE_AGE = 150

    def __init__(
        self,
        enabled: bool = True,
        fade_age: Optional[int] = None,
        hunger_eat_bias: float = HUNGER_EAT_BIAS,
        obs_spec: ObservationSpec = DEFAULT_OBSERVATION_SPEC,
    ):
        """
        Initialize the instinct module.

        Args:
            enabled: Master switch for all instinct biases
            fade_age: Age in ticks at which strength reaches 0
                      (None keeps strength constant at 1.0)
            hunger_eat_bias: Extra EAT logit bias at maximum hunger
            obs_spec: Observation layout specification
        """
        self.enabled = enabled
        self.fade_age = fade_age
        self.hunger_eat_bias = hunger_eat_bias
        self.obs_spec = obs_spec

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "InstinctModule":
        """
        Build an InstinctModule from a ``brain.instincts`` config dict.

        Recognised keys (all optional):
            enabled (bool):        master switch          [default: true]
            fade_age (int|null):   ticks until strength 0 [default: 150,
                                   null = never fade (legacy v2 behaviour)]
            hunger_eat_bias (float): EAT prior at max hunger [default: 3.0]

        Args:
            config: Config dict, or None for defaults

        Returns:
            Configured InstinctModule
        """
        config = config or {}
        return cls(
            enabled=config.get("enabled", True),
            fade_age=config.get("fade_age", cls.DEFAULT_FADE_AGE),
            hunger_eat_bias=config.get("hunger_eat_bias", cls.HUNGER_EAT_BIAS),
        )

    def strength_at(self, age: int) -> float:
        """
        Instinct strength for an agent of the given age.

        Returns 1.0 for life when fading is disabled, otherwise decays
        linearly from 1.0 at age 0 to 0.0 at ``fade_age``.

        Args:
            age: Agent age in ticks

        Returns:
            Strength factor in [0, 1]
        """
        if not self.enabled:
            return 0.0
        if self.fade_age is None:
            return 1.0
        return max(0.0, 1.0 - age / self.fade_age)

    def apply(
        self,
        logits: np.ndarray,
        observation: np.ndarray,
        action_mask: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        """
        Add instinct biases to action logits (in place).

        Args:
            logits: Action logits (modified in place)
            observation: Observation vector (read via obs_spec)
            action_mask: Binary mask of valid actions (1 = valid)
            strength: Scale factor for all biases

        Returns:
            The (modified) logits array
        """
        if not self.enabled or strength <= 0.0 or action_mask is None:
            return logits

        # Interaction instincts — only when contextually valid
        if action_mask[Action.PICK_UP.value] > 0:
            logits[Action.PICK_UP.value] += self.PICK_UP_BIAS * strength
        if action_mask[Action.EAT.value] > 0:
            logits[Action.EAT.value] += self.EAT_BIAS * strength

            # Hunger-scaled EAT prior (auto-eat replacement): the lower
            # the agent's energy, the stronger the urge — but never a
            # forced action, and it fades with age like everything else.
            spec = self.obs_spec
            if len(observation) > spec.energy_urgency:
                urgency = float(observation[spec.energy_urgency])
                if urgency > 0.0:
                    logits[Action.EAT.value] += (
                        self.hunger_eat_bias * urgency * strength
                    )
        if action_mask[Action.USE.value] > 0:
            logits[Action.USE.value] += self.USE_BIAS * strength

        self._apply_turn_toward_food(logits, observation, action_mask, strength)

        return logits

    def _apply_turn_toward_food(
        self,
        logits: np.ndarray,
        observation: np.ndarray,
        action_mask: np.ndarray,
        strength: float,
    ) -> None:
        """
        Bias the correct turn when food is nearby but not ahead.

        Uses the egocentric vision grid (already rotated so that
        col < center is the agent's left) to decide which turn
        actually points toward the food.
        """
        spec = self.obs_spec
        if len(observation) <= spec.food_dir_match:
            return

        food_prox = observation[spec.nearest_food_prox]
        food_dir = observation[spec.food_dir_match]
        food_ahead_sig = observation[spec.food_ahead]

        # Food is nearby but agent isn't facing it — encourage turning
        if not (
            food_prox > self.PROXIMITY_GATE
            and food_dir < self.DIRECTION_GATE
            and food_ahead_sig < self.AHEAD_GATE
        ):
            return

        base_bias = self.TURN_TOWARD_FOOD_BIAS * food_prox * strength

        # Sum food-like type encodings on each side of the vision grid
        type_enc = spec.vision_grid(observation)[:, :, 0]
        food_like = np.where(type_enc >= self.FOOD_TYPE_THRESHOLD, type_enc, 0.0)
        center = spec.vision_shape[1] // 2
        left_score = float(food_like[:, :center].sum())
        right_score = float(food_like[:, center + 1 :].sum())

        left_valid = action_mask[Action.TURN_LEFT.value] > 0
        right_valid = action_mask[Action.TURN_RIGHT.value] > 0

        if left_score > right_score + self.SIDE_MARGIN:
            if left_valid:
                logits[Action.TURN_LEFT.value] += base_bias
            if right_valid:
                logits[Action.TURN_RIGHT.value] += base_bias * self.OFF_SIDE_TURN_FACTOR
        elif right_score > left_score + self.SIDE_MARGIN:
            if right_valid:
                logits[Action.TURN_RIGHT.value] += base_bias
            if left_valid:
                logits[Action.TURN_LEFT.value] += base_bias * self.OFF_SIDE_TURN_FACTOR
        else:
            # Food roughly centred / behind — boost both turns equally
            if left_valid:
                logits[Action.TURN_LEFT.value] += base_bias
            if right_valid:
                logits[Action.TURN_RIGHT.value] += base_bias
