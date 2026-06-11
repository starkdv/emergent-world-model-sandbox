"""
Latent rollout planner — model-based action selection.

With a latent dynamics head, the brain can *imagine*: from the current
hidden state, simulate candidate action sequences entirely in latent
space (no world access, no rendering back to observations):

    repeat depth times:
        ẑ', r̂ = dynamics(h, a)        imagine the consequence
        h     = GRU(ẑ', h)            advance imagined memory
    score = Σ_k γ^k·r̂_k + γ^D·V(ẑ_D, h_D)   (value bootstrap at horizon)

Random shooting: sample N candidate sequences (the first action drawn
from the *valid* action set), score each rollout, and pick the first
action of the best sequence. This is deliberately the simplest
model-based control loop — it exercises the world model end-to-end and
is the seed for Dreamer-style imagination and dream-based evolution.

Off by default (``brain.world_model.planner.enabled``): rollouts cost
``samples × depth`` extra forward passes per decision, and a planner is
only as good as its learned model.

Author: Karan Vasa
Date: June 2026
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from agents.brain import Brain


class LatentPlanner:
    """
    Random-shooting planner over imagined latent rollouts.

    Attributes:
        depth: Imagination horizon (steps)
        samples: Number of candidate action sequences
        gamma: Discount factor inside the rollout
    """

    def __init__(self, depth: int = 3, samples: int = 16, gamma: float = 0.95):
        """
        Initialize the planner.

        Args:
            depth: Imagination horizon
            samples: Candidate action sequences per decision
            gamma: Discount applied to imagined rewards
        """
        self.depth = depth
        self.samples = samples
        self.gamma = gamma

    @classmethod
    def from_config(cls, config: dict) -> "LatentPlanner":
        """Build from a ``brain.world_model.planner`` config dict."""
        config = config or {}
        return cls(
            depth=config.get("depth", 3),
            samples=config.get("samples", 16),
            gamma=config.get("gamma", 0.95),
        )

    def plan(
        self,
        brain: "Brain",
        h: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        Choose the first action of the best imagined rollout.

        Args:
            brain: Brain with a dynamics head (``has_world_model``)
            h: Current GRU hidden state (post-observation)
            action_mask: Valid-action mask for the FIRST step (later
                imagined steps cannot be masked — validity depends on
                world state the model does not expose)

        Returns:
            Index of the chosen first action
        """
        n_actions = brain.output_size
        valid_first = (
            np.flatnonzero(action_mask)
            if action_mask is not None
            else np.arange(n_actions)
        )
        if len(valid_first) == 0:
            valid_first = np.arange(n_actions)

        best_action = int(valid_first[0])
        best_score = -np.inf

        for _ in range(self.samples):
            first = int(np.random.choice(valid_first))
            score = self._rollout(brain, h, first)
            if score > best_score:
                best_score = score
                best_action = first

        return best_action

    def _rollout(self, brain: "Brain", h: np.ndarray, first_action: int) -> float:
        """Imagine one action sequence; return its discounted score."""
        h_sim = h
        z_sim = None
        score = 0.0
        discount = 1.0

        action = first_action
        for _ in range(self.depth):
            z_sim, r_pred = brain.predict_next_latent(h_sim, action)
            score += discount * r_pred
            discount *= self.gamma
            h_sim = brain._gru_step(z_sim, h_sim)
            action = int(np.random.randint(brain.output_size))

        # Bootstrap with the critic's view of the imagined end state
        score += discount * brain._value(z_sim, h_sim)
        return score
