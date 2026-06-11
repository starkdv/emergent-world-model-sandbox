"""
Curiosity — intrinsic reward from world-model prediction error.

When the brain has a latent dynamics head (``brain.world_model.enabled``),
each transition yields a prediction error

    err_t = mean( (ẑ_{t+1} − z_{t+1})² )

where ẑ_{t+1} = dynamics(h_{t+1}, a_t) and z_{t+1} = encode(obs_{t+1}).
States the model predicts poorly are *surprising*; rewarding surprise
pushes agents toward unexplored dynamics without hand-crafted
exploration bonuses (the emergence-first way to get exploration).

The raw error scale is arbitrary and drifts as the model trains, so it
is normalised by running statistics (Welford) and clipped:

    intrinsic_t = weight · clip( (err_t − μ) / σ , 0, clip )

Only positive surprises are rewarded (err below the running mean means
"more boring than usual" — that should not be punished). The weight can
decay multiplicatively per call so curiosity hands over to extrinsic
reward as the run matures.

Author: Karan Vasa
Date: June 2026
"""

import numpy as np


class CuriosityModule:
    """
    Normalised, decaying intrinsic-reward signal from prediction error.

    Attributes:
        weight: Current scale of the intrinsic reward
        decay: Multiplicative weight decay per reward computation
        clip: Max value of the normalised (z-scored) error
    """

    def __init__(
        self,
        weight: float = 0.1,
        decay: float = 1.0,
        clip: float = 3.0,
        warmup: int = 20,
    ):
        """
        Initialize the curiosity module.

        Args:
            weight: Initial intrinsic-reward scale
            decay: Per-step multiplicative decay of the weight
                   (1.0 = constant curiosity)
            clip: Cap on the normalised error (standard deviations)
            warmup: Number of samples to collect before emitting rewards
                    (lets the running statistics stabilise first)
        """
        self.weight = weight
        self.decay = decay
        self.clip = clip
        self.warmup = warmup

        # Welford running statistics of the raw prediction error
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    @classmethod
    def from_config(cls, config: dict) -> "CuriosityModule":
        """Build from a ``learning.curiosity`` config dict."""
        config = config or {}
        return cls(
            weight=config.get("weight", 0.1),
            decay=config.get("decay", 1.0),
            clip=config.get("clip", 3.0),
            warmup=config.get("warmup", 20),
        )

    def _update_stats(self, err: float) -> None:
        """Welford online update of mean/variance."""
        self._count += 1
        delta = err - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (err - self._mean)

    @property
    def _std(self) -> float:
        if self._count < 2:
            return 1.0
        return float(np.sqrt(self._m2 / (self._count - 1)) + 1e-8)

    def intrinsic_reward(
        self, predicted_latent: np.ndarray, actual_latent: np.ndarray
    ) -> float:
        """
        Intrinsic reward for one transition.

        Args:
            predicted_latent: ẑ_{t+1} from the dynamics head
            actual_latent: z_{t+1} from encoding the real next observation

        Returns:
            Normalised, clipped, weighted curiosity reward (≥ 0)
        """
        err = float(np.mean((predicted_latent - actual_latent) ** 2))
        self._update_stats(err)

        if self._count <= self.warmup:
            return 0.0

        # Only above-average surprise is rewarded
        normalised = max(0.0, (err - self._mean) / self._std)
        reward = self.weight * min(normalised, self.clip)

        self.weight *= self.decay
        return reward
