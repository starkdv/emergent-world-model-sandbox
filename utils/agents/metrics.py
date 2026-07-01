"""
Per-generation metrics CSV (World upgrade W6c).

A long run is hard to reason about from console spam alone. ``MetricsWriter``
appends one row every ``generation_length`` ticks capturing the population and
ecology aggregates — population, food/plant/seed counts, mean energy/age, and
the soil fertility/moisture means — so a run can be plotted and compared
without re-parsing the per-action logs. It is intentionally light: one O(agents)
pass per generation, gated behind an explicit ``--metrics-csv`` flag.

Author: Karan Vasa
"""

import csv
import os
from typing import Optional

FIELDS = [
    "generation",
    "tick",
    "alive_agents",
    "total_food",
    "total_plants",
    "total_seeds",
    "mean_energy",
    "mean_age",
    "max_age",
    "mean_fitness",
    "avg_fertility",
    "avg_moisture",
    "wm_rollout_error",
]


class MetricsWriter:
    """Append per-generation aggregate rows to a CSV."""

    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDS)
        self._writer.writeheader()
        self._fh.flush()

    def record(self, world, generation: int) -> dict:
        """Compute and append one row for the current world state."""
        agents = [a for a in world.agents.values() if getattr(a, "alive", True)]
        n = len(agents)
        counts = world.get_cached_object_counts()
        if n:
            mean_energy = sum(a.energy for a in agents) / n
            mean_age = sum(a.age for a in agents) / n
            max_age = max(a.age for a in agents)
            mean_fitness = sum(getattr(a, "fitness", 0.0) for a in agents) / n
        else:
            mean_energy = mean_age = max_age = mean_fitness = 0.0
        try:
            avg_fert, avg_moist = world.get_cached_soil_stats()
        except Exception:
            avg_fert, avg_moist = 0.0, 0.0

        # World-model quality: mean k-step open-loop rollout error EMA over
        # agents whose learner measures it (PPO + world-model head).
        wm_errs = [
            e
            for a in agents
            if (e := getattr(getattr(a, "learner", None), "wm_rollout_error_ema", None))
            is not None
        ]
        wm_err = round(sum(wm_errs) / len(wm_errs), 5) if wm_errs else ""

        row = {
            "generation": generation,
            "tick": world.tick,
            "alive_agents": counts["alive_agents"],
            "total_food": counts["total_food"],
            "total_plants": counts["total_plants"],
            "total_seeds": counts["total_seeds"],
            "mean_energy": round(float(mean_energy), 3),
            "mean_age": round(float(mean_age), 2),
            "max_age": int(max_age),
            "mean_fitness": round(float(mean_fitness), 3),
            "avg_fertility": round(float(avg_fert), 4),
            "avg_moisture": round(float(avg_moist), 4),
            "wm_rollout_error": wm_err,
        }
        self._writer.writerow(row)
        self._fh.flush()
        return row

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
