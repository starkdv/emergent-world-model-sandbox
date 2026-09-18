"""
Per-generation metrics CSV (World upgrade W6c).

A long run is hard to reason about from console spam alone. ``MetricsWriter``
appends one row every ``generation_length`` ticks capturing the population and
ecology aggregates — population, food/plant/seed counts, mean energy/age, and
the soil fertility/moisture means — so a run can be plotted and compared
without re-parsing the per-action logs. It is intentionally light: one O(agents)
pass per generation, gated behind an explicit ``--metrics-csv`` flag.

It also carries the **evolution** aggregates, so a long campaign can measure
trait dynamics without per-tick agent logs at all (which at 80 agents x 20k
ticks is millions of rows and makes the run I/O bound rather than
compute bound): per-trait mean/SD, Sarle's bimodality coefficient for
evolutionary branching, the realised foraging income against the population's
energy burn, and the reproduction subsidy currently in force.

Author: Karan Vasa
"""

import csv
import os
from typing import Optional

import numpy as _np


def _bimodality(x: "_np.ndarray") -> float:
    """
    Sarle's bimodality coefficient, BC = (g^2+1)/(k + 3(n-1)^2/((n-2)(n-3))).

    Above 5/9 the distribution is flatter-or-more-split than uniform, which is
    the signature of evolutionary branching. Mirrors
    scripts/analyze_evolution.bimodality_coefficient so the live metric and
    the post-hoc instrument agree.
    """
    n = x.size
    if n < 4:
        return float("nan")
    sd = x.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    z = (x - x.mean()) / sd
    g = float((z**3).mean())
    k = float((z**4).mean() - 3.0)
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if abs(denom) < 1e-12:
        return float("nan")
    return float((g**2 + 1.0) / denom)


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
    # --- evolution (agents/ecology.py, scripts/analyze_evolution.py) ---
    "max_generation",
    "mean_generation",
    "lineages",
    "birth_subsidy",
    "eats_per_agent_per_1k",
    "energy_income_per_tick",
    "energy_burn_per_tick",
    "energy_ratio",
    "body_size_mean",
    "body_size_sd",
    "body_size_bc",
    "visual_acuity_mean",
    "visual_acuity_sd",
    "visual_acuity_bc",
    "diet_mean",
    "diet_sd",
    "diet_bc",
]


class MetricsWriter:
    """Append per-generation aggregate rows to a CSV."""

    def __init__(self, path: str):
        self.path = path
        self._last_tick = 0
        self._last_eat_energy = 0.0
        self._last_eat_count = 0
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

        # --- evolution aggregates -------------------------------------
        from agents import ecology

        evo = {}
        # Tolerate agent stand-ins without a genome or traits (test fixtures
        # use SimpleNamespace); the evolution columns are then simply blank.
        genomes = [g for a in agents if (g := getattr(a, "genome", None)) is not None]
        gens = [int(g.generation) for g in genomes]
        evo["max_generation"] = max(gens) if gens else 0
        evo["mean_generation"] = round(sum(gens) / len(gens), 2) if gens else 0.0
        evo["lineages"] = len({g.lineage_id for g in genomes}) if genomes else 0
        evo["birth_subsidy"] = round(ecology.birth_subsidy_at(world.tick), 4)

        # Realised foraging income vs the population's energy burn. The
        # baseline runs at ~3.5% -- the population is funded by the birth
        # subsidy, not by eating -- so this ratio is the headline number for
        # whether the ecology is actually self-supporting.
        window = max(world.tick - self._last_tick, 1)
        income = (
            getattr(world, "eat_energy_total", 0.0) - self._last_eat_energy
        ) / window
        burn = sum(
            getattr(a, "metabolism_rate", 0.0)
            + ecology.acuity_upkeep(getattr(a, "traits", {}))
            for a in agents
        )
        evo["energy_income_per_tick"] = round(float(income), 4)
        evo["energy_burn_per_tick"] = round(float(burn), 4)
        evo["energy_ratio"] = round(float(income / burn), 4) if burn > 1e-9 else ""
        eats = getattr(world, "eat_count_total", 0) - self._last_eat_count
        evo["eats_per_agent_per_1k"] = (
            round(1000.0 * eats / window / n, 4) if n else 0.0
        )
        self._last_tick = world.tick
        self._last_eat_energy = getattr(world, "eat_energy_total", 0.0)
        self._last_eat_count = getattr(world, "eat_count_total", 0)

        for trait, default in (
            ("body_size", 1.0),
            ("visual_acuity", 2.0),
            ("diet", 0.5),
        ):
            values = [
                float(getattr(a, "traits", {}).get(trait, default)) for a in agents
            ]
            if values:
                arr = _np.asarray(values, dtype=float)
                evo[f"{trait}_mean"] = round(float(arr.mean()), 4)
                evo[f"{trait}_sd"] = round(float(arr.std()), 4)
                evo[f"{trait}_bc"] = round(_bimodality(arr), 4)
            else:
                evo[f"{trait}_mean"] = evo[f"{trait}_sd"] = evo[f"{trait}_bc"] = ""

        row = {
            **evo,
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
