"""Print every table in README.md from the per-run CSVs.

The baseline arm (today's setup, 8-tick window, capped learning slots) is
read from ../sample_long_context/L8_s*_metrics.csv.
"""

import csv
import glob
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {
    "baseline": os.path.join(HERE, "..", "sample_long_context", "L8"),
    "fair": os.path.join(HERE, "fair"),
    "fair_shared": os.path.join(HERE, "fair_shared"),
    "fair_shared_big": os.path.join(HERE, "fair_shared_big"),
}
COLS = [
    "wm_rollout_error",
    "wm_rel_persistence",
    "wm_rel_variance",
    "mean_fitness",
    "mean_age",
    "alive_agents",
    "total_plants",
    "frac_trained",
    "mean_updates",
]


def load(prefix):
    return [
        list(csv.DictReader(open(p)))
        for p in sorted(glob.glob(prefix + "_s[0-9]_metrics.csv"))
    ]


def cell(vals):
    if not vals:
        return "-"
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):.3g} +/- {sd:.2g}"


if __name__ == "__main__":
    runs = {k: load(v) for k, v in ARMS.items()}
    for col in COLS:
        print(f"\n{col}")
        print("| gen | " + " | ".join(ARMS) + " |")
        print("|---|" + "---|" * len(ARMS))
        for g in range(6):
            row = [
                cell(
                    [
                        float(r[g][col])
                        for r in runs[a]
                        if len(r) > g and r[g][col] != ""
                    ]
                )
                for a in ARMS
            ]
            print(f"| {g + 1} | " + " | ".join(row) + " |")
    print("\nper-run gen 4-6 mean fitness / gen 6 wm_rel_persistence")
    for a in ARMS:
        fit = [st.mean(float(r[g]["mean_fitness"]) for g in (3, 4, 5)) for r in runs[a]]
        rel = [float(r[5]["wm_rel_persistence"]) for r in runs[a]]
        print(a, [round(x, 1) for x in fit], cell(fit), "|", [round(x, 2) for x in rel])
