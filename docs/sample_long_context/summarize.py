"""Print every table in README.md from the per-run CSVs in this folder."""

import csv
import glob
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["L8", "cascade512", "L512", "L32"]
COLS = [
    "wm_rollout_error",
    "wm_rel_persistence",
    "wm_rel_variance",
    "mean_fitness",
    "mean_age",
    "frac_trained",
    "mean_updates",
]


def load(arm):
    paths = sorted(glob.glob(os.path.join(HERE, f"{arm}_s*_metrics.csv")))
    return [list(csv.DictReader(open(p))) for p in paths]


def cell(vals):
    if not vals:
        return "-"
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):.3g} +/- {sd:.2g} (n={len(vals)})"


for col in COLS:
    print(f"\n{col}")
    print("| gen | " + " | ".join(ARMS) + " |")
    print("|---|" + "---|" * len(ARMS))
    for g in range(6):
        row = []
        for arm in ARMS:
            vals = [
                float(r[g][col]) for r in load(arm) if len(r) > g and r[g][col] != ""
            ]
            row.append(cell(vals))
        print(f"| {g + 1} | " + " | ".join(row) + " |")

print("\nmean fitness, gen 4-6 average per complete run")
for arm in ARMS:
    fits = [
        st.mean(float(r[g]["mean_fitness"]) for g in (3, 4, 5))
        for r in load(arm)
        if len(r) == 6
    ]
    print(arm, [round(f, 1) for f in fits], cell(fits))
