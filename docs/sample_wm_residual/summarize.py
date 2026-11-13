"""Summarise the 4 arms x 4 seeds in this folder (mean +/- SD over seeds)."""

import csv
import glob
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["legacy", "residual", "legacy_zero", "residual_zero"]
COLS = ["wm_rollout_error", "wm_rel_persistence", "wm_rel_variance", "mean_fitness"]


def load(arm):
    runs = []
    for path in sorted(glob.glob(os.path.join(HERE, f"{arm}_s*_metrics.csv"))):
        with open(path, newline="") as fh:
            runs.append(list(csv.DictReader(fh)))
    return runs


def fmt(vals):
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.3g} +/- {sd:.2g}"


for col in COLS:
    print(f"\n{col}")
    print("gen | " + " | ".join(ARMS))
    for g in range(6):
        cells = []
        for arm in ARMS:
            vals = [float(r[g][col]) for r in load(arm)]
            cells.append(fmt(vals))
        print(f"{g + 1} | " + " | ".join(cells))
    late = []
    for arm in ARMS:
        per_run = [st.mean(float(r[g][col]) for g in (3, 4, 5)) for r in load(arm)]
        late.append(fmt(per_run))
    print("gen 4-6 | " + " | ".join(late))
