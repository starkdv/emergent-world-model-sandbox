# World-model quality: measuring (and training) open-loop rollout error

The planning studies (`sample_planning_multiseed/`, `sample_planning_warmup_sweep/`)
kept implicating **compounding open-loop model error** — the planner rolls the
dynamics head several steps feeding its own predictions back, but the head is
trained only on 1-step prediction and nothing measured that regime. This study
uses the model-quality toolkit (CHANGELOG "M1–M3") to measure it directly and to
test whether training it directly (the multi-step consistency loss) helps.

## Arms

| arm | config | world-model training |
|---|---|---|
| **1step** | `config/wm_quality_1step_v35.yaml` | standard 1-step auxiliary loss |
| **multistep** | `config/wm_quality_multistep_v35.yaml` | + k-step open-loop consistency loss (k=3, coef 0.5) |

Both: v3.5 + PPO, 64×64, shooting planner, curiosity off, seeds 1–3, 6,000
ticks, and both log the **k-step rollout-error diagnostic**
(`learning.ppo.rollout_metric_k: 3` → `wm_rollout_error` column in the metrics
CSV): every learner update, the dynamics head is rolled open-loop 3 steps from
real hidden states (own predictions fed back, real logged actions) and its
latent MSE vs the real encoded latents is recorded (EMA, population mean per
generation).

## Results (open-loop latent MSE at k=3; mean ± std over 3 seeds)

| tick | 1-step | multi-step |
|---|---|---|
| 1000 | 6.56 ± 0.09 | **6.09 ± 0.11** |
| 2000 | 6.03 ± 0.11 | 6.08 ± 0.11 |
| 3000 | 5.48 ± 0.31 | 5.59 ± 0.28 |
| 4000 | 4.79 ± 0.40 | 4.60 ± 0.35 |
| 5000 | 3.64 ± 0.37 | 3.91 ± 0.21 |
| 6000 | 2.89 ± 0.43 | 2.93 ± 0.52 |

(Per-run data: the six `*_metrics.csv` files here; figure:
`paper/figures/rollout_error.png`.)

## Findings

1. **The diagnostic works and quantifies the cold-start problem.** Error falls
   monotonically in all 6 runs (−56% over the run). At tick 1,000 it is
   **2.3× its tick-6,000 level** — the measured version of the warmup
   hypothesis. And it is *still falling* at 6,000 ticks: the model never
   "finishes" training at these horizons, consistent with no fixed warmup
   switch point beating the baseline in the sweep.
2. **The multi-step loss helps exactly where the model is worst**: −7% error at
   tick 1,000 (the only point where the arms' seed ranges don't overlap:
   6.00–6.24 vs 6.49–6.69); indistinguishable from tick 2,000 on. It
   accelerates early open-loop competence without changing converged quality
   at this capacity.
3. **No downstream fitness change**: final fitness 63.2 ± 10.7 (1-step) vs
   56.2 ± 6.0 (multi-step), peak 66.1 ± 6.7 vs 57.4 ± 7.6 — overlapping at
   n=3. Early model error is real and reducible, but not the only binding
   constraint on model-based planning here.

Practical use: gate model-heavy planners on the *measured* error instead of a
tick count via `brain.world_model.planner.warmup_error_threshold` (and
`learning.ppo.imagination.warmup_error` for imagination).

## Reproduce

```bash
for seed in 1 2 3; do
  for cfg in wm_quality_1step wm_quality_multistep; do
    python main.py --no-viz --config config/${cfg}_v35.yaml \
      --learning --mode rl --seed $seed --generations 6 \
      --metrics-csv ${cfg}_s${seed}.csv --log --log-dir run
  done
done
# the wm_rollout_error column is the diagnostic
```
