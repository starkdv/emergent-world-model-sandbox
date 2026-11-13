# sample_wm_residual: how good is the world model, really? (16 runs)

**Headline.** Measured against a do-nothing baseline, the ecology's world model
is far worse than it looked. The documented 3-step rollout error of ~2.9 is
**23x worse than predicting "the latent does not change"** and 26x worse than
predicting the batch mean. Most of that error is the random genome
initialisation, not the world: **zeroing the head's output layer at birth cuts
the error ~10x (2.94 -> 0.29)**. The residual head this study set out to test
does **not** help here. The model is still worse than the do-nothing baseline
after every fix (2.1x at best). No arm changes fitness measurably at n = 4.

This is a measurement correction, not a new capability. It changes how
Sections 6.4-6.7 of the paper should be read: the planners there searched
over a model that was worse than no model at all.

## Question

The rollout-error instrument (`wm_rollout_error`, M1) reports raw latent MSE
with no reference point. A review of the brain architecture
(thread notes, Nov 2026) found that the same dynamics head, run on a much
easier robot world, lost to the guess "z' = z". Two questions:

1. Is the ecology head also worse than that do-nothing guess?
2. Does a residual head (predict the change, `z' = z + delta(h, a, z)`) fix it?

## New instrument (this PR)

Every learner update now also computes, on the same windows and masks as M1:

- `wm_persistence_error`: k-step error of the do-nothing guess `z_{t+j} = z_t`.
- `wm_latent_var`: per-dimension variance of the real target latents.

The metrics CSV gains two columns, both averaged over living agents:

| column | meaning | read it as |
|---|---|---|
| `wm_rel_persistence` | model error / persistence error at horizon k | < 1: the model beats "nothing changes" |
| `wm_rel_variance` | model error / latent variance at horizon k | < 1: the model beats predicting the mean (R^2 > 0) |

These are per-agent ratios (EMA, beta 0.9) averaged over agents, so an agent
standing still (tiny persistence error) pulls the mean up. The raw
`wm_rollout_error` column is unchanged and comparable with older runs.

## Arms

All four share `config/wm_quality_multistep_v35.yaml` (v3.5, PPO, 64x64,
shooting planner depth 2 x 6 samples, M3 multi-step loss k=3, curiosity off,
24 founders). Only the `brain.world_model` block differs. Seeds 1-4, 6
generations x 1,000 ticks.

| arm | config | `residual` | `zero_init` |
|---|---|---|---|
| `legacy` | `configs/legacy.yaml` | false | false |
| `residual` | `configs/residual.yaml` | true | false |
| `legacy_zero` | `configs/legacy_zero.yaml` | false | true |
| `residual_zero` | `configs/residual_zero.yaml` | true | true |

`legacy` reproduces `docs/sample_wm_quality` (multistep arm): 2.94 +/- 0.68 at
tick 6,000 here vs 2.93 +/- 0.52 there.

## Results (mean +/- SD over 4 seeds; per-run CSVs in this folder)

**3-step open-loop latent MSE (`wm_rollout_error`)**

| gen | legacy | residual | legacy_zero | residual_zero |
|---|---|---|---|---|
| 1 | 6.2 +/- 0.73 | 27.6 +/- 2.6 | 0.442 +/- 0.031 | 0.529 +/- 0.085 |
| 2 | 5.94 +/- 1 | 24.5 +/- 2 | 0.419 +/- 0.047 | 0.672 +/- 0.1 |
| 3 | 4.99 +/- 0.7 | 19 +/- 2.2 | 0.366 +/- 0.055 | 0.714 +/- 0.14 |
| 4 | 4.37 +/- 0.74 | 16 +/- 3.3 | 0.346 +/- 0.074 | 0.691 +/- 0.13 |
| 5 | 3.48 +/- 0.6 | 13.9 +/- 3.9 | 0.304 +/- 0.1 | 0.657 +/- 0.12 |
| 6 | 2.94 +/- 0.68 | 12 +/- 2.8 | 0.29 +/- 0.074 | 0.668 +/- 0.066 |

**Model error / do-nothing error (`wm_rel_persistence`, < 1 is good)**

| gen | legacy | residual | legacy_zero | residual_zero |
|---|---|---|---|---|
| 1 | 43 +/- 8.7 | 195 +/- 38 | 3.37 +/- 0.57 | 3.44 +/- 0.45 |
| 2 | 41.9 +/- 7.2 | 169 +/- 45 | 2.94 +/- 0.49 | 4.11 +/- 0.53 |
| 3 | 33.9 +/- 3.5 | 148 +/- 38 | 2.68 +/- 0.84 | 4.04 +/- 0.56 |
| 4 | 30.7 +/- 3.6 | 120 +/- 35 | 2.45 +/- 0.91 | 3.94 +/- 0.52 |
| 5 | 26.2 +/- 5.2 | 101 +/- 44 | 2.47 +/- 1.3 | 3.83 +/- 0.26 |
| 6 | 23.1 +/- 5.6 | 92.3 +/- 34 | 2.07 +/- 0.7 | 3.86 +/- 0.37 |

**Model error / latent variance (`wm_rel_variance`, < 1 is good)**

| gen | legacy | residual | legacy_zero | residual_zero |
|---|---|---|---|---|
| 1 | 49.3 +/- 7.6 | 207 +/- 29 | 3.61 +/- 0.33 | 4.24 +/- 0.74 |
| 6 | 25.8 +/- 7.6 | 104 +/- 35 | 2.29 +/- 0.57 | 4.8 +/- 0.43 |

**Mean fitness (gen 4-6 average per run)**

| legacy | residual | legacy_zero | residual_zero |
|---|---|---|---|
| 51.9 +/- 14 | 60.1 +/- 4.4 | 55.4 +/- 14 | 50.7 +/- 4.3 |

(`python docs/sample_wm_residual/summarize.py` prints every table, all
generations.)

## Findings

1. **The legacy world model never beats doing nothing.** At its best (tick
   6,000) its 3-step error is 23x the persistence error and 26x the latent
   variance, on all 4 seeds. The "weak model" of paper conclusion 3 is weaker
   than any trained regressor should be: it has not even learned the mean.
2. **The cause is initialisation, not the world.** Genomes start at
   N(0, 0.5^2), so the head's first outputs are ~N(0, 2^2) per dimension
   against a per-dimension latent variance of ~0.1. Each agent's PPO learner (64-step batches,
   Adam lr 3e-4, global grad-clip 0.5 shared with the policy) shrinks that
   only slowly, and every birth re-mixes it by crossover. The paper's
   "error falls 56% over the run" was mostly this decay. Zeroing `dyn.Wz` and
   `dyn.bz` at birth removes it: 14x lower error at tick 1,000, 10x at 6,000,
   non-overlapping on every seed at every generation.
3. **The residual head does not help in this ecology.** With random init it
   is 4x worse than legacy (each step's random delta accumulates into z).
   With zero init it starts exactly at persistence, then drifts *away* from
   it during training (3.4x -> 3.9x) and ends 2.3x worse than `legacy_zero`.
   On a smooth synthetic sequence the same code beats persistence (3-step
   0.0052 vs 0.0066), so this is a property of the ecology's training regime,
   not a bug. Likely contributors, untested: uniform crossover mixes two
   parents' delta heads element-wise at every birth; 64-step per-agent
   batches; and large discontinuous latent jumps when the 5x5 view shifts,
   where an additive error compounds.
4. **Even the best arm is 2.1x worse than "nothing changes"** and 2.3x worse
   than predicting the mean. No planner in this repo has yet had a world model
   that beats the trivial baselines.
5. **No downstream fitness change at n = 4.** All four arms overlap
   (residual's tighter spread is not a win; its model is the worst).

## What this means

- The planner verdicts (P1-P3, CEM, imagination, warmup) stand as measured
  results but should be read as "search over a model worse than no model".
  "Does planning pay with a *good* model?" remains untested in this repo.
- New experiments that use the world model should set
  `brain.world_model.zero_init: true`. It defaults to false only so that the
  published runs reproduce bit-for-bit.
- `wm_rel_persistence < 1` is the minimum bar a model should clear before a
  planner or imagination is gated on it. The M2 readiness threshold (4.5 raw)
  is easily passed by a model that is worse than no model.
- Next levers, in order: share world-model training across the population
  (each agent's 64-step learner is the bottleneck), exclude the dynamics head
  from crossover (inherit it from one parent), then revisit the residual head.

## Reproduce

```bash
for seed in 1 2 3 4; do
  for arm in legacy residual legacy_zero residual_zero; do
    OMP_NUM_THREADS=1 python main.py --no-viz \
      --config docs/sample_wm_residual/configs/${arm}.yaml \
      --learning --mode rl --seed $seed --generations 6 \
      --metrics-csv docs/sample_wm_residual/${arm}_s${seed}_metrics.csv
  done
done
python docs/sample_wm_residual/summarize.py
```

Each run takes ~10 minutes single-threaded. Set `OMP_NUM_THREADS=1` when
running seeds in parallel; torch's default threading slows parallel runs
several-fold.
