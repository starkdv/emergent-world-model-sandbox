# sample_population_learning: every agent trains, shared world model, bigger brain (12 runs)

**Headline.** Making every agent train is the biggest world-model improvement
measured in this repo so far. With the learning-slot cap lifted, the share of
agents that ever train rises from 40% to 93%. The 3-step error drops from
0.30 to 0.23, and the error over the do-nothing baseline from 2.4x to 1.4x.
The 1-SD bands do not overlap from generation 2 on. On top of that,
population-shared world-model training adds nothing. A bigger brain makes the
model worse (3.3x). No arm changes fitness at n = 4, and no arm beats "nothing
changes" yet.

## Question

The previous study (`docs/sample_long_context`) found that only 12 of 30
agents ever ran a PPO update, because of the world's first-come learning-slot
cap. Three follow-ups, applied cumulatively:

1. **fair:** every agent trains (`learning.max_updates_per_tick: 4 -> 16`; it
   is the code default, and it exceeds the ~10 agents eligible per tick).
2. **fair_shared:** adds `learning.ppo.world_model_shared` (new, this PR).
   Every finished chunk from every agent goes into one process-wide pool of
   256 chunks. Each update also samples 8 pooled chunks, encodes them with
   the learning agent's own encoder and GRU (detached), and adds the
   world-model loss (horizons 1..3) on them. Policy and value still train
   only on the agent's own experience.
3. **fair_shared_big:** adds a bigger brain, with the GRU going from 48 to
   128 and the dynamics hidden layer from 32 to 128.

The baseline is today's setup: `L8` from `docs/sample_long_context`, the same
config with the cap at 4. All arms use `zero_init` and v3.5, with seeds 1-4
and 6 generations x 1,000 ticks.

## Results (mean +/- SD over 4 seeds; `python summarize.py` prints all)

**3-step open-loop latent MSE**

| gen | baseline | fair | fair_shared | fair_shared_big |
|---|---|---|---|---|
| 1 | 0.492 +/- 0.018 | 0.348 +/- 0.041 | 0.406 +/- 0.031 | 0.618 +/- 0.021 |
| 2 | 0.455 +/- 0.012 | 0.283 +/- 0.046 | 0.316 +/- 0.046 | 0.614 +/- 0.011 |
| 3 | 0.416 +/- 0.052 | 0.252 +/- 0.029 | 0.263 +/- 0.043 | 0.574 +/- 0.026 |
| 4 | 0.394 +/- 0.066 | 0.240 +/- 0.027 | 0.253 +/- 0.036 | 0.548 +/- 0.031 |
| 5 | 0.343 +/- 0.029 | 0.218 +/- 0.034 | 0.225 +/- 0.025 | 0.539 +/- 0.045 |
| 6 | 0.304 +/- 0.016 | 0.229 +/- 0.023 | 0.231 +/- 0.020 | 0.527 +/- 0.081 |

**Model error / do-nothing error (`wm_rel_persistence`, < 1 is good)**

| gen | baseline | fair | fair_shared | fair_shared_big |
|---|---|---|---|---|
| 1 | 3.29 +/- 0.60 | 2.28 +/- 0.24 | 2.70 +/- 0.34 | 4.01 +/- 0.22 |
| 2 | 3.51 +/- 0.89 | 1.76 +/- 0.34 | 1.99 +/- 0.35 | 3.78 +/- 0.17 |
| 4 | 3.14 +/- 0.47 | 1.32 +/- 0.16 | 1.59 +/- 0.31 | 3.33 +/- 0.55 |
| 6 | 2.38 +/- 0.36 | 1.39 +/- 0.23 | 1.39 +/- 0.23 | 3.34 +/- 0.46 |

**Fitness and training coverage**

| | baseline | fair | fair_shared | fair_shared_big |
|---|---|---|---|---|
| mean fitness, gen 4-6 | 51.7 +/- 6.4 | 51.8 +/- 4.2 | 53.5 +/- 7.4 | 52.0 +/- 3.6 |
| agents ever trained | 40% | 93% | 93% | 94% |
| updates per agent (gen 6) | 31 | 136 | 138 | 139 |
| run time (s, 6 gens) | ~730 | ~1,000-1,900 | ~1,250-2,350 | ~1,700-2,600 |

(Run times are not comparable across arms: runs shared 4 cores, with
different numbers of runs in parallel.)

## Findings

1. **In-lifetime training coverage was the main limit on the world model.**
   With 4.4x more updates per agent, error over persistence falls from 2.4x
   to 1.4x, and error over the latent variance from 2.5x to 1.7x. The bands
   separate from gen 1 (latent MSE) and gen 2 (relative error) onwards.
   Every published PPO run with `max_updates_per_tick: 4` trained only
   ~40% of its agents.
2. **Shared world-model training adds nothing on top.** It is slightly worse
   early (gen 1-4) and equal by gen 6. Once every agent trains on its own
   data, the pooled chunks do not add useful signal at this coefficient.
   Untested variants: train the representation on pooled data too (here it
   is detached), and weight own data higher.
3. **A bigger brain is worse, as predicted.** At 128 GRU units and a 128-unit
   dynamics layer, the error is 2.3x higher than fair_shared at gen 6
   (3.3x persistence). It is still undertrained at ~140 updates per agent.
   Capacity is not the limit at this data budget.
4. **No fitness change in any arm at n = 4.** A better world model still does
   not show up in fitness. That matches every earlier world-model result in
   this repo: the policy does not depend on the model enough for its quality
   to matter yet.
5. **Still no model beats "nothing changes"** (best 1.39x). The gap is now
   small enough that a planner gated on `wm_rel_persistence < 1` could
   plausibly open within a longer run.

## Reproduce

```bash
for seed in 1 2 3 4; do
  for arm in fair fair_shared fair_shared_big; do
    OMP_NUM_THREADS=1 python main.py --no-viz \
      --config docs/sample_population_learning/configs/${arm}.yaml \
      --learning --mode rl --seed $seed --generations 6 \
      --metrics-csv docs/sample_population_learning/${arm}_s${seed}_metrics.csv
  done
done
python docs/sample_population_learning/summarize.py
```
