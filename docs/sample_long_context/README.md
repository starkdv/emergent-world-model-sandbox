# sample_long_context: longer training memory and the 8-to-512 cascade (12 runs + 5 partial)

**Headline.** The cascade trains on the last 8 ticks every chunk, the last 16
every 2nd chunk, the last 32 every 4th, and so on up to the last 512. It lowers
the world model's error in the first four generations (3-step error 0.31 vs
0.39 at gen 4; error over persistence 2.2x vs 3.1x). By gen 6 both arms are
equal (0.31 vs 0.30). Fitness does not change at n = 4 (50.5 +/- 8.6 vs
51.7 +/- 6.4). This is the same "helps early, converges equal" pattern as the
multi-step loss (M3). A plain 512-tick window was too slow to finish
(it managed 2 generations in the time L8 ran all 6). After that one generation it looks
like the cascade, but with only 23% of agents trained instead of 40%.

**Side finding.** Only 40% of agents ever run a single PPO update in their
lifetime, in every arm. The cause is the world's learning-slot cap
(`learning.max_updates_per_tick: 4`, train interval 3), which grants slots
first-come. The same 12 of 30 agents win them every time, so the other 60%
live on inherited weights alone. This applies to every published PPO run
that uses this config family.

## Question

Agents train by backprop through time on 8-tick chunks (`learning.ppo.seq_len`).
The GRU's hidden state carries over for the whole life at run time. But
gradients only reach 8 ticks back, so the brain cannot *learn* to keep
information longer than that. Does a longer training window help?

## What was added

- `learning.ppo.cascade: {enabled, max_len}` (off by default). The learner
  keeps the last `max_len` steps along with each step's pre-step hidden state.
  At chunk n it queues the most recent `seq_len * 2^j` steps for every j where
  `2^j` divides n. After the regular update, it trains once on each queued
  window. The short-window update is unchanged. Each level costs about one
  extra BPTT step per tick. With the flag off, the trained weights are
  bit-identical to the old code.
- Learner counters `update_count` and `cascade_updates`. Metrics CSV columns
  `frac_trained` (share of living agents with at least one update) and
  `mean_updates` (mean updates per agent).
- Tests: `tests/test_ppo_cascade.py`.

## Arms

All arms share `docs/sample_wm_residual/configs/legacy_zero.yaml` (v3.5, PPO,
zero-init dynamics head, M3 k=3). Seeds 1-4, 6 generations x 1,000 ticks.
Agents live at most 1,000 ticks.

| arm | seq_len | batch_size (chunks) | chunk_buffer | cascade |
|---|---|---|---|---|
| `L8` | 8 | 8 | 24 | off |
| `cascade512` | 8 | 8 | 24 | 16, 32, ..., 512 |
| `L512` | 512 | 1 | 2 | off |
| `L32` (1 seed, hint only) | 32 | 2 | 6 | off |

Plain 512 needs `batch_size: 1`: at 8 chunks it would need 4,096 ticks of
experience before its first update, which no agent lives to see.

`L8` uses the same config as `legacy_zero` in `sample_wm_residual`, and
matches it within noise (gen 6 error 0.304 vs 0.29). It is not bit-identical:
headless runs are not exactly reproducible across invocations.

## Results (mean +/- SD over seeds; `python summarize.py` prints all)

**3-step open-loop latent MSE (`wm_rollout_error`)**

| gen | L8 | cascade512 | L512 |
|---|---|---|---|
| 1 | 0.492 +/- 0.018 | 0.455 +/- 0.022 | 0.476 +/- 0.026 |
| 2 | 0.455 +/- 0.012 | 0.400 +/- 0.061 | - |
| 3 | 0.416 +/- 0.052 | 0.355 +/- 0.045 | - |
| 4 | 0.394 +/- 0.066 | 0.307 +/- 0.017 | - |
| 5 | 0.343 +/- 0.029 | 0.328 +/- 0.073 | - |
| 6 | 0.304 +/- 0.016 | 0.306 +/- 0.060 | - |

**Model error / do-nothing error (`wm_rel_persistence`, < 1 is good)**

| gen | L8 | cascade512 | L512 |
|---|---|---|---|
| 1 | 3.29 +/- 0.60 | 2.82 +/- 0.24 | 2.84 +/- 0.32 |
| 4 | 3.14 +/- 0.47 | 2.22 +/- 0.35 | - |
| 6 | 2.38 +/- 0.36 | 2.03 +/- 0.52 | - |

**Fitness and training coverage**

| | L8 | cascade512 | L512 (gen 1) |
|---|---|---|---|
| mean fitness, gen 4-6 | 51.7 +/- 6.4 | 50.5 +/- 8.6 | - |
| mean fitness, gen 1 | 29.3 +/- 7.7 | 34.4 +/- 6.5 | 36.8 +/- 6.0 |
| agents ever trained | 40% | 40% | 23% |

## Findings

1. **The cascade speeds up world-model learning early but not the end
   point.** Error is lower in gens 1-4, and the 1-SD bands separate at gen 4.
   By gen 6 the arms are equal. Still no arm beats "nothing changes"
   (best: 2.0x).
2. **No fitness effect at n = 4.** The spreads overlap completely.
3. **A plain 512-tick window is impractical here.** Every update
   backpropagates through 512 steps, and updates run every 3 ticks. Started
   side by side, L512 finished 2 generations while L8 finished 6. A container
   restart cut the 4 seeded runs off after generation 1. Agents also wait 512 ticks before their first update, so
   only 23% train at all. The cascade gets long windows at about 2x the run
   time of L8 (1,380 s vs 730 s per run).
4. **The cap on learning slots, not the window, may be the bigger
   limiter.** Sixty percent of agents never learn during their lifetime.
   Any gain from in-lifetime learning, including the cascade, reaches at most
   40% of the population.

## Next

- Fix the cap on learning slots (round-robin, or a higher cap in headless
  runs), then re-test the cascade with every agent training.
- Re-run plain `L512` to completion when compute allows, as the clean control.

## Reproduce

```bash
for seed in 1 2 3 4; do
  for arm in L8 cascade512 L512; do
    OMP_NUM_THREADS=1 python main.py --no-viz \
      --config docs/sample_long_context/configs/${arm}.yaml \
      --learning --mode rl --seed $seed --generations 6 \
      --metrics-csv docs/sample_long_context/${arm}_s${seed}_metrics.csv
  done
done
python docs/sample_long_context/summarize.py
```
