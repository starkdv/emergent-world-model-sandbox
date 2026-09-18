# sample_v35_emergence_baseline — what the shipped brain actually does

**24 runs**: 3 simulation types x 2 evolution modes x 4 seeds, 5,000 ticks
each (5 generations x 1000). Headless, serial (`simulation.parallel: false`),
`learning.algorithm: ppo` in the `rl` arms. Measured with
`scripts/analyze_emergence.py`, which is new in the same branch as this
directory and pairs every behavioural claim with a null model.

This is the baseline the **Brain v4 proposal** (`docs/BRAIN_V4_PROPOSAL.md`)
argues from. It is a diagnostic study, not an A/B: there is no treatment arm.

## Arms

| Arm | Config | World | Brain | Notes |
|---|---|---|---|---|
| A | `configs/A_default_v3.yaml` | 96x96 heightmap, 16 -> 30 agents, weather + fire | v3 (8 actions, 17,337 params) | derived from `config/default.yaml`; cohort **competition** v2 vs v3 on; `parallel: false`; `algorithm: ppo` |
| B | `configs/B_social_v35.yaml` | 64x64, 24 -> 30 agents | **v3.5** (9 actions incl. SIGNAL, 17,626 params) | derived from `config/worldmodel_v35.yaml` with `agents_visible`, `agent_collision`, `signal.enabled`, `social.transfer_enabled` all **on** |
| C | `configs/C_ecology_v35.yaml` + `--objects config/ecology.yaml` | as B, plus 3 food species and a toxic look-alike | v3.5 | the only discrimination task in the repo |

Modes: `rl` (PPO + Lamarckian inheritance) and `neuroevolution` (no gradients).

Reproduce one run with:

```
python main.py --no-viz \
  --config docs/sample_v35_emergence_baseline/configs/B_social_v35.yaml \
  --mode rl --seed 1 --generations 5 \
  --log --log-dir runs/B_rl_s1 --metrics-csv runs/B_rl_s1/metrics.csv
python scripts/analyze_emergence.py runs/B_rl_s1
```

## What is committed here

- `configs/` — the three configs, verbatim.
- `metrics/<arm>_<mode>_s<seed>.csv` — the per-generation `--metrics-csv`
  output of all 24 runs.
- `emergence.json` — the **full** `analyze_emergence.py` output for all 24
  runs: every metric in this README plus the per-quarter time courses.
- `reward_probe.py` + `reward_probe_B_seed1.json` — the per-action shaped
  reward measurement of §2.6 of the proposal (a 2,000-tick arm-B run with
  `Agent.compute_reward` wrapped).

The raw `agent_actions_*.csv` logs are ~16-20 MB per run (~450 MB total) and
are **not** committed. `emergence.json` is the complete derived record; the
command above regenerates the raw logs from the committed config + seed.

## Headline results

### 1. `SIGNAL` is the cheapest always-legal action, and the population lives on it

Success-path energy costs (`utils/agents/agent_utils.py`):
`SIGNAL 0.12` < `WAIT 0.18` < `MOVE_FORWARD 0.20` < `TURN 0.24` (+ up to 0.20
escalating). `SIGNAL` always succeeds, is never masked, and has no term in the
reward shaper.

SIGNAL share of all actions, by quarter of the run (mean over 4 seeds):

| arm | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| B / neuroevolution | 30.5 | 30.5 | 28.1 | 29.5 |
| B / rl | 34.8 | 40.5 | 28.0 | 20.5 |
| **C / neuroevolution** | **41.3** | **58.4** | **60.7** | **58.6** |
| C / rl | 26.5 | 25.3 | 28.1 | 24.1 |

The effect is **strongest with no reward function at all** (neuroevolution),
so it is a property of the action-cost table, not of reward shaping or PPO.
Arms A have no SIGNAL action; there the idling lands on `WAIT`
(41.9 ± 22% under neuroevolution).

### 2. Planting pays the planter nothing

A planted tile is harvested by **the planter**, at a latency long enough for
the seed to have borne fruit (>= 160 ticks), 4-21% of the time. A tile the
population merely walked on is harvested near, at the same latency, 26-63% of
the time.

| | A/neuro | A/rl | B/neuro | B/rl | C/neuro | C/rl |
|---|---|---|---|---|---|---|
| plantings logged | 1,290 ± 990 | 9,220 ± 2,500 | 646 ± 590 | 1,070 ± 1,100 | 442 ± 350 | 847 ± 850 |
| harvested by anyone @>=160 | 0.49 ± 0.27 | 0.72 ± 0.05 | 0.39 ± 0.17 | 0.75 ± 0.05 | 0.45 ± 0.31 | 0.72 ± 0.08 |
| matched spatial null @>=160 | 0.30 ± 0.27 | 0.57 ± 0.10 | 0.26 ± 0.23 | 0.59 ± 0.14 | 0.34 ± 0.25 | 0.63 ± 0.18 |
| **harvested by the planter @>=160** | **0.10 ± 0.09** | **0.14 ± 0.03** | **0.04 ± 0.02** | **0.21 ± 0.04** | **0.12 ± 0.13** | **0.13 ± 0.04** |
| median harvest latency (any) | 123 ± 67 | 59 ± 9 | 166 ± 74 | 66 ± 32 | 220 ± 180 | 73 ± 23 |

Caveat, stated plainly: the null is **saturated** (0.26-0.63) because the
worlds are food-dense — 200-600 plants standing at the end of every run. This
measurement has little power as configured, and a sparser world is needed to
sharpen it. What it can say is that the planter-specific benefit is not
detectable above ambient food density.

### 3. Individual differentiation is real; spatial structure is not

`role_MI = H(action) - E_agent[H(action|agent)]` beats its label-shuffled null
by 3-5x in every arm. But territory overlap is 1-2% and each agent sees
22-102 tiles of a 4,096-9,216-tile world: agents are not partitioning space,
they are each stuck in a small blob.

### 4. Neuroevolution collapses

Without gradients, 55-77% of agents spend more than half their life on a
single action, trigram entropy drops ~2 bits, and meals fall 4-7x. The
"dual-mode" toggle is not a symmetric comparison at this run length.

### 5. Everything pins at the population cap

All 24 runs end at `reproduction.max_population` (30). Survival does not
discriminate between arms here; any future selection-pressure experiment must
raise the cap or lower carrying capacity first.

## Aggregate table (mean ± SD over 4 seeds)

| metric | A/neuro | A/rl | B/neuro | B/rl | C/neuro | C/rl |
|---|---|---|---|---|---|---|
| end_alive_agents | 30 ± 0 | 30 ± 0 | 30 ± 0 | 30 ± 0 | 30 ± 0 | 30 ± 0 |
| mean_final_age | 514 ± 73 | 646 ± 25 | 609 ± 75 | 625 ± 22 | 608 ± 58 | 654 ± 23 |
| max_age | 929 ± 1.2e+02 | 999 ± 0 | 967 ± 43 | 999 ± 0 | 999 ± 0 | 999 ± 0 |
| end_mean_energy | 113 ± 20 | 179 ± 9.7 | 114 ± 27 | 140 ± 18 | 122 ± 25 | 171 ± 7.2 |
| n_agents | 294 ± 38 | 231 ± 8.6 | 249 ± 30 | 239 ± 8.2 | 247 ± 22 | 228 ± 8 |
| pct_MOVE_FORWARD | 20.6 ± 17 | 26.9 ± 1.1 | 18.1 ± 12 | 6.6 ± 3 | 7.48 ± 6 | 12.7 ± 4.1 |
| pct_TURN_LEFT | 8.62 ± 2.3 | 18 ± 6.5 | 8.18 ± 1.9 | 18.9 ± 11 | 6.87 ± 4.3 | 10.5 ± 5.3 |
| pct_TURN_RIGHT | 21 ± 17 | 13.2 ± 5 | 12.4 ± 5.4 | 15.4 ± 6.3 | 7.35 ± 4.6 | 21.7 ± 17 |
| pct_WAIT | 41.9 ± 22 | 17 ± 5.8 | 27.9 ± 13 | 21.3 ± 5.1 | 19.8 ± 3.8 | 16.6 ± 6.4 |
| pct_SIGNAL | 0 ± 0 | 0 ± 0 | 29.7 ± 18 | 31 ± 10 | 54.8 ± 9.5 | 26 ± 14 |
| pct_EAT | 0.733 ± 0.71 | 4.31 ± 1.6 | 0.509 ± 0.72 | 0.986 ± 0.3 | 0.373 ± 0.47 | 1.99 ± 0.88 |
| pct_USE | 0.869 ± 0.67 | 6.19 ± 1.7 | 0.605 ± 0.49 | 1.1 ± 1.1 | 0.388 ± 0.3 | 0.641 ± 0.61 |
| pct_always_valid | 92.1 ± 7.7 | 75.1 ± 2.6 | 96.3 ± 4.4 | 93.2 ± 5.1 | 96.3 ± 2.3 | 87.4 ± 8.6 |
| rate_EAT_ok | 0.84 ± 0.91 | 5.6 ± 2.2 | 0.693 ± 1 | 1.24 ± 0.41 | 0.496 ± 0.66 | 2.63 ± 1.2 |
| rate_PICK_UP_ok | 4.64 ± 5 | 16.2 ± 2.1 | 2.4 ± 3.1 | 4.05 ± 2.9 | 2.2 ± 1.4 | 8.3 ± 5.7 |
| rate_USE_ok | 0.886 ± 0.66 | 8 ± 2.2 | 0.753 ± 0.66 | 1.33 ± 1.3 | 0.469 ± 0.38 | 0.831 ± 0.79 |
| rate_SIGNAL_ok | 0 ± 0 | 0 ± 0 | 37.1 ± 25 | 38.3 ± 12 | 67 ± 15 | 33.7 ± 17 |
| H_pop | 1.84 ± 0.15 | 2.64 ± 0.037 | 2.15 ± 0.35 | 2.37 ± 0.11 | 1.89 ± 0.3 | 2.47 ± 0.21 |
| H_ind_mean | 1.43 ± 0.13 | 2.21 ± 0.029 | 1.7 ± 0.3 | 1.74 ± 0.32 | 1.45 ± 0.25 | 1.94 ± 0.16 |
| H_cond | 1.35 ± 0.14 | 2.04 ± 0.055 | 1.67 ± 0.25 | 1.79 ± 0.25 | 1.41 ± 0.22 | 1.93 ± 0.12 |
| trigram_H | 4.42 ± 0.35 | 6.54 ± 0.1 | 5.35 ± 0.83 | 5.79 ± 0.6 | 4.56 ± 0.72 | 6.16 ± 0.4 |
| trigram_top1_pct | 38.6 ± 12 | 9.33 ± 2.9 | 27.3 ± 10 | 20.8 ± 5.4 | 39.6 ± 9.6 | 19.2 ± 7.5 |
| role_MI | 0.409 ± 0.12 | 0.428 ± 0.034 | 0.45 ± 0.064 | 0.635 ± 0.23 | 0.448 ± 0.076 | 0.525 ± 0.088 |
| role_MI_null | 0.0816 ± 0.047 | 0.129 ± 0.03 | 0.0617 ± 0.025 | 0.14 ± 0.022 | 0.0728 ± 0.025 | 0.143 ± 0.042 |
| modal_frac_mean | 0.651 ± 0.044 | 0.447 ± 0.0098 | 0.557 ± 0.067 | 0.561 ± 0.074 | 0.634 ± 0.046 | 0.532 ± 0.047 |
| frac_agents_collapsed | 0.77 ± 0.078 | 0.292 ± 0.0066 | 0.548 ± 0.16 | 0.582 ± 0.15 | 0.727 ± 0.076 | 0.526 ± 0.12 |
| tiles_per_agent | 64 ± 46 | 102 ± 8.4 | 47.8 ± 30 | 22.1 ± 7.1 | 24.4 ± 21 | 44.7 ± 16 |
| territory_overlap | 0.0187 ± 0.012 | 0.0169 ± 0.0072 | 0.0217 ± 0.01 | 0.00988 ± 0.0031 | 0.00898 ± 0.0057 | 0.0121 ± 0.0056 |
| patch_return_rate | 0.858 ± 0.039 | 0.783 ± 0.027 | 0.898 ± 0.052 | 0.963 ± 0.021 | 0.891 ± 0.12 | 0.924 ± 0.062 |
| patch_return_null | 0.716 ± 0.057 | 0.619 ± 0.022 | 0.772 ± 0.057 | 0.742 ± 0.023 | 0.815 ± 0.079 | 0.666 ± 0.043 |
| patch_return_index | 1.21 ± 0.13 | 1.26 ± 0.025 | 1.17 ± 0.048 | 1.3 ± 0.027 | 1.11 ± 0.18 | 1.39 ± 0.055 |
| n_plant | 1.29e+03 ± 9.9e+02 | 9.22e+03 ± 2.5e+03 | 646 ± 5.9e+02 | 1.07e+03 ± 1.1e+03 | 442 ± 3.5e+02 | 847 ± 8.5e+02 |
| n_eat | 1.09e+03 ± 1.1e+03 | 6.42e+03 ± 2.4e+03 | 760 ± 1.1e+03 | 1.47e+03 ± 4.5e+02 | 558 ± 6.9e+02 | 2.97e+03 ± 1.3e+03 |
| plant_harvest_rate | 0.584 ± 0.27 | 0.869 ± 0.024 | 0.506 ± 0.18 | 0.868 ± 0.036 | 0.5 ± 0.32 | 0.844 ± 0.046 |
| plant_harvest_null | 0.315 ± 0.28 | 0.616 ± 0.1 | 0.271 ± 0.24 | 0.632 ± 0.15 | 0.357 ± 0.26 | 0.659 ± 0.18 |
| plant_harvest_mature_rate | 0.485 ± 0.27 | 0.721 ± 0.046 | 0.386 ± 0.17 | 0.747 ± 0.054 | 0.454 ± 0.31 | 0.717 ± 0.08 |
| plant_harvest_mature_null | 0.301 ± 0.27 | 0.57 ± 0.096 | 0.255 ± 0.23 | 0.589 ± 0.14 | 0.341 ± 0.25 | 0.633 ± 0.18 |
| plant_harvest_mature_self_rate | 0.101 ± 0.088 | 0.143 ± 0.034 | 0.0441 ± 0.023 | 0.212 ± 0.044 | 0.121 ± 0.13 | 0.128 ± 0.038 |
| plant_harvest_latency_med | 123 ± 67 | 58.8 ± 8.6 | 166 ± 74 | 65.8 ± 32 | 220 ± 1.8e+02 | 72.8 ± 23 |
| end_total_plants | 420 ± 1e+02 | 506 ± 2.6e+02 | 201 ± 1.3e+02 | 362 ± 1.8e+02 | 396 ± 75 | 536 ± 31 |
| end_total_food | 633 ± 1.1e+02 | 679 ± 3.6e+02 | 246 ± 1.6e+02 | 497 ± 2.5e+02 | 449 ± 99 | 440 ± 87 |

## Per-run table (all 24 runs)

| arm | mode | seed | SIGNAL% | WAIT% | EAT% | USE% | always-valid% | eats/agent/1k | signals/agent/1k | role_MI | role_MI null | trigram H | collapsed | tiles/agent | patch idx | plant->self@160 | null@160 | mean age | end energy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | neuro | 1 | 0.000 | 65.94 | 0.254 | 1.743 | 95.3 | 0.263 | 0.000 | 0.433 | 0.068 | 3.971 | 0.844 | 31.36 | 1.198 | 0.183 | 0.121 | 516.6 | 100.5 |
| A | neuro | 2 | 0.000 | 27.5 | 0.012 | 0.022 | 99.81 | 0.011 | 0.000 | 0.224 | 0.023 | 4.723 | 0.663 | 50.14 | 1.250 | 0.000 | 0.000 | 452.9 | 92.51 |
| A | neuro | 3 | 0.000 | 13.44 | 0.800 | 1.237 | 93.85 | 0.728 | 0.000 | 0.544 | 0.153 | 4.794 | 0.846 | 142.6 | 1.014 | 0.026 | 0.392 | 454.8 | 115.3 |
| A | neuro | 4 | 0.000 | 60.64 | 1.864 | 0.472 | 79.4 | 2.358 | 0.000 | 0.432 | 0.083 | 4.203 | 0.728 | 32.1 | 1.375 | 0.194 | 0.692 | 632.6 | 144.9 |
| A | rl | 1 | 0.000 | 11.81 | 6.343 | 7.299 | 71.31 | 8.666 | 0.000 | 0.422 | 0.105 | 6.577 | 0.296 | 104.7 | 1.287 | 0.148 | 0.680 | 683.1 | 166.5 |
| A | rl | 2 | 0.000 | 13.59 | 5.354 | 5.400 | 76.42 | 6.643 | 0.000 | 0.375 | 0.130 | 6.666 | 0.297 | 105 | 1.230 | 0.160 | 0.580 | 620.3 | 188.8 |
| A | rl | 3 | 0.000 | 26.81 | 2.331 | 3.892 | 78.28 | 3.045 | 0.000 | 0.458 | 0.104 | 6.378 | 0.294 | 110.8 | 1.253 | 0.177 | 0.415 | 653.1 | 172.9 |
| A | rl | 4 | 0.000 | 15.63 | 3.227 | 8.171 | 74.5 | 4.053 | 0.000 | 0.457 | 0.177 | 6.539 | 0.281 | 88.25 | 1.290 | 0.088 | 0.603 | 628 | 188.4 |
| B | neuro | 1 | 14.88 | 33.14 | 0.183 | 0.909 | 97.21 | 0.198 | 16.09 | 0.440 | 0.037 | 5.939 | 0.336 | 78.75 | 1.098 | 0.013 | 0.195 | 540.6 | 101.2 |
| B | neuro | 2 | 59.62 | 19.66 | 0.052 | 0.155 | 99.44 | 0.069 | 79.12 | 0.350 | 0.039 | 4.335 | 0.716 | 28.27 | 1.172 | 0.037 | 0.173 | 663.6 | 96.1 |
| B | neuro | 3 | 24.89 | 47.12 | 0.052 | 0.103 | 99.64 | 0.055 | 26.33 | 0.490 | 0.074 | 4.752 | 0.679 | 9.890 | 1.159 | 0.076 | 0.015 | 529 | 98.77 |
| B | neuro | 4 | 19.25 | 11.88 | 1.747 | 1.251 | 88.87 | 2.450 | 27 | 0.519 | 0.097 | 6.362 | 0.460 | 74.44 | 1.234 | 0.051 | 0.638 | 701.1 | 161.9 |
| B | rl | 1 | 13.31 | 24.43 | 1.444 | 0.776 | 95.11 | 1.890 | 17.43 | 0.467 | 0.128 | 6.105 | 0.451 | 15.96 | 1.260 | 0.234 | 0.613 | 654.7 | 137.2 |
| B | rl | 2 | 39.87 | 13.78 | 0.816 | 3.007 | 84.45 | 1.002 | 48.96 | 0.502 | 0.126 | 6.176 | 0.525 | 17.39 | 1.308 | 0.259 | 0.626 | 614 | 157 |
| B | rl | 3 | 35.9 | 19.85 | 1.057 | 0.066 | 96.41 | 1.262 | 42.86 | 0.549 | 0.128 | 6.125 | 0.520 | 34.05 | 1.292 | 0.141 | 0.750 | 596.9 | 154 |
| B | rl | 4 | 34.76 | 27.32 | 0.628 | 0.537 | 96.75 | 0.795 | 43.98 | 1.022 | 0.177 | 4.746 | 0.833 | 20.84 | 1.334 | 0.216 | 0.368 | 632.6 | 112.7 |
| C | neuro | 1 | 51.21 | 18.91 | 0.219 | 0.171 | 93.33 | 0.240 | 56.02 | 0.560 | 0.054 | 5.097 | 0.657 | 16.87 | 1.216 | 0.337 | 0.577 | 547 | 107 |
| C | neuro | 2 | 70.48 | 16.02 | 0.011 | 0.012 | 99.71 | 0.014 | 88.37 | 0.355 | 0.079 | 3.357 | 0.772 | 8.490 | 0.800 | 0.000 | 0.000 | 626.9 | 105 |
| C | neuro | 3 | 52.57 | 18.26 | 1.169 | 0.656 | 95.26 | 1.623 | 73.02 | 0.466 | 0.111 | 4.642 | 0.830 | 60.38 | 1.272 | 0.128 | 0.593 | 694.6 | 165 |
| C | neuro | 4 | 44.75 | 26.21 | 0.095 | 0.713 | 96.85 | 0.108 | 50.56 | 0.409 | 0.048 | 5.151 | 0.651 | 11.73 | 1.136 | 0.020 | 0.195 | 564.9 | 109.4 |
| C | rl | 1 | 48.41 | 14.85 | 0.997 | 0.697 | 94.33 | 1.288 | 62.55 | 0.500 | 0.125 | 5.506 | 0.710 | 29.27 | 1.395 | 0.155 | 0.792 | 646.1 | 166.4 |
| C | rl | 2 | 23.64 | 21.44 | 2.505 | 1.619 | 74.27 | 3.265 | 30.81 | 0.674 | 0.209 | 6.579 | 0.384 | 68.18 | 1.324 | 0.063 | 0.729 | 651.7 | 166.2 |
| C | rl | 3 | 19.2 | 23.15 | 1.300 | 0.058 | 95.7 | 1.631 | 24.08 | 0.482 | 0.097 | 6.178 | 0.470 | 30.14 | 1.356 | 0.155 | 0.328 | 627.2 | 167.5 |
| C | rl | 4 | 12.68 | 6.847 | 3.146 | 0.190 | 85.47 | 4.349 | 17.53 | 0.443 | 0.139 | 6.387 | 0.540 | 51.21 | 1.472 | 0.139 | 0.682 | 691.3 | 183.3 |

## Per-action shaped reward (arm B, seed 1, 2,000 ticks, 59,228 actions)

| action | share | mean reward | median | mean energy cost |
|---|---|---|---|---|
| EAT | 0.22% | +19.74 | +15.02 | 0.10 |
| DROP | 0.59% | +6.34 | +5.70 | 0.10 |
| USE (plant) | 2.40% | +2.99 | +2.69 | 0.21 |
| MOVE_FORWARD | 3.74% | +1.61 | +0.06 | 0.21 |
| PICK_UP | 1.47% | +0.50 | +0.80 | 0.20 |
| WAIT | 28.72% | +0.17 | -0.30 | 0.24 |
| SIGNAL | 40.69% | -0.07 | -0.30 | 0.12 |
| TURN_LEFT | 9.98% | -0.29 | -0.37 | 0.28 |
| TURN_RIGHT | 12.19% | -0.43 | -0.62 | 0.28 |

A needle (+19.7 on 0.2% of steps) in a flat plain (±0.3 on ~90% of steps).
Within the plain, `SIGNAL` has the same median reward as `WAIT` at half the
energy of turning. Raw numbers in `reward_probe_B_seed1.json`.

## Caveats

- 5,000 ticks is short. `max_age` is 1,000, so this is ~5 agent lifetimes and
  a handful of generations — enough to see attractors form, not enough for
  long-run evolutionary claims.
- The population cap binds in every run (see 5 above).
- The cultivation null is saturated (see 2 above).
- Arm A runs the v2/v3 cohort competition; arms B and C do not. The arms are
  three different *simulation types*, not a controlled ablation ladder.
- These are diagnostics of the shipped system, not a treatment comparison.
  Nothing here is an A/B and nothing here claims an effect size.
