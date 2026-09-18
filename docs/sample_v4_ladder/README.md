# sample_v4_ladder — Brain v4 measured against v3.5. It loses.

**12 runs**: 3 arms x 4 seeds, 5,000 ticks each (5 generations x 1000).
Headless, serial, `learning.algorithm: ppo`, measured with
`scripts/analyze_emergence.py`. The world is **identical** to arm B of
`docs/sample_v35_emergence_baseline/`, so the ladder below moves one group of
knobs at a time and the v3.5 column is a like-for-like comparison.

**Headline: Brain v4 does not beat Brain v3.5 on any pre-registered metric at
this compute budget, and the episodic place memory specifically makes things
worse.** Predictions P2, P3 and P4 of `docs/BRAIN_V4_PROPOSAL.md` all fail.
The machinery is built, tested and ablatable; the claim that it helps is not
supported.

## Arms

| Arm | Config | What it adds |
|---|---|---|
| `v3.5 B/rl` | `docs/sample_v35_emergence_baseline/configs/B_social_v35.yaml` | the reference — legacy shaping, instincts on, legacy action costs and fitness |
| `v40_scoring` | `configs/v40_scoring.yaml` (= `config/v4_baseline.yaml`) | V4.0 only: `reward.preset: minimal`, flat action costs with `SIGNAL` at `WAIT`'s price, `fitness_model: reproduction`, instincts off. Same v3.5 brain. |
| `v4_full` | `configs/v4_full.yaml` | V4.0 + the whole v4 architecture (slow core, 8 episodic slots, dual-discount critic, `drives` reward, 4-channel costly signalling) |
| `v4_nomem` | `configs/v4_nomem.yaml` | `v4_full` with `brain.v4.memory_slots: 0` — the episodic place-memory ablation. Same genome length. |

Reproduce one run:

```
python main.py --no-viz --config docs/sample_v4_ladder/configs/v4_full.yaml \
  --mode rl --seed 1 --generations 5 \
  --log --log-dir runs/v4_full_s1 --metrics-csv runs/v4_full_s1/metrics.csv
python scripts/analyze_emergence.py runs/v4_full_s1
```

## Results (mean ± SD over 4 seeds)

| metric | v3.5 B/rl | V4.0 scoring | v4 full | v4 no-memory |
|---|---|---|---|---|
| SIGNAL share % | 31 ± 10 | 28.6 ± 14 | 25.2 ± 10 | 12.1 ± 3.2 |
| always-valid share % | 93.2 ± 5.1 | 95.3 ± 4.3 | 96.2 ± 3.3 | 92.5 ± 6 |
| eats/agent/1k | 1.24 ± 0.41 | 0.954 ± 0.8 | 0.692 ± 0.69 | 1.24 ± 0.73 |
| plants/agent/1k | 1.33 ± 1.3 | 0.291 ± 0.21 | 1.63 ± 1.6 | 3.15 ± 3.6 |
| signals/agent/1k | 38.3 ± 12 | 33.8 ± 15 | 28 ± 12 | 14.5 ± 3.4 |
| trigram entropy (bits) | 5.79 ± 0.6 | 5.64 ± 0.91 | 4.82 ± 1.2 | 5.14 ± 0.5 |
| H(a_t \| a_t-1) (bits) | 1.79 ± 0.25 | 1.73 ± 0.33 | 1.44 ± 0.48 | 1.58 ± 0.14 |
| role_MI (bits) | 0.635 ± 0.23 | 0.688 ± 0.21 | 0.701 ± 0.26 | 0.64 ± 0.15 |
| role_MI null | 0.14 ± 0.022 | 0.0709 ± 0.0016 | 0.0742 ± 0.01 | 0.154 ± 0.062 |
| agents collapsed | 0.582 ± 0.15 | 0.56 ± 0.18 | 0.644 ± 0.19 | 0.687 ± 0.081 |
| tiles visited/agent | 22.1 ± 7.1 | 43.9 ± 23 | 45 ± 36 | 34 ± 29 |
| territory overlap | 0.00988 ± 0.0031 | 0.0168 ± 0.0062 | 0.012 ± 0.0071 | 0.0167 ± 0.013 |
| E1 patch-return index | 1.3 ± 0.027 | 1.33 ± 0.07 | 1.3 ± 0.11 | 1.26 ± 0.094 |
| E2 plant->self @>=160 | 0.212 ± 0.044 | 0.199 ± 0.081 | 0.118 ± 0.1 | 0.097 ± 0.038 |
| E2 matched null | 0.589 ± 0.14 | 0.411 ± 0.11 | 0.347 ± 0.18 | 0.478 ± 0.13 |
| mean lifespan | 625 ± 22 | 609 ± 42 | 557 ± 69 | 607 ± 35 |
| end mean energy | 140 ± 18 | 138 ± 28 | 120 ± 26 | 125 ± 7.9 |
| end plants | 362 ± 1.8e+02 | 202 ± 1.5e+02 | 249 ± 78 | 189 ± 1.1e+02 |
| agents ever alive | 239 ± 8.2 | 246 ± 16 | 272 ± 37 | 246 ± 14 |
| wm rollout error | — | — | 4.02 ± 2.2 | 1.18 ± 1.1 |

## What this says

### 1. V4.0 (scoring integrity) is a null result, and it refutes L2

`docs/BRAIN_V4_PROPOSAL.md` §3 L2 diagnosed the SIGNAL attractor as caused by
the action-cost inversion (`SIGNAL` 0.12 < `WAIT` 0.18) and prediction P4 said
cost parity would collapse the always-valid-action share. It does not:
**31.0 ± 10% → 28.6 ± 14%** SIGNAL, with the always-valid share *rising*
93.2 → 95.3%. Removing the dense shaping and the instincts costs foraging
(1.24 → 0.95 eats/agent/1k) and planting (1.33 → 0.29) and leaves survival
flat (625 → 609 ticks).

So making a meaningless signal cost the same as waiting does not stop agents
emitting it. The likelier driver, on this evidence, is that under the
`minimal` diet with instincts off there is **almost no gradient signal at
all** — `EAT` is 0.7% of steps — so each seed's policy drifts into whatever
arbitrary attractor it finds first. The per-seed spread supports that: SIGNAL
share across the four seeds is 19.0 / 33.3 / 49.4 / 12.6%.

P4's other half — "survival gets worse, and that is the point" — does hold.

### 2. The full v4 architecture is a regression

Against v3.5, `v4_full` is worse on foraging (0.69 vs 1.24 eats/agent/1k),
behavioural diversity (trigram entropy 4.82 vs 5.79), collapse (0.64 vs 0.58)
and lifespan (557 vs 625), and **flat on both of the metrics it was built
for**:

- **E1 / P3 (return to a known patch)**: 1.30 ± 0.11 against v3.5's
  1.30 ± 0.03. The episodic place memory changes nothing.
- **E2 / P2 (cultivation with intent)**: plant→self-harvest at maturity
  latency 0.118 ± 0.10 against a matched null of 0.347 ± 0.18 — still *below*
  the null, exactly as in v3.5.

### 3. The episodic place memory is the component that hurts

`v4_nomem` is `v4_full` with `memory_slots: 0` and the same genome length, and
it is **better on almost everything**: eats 1.24 vs 0.69, lifespan 607 vs 557,
trigram entropy 5.14 vs 4.82, SIGNAL share 12.1 vs 25.2%.

The mechanism shows up in the world model:

| | v4 full | v4 no-memory |
|---|---|---|
| `wm_rollout_error` | **4.02 ± 2.2** | **1.18 ± 1.1** |
| per-seed | 5.93, 0.23, 5.36, 4.57 | 0.53, 0.34, 0.73, 3.14 |
| seeds above 2.0 | 3 / 4 | 1 / 4 |

Across all 8 v4 runs, `corr(wm_rollout_error, eats/agent/1k) = -0.74`,
`corr(wm_rollout_error, lifespan) = -0.64`.

The proposed explanation, which is a design flaw and not a tuning problem:
**the memory read is part of the latent `z = [s ‖ vision ‖ memory]` that the
dynamics head has to predict, and it is discontinuous.** A slot write
replaces 48 of its dimensions in a single tick. The world model is being
asked to regress onto a signal with step changes it cannot see coming, its
open-loop error blows up, and everything downstream that depends on it — the
empowerment drive, the multi-step consistency loss, the curiosity term —
degrades with it.

The obvious fix is to keep the memory read out of the dynamics head's
prediction target (predict `z_pre` only, while the policy and critic still
read the full `z`). That decouples the two and is a one-line change to the
spec's `dyn.Wz` width — but it is a **new study**, not a patch to slip into
this one.

### 4. What is NOT concluded here

- **Not** that the slow core, the dual-discount critic, the evolved drives or
  the costly comm channel fail. This ladder has three arms; only the episodic
  memory is isolated by an ablation. Everything else moves as a bundle.
- **Not** that v4 cannot work. 5,000 ticks is ~5 agent lifetimes and a handful
  of generations, against a model with 31% more parameters, a slow core whose
  time constants reach 512 ticks and a BPTT window of 32. Undertraining is the
  prime suspect and this campaign cannot rule it out.
- **Not** a claim about any effect size. Every SD here is large relative to
  its mean (trigram entropy ±1.2, tiles/agent ±36, eats ±0.69 on 0.69).

## Recommended defaults, on this evidence

- Keep `brain.version: 3.5` as the recommended architecture. v4 is available
  and correct, not yet better.
- If running v4, run it with **`brain.v4.memory_slots: 0`** until the latent
  discontinuity is fixed.
- The V4.0 scoring knobs (`action_cost_model: flat`,
  `fitness_model: reproduction`) remain the honest baseline for emergence
  work even though they cost performance — that was always their point — but
  they do not fix the always-valid-action attractor and should not be sold as
  doing so.

## What is committed here

- `configs/` — the three configs, verbatim.
- `metrics/<arm>_rl_s<seed>.csv` — per-generation `--metrics-csv` output, all
  12 runs.
- `emergence.json` — the **full** `analyze_emergence.py` output for all 12
  runs, including per-quarter time courses.

Raw `agent_actions_*.csv` logs (~16-25 MB per run) are not committed; the
command above regenerates them from the committed config + seed.

## Caveats

- 5,000 ticks, 4 seeds. See "What is NOT concluded" above.
- The population cap (`max_population: 30`) binds in every run, as it did in
  the v3.5 baseline. Survival does not discriminate strongly here.
- The cultivation null is saturated (0.35-0.59) because the worlds are
  food-dense. This measurement wants a scarcer world.
- `v40_scoring` changes five knobs at once against v3.5 (reward diet, action
  costs, signal cost, fitness model, instincts). It attributes to the bundle,
  not to any one knob.
