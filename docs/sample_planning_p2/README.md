# Planner P2 — categorical CEM + λ-returns (measured)

> **Multi-seed update (4 seeds): this CEM win did NOT replicate.** The +32% was seed luck; across seeds CEM ties the baseline on peak fitness and is *worse* on final fitness while ~20% slower. Not recommended by default at this scale. See `../sample_planning_multiseed/`.

Phase **P2** of [`../PLANNING_PROPOSAL.md`](../PLANNING_PROPOSAL.md). Two
self-contained upgrades to the latent planner (`agents/planner.py`):

- **Categorical CEM** (`strategy: cem`): instead of one round of random/policy
  rollouts, maintain a per-step action distribution; each iteration sample a
  population of sequences, keep the top-scoring **elites**, and refit the
  distribution toward them. Concentrates the search on promising sequences
  (cross-entropy method / MPPI-style).
- **λ-returns** (`lam`): score a rollout with a TD(λ) return over the imagined
  trajectory instead of only an end-of-horizon bootstrap.

Unit tests: `tests/test_planner.py`. **Not done in P2:** model-error discipline
via a dynamics *ensemble* — it requires a genome-length change (append-only bump
+ migration), so it is deferred and documented in the proposal.

## Matched A/B (same seed 42, 64×64, v3.5 + PPO, 2000 ticks, depth 2)

| Variant | rollouts / decision | peak fitness | EAT | seeds planted | WAIT | ticks/s |
|---|---|---|---|---|---|---|
| `shooting` (baseline) | 6 | 33.8 | 306 | 230 | 21.9% | 6.54 |
| `policy_shooting` (P1) | 6 | 41.1 | 341 | 289 | 22.6% | 6.35 |
| **`cem` (P2)** | 18 (3×6) | **44.5** | **487** | **396** | 26.9% | 5.19 |

## Finding

- **CEM is the strongest controller**: peak fitness **+32% over baseline** and
  **+8% over P1**, with the most foraging (eating +59%, planting +72% vs
  baseline). The cross-entropy refinement finds better action sequences than a
  single round of shooting.
- **Cost**: CEM does `cem_iters × samples` rollouts per decision (3× here) for a
  **~20% throughput drop** (5.19 vs 6.54 ticks/s) — the rollouts are not the only
  per-tick cost, so wall-clock scales sub-linearly with the rollout count.
- Ordering on competence is **cem > policy_shooting > shooting**, matching the
  proposal's expectation that better search compounds the P1 gain.

Recommended P2 config: [`config/planning_p2_v35.yaml`](../../config/planning_p2_v35.yaml)
(`strategy: cem`, `cem_iters: 3`, `cem_elite_frac: 0.3`). λ-returns (`lam < 1`)
are available and unit-tested but were left at `lam: 1.0` here.

## Caveats

Single seed, 2000 ticks, 64×64, population 30, in-silico fitness — effect sizes,
not significance. CEM's budget (18 rollouts) is larger than the baselines' (6);
the comparison shows quality-per-tick is favourable but not a same-budget test.
Multi-seed and same-budget sweeps are future work.

## Reproduce

```bash
python main.py --no-viz --config config/planning_p2_v35.yaml \
    --learning --mode rl --seed 42 --generations 2 --log --log-dir cem
python scripts/analyze_logs.py --file cem/agent_actions_*.csv
```

Full reports: `*_analysis.txt`; CEM per-generation metrics: `cem_metrics.csv`.
See `../sample_planning_p1/` for the P1 (policy_shooting) A/B.
