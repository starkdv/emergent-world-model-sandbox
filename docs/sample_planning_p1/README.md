# Planner P1 — policy-guided rollouts (measured)

Phase **P1** of [`../PLANNING_PROPOSAL.md`](../PLANNING_PROPOSAL.md): replace the
random-shooting planner's **uniform-random rollout continuations** with
continuations sampled from the agent's **own policy** (Dreamer-style imagination),
so a rollout estimates "take action *a₀*, then act like myself" instead of "*a₀*
then a random walk."

Implemented in `agents/planner.py` behind `brain.world_model.planner.strategy`
(legacy `shooting` stays the default). Unit tests: `tests/test_planner.py`.

## Matched A/B (same seed 42, 64×64, v3.5 + PPO, 2000 ticks)

Three planners, identical except for the planner config:

| Variant | strategy | first_action | normalize |
|---|---|---|---|
| **shooting** (baseline) | `shooting` | uniform | off |
| **policy-greedy** | `policy_shooting` | **policy** | on |
| **policy-shooting** (P1) | `policy_shooting` | uniform | off |

| Metric | shooting | policy-greedy | **policy-shooting (P1)** |
|---|---|---|---|
| Avg peak fitness | 33.8 | 26.7 | **41.1  (+21%)** |
| Mean fitness (final gen) | 46.5 | 23.8 | **48.9** |
| Avg tiles / agent | 45.4 | 15.5 | 40.8 |
| EAT attempts | 306 | 38 | **341** |
| Seeds planted | 230 | 45 | **289  (+26%)** |
| WAIT share | 21.9% | 39.3% | 22.6% |
| ticks/s | 6.54 | 6.29 | 6.35 |

## Finding

- **Policy-guided continuations help** (policy-shooting beats baseline on fitness
  +21% and planting +26%) **at essentially equal cost** (≈3% slower) — the
  lower-variance value estimate makes the planner pick better first actions.
- **Biasing the *first* action toward the policy hurts** in a from-scratch run
  (policy-greedy collapses: WAIT 39%, exploration and eating crash). The policy
  is immature early, so committing the search to it removes the exploration that
  random first actions provide. **Lesson: keep the first action exploratory while
  the policy is still learning; only the continuations should be policy-guided.**
  Normalization was also on for the greedy variant; the principled P1 leaves it
  off.

The recommended P1 config is therefore `strategy: policy_shooting`,
`first_action: uniform`, `normalize: false` — shipped as
[`config/planning_p1_v35.yaml`](../../config/planning_p1_v35.yaml).

## Caveats

Single seed, 2000 ticks, 64×64, population 30, in-silico fitness. Effect sizes,
not significance; multi-seed replication is future work. `policy_topk` first
actions and `commit > 1` (control horizon) are implemented and unit-tested but
not swept here.

## Reproduce

```bash
# baseline (random shooting) and P1 (policy_shooting), same seed
python main.py --no-viz --config config/planning_curiosity_v35.yaml \
    --learning --mode rl --seed 42 --generations 2 --log --log-dir shoot
python main.py --no-viz --config config/planning_p1_v35.yaml \
    --learning --mode rl --seed 42 --generations 2 --log --log-dir p1
python scripts/analyze_logs.py --file shoot/agent_actions_*.csv
python scripts/analyze_logs.py --file p1/agent_actions_*.csv
```

Full analyzer reports: `*_analysis.txt`; per-generation metrics: `*_metrics.csv`.
