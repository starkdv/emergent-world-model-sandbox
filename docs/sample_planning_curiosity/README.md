# Planning + curiosity vs baseline — Brain v3.5

Does giving agents a **world model to plan and be curious with** change how they
behave? This is a controlled A/B: two headless Brain v3.5 + PPO runs, **same
seed (42), same 3,000 ticks, same world**, differing only in whether each agent
has a **per-agent latent world model** driving:

- **Planning** — `agents/planner.py` `LatentPlanner`: from the current hidden
  state the agent imagines short latent rollouts for each candidate action and
  picks the one with the best predicted return (depth 2, 6 samples).
- **Curiosity** — `agents/curiosity.py`: intrinsic reward = prediction error of
  the agent's own latent dynamics head, so *surprising* transitions are
  rewarded (exploration without a hand-crafted bonus).

| | Baseline | Planning + curiosity |
|---|---|---|
| config | `config/worldmodel_v35.yaml` | `config/planning_curiosity_v35.yaml` |
| `brain.world_model.enabled` | false | **true** |
| `brain.world_model.planner.enabled` | false | **true** |
| `learning.curiosity.enabled` | false | **true** |

Both: v3.5 (78-dim obs + SIGNAL), `learning.algorithm: ppo`, 64×64 world,
population capped at 30, `--seed 42 --generations 3` (3,000 ticks).

## Reproduce

```bash
python main.py --no-viz --config config/worldmodel_v35.yaml \
    --learning --mode rl --seed 42 --generations 3 --log --log-dir base
python main.py --no-viz --config config/planning_curiosity_v35.yaml \
    --learning --mode rl --seed 42 --generations 3 --log --log-dir treat
python scripts/analyze_logs.py --file base/agent_actions_*.csv
python scripts/analyze_logs.py --file treat/agent_actions_*.csv
```

## Result — agents behave clearly differently

The planning + curiosity agents are **more exploratory and far more
goal-directed**, and they survive and score better.

| Metric | Baseline | Planning+curiosity | Change |
|---|---|---|---|
| Avg peak fitness | 38.7 | **57.8** | **+49%** |
| Mean fitness (final gen) | 40.8 | **70.7** | **+73%** |
| Avg lifespan | 422 | **548** ticks | +30% |
| Max age (final gen) | 718 | **959** | +34% |
| Mean energy (final gen) | 109 | **165** | +51% |
| Tiles visited **per agent** | 27.7 | **54.9** | **+98%** (curiosity → ~2× exploration) |
| Unique tiles (global) | 1,886 | 2,358 | +25% |
| EAT attempts | 185 | **1,196** | **+546%** |
| PICK_UP share | 0.7% | **3.7%** | ~5× |
| USE (plant seed) / seeds planted | 226 | **929** | ~4× |
| Strategy entropy | 1.61 (51%) | **2.44 (77%)** | richer action repertoire |
| Behavioural novelty (pairwise JS) | 0.320 | 0.092 | **lower** — agents *converge* on a shared effective plan |

### Action mix — less aimless turning, more doing

| Action | Baseline | Planning+curiosity |
|---|---|---|
| TURN_LEFT + TURN_RIGHT | **51.4%** | 35.6% |
| MOVE_FORWARD | 11.0% | **15.9%** |
| PICK_UP | 0.7% | **3.7%** |
| EAT | 0.2% | **1.3%** |
| USE | 0.3% | **1.0%** |
| WAIT | 14.0% | 22.8% |

## Reading it

- **Curiosity widened exploration**: per-agent tile coverage nearly **doubled**
  (27.7 → 54.9) — agents seek out novel, surprising states instead of circling.
- **Planning made them goal-directed**: the baseline spends **half its actions
  just turning** and almost never interacts (EAT 0.2%, USE 0.3%); the planning
  agents move forward more and run the full **forage → eat → plant** loop far
  more often (EAT ×6.5, PICK_UP ×5, seeds planted ×4), which shows up as higher
  energy, lifespan and fitness.
- **They converge rather than diverge**: strategy entropy *per agent* rises
  (each agent uses a richer repertoire) while *pairwise* behavioural novelty
  **falls** — agents independently discover the *same* effective planned
  strategy, the signature of a shared, useful world model rather than noise.

This is the per-agent latent world model from Brain v3 Phase 4 in action; it is
distinct from the offline `PopulationWorldModel` in `docs/sample_world_model/`
(that one is for dream-based evolution). Both are "world models"; this page is
about the one agents use *online* to plan and explore.

Full analyzer reports: `baseline_analysis.txt`, `planning_curiosity_analysis.txt`.
Per-generation metrics: `*_metrics.csv`.

> The planner here is a minimal random-shooting MPC (replans every tick, random
> continuation actions). See **`docs/PLANNING_PROPOSAL.md`** for how it works
> today and a staged plan to make it much stronger (policy-guided rollouts, CEM,
> Dreamer-style imagination learning).
