# Sample brain-cohort competition run

Two brain architectures competing in **one shared world**: ~15% of the
founding agents run the **old** brain (v2, legacy GRU-MLP) and the rest run the
**new** brain (v3, attention). Offspring breed true — a v2 parent produces v2
children — so the two cohorts compete head-to-head over evolutionary time.

Produced by the headless competition runner (4,000 ticks on
`config/default.yaml`, which carries the `competition:` block):

```bash
python scripts/competition_run.py --ticks 4000 --out <dir>
```

The per-action log carries a `cohort` column, so `scripts/analyze_logs.py`
emits a **⚔️ COHORT COMPARISON** section comparing the two populations.

Files here:

| File | What |
|---|---|
| `metrics.csv` | One aggregate row per generation (population, food/plant/seed counts, mean energy/age, mean fitness, soil means). |
| `analysis.txt` | Full analyzer report including the ⚔️ COHORT COMPARISON section. |
| `agent_actions_sample.csv.gz` | Per-action log **down-sampled 1-in-40** (~9.6k of 385k rows) so it fits in git. The `cohort` column is the last field. Re-run the command above for the full ~27 MB log. |

## Headline results

Founders: `{v2-old: 1, v3-new: 7}` → final at tick 4000: `{v3-new: 96, v2-old: 4}`.
The new architecture out-reproduced and out-survived the old one.

| Cohort | Agents (lifetime) | Mean age | Max age | Mean fitness | PICK_UP share |
|---|---|---|---|---|---|
| **v3-new** | 1,715 | 101.3 | **493** | **10.57** | 3% |
| v2-old | 255 | 98.3 | 259 | 8.95 | 1% |

- **v3-new wins on every survival/fitness proxy**: higher mean fitness
  (10.57 vs 8.95), longer max lifespan (493 vs 259 ticks), and more
  resource-gathering (3% vs 1% PICK_UP).
- The cohort started at a single v2 agent (15% of 8 founders) and bred true to
  ~12% of the population mid-run before the new brain's advantage compounded and
  drove it down to 4% by tick 4000.
- Both cohorts reached 100% EAT success; the difference is in foraging
  efficiency and reproduction, not basic feeding competence.

## How to read the cohort column

`agent_actions_sample.csv.gz` is the standard AgentLogger action log with one
extra trailing column, `cohort` (`v2-old` or `v3-new`). To recompute the
comparison from the sample:

```bash
zcat agent_actions_sample.csv.gz > /tmp/actions.csv
python scripts/analyze_logs.py --file /tmp/actions.csv
```

## Reproduce / tune

The competition is driven by the `competition:` block in `config/default.yaml`:

```yaml
competition:
  enabled: true
  old_fraction: 0.15      # fraction of founders on the old brain
  old_brain_version: 2    # legacy architecture
  old_label: "v2-old"
  new_label: "v3-new"
```

Set `enabled: false` for a single-cohort world. Note the two architectures must
share an observation layout: v2 and v3 are both 72-dim (compatible); v3.5 is
78-dim (SIGNAL) and **cannot** share a world with v2/v3.
