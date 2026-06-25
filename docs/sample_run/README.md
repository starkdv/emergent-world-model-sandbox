# Sample headless run (config-driven heightmap world)

Artifacts from a 5,000-tick headless run on `config/default.yaml` (brain v3 +
RL, heightmap biomes, reproduction on), produced with:

```bash
python main.py --no-viz --mode rl --generations 5 \
  --log --log-dir <dir> --log-frequency 5 \
  --metrics-csv <dir>/metrics.csv
```

Files here:

| File | What |
|---|---|
| `metrics.csv` | One aggregate row per generation (population, food/plant/seed counts, mean energy/age, mean fitness, soil means). |
| `analysis.txt` | Full `scripts/analyze_logs.py` report (action distribution, success rates, interaction detail, energy economy, per-species, 🧬 SOCIETY/ROLES, behavioural diversity, action n-grams, temporal phases, instinct-fade, verdict). |
| `agent_actions_sample.csv.gz` | The per-action log **down-sampled 1-in-40** (~10.7k of 427,538 rows) so it fits in git. Re-run the command above for the full ~42 MB log. |

## Headline results

- Population 20 → **100 (cap), held all run** — no die-off; 549 agents lived across the run (reproduction).
- Mean energy 104 → **179**; mean fitness 21 → **238**.
- Emergent **pick → plant → harvest** farming loop: 34,461 seeds planted by 320 agents; plants 220 → 1,104.
- EAT success 100%; adults (post-instinct-fade) still feed (the emergence-first claim, measured).
- Role entropy 1.79 bits (6 dominant roles); behavioural novelty (pairwise JS) 0.41.

Regenerate locally with the command above; logs land wherever `--log-dir`
points (default `data/logs/`, which is gitignored).
