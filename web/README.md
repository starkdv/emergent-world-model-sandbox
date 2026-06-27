# 3D Voxel Frontend (`web/`)

A Minecraft-style voxel view of the simulation. The Python sim streams its
state read-only (the F0 bridge) over Server-Sent Events (the F3a server); this
browser client (Three.js) renders it live — phases F0/F3a/F3b of
[`docs/FRONTEND_3D_PROPOSAL.md`](../docs/FRONTEND_3D_PROPOSAL.md).

## Run

```bash
# from the repo root — builds the world FROM config/default.yaml (size, biomes,
# population, learning) and serves the viewer; stdlib only, no extra Python deps
python -m render.server
# then open http://127.0.0.1:8000
```

By default the world is **config-driven**: it uses `config/default.yaml` for the
map size, terrain generator (heightmap → biomes), climate, and population, then
spawns learning agents with reproduction on so the population sustains itself
(agents forage and breed instead of dying out).

`python -m render.server --help` prints all modes with explanations. The
common ones:

```bash
python -m render.server --config config/training_easy.yaml   # a different world
python -m render.server --demo                               # fixed self-contained scene
python -m render.server --checkpoint data/states/run.pkl     # fly around a saved run (W6b)
python -m render.server --tps 15

# record a run, then replay it on a loop (no re-simulation)
python -m render.recorder --out run.jsonl --ticks 600
python -m render.server --replay run.jsonl
```

### Brains, planning, curiosity, logging, training (what each does)

The world is whatever `--config` says — pick the brain there. Learning (PPO) is
**on by default** so agents improve while you watch (and the planner's world
model trains); add `--no-learn` to freeze them.

```bash
# Watch the v3.5 social brain (78-dim obs + SIGNAL) learn with PPO
python -m render.server --config config/worldmodel_v35.yaml

# Watch agents PLAN + be CURIOUS with a per-agent world model.
# planner  → each agent imagines short latent rollouts and picks the best action
# curiosity→ intrinsic reward for surprising transitions (more exploration)
python -m render.server --config config/planning_curiosity_v35.yaml

# Capture data WHILE watching, then train an offline world model from it
python -m render.server --config config/worldmodel_v35.yaml \
    --world-model-log --log --log-dir data/logs
python scripts/train_world_model.py \
    --transitions "data/logs/transitions_*.csv" \
    --config config/worldmodel_v35.yaml \
    --out data/world_models/wm.pt --report data/world_models/wm.txt

# Analyse behaviour (e.g. planning vs not — see docs/sample_planning_curiosity/)
python scripts/analyze_logs.py --file data/logs/agent_actions_*.csv

# Start agents from pre-trained genome weights (.npz) instead of random
python -m render.server --config config/worldmodel_v35.yaml \
    --load-weights data/weights/best.npz
```

| Flag | What happens |
|---|---|
| `--config FILE` | Builds the real world from that YAML — **the brain version lives here** (`config/worldmodel_v35.yaml` = v3.5 + PPO; `config/planning_curiosity_v35.yaml` = v3.5 + planner + curiosity). |
| (default) | RL learning **on** — agents learn live; if the brain has a world-model head it trains too. |
| `--no-learn` | Freeze policies (ablation / faster). The planner then uses an untrained world model, so it won't get better. |
| `--log` | Per-action + per-state CSVs → feed `scripts/analyze_logs.py`. |
| `--world-model-log` | Transition CSVs `f(obs,a)→(next_obs,r,done)` → feed `scripts/train_world_model.py`. |
| `--log-dir DIR` | Where the above go (default `data/logs`). |
| `--load-weights NPZ` | Seed every agent from trained genome weights (migrated onto the configured brain if needed). |

Two "world models" — don't confuse them: the **planner/curiosity** use a
*per-agent* latent head that lives in each genome and trains live (Brain v3
Phase 4). `scripts/train_world_model.py` trains a separate *population* world
model offline from the transition logs (for dream-based evolution,
`agents/dream.py`). See `docs/sample_planning_curiosity/` and
`docs/sample_world_model/` for measured results of each.

> **Codespaces:** the server binds `127.0.0.1:8000`; forward that port (VS Code
> does it automatically — click the popup / the Ports tab) and open the URL.
> A planning world runs slower than `--tps` (the planner imagines rollouts every
> decision); the view stays live, it just advances fewer ticks/second.

## Controls

| Input | Action |
|---|---|
| drag | orbit |
| scroll | zoom |
| WASD / Q,E | free-fly (move / down,up) |
| F | follow the next agent (chase cam; a bright **yellow marker** floats above it). When the followed agent dies, the camera **auto-hands off to the nearest living agent** so it never freezes on a corpse. Press F to cycle; cycles back to free cam |
| V | toggle the **5×5 vision grid** overlay — highlights the exact tiles the selected/followed agent perceives (yellow = its own tile, brighter cyan = tiles ahead). Egocentric, rotates with the agent's facing; mirrors `utils/agents/perception.py` |
| R | reset camera |
| click anything | inspect it — **agent** (cohort, energy, lineage, generation), **tree/berry** (category, type, maturity/freshness, on-fire), or **tile** (terrain, elevation, fertility, moisture); click empty space to clear |

## This is the real simulation, not a demo

By default the viewer runs the **actual** simulation built from
`config/default.yaml` — the same `World`, systems, and the configured **brain**
that `main.py` runs. The bridge is **read-only**; it never fabricates or alters
anything. (`--demo` exists only for a fixed offline scene.) Every mechanic the
brain/world code produces is surfaced — here is the fidelity map:

| Simulation mechanic | Phase | In the 3D world |
|---|---|---|
| Elevation / heightmap | W2 | voxel column height |
| Biomes (soil/rock/water/sand) | W2 | block palette; water = translucent layer |
| Fertility / moisture | W0/W1 | terrain tint; in the tile inspector |
| Day/night, light, season | W1 | sky color, sun angle, lighting; HUD season |
| Weather (rain / drought) | W1 | falling rain streaks / dusty haze tint; HUD |
| Plants growing | — | tree model **scales with maturity**; inspector shows maturity |
| Food freshness / decay | — | inspector freshness; consumed → orange puff |
| Seeds (viability) | — | seed model; inspector viability |
| Hazards / thorns | W3 | spiky model; tile/agent hazard |
| Wildfire | W3 | burning objects glow + flicker + flame particles |
| Agents (pos/facing/energy) | — | creature grounded on the surface, yawed, energy bar |
| Agent age | — | inspector (% of max age) |
| Agent **current action** (brain output) | brain | inspector "last action" |
| Inventory (carrying food/seed) | — | inspector 🍒/🌱 |
| Lineage / generation | — | per-lineage body tint; inspector |
| Reproduction (births) | — | green birth pop; population in HUD |
| Death | — | grey death puff; removed |
| SIGNAL / pheromone field | W4 | cyan ground glow; HUD "signal" |
| Trade (give) | W5 | HUD "trade" flag (per-give particles: TODO) |
| Brain architecture | v2/v3/3.5 | HUD `brain` (output 9 = v3.5 + SIGNAL) |

Inherently internal (not directly drawable): the neural weights themselves,
the GRU hidden state, and the world-model latent — these are *expressed*
through the agent's behaviour and current action rather than shown as geometry.
The kin-similarity sense is **Brain v3.6** (designed, not yet built —
`BRAIN_V3_PROPOSAL.md` §9).

## How it connects

```
Python sim ──► render/state_bridge.py ──► render/server.py (SSE) ──► web/main.js
 (unchanged)     snapshot + per-tick deltas    /api/stream            Three.js
```

The bridge is **read-only** — the viewer never affects the simulation. Three.js
(r160) is **vendored locally** in `web/vendor/three/` and resolved via an import
map (no build step, no `node_modules`, **no internet** — works in Codespaces /
behind a firewall). Blocky rendering is the default; smooth terrain is a planned
client-side toggle over the same elevation field (proposal §11).

**Building a different frontend (Unity, etc.)?** The full wire protocol — SSE
framing, snapshot/delta JSON schemas, terrain-grid decoding, coordinate
conventions, and a Unity C# client — is in
[`docs/UNITY_STREAM.md`](../docs/UNITY_STREAM.md).
