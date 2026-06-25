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

Useful flags:

```bash
python -m render.server --config config/training_easy.yaml   # a different world
python -m render.server --demo                               # fixed self-contained scene
python -m render.server --checkpoint data/states/run.pkl     # fly around a saved run (W6b)
python -m render.server --tps 15

# record a run, then replay it on a loop (no re-simulation)
python -m render.recorder --out run.jsonl --ticks 600
python -m render.server --replay run.jsonl
```

## Controls

| Input | Action |
|---|---|
| drag | orbit |
| scroll | zoom |
| WASD / Q,E | free-fly (move / down,up) |
| F | follow the next agent (chase cam; cycles, then back to free cam) |
| R | reset camera |
| click anything | inspect it — **agent** (energy, lineage, generation), **tree/berry** (category, type, on-fire), or **tile** (terrain, elevation, fertility, moisture); click empty space to clear |

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
loads from a CDN via an import map (no build step / `node_modules` needed).
Blocky rendering is the default; smooth terrain is a planned client-side toggle
over the same elevation field (proposal §11).
