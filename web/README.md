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

## What you see

- **Terrain** — voxel columns whose height is the W2 `elevation` field; block
  color by biome (soil/grass, rock, sand) shaded by fertility; water as a
  translucent layer.
- **Agents** — small creatures grounded on their tile's surface, tinted per
  lineage, yawed to their facing direction, with an energy bar; they tween
  between ticks so motion over hills is smooth.
- **Objects** — distinct models per category (berry/fruit, toxic nightshade,
  plant, seed, fertilizer, thorns).
- **Signals** — a cyan ground glow where agents have emitted pheromone (W4).
- **Sky** — skybox color, sun angle, and light follow the W1 day/night cycle;
  HUD shows season and weather (rain/drought).

## How it connects

```
Python sim ──► render/state_bridge.py ──► render/server.py (SSE) ──► web/main.js
 (unchanged)     snapshot + per-tick deltas    /api/stream            Three.js
```

The bridge is **read-only** — the viewer never affects the simulation. Three.js
loads from a CDN via an import map (no build step / `node_modules` needed).
Blocky rendering is the default; smooth terrain is a planned client-side toggle
over the same elevation field (proposal §11).
