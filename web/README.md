# 3D Voxel Frontend (`web/`)

A Minecraft-style voxel view of the simulation. The Python sim streams its
state read-only (the F0 bridge) over Server-Sent Events (the F3a server); this
browser client (Three.js) renders it live — phases F0/F3a/F3b of
[`docs/FRONTEND_3D_PROPOSAL.md`](../docs/FRONTEND_3D_PROPOSAL.md).

## Run

```bash
# from the repo root — builds a heightmap world (elevation + day/night sky +
# agents) and serves the viewer; stdlib only, no extra Python deps
python -m render.server
# then open http://127.0.0.1:8000
```

Useful flags:

```bash
python -m render.server --width 100 --height 100 --agents 30 --tps 15
python -m render.server --checkpoint data/states/run.pkl   # fly around a saved run (W6b)
```

## Controls

| Input | Action |
|---|---|
| drag | orbit |
| scroll | zoom |
| WASD / Q,E | free-fly (move / down,up) |
| F | follow the next agent (cycles, then back to free cam) |
| R | reset camera |

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
