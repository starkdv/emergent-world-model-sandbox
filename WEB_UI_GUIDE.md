# Web UI Guide — Three.js Browser Renderer

**Author:** Karan Vasa
**License:** MIT

The Emergent World-Model Sandbox ships with a full, **simulation-grade 3D web
UI** built on [Three.js](https://threejs.org/). It runs the live Python
simulation behind a lightweight HTTP server and renders the world in real time
in any modern browser — a polished, "spectator-mode" alternative to the Pygame
and ModernGL desktop renderers.

This realises the **browser-based spectator client** described throughout
[SUGGESTIONS.md](SUGGESTIONS.md) (Parts 4, 7, and 9).

---

## Quick Start

```bash
# Launch the web UI (defaults to http://127.0.0.1:8000/)
python main.py --web

# Common variants
python main.py --web --open-browser            # auto-open the browser
python main.py --web --port 9000               # custom port
python main.py --web --host 0.0.0.0            # expose on your LAN
python main.py --web --mode neuroevolution     # pure evolution
python main.py --web --objects config/custom_objects.yaml   # custom objects
```

Then open the printed URL. Press **Ctrl+C** in the terminal to stop the server.

> **Note:** Three.js is loaded from a CDN (`cdn.jsdelivr.net`) via an ES-module
> import map. An internet connection is required the first time; the browser
> caches the modules afterwards. The Python server itself has **no extra
> dependencies** beyond the standard library.

---

## What You See

| Region | Contents |
|--------|----------|
| **3D viewport** | Terrain (height + colour by type), objects, and agents in a live, smoothly-animated 3D scene. Orbit / pan / zoom with the mouse. |
| **Top bar** | Live stats — status, tick, agents, generation, average energy, food, plants, simulation TPS, and render FPS. |
| **Simulation panel** (top-left) | Play/Pause, Step, Reset, a speed slider (0–20×), and toggles for Grid, Trails, and 3D terrain. |
| **Left rail** | Three tabs — **Objects** (UI card for every registered type), **Spawn** (place objects by clicking the world), and **Legend**. |
| **Inspector** (right) | Full live detail for the clicked agent / object / tile. |
| **Graph panel** (bottom-left) | Rolling population & average-energy line chart. |
| **Tooltip** | Hover an entity for a quick summary. |

---

## Controls

### Mouse
- **Left-drag** — orbit the camera
- **Right-drag** — pan
- **Scroll** — zoom
- **Hover** — tooltip for the entity under the cursor
- **Click** — open the inspector for an agent / object / tile (or place an
  object when the Spawn tool is active)

### Keyboard
| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `S` | Step one tick |
| `R` | Reset the world |
| `G` | Toggle grid overlay |
| `T` | Toggle agent trails |
| `Esc` | Close inspector / cancel spawn selection |

---

## UI For Every Object

The **Objects** tab is generated directly from the live
`ObjectRegistry`, so **every** built-in *and* custom (YAML) object type gets a
card showing:

- a colour swatch + display name + category
- the component badges it carries (`edible`, `seed`, `plant`, `fertilizer`,
  `tool`, `tile_effect`)
- expandable detail for each component and every cross-cutting spec
  (physics, interaction, observation, and tile-effect parameters)

Because the cards are data-driven, adding a new object via
`--objects my_objects.yaml` automatically gives it a full UI card, a spawn
entry, a legend colour, and correct 3D rendering — **no client changes
required.**

### Visual encoding

| Category | 3D shape | Colour |
|----------|----------|--------|
| food | sphere | registry colour, gentle glow scaled by freshness |
| seed | diamond (octahedron) | tan → green by growth; **gold glow** if agent-planted |
| plant | cone | registry green, scales with maturity, glows when mature |
| fertilizer | box | registry colour |
| tool / other | box / icosahedron | registry colour |
| terrain (sand) | rendered as the tile itself | terrain palette |
| agent | cone (oriented by facing) | energy gradient (green → red) |

---

## Spawning Objects

1. Open the **Spawn** tab in the left rail.
2. Click an object type to select it (the cursor becomes a crosshair).
3. Click any tile in the 3D world to place a fresh instance there.
4. Click the selected item again — or press `Esc` — to cancel.

Spawning respects the world's stacking rules, exactly like agent-dropped items.

---

## Architecture

```
Browser (Three.js)                        Python server (stdlib only)
─────────────────────                     ────────────────────────────
web/index.html         ──── GET /  ─────▶ utils/ui/web_server.py
web/static/js/app.js    ─── GET /api/* ─▶   WebSimulationServer
  ├─ net.js   (fetch)                         ├─ background sim thread
  ├─ world3d.js (scene)  ◀── JSON state ──    │    world.update() @ TPS×speed
  └─ ui.js    (panels)   ─── POST control ▶   └─ utils/ui/web_serialize.py
                                                   meta / state / terrain / inspect
```

- **State is polled** (~20 Hz) rather than streamed, which keeps the server
  dependency-free and robust across browsers. The client **interpolates** agent
  positions between snapshots for smooth motion.
- **Terrain** is only re-fetched when it actually changes (sand spread /
  reclaim / reset), tracked by a cheap server-side version counter.
- A single lock guards all world access, so every JSON snapshot is internally
  consistent even while the simulation thread is running.

### JSON API

| Method & Route | Purpose |
|----------------|---------|
| `GET /api/meta` | Static metadata: world size, terrain palette, **every** object definition, action list, observation layout. |
| `GET /api/state` | Per-frame snapshot: tick, counts, stats, objects, agents. |
| `GET /api/terrain` | Flat terrain-type grid (+ version). |
| `GET /api/inspect/agent/<id>` | Full agent detail (energy, traits, inventory…). |
| `GET /api/inspect/object/<id>` | Full object detail (all components). |
| `GET /api/inspect/tile?x=&y=` | Full tile detail (terrain, fertility, contents). |
| `POST /api/control` | `{cmd}` ∈ `pause`, `resume`, `toggle`, `step`, `set_speed`, `spawn`, `reset`. |

---

## Files

| File | Role |
|------|------|
| `utils/ui/web_server.py` | HTTP server + background simulation thread + control commands. |
| `utils/ui/web_serialize.py` | Pure world → JSON serialisation (meta / state / terrain / inspection). Unit-testable, no browser needed. |
| `web/index.html` | Page shell + Three.js import map. |
| `web/static/css/style.css` | Dark-theme UI styling. |
| `web/static/js/net.js` | `fetch` wrapper for the JSON API. |
| `web/static/js/world3d.js` | Three.js scene: terrain, objects, agents, trails, picking. |
| `web/static/js/ui.js` | DOM panels: HUD, object registry, spawn, legend, inspector, graph. |
| `web/static/js/app.js` | Orchestration: scene/camera/lights, render loop, polling, input. |

---

## Troubleshooting

- **Blank page / "Failed to connect"** — make sure the server is still running
  in the terminal and you opened the exact URL it printed.
- **Loading spinner never clears** — the browser could not reach the Three.js
  CDN. Check your internet connection (only needed once) or use the desktop
  renderers (`--gui` / `--gui --gpu`) offline.
- **Reset button disabled** — reset requires the world factory, which is always
  provided by `main.py`; it is only unavailable if you embed
  `WebSimulationServer` without a `world_factory`.
- **Port already in use** — pick another with `--port`.
