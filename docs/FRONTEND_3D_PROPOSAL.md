# Frontend 3D / Voxel World — Architecture Proposal

**Status:** PROPOSAL — for review. No code yet. This document is the plan for
moving the simulation's *visualization* from the current 2D/isometric renderers
to a **Minecraft-like voxel world**. Author: Karan Vasa.

**Branch (when started):** `claude/frontend-3d`
**Scope (proposed):** a new renderer package; a read-only **world→render state
bridge**; no change to `world/`, `agents/`, or the simulation loop semantics.
**Inputs reviewed:** `utils/render.py` (ASCII), `utils/ui/pygame_renderer.py`
(2D top-down), `utils/ui/gpu_renderer.py` (ModernGL isometric 2.5D),
`world/tiles.py`, `world/terrain_generation.py` (W2 heightmap → elevation),
`world/environment.py` (day/night/season/weather), `world/checkpoint.py` (W6b
full-state serialization), `WORLD_UPGRADE_PROPOSAL.md` (the persistent-world /
spectator track).

---

## 1. Why move off 2D / isometric

The simulation already produces everything a 3D world needs, but the current
views flatten it:

- **W2 gave every tile a real `elevation` ∈ [0, 1]** (mountains, downhill
  rivers, basins) — but the 2D renderer ignores height entirely and the
  isometric one only fakes a little depth. The terrain *is* 3D data rendered in
  2D.
- **W1 gives a live sky** — `time_of_day`, `light`, `temperature`, `season`,
  rain/drought — with nowhere to put it. A voxel world with a day/night sky and
  weather makes the climate legible at a glance.
- **W3–W5 made the world social and ecological** — species, hazards, signals,
  territories, trades. These read far more naturally as 3D entities and volumes
  (a pheromone field as a translucent ground glow; territory as space agents
  occupy) than as 2D tiles.
- **The persistent-world / spectator track** (`WORLD_UPGRADE_PROPOSAL.md` §
  "Determinism, checkpointing, and an event stream") wants an always-on world
  people can *watch*. A browser voxel client is the natural surface for that;
  the desktop Pygame/ModernGL windows are not shareable.

The goal is not a game — there is no player avatar, no block-breaking by a
user. It is a **voxel rendering of an autonomous world**: the same agents and
ecology we already simulate, shown as a living Minecraft-style landscape you can
fly around and inspect.

## 2. What "Minecraft-like" means here (concretely)

| Element | Source in the sim today | Voxel representation |
|---|---|---|
| Ground height | `tile.elevation` ∈ [0,1] (W2) | a column of blocks `floor(elevation × MAX_H)` tall (MAX_H ≈ 16–32) |
| Terrain / biome | `tile.terrain_type` (soil/rock/water/sand) + fertility/moisture | block palette: grass/dirt, stone, water (translucent), sand; shaded by fertility/moisture |
| Water | `WATER` tiles + moisture | translucent blue blocks at the local water level; rivers follow the W2 carve |
| Plants / food / seeds | `WorldObject` + components (`type_id`) | small block/cross-sprite models on top of the column (berry bush, sapling, fruit) |
| Hazards (W3 thorns, fire) | `tile_effect.contact_damage`, `FireSystem` | spiky blocks; fire = emissive animated blocks |
| Agents | `agent.x/y`, `direction`, `energy`, `alive` | a low-poly creature/blocky mob, oriented by `direction`, with an energy bar billboard |
| Signals (W4) | `world.pheromones` float grid | translucent colored glow on the ground cell, intensity = field value |
| Sky / light | `environment.time_of_day`, `light`, weather | skybox color + sun/moon angle from `time_of_day`; rain/drought particle + fog |
| Trade / give (W5) | `interaction_kind="give"` events | brief particle/arc between two agents |

Everything above already exists as plain data on `World`; the renderer is a
*pure consumer*.

## 3. Architecture — three options

### Option A (recommended): Web client (Three.js) + thin Python state server

```
  Python sim ──► StateBridge ──► WebSocket/JSON(+binary) ──► Browser (Three.js)
  (unchanged)    (new, read-only)   per-tick deltas          voxel renderer + camera
```

- The sim runs headless and streams world state; the browser does all
  rendering with WebGL via Three.js (or babylon.js).
- **Pros:** shareable (just a URL — exactly what the persistent-world/spectator
  track needs); GPU-accelerated voxel meshing libraries exist; decouples render
  from sim so the sim stays pure Python; multiple spectators for free; trivial
  to record/replay from the checkpoint + delta stream.
- **Cons:** two languages (Python + TS/JS); needs a small server (FastAPI +
  websockets) and a build step (Vite). Network bandwidth must be managed with
  deltas + chunking.

### Option B: Native 3D in-process (extend ModernGL, or Ursina/Panda3D)

- Promote `gpu_renderer.py` from isometric 2.5D to a true 3D voxel camera, or
  adopt a Python 3D engine (Ursina is Minecraft-oriented and quick; Panda3D is
  heavier but capable).
- **Pros:** single language; reuses the existing ModernGL context, instancing,
  and the `--gui --gpu` entry point; no network. Fastest path to "something on
  screen."
- **Cons:** not shareable (desktop window only); ties rendering to the sim
  process; Ursina/Panda3D add heavy deps; we'd hand-write voxel meshing.

### Option C: Offline export (glTF / `.vox` / schematic per checkpoint)

- Convert a W6b checkpoint into a standard 3D asset, viewed in any glTF viewer
  or imported into Blender / a voxel editor.
- **Pros:** trivial; great for stills, papers, and debugging a single frame.
- **Cons:** not live. Best as a *complement*, not the main view.

**Recommendation:** **Option A** as the strategic target (it is the only one
that serves the persistent-world/spectator goal), with **Option C** delivered
first as a cheap milestone (it forces us to nail the voxel mapping with zero
networking), and the existing 2D/iso renderers kept as-is for local debugging.
If a no-network desktop view is wanted sooner, Option B's ModernGL extension is
the fallback. The phased plan in §7 reflects this.

## 4. The world→render state bridge (the key new component)

A single read-only module (`render/state_bridge.py`) is the contract between
sim and renderer, regardless of which front end consumes it. It mirrors the W6b
checkpoint serialization but is **lossy, render-focused, and delta-friendly**.

- **Full snapshot** (on connect / first frame): a compact description of the
  static-ish world — per-tile `(elevation_quantized, terrain, biome_shade)`,
  packed as typed arrays (one byte per field), plus the object list and agent
  list. Chunked by region so the client can stream/cull.
- **Per-tick delta**: only what changed — moved/spawned/removed objects,
  agent `(x, y, dir, energy, alive)`, dirty pheromone cells, and the scalar
  sky state (`time_of_day`, `light`, weather). Tiles rarely change, so the
  terrain mesh is sent once and only re-sent for the cells that mutate
  (fire burns, sand spreads, rivers — all already localized).
- **Encoding**: JSON for structure + base64/binary typed arrays for the grids
  (a 100×100 terrain byte-grid is 10 KB raw, ~negligible gzipped). The delta
  stream is what keeps bandwidth flat over a long run.
- **Reuses W6b**: the bridge and the checkpoint share the same field
  extraction; a checkpoint *is* a valid full snapshot, so "load a `.pkl` and
  fly around it" (Option C / replay) falls out for free.

This bridge is the only sim-touching code, and it is **read-only** — it never
mutates the world, so it cannot affect determinism or the science.

## 5. Voxel terrain construction

- **Heightmap → columns.** `h = round(elevation × MAX_HEIGHT)`; fill the column
  with a biome-appropriate stack (e.g. stone below, dirt+grass on top; sand for
  beaches; water filled to a sea level derived from the W2 water quantile).
- **Chunking.** Partition the grid into 16×16 chunks (Minecraft-sized). Each
  chunk meshes independently and re-meshes only when one of its tiles is dirty
  — and tile mutations in this sim are already rare and localized.
- **Greedy meshing + face culling.** Standard voxel optimizations: merge
  coplanar faces, never emit faces between two solid blocks. A 100×100×~24 world
  is tiny by voxel standards; this is comfortably real-time.
- **Shading.** Tint grass by fertility, darken by moisture; a simple ambient +
  sun-direction lambert from `environment` for time-of-day. Water is
  translucent with a slight animated normal.

## 6. Entities, camera, and overlays

- **Agents.** One instanced low-poly model per agent, positioned at its
  column-top, yawed to `direction`, tinted/scaled by `energy`; a small
  floating energy bar (billboard). Dead agents fade out. Newborns (reproduction)
  pop in.
- **Objects.** Berries/plants/seeds as cross-sprites or tiny blocks keyed by
  `type_id` (reuse the palette the 2D renderers already define).
- **Signals & territory.** Pheromone field → a translucent emissive decal on the
  ground cell, alpha = field value (decays as the field does). Optional
  "territory" heat overlay from the W5 analyzer's visited-cell data.
- **Camera.** Three modes: **orbit** (drag to rotate around a point), **free-fly**
  (WASD + mouse, for inspection), and **follow-agent** (lock onto one agent —
  great for watching a single policy behave). Reuse the pan/zoom feel from the
  existing isometric renderer.
- **HUD.** Tick, population, season/weather, selected-agent inspector (energy,
  age, inventory, lineage) — the data is all already on the objects.

## 7. Phased delivery plan

| Phase | Deliverable | Acceptance |
|---|---|---|
| **F0** | **State bridge + voxel mapping spec.** `render/state_bridge.py` producing a full snapshot from a live world or a W6b checkpoint; documented byte layout. | A checkpoint serializes to a render snapshot; unit-tested field-by-field. |
| **F1** | **Offline voxel export (Option C).** Snapshot → glTF/`.vox`. Validates the heightmap→column mapping and palette with zero networking. | A checkpoint renders as a correct voxel scene in any glTF viewer. |
| **F2** | **Static web viewer (Option A, no live link).** Three.js loads one snapshot, builds chunked greedy meshes, free-fly camera, sky from the snapshot's `time_of_day`. | 100×100 world at 60 fps; terrain/biomes/water read correctly. |
| **F3** | **Live streaming.** FastAPI + WebSocket server; per-tick deltas; agents/objects move in real time; day/night animates. | A running sim is watchable live in the browser; bandwidth flat over 10k ticks. |
| **F4** | **Entities & overlays polish.** Agent models + energy bars, signal glow, follow-agent camera, HUD inspector, trade/fire particles. | A spectator can follow one agent and read its state; signals visible. |
| **F5** | **Spectator/persistent-world hooks.** Multi-client, replay-from-checkpoint scrubbing, shareable URL. | Two browsers watch the same world; replay a saved run. |

F0–F1 are pure Python and land first (cheap, testable, no JS). The web stack
(F2+) is where the TS/Vite/Three.js dependency enters; it can be deferred or
swapped for Option B (native ModernGL 3D) if a desktop-only view is preferred.

## 8. What deliberately does NOT change

- The simulation loop, world systems, brains, learners — untouched. The
  renderer is a read-only consumer.
- Determinism and checkpointing (W6b) are unaffected; the bridge reuses the
  same serialization and never mutates state.
- The existing renderers (`ConsoleRenderer`, `PygameRenderer`,
  `IsometricRenderer`) stay as local-debug tools; the voxel view is additive,
  selected by a new flag/entry point.
- The observation/genome layouts — the voxel world shows what agents do, it does
  not change what they perceive.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Two-language complexity (Python + JS) | Keep the bridge the *only* contract; the JS side is a pure renderer with no sim logic. F0–F1 stay pure Python so value lands before any JS. |
| Bandwidth at high agent counts / long runs | Send terrain once; per-tick deltas only; chunk + frustum-cull on the client; binary-pack grids. |
| Re-meshing cost on terrain change | Chunked meshing; only dirty chunks re-mesh, and tile mutations are already rare and local (fire, sand, rivers). |
| Scope creep into "a game" | Explicit non-goal: no player, no block editing. It is a *viewer* of an autonomous world. |
| Dependency weight (Three.js/Vite or Ursina/Panda3D) | Option A's deps live in a separate web subproject, not the sim's Python env; Option C needs only a glTF writer. |
| Effort vs. payoff if only used locally | Deliver Option C first (days, not weeks); commit to the web stack only once the spectator goal is confirmed. |

## 10. Open questions for review

1. **Target surface:** web/browser (Option A — shareable, the spectator goal)
   or native desktop 3D (Option B — simpler, no network)? The plan assumes A
   with C first; confirm.
2. **Fidelity:** blocky Minecraft aesthetic (fast, on-theme) vs. smoothed
   terrain (prettier, more work)? The plan assumes blocky.
3. **Live vs. replay-first:** is watching a *live* run the priority (F3), or is
   scrubbing *recorded* checkpoints (F1/F5) enough to start?
4. **Agent look:** abstract blocky mobs vs. distinct per-species models?
5. **Where should the web subproject live** — in this repo (`web/`) or a
   separate one consuming the bridge as an API?

---

*Once the answers to §10 are settled, F0 (the state bridge) can start
immediately — it is pure Python, reuses the W6b serialization, and unblocks
every later phase.*
