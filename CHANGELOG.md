# Changelog

## [Unreleased] — fix: 3D viewer — seeds/plants/berries now look different + trees grow

- **Seeds, plants and berries all rendered as the same red blob → fixed.** The
  object type_ids are `berry`, `berry_plant`, `berry_seed` — and the client
  classified by **type_id substring before the authoritative `category`**, so
  `berry_plant`/`berry_seed` matched `"berry"` and became *food*. Everything
  drew as the red food icosahedron. `categoryOf` now trusts the bridge's
  `category` field first; type_id is only a fallback (seed/plant before berry).
  Result: berries = red icosahedra, seeds = tan cubes, plants = trees.
- **Plants now visibly grow like trees.** A plant scales from a small sapling
  (0.35×) to a full tree (1.4×) with maturity, and its color **lerps from pale
  yellow-green (young) to deep forest green (mature)** via per-instance color —
  so young and mature plants are obviously different. (The bridge already
  streams maturity each tick after the earlier delta fix.)

## [Unreleased] — fix: 3D viewer (follow marker, visible objects, live growth, offline)

Diagnosed by driving the real viewer in a headless browser against a live
`render.server` (screenshots + DOM/state probes), not by inspection.

- **Follow marker never appeared → fixed.** The follow camera tracked agents by
  a snapshot **index**, so the instant the followed agent died (avg lifespan
  ~195 ticks → seconds of wall-clock) the camera froze on a dead id and the
  marker vanished. Rewrote follow to track a live agent **by id** and, when it
  dies, **auto-hand off to the nearest living agent**. The marker is now bigger,
  drawn on top (never hidden behind hills), and the chase converges faster.
- **Seeds / trees hard to see → enlarged + recolored.** Trees were nearly the
  same green as the soil and all objects were small. Trees are now taller with a
  brown trunk + dark-green canopy; berries/seeds are bigger and brighter.
- **Plants didn't visibly grow → fixed.** `StateTracker` only re-emitted an
  object when its **position** changed, so a plant maturing in place never
  updated. It now re-emits when the render-relevant state changes (position,
  category, or maturity/freshness bucketed to 0.1), so trees grow live.
- **No more CDN dependency.** Three.js (r160) is **vendored** in
  `web/vendor/three/` and resolved via the import map, so the viewer works with
  **no internet** (Codespaces / firewalled). Previously a blocked `unpkg.com`
  left the page blank.
- **HUD brain version was wrong in cohort mode.** It sampled the *first* agent's
  brain — and the v2-old founders are spawned first — so a 96%-v3 world showed
  "v2". The bridge now reports the **full `brain_versions` distribution** (in
  both snapshot and every delta) and the HUD shows the live mix
  (e.g. `v3·96 + v2·4`); `brain_class`/`brain_output_size` now describe the
  *majority* brain. Documented in `docs/UNITY_STREAM.md`.

## [Unreleased] — feature: brain-cohort competition (old vs new, in one world)

- **Two brain architectures compete in a single shared world.** A new
  `competition:` block in `config/default.yaml` seeds ~`old_fraction` (default
  15%) of the founding agents with the **old** brain (v2) and the rest with the
  **new** brain (v3). Each agent records `brain_config_used` + a `cohort` label.
- **Offspring breed true.** `clone_agent` temporarily restores the parent's
  brain config while constructing the child, so a v2 parent produces v2 children
  (matching genome length) and the two cohorts compete over evolutionary time.
- **Analyzer compares the cohorts.** `scripts/analyze_logs.py` emits a new
  **⚔️ COHORT COMPARISON** section (per-cohort agents, actions, mean/max age,
  mean fitness, EAT%, action mix) whenever the action log has ≥2 cohorts. The
  log gained a trailing `cohort` column (`utils/data/agent_logger.py`).
- **UI marker on the followed agent.** The 3D web client floats a bobbing,
  spinning cone over the agent you follow, and the inspector shows its `cohort`.
- **Headless runner.** `scripts/competition_run.py` builds the config-driven
  world, runs N ticks with per-action logging + per-generation metrics, then
  writes `analysis.txt` (with the cohort section).
- **Published sample run** in `docs/sample_competition/` (4,000 ticks): founders
  `{v2-old:1, v3-new:7}` → `{v3-new:96, v2-old:4}`. The new architecture wins on
  mean fitness (10.57 vs 8.95) and max lifespan (493 vs 259). Includes
  `metrics.csv`, `analysis.txt`, and a 1-in-40 down-sampled action log.
- Note: cohorts must share an observation layout — v2/v3 are 72-dim
  (compatible); v3.5 is 78-dim (SIGNAL) and cannot share a world with v2/v3.
- Tests: `TestCohortComparison` in `tests/test_w5_society.py`.

## [Unreleased] — docs: external/Unity stream protocol

- **docs/UNITY_STREAM.md** (new): the wire contract for building an external
  frontend (Unity/Unreal/custom) on the live stream. Documents the transport
  (**SSE over HTTP**, not WebSocket — with the implications spelled out), the
  `/api/snapshot` and `/api/stream` endpoints, SSE framing, the full
  **snapshot/delta JSON schemas** (terrain grids, objects, agents, sky,
  burning, signals, feature/brain flags), base64 terrain-grid **decoding**,
  coordinate/render conventions, a copy-pasteable **Unity C# SSE client**, a
  poll-`/api/snapshot` fallback, an optional WebSocket-endpoint path, and
  schema **versioning**. Linked from README, USER_GUIDE, and web/README.

## [Unreleased] — docs: comprehensive testing guide

- **docs/TESTING.md** (new): documents the full **512-test** suite file-by-file
  — what each file verifies and its **pass criteria** — grouped by subsystem
  (brain/genome, world/terrain, objects/ecology, actions/rewards, learning,
  world-model/dreams, substrate, social/analyzers, frontend, infra, scenarios).
  Notes the one known-flaky stochastic test and how to add new tests. Linked
  from README, USER_GUIDE, and the proposal docs.

## [Unreleased] — perf (instanced objects), water, large biome world, user guide

- **Perf — no more lag as the world fills**: the web client now renders objects
  with **per-category InstancedMesh** (one draw call per category instead of
  thousands of meshes). Trees are a single merged geometry that scales with
  maturity, so they are clearly visible. Click-to-inspect works via instance
  ids; fire shows as flame particles at burning objects.
- **Water**: rendered as **one flat sea-level surface** (was stair-stepped
  per-column layers), and **WATER is now impassable** so agents stay on land
  instead of walking on water (`Tile.is_passable` excludes water + rock).
- **Default config = large biome world**: 160×160 heightmap with a richer biome
  mix (more sand for **deserts**), **day/night + weather**, **wildfire**, and a
  **small starting population (8) that breeds** (reproduction on). Tuned for a
  watchable, self-sustaining 3D world.
- **docs/USER_GUIDE.md** (new): task-oriented guide — which mode/brain/learner/
  reward to pick and why; world features (biomes, weather, fire, desertification);
  the **world model + planning**; **dream-based evolution**; the **3D viewer**;
  **exporting** the world (checkpoint / PLY / recording); and **Blender**
  round-tripping. Linked from the README.
- Tests: `test_tile_passable` updated for water-impassable; full suite green
  (510 + the known-flaky dream test which passes in isolation).

## 3D viewer: true-to-sim fidelity pass

The viewer is the **real simulation**, not a demo: `python -m render.server`
runs the world from `config/default.yaml` (same `World`, systems, and configured
brain as `main.py`); the read-only bridge now surfaces the full per-entity state
so every brain/world mechanic is visible.

- **Bridge** (`render/state_bridge.py`, read-only): agent view gains **age**,
  **last action** (the brain's most recent decision), **inventory** (count +
  has_food/has_seed); object view gains **value** (food freshness / plant
  maturity / seed viability) and **planted-by-agent**; snapshot reports
  **brain class + output size** (v2 / v3 / v3.5) and the **signal/transfer**
  feature flags.
- **Client** (`web/`): inspector now shows agent age/last-action/carrying and
  object freshness/maturity/viability; **plants scale with maturity** (saplings
  small → mature trees full); **rain** falls and **drought** hazes the sky
  (W1 weather, previously HUD-only); HUD shows the **brain architecture** and
  active social features.
- **Docs**: `web/README.md` gains a full **fidelity map** (every W0–W6 / brain
  mechanic → how it appears in 3D) and states plainly that the default is the
  real sim. What stays internal (neural weights, GRU/world-model latent) is
  called out.
- Tests: 3 new bridge assertions (agent full-state, object value, feature
  flags). Full suite green.

## 3D viewer: config-driven world, clickable, visual fixes

Addresses the Codespace report (dark surfaces, no trees, no day/night,
mountainous, agents dying, nothing clickable):

- **Config-driven world** (`render/sim_session.session_from_config`,
  `render.server --config`, default `config/default.yaml`): the viewer now
  builds the *configured* world — size, heightmap **biomes**, climate,
  population — instead of a fixed demo. Initial resources are populated
  (trees/berries/seeds) and agents spawn with **lifetime learning** and
  **reproduction on**, so they forage and breed instead of dying out (verified
  20 → 39 agents over 150 ticks). `--demo` keeps the old fixed scene.
- **Dark surfaces fixed**: the terrain material had `vertexColors: true` but a
  box has no vertex colors, so every tile rendered black. Removed it →
  per-instance biome colors show, and the W1 **day/night** lighting is visible.
- **Less mountainous**: voxel column height `MAX_H` 18 → 10 (gentler hills),
  and `config/default.yaml` now defaults `terrain.generator: heightmap`
  (biomes) with broader/smoother features (`feature_scale` 12→18,
  `persistence` 0.5→0.45).
- **Trees**: plants now render as an actual tree (trunk + foliage) instead of a
  small cone, so vegetation is visible.
- **Click-to-inspect everything**: raycasting now covers agents, objects
  (trees/berries — category/type/on-fire), and **terrain tiles** (terrain,
  elevation, fertility, moisture) via the InstancedMesh `instanceId`; the
  inspector panel is now generic.

## config default + 3D viewer fixes (earlier)

- **Default brain is now v3** (the attention brain) in `config/default.yaml`
  — was v2. Set `brain.version: 2` for the legacy baseline, `3.5` for the
  social brain. README/table/snippets updated to match.
- **3D viewer demo world is now populated and larger** (`render/sim_session.py`
  `build_demo_world`): it previously generated terrain but **no objects**, so
  the scene was empty (no trees/food) — fixed by scattering berry-plants
  ("trees", half pre-matured to bear fruit), berries, and seeds, with resource
  spawning on so food regenerates. Default size 64²→**96²**, agents 12→**24**,
  demo brain v2→**v3**. Server CLI defaults match.
- **3D viewer camera & lighting fixes**: the camera now frames the world by its
  size (was a fixed faraway position → the map looked tiny), an always-on
  ambient light plus higher hemisphere/sun floors keep the scene readable, and
  the night sky floor was lifted (deep night was rendering near-black). These
  address the "agents not moving / only night / map too small" reports — agents
  were in fact moving in the sim; the empty, dark, distant view hid it.

## 3D voxel frontend (Phases F0–F5)

Plan, decisions, and roadmap: `docs/FRONTEND_3D_PROPOSAL.md` · run guide:
`web/README.md`. A live Minecraft-style voxel view of the world, rendered in
the browser. The sim is unchanged — a **read-only** bridge streams its state.

- **F0 — state bridge** (`render/state_bridge.py`): `world_snapshot` (terrain
  packed as base64 byte-grids from the W2 elevation field, plus objects/agents/
  sky) + `StateTracker.delta` (per-tick moves/spawns/removals, pheromone cells,
  sky scalars). Read-only — never mutates the world. 8 tests.
- **F3a — live server** (`render/server.py`, `render/sim_session.py`):
  Server-Sent-Events over the stdlib HTTP server (no new deps), `/api/snapshot`
  + `/api/stream`; `SimSession` (snapshot/step) with demo-world and
  checkpoint-resume factories. `python -m render.server`. 5 tests.
- **F3b — web client** (`web/`): Three.js voxel renderer — chunk-free instanced
  terrain columns (blocky) with biome palette + water, per-lineage agents
  grounded on the surface and tweened between ticks, distinct per-category
  object models, pheromone glow, W1 day/night sky; orbit/free-fly/follow camera.
- **F1 — offline export** (`render/voxel_export.py`): snapshot → colored ASCII
  PLY (Blender/MeshLab) with exposed-face culling. `python -m render.voxel_export`.
  5 tests.
- **F4 — polish**: click-to-inspect agents, chase-follow camera, idle bob +
  smooth yaw, signal pulse, and lifecycle particle bursts (birth/death/consume)
  derived from deltas. Multi-client streaming verified.

18 new render tests (full suite 497). **Next:** F5 (replay scrubbing),
trade/fire particles (needs a read-only event channel), optional smooth terrain.

## World upgrade, Phases W0–W6 (complete)

Plan and rationale: `docs/WORLD_UPGRADE_PROPOSAL.md`.

> **▶ Open / next up (sim side):** **Brain v3.6 — kin-similarity sense
> (Observation v3).** The one piece deferred from W5 because it touches the
> genome (`nearest_agent_kin`, obs 78→79). Fully designed in
> `BRAIN_V3_PROPOSAL.md` §9; this is the next sim work item.

### Phase W6c — Substrate: reward-shaping diet + per-generation metrics CSV

**In simple terms:** the reward the learner sees is now a choosable "diet," and
a run can write a compact metrics CSV. The `minimal` diet is the world-side
analogue of fading the brain's instincts — it lets us ask whether the world's
own pressures (plus curiosity) are enough to learn from, instead of hand-built
dense rewards.

- **`reward.preset`** — `legacy` (default) keeps the full dense shaping tuned
  across the Brain v2/v3 work, **bit-identical** (its inline constants are
  untouched). `minimal` strips reward to **eat / death / energy-delta only**:
  a successful net-positive EAT, the death penalty, and a small per-step
  metabolism penalty — nothing else (no exploration/anti-loop/anti-spin/
  turn-toward-food terms).
- **`RewardConfig`** (`utils/agents/learning_utils.py`) holds the preset + the
  headline magnitudes (`eat_base`, `eat_energy_gain_coef`,
  `metabolism_penalty_coef`, `death_penalty`), behind a module-level active
  config (`get/set_active_reward_config`) that `main.py` sets from
  `config['reward']` — the same pattern as the observation active-spec.
- **Per-generation metrics CSV** — `utils/agents/metrics.py` `MetricsWriter`,
  wired as `--metrics-csv PATH`. One aggregate row per generation: population,
  food/plant/seed counts, mean energy/age, max age, mean fitness, and the soil
  fertility/moisture means. One O(agents) pass per generation.
- **Config**: new `reward:` block; new `--metrics-csv` flag.
- **Tests**: 11 new (`tests/test_w6c_reward_diet.py`) — `RewardConfig.from_dict`
  fallback, the minimal diet's four-term behaviour (eat rewarded, no-gain eat
  ignored, metabolism penalty, death penalty, no exploration/failed-eat terms),
  legacy-richer-than-minimal sanity, and the metrics writer. CLI smoke-tested
  (`minimal` + metrics CSV).

This completes **W6** (and the W0–W6 world-upgrade arc).

### Phase W6b — Substrate: full checkpointing (--save-state / --load-state)

**In simple terms:** a long run can now be stopped and resumed *exactly* where
it left off. Save a checkpoint; later, load it and the simulation continues on
a bit-identical trajectory (in serial mode) — the prerequisite for the
persistent-world track and for reproducible long experiments.

- **`world/checkpoint.py`** — `save_state(world, path, config=…)` and
  `load_state(path, config=…)`. Captured: world scalars + feature flags + the
  config; the tile grid (terrain/fertility/moisture/elevation/occupancy);
  every `WorldObject` and its components (plain Python — pickled directly); the
  pheromone field; the environment-engine state; each agent's genome, physical
  state, GRU hidden state, and anti-spin action-streak counters; the id
  counters; and **both** RNG streams (Python `random` and NumPy global).
- **Faithful resume** — brains/learners are rebuilt from the genome on load
  (no RNG cost); RNG state is restored **last**, after every agent is
  constructed (each draws one `np.random.randint` for facing), so the resumed
  stream is byte-identical. The anti-spin counters are captured because they
  drive the next action's energy cost — without them, energy diverged by ~0.2.
- **CLI** — `--save-state PATH` writes a checkpoint at the end of a headless
  run; `--load-state PATH` resumes from one (replacing the freshly-built world;
  that build is harmless because RNG is restored last).
- **Tests**: 4 new (`tests/test_checkpoint.py`) — save→continue vs
  save→load→run **bit-identical equality** (the acceptance criterion),
  tick/population restore, immediate-resume equality (RNG captured at the right
  moment), and version rejection. CLI save→resume smoke-tested.
- **Deferred (W6c)**: per-generation metrics CSV, reward-shaping config +
  `minimal` preset.

### Phase W6a — Substrate: spatial index for the nearest-food scans

**In simple terms:** the single biggest tick-rate sink was three "where's the
nearest food?" scans that walked up to 21×21 = 441 tiles each, several times
per agent per tick, mostly over empty tiles. W6a replaces those walks with a
coarse-cell index that only looks where food actually is — **~3.5× faster** on
the radius-10 reward scan — with **bit-identical results**.

- **`world/spatial_index.py`** — `SpatialIndex`, a coarse square-cell bucket
  (`cell_size` default 8) of edible-object ids by position, with
  add/remove/move/query_box. Tracks only edible objects (membership is stable
  while they sit in the world), so maintenance is a few hooks, not a per-tile
  mirror.
- **Maintenance** — hooked into `World.add_object` / `remove_object` /
  `move_object` and the two agent sites that move a berry between a tile and an
  inventory (`execute_pick_up`, `execute_drop`). All system removals already
  route through `remove_object`, so the index cannot go stale.
- **`World.nearest_edible(ax, ay, scan_r)`** — one helper consolidating the
  three duplicated scans (perception stimulus, RewardShaper distance +
  direction). Uses the index when present (visits only objects in overlapping
  cells) and an identical bounded tile scan otherwise. It is an *acceleration
  structure, not a source of truth*: every candidate is verified against live
  tile state and ties break row-major (min y, then x), so the nearest-food
  result is identical with the index on or off.
- **Call sites refactored** — `perception._encode_stimulus`,
  `RewardShaper._find_nearest_food_distance`,
  `RewardShaper._compute_food_dir_match` all delegate to `nearest_edible`.
- **Config**: new `performance:` block (`spatial_index: true`,
  `spatial_index_cell: 8`); wired through `World(performance_config=…)` and
  `main.py`.
- **Tests**: 11 new (`tests/test_spatial_index.py`) — index unit behaviour,
  **index-on == index-off across 20 random worlds** (the contract),
  row-major tie-break, maintenance through add/remove/move/pickup/drop, and an
  end-to-end run with the index on and off. Full suite: **469 tests**.
- **Deferred (W6b/c)**: full checkpointing (`--save-state/--load-state`),
  per-generation metrics CSV, reward-shaping config + `minimal` preset.

### Phase W5 — Social dynamics & open-endedness instruments

**In simple terms:** agents can now hand inventory to each other ("give"),
and the analyzer reports the division-of-labor metrics the project needs to
*see* whether anything social has emerged. No genome change — these are
capability + measurement, not new perception.

- **Trade via USE (opt-in).** `execute_use` now checks for a living agent on
  the tile in front *before* falling through to seed/fertilizer logic; with
  `social.transfer_enabled: true` it transfers the first inventory item to
  that agent (recipient must have space) and emits `interaction_kind="give"`.
  The action mask exposes USE whenever a trade is possible, so an agent
  carrying a non-plantable berry can still learn the give path. Gate stays
  off by default — bit-compatible with W4 runs.
- **🧬 SOCIETY / ROLES analyzer section.** A new
  `_compute_society_metrics` in `scripts/analyze_logs.py` derives four
  division-of-labor instruments from the existing log schema (no new
  columns needed):
  - **Role-entropy** — normalised Shannon entropy over agents' dominant
    actions (1.0 = roles spread evenly across the action vocabulary; 0 =
    one shared role). Also reports the role histogram (`EAT=4, WAIT=2 …`).
  - **Behavioural novelty** — mean pairwise Jensen-Shannon divergence
    (bits) between agents' action-frequency distributions, sample-capped
    at 64 agents for runtime. 0 = identical strategies; high = strategies
    diverged.
  - **Territory** — per-agent centroid, bbox area, visited-cell count and
    position entropy; aggregate mean bbox + mean Jaccard overlap over
    visited-cell sets (overlap-rate proxy without paying for a spatial
    index).
  - **Trade** — count + rate of `give` actions, distinct givers + recipient
    sites, give/signal ratio, plus a "blocked: recipient full" line when
    that path fires.
- **Config**: new top-level `social:` block (`transfer_enabled: false`),
  wired through `World(social_config=…)` and `main.py`. `social.transfer_enabled`
  is the only new switch; the analyzer metrics need no config.
- **Tests**: 13 new (`tests/test_w5_society.py`) — trade path (off,
  recipient-full, success, fall-through to plant), mask widens when only a
  trade is possible, and society metrics (role-entropy high/low, novelty
  zero/positive, territory bbox/overlap, give counting, missing-column
  guards). Full suite: **458 tests** green.
- **Deferred for the next batched genome bump (Brain v3.6 / Observation v3):**
  a `nearest_agent_kin` similarity sense. Append-only and migration-safe,
  but a break — queued instead of squeezed in mid-cycle.

### Phase W4 (part 2/2) — Brain v3.5: Observation v2 + SIGNAL (the genome break)

**In simple terms:** the batched genome break from part 1's plan is now built,
as **Brain v3.5** — the v3 attention brain with widened I/O so agents can live
in each other's world. Opt-in (`brain.version: 3.5`); the default world is
unchanged and bit-compatible.

- **Observation v2 (78-dim).** A six-feature EXTRA block is appended after the
  legacy 72 (so the prefix is identical): `time_of_day` sin/cos, tile
  temperature, nearest-agent proximity, nearest signal, on-hazard. The layout
  lives in `ObservationSpec` (`build_observation_spec(version=2)`,
  `OBSERVATION_SPEC_V2`) behind a module-level *active spec*
  (`set_observation_version`) that perception and the brain both read.
- **SIGNAL action + pheromone field.** `Action.SIGNAL` (id 8) deposits a value
  onto the agent's tile in a decaying (optionally diffusing) float field
  (`world.pheromones`); other agents sense it via the EXTRA block. Signals
  carry no built-in meaning — any protocol must emerge. Gated by
  `signal.enabled`; when off, SIGNAL is masked so an 8-action brain is
  untouched. `get_action_mask` now sizes by the brain's `output_size`.
- **Brain v3.5.** `BrainV3` resolves the active spec, derives `state_inputs`
  (28 under v2) and includes the EXTRA slice in the state path;
  `create_brain`/`calculate_weight_count` select v3.5 with `output_size` fixed
  at 9. **v3.5-base = 17,626 params** (v3 + 289, <2%).
- **Migration.** `adapt_loaded_genome` migrates a v3 genome into v3.5 on
  `--load-weights` via the part-1 `migrate_genome` (top-left copy): same
  original-action logits and value to floating-point tolerance; the EXTRA rows
  and SIGNAL column start at zero. The A2C-v3 and PPO-torch batched forwards
  were updated to include the EXTRA slice (without which v3.5 RL runs crashed).
- **Config**: `brain.version: 3.5` documented; new `signal:` block; `main.py`
  activates the v2 observation layout and migrates loaded weights for v3.5.
- **Tests**: 19 new (`tests/test_brain_v35.py`) — spec, perception features,
  weight count, migration bit-identity (v3→v3.5), SIGNAL masking, pheromone
  decay, end-to-end. A2C + PPO end-to-end runs verified. Full suite: 439.
- **Analyzer** — `scripts/analyze_logs.py` gained a **SOCIAL / SIGNAL** section:
  SIGNAL usage rate, **signal entropy** (how evenly signalling is shared across
  agents vs concentrated in specialists), and an **agent-proximity response**
  breakdown (action mix / SIGNAL rate bucketed by nearest-agent proximity, and
  mean proximity when signalling vs overall) — the W4 "signal entropy and
  agent-proximity response measurable" criterion. The world-model logger now
  sizes its `obs_*` columns *after* the v3.5 observation layout is active (so
  the EXTRA features are logged), and a pre-existing crash on int-coded action
  columns in transition logs was fixed. 6 tests (`tests/test_signal_analyzer.py`).

### Phase W4 (part 1/2) — Agents in each other's world + genome-migration tool

**In simple terms:** agents have been blind to each other and unable to
contest space — every social research direction was blocked behind that
(proposal P3). This first W4 increment lets agents *see* each other and
optionally *block* each other, and ships the genome-migration utility the
upcoming Observation-v2/SIGNAL break will use. Everything is opt-in; the
default world is unchanged.

- **`migrate_genome(old_flat, old_spec, new_spec)`** (`agents/brain/spec.py`):
  a generic top-left-corner copy across any **append-only** spec change.
  Because new observation features become extra rows at the end of the first
  weight matrix and a new action becomes an extra policy column, the old
  genome always sits in the new tensor's top-left corner; new rows/columns
  stay zero. Result: a migrated brain's logits for the original actions and
  its value are **bit-identical** to the old brain's (the W4 migration
  guarantee), verified for both v2 and v3 layouts.
- **Agents visible in vision** (`world.agents_visible`, default off): a tile
  holding another living agent reads as `(0.40, energy-ratio)` in the vision
  grid, overriding the terrain/object underneath. Self is excluded. This is
  the P3 unblock — agents can finally perceive each other.
- **Tile exclusivity** (`world.agent_collision`, default off): a living
  agent blocks a tile, so space itself becomes a contested resource.
- **Config**: `world.agents_visible` / `world.agent_collision` (wired via
  `main.py`).
- **Tests**: 10 new in `tests/test_agents_in_world.py` (migration top-left
  copy for v2 obs/action growth and v3, behavioural bit-identity,
  agents-in-vision enabled/disabled/self-excluded, collision). Full suite:
  420 passing.

**W4 part 2/2 (Brain v3.5) followed** — see the entry above; this part-1
increment shipped the migration tool and the agent-awareness toggles that
part 2 built on.

### Phase W3 — Ecology & hazards: toxicity, species, thorns, wildfire

**In simple terms:** the world had one food and one viable strategy. W3 adds
real ecological trade-offs and disturbances — multiple food species, a
poisonous look-alike you must learn to avoid, thorn hazards, and wildfire —
mostly as data on the W0 registry plus three small mechanics. The default
world is unchanged; the new content is opt-in or shipped as a loadable pack.

- **Toxicity wired into EAT.** The dormant `toxicity` field is now a physical
  consequence: net energy = `calories × freshness − toxicity × freshness ×
  30` (`TOXICITY_DAMAGE`). Poisonous food can cost more energy than it gives
  and can be fatal. Nothing labels a food good/bad — the agent discovers it
  from the energy/survival signal (`guideline.md` §8). EAT now records the
  eaten **species** in the action log, and the reward shaper only pays the
  survival bonus when the eat actually netted energy (poison is never
  rewarded — emergent discrimination, not a scripted rule).
- **Food species pack** — new `config/ecology.yaml` (load with `--objects`):
  a fast/cheap **shrub berry** (+10), a slow/rich **tree fruit** (+45), and a
  net-negative **nightshade** look-alike (the discrimination task), each with
  a distinct `vision_encoding`. Built with W0 `extends:` (a few lines each).
- **Contact hazards** — new `TileEffectSpec.contact_damage` field and a
  built-in **thorns** object that costs energy to step onto (a pressure, not
  a wall), applied through the normal movement energy/reward path.
- **Wildfire** — new `FireSystem` (opt-in `fire.enabled`). Plants on hot, dry
  tiles ignite (heat from the W1 `environment.temperature`, dryness from tile
  moisture), fire spreads to adjacent plants and burns them out (returning
  ash/fertility), and it **self-extinguishes at water/wet boundaries**. The
  dramatic, learnable disturbance the blunt calamity never was. Disabled =
  no-op. New `fire:` config block.
- **Analyzer** — `scripts/analyze_logs.py` gains a per-species consumption
  section (counts, share, mean net energy, toxic flag); `scripts/objects.py
  preview` shows each food's net energy and flags poisons.
- **Tests** — 14 new in `tests/test_ecology.py` (toxicity math, species
  logging, poison-not-rewarded, thorns contact damage, fire ignite/spread/
  self-extinguish/nutrient-return/disabled, and the ecology pack). Full
  suite: 410 passing. End-to-end run verified with fire + ecology enabled.
- *Deferred:* a dedicated invasive-species mechanic (the fast shrub already
  fills that niche) and a flood event.

### Phase W2 — Living terrain: heightmap, mountains, rivers, biomes (opt-in)

**In simple terms:** terrain used to be a uniform random shuffle — no
elevation, no rivers, no biomes; just scattered tiles. W2 adds an
elevation-first generator: a smooth height surface where the high ground
becomes mountains, water settles in basins and **flows downhill into
rivers**, and soil/sand fall out of a geography-driven moisture field with
**fertile river corridors**. It is opt-in (`terrain.generator: heightmap`);
the legacy flat generator stays the default and is unchanged.

- **New `world/terrain_generation.py`**: pure-NumPy fractal value-noise
  elevation (no new dependencies); mountains = the highest `rock_ratio` of
  tiles; lakes settle in the lowest basins; **rivers** are traced by
  steepest descent from the peaks into water/edges (within the water
  budget); a moisture field is derived from elevation + BFS distance to
  water; the driest land becomes desert **sand**; **fertility is highest in
  river corridors**. Fully deterministic for a given (width, height, seed).
- **Elevation is now a first-class tile field** (`tile.elevation`, [0,1]).
  The legacy generator leaves it at 0.0 (flat), so existing worlds are
  bit-compatible. It is **not** added to the agent observation yet — that
  genome break is reserved for W4 (Observation v2).
- **Slope movement cost**: moving uphill costs extra energy proportional to
  the elevation climbed (`SLOPE_CLIMB_COST`); flat terrain is unchanged, so
  movement is identical under the legacy generator.
- **Config** (`terrain:` in `config/default.yaml`): `generator:
  legacy|heightmap` plus a `heightmap:` block (`feature_scale`, `octaves`,
  `persistence`, `river_sources`). The existing rock/water/sand ratios are
  reused by the heightmap generator (honoured via elevation/moisture
  quantiles).
- **New `scripts/terrain.py preview`**: one-second ASCII preview of a seed's
  world (terrain or raw height field) before running a full simulation.
- **Tests**: 15 new in `tests/test_terrain_generation.py` (elevation range/
  smoothness/determinism, mountains-high/water-low, downhill rivers,
  spatial coherence, fertile corridors, World integration, legacy flatness,
  slope cost). End-to-end heightmap run verified through `main.py`. Full
  suite: 396 passing.
- **Deferred to a later W2 increment** (noted in the proposal): per-tick
  moisture diffusion, slow erosion, and 3×3 nutrient return. The current
  increment delivers the headline geography (mountains/rivers/biomes,
  elevation, slope cost) and meets the W2 acceptance criteria.

### Bug B5 fixed — runaway plant/food accumulation (plant carrying capacity)

**In simple terms:** mature plants spawn berries, berries decay into
seeds, seeds germinate into new plants — and nothing ever stopped a new
plant from sprouting next to existing ones. Each plant produced ~20
offspring over its life (a growth rate ~20× replacement), so plants — and
the berries they spawn — kept accumulating until they tiled the world. A
no-agent 100×100 world climbed past 2,600 objects at 8k ticks and was
still rising; a smaller world plateaued only at ~65% plant coverage.

- **Cause:** `SeedGerminationSystem` checked terrain, fertility, moisture,
  and a success probability — but never local crowding. There was no
  carrying capacity, so the lifecycle was a pure exponential bounded only
  by the one-plant-per-tile rule (i.e. world saturation).
- **Fix (a real pressure, not a script — `guideline.md` §8):**
  competition for space/light. A seed will not germinate if the
  surrounding window already holds `max_neighbor_plants` plants; it waits
  and eventually rots, exactly like the existing `blocks_growth` path.
  New helper `_count_plants_in_radius`.
- **Config** (`plants:` in `config/default.yaml`): `max_neighbor_plants: 3`
  and `neighbor_radius: 2` (a 5×5 window). With these defaults plant
  coverage plateaus flat at ~24% of plantable tiles (food stays abundant)
  instead of saturating to ~65% and climbing. `max_neighbor_plants: 0`
  restores the legacy unbounded behaviour for A/B comparison.
- **Threaded through** `World` → `WorldSystemManager` →
  `SeedGerminationSystem` and read in `main.py`.
- 5 new tests (`tests/test_systems.py::TestGerminationCarryingCapacity`),
  including a long-run regression asserting the population stays well
  below world saturation. Full suite: 381 passing.

### Phase W1 — Environment engine: day/night, seasons, weather (opt-in)

**In simple terms:** the world had a frozen climate — eternal noon, no
seasons, and "rain" that fell every tick forever. A new environment
engine gives it a real clock: days and nights, a slow seasonal
temperature wave, and rain/drought events. Plants, food, spoilage,
soil moisture, and agent metabolism all now respond to it. It is off
by default; switch it on with `environment.enabled: true`.

- **New `world/environment.py`** — `EnvironmentSystem`, the single
  source of climate. Each tick (before every other system) it computes:
  - **light**: sinusoid over `day_length` ticks, floored at `min_light`;
  - **temperature**: `base_temperature` + seasonal sinusoid
    (`season_length`, `season_temp_amplitude`) + a day/night offset;
  - **weather**: stochastic rain (`rain_start_chance`, `rain_duration`)
    and drought events (droughts suppress rain and multiply evaporation).
- **Existing systems consume plain multipliers** (all exactly 1.0 when
  disabled, keeping the legacy baseline bit-compatible):
  - plant growth × light × temperature comfort window
    (`temperature_response`: full rate in [0.3, 0.7], zero past 0.1/0.9);
  - germination × temperature window; food production × light;
  - freshness decay × (0.5 + temperature) — heat spoils, cold preserves;
  - agent metabolism × (1 + `metabolism_temp_coef` · 2·|temp − 0.5|) in
    **both** the serial and parallel agent paths.
- **Bug B1 fixed (soil moisture only ever rose).** The legacy soil model
  added a constant +0.0008/tick "rain" against −0.0002/tick evaporation,
  so every tile saturated at 1.0 and moisture constrained nothing.
  With the environment enabled, evaporation scales with temperature and
  light (×`drought_evaporation_factor` in droughts) and recovery arrives
  **only** during rain events or on tiles adjacent to water
  (`world.water_adjacent`, computed once). Moisture is now a real,
  non-monotonic constraint. (Legacy arithmetic is preserved verbatim
  when disabled.)
- **Bug B2 fixed (sand germination was impossible).** Sand clamped tile
  fertility/moisture to 0.05, strictly below the seed requirements
  (0.3/0.2), so its ×0.1 germination multiplier never even applied. The
  clamps now sit *at* the thresholds (0.30/0.20) so the multiplier is
  what makes sand harder — rare desert germination is possible again.
- **Config**: new documented `environment:` block in
  `config/default.yaml` (`enabled: false` by default).
- **Validation**: 34 new tests (`tests/test_environment.py`) covering the
  clock, the response curve, weather lifecycles, every multiplier,
  disabled-mode neutrality/legacy parity, the B1 fix empirically
  (moisture falls in dry spells, recovers in rain, non-monotonic over a
  cycle), and the B2 fix (a seed germinates on sand). A/B 2,000-tick
  runs (seed 42): population stays viable with the environment on
  (99 vs 100 agents alive) while scarcity becomes real
  (67 vs 246 food items, 66 vs 293 plants at the final tick).

### Phase W0 — Registry hardening & custom-object UX

**In simple terms:** defining a custom object used to fail silently — a
typo'd section name registered a useless object, a wrong field crashed
with a context-free error, and a definition took ~60 lines of copying.
Definitions are now validated with actionable errors, can inherit from
builtins with `extends:`, and ship with a one-second toolbox CLI.

- **New `world/object_validation.py`**: schema validation against the
  spec dataclasses (unknown sections/fields are errors with
  *did-you-mean* suggestions, all collected and reported together);
  cross-reference checking at load time (`grows_into`, `produces`,
  `decompose_into`, `spread_type_id` must name real types);
  `extends: <type_id>` deep-merge inheritance (spawn counts are never
  inherited implicitly); `vision_encoding: auto` allocates a free value
  inside per-category bands; collision warnings when two types are
  closer than 0.02 in encoding space (agents cannot tell them apart).
- **`SpawnSpec` gains `respawn_rate` / `max_count`** — custom foods can
  now replenish like builtin berries (`_respawn_registry_types` in the
  spawn system, terrain-filtered, capped).
- **New `scripts/objects.py`** — `validate` (schema + spawn dry-run with
  "will NEVER appear" warnings), `list` (table of all registered types),
  `preview` (plain-language explanation of how the simulation will treat
  each type).
- **New `docs/OBJECTS_GUIDE.md`** — 60-second authoring loop, the three
  rules that prevent silent failures, full schema reference, cookbook.
- **`config/custom_objects.yaml` rewritten** around `extends:` — the
  golden-apple example is now ~8 substantive lines (was ~60).
- **`main.py` refuses invalid object files at startup** with the full
  error report instead of running a broken world.
- 19 new tests (`tests/test_object_validation.py`).

## [Unreleased] — Brain v3, Phases 1–4 + Dream-Based Evolution

### Repository reorganization: docs/ and scripts/

**In simple terms:** the repo root was cluttered with ten documentation
files and three loose scripts. Documentation now lives in `docs/`
(README and CHANGELOG stay at the root, as is conventional), runnable
tools live in `scripts/`, and one redundant document was removed.

- **Moved to `docs/`**: BRAIN_V3_PROPOSAL, BRAIN_V2_V3_COMPARISON,
  ECOSYSTEM, PROJECT_OVERVIEW_TECHNICAL, SUGGESTIONS,
  WORLD_MODEL_LOGGING_FORMAT, guideline, CLI_GUIDE.
- **Moved to `scripts/`** (each with a repo-root path bootstrap so they
  run from any directory): `analyze_v3_1.py` → **`scripts/analyze_logs.py`**
  (renamed — the version-numbered name was stale),
  `analyze_observation_sensitivity.py`, `dream_evolve.py`.
- **Deleted** `PROJECT_OVERVIEW.md` — its non-technical narrative is fully
  covered by the README's What's-New/plain-words sections;
  `docs/PROJECT_OVERVIEW_TECHNICAL.md` remains for reviewer audiences.
- **`docs/CLI_GUIDE.md` rewritten**: complete `main.py` flag table
  (including `--mode`, `--world-model-log`, weight-loading caveats and
  the parallel-determinism note), a section per script with usage, and
  an updated troubleshooting/directory guide.
- All cross-references updated (README links, docs↔docs relative links,
  code comments, `config/default.yaml` pointers, `.flake8`
  per-file-ignores).
- **Fix** (`scripts/analyze_logs.py`): runs with zero object interactions
  produce all-NaN `object_type`/`interaction_kind` columns, which pandas
  reads as float64 — the string backfill then raised under pandas ≥ 3.
  Text columns are now coerced to string dtype before backfilling.

### Reconstructed `utils/data/agent_logger.py` (the `--log` pipeline)

**In simple terms:** one source file referenced everywhere — the CSV logger
behind the `--log` flag — had never actually been committed to the
repository, so `--log` crashed on a fresh clone and six test files could not
run. The module has been reconstructed from its complete usage surface and
committed; the whole test suite is now collectable.

**Root cause found:** `.gitignore` contained the unanchored pattern `data/`,
which matches *any* directory named `data` — including `utils/data/` — so
`git add` silently skipped the file. The pattern is now anchored to the
repository root (`/data/`).

**Technical detail:**

- `AgentLogger`: per-action CSV (`agent_actions_*.csv`) + per-tick agent
  state snapshots (`agent_states_*.csv`); persistent file handles with
  batched flushes (headers and state snapshots flushed immediately so files
  are readable mid-run); lock-guarded writes because `log_all_states` runs
  on the simulation's I/O pool while `log_action` runs on the main thread;
  accepts both the live agent dict (serial path) and snapshot lists
  (parallel path).
- `WorldModelLogger`: subclass of `AsyncWorldModelLogger` so the transition
  CSV schema keeps a single source of truth.
- `utils/data/__init__.py` restored to direct imports (the defensive
  import added earlier is no longer needed).
- Unblocked: `tests/test_logging.py`, `tests/test_world_model_logger.py`,
  `tests/test_exploration_sweep.py`, all three `tests/scenarios/` modules,
  and the previously failing `tests/test_improved_rewards.py` /
  `tests/test_parallel.py::test_logging_offloaded` — the full suite is now
  collectable and green.

### Dream-based evolution (capstone of the world-model stack)

**In simple terms:** the simulation can now train a model of the *world
itself* from its own logs and run evolution **inside that model** — thousands
of imagined episodes per second instead of full simulation. The loop is:
collect real experience → train the population world model → evolve genomes
in the dream → ground the champions back in the real environment (mandatory:
dream fitness is a proxy, and evolution will exploit the model's mistakes).

**Technical detail:**

- New `agents/dream.py`:
  - `PopulationWorldModel` — observation-space, policy-agnostic dynamics
    model `f(obs, onehot a) → (Δobs, r̂, p̂_done)` (two tanh layers + three
    heads; predicts observation *deltas* so a mostly-static world centres
    the targets at zero). MSE on Δobs/reward, BCE on done; save/load.
  - `load_transitions_csv` — loader for the `--world-model-log` schema.
  - `evaluate_in_dream` — roll a genome's policy in imagination from real
    seed observations; accumulated predicted reward = dream fitness.
  - `dream_evolution` — (μ+λ) loop with elites + Gaussian mutation, every
    evaluation imagined.
- New CLI `dream_evolve.py`: transitions CSV → trained model → dream
  generations → top-5 champions saved in the same `.npz` format
  `main.py --load-weights` consumes (the grounding step).
- **Fix:** `utils/data/__init__.py` imported the never-committed
  `agent_logger` module unconditionally, making the whole `utils.data`
  package unimportable — `--world-model-log` was broken on a fresh clone
  and 7 test modules could not even be collected. The import is now
  defensive; `tests/test_async_logger.py` (5 tests) is unblocked.
- **Fix:** `AsyncWorldModelLogger` wrote a hardcoded 64 `obs_*` columns
  while observations are 72-dimensional, misaligning every CSV column
  after `obs_63`. The header now follows `get_observation_size()`.
- Validated end-to-end: 400-tick logged run (9.5k transitions) → world
  model trained → 8 dream generations (mean dream fitness 31 → ~50) →
  champions grounded in the real environment (population remains viable).
- New tests: `tests/test_dream_evolution.py` (10 — CSV loader vs logger
  schema, header-width regression, model learns action-conditional toy
  dynamics, save/load, dream evaluation rewards what the model rewards,
  dream evolution beats random-genome baseline).

### Phase 4 — Learned world model: dynamics head, curiosity, latent planning (opt-in)

**In simple terms:** agents can now carry a tiny *imagination*: a learned model
that predicts "if I do this, what will I perceive next, and what reward will I
get?". Three things fall out of it. (1) **Curiosity** — when reality differs
from the prediction, the surprise itself becomes a reward, so agents explore
without any hand-coded exploration bonus. (2) **Planning** — before acting, an
agent can imagine a few action sequences in its head and pick the most
promising first move. (3) The model lives in the genome, so good imaginations
are inherited and evolved. Everything is off by default and works with both
brain versions.

**Technical detail:**

- **Latent dynamics head** (`brain.world_model.enabled`): appended to the
  genome layout (prefix unchanged → migration-friendly), for both brain
  versions: `d = tanh([h ‖ onehot(a)]·W1+b1)`, `ẑ' = d·Wz+bz`,
  `r̂ = d·Wr+br`. v2: +2,401 params (8,873 → 11,274); v3: +3,441
  (17,337 → 20,778). New `Brain.encode`, `Brain.predict_next_latent`,
  `Brain.has_world_model`.
- **PPO training** (`learning.ppo.world_model_coef`): auxiliary loss
  `‖ẑ_{t+1} − sg(z_{t+1})‖² + (r̂ − r)²` over sequence chunks, with
  stop-gradient latent targets (the policy/value losses anchor the
  representation; the dynamics head chases it — prevents latent collapse).
  Under a2c or pure neuroevolution the head evolves only.
- **Curiosity** (`agents/curiosity.py`, `learning.curiosity`): intrinsic
  reward = `weight · clip(zscore⁺(prediction error), 0, clip)` with Welford
  running statistics, warmup, and optional weight decay; only above-average
  surprise is rewarded. Attached via `Agent.enable_learning`; inherited by
  offspring.
- **Latent rollout planner** (`agents/planner.py`,
  `brain.world_model.planner`): random-shooting over imagined rollouts —
  predict ẑ', advance the GRU on the imagined latent, accumulate discounted
  r̂, bootstrap with the critic at the horizon; first action of the best
  rollout wins. First-step actions respect the action mask.
- **Decision/reward logic unified**: new `Agent.choose_action` and
  `Agent.compute_reward` are the single sources used by BOTH the serial
  path and `utils/parallel.py` — the duplication that caused the Phase-2
  drift bug is gone structurally.
- Config: documented `brain.world_model` + `learning.curiosity` blocks;
  startup banner shows world-model/planner status.
- New tests: `tests/test_world_model.py` (20 tests — spec counts and
  prefix stability, prediction shapes, curiosity warmup/normalisation/
  decay/clipping, planner mask-respect and preference for model-predicted
  reward, PPO gradient reach into dyn.*, prediction error decreasing on a
  learnable toy world, agent/offspring integration).

### Documentation cleanup

- Deleted `WORLD_MODEL_IMPLEMENTATION_GUIDE.md` — its plan is superseded by
  the implemented Phase 4 (latent-space model sharing the policy encoder,
  rather than the guide's standalone observation-space MLP). The surviving
  idea from it — a population-level offline model trained from transition
  logs for dream-based evolution — is tracked in BRAIN_V3_PROPOSAL.md §5
  and SUGGESTIONS.md §5.1.
- Deleted `todo.md` — a historical phase-completion checklist (Nov 2025)
  fully superseded by SUGGESTIONS.md (forward roadmap), CHANGELOG.md
  (history), and BRAIN_V3_PROPOSAL.md (phase status).
- Updated SUGGESTIONS.md checkboxes (attention, GRU size, value head,
  curiosity, GAE, PPO, world model items now DONE with pointers),
  PROJECT_OVERVIEW_TECHNICAL.md status (world model now trained, not just
  logged), ECOSYSTEM.md (status banner pointing to the current docs).

### Phase 3b — PPO learner: real lifetime learning (opt-in, `learning.algorithm: ppo`)

**In simple terms:** until now, "learning during a lifetime" only adjusted the
brain's final output layer — everything underneath (perception, memory) could
only change across generations by mutation. The new PPO learner trains the
**whole network** while the agent lives: it replays short episodes in order
(so memory is trained too), uses a better estimate of "was that action good?"
(GAE), and clips every update so a single bad batch can't wreck the policy.
Learned weights are still written back into the genome, so offspring inherit
them. The old learner remains the default for controlled comparisons.

**Technical detail:**

- New `agents/ppo.py` — `PPOSequenceLearner` + `TorchBrainMirror`:
  - Persistent torch parameter mirror per agent (one `nn.Parameter` per
    ParamSpec tensor) with a persistent Adam optimizer; functional forward
    re-expressed in torch for **both** brain versions, so autograd reaches
    encoder/attention, GRU (through time), and both heads.
  - **Sequence replay**: time-ordered chunks of `seq_len: 8` steps with the
    chunk-start hidden state; the GRU is re-run over each chunk during
    learning (truncated BPTT; stored-state strategy à la R2D2).
  - **GAE(λ)** advantages (`compute_gae`, default λ=0.95) with per-batch
    advantage normalisation; value targets R = Â + V.
  - **PPO clipped surrogate** (ε=0.2) against the recorded behaviour
    log-probs (`Brain.decide_with_logprob`), + value loss (0.5) and entropy
    bonus (0.01); gradient-norm clipping (0.5).
  - Lamarckian sync after every update via `ParamSpec.pack`.
- `Agent.enable_learning(algorithm="ppo", ppo_config=...)`; offspring inherit
  the algorithm; graceful fallback to a2c when torch is unavailable.
- Config: `learning.algorithm` + documented `learning.ppo` block
  (`learning_rate`, `seq_len`, `batch_size`, `gae_lambda`, `clip_epsilon`,
  `value_coef`, `entropy_coef`, `epochs`, `grad_clip`, `chunk_buffer`).
- **Fix: the parallel agent-update path (`utils/parallel.py`, the default
  `simulation.parallel: true` pipeline) had drifted from `Agent.update`** —
  it still contained the removed auto-eat override and non-fading instinct
  calls, so Phase 2 behaviour was not active in default headless runs (the
  serial path and the test suite were correct). The parallel path now mirrors
  `Agent.update` exactly (fading instincts, no auto-eat, PPO step storage and
  terminal handling). Refreshed baselines with fading genuinely active:
  RL a2c seed 42: 64 alive @1000; neuroevolution: 16–33 across seeds — harder,
  viable, no extinctions.
- New docs: **`BRAIN_V2_V3_COMPARISON.md`** — full math-level comparison of
  both architectures (GRU gates, attention scaling, parameter/FLOP budgets)
  and both learners (policy-gradient algebra, GAE derivation, clipped
  surrogate), written for a CS-student audience.
- New tests: `tests/test_ppo_learner.py` (14 tests — GAE vs hand-computed
  values, chunking/padding/terminal handling, full-network gradient reach for
  v2 AND v3, Lamarckian sync, clip-bounds-off-policy, agent/offspring
  integration, no-torch fallback).

### Phase 3 — Attention brain architecture (opt-in, `brain.version: 3`)

**In simple terms:** there is now a second, smarter brain design you can switch
on in the config. Instead of looking at all 25 vision tiles with equal, fixed
importance, the new brain *attends*: a tiny attention mechanism, steered by the
agent's internal state (hunger, inventory, …), decides which tiles matter right
now. Its memory is larger, and its value estimate ("how good is my situation?")
can see the current situation directly instead of only remembered context. The
old brain remains the default, so the two can be compared scientifically under
identical conditions.

**Technical detail:**

- New `agents/brain/v3.py` — `BrainV3(Brain)`, ≈17,337 parameters
  (vs ≈8,873 for v2):
  - **Shared tile embedding**: each vision tile becomes a token
    `[type, value, pos_row, pos_col]` embedded by one 4×8 matrix shared
    across all tiles (position-equivariant; scales to larger vision radii
    with the same weights). Fixed positional encoding, not learned.
  - **Single-head attention pool**: query derived from the state encoding
    (22 non-vision features → 40), keys/values from tile embeddings;
    softmax(k·q/√E) pools the tiles.
  - **Latent z = [state 40 | pooled vision 8]** feeds a **GRU(48)**.
  - **Value MLP reads [z, h]** (96 → 16 → 1) — the critic gets a direct
    view of the present state ("separate value head" roadmap item).
- `Brain` was refactored into overridable `_encode` / `_value` /
  `_build_nested` template methods; v2 behaviour is unchanged (verified by
  the existing equivalence tests).
- New `Brain.rebind(genome)` re-binds parameter views after genome
  replacement; `clone_agent`, `inherit_knowledge`, and pretrained-weight
  loading now use it (architecture-agnostic, replaces hand-rolled
  reconstruction).
- Factory `agents.brain.create_brain(genome, brain_config, instincts)` and
  `calculate_weight_count_for_config(brain_config)` map the YAML `brain`
  section to the right class; `Agent.brain_config` (class-level, set in
  `main.py`) makes offspring inherit the architecture.
- Learner support: `AgentLearner.learn` routes version-3 brains to a
  dedicated NumPy batch path (attention forward + policy-head and
  value-MLP-output updates — same heads-only update depth as v2; the
  full-backprop learning upgrade is Phase 3b).
- Config: `brain.version` (default **2**) and a `brain.v3` size block
  (`embed_dim`, `state_dim`, `gru_hidden_size`, `value_hidden`).
- Validation: 1000-tick headless v3 runs — RL seed 42: 100 alive;
  neuroevolution seed 42: 48 alive (vs 100 for v2 — the expected
  capacity-vs-evolvability trade-off of mutating 2× the parameters, and
  the reason v2 remains the default baseline).
- New tests: `tests/test_brain_v3.py` (15 tests — spec count, factory
  dispatch, masking, attention/positional sensitivity, rebind, learner
  updates + Lamarckian sync, evolution integration).

### Phase 2 — Fading instincts & auto-eat removal (emergence integrity)

**In simple terms:** survival "training wheels" now come off as agents grow up.
Newborns are still nudged toward picking up food, eating when hungry, and turning
toward food — but the nudges weaken every tick and disappear entirely at age 150
(configurable). The old hidden rule that *forced* hungry agents to eat was removed.
Result: every behaviour you observe in an adult agent is genuinely produced by its
evolved/learned network, which is what makes the project's "emergence-first" claim
honest and testable.

**Technical detail:**

- `agents/brain/instincts.py`: `InstinctModule.strength_at(age)` now drives a
  linear fade (1.0 at birth → 0.0 at `fade_age`, default 150 ticks). All instinct
  biases — PICK_UP +1.5, EAT +1.0, USE +0.5, turn-toward-food +0.8×proximity —
  are multiplied by this strength.
- **Auto-eat override removed** (`agents/agent.py`): the hardcoded
  "force EAT below 50% energy while holding food" rule is gone. Replaced by a
  hunger-scaled EAT prior inside the instinct module:
  `+hunger_eat_bias × energy_urgency × fade_strength` (default bias 3.0). The
  policy can override it, and it fades with age like every other instinct.
- New config section `brain.instincts` (`enabled`, `fade_age`, `hunger_eat_bias`)
  in `config/default.yaml` and `config/training_easy.yaml`; wired class-level via
  `Agent.instinct_config` in `main.py` so offspring inherit the configuration.
  Setting `fade_age: null` reproduces the legacy (never-fading) behaviour;
  `enabled: false` gives a pure network from birth — both useful as ablations.
- Instinct modules are preserved across all brain rebuilds (offspring cloning in
  `agents/evolution.py`, knowledge inheritance in `agents/agent.py`, pretrained
  weight loading in `utils/agents/learning_utils.py`).
- Validation: 1000-tick headless A/B runs remain population-stable in both
  evolution modes (RL seed 42: 94 alive vs 100 before; neuroevolution seed 42:
  100 vs 92; harsh seed 7 RL: 23 with fade vs 30 without — a modest, intended
  increase in selection pressure, no collapse).
- New tests: `tests/test_instinct_fading.py` (config plumbing, hunger bias
  scaling/gating, fade-through-agent path, adult policy == pure network,
  instinct survival across clone/inherit, end-to-end "hungry agent still eats").

### Phase 1 — Spec-driven brain refactor (enabling groundwork)

**In simple terms:** the brain's wiring diagram used to be written out by hand in
three different files that all had to agree byte-for-byte; now it is declared once
and everything else is derived from it. This makes upcoming architecture upgrades
(bigger memory, attention, world-model heads) one-line changes instead of
error-prone three-file surgery. No behaviour changed in this phase.

**Technical detail:**

- `agents/brain.py` became the `agents/brain/` package:
  - `spec.py` — `ParamSpec` (declarative genome layout: weight counting,
    zero-copy unpacking, packing, version tag for future genome migration) and
    `ObservationSpec` (named observation fields replacing magic indices).
  - `modules.py` — pure neural functions (sigmoid, softmax, GRU step).
  - `instincts.py` — instinct biases extracted from `Brain.forward`; the network
    is now a pure function of (observation, hidden state, parameters).
- `AgentLearner._sync_genome_weights` (Lamarckian inheritance) reduced from ~50
  hand-maintained lines to a single `ParamSpec.pack` call.
- `utils/agents/brain_utils.py` kept as a thin backward-compatibility shim.
- Fixed NumPy ≥ 2.0 incompatibility in the value head
  (`float()` on a shape-`(1,)` array), which broke 8 tests on fresh installs.
- New tests: `tests/test_brain_spec.py`, including byte-for-byte layout
  equivalence against a frozen copy of the legacy unpacker.

### Earlier

#### Added
- `agents/agent.py`: Improved agent lifecycle and GRU hidden-state integration.
- `tests/test_evolution.py`: New tests covering mating, selection, and lineage tracking.

#### Changed
- `agents/learning.py`: Refactored actor-critic update loop for numerical stability and batch handling.

#### Notes
Consolidated migration to Brain v2 (GRU + Actor-Critic), reorganized utilities into
`utils/agents/`, and updated tests to match new APIs.
