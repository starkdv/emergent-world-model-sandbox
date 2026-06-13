# Changelog

## [Unreleased] — World upgrade, Phases W0–W4

Plan and rationale: `docs/WORLD_UPGRADE_PROPOSAL.md`.

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

**Staged next (W4 part 2/2) — now scoped as Brain v3.5.** The single batched
genome break — Observation-v2 feature block (`time_of_day` sin/cos, tile
temperature, nearest-agent proximity/signal, on-hazard) + the **SIGNAL**
action and pheromone field (`output_size` 8→9) — is a minor bump of the v3
attention brain with widened I/O. Designed in full (architecture diagram,
genome/param deltas, migration, implementation checklist) in
`docs/BRAIN_V3_PROPOSAL.md` §8. Deferred deliberately: the action-count change
ripples through the action mask, instincts, PPO replay, the dream model, and
the world-model logger's one-hot, and warrants its own carefully validated
increment on top of the migration tool landed here.

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
