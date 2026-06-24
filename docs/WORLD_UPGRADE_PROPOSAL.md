# World Upgrade — Architecture Proposal

**Status:** In progress — W0 ✅ and W1 ✅ implemented (June 2026, see
CHANGELOG); W2 partially implemented (heightmap/rivers/biomes/slope shipped;
moisture diffusion + erosion deferred); W3 partially implemented (toxicity,
species pack, thorns, wildfire shipped; invasive-species + flood deferred);
W4 implemented as Brain v3.5 (agents-in-vision, tile collision, genome
migration, Observation-v2 senses, SIGNAL + pheromone field); W5–W6 remain
proposals
**Branch:** `claude/world-upgrade`
**Scope:** `world/` (tiles, systems, world, object registry), `utils/agents/`
(perception, reward shaping, action execution), `config/default.yaml`
**Inputs reviewed:** full `world/` code inventory, `docs/SUGGESTIONS.md`
Parts 1/3/6 (environment, ecology, physics), `docs/guideline.md` §8
(emergence-first), the Brain v3 stack this world must now feed

---

## 1. Why upgrade — what the world actually is today

The brain got a generational upgrade (attention, full-network learning,
world models, dream evolution); the world it lives in did not. Today's
world, precisely:

- **Terrain**: 4 tile types (SOIL/ROCK/WATER/SAND) placed by **uniform
  random shuffle** — no spatial coherence, no biomes, no rivers. Only ROCK
  blocks movement; edges are walls.
- **One food chain**: berry_plant → berry (20 cal) → berry_seed → plant.
  One species, one nutrition value. `EdibleComponent.toxicity` exists but
  is **wired to nothing**.
- **Static climate**: moisture recovers at a constant 0.0008/tick
  ("simulates rain" — every tick, everywhere, forever). No day/night, no
  weather, no seasons, no temperature. The only environmental dynamics are
  sand spread (one tile-effect object) and a blunt calamity (delete 70% of
  resources every 500 ticks).
- **No inter-agent world**: agents can stack on one tile, cannot sense
  each other (not even in vision!), cannot signal, trade, or fight. The
  world is effectively N independent single-agent worlds sharing a food
  budget.
- **Strong machinery already in place**: a clean ECS object registry that
  builds objects (edible/seed/plant/fertilizer/tile-effect specs) from
  YAML; a 7-system tick pipeline; per-tick caches; a learning scheduler;
  row-parallel soil updates.

### The four structural problems

**P1 — The world is too static for the brain we built.** Curiosity rewards
*surprise* and the dynamics head learns *change* — but outside of agent
actions, almost nothing in this world changes in a learnable way. A
non-stationary "real selection pressure" environment is the project's own
stated methodology (technical overview §3); today that pressure is one
periodic mass-deletion event.

**P2 — One niche ⇒ one strategy.** With a single food source, uniform
terrain, and no agent-agent channel, there is exactly one viable strategy
(forage berries). Open-endedness, division of labor, and behavioral
diversity — the §5.2 research agenda — have nothing to differentiate on.

**P3 — Agents are invisible to each other.** The 5×5 vision encodes tiles
and objects but **not other agents**. Every social research direction
(communication, kin selection, territory, cooperation) is blocked behind
this single gap.

**P4 — Hidden scaling and engineering debt.** Perception scans an 11×11
window per agent per tick and the RewardShaper scans **21×21 per action**
(O(441) tile loops × population × ticks). Long experiments can't be
checkpointed (only weights are saved). And the RewardShaper has grown into
~35 hand-crafted terms (food-distance shaping, spin penalties, loop
penalties...) — a quiet tension with the emergence-first claim that
fading instincts just fixed on the brain side.

### Verified dynamics bugs (measured, not suspected)

| # | Bug | Evidence | Consequence |
|---|---|---|---|
| B1 ✅ fixed (W1) | **Moisture never decreases.** Recovery (+0.0008/tick, every soil tile, unconditional) exceeds evaporation (−0.0002) plus even max plant draw (−0.0005): net **+0.0006 empty / +0.0001 planted** — monotonic. | Measured: avg moisture 0.51 → 0.95 over 900 ticks; 57.5% of tiles fully saturated and climbing | The moisture dimension is functionally dead: every germination check passes after ~1.5k ticks, and the observation channel carries no information |
| B2 ✅ fixed (W1) | **Germination on sand is impossible, not "10× harder".** Sand clamps moisture/fertility to 0.05, but germination requires moisture ≥ 0.2 and fertility ≥ 0.3 — the check fails before the 0.1 multiplier is ever consulted. | `TileEffectSpec` overrides vs `SeedGerminationSystem` thresholds | Sand's germination_multiplier is dead config; sand reclamation (which needs a plant ON sand) can only occur when sand spreads under an existing plant |
| B3 | **Water is cosmetic.** Water tiles force their own moisture to 1.0 and block planting — nothing else. No drinking/thirst, no crossing cost, no moisture sharing with neighbours. | inventory §1/§6 | "Water" currently means "tile you can't plant on" |
| ~~B4~~ | ~~Inventory is a stasis field~~ — **RETRACTED**: verified false. `DecaySystem` iterates `world.objects`, which includes carried items (pickup only removes the tile link), so carried food DOES spoil (measured: freshness 1.0 → 0.5 in 50 ticks while held). | empirical test | No fix needed |
| B5 ✅ fixed | **Runaway plant/food accumulation (no carrying capacity).** Each mature plant drops ~40 berries over its life; 70% decompose into seeds that germinate at 75% — ~20 offspring per plant, with no crowding check anywhere. | Measured (no agents): a 100×100 world climbs past 2,600 objects at 8k ticks and is still rising; a small world plateaus only at ~65% plant coverage | Plants and the berries they spawn tile the world; food count is meaningless and the world saturates |

These are first-class fix targets: **W1's weather model is the B1/B3 fix**
(evaporation scaled by temperature/light *exceeding* base recovery, recovery
arriving via rain events and water-adjacency diffusion — moisture becomes a
real, spatially structured constraint), **W2's elevation/water rework
addresses B3**, B2 is corrected by aligning sand's clamps with germination
thresholds (or vice versa) so the multiplier actually expresses "harder",
and B4 turned out to already behave correctly (retracted above). **B5 is
fixed by a density-dependent germination cap** (competition for
space/light): a seed will not establish where its neighbourhood already
holds `max_neighbor_plants` plants, giving the ecology a real carrying
capacity instead of unbounded exponential growth.

---

## 2. Design principles

1. **Pressures, not scripts** (`guideline.md` §8). Every addition is a new
   *physics* or *resource economics*; never a scripted behavior, never a
   new hand-crafted reward term. Agents must discover what night, poison,
   or neighbors mean.
2. **One genome break, maximum.** New senses extend the 72-dim observation
   → input-layer sizes → genome length. Therefore: world dynamics that
   agents can perceive through *existing* channels (vision encodings,
   food availability, energy) ship freely; everything needing new senses
   is batched into a **single Observation-v2 change** with a
   spec-versioned, behavior-preserving migration (new encoder input rows
   zero-initialized — old genomes act identically until evolution/learning
   uses the new inputs).
3. **Registry-first.** The YAML object registry is the cheapest extension
   surface in the codebase — new species, foods, and hazards should be
   data, not code, wherever possible.
4. **One modulation point.** Day/night, seasons, and weather are all
   "global scalar fields that multiply existing rates". One
   `EnvironmentSystem` computes `light`, `temperature`, `rain` per tick;
   existing systems consume multipliers. No scattered special cases.
5. **Everything config-gated, everything A/B-able** — same discipline as
   the brain phases: each feature is an ablation switch, defaults chosen
   so the current baseline remains reproducible (`environment.enabled:
   false` reproduces today's world exactly).

---

## 3. Proposed architecture

```
                       EnvironmentSystem (NEW — runs first each tick)
                       light(t)  ·  temperature(tile,t)  ·  rain(t)  ·  season(t)
                                          │ multipliers
        ┌─────────────┬─────────────┬─────┴───────┬──────────────┬────────────┐
        ▼             ▼             ▼             ▼              ▼            ▼
  PlantGrowth   Germination     Decay      ResourceSpawn   SoilDynamics   agent
  (× light ×    (× temp        (× temp/                    (rain replaces  metabolism
   temp curve)    window)        moisture)                  constant       (× temp)
                                                            recovery)
        ▼
  TileEffect engine (EXTENDED): contact_damage, flammability → FireSystem
        ▼
  ObjectRegistry (EXTENDED data): plant/food species, toxins, hazards
        ▼
  Agent layer: agents visible in vision · SIGNAL channel · Observation v2
```

Key decisions, with rationale:

1. **`EnvironmentSystem` as the single source of climate** (SUGGESTIONS
   §6.1). World gains `time_of_day`, `season`, and a per-tile
   `temperature` derived cheaply (base + season + water/rock proximity
   computed once at generation, modulated globally per tick — not a
   per-tile simulation). Rain becomes *events* (the current constant
   moisture drip becomes the no-weather fallback), droughts are the
   inverse. Calamities become one member of a general **event system**
   (drought, flood, blight) instead of a special case.

2. **Coherent terrain generation** (SUGGESTIONS §3.1 biomes). Replace the
   uniform shuffle with smoothed value-noise (pure NumPy, no new deps):
   terrain clusters into forest (fertile/moist), plains, wetland, desert
   (sand), rocky ridges; water generates as connected bodies so moisture
   diffusion (below) creates *river corridors of fertility*. Same config
   ratios, spatially coherent layout. `terrain.generator: legacy|biomes`.

3. **Soil becomes a transport medium** (SUGGESTIONS §6.2): water-adjacent
   moisture diffusion (gradient flow), 3×3 nutrient return on death
   (fertile patches where things die), decay rate coupled to temperature
   and moisture. All effects ride existing tile properties — zero
   observation change, but the world develops *geography agents can
   exploit*.

4. **Ecology via the registry** (SUGGESTIONS §6.3): 2–3 plant/food species
   as YAML data — e.g. fast/low-calorie shrub vs slow/high-calorie tree
   fruit vs a wetland reed — with distinct `vision_encoding`s so v3's
   attention can tell them apart. **Wire the dormant `toxicity` field
   into EAT** (energy loss proportional to toxicity × freshness):
   poisonous food = the first discrimination task. An invasive species
   and contact-hazard objects (thorns: `contact_damage` on the
   tile-effect spec) are then pure data. Fire (`FireSystem`) consumes
   the W1 temperature/moisture fields: plants on hot dry tiles ignite,
   spread, return nutrients — the dramatic, learnable disturbance the
   blunt calamity never was.

5. **Agents enter each other's world.** Three sub-steps in strictly
   increasing invasiveness:
   a. **Agents visible in vision** (a vision encoding for "agent here") —
      this is the P3 unblock and the only one that must come first;
   b. **Tile exclusivity option** (`world.agent_collision: true`) so
      space itself becomes contested;
   c. **SIGNAL** — a 9th action emitting a 1-float value that decays on
      the tile (a pheromone field), sensed via Observation v2. Signals
      have *no built-in meaning* — emergence-first. Markers (§3.2) come
      free: DROP a zero-cost marker object, already supported by the
      registry.

6. **Observation v2 — the single batched genome break.** One new stimulus
   block appended to the observation: `time_of_day (sin, cos)`,
   `tile_temperature`, `nearest_agent_proximity`, `nearest_agent_signal`,
   `on_hazard`. Appended *after* the current 72 (layout prefix unchanged,
   exactly like the dynamics head), `ObservationSpec` versioned,
   encoder input rows for new features zero-initialized on migration so
   **existing genomes keep their behavior bit-for-bit** until mutation/
   learning touches the new rows. SIGNAL extends `output_size` 8→9 under
   the same migration (new policy column zero-init = never chosen until
   learned/evolved).

7. **Reward-shaping diet** (the P4 integrity item, SUGGESTIONS Part 1).
   Move every shaping magnitude into config; define a `reward.preset:
   minimal` (eat/death/energy-delta only) alongside `legacy`. The new
   world features get **no new shaping terms at all** — curiosity and the
   environment itself are the exploration signal now. This is the world-
   side completion of what fading instincts did for the brain.

8. **Substrate** (SUGGESTIONS §4.2): a tile-bucket **spatial index**
   maintained by add/remove_object — replaces the 21×21 and 11×11 scans
   in RewardShaper/perception with O(objects nearby) lookups (the single
   biggest speedup available, and it makes large worlds viable);
   **checkpointing** (world + agents + RNG → resumable long runs — also
   the prerequisite for the persistent-world track); per-generation
   metrics CSV.

---

## 4. What deliberately does NOT change

- The 8 primitive actions keep their semantics (SIGNAL is additive, W4).
- The brain package, learners, world model, dream evolution — untouched;
  they consume whatever the world produces.
- The existing transition-log schema (columns are appended, never
  reordered — same rule as the genome).
- Emergence-first: no pathfinding, no scripted "drink"/"flee" behaviors,
  no kin-bonus reward by default (group fitness bonus from §3.3 is
  implemented only as an explicitly-labeled ablation flag, off).

---

## 5. Phased delivery plan

Each phase independently shippable, config-gated, tests + A/B survival
runs before merge — the same discipline as Brain v3 Phases 1–4.

**W0 — Registry hardening & custom-object UX (prerequisite for W3).
✅ DONE (June 2026 — see CHANGELOG "Phase W0").**
Schema validation with named errors, cross-reference checking, `extends:`
inheritance, the object toolbox CLI, respawn spec (§8–§9 below).
*Acceptance:* every failure mode in §8 produces a clear, actionable error
at load time; the golden-apple example shrinks to ≤10 lines via `extends`.

**W1 — Environment engine (no observation change).
✅ DONE (June 2026 — see CHANGELOG "Phase W1"; includes the B1 and B2
fixes; calamity generalization deferred — calamity remains its own
system for now).**
`EnvironmentSystem` (day/night, seasons, rain/drought events, derived
temperature), multipliers consumed by growth/germination/spawn/decay/soil/
metabolism; calamity generalized into the event system.
*Acceptance:* `environment.enabled: false` is bit-compatible with today;
enabled defaults keep populations viable over 2,000-tick runs; each cycle
visible in the world-state logs.

**W2 — Living terrain (elevation becomes first-class).
🟡 PARTIALLY DONE (June 2026 — see CHANGELOG "Phase W2").**
Heightmap generator (smoothed noise): mountains = high rock, water settles
and **flows downhill into actual rivers**, biomes derived from
elevation × moisture (forest/plains/wetland/desert); slope movement cost;
moisture diffusion, 3×3 nutrient return, temperature/moisture-coupled
decay, slow erosion. This delivers rivers/mountains/real biomes *within*
the lab world (a heightmap is exactly how game terrain works) and the
elevation field is the bridge asset for the 3D track (§13) — the GPU
isometric renderer is already built to display it.
*Acceptance:* rivers connect high→low and create fertility corridors;
slopes measurably shape movement; legacy flat generator still selectable.

> **Shipped (W2, first increment):** `world/terrain_generation.py`
> (pure-NumPy value-noise elevation; mountains from the elevation quantile;
> lakes in basins + downhill steepest-descent rivers within the water
> budget; moisture from elevation + distance-to-water; desert sand from the
> driest land; **fertile river corridors**), `tile.elevation` as a
> first-class field (legacy generator stays flat at 0.0 → bit-compatible),
> slope-based movement energy cost, `terrain.generator: legacy|heightmap`
> config, the `scripts/terrain.py preview` ASCII tool, and 15 tests. All
> three acceptance criteria are met.
>
> **Deferred to a later W2 increment:** per-tick moisture *diffusion*, slow
> *erosion*, and 3×3 nutrient return on death (the current nutrient return
> stays single-tile). Elevation is deliberately **not** in the observation
> vector yet — that is the W4 genome break.

**W3 — Ecology & hazards (mostly data).
🟡 PARTIALLY DONE (June 2026 — see CHANGELOG "Phase W3").**
2–3 plant/food species (YAML), toxicity wired into EAT, invasive species,
thorn hazards (`contact_damage`), `FireSystem`, flood event.
*Acceptance:* analyzer shows per-species consumption; agents' species
preference is measurable (the first discrimination result); fire spreads
and self-extinguishes at water/moisture boundaries.

> **Shipped (W3):** toxicity wired into EAT (net = calories − toxicity ×
> freshness × 30) with species recorded per eat and the reward shaper gated
> on realised energy gain; a 3-species food pack (`config/ecology.yaml`:
> shrub berry / tree fruit / toxic nightshade) via W0 `extends`;
> `TileEffectSpec.contact_damage` + a built-in `thorns` hazard; `FireSystem`
> (opt-in `fire.enabled`) consuming the W1 climate, spreading and
> self-extinguishing at wet boundaries; a per-species consumption section in
> `scripts/analyze_logs.py`; 14 tests. All three acceptance criteria met.
>
> **Deferred:** a dedicated invasive-species mechanic (the fast shrub fills
> that niche through reproduction) and the flood event.

**W4 — Agents in the world + Observation v2 (the one genome break).
✅ DONE (June 2026 — see CHANGELOG "Phase W4"; implemented as Brain v3.5).**
Agents visible in vision; optional tile exclusivity; pheromone field +
SIGNAL action; Observation v2 block + `output_size` 9 with zero-init
spec migration (old populations load and behave identically).
*Acceptance:* migration test proves bit-identical pre/post behavior for
old genomes; signal entropy and agent-proximity response measurable in
new analyzer sections.

> **Shipped (W4 part 1):** the genome-migration tool `migrate_genome`
> (generic top-left copy), **agents visible in vision** (`world.agents_visible`,
> the P3 unblock), and **tile exclusivity** (`world.agent_collision`).
>
> **Shipped (W4 part 2) — Brain v3.5.** The batched **Observation v2** block
> (six social/climate senses, 78-dim) + the **SIGNAL** action and decaying
> pheromone field (`output_size` 9), selected by `brain.version: 3.5` and
> `signal.enabled`. A v3 genome auto-migrates into v3.5 with the same
> original-action behaviour to float tolerance. Full design + as-built notes:
> **`BRAIN_V3_PROPOSAL.md` §8 (Brain v3.5)**. Both acceptance criteria are met
> (migration identity; per-agent senses/signal are observable — a dedicated
> signal-entropy analyzer view is a small follow-up).

**W5 — Social dynamics & open-endedness instruments.
✅ DONE (June 2026 — see CHANGELOG "Phase W5").**
Inventory transfer ("give" via USE on a facing agent, ablatable through
`social.transfer_enabled`), and a new analyzer **🧬 SOCIETY / ROLES** section
that prints role-entropy (division of labor), pairwise behavioural novelty
(mean Jensen-Shannon divergence), per-agent territory (bounding-box area,
visited cells, mean Jaccard overlap), and trade counts (give actions, givers,
recipient sites, give/signal ratio). 13 new tests.
*Acceptance:* division-of-labor metrics computable on long runs (`role_entropy_norm`,
`novelty_mean_js`, `mean_territory_overlap` print on every run with ≥2 agents);
all features individually ablatable (trade is its own config flag; metrics
require no flag and are derived from the existing log schema).

> **Deferred for a separate genome bump:** the kin-similarity observation
> feature. Adding it would require an Observation v3 break — append-only and
> migration-safe like v3 → v3.5, but a break nonetheless — so it stays
> queued until the next batched genome refresh (**Brain v3.6**) instead of
> being squeezed in mid-cycle. The full design is on record:
> **`BRAIN_V3_PROPOSAL.md` §9 (Brain v3.6 — the kin-similarity sense)** —
> a single `nearest_agent_kin` input (idx 78, obs grows 78→79) computed from
> a birth-time genome fingerprint, +40 params, migrated by the existing
> `migrate_genome`. Everything kin-sense would unlock today (group selection,
> family clusters, allele tracking) is already observable through
> territory/overlap + lineage logs, so this is research priority not a
> blocker.

**W6 — Substrate (can interleave with any phase).**
Spatial object index (perception + RewardShaper + spawn use it),
full checkpointing (`--save-state/--load-state`), per-generation metrics
CSV, reward-shaping config + `minimal` preset.
*Acceptance:* ≥3× tick-rate improvement at 100 agents on 100×100;
checkpoint→resume reproduces identical state in serial mode.

> **Shipped (W6a — spatial index).** `world/spatial_index.py`: a coarse-cell
> bucket of edible objects, maintained by `add/remove/move_object` plus the
> pickup/drop sites. The three nearest-food scans (perception stimulus,
> RewardShaper distance + direction) now delegate to a single
> `World.nearest_edible(ax, ay, scan_r)` that queries the index when present
> and falls back to an identical bounded tile scan otherwise. It is an
> *acceleration structure, not a source of truth* — every candidate is
> verified against live tile state and tie-breaks row-major — so results are
> bit-identical with the index on or off (asserted over 20 random worlds).
> Benchmarked **~3.5× faster** on the radius-10 reward scan with sparse food.
> `performance.spatial_index` (default on) toggles it. 11 tests
> (`tests/test_spatial_index.py`).
>
> **Remaining (W6b/c):** full checkpointing (`--save-state/--load-state`),
> per-generation metrics CSV, and the reward-shaping config + `minimal` preset.

**Suggested order:** W0 + W6a (spatial index) → W1 → W2 → W3 → W4 → W5, with
W6b/c (checkpointing, reward diet) slotted between. W1–W3 are pure
selection-pressure upgrades that the *existing* brains can be re-evaluated
against immediately; W4 is scheduled later precisely because it is the
genome break.

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| New pressures collapse populations (night + drought + poison stacking) | Each system has its own enable flag and gentle defaults; A/B survival runs per phase (the Phase-2 fade methodology); the event system's intensities are dials |
| Observation change invalidates saved populations | Single batched break (W4), append-only layout, zero-init migration with a bit-identical-behavior test |
| World becomes expensive (temperature, diffusion, fire) | Temperature is derived, not simulated; diffusion is vectorized row-parallel like SoilDynamics; spatial index lands *first*; perf benchmark added to CI |
| Hand-tuned ecology (species balance) becomes new hidden scripting | Species are plain YAML the user can read; balance validated by measurement (analyzer per-species sections), not embedded constants |
| Reward-shaping diet degrades learning | `legacy` preset remains default until A/B learning-curve comparison justifies switching; the diet is itself the experiment |
| Signals/trading hard-code cooperation | SIGNAL carries no semantics; trading is a capability, not an incentive; group-bonus stays an off-by-default labeled ablation |

---

## 7. Decision summary

1. **Build the climate engine first** (W1) — it's the multiplier that
   makes every existing system non-stationary, feeds curiosity, and costs
   no genome change.
2. **Make terrain mean something** (W2) — biomes + moisture transport turn
   "where" into a strategy dimension.
3. **Diversify the food web through the registry** (W3) — species and
   toxins are data; the first discrimination/preference results come here.
4. **Let agents perceive each other, then talk** (W4) — one batched,
   migration-safe genome break unlocks the entire social research agenda.
5. **Pay the engineering debts** (W6) — spatial index, checkpointing, and
   the reward-shaping diet keep the science honest and the simulation fast.

---

## 8. Object-system audit — issues found and upgrades

The ECS registry is the right architecture, but a code audit (verified
empirically, not just by reading) found real defects:

| # | Issue | Evidence | Severity |
|---|---|---|---|
| O1 | **Silent section typos.** `ObjectDefinition.from_dict` looks sections up by key and ignores anything unknown — a YAML typo like `edibel:` registers the object *successfully* with no food component and no warning. | Reproduced: `load_from_config({'my_food': {'edibel': ...}})` → loads, `edible is None` | High — this is the core "too difficult to use" failure |
| O2 | **Mixed failure modes.** A typo at *field* level (`callories:`) crashes with a context-free `TypeError: unexpected keyword argument` instead — so users get silence for one mistake class and a stack trace without the offending type_id for the other. | `Spec(**data[...])` dataclass kwargs | High |
| O3 | **Dangling cross-references accepted.** `grows_into`, `produces`, `decompose_into`, `spread_type_id` are never checked against the registry; `grows_into: NONEXISTENT` loads fine and raises `KeyError` minutes later inside `SeedGerminationSystem`. | Reproduced at load time | High |
| O4 | **Custom foods never regenerate.** `ResourceSpawnSystem`'s safety net hardcodes `"berry"` (`systems.py:613`); a standalone custom food exists only as `spawn.initial_count` copies — once eaten, gone forever. There is no respawn spec. | `_spawn_resource_near(world, x, y, "berry")` | Medium — major surprise for content authors |
| O5 | **Vision-encoding collisions unmanaged.** `observation.vision_encoding` is a hand-picked float; nothing warns when a custom object collides with a builtin (berry=1.0, plant=0.75, seed=0.6, fert=0.4, sand=0.15) — colliding types are *indistinguishable to every brain*, silently. | No check in `register()` | Medium |
| O6 | **Dead surface.** `EdibleComponent.toxicity` is parsed, documented ("1.0 = deadly") and wired to nothing; `ToolComponent`/`ToolSpec` ("DIG") are registered but no system consumes them. Documented promises that don't exist. | inventory §6 | Medium |
| O7 | **No reuse.** No `extends:`/defaults mechanism — every object repeats up to 8 boilerplate sections (the example file needs 267 lines for 4 objects, half of it comment-documentation because no other docs exist). | `config/custom_objects.yaml` | Medium |
| O8 | **Global mutable singleton.** `ObjectRegistry` is class-level state; tests need autouse reset fixtures, and two worlds in one process (island-model evolution, A/B framework) would share definitions. | `cls._definitions` | Low now, blocks W6/track work later |

**Upgrades (Phase W0):**
1. **Validating loader**: explicit allow-list of section/field names with
   "unknown section 'edibel' in 'my_food' — did you mean 'edible'?" errors;
   all errors collected and reported together with type_id context.
2. **Cross-reference check** after bulk load: every `grows_into`/`produces`/
   `decompose_into`/`spread_type_id` must resolve (builtins included).
3. **`extends:` inheritance + top-level `defaults:`** — an object inherits a
   registered definition and overrides only what differs.
4. **Respawn spec**: `spawn.respawn_rate` + `spawn.max_count` consumed by
   `ResourceSpawnSystem`, replacing the hardcoded berry safety net
   (builtins re-expressed through the same mechanism — one code path).
5. **Vision-encoding management**: collision warnings; optional
   `vision_encoding: auto` allocating within per-category reserved bands
   (food 0.85–1.0, plant 0.65–0.8, seed 0.5–0.65, hazard 0.1–0.3).
6. **Wire or remove dead surface**: toxicity → EAT (W3); ToolSpec removed
   until a system exists (honest schema).
7. **Instance registries**: `ObjectRegistry` becomes instantiable
   (`world.registry`), with the class-level API kept as a default-instance
   facade for backward compatibility.

## 9. Custom-object usability plan

The current authoring loop is: read a 90-line comment block → copy 60 lines
of boilerplate → run a full simulation → discover nothing spawned (or
crashed at tick 800). Target loop: *write 8 lines → validate in one second
→ preview → run*.

- **`scripts/objects.py` toolbox CLI**:
  - `validate config/my_objects.yaml` — schema + cross-refs + encoding
    collisions + "will anything actually spawn?" dry-run, in <1s;
  - `list` — table of all registered types (builtins + file): components,
    encodings, spawn behaviour;
  - `preview my_objects.yaml` — instantiate each object in a probe world
    and print exactly how agents will perceive it (vision encoding, value
    channel) and what each system will do to it per tick.
- **Minimal-first authoring** with `extends`:
  ```yaml
  objects:
    golden_apple:
      extends: berry          # inherit everything that makes berry work
      edible:   { calories: 60.0 }
      physics:  { decay_rate: 0.02, decompose_into: "" }
      observation: { vision_encoding: 0.95 }
      spawn:    { initial_count: 10, respawn_rate: 0.005 }
  ```
- **`docs/OBJECTS_GUIDE.md`**: schema reference generated from the spec
  dataclasses (single source of truth — the doc cannot drift), a cookbook
  (food / poison / plant chain / hazard / terrain effect), and the three
  rules people actually trip on (spawn required, encodings must differ,
  chains must close).
- `main.py --objects` prints the validator's summary at startup and
  **refuses to start on errors** (today it starts and misbehaves).

## 10. Target-track integration: Robotics & Entertainment

The two end-state platforms (SUGGESTIONS Parts 8–9) pull the world in
different directions — robotics wants *physics and reproducibility*,
entertainment wants *persistence, spectators, and user content*. The
upgrade should build the three seams both tracks share, and avoid
grid-world features that neither needs:

1. **A formal Environment interface** (the keystone). Extract what `main.py`
   implicitly does into `Environment`: `reset(seed) → obs`,
   `step(actions) → (obs, rewards, dones, info)`, plus declared
   observation/action-space descriptors (ObservationSpec already is one).
   The grid world becomes implementation #1; the robotics physics world
   (MuJoCo/PyBullet — external engines, not a port of this code) becomes
   implementation #2 behind the same interface; a thin Gymnasium adapter
   makes the whole RL ecosystem and standard benchmarking available. The
   brains, learners, world-model and dream stack already only consume
   (obs, action, reward, done) — they transfer unchanged.
2. **Determinism, checkpointing, and an event stream** (W6 + one addition).
   Robotics needs bit-reproducible episodes (serial mode + RNG-state
   checkpoints); entertainment needs a persistent always-on world (resume
   from checkpoint) and spectators. Add a versioned **state-delta stream**
   (per-tick JSON/msgpack: births, deaths, moves, events) — today both
   renderers poke directly at `world` internals; making them consume the
   stream turns "browser spectator client" from a rewrite into a websocket.
3. **The registry as the entertainment content pipeline.** User-authored
   creatures/objects (Part 9) are exactly the custom-object path — §9's
   validation/UX work *is* track work: the validator becomes the upload
   gate, `preview` becomes the content-creator tool, and instance
   registries (O8) allow per-server content sets.

What we deliberately do **not** do for the tracks yet: no elevation/3D in
the grid world (robotics gets real physics engines instead), no networking
layer (the event stream is its prerequisite), no continuous-action head
(brain-side change, scheduled when a physics environment exists).

## 11. Should the world stay on Python?

**Recommendation: yes for now — with a structure-of-arrays migration and a
measured escalation ladder, not a rewrite.**

The honest numbers: at 100 agents / 100×100 the simulation runs ~5–20
ticks/s (feature-dependent); a 1000-tick run is minutes. Profiling shows
the cost is dominated by (a) O(441)-tile Python scans per agent-action in
reward shaping/perception, (b) per-object Python loops in `systems.py`,
(c) per-agent torch updates — i.e. **algorithmic and layout problems, not
language ceiling**. Killing the project's research velocity (every
scientist-facing surface — registry, configs, analyzers, tests, torch — is
Python) for a rewrite before exhausting those is the wrong trade.

The ladder — each rung is taken only if the previous one misses its target:

| Rung | Change | Expected effect | Trigger |
|---|---|---|---|
| 1 (W6a) | Spatial object index | kills the O(441) scans → ~3–5× | now |
| 2 | **World v2 layout: structure-of-arrays.** Tile fertility/moisture/temperature become NumPy arrays (`world.fertility[y, x]`), systems become vectorized array ops (SoilDynamics/Environment/diffusion are pure stencils). This is also what makes W1–W2 cheap to add. | ~5–20× on systems; large worlds viable | with W1/W2 |
| 3 | Numba `@njit` on the few remaining hot kernels (germination, decay loops) — optional dependency, zero architecture change | 2–10× on those kernels | if 1000 agents @10 t/s unmet |
| 4 | Rust/C++ core for the tick loop behind the §10 Environment interface (PyO3), Python keeps config/registry/science/learning | order of magnitude | only for the persistent entertainment server, if rung 3 misses |

Two structural notes: the §10 Environment interface is exactly what makes
rung 4 a *swap* instead of a rewrite, so building it now keeps the Python
question reversible; and the robotics track never needed this world ported
— its 3D environments come from existing engines with Python APIs. Python
remains the contract; compiled code remains an implementation detail.

## 12. The stated destination: a 3D game-like world

The owner's aim is explicit: move beyond the 2D tile grid toward a 3D
game-like world. The honest engineering translation of that aim:

**What's true:** spatially coherent rivers, mountains, and biomes are not
expressible in the *current flat* tile world — terrain is a uniform random
shuffle with no elevation, and that is a real ceiling.

**What's also true:** the cheapest 80% of "3D game-like" is a
**heightmap (2.5D)**, not a polygon engine — this is how most game terrain
actually works. W2 makes elevation a first-class simulated field: water
flows downhill into rivers, mountains block and cost energy, biomes fall
out of elevation × moisture, and the existing ModernGL isometric renderer
displays it. That step needs no engine, no new language, no genome break.

**The staged road:**

| Stage | World | Perception / actions | Status |
|---|---|---|---|
| A (now → W2) | Heightmap 2.5D lab world | current tile tokens + elevation channel; discrete actions | this proposal |
| B | **Engine-backed 3D world** behind the §10 Environment interface — Godot (headless, open-source, good Python bridges) for the entertainment world; MuJoCo/Isaac for the robotics flavour | depth/raycast or low-res camera → v3's tile-token attention generalises to sensor patches (it was designed for this); locomotion → **continuous Gaussian action head** (the scheduled brain-side change) | after W4, as the tracks demand |
| C | Persistent multiplayer 3D world (entertainment Part 9) | Stage B + the event stream/networking | far |

**What transfers unchanged across stages** — and why the lab world is not
throwaway work: the brains (they consume ObservationSpec/action descriptors,
not tiles), both learners, curiosity/planner, the dream-evolution stack
(observation-space, world-agnostic), the registry as the content format,
and every analyzer. **What gets replaced per stage:** the perception
front-end and the action set — exactly the two seams ObservationSpec and
the Environment interface were built to isolate.

**Role split after Stage B:** the grid/heightmap world remains the *fast
lab* (thousands of ticks/second of evolution, interpretable, cheap dream
training); the 3D world is the *product* (spectators, robotics transfer).
Same genomes, same science, two worlds behind one interface — evolve in
the lab, ground in 3D, exactly the dream-evolution pattern already proven.

## 13. Updated decision summary

1. **W0 first**: harden the registry and fix the custom-object experience —
   cheapest phase, removes the worst user-facing pain, and W3's
   data-driven ecology depends on it.
2. **Adopt the SoA world layout when building W1–W2** (rung 2) — climate
   and diffusion want to be array stencils anyway; doing both at once
   avoids touching the systems twice.
3. **Build the Environment interface during W6** — it serves robotics
   (Gymnasium/physics engines), entertainment (event stream/spectators),
   and keeps the language decision reversible.
4. **Stay on Python** until the measured ladder says otherwise; the next
   two rungs are algorithmic and cost no rewrite.
5. **Fix the dead dynamics first-class** (§1 bugs): W1 exists precisely to
   make moisture a real constraint again — until then, one of the four
   observation channels agents are evolving against is a constant.
6. **Aim every terrain decision at the 3D destination** (§12): elevation
   as a first-class field now (heightmap rivers/mountains/biomes), an
   engine-backed 3D world behind the Environment interface next — the
   lab world stays as the fast-evolution substrate, not a dead end.
