# World Upgrade — Architecture Proposal

**Status:** Proposal (no code changes yet)
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

**W1 — Environment engine (no observation change).**
`EnvironmentSystem` (day/night, seasons, rain/drought events, derived
temperature), multipliers consumed by growth/germination/spawn/decay/soil/
metabolism; calamity generalized into the event system.
*Acceptance:* `environment.enabled: false` is bit-compatible with today;
enabled defaults keep populations viable over 2,000-tick runs; each cycle
visible in the world-state logs.

**W2 — Living terrain.**
Biome generator (smoothed noise, connected water), moisture diffusion,
3×3 nutrient return, temperature/moisture-coupled decay, slow erosion.
*Acceptance:* biome maps render correctly in both renderers; fertility
"river corridors" measurably form; legacy generator still selectable.

**W3 — Ecology & hazards (mostly data).**
2–3 plant/food species (YAML), toxicity wired into EAT, invasive species,
thorn hazards (`contact_damage`), `FireSystem`, flood event.
*Acceptance:* analyzer shows per-species consumption; agents' species
preference is measurable (the first discrimination result); fire spreads
and self-extinguishes at water/moisture boundaries.

**W4 — Agents in the world + Observation v2 (the one genome break).**
Agents visible in vision; optional tile exclusivity; pheromone field +
SIGNAL action; Observation v2 block + `output_size` 9 with zero-init
spec migration (old populations load and behave identically).
*Acceptance:* migration test proves bit-identical pre/post behavior for
old genomes; signal entropy and agent-proximity response measurable in
new analyzer sections.

**W5 — Social dynamics & open-endedness instruments.**
Kin-similarity observation feature, inventory transfer (trade via USE on
an adjacent agent), territory readout from the pheromone field;
role-entropy and behavioral-novelty metrics in `scripts/analyze_logs.py`.
*Acceptance:* division-of-labor metrics computable on long runs; all
features individually ablatable.

**W6 — Substrate (can interleave with any phase).**
Spatial object index (perception + RewardShaper + spawn use it),
full checkpointing (`--save-state/--load-state`), per-generation metrics
CSV, reward-shaping config + `minimal` preset.
*Acceptance:* ≥3× tick-rate improvement at 100 agents on 100×100;
checkpoint→resume reproduces identical state in serial mode.

**Suggested order:** W6a (spatial index) → W1 → W2 → W3 → W4 → W5, with
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
