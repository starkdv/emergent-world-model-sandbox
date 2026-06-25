# Emergent World-Model Sandbox

**Author:** Karan Vasa  
**License:** MIT  

A simulation sandbox with evolving agents that learn survival strategies and exhibit emergent behaviors through evolution in a physically consistent, resource-based 2D world.

## Overview

This project implements a **2D grid world** where:

- **Agents** move around, consume resources, and manipulate objects using only primitive actions
- **Evolution** drives the emergence of complex behaviors like farming, cooperation, and communication
- **World physics** governs resource transformation, plant growth, and environmental dynamics
- **Dual-mode evolution** — choose between **RL mode** (online Actor-Critic learning with Lamarckian inheritance) or **pure neuroevolution** (no gradient learning, genome-only) via `--mode` flag or config
- **No hardcoded behaviors** — all complex strategies emerge through natural selection and (optionally) reinforcement learning

## Key Features

- 🌱 **Emergent Agriculture**: Agents discover seed planting and farming through evolution
- 🏜️ **Environmental Hazards**: Sand terrain spreads and degrades soil unless trees block it
- 🧩 **Custom Object System**: Define new objects via YAML — foods, plants, terrain effects, structures
- 🤝 **Cooperation**: Group behaviors emerge without explicit programming
- 🧠 **Neural Evolution + Online Learning**: Agents use evolved GRU Actor-Critic networks with optional real-time RL
- 🔀 **Dual Evolution Mode**: RL mode (gradient learning + Lamarckian inheritance) or pure neuroevolution — selectable via `--mode rl` / `--mode neuroevolution`
- 🎮 **Three Renderers**: Pygame 2D GUI, a GPU-accelerated isometric 2.5D renderer (ModernGL), **and** a live **3D voxel (Minecraft-like) web view** streamed to the browser via Three.js (`python -m render.server`)
- 🎯 **Fading Instincts**: Survival-bootstrapping biases (PICK_UP, EAT, USE, turn-toward-food, hunger-eat) that genuinely fade to zero with agent age — adults act purely on their learned network
- 🔮 **Learned World Model** (opt-in): a latent dynamics head in the genome predicts *(next perception, reward)* — powering **curiosity** (surprise as intrinsic reward) and **imagination-based planning** (latent rollouts)
- 🧪 **Two Learners**: legacy heads-only A2C (the control) or full-network PPO with sequence replay, GAE(λ), and clipped updates — both Lamarckian
- 🔬 **Scientific Analysis**: Comprehensive action-distribution and survival-metric analysis tools
- ⚡ **Optimised Engine**: Per-tick caching, set-based tile indexing, persistent log file handles

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/starkdv/emergent-world-model-sandbox.git
cd emergent-world-model-sandbox

# Create and activate virtual environment (REQUIRED)
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Simulations

```bash
# Pygame 2D GUI (default)
python main.py --gui --seed 42

# GPU isometric 2.5D renderer (requires OpenGL 3.3+)
python main.py --gpu --seed 42

# 3D voxel (Minecraft-like) web view — live in the browser, no extra deps
python -m render.server          # then open http://127.0.0.1:8000
# (see the "3D Voxel Frontend" section below)

# Pure neuroevolution mode (no gradient learning)
python main.py --gui --mode neuroevolution

# RL mode (online Actor-Critic learning + Lamarckian inheritance)
python main.py --gui --mode rl

# Run with custom configuration
python main.py --config config/training_easy.yaml --gui

# Add custom objects (superfoods, mushrooms, etc.)
python main.py --gui --objects config/custom_objects.yaml

# Enable CSV logging for post-run analysis
python main.py --gui --log --log-frequency 10

# Headless data collection
python main.py --no-viz --generations 1000 --log
```

### GUI Controls (Pygame & GPU renderer)

- **SPACE**: Pause/Resume simulation
- **G**: Toggle grid overlay
- **R**: Reset camera to centre
- **WASD / Arrow Keys**: Pan camera
- **Mouse Wheel**: Zoom in/out
- **Left Click + Drag**: Pan camera
- **Hover over tiles**: Inspect terrain, objects, agent details
- **ESC**: Exit

> 💡 Pause with **SPACE** then hover for easier tile inspection.

**New to the project? Start with the [USER_GUIDE.md](docs/USER_GUIDE.md)** —
which mode to pick and why, training, the dream world, planning, the 3D viewer,
exporting the world, and Blender round-tripping.

See [CLI_GUIDE.md](docs/CLI_GUIDE.md) for the full command-line reference, and
**[Modes & Feature Toggles](#modes--feature-toggles--complete-reference)**
below for every mode, its prerequisites, and what enabling it does.

### 3D Voxel Frontend (Minecraft-like, in the browser)

A live voxel view of the world, rendered in the browser with Three.js. The
Python sim streams its state **read-only** (it never affects the simulation)
over Server-Sent Events; the browser builds the world as voxel columns from the
W2 `elevation` field, with biome blocks, water, a W1 day/night sky, per-lineage
agents grounded on the terrain, distinct per-category object models (trees,
berries, …), and a pheromone-signal glow. **Click anything** — an agent, a tree,
or a tile — to inspect its properties.

By default the world is built **from `config/default.yaml`** (size, heightmap
biomes, climate, population), with learning agents and reproduction on so the
population sustains itself.

```bash
# live view — stdlib only, no extra Python dependencies
python -m render.server                       # → http://127.0.0.1:8000 (config-driven)
python -m render.server --config config/training_easy.yaml
python -m render.server --demo                # fixed self-contained scene
python -m render.server --checkpoint data/states/run.pkl   # fly around a saved run (W6b)

# record a run's render stream, then replay it on a loop (no re-simulation)
python -m render.recorder --out run.jsonl --ticks 600
python -m render.server --replay run.jsonl     # → http://127.0.0.1:8000

# offline: export a frame to a colored PLY (opens in Blender / MeshLab)
python -m render.voxel_export --out world.ply
python -m render.voxel_export --checkpoint data/states/run.pkl --out run.ply
```

**Controls:** drag = orbit · scroll = zoom · WASD/QE = free-fly · **F** = follow
the next agent (chase cam) · **R** = reset camera · **click anything** to inspect
— an agent (energy, lineage, generation), a tree/berry (category, type), or a
tile (terrain, elevation, fertility, moisture). Lifecycle events show as
particle bursts (birth/death/consume); wildfire glows and flickers.

Architecture (the bridge is read-only, so determinism/checkpointing are
untouched), full design + roadmap in
**[FRONTEND_3D_PROPOSAL.md](docs/FRONTEND_3D_PROPOSAL.md)** and
**[web/README.md](web/README.md)**:

```
Python sim ──► render/state_bridge.py ──► render/server.py (SSE) ──► web/main.js
 (unchanged)     snapshot + per-tick deltas   /api/stream            Three.js voxels
```

## What's New — Brain v3 Upgrade (Phases 1–4)

**In one sentence:** the brain's internals were reorganised so it can grow,
the survival "training wheels" now genuinely come off as agents mature, an
opt-in attention-based brain is available alongside the legacy one, lifetime
learning now reaches every weight, and agents can carry a **learned world
model** that powers curiosity and imagination-based planning.

**In plain words:**
- The neural network's wiring diagram now lives in *one* place
  (`agents/brain/spec.py`) instead of three hand-synchronised copies, so
  upgrading the architecture no longer risks silently corrupting genomes.
- Baby agents still get helpful nudges ("pick that up", "eat when hungry",
  "turn toward food"), but these nudges now **weaken every tick and vanish at
  age 150** — adult behaviour is 100% produced by evolved/learned weights.
- The old hidden rule that *forced* hungry agents to eat was removed. Hungry
  agents now merely feel a strong urge to eat (which also fades with age), so
  "knowing when to eat" becomes something evolution and learning must solve —
  as the project's emergence-first principle demands.
- A new opt-in **attention brain** (`brain.version: 3`) is available: agents
  perceive their surroundings through a small attention mechanism steered by
  their internal state, and the value estimate sees the current situation
  directly. The legacy brain stays the default so the two can be compared
  under identical conditions (see *Neural Architecture* below).
- A new opt-in **PPO learner** (`learning.algorithm: ppo`) makes lifetime
  learning real: previously only the network's output layer learned during a
  lifetime; now gradients reach **every weight** (perception, memory, heads)
  via sequence replay, GAE(λ) advantages, and PPO-clipped updates — and the
  learned weights are still inherited by offspring (Lamarckian).
- An opt-in **learned world model** (`brain.world_model.enabled`): a latent
  dynamics head in the genome predicts *(next latent, reward)* from
  *(memory, action)*. It unlocks **curiosity** (prediction error as
  intrinsic reward — exploration without hand-crafted bonuses) and a
  **latent rollout planner** (the agent imagines action consequences in
  latent space and picks the best first move).
- **Dream-based evolution** (`scripts/dream_evolve.py`): a population-level world
  model is trained offline from the transition logs and used as a virtual
  environment — genomes evolve *inside the dream* at a fraction of the cost
  of real simulation, then champions are grounded back in the real world.

Full design rationale is in [BRAIN_V3_PROPOSAL.md](docs/BRAIN_V3_PROPOSAL.md);
change details are in [CHANGELOG.md](CHANGELOG.md). For a complete,
math-level comparison of the two brains and the two learners, see
[**BRAIN_V2_V3_COMPARISON.md**](docs/BRAIN_V2_V3_COMPARISON.md).

## Modes & Feature Toggles — Complete Reference

Everything below is opt-in and independently switchable, so any combination
can be run as a controlled experiment. Quick map:

| # | Mode | Switch | Default | Requires |
|---|------|--------|---------|----------|
| 1 | Evolution mode | `--mode rl\|neuroevolution` / `evolution.mode` | `rl` | — |
| 2 | Brain version | `brain.version: 2\|3\|3.5` | `3` | 3.5 adds social senses + SIGNAL |
| 2b | Signalling (v3.5) | `signal.enabled` | off | `brain.version: 3.5` |
| 3 | Fading instincts | `brain.instincts` | on, fade at 150 | — |
| 4 | Learning algorithm | `learning.algorithm: a2c\|ppo` | `a2c` | RL mode; PPO needs torch |
| 5 | World model | `brain.world_model.enabled` | off | — (training needs PPO) |
| 6 | Curiosity | `learning.curiosity.enabled` | off | world model + RL mode |
| 7 | Latent planner | `brain.world_model.planner.enabled` | off | world model |
| 8 | Dream evolution | `python scripts/dream_evolve.py` | — | torch + `--world-model-log` data |
| 9 | Environment engine | `environment.enabled` | off | — |
| 10 | Terrain generator | `terrain.generator: legacy\|heightmap` | `legacy` | — |
| 11 | Wildfire | `fire.enabled` | off | — (hotter/drier = more fire) |
| 12 | Ecology species pack | `--objects config/ecology.yaml` | — | — |
| 13 | Agents visible in vision | `world.agents_visible` | off | — |
| 14 | Tile exclusivity | `world.agent_collision` | off | — |

All YAML keys live in `config/default.yaml` (heavily commented). CLI flags
override config.

### 1. Evolution mode — `rl` vs `neuroevolution`

**Enable:** `python main.py --mode rl` or `--mode neuroevolution`; or set
`evolution.mode` in the config. Resolution priority: `--mode` flag >
legacy `--learning` flag (= rl) > config (default `rl`).
**Prerequisites:** none.
**When `rl` is enabled:** every agent gets a lifetime learner (see #4);
learned weights are synced back into the genome after every update and
inherited by offspring with mutation (**Lamarckian inheritance**).
**When `neuroevolution` is enabled:** no gradients ever run — behaviour
changes only through mutation + selection across generations. All
`learning.*` settings are ignored. This is the scientific control for
every "does learning help?" question.

### 2. Brain version — legacy GRU-MLP vs attention brain

**Enable:** `brain.version: 3` in the config (`2` is the default baseline).
**Prerequisites:** none — v3 runs on plain NumPy for acting; the A2C
learner has a dedicated v3 path. **Start a fresh run**: v3 genomes are
17,337 weights vs v2's 8,873, so weights/populations saved under the other
version cannot be loaded.
**What happens:** perception switches from one dense layer to 25 shared
tile tokens pooled by attention (the agent's internal state decides which
tiles matter each tick), memory grows to GRU(48), and the value estimate
reads the current state directly instead of only memory. ~2.4× the compute
per tick (still microseconds). Under pure neuroevolution, early
populations are measurably weaker (more parameters to search) — that
capacity-vs-evolvability trade-off is itself an experiment. Three size
recipes (`v3-small` 9,617 / `v3-base` 17,337 / `v3-large` 29,537 params)
are set via the `brain.v3` keys — see
[v3 size presets](#v3-size-presets--small--base--large).

### 3. Fading instincts

**Enable:** on by default. Tune under `brain.instincts`:
`fade_age: 150` (ticks until strength 0; `null` = never fade, the legacy
behaviour), `enabled: false` (no instincts at all — hard mode),
`hunger_eat_bias: 3.0`.
**Prerequisites:** none. Applies identically in both evolution modes.
**What happens when fading is on (default):** newborns get additive
logit nudges (pick up food +1.5, eat +1.0, eat-when-hungry +3.0×urgency,
plant +0.5, turn-toward-food) that shrink every tick and hit zero at
`fade_age` — adults act purely on evolved/learned weights. Survival
pressure rises versus the legacy never-fading setting (fewer agents
survive a fixed window); populations remain viable at the defaults.
**If you disable instincts entirely:** most random-weight newborns starve
before learning/selection can act — useful only as an ablation.

### 4. Learning algorithm — `a2c` vs `ppo`

**Enable:** `learning.algorithm: ppo` (tunables under `learning.ppo`).
**Prerequisites:** RL mode (#1); PyTorch installed — without torch the
agent prints `[LEARN] torch unavailable — falling back to a2c` and uses
the legacy learner.
**When `a2c` (default):** the legacy learner — random single-transition
replay, TD(0) advantage, manual gradients on the **output heads only**;
the encoder and GRU change only through evolution. Cheap (µs/update) and
the control condition.
**When `ppo`:** gradients reach **every weight** (perception, GRU through
time, heads) via a persistent torch mirror; time-ordered 8-step sequence
replay, GAE(λ) advantages, PPO-clipped updates, Adam + grad clipping.
Real lifetime learning — and still Lamarckian. Cost: ~ms per update on
CPU (a v2+PPO 1000-tick run ≈ a few minutes; PPO+v3 ≈ 0.5 s/tick); set
`learning.compute_device: cuda` at scale.

### 5. Learned world model (per-agent dynamics head)

**Enable:** `brain.world_model.enabled: true` (width via `hidden: 32`).
Works with both brain versions.
**Prerequisites:** none to enable — but the head is only *trained* by the
PPO learner (#4); under a2c or neuroevolution it evolves like any other
genome weights. **Changes the genome length** (v2: 11,274; v3: 20,778),
so start fresh — saved weights without the head cannot be loaded.
**What happens:** the genome gains a small head predicting *(next latent,
reward)* from *(memory, action)* — the agent's "imagination". On its own
it changes nothing visible; it is the prerequisite that unlocks #6 and #7,
and PPO adds its prediction-error loss to training. Startup banner shows
`world model ON`.

### 6. Curiosity (intrinsic reward)

**Enable:** `learning.curiosity.enabled: true` (plus `weight`, `decay`,
`clip`, `warmup`).
**Prerequisites:** world model (#5) **and** RL mode (#1) — curiosity
shapes the learning reward, so it is attached with the learner and is
silently inactive without the dynamics head.
**What happens:** each tick the dynamics head's prediction error is
z-scored against running statistics; above-average surprise becomes a
positive reward bonus (`weight × clip(zscore⁺, 0, clip)`), zero during the
first `warmup` steps. Agents are pushed toward transitions their model
cannot yet predict — exploration emerges instead of being hand-crafted.
Expect noisier early behaviour; use `decay < 1` to anneal curiosity away
over a long run. Offspring inherit the setting.

### 7. Latent rollout planner (imagination-based actions)

**Enable:** `brain.world_model.planner.enabled: true` (plus `depth`,
`samples`, `gamma`).
**Prerequisites:** world model (#5). Meaningful only once the dynamics
head is competent — i.e. trained with PPO (#4) or after substantial
evolution; with a random head it is expensive noise.
**What happens:** instead of sampling the policy directly, the agent
imagines `samples` action sequences of length `depth` entirely in latent
space (predict ẑ′ → advance the GRU → accumulate predicted reward →
critic bootstrap) and takes the first action of the best rollout. Cost:
`samples × depth` extra forward passes per agent per tick (default 16×3).
The policy network still runs (memory must advance), and under PPO the
planner's choices are recorded and learned from.

### 8. Dream-based evolution (offline)

**Enable:** a three-step pipeline, not a config key:

```bash
# 1. Collect real experience (any mode; more data = better model)
python main.py --no-viz --world-model-log --seed 42

# 2. Train a population world model on the logs, evolve genomes inside it
python scripts/dream_evolve.py --transitions "data/logs/transitions_*.csv"

# 3. GROUND the champions in the real environment (mandatory)
python main.py --load-weights data/weights/dream_best.npz --no-viz
```

**Prerequisites:** PyTorch; transition CSVs from `--world-model-log`
(≥ ~500 rows minimum, thousands recommended; logs written before the
June 2026 schema fix are column-misaligned and will be skipped); the
`--config` passed to `scripts/dream_evolve.py` must match the brain that will
load the result (version + world-model setting decide genome length).
**What happens:** an observation-space model of the *environment itself*
is trained from the logs, then a (μ+λ) evolutionary loop evaluates every
genome by imagined rollouts — thousands of episodes per second instead of
full simulation. The top-5 champions are saved in the same `.npz` format
`main.py --load-weights` consumes. Dream fitness is a **proxy**: evolution
exploits model errors, which is why step 3 is not optional.

### 9. Environment engine — day/night, seasons, weather

**Enable:** `environment.enabled: true` in `config/default.yaml`
(see the heavily-commented `environment:` block for every knob).
**Prerequisites:** none — works in every mode and with every brain.
**What happens:** one system computes a global climate each tick —
a light cycle (`day_length` ticks per day), a slow seasonal temperature
wave (`season_length`), and stochastic rain/drought events — and the
existing systems consume plain multipliers from it: plants grow with
light × a temperature comfort window, seeds germinate only in livable
temperatures, food production follows daylight, spoilage speeds up in
heat, and agent metabolism rises at both temperature extremes. It also
**replaces the soil moisture model**: moisture now *evaporates*
(faster when hot/bright, doubled in droughts) and recovers **only**
during rain or next to water tiles — fixing a long-standing bug where
moisture could only ever rise. The intent is emergence-first: nothing
scripts the agents, but day/night rhythms, seasonal scarcity, and
weather shocks become real pressures their brains can discover and
exploit. Disabled (the default), every multiplier is exactly 1.0 and
the simulation is bit-compatible with the pre-upgrade baseline.

### 10. Terrain generator — `legacy` vs `heightmap`

**Enable:** `terrain.generator: heightmap` in `config/default.yaml`
(with an optional `terrain.heightmap:` block of tunables).
**Prerequisites:** none (pure NumPy; works in every mode).
**What happens:** instead of the legacy uniform random shuffle, terrain is
built from an **elevation heightmap**: the high ground becomes mountain
**rock**, water settles into basins and **flows downhill into rivers**
(steepest-descent tracing from the peaks), and soil/desert-**sand** fall out
of a geography-driven moisture field — with **fertile river corridors**.
Elevation becomes a first-class per-tile field, so **moving uphill costs
extra energy** (slope cost). The rock/water/sand ratios you already set are
honoured via elevation quantiles. Elevation is *not* in the agent
observation yet (that genome break is reserved for a later phase), so old
genomes run unchanged; `legacy` (the default) keeps the flat world exactly
as before.

Preview any seed as ASCII before running a simulation:

```bash
python scripts/terrain.py preview --seed 42                  # terrain map
python scripts/terrain.py preview --seed 42 --elevation      # height field (0–9)
python scripts/terrain.py preview --config config/default.yaml
```

### 11. Wildfire — `fire.enabled`

**Enable:** `fire.enabled: true` in `config/default.yaml` (tunables under
`fire:`). Most dramatic with the environment engine on and a warm
`base_temperature`.
**Prerequisites:** none (reads the W1 climate if present; otherwise uses a
moderate constant heat).
**What happens:** plants on **hot, dry** tiles can ignite; fire spreads to
adjacent plants and burns them out, returning ash (fertility) to the soil.
Heat comes from `environment.temperature` and dryness from each tile's
moisture, so fire is rare in the wet/cold and dangerous in the hot/dry — and
it **self-extinguishes at water/wet boundaries** (a damp strip of vegetation
is a firebreak). This is a real, learnable disturbance rather than the blunt
periodic calamity. Disabled (default) → no-op.

### 12. Ecology species pack — `--objects config/ecology.yaml`

**Enable:** `python main.py --objects config/ecology.yaml` (combine with any
mode/renderer).
**Prerequisites:** none.
**What happens:** the single-berry food web becomes an ecology with
trade-offs: a fast/cheap **shrub berry** (+10 energy), a slow/rich **tree
fruit** (+45), and a **nightshade** look-alike that is *net-negative* to eat.
Toxicity is now a real physical consequence of EAT (net energy =
`calories × freshness − toxicity × freshness × 30`), so the nightshade is the
first **discrimination task**: nothing flags it as poison — agents must learn
to avoid it from the energy hit. Each species has a distinct `vision_encoding`
so the v3 attention brain can tell them apart, and `scripts/analyze_logs.py`
reports **per-species consumption** (with a toxic flag) so preference and
poison-avoidance are measurable. Built-in **thorns** (`contact_damage`) add a
crossable-but-costly terrain hazard. Validate/preview any pack first:

```bash
python scripts/objects.py validate config/ecology.yaml
python scripts/objects.py preview  config/ecology.yaml   # shows net energy + ⚠ POISON
```

### 13. Agents visible in vision — `world.agents_visible`

**Enable:** `world.agents_visible: true`.
**What happens:** another living agent occupying a vision tile reads as the
encoding `0.40` with that agent's energy ratio in the value slot, overriding
the terrain/object underneath (an agent never sees itself). This is the
unblock for every social research direction — until now agents were
invisible to each other. It changes vision *content*, not the observation
size, so existing genomes run unchanged; they simply start perceiving a new
value where agents stand.

### 14. Tile exclusivity — `world.agent_collision`

**Enable:** `world.agent_collision: true`.
**What happens:** a tile occupied by a living agent blocks `MOVE_FORWARD`, so
physical space becomes a contested resource (territory, crowding, blocking).
Off by default, agents may overlap exactly as before.

> **W4 status:** these two toggles, the genome-migration tool, **and** the
> batched **Observation v2 + SIGNAL** genome break are all shipped — the break
> landed as **Brain v3.5** (`brain.version: 3.5`; see the Neural Architecture
> section and [BRAIN_V3_PROPOSAL.md §8](docs/BRAIN_V3_PROPOSAL.md)).

### 15. Inventory transfer — `social.transfer_enabled` (W5)

**Enable:** `social.transfer_enabled: true`.
**What happens:** the existing USE action now checks the tile in front first;
if a living agent is there with inventory space, USE hands them the first
inventory item (logged as `interaction_kind="give"`). Trade carries no
built-in reward — any protocol (request via SIGNAL, donor specialisation,
reciprocity) must emerge from selection. Off by default → legacy USE
behaviour, bit-compatible with W4 runs.

> **W5 status:** trade is shipped, and the analyzer's new **🧬 SOCIETY /
> ROLES** section prints role-entropy, behavioural novelty (mean pairwise JS),
> territory (bbox + Jaccard overlap) and trade counts on every run with ≥2
> agents — no flag required. A `nearest_agent_kin` similarity sense is
> deferred to the next batched genome bump (Brain v3.6 / Observation v3).

### Supporting flags & settings

| Switch | Effect |
|---|---|
| `--world-model-log` | Write transition/episode/world-state CSVs (the dream-evolution fuel). ~1 KB per transition. |
| `--log` (+ `--log-dir`, `--log-frequency`) | Write per-action + per-tick agent-state CSVs for analysis. |
| `--load-weights F.npz` / `--save-weights` | Seed agents from saved weights / save the best at the end. **Weight length must match the configured brain** (version + world model), otherwise loading fails. |
| `--save-state F.pkl` / `--load-state F.pkl` | (W6b) Write a full checkpoint (world + agents + RNG) at the end of a run / resume from one. In serial mode (`simulation.parallel: false`) a resumed run is bit-identical to an uninterrupted one. |
| `--metrics-csv F.csv` | (W6c) Append one aggregate row per generation (population, food/plant/seed counts, mean energy/age, mean fitness, soil means) during a headless run. |
| `--seed N` | Reproducible world generation. Note: with `simulation.parallel: true` (default) agent updates are threaded and runs are not bit-reproducible; set `parallel: false` for determinism. |
| `--gui` / `--gpu` / `--no-viz` | Pygame 2D / ModernGL isometric / headless. |
| `python -m render.server` | (Frontend) Serve the live 3D voxel web view (SSE); `--checkpoint`, `--width/--height/--agents/--tps`. |
| `python -m render.voxel_export` | (Frontend F1) Export a frame to colored PLY; `--checkpoint`, `--out`. |
| `learning.compute_backend/device` | `numpy`/`torch`, `cpu`/`cuda`/`mps` for the learners. |

**The full stack in one config:** `brain.version: 3` +
`brain.world_model.enabled: true` + `brain.world_model.planner.enabled:
true` + `learning.algorithm: ppo` + `learning.curiosity.enabled: true`,
run with `--mode rl`. Budget CPU accordingly (or use `cuda`) — every
feature stacks its per-tick cost.

## Architecture

```
emergent-world-model/
├── world/           # Tiles, objects, systems, object registry, tile-effect engine
├── agents/          # Agent lifecycle, evolution, genome
│   ├── brain/       # Brain package: v2 + v3 (attention) architectures,
│   │                #   spec (genome/observation layouts), instincts (fading)
│   ├── learning.py  # A2C learner (legacy, heads-only — the control)
│   ├── ppo.py       # PPO learner (full-network backprop, GAE, clipping)
│   ├── curiosity.py # Intrinsic reward from world-model prediction error
│   ├── planner.py   # Latent rollout planner (imagination-based actions)
│   └── dream.py     # Population world model + dream-based evolution
├── simulation/      # Simulation management
├── render/          # 3D frontend (read-only): state_bridge (snapshot+deltas),
│                    #   sim_session, server (SSE), voxel_export (PLY)
├── web/             # Three.js voxel web client (index.html, main.js, style.css)
├── utils/
│   ├── agents/      # Perception (72/78-dim obs), action execution, reward shaping
│   ├── data/        # AgentLogger, WorldModelLogger (async + persistent handles)
│   └── ui/          # Pygame renderer, GPU isometric renderer (ModernGL)
├── config/          # YAML configs + custom object definitions
├── scripts/         # Analysis tools + dream-evolution CLI
├── docs/            # All project documentation
├── tests/           # 500+ unit tests
└── data/            # Logs, exported data, saved weights
```

## Neural Architecture

Two brain versions are available, selected by `brain.version` in the config.
Both are compact recurrent **Actor-Critic** networks; v2 is the default and
the controlled baseline for experiments.

**v2 — legacy GRU-MLP (≈8 873 parameters):**

```
Observation (72) → Encoder [FC→tanh] (32) → GRU (32) → Policy head (8 actions)
                                                       → Value head  (1)
```

**v3 — attention brain (`brain.version: 3`, ≈17 337 parameters):**

```
vision 5×5×2 ──→ 25 tile tokens (+ positional enc) → shared embed (4→8) ─┐
agent state + stimulus + inventory ──→ state encoder (22→40) ──┐         │
                                          │ (attention query)  │  keys/values
                                          ▼                    ▼         ▼
                                  attention pool ◄────────────────────────┘
                                          │
                       latent z = [state 40 | pooled vision 8]
                                          │
                                     GRU (48)
                                          │ h
              Policy head (8, masked)  ·  Value MLP ([z,h] → 16 → 1)
```

Why v3 (full rationale in [BRAIN_V3_PROPOSAL.md](docs/BRAIN_V3_PROPOSAL.md)):
- **One tile embedding shared by all 25 tiles** makes perception
  position-equivariant and lets it scale to larger vision radii with the
  *same* weights — v2's dense vision layer spends ~1 600 parameters just
  memorising tile positions.
- **Attention driven by the agent's internal state** lets the network focus
  on relevant tiles (e.g. food when hungry) instead of weighting all 25
  cells equally.
- **The value head reads [z, h]** — the critic sees the current state
  directly instead of only what the GRU chose to remember.

```yaml
brain:
  version: 3        # 3 (default) = attention brain, 2 = legacy GRU-MLP baseline
  v3:
    embed_dim: 8
    state_dim: 40
    gru_hidden_size: 48
    value_hidden: 16
```

**v3.5 — the social brain (`brain.version: 3.5`, World phase W4):** a *minor*
bump of v3 that widens its I/O so agents can live in each other's world —
**six new observation inputs** (time-of-day sin/cos, tile temperature,
nearest-agent proximity & signal, on-hazard → state encoder 22→**28**) and
**one new action**, **SIGNAL**, with a decaying pheromone field (policy head
8→**9**). The attention/GRU/value core is unchanged; only the input and output
layers grow (**+289 params, <2%**, v3.5-base ≈ 17 626). An existing v3 genome
auto-migrates into v3.5 on `--load-weights` with the same behaviour on the
original actions (to float tolerance; new weights zero-init). Enable SIGNAL +
the pheromone field with `signal.enabled: true`, and combine with
`world.agents_visible` so agents both see and signal each other. Full design +
as-built notes: **[BRAIN_V3_PROPOSAL.md §8](docs/BRAIN_V3_PROPOSAL.md)**.

```bash
python main.py --gui --config config/default.yaml \
  # set brain.version: 3.5, signal.enabled: true, world.agents_visible: true
```

**v3.5 architecture (the attention core is byte-for-byte v3; only the shaded
input/output layers grow):**

```
   Observation v2 (78 = 72 v1 prefix ⧺ 6 EXTRA)        ObservationSpec(version=2)
   │
   ├─ vision 5×5×2 (50) ── AGENT-AWARE: a tile with another living agent
   │      │                encodes (0.40, its energy)            [W4 part 1]
   │      ▼
   │   per-tile embed (4→E, shared) → 25 tile tokens (+ fixed positional enc)
   │      │
   │      ├──────────────── keys / values ─────────────┐
   │                                                    │
   ├─ STATE block (28) = agent_state(8) ⧺ stimulus(8) ⧺ inventory(6)
   │      │                                ⧺ EXTRA(6)  ◄── NEW v3.5 senses
   │      │   EXTRA = [time_of_day sin, time_of_day cos, tile_temperature,
   │      │            nearest_agent_proximity, nearest_agent_signal, on_hazard]
   │      ▼
   │   state encoder (28→S) ── s ──► attention query ──► softmax pool ─► e (E)
   │      │                                                            │
   └──────┴──────────────── concat → latent  z = [s ‖ e]  (Z = S+E) ──┘
                                       │
                          Memory core: GRU (H)
                                       │ h
          ┌──────────────────┬─────────┼──────────────────┬────────────────┐
          ▼                  ▼          ▼                  ▼                ▼
     Policy head        Value MLP   Dynamics head      (instincts        SIGNAL
     h → 9 logits       [z,h]→V     (h, onehot a9)      fade w/ age)     = action #8
     (8 + SIGNAL),                  → ẑ_{t+1}, r̂                         in policy
     SIGNAL masked                  optional world model                 head ↑
     unless signal.enabled

   SIGNAL execution → writes a float on the agent's tile → PheromoneField
   (decays each tick, optional diffusion) → re-sensed as EXTRA[nearest_agent_signal].
   Signals carry NO built-in meaning — any protocol must emerge.
```

| Tensor | v3 | v3.5 | Δ (S=40, H=48) |
|---|---|---|---|
| `state_enc.W` | (22, S) | (28, S) | +6·S = **+240** |
| `policy.W` / `policy.b` | (H, 8)/(8,) | (H, 9)/(9,) | +H+1 = **+49** |
| **total** | 17 337 | **17 626** | **+289 (<2%)** |

**Worked example — running the social brain end to end.** Turn on the v3.5
brain together with the world features it perceives, run headless with logging,
then read the social metrics:

```bash
# 1) Enable the social stack in config/default.yaml (or a copy):
#      brain.version: 3.5
#      signal:  {enabled: true}        # SIGNAL action + pheromone field
#      world:   {agents_visible: true, agent_collision: true}
#      environment: {enabled: true}    # day/night so the clock senses vary
#
# 2) Run with RL + per-action logging:
python main.py --no-viz --mode rl --log --config config/default.yaml

# 3) Read what emerged (the analyzer auto-picks the latest log):
python scripts/analyze_logs.py
```

The analyzer prints two social sections that are meaningful only for v3.5 runs:

```
🛰️  SOCIAL / SIGNAL
  SIGNAL actions:         1843 (4.21% of actions)
  Signalling agents:      18
  Signal entropy:         3.94 bits (92.1% of max — 1.0 = shared evenly)
  Agent-proximity response (action mix by how close others are):
    proximity   actions  SIGNAL%  dominant
    alone         30214    1.10%  MOVE_FORWARD
    near          11883    6.40%  MOVE_FORWARD
    close          4120   18.07%  SIGNAL          ← agents signal far more in company

🧬 SOCIETY / ROLES
  Role entropy:           0.91 bits (45% of max — division of labor)
  Behavioural novelty:    mean pairwise JS = 0.23 bits
  Territory:              mean bbox = 1483 tiles, mean Jaccard overlap = 0.04
```

(Numbers are illustrative — a real run varies.) The signalling-rises-with-
proximity pattern and a non-trivial role entropy are the emergence signals
v3.5 exists to make measurable. To *watch* the same run as a 3D voxel world,
add `--save-state run.pkl` and open it with
`python -m render.server --checkpoint run.pkl` (see the
[3D Voxel Frontend](#3d-voxel-frontend-minecraft-like-in-the-browser) section).

> **▶ Next up — v3.6 (the kin-similarity sense).** The one piece deferred from
> World phase W5 because it touches the genome: a single `nearest_agent_kin`
> input (obs 78→79, +40 params) computed from a birth-time genome fingerprint,
> so agents can perceive how related a neighbour is and kin selection can
> emerge. It is **designed and is the current open work item** — full design in
> **[BRAIN_V3_PROPOSAL.md §9](docs/BRAIN_V3_PROPOSAL.md)**.

#### v3 size presets — small / base / large

There are no named presets in the config; the three sizes from
[BRAIN_V3_PROPOSAL.md](docs/BRAIN_V3_PROPOSAL.md) §3.2 are **recipes for the
`brain.v3` keys**. Exact parameter counts (from `ParamSpec`, verified by
tests; add the dynamics head from `brain.world_model` on top: +~2–4k):

| Preset | embed_dim (E) | state_dim (S) | gru_hidden_size (H) | Params | Use case |
|---|---|---|---|---|---|
| `v3-small` | 8 | 32 | 32 | **9,617** | evolvability parity with v2 (8,873) — isolates the *architecture* effect from the *capacity* effect |
| `v3-base` (default) | 8 | 40 | 48 | **17,337** | the standard v3 |
| `v3-large` | 12 | 52 | 64 | **29,537** | capacity study — how far can mutation/PPO push a bigger brain? |

```yaml
# v3-small — drop-in:
brain:
  version: 3
  v3: { embed_dim: 8, state_dim: 32, gru_hidden_size: 32, value_hidden: 16 }

# v3-large — drop-in:
brain:
  version: 3
  v3: { embed_dim: 12, state_dim: 52, gru_hidden_size: 64, value_hidden: 16 }
```

Notes: the GRU input is always `z = state_dim + embed_dim`; each preset is
a **different genome length**, so saved weights/populations don't transfer
between presets (start fresh, or use dream evolution with a matching
`--config`). For pure neuroevolution, prefer `v3-small` — search degrades
as the genome grows.

### Lifetime Learning: A2C vs PPO

Two learning algorithms are available in RL mode (`learning.algorithm`):

| | `a2c` (default) | `ppo` |
|---|---|---|
| Trains | output heads only | **every parameter** (perception, GRU, heads) |
| Replay | random single transitions | time-ordered sequence chunks (BPTT) |
| Advantage | TD(0) | GAE(λ) |
| Update safety | none | PPO clipped ratio + grad-norm clip |
| Backend | NumPy | torch (falls back to a2c without it) |

Both are Lamarckian — learned weights are packed back into the genome and
inherited by offspring. The full derivations (GRU gates, attention scaling,
policy-gradient algebra, GAE telescoping, the clipped surrogate) are in
[BRAIN_V2_V3_COMPARISON.md](docs/BRAIN_V2_V3_COMPARISON.md).

### Learned World Model, Curiosity & Planning

With `brain.world_model.enabled: true`, the genome gains a small **latent
dynamics head** (works with both brain versions):

```
d  = tanh([h ‖ onehot(a)]·W1 + b1)      h = GRU memory, a = imagined action
ẑ' = d·Wz + bz                          predicted next latent
r̂  = d·Wr + br                          predicted reward
```

It is trained end-to-end by the PPO learner (auxiliary loss against the real
next latent, stop-gradient targets) — or simply evolves in pure
neuroevolution mode. It unlocks two capabilities:

- **Curiosity** (`learning.curiosity`): the prediction error, z-scored over
  running statistics and clipped, becomes an intrinsic reward. Surprising
  transitions are rewarded; exploration emerges instead of being hand-coded.
- **Planning** (`brain.world_model.planner`): random-shooting rollouts
  entirely in latent space — imagine ẑ′, advance the GRU, accumulate r̂,
  bootstrap with the critic at the horizon — and take the best first action.

```yaml
brain:
  world_model: { enabled: true, hidden: 32,
                 planner: { enabled: false, depth: 3, samples: 16 } }
learning:
  curiosity:   { enabled: true, weight: 0.1, decay: 1.0 }
```

### Dream-Based Evolution

The capstone of the world-model stack: a **population-level** model of the
environment itself (observation-space, policy-agnostic — any genome can be
evaluated in it), trained offline from the transition logs, used as a cheap
virtual world for evolution. Thousands of imagined episodes per second
instead of full simulation:

```bash
# 1. Collect real experience
python main.py --no-viz --world-model-log --mode rl --learning --seed 42

# 2. Train the population world model + evolve genomes inside the dream
python scripts/dream_evolve.py --transitions "data/logs/transitions_*.csv" \
    --generations 20 --population 32

# 3. GROUND the dream champions in the real environment (mandatory —
#    dream fitness is a proxy and evolution exploits model errors)
python main.py --load-weights data/weights/dream_best.npz --no-viz --seed 42
```

Implementation: `agents/dream.py` (model + dream rollouts + (μ+λ) evolution)
and `scripts/dream_evolve.py` (CLI). The model predicts observation *deltas*, reward,
and episode termination from `(obs, action)`.

**Observation layout (72 features):**

| Range | Feature group | Description |
|-------|--------------|-------------|
| 0–7 | Agent state | Energy, age, direction one-hot, inventory space, metabolism |
| 8–57 | Vision 5×5×2 | Egocentric (agent-aligned; rotates with facing), type + value encoding per tile, terrain-layer aware |
| 58–65 | Stimulus | food_on_tile, seed_on_tile, food_ahead, resource_ahead, nearest_food_prox, food_dir_match, energy_urgency, can_interact |
| 66–71 | Inventory | Fullness, has_food, has_seed, has_fertilizer, total_calories, count |

### Fading Bootstrap Instincts

Newborn agents have random (or freshly mutated) weights, so without help most would
starve before evolution or learning can act. Instincts solve this: small additive
biases on the action logits that make survival-relevant actions more likely **when
they are contextually valid**. Crucially, every bias is multiplied by a strength
factor that **fades linearly from 1.0 at birth to 0.0 at `fade_age`** — so instincts
scaffold juveniles, then hand control entirely to the learned network. Any behaviour
you observe in an adult agent is genuinely produced by its evolved/learned weights.

| Condition | Action biased | Logit boost (× fade strength) |
|-----------|--------------|------------------------------|
| PICK_UP valid | PICK_UP | +1.5 |
| EAT valid | EAT | +1.0 |
| EAT valid **and hungry** | EAT | +3.0 × energy_urgency |
| USE valid | USE | +0.5 |
| Food nearby but not ahead | TURN_LEFT / TURN_RIGHT | +0.8 × proximity |

Configured in `config/default.yaml`:

```yaml
brain:
  instincts:
    enabled: true        # false = pure network from birth (ablation)
    fade_age: 150        # ticks until strength reaches 0 (null = never fade)
    hunger_eat_bias: 3.0 # EAT prior at maximum hunger
```

> **Note:** older versions *forced* an EAT action whenever energy dropped below 50%
> while food was held. That hardcoded override has been removed — the hunger-scaled
> EAT bias above is a strong prior the policy can still override, and it fades with
> age like everything else. Implementation: `agents/brain/instincts.py`.

## Core Concepts

### Primitive Actions Only
Agents can only perform basic actions:
- `MOVE_FORWARD` — Move one tile in facing direction
- `TURN_LEFT / TURN_RIGHT` — Change facing direction
- `PICK_UP` — Pick up objects from current tile
- `DROP` — Drop objects from inventory
- `EAT` — Consume edible objects from inventory
- `USE` — Plant seeds on current tile
- `WAIT` — Do nothing this tick

Actions are **masked** when invalid (e.g. `DROP` is masked in no-stacking mode when there is no legal drop location).

### Emergent Behaviors
Complex behaviors emerge from combinations of primitive actions:
- **Farming**: Agents learn to pick up seeds, plant them, and return to harvest
- **Tool Use**: Discovery and utilisation of beneficial objects
- **Cooperation**: Coordinated group activities
- **Communication**: Evolved signalling between agents

### Evolution System
- **Dual Mode**: `--mode rl` (gradient learning + Lamarckian weight inheritance) or `--mode neuroevolution` (pure evolution, no gradients). Set in config via `evolution.mode` or CLI.
- **Neural Networks**: Each agent has an evolved GRU Actor-Critic brain
- **Lamarckian Inheritance**: In RL mode, learned weights are synced back to the genome and passed to offspring (with mutation), so children inherit both genetic and learned knowledge
- **Genetic Algorithm**: Tournament selection, crossover, and mutation
- **Fitness-based Selection**: Survival and reproduction based on energy accumulation
- **Trait Inheritance**: Both neural weights and behavioural traits evolve

## Configuration

Key simulation parameters in `config/default.yaml` (see
[Modes & Feature Toggles](#modes--feature-toggles--complete-reference) for
the full enable/prerequisite/effect reference):
- **Evolution mode** (`evolution.mode`): `"rl"` or `"neuroevolution"`
- World size, terrain generation (soil, rock, water, sand ratios)
- Population size, genetic parameters
- Resource availability and growth rates
- Learning scheduler (interval, budget, adaptive mode)
- Visualisation and data collection settings

## World & Ecosystem Parameters

The world is a 2D grid of tiles (soil/rock/water/sand), each carrying
`fertility` and `moisture`. A pipeline of systems runs every tick to grow
plants, germinate seeds, decay food, cycle soil nutrients, and (optionally)
drive a day/night/seasonal/weather climate. Every parameter below lives in
`config/default.yaml`; deeper mechanics are documented in
[docs/ECOSYSTEM.md](docs/ECOSYSTEM.md).

### World & terrain (`world:`, `terrain:`)

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `world.width` / `world.height` | 100 / 100 | Grid dimensions in tiles |
| `world.initial_resources` | 50 | Objects scattered at world start |
| `world.resource_spawn_rate` | 0.01 | Safety-net berry spawn probability when food is depleted |
| `world.allow_stacking` | false | If false, one object per tile (overflow placed nearby) |
| `world.agents_visible` | false | (W4) Other living agents appear in vision as encoding 0.40 with their energy ratio (the P3 unblock) |
| `world.agent_collision` | false | (W4) A living agent blocks a tile, so space becomes contested |
| `social.transfer_enabled` | false | (W5) USE on a facing agent transfers the first inventory item (recipient must have space). Logged as `interaction_kind="give"` |
| `world.seed` | null | RNG seed (null = random each run) |
| `terrain.soil_ratio` | 0.695 | Fraction of plantable soil tiles |
| `terrain.rock_ratio` | 0.20 | Impassable rock (blocks movement) |
| `terrain.water_ratio` | 0.10 | Water (not plantable; feeds moisture to adjacent soil when the environment engine is on) |
| `terrain.sand_ratio` | 0.005 | Spreading hazard that slows growth/germination 10× |
| `terrain.fertility_range` | [0.3, 1.0] | Initial fertility drawn per soil tile |
| `terrain.moisture_range` | [0.2, 0.8] | Initial moisture drawn per soil tile |
| `terrain.generator` | legacy | `legacy` (uniform shuffle, flat) or `heightmap` (elevation: mountains, rivers, biomes, slope cost — see [mode #10](#10-terrain-generator--legacy-vs-heightmap)) |
| `terrain.heightmap.feature_scale` | 12 | Heightmap: coarse-noise cell size (bigger = smoother, broader features) |
| `terrain.heightmap.octaves` | 4 | Heightmap: layered noise detail levels |
| `terrain.heightmap.persistence` | 0.5 | Heightmap: amplitude falloff per octave (lower = smoother) |
| `terrain.heightmap.river_sources` | 6 | Heightmap: how many downhill rivers to trace from the peaks |
| `performance.spatial_index` | true | (W6a) Coarse-cell index of edible objects — speeds the nearest-food scans ~3.5× with identical results. Disable to A/B or debug |
| `performance.spatial_index_cell` | 8 | (W6a) Index cell edge in tiles (bigger = fatter buckets) |
| `reward.preset` | legacy | (W6c) `legacy` (full dense shaping) or `minimal` (eat/death/energy-delta only — the reward-diet ablation; pair with `learning.curiosity`) |

### Plants & seeds (`plants:`)

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `growth_time` | 50 | Ticks a seed must sit in soil before it can germinate |
| `mature_age` | 100 | Plant age at which it starts producing berries |
| `max_age` | 500 | Plant age at death (returns fertility to soil) |
| `seed_spawn_rate` | 0.1 | Per-tick chance a mature plant spawns a berry |
| `required_fertility` | 0.3 | Minimum tile fertility for germination |
| `required_moisture` | 0.2 | Minimum tile moisture for germination |
| `germination_success_rate` | 0.75 | Probability a ready seed germinates (× temperature window when the environment engine is on) |
| `fertility_consumption_per_tick` | 0.001 | Fertility a plant draws from its tile |
| `moisture_consumption_per_tick` | 0.0005 | Moisture a plant draws from its tile |
| `seed_max_age` | 200 | Ticks before an ungerminated seed rots away |
| `max_neighbor_plants` | 3 | **Carrying capacity (B5):** a seed will not germinate if this many plants are already within `neighbor_radius`. `0` disables it (legacy unbounded growth) |
| `neighbor_radius` | 2 | Chebyshev radius of the crowding window (2 = a 5×5 area) |

> **Why the carrying capacity matters:** without it, each mature plant
> produces ~20 offspring over its life, so plants — and the berries they
> spawn — accumulate until they tile the world (a small world saturates at
> ~65% plant coverage). With the defaults, plant coverage plateaus flat at
> ~24% of plantable tiles while food stays abundant.

### Resources & soil (`resources:`, `soil:`)

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `resources.berry_calories` | 20.0 | Energy a fresh berry gives when eaten (× freshness) |
| `resources.berry_freshness_decay` | 0.01 | Freshness lost per tick (≈100-tick shelf life) |
| `resources.seed_drop_chance` | 0.7 | Chance a fully decayed berry drops a seed |
| `soil.fertility_recovery_rate` | 0.0005 | Fertility regained per tick on empty soil |
| `soil.fertility_return_on_death` | 0.15 | Fertility returned when a plant/berry decomposes |
| `soil.moisture_evaporation_rate` | 0.0002 | Legacy moisture loss/tick (**superseded** by the environment engine when enabled) |
| `soil.moisture_recovery_rate` | 0.0008 | Legacy moisture gain/tick (the old "rain" — this was bug B1; **superseded** when the environment engine is enabled) |

### Environment engine (`environment:`) — opt-in (W1)

Off by default; when `enabled: true` it computes a global climate each tick
and feeds plain multipliers to the systems above. See
[mode #9](#9-environment-engine--daynight-seasons-weather) for the full
effect description.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `enabled` | false | Master switch (false = legacy static climate, bit-compatible) |
| `day_length` | 200 | Ticks per day/night cycle (drives `light`) |
| `min_light` | 0.25 | Light floor at deepest night (1.0 = noon) |
| `season_length` | 2000 | Ticks per seasonal cycle (drives the temperature baseline) |
| `season_temp_amplitude` | 0.25 | Seasonal temperature swing around `base_temperature` |
| `base_temperature` | 0.5 | Yearly mean temperature (0.5 = centre of the comfort band) |
| `daynight_temp_amplitude` | 0.10 | Day-vs-night temperature swing |
| `metabolism_temp_coef` | 0.5 | Extra agent energy drain at temperature extremes |
| `base_evaporation` | 0.0012 | Soil moisture loss/tick, scaled by temperature & light |
| `water_adjacency_recovery` | 0.002 | Moisture gain/tick on soil next to water tiles |
| `weather.rain_start_chance` | 0.01 | Per-tick chance rain begins (rain is the only moisture recovery) |
| `weather.rain_duration` | 60 | Length of a rain event |
| `weather.rain_recovery` | 0.004 | Moisture gain/tick while raining |
| `weather.drought_start_chance` | 0.002 | Per-tick chance a drought begins |
| `weather.drought_duration` | 150 | Length of a drought |
| `weather.drought_evaporation_factor` | 2.0 | Evaporation multiplier during drought |

### Sand hazard (`sand:`) — W0 / B2

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `germination_multiplier` | 0.1 | Germination is 10× harder on sand |
| `growth_multiplier` | 0.1 | Plant growth is 10× slower on sand |
| `spawn_rate_multiplier` | 0.3 | 70% less food production on sand |
| `fertility_override` | 0.30 | Clamps tile fertility on sand (B2: at the seed threshold, so the ×0.1 multiplier is the real difficulty) |
| `moisture_override` | 0.20 | Clamps tile moisture on sand (B2: at the seed threshold) |
| `spread_interval` / `spread_chance` / `spread_radius` | 200 / 0.05 / 1 | How fast unprotected soil turns to sand |
| `spread_blocked_by` | [plant] | Categories that stop sand spreading |
| `reclaim_terrain` / `reclaim_interval` | soil / 150 | A plant sitting on sand reclaims it back to soil |

### Population pressure (`reproduction:`, `calamity:`)

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `reproduction.enabled` | false | Enable in-simulation asexual reproduction |
| `reproduction.energy_threshold` | 0.60 | Fraction of max energy a parent needs to reproduce |
| `reproduction.energy_split` | 0.40 | Fraction of energy the parent loses to the offspring |
| `reproduction.min_age` / `cooldown_ticks` | 100 / 50 | Earliest reproduction age / ticks between births |
| `reproduction.mutation_std` | 0.02 | Gaussian mutation applied to offspring weights |
| `reproduction.max_population` | 100 | Hard cap on agent count (null = unlimited) |
| `calamity.enabled` | true | Enable periodic resource-destroying disasters |
| `calamity.interval` | 500 | Ticks between calamities |
| `calamity.destruction_rate` | 0.70 | Fraction of targeted objects destroyed each event |
| `calamity.affect_plants` / `affect_food` / `affect_seeds` | true / true / false | What a calamity destroys (seeds preserved by default for recovery) |

### Wildfire (`fire:`) — W3, opt-in

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `fire.enabled` | false | Master switch (no-op when off) |
| `fire.ignite_chance` | 0.0004 | Base per-tick chance a hot, dry plant ignites |
| `fire.spread_chance` | 0.30 | Per-tick chance a burning plant ignites an adjacent plant |
| `fire.burn_duration` | 6 | Ticks a plant burns before it is consumed |
| `fire.moisture_threshold` | 0.4 | Tile moisture above this is a firebreak (won't ignite/spread) |
| `fire.nutrient_return` | 0.25 | Fertility (ash) returned to a burned tile |

> Ignition/spread scale with heat (`environment.temperature`) and dryness
> (tile moisture), so fire needs the environment engine (or a warm constant)
> to be common. See [mode #11](#11-wildfire--fireenabled).

### Edible / toxicity (per-object, W3)

Edible objects (built-ins and custom) carry `edible.calories`,
`edible.toxicity`, and `edible.freshness`. Eating yields **net energy =
`calories × freshness − toxicity × freshness × 30`** — so a toxic food can be
net-negative. Set `toxicity` on any food (e.g. in a `--objects` pack) to
create a discrimination task. See [mode #12](#12-ecology-species-pack--objects-configecologyyaml).

## Custom Objects

Create new world objects entirely from YAML — no code changes required.
See `config/custom_objects.yaml` for a full reference with worked examples.

```yaml
# config/my_objects.yaml
objects:
  superfood:
    display_name: "Superfood"
    category: "food"
    edible:
      calories: 80.0
      toxicity: 0.0
      freshness: 1.0
    physics:
      decay_rate: 0.05
    interaction:
      pickable: true
    observation:
      vision_encoding: 0.95
      value_source: "freshness"
```

```bash
python main.py --gui --objects config/my_objects.yaml
```

### Object Components

| Component | Purpose |
|-----------|---------|
| `edible` | Food with calories, toxicity, freshness decay |
| `seed` | Plantable seed that grows into a plant |
| `plant` | Grows, matures, produces resources |
| `fertilizer` | Boosts soil fertility in a radius |
| `interaction` | Controls pickable, usable, passable, blocks_growth |
| `tile_effect` | Environmental multipliers, spreading, terrain conversion |
| `physics` | Decay, decomposition, nutrient return |
| `observation` | How agents perceive the object |

### Tile Effect System

Objects with `tile_effect` modify the environment around them:

- **Growth/germination multipliers** — sand uses 0.1 (10× slower)
- **Spawn rate multiplier** — reduces food production on affected tiles
- **Spreading** — converts neighbouring soil tiles over time
- **Blocked by** — plants can prevent spread (trees block sand)
- **Terrain conversion** — permanently changes tile type (sand, rock, etc.)
- **Fertility/moisture override** — clamps tile values

Built-in terrain hazard: **Sand** (5 % of default map)
- 10× harder germination, 10× slower growth, 70 % less food
- Spreads to adjacent soil every 200 ticks if no plant is nearby

## Data Analysis

### Action-Distribution Analysis
```bash
python scripts/analyze_logs.py                       # latest log in data/logs
python scripts/analyze_logs.py --file path/to.csv    # a specific log
python scripts/analyze_logs.py --fade-age 150        # match brain.instincts.fade_age
```
Works on both log schemas (`agent_actions_*.csv` from `--log` and
`transitions_*.csv` from `--world-model-log`). Reports action distribution,
success rates, energy economy, population dynamics, lifespans, spatial
coverage, farming pipeline, behavioural diversity, action n-grams, temporal
phases — plus two Brain-v3-era sections:

- **🍼 Instinct fade phases** — juvenile (age < fade_age) vs adult behaviour
  split: do agents still eat/forage once the instinct scaffold is gone?
  (the emergence-first claim, measured per run)
- **💀 Death analysis** — death reasons and age-at-death distribution
  (transitions format; e.g. a starvation spike just past fade_age means
  agents aren't learning to eat before the training wheels come off)
- **🍽️ Per-species consumption** (W3) — what gets eaten, with net energy and a
  toxic-food flag, so species preference / poison-avoidance is measurable
- **🛰️ Social / SIGNAL** (Brain v3.5 / W4) — SIGNAL usage rate, **signal
  entropy** (shared behaviour vs specialists), and an **agent-proximity
  response** (action mix bucketed by how close other agents are) — does
  signalling rise in company?
- **🧬 Society / Roles** (W5) — **role-entropy** (normalised Shannon over
  agents' dominant actions: division of labor), **behavioural novelty**
  (mean pairwise Jensen-Shannon divergence between agent action
  distributions), **territory** (per-agent bbox + mean Jaccard overlap),
  and **trade** counts (gives, givers, recipient sites, give/signal ratio).

### Observation Sensitivity
```bash
python scripts/analyze_observation_sensitivity.py                      # v2 brain
python scripts/analyze_observation_sensitivity.py --brain 3            # attention brain
python scripts/analyze_observation_sensitivity.py --brain 3 --world-model
```
Perturbs each of the 72 observation features and ranks how much the policy
and value change — in three views: RAW network, RUNTIME JUVENILE (mask +
full-strength instincts), and RUNTIME ADULT (instincts fully faded, the
pure network that governs adult behaviour).

### CSV Logging

```bash
python main.py --gui --log
python main.py --gui --log --log-dir data/logs/run1 --log-frequency 10
```

**Generated files:**
- `agent_actions_*.csv` — every action taken with result, energy, position
- `agent_states_*.csv` — per-tick agent state snapshots
- World-model transition logs (`transitions_*.csv`, `world_states_*.csv`, `episodes_*.csv`)

```python
import pandas as pd
actions = pd.read_csv("data/logs/agent_actions_20260219.csv")
print(actions['action'].value_counts(normalize=True))
```

## Performance

Key optimisations applied to maintain high FPS:

- **Per-tick caching** — `world.get_cached_object_counts()` / `get_cached_soil_stats()` computed once per tick
- **`is_terrain` flag** — cached on `WorldObject` at creation; replaces repeated registry lookups
- **Set-based tile index** — `Tile.object_ids` is a `set` (O(1) membership, no duplicates)
- **Persistent log handles** — `WorldModelLogger` keeps file handles open across ticks; flushes every 50 writes
- **GPU renderer caching** — object/agent instance lists rebuilt only when `world.tick` changes
- **Learning scheduler** — staggered training slots, capped budget per tick, adaptive scheduling

## Visualization

- **Pygame renderer** — real-time 2D orthographic view with HUD and tile inspector
- **GPU isometric renderer** (`--gpu`) — ModernGL 2.5D isometric view, GLSL shaders, frustum culling
- **Data analysis** — action-distribution script with target comparison

## Research Applications

This sandbox is designed for studying:
- **Artificial Life**: How complex behaviours emerge from simple rules
- **Evolutionary Computation**: Natural selection in neural network populations
- **Multi-agent Systems**: Coordination and communication without central control
- **Online RL**: Actor-Critic agents learning in a non-stationary multi-agent environment

## Contributing

1. Follow the development guidelines in `docs/guideline.md`
2. All functions must have proper docstrings
3. Focus on emergent behaviours, never hardcode high-level strategies
4. Maintain separation between world physics and agent logic
5. Write comprehensive tests — run `pytest` before opening a PR (320+ tests, all must pass)

## Future Extensions

- **Communication Evolution**: Emergent language and signalling via SIGNAL action
- **Tool Construction**: Building and using complex tools
- **Environmental Dynamics**: Day/night cycles, seasons, temperature, and weather
- **Speciation (NEAT-style)**: Species-level diversity protection
- **Recurrent population world model**: GRU/sequence version of the dream
  model for partially observable dynamics; dream curricula

See [SUGGESTIONS.md](docs/SUGGESTIONS.md) for the full 80+ item roadmap.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this work in research, please cite:

```
Vasa, K. (2025). Emergent World-Model Sandbox: Evolution of Complex Behaviors
in Artificial Life Simulations. GitHub repository.
```

---

*This project demonstrates how complex, intelligent behaviours can emerge from simple rules and evolutionary pressure, without any explicit programming of high-level strategies.*
