# Suggestions & Roadmap

**Author:** Karan Vasa  
**Date:** February 26, 2026  
**Current State:** v3.2 — GRU Actor-Critic agents with **dual-mode evolution** (RL or pure neuroevolution, selectable via `--mode` / config), Lamarckian weight inheritance, reproduction, calamities, and balanced turn/exploration behavior.  
**Vision:** Two-track platform — (1) **Robotics Learning** (sim-to-real controller training) and (2) **Online Simulated World** (entertainment, games, social sandbox) — both powered by the same neuroevolution + world model engine.

---

## Current Metrics (Latest Run)

| Metric | Value | Target | Status |
|---|---|---|---|
| TURN_LEFT | 20.8% | ~equal to R | ✅ |
| TURN_RIGHT | 22.0% | ~equal to L | ✅ |
| WAIT | 19.0% | 32-35% | ⚠️ Low |
| Survival | 3358 ticks | >1500 | ✅ |
| EAT success | 100% | >90% | ✅ |
| Population | 423 agents | — | Healthy |
| Return cycles | 12.48% | <15% | ✅ |
| Revisits | 40.9% | <45% | ✅ |

---

> **How to read this document:** Parts 1–7 build the sandbox's core capabilities — each section is annotated with how it feeds into the two ultimate tracks (Robotics & Entertainment). Parts 8–9 define the end-state platforms. The suggested progression (Phases A–M) interleaves both tracks so the entertainment and robotics systems grow together.

---

## Part 1: Immediate Tuning & Polish

Quick wins that require minimal code changes. These tunings establish a **stable behavioral baseline** — essential before exposing agents to online arena play (Entertainment Track) or transferring evolved controllers to physical hardware (Robotics Track). A well-calibrated sandbox is the foundation both tracks are built on.

- [ ] **Tune WAIT back to 32-35%** — Reduce exploration movement bonus from +0.03 to +0.01, lower new-tile bonus from +0.04 to +0.02. The turn-balance fixes made agents more active, overshooting the movement target.
- [ ] **Equalize turn energy cost** — Turns cost 0.24 vs MOVE 0.20. Matching them (e.g. both 0.20) removes residual forward bias without breaking balance.
- [ ] **Add per-generation metrics CSV** — Log generation-level stats (avg fitness, avg survival, action distribution, population size) alongside the per-action log for easier trend analysis.
- [ ] **Configurable reward shaping** — Move the hardcoded reward magnitudes (exploration +0.03, backtrack -0.10, food proximity scaling, etc.) into `config/default.yaml` so tuning doesn't require code edits.
- [ ] **Observation sensitivity dashboard** — Wrap `scripts/analyze_observation_sensitivity.py` into a CLI that reads the latest log and prints a ranked feature importance table automatically.

---

## Part 2: Architecture & Learning Improvements

Deeper changes to make agents smarter. The brain architecture designed here becomes the **deployable controller** for physical robots (compact enough for embedded hardware at ~8K parameters) and the **scalable agent brain** for thousands of concurrent creatures in the online world. Every improvement here accelerates both the Robotics and Entertainment tracks.

### 2.1 Brain Architecture

- [x] **Attention over vision grid** — DONE (Brain v3, `brain.version: 3`): 25 tile tokens + positional encoding, shared 4×8 embedding, single-head attention pool with the query derived from agent state. See `agents/brain/v3.py` and BRAIN_V2_V3_COMPARISON.md §4. *→ Scales to camera-like sensors for robotics and enables larger vision radii for the online world.*
- [x] **Larger GRU / stacked GRU** — DONE (partially): Brain v3 uses GRU(48) (configurable via `brain.v3.gru_hidden_size`). Stacked (2-layer) GRU remains open.
- [x] **Separate value head architecture** — DONE (Brain v3): the value MLP reads `[z, h]` — the current latent plus memory — so the critic no longer depends solely on what the GRU stored.
- [x] **Curiosity-driven exploration (ICM / RND)** — DONE (Phase 4, `learning.curiosity` + `brain.world_model`): the latent dynamics head's prediction error, z-score-normalised and clipped, is the intrinsic reward. See `agents/curiosity.py`.

### 2.2 Learning System

- [x] **Dual-mode evolution** — Added `--mode rl` / `--mode neuroevolution` CLI flag and `evolution.mode` config key. RL mode enables online Actor-Critic learning with Lamarckian inheritance (learned weights synced to genome and passed to offspring + mutation). Neuroevolution mode disables all gradient learning — agents act purely from evolved genome weights + instincts. Legacy `--learning` flag still works.
- [x] **GAE (Generalized Advantage Estimation)** — DONE (PPO learner, `learning.ppo.gae_lambda`). See `agents/ppo.py:compute_gae`.
- [x] **PPO-style clipped updates** — DONE (`learning.algorithm: ppo`): clipped surrogate (ε configurable), full-network backprop via a persistent torch mirror, sequence replay with truncated BPTT, Adam + grad-norm clipping. A2C remains the default control.
- [ ] **Curriculum learning** — Start agents in a resource-rich, small world and progressively increase difficulty (larger world, fewer resources, stronger calamities). Configurable schedule in YAML. *→ Foundation for robotics task curriculum (Part 8.5) and entertainment difficulty progression (survival gauntlet, Part 9.2).*
- [ ] **Population-level knowledge distillation** — Periodically average weights of top-N agents and inject the averaged weights into the lowest-performing agents, accelerating convergence. *→ Accelerates robotics training convergence and creates visible "teaching" events for online spectators.*
- [ ] **Hindsight experience replay** — When an agent dies near food, rewrite the trajectory reward as if it had eaten, providing learning signal for "almost successful" behaviors. *→ Sample-efficient learning critical for expensive robotics physics simulations.*

### 2.3 Evolution

- [x] **Lamarckian weight inheritance verified** — Confirmed that `_sync_genome_weights()` is called at the end of every `learn()` call, writing brain params back to `genome.weights`. On reproduction, `clone_agent()` deep-copies the genome (including synced weights) and applies mutation. Offspring reliably inherit learned weights.
- [ ] **Speciation (NEAT-style)** — Group agents by genome similarity and enforce mating within species. Prevents premature convergence and maintains behavioral diversity. *→ Essential for rich online world dynamics and diverse robotics solution exploration.*
- [ ] **Adaptive mutation rate** — Agents with stagnant fitness increase their mutation rate automatically, while successful lineages stay conservative. *→ Prevents evolutionary stagnation in always-on entertainment worlds and robotics training plateaus.*
- [ ] **Trait-linked behaviors** — Let genome traits (metabolism, vision) influence brain architecture (e.g. metabolism_rate scales energy-urgency input gain, vision_radius controls observation grid size).
- [ ] **Sexual reproduction** — Current reproduction is asexual fission. Add mate selection where agents choose partners based on fitness/proximity, enabling richer evolutionary dynamics. *→ Creates mating events for entertainment spectacle and enables the creature designer's breeding system (Part 9.1).*

---

## Part 3: World & Ecosystem Expansion

Make the world more complex and interesting. This is the **content engine** for the Entertainment Track — richer ecosystems create the emergent drama that keeps users watching, competing, and streaming. For the Robotics Track, ecosystem complexity provides diverse training environments that produce more robust and adaptable controllers.

### 3.1 Environment

- [ ] **Day/night cycle** — Global light level oscillates over N ticks. Agents get a `time_of_day` observation feature. Food spawns slower at night; energy cost rises. Encourages temporal strategies (forage by day, rest at night).
- [ ] **Weather system** — Rain events increase moisture globally, drought events decrease it. Affects plant growth and food availability. Adds another survival pressure dimension.
- [ ] **Biomes** — Cluster terrain types into biomes (forest = high fertility/moisture, desert = sand-heavy, wetland = water-heavy) with transition zones. Different biomes have different resource densities.
- [ ] **Predator objects / hazard tiles** — Add tiles or objects that damage agents on contact (lava, thorns, predators). Forces agents to learn avoidance alongside food-seeking.
- [ ] **Movable objects** — Let agents push rocks or build simple walls. Enables tool-use and environmental modification behaviors.

### 3.2 Agent Capabilities

- [ ] **Communication action** — Add a SIGNAL action that emits a value visible to nearby agents in their observation. Enables emergent proto-language for cooperation (e.g., signaling food location). *→ Foundation for online world social dynamics (Part 9.4) and multi-robot coordination for robotics teams.*
- [ ] **Trading / sharing** — Agents can transfer inventory items to adjacent agents. Opens cooperation and specialization (forager + planter roles). *→ Core mechanic for the online ecosystem economy (Part 9.3) and cooperative robotics task-sharing.*
- [ ] **Memory markers** — Agents can DROP a "marker" object (zero-cost) that other agents can see in their vision. Persistent spatial communication.
- [ ] **Multi-step actions** — Complex actions like "build shelter" that take multiple ticks and provide lasting benefits (reduced energy drain). Requires planning. *→ Extends to multi-step robotic manipulation tasks (pick-place-assemble) and entertainment crafting mechanics.*
- [ ] **Age-based trait expression** — Young agents have higher exploration drive (higher entropy), old agents become more exploitative. Simulates natural behavioral development.

### 3.3 Social Dynamics

- [ ] **Kin recognition** — Agents can detect lineage similarity in nearby agents via observation. Enables kin selection behaviors (helping relatives). *→ Drives clan-like behavior that maps directly to the entertainment clan system (Part 9.4).*
- [ ] **Territory marking** — Agents that stay in an area leave "scent" that decays. Other agents can detect territory density and avoid or contest it. *→ Creates visible territorial dynamics for spectator entertainment and spatial strategy emergence.*
- [ ] **Group fitness bonus** — Agents near kin get a small energy bonus, encouraging cluster formation and cooperative survival strategies. *→ Foundation for co-op boss mode (Part 9.2) and multi-robot cooperative tasks.*

---

## Part 4: Tooling & Infrastructure

Improve the development and analysis workflow. This section builds the **backbone of both platforms** — the live dashboard evolves into the spectator stats overlay (Entertainment) and training monitor (Robotics), distributed simulation powers biome servers and multi-user training jobs, and checkpointing enables the persistent online world.

### 4.1 Analysis & Visualization

- [ ] **Live training dashboard** — Real-time web dashboard (e.g., Streamlit or Flask) showing population curves, action distributions, fitness over time, and a mini-map of agent positions. *→ Evolves into the robotics training dashboard (Part 8.7) and entertainment spectator stats overlay (Part 9.1).*
- [ ] **Trajectory replay** — Save agent paths and replay them in the GUI (scrub forward/backward). Essential for debugging loop behavior and understanding emergent strategies. *→ Becomes the replay & highlights system for entertainment (Part 9.4) and robotics episode analysis.*
- [ ] **Heatmap overlay** — GUI toggle that colors tiles by agent visit frequency, food density, or death locations. Reveals spatial patterns instantly. *→ Feeds into the spectator mode's visual overlays and robotics environment analysis.*
- [ ] **Lineage tree visualization** — Graph showing parent-child relationships across generations with fitness annotations. Identifies successful evolutionary branches. *→ Core of the adopt-an-agent family tree (Part 9.1) and creature wiki (Part 9.4).*
- [ ] **A/B experiment framework** — Script that runs two configurations side-by-side with different seeds and produces a comparison report (metrics table + significance tests). *→ Enables automated hyperparameter search for robotics tasks and balance testing for entertainment game modes.*

### 4.2 Performance & Scalability

- [ ] **GPU-accelerated learning** — Full torch backend for batch PPO updates across all agents simultaneously. Currently learning is per-agent sequential. *→ Required for scalable robotics training and running 1000+ agents on entertainment servers.*
- [ ] **Spatial hashing for world queries** — Replace O(n²) food scans in `_find_nearest_food_distance` and `_compute_food_dir_match` with a grid-based spatial hash for O(1) neighborhood lookups.
- [ ] **C extension for world update** — The core tile update loop (`systems.py`) could be a Cython/C module for 5-10× speedup on large worlds.
- [ ] **Distributed simulation** — Run multiple independent worlds in parallel (multiprocessing) with periodic weight migration between populations. Island model evolution. *→ Foundation for entertainment biome servers (Part 9.3) and multi-user robotics training jobs (Part 8.7).*
- [ ] **Checkpointing** — Save/load full simulation state (world + all agents + RNG state) for deterministic resume. Currently only weights are saved. *→ Required for the persistent online world (Part 9.3) and resumable robotics training runs.*

### 4.3 Testing & CI

- [ ] **Behavioral regression tests** — Run a fixed-seed 500-tick simulation and assert action distribution stays within expected bands. Catches unintentional balance regressions. *→ Prevents entertainment balance-breaking patches and ensures robotics controller stability across updates.*
- [ ] **Property-based testing** — Use Hypothesis to generate random configs/world states and verify invariants (energy conservation, population bounds, no duplicate object IDs).
- [ ] **Performance benchmarks** — Tracked benchmark suite that measures ticks/second at various population sizes. Alerts on regressions.
- [ ] **CI pipeline** — GitHub Actions running pytest + benchmarks on push. Blocks merge on test failures. *→ Essential for multi-contributor platform development across both tracks.*

---

## Part 5: Research Directions

Longer-term explorations that could produce publishable results. The **world model** (Section 5.1) is the single most important technology for the Robotics Track — it enables 100–1000× faster training via "dreaming." **Emergent complexity** (Section 5.2) IS the entertainment value proposition — a world that never stops producing novel behaviors. **Interpretability** (Section 5.3) feeds both the creature wiki for users and debuggable robot controllers for engineers.

### 5.1 World Models

- [x] **Transition predictor** — DONE (Phase 4, `brain.world_model`): a latent dynamics head in the genome predicts (next latent, reward) from (hidden state, action); prediction error is the curiosity reward (`learning.curiosity`). Implemented in latent space (sharing the policy encoder) rather than observation space — see BRAIN_V3_PROPOSAL.md §3.1. *→ THE core technology for the Robotics Track — this becomes the full world model (Part 8.4) that enables dream-based evolution.*
- [x] **Dreamer-style imagination** — DONE (first version, `brain.world_model.planner`): random-shooting rollouts in latent space (imagine ẑ', advance the GRU, accumulate r̂, bootstrap with the critic) select the agent's next action. See `agents/planner.py`. Population-level offline model + dream-based evolution also DONE: `scripts/dream_evolve.py` / `agents/dream.py` train an observation-space model from transition logs and evolve genomes inside it ((μ+λ) selection, mandatory real-world grounding documented).
- [ ] **Model-based rollouts** — Before acting, simulate K steps in the learned model and pick the action sequence with highest expected return. Adds planning depth without environment cost.

### 5.2 Emergent Complexity

- [ ] **Open-ended evolution** — Measure "interestingness" of evolved behaviors over very long runs (10K+ generations). Track behavioral novelty metrics to see if the system keeps producing new strategies or converges. *→ THE entertainment value proposition — a world that never gets boring because it continuously invents new behaviors.*
- [ ] **Division of labor** — With communication + trading, observe whether agents spontaneously specialize into distinct roles (forager, planter, scout). Quantify with role entropy metrics. *→ Visible role emergence drives user engagement in the online world and inspires multi-robot specialization for robotics teams.*
- [ ] **Cultural transmission** — If weight distillation (Part 2.2) enables "teaching", track whether behavioral innovations spread through the population faster than genetic inheritance alone. *→ "Cultural events" become shareable content moments for entertainment streaming (Part 9.4).*
- [ ] **Arms race dynamics** — With predator agents or competing populations, observe co-evolutionary dynamics where both sides continuously adapt.

### 5.3 Interpretability

- [ ] **GRU hidden state analysis** — PCA/t-SNE on hidden states across many agents to find behavioral clusters. Does the GRU naturally encode "hungry", "exploring", "returning to food"? *→ Powers the auto-generated creature wiki (Part 9.4) behavioral profiles and robot controller debugging.*
- [ ] **Policy distillation to decision trees** — Train a decision tree to mimic the neural policy. Produces human-readable rules like "IF food_ahead AND energy < 50% THEN MOVE_FORWARD". *→ Makes creature behaviors explainable for entertainment users and robot controllers debuggable for engineers.*
- [ ] **Causal intervention studies** — Clamp specific observation features and measure behavioral change. Extends the sensitivity analysis to causal claims.
- [ ] **Genome-behavior mapping** — Correlate genome weight regions with behavioral traits. Which weight clusters control exploration vs. exploitation? *→ Enables the genome marketplace (Part 9.2) to show meaningful trait labels and robotics engineers to understand evolved controllers.*

---

## Part 6: World Physics Additions

New physical systems that integrate with the existing tile/object/system architecture in `world/systems.py` and `world/tiles.py`. These systems serve dual purpose: they create **differentiated biome content** for the Entertainment Track's server types (jungle, desert, volcanic, ocean — Part 9.3) and provide **diverse, time-varying environments** that produce robust robotics controllers via natural curriculum learning.

### 6.1 Light & Temperature

- [ ] **Day/night cycle** — Add a global `world.time_of_day` float (0.0–1.0) that oscillates with configurable period (e.g. 200 ticks = 1 day). Expose as an observation feature (index slot available in agent state). Effects:
  - Plant growth rate × light_level (plants don't grow at night)
  - Food spawning rate × light_level
  - Agent vision radius shrinks at night (e.g. 5 → 3 tiles)
  - New config section `daynight:` with `cycle_length`, `min_light`, `max_light`
  - *→ Biome differentiation content (Part 9.3) and time-varying environments for robust robotics controllers.*
- [ ] **Temperature per tile** — Each tile gets a `temperature` float (0.0–1.0). Water tiles radiate cold, rock tiles absorb and radiate heat. Temperature affects:
  - Plant growth speed (optimal range 0.3–0.7, dies outside 0.1–0.9)
  - Agent metabolism rate (cold = faster energy drain, hot = faster too, mild = baseline)
  - Seed germination success (temperature-dependent probability modifier)
- [ ] **Seasons** — Slow sinusoidal modulation of global temperature and moisture over ~2000 ticks. Spring: high moisture + rising temp. Summer: peak temp + low moisture. Autumn: dropping temp + seed drop bonus. Winter: low temp + minimal growth. Configurable in YAML.
- [ ] **Fire propagation** — Low-moisture, high-temperature tiles with plants can catch fire (probabilistic). Fire spreads to adjacent plant tiles, destroys plants, returns nutrients to soil, and increases temperature locally. Water tiles act as firebreaks. New `FireSystem` in `systems.py`. *→ Dramatic events for spectator mode and hazard-avoidance training for robotics.*

### 6.2 Fluid & Soil Dynamics

- [ ] **Water flow** — Water tiles spread moisture to adjacent soil tiles via gradient diffusion each tick (moisture flows from high to low). Creates natural "river" moisture corridors. Rate configurable in `soil:` config section.
- [ ] **Erosion** — High-moisture soil tiles adjacent to water gradually lose fertility. Rock tiles adjacent to water slowly erode into sand over very long timescales. Adds geological change.
- [ ] **Nutrient diffusion** — When a plant dies and returns nutrients, fertility spreads to a 3×3 area (diminishing with distance) instead of only the death tile. Creates natural "fertile patches".
- [ ] **Flood events** — Periodic water level rise (new calamity type) that temporarily makes water-adjacent soil tiles impassable and destroys objects there. Configurable as a calamity subtype. *→ Scheduled world events for entertainment (Part 9.3) and dynamic terrain challenges for robotics curriculum.*
- [ ] **Aquifer system** — Underground moisture layer that tiles draw from. Deep soil tiles have higher baseline moisture restoration. Could add a `depth` property to tiles.

### 6.3 Ecology & Biodiversity

- [ ] **Multiple plant species** — Currently one plant type. Add 2–3 species with different growth rates, seed drop rates, calorie values, and soil requirements. Encoded as different `type_id` values in the object registry. Agents can learn preferences. *→ Biodiversity content for online world biomes and diverse reward landscapes for evolution.*
- [ ] **Pollination** — Plants within N tiles of another mature plant of the same species produce seeds faster. Encourages biodiversity clustering. Implemented as a proximity check in `PlantGrowthSystem`.
- [ ] **Decomposition chain** — Dead plants → organic matter object (new type) → slowly converts to fertility. Adds an intermediate step to nutrient cycling. Organic matter visible in agent vision.
- [ ] **Fungal network** — Tiles with plants connected to other plant tiles (within radius 3) share nutrients at a slow rate. Creates underground resource sharing. Visualized as subtle connections in GPU renderer.
- [ ] **Invasive species** — A fast-growing, low-calorie plant type that crowds out native plants by germinating faster. Creates ecological competition that agents must navigate. *→ Dynamic ecosystem content that keeps the online world unpredictable for users.*

### 6.4 Physics Interactions

- [ ] **Wind system** — Global wind direction vector that rotates slowly. Affects seed dispersal (seeds "blow" in wind direction when dropped), fire spread direction, and sand spread bias. Agent observation gets a `wind_dir` feature.
- [ ] **Gravity on slopes** — Add an `elevation` float to tiles (generated via Perlin noise). Objects on high tiles roll downhill probabilistically. Agents spend more energy moving uphill. Elevation visible in isometric renderer. *→ Precursor to 3D terrain for robotics locomotion training (Part 8.1) and visual depth for entertainment.*
- [ ] **Decay acceleration** — Food on hot tiles or wet tiles decays faster. Food on cold tiles decays slower. Links the temperature and moisture systems to the existing `DecaySystem`.
- [ ] **Rock weathering** — Rock tiles at high moisture slowly convert to sand, then to soil over very long timescales. Creates slow geological evolution of the world map.

---

## Part 7: UI & Renderer Upgrades

Improvements to both the Pygame 2D renderer (`pygame_renderer.py`) and the ModernGL isometric renderer (`gpu_renderer.py`). These features directly evolve into the **browser-based spectator client** (Entertainment Track — camera following, overlays, speed control, smooth visuals) and the **training visualization dashboard** (Robotics Track — live fitness graphs, trajectory replay, heatmaps). Every render feature built here is reused in Phases G–I.

### 7.1 Information Displays (Both Renderers)

- [ ] **Population graph panel** — Bottom-right panel showing a live line chart of population count over the last 500 ticks. Uses a rolling buffer of values. Renders as a simple polyline on a small surface. Toggle with `P` key.
- [ ] **Action distribution bar chart** — Small horizontal bar chart in the HUD showing the current-tick action breakdown (MOVE/TURN/WAIT/EAT/etc.) with color-coded bars. Auto-updates each tick. Toggle with `D` key.
- [ ] **Energy histogram** — HUD panel showing distribution of agent energy levels as a mini histogram (10 bins, 0–max_energy). Reveals whether agents are collectively healthy or starving. Toggle with `E` key.
- [ ] **Generation counter and fitness** — Display current generation number, best fitness this generation, and average fitness in the main stats panel. Already tracked internally, just needs HUD rendering.
- [ ] **Agent trail visualization** — Toggle (`T` key) that draws the last 20 positions of each agent as fading dots. Instantly reveals looping, backtracking, and exploration patterns without needing post-run analysis. *→ Spectator mode visualization for tracking adopted agents and debugging robotics trajectories.*
- [ ] **Death markers** — Show a small `×` at positions where agents died for the last 100 ticks. Fades over time. Reveals danger zones (resource deserts, map edges). *→ Danger-zone visualization for spectators and robotics environment hazard analysis.*

### 7.2 Overlays & Heatmaps

- [ ] **Fertility heatmap** — Keyboard toggle (`F` key) that overlays tile fertility as a green-to-brown gradient. Helps visually identify fertile farming zones.
- [ ] **Moisture heatmap** — Keyboard toggle (`M` key) that overlays moisture as a blue-to-yellow gradient. Shows hydration landscape.
- [ ] **Visit frequency heatmap** — Keyboard toggle (`V` key) that tracks how often each tile is visited by any agent and renders as a cool-to-hot gradient. Reveals agent spatial coverage and territory formation. *→ Territory visualization for entertainment spectators and robotics coverage analysis.*
- [ ] **Food density heatmap** — Keyboard toggle that shows food object count per 5×5 area as a density overlay. Reveals resource clustering and depletion zones.
- [ ] **Temperature overlay** — If temperature physics is added (Part 6.1), visualize it as a red-to-blue gradient overlay.

### 7.3 Interaction & Controls

- [ ] **Click-to-follow agent** — Left-click on an agent to "lock" the camera to it. Camera follows the agent as it moves. Shows detailed stats panel for that agent. Click elsewhere or press `ESC` to unlock. *→ Becomes the spectator mode camera system (Part 9.1) and adopt-an-agent tracking view.*
- [ ] **Speed control (1×–20×)** — Keyboard shortcuts (`1`–`5` or `+`/`-`) to adjust simulation speed. 1× = normal, 5× = run 5 ticks per frame, 20× = fast-forward. HUD shows current speed. *→ Required for spectator fast-forward and accelerated robotics training visualization.*
- [ ] **Step mode** — Press `.` (period) to advance exactly one tick while paused. Essential for debugging specific agent decisions.
- [ ] **Pin info panel** — Right-click on a tile to pin its info panel permanently. Can have multiple pinned panels. Click the `×` button to unpin.
- [ ] **Screenshot hotkey** — Press `F12` to save a timestamped PNG screenshot to `data/exports/screenshots/`. Uses `pygame.image.save()`.
- [ ] **Minimap** — Small overview of the entire world in the corner (50×50 pixel area), showing terrain colors and agent dots. Camera viewport drawn as a white rectangle. *→ Essential UI element for both spectator mode navigation and robotics environment overview.*
- [ ] **Keyboard shortcut help overlay** — Press `?` to toggle a semi-transparent overlay listing all keyboard shortcuts and their current state (toggle on/off).

### 7.4 Isometric Renderer Specific (gpu_renderer.py)

- [ ] **Elevation rendering** — If elevation physics is added, render tiles at different heights in the isometric view. Higher tiles render higher on screen, creating a natural 3D landscape effect.
- [ ] **Water animation** — Animate water tiles with a subtle color oscillation (shift the blue channel by sin(time)) to give a "flowing" look. Minimal GPU cost (uniform parameter in shader).
- [ ] **Day/night lighting** — If day/night cycle is added, modulate the global tile color by light level. Night = darker, bluer. Dawn/dusk = warm orange tint. Implemented as a uniform multiplier in the tile fragment shader. *→ Dramatic visual content for the entertainment spectator stream.*
- [ ] **Particle effects** — Spawn small particle sprites on events: eating (green sparkle), death (red puff), reproduction (yellow burst), planting (brown scatter). Uses a simple GPU particle system with instanced quads. *→ Event feedback for spectators — births, deaths, and farming become visually exciting moments.*
- [ ] **Shadow casting** — Render simple shadows below agents and tall plants to enhance depth perception in the isometric view.
- [ ] **Smooth agent movement** — Interpolate agent positions between ticks so movement appears smooth instead of jumping tile-to-tile. Store previous and current positions, lerp during render. *→ Critical for entertainment visual quality — spectators and streamers need polished visuals.*

### 7.5 Pygame Renderer Specific (pygame_renderer.py)

- [ ] **Sprite-based objects** — Replace the current colored-circle objects with small pixel-art sprites (berry = red circle with leaf, plant = green stalk, seed = brown dot). Load from a sprite sheet PNG.
- [ ] **Agent body shape** — Replace the triangle with a more expressive shape that shows energy level (shrinks when low energy), direction (clear arrow head), and flashes when eating.
- [ ] **Smooth scrolling** — Add momentum-based camera panning (drag-and-release continues scrolling with deceleration). Current pan is instant.
- [ ] **Tile grid toggle improvements** — When grid is off, add subtle tile borders only at terrain transitions (soil→rock, rock→water). Less visual noise than a full grid, more informative.
- [ ] **Object count badges** — When tiles have multiple objects (stacking mode), show a small number badge in the corner of the tile.
- [ ] **Zoom-dependent detail** — At low zoom, hide individual objects and show tile color averages. At high zoom, show full detail including agent direction arrows and item icons.

---

## Part 8: Robotics Learning Platform (Ultimate Goal)

> **Vision:** Transform the Emergent World-Model Sandbox into a **sim-to-real robotics learning platform** where users upload robot morphologies, train locomotion/manipulation controllers via neuroevolution + world models in our simulated environments, and export learned genome weights to physical hardware.

### 8.0 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE / API                          │
│  Upload URDF/SDF → Define task → Configure terrain → Train     │
└─────────────┬───────────────────────────────────┬───────────────┘
              │                                   │
     ┌────────▼────────┐                 ┌────────▼────────┐
     │  PHYSICS WORLD   │   train model  │   WORLD MODEL    │
     │  (PyBullet /     │ ──────────────>│   (Learned NN    │
     │   MuJoCo / 2D)   │                │    dynamics)     │
     │                  │<───validate────│                  │
     └────────┬────────┘                 └────────┬────────┘
              │                                   │
     ┌────────▼──────────────────────────────────▼────────┐
     │           EVOLUTION + LEARNING ENGINE                │
     │  Neuroevolution | RL (Actor-Critic) | Hybrid        │
     │  Lamarckian inheritance | Speciation | Curriculum   │
     └────────────────────────┬───────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   WEIGHT EXPORT    │
                    │   .npz / .onnx /   │
                    │   ROS2 msg format  │
                    └───────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  PHYSICAL ROBOT    │
                    └───────────────────┘
```

### 8.1 Physics Engine Integration

Move from the discrete grid world to continuous physics for realistic locomotion.

- [ ] **2D physics sandbox (Phase 1)** — Integrate Box2D (via `pybox2d`) as an alternative world mode. Keep the grid world for ecology experiments. 2D physics sufficient for: bipedal walkers, crawlers, wheeled robots, simple arms. Configurable via `world.physics_engine: "grid" | "box2d"`.
- [ ] **3D physics sandbox (Phase 2)** — Add PyBullet or MuJoCo backend for full 3D robotics. URDF/SDF model loading. Gravity, friction, contact dynamics. Configurable via `world.physics_engine: "pybullet" | "mujoco"`.
- [ ] **Terrain generator** — Procedural terrain: flat, slopes, stairs, gaps, rubble, deformable surfaces. Defined via YAML or Perlin-noise parameters. Different difficulty levels for curriculum learning.
- [ ] **Sensor simulation** — IMU (acceleration, gyro), joint encoders (position, velocity), contact sensors (foot pressure), distance sensors (LIDAR-like rays). Map to observation vector same way current vision grid works.
- [ ] **Timestep control** — Configurable physics dt (e.g. 1/240s for MuJoCo). Decouple physics steps from brain decision steps (e.g., brain decides every 10 physics steps = 24 Hz control).

### 8.2 Continuous Action Space

Current 8 discrete actions → N continuous joint torques.

- [ ] **Continuous policy head** — Replace discrete softmax with Gaussian policy: brain outputs (μ, σ) per joint → sample torques from N(μ, σ²). Tanh-squashed for bounded torques.
- [ ] **Hybrid action space** — Some actions discrete (e.g. gripper open/close) + some continuous (joint torques). Brain outputs both heads simultaneously.
- [ ] **Action scaling config** — Per-joint torque limits and scaling factors defined in the morphology YAML. Brain outputs normalized [-1, 1], scaled to physical range.
- [ ] **Motor primitives (CPG)** — Optional Central Pattern Generator layer between brain and joints. Brain modulates CPG parameters (frequency, amplitude, phase offsets) instead of raw torques. Much easier to evolve stable gaits.
- [ ] **Reflexes layer** — Equivalent to current instinct biases but for locomotion: stumble recovery, contact-triggered knee flexion, balance correction. Bootstraps learning like PICK_UP/EAT instincts do now.

### 8.3 Configurable Morphology

Users define robot body plans, loaded from standard formats.

- [ ] **URDF/SDF parser** — Load robot definitions from standard URDF (ROS) or SDF (Gazebo) files. Parse links, joints, masses, inertias. Map joint count → brain output size automatically.
- [ ] **YAML morphology format** — Simpler alternative to URDF for 2D robots. Define body segments, joints, lengths, masses, motor strengths in YAML. Lower barrier to entry.
- [ ] **Morphology → brain auto-sizing** — Observation size = sensor_count, output size = joint_count. Brain architecture auto-configured from morphology. Genome weight count adapts automatically.
- [ ] **Morphology evolution (optional)** — Let evolution modify body parameters (leg length, mass distribution, joint limits) alongside brain weights. Co-evolve body and brain. Genome traits already support this — extend with morphology genes.
- [ ] **Morphology library** — Ship built-in morphologies: `biped_simple`, `quadruped`, `hexapod`, `wheeled_2wd`, `arm_3dof`, `snake`. Users start from these or create custom.

### 8.4 World Model (Learned Dynamics)

The key to sample-efficient training — learn the environment dynamics, then "dream."

- [ ] **Transition predictor** — Small MLP or GRU that predicts `(next_obs, reward)` given `(obs, action)`. Train on collected physics simulation data. This IS the world model.
- [ ] **Data collection pipeline** — During physics training, log all (obs, action, next_obs, reward) tuples. Structured storage in the existing `data/` directory. Use existing `WorldModelLogger` format.
- [ ] **World model training loop** — Offline: train predictor on collected data. Minimizes ‖predicted_next_obs − actual_next_obs‖². Separate from agent training.
- [ ] **Dream-based evolution** — Run neuroevolution episodes INSIDE the world model instead of the physics sim. 100-1000× faster (no physics computation). Periodically validate best agents in real physics to prevent model exploitation.
- [ ] **Dreamer-style imagination** — Agent uses world model to roll out K-step imagined trajectories and evaluates them with its value function. Plans before acting. Model-based RL integrated with the existing Actor-Critic.
- [ ] **Model ensemble** — Train 3-5 world models with different random seeds. Use disagreement between them as uncertainty estimate. Agents penalized for visiting high-uncertainty states (prevents exploiting model errors).
- [ ] **Incremental model refinement** — As agents discover new behaviors, the world model encounters out-of-distribution states. Auto-detect high prediction error and collect more data in those regions. Active learning for the world model.
- [ ] **Latent world model** — Instead of predicting raw observations, learn a compressed latent space (VAE or AE) and predict latent dynamics. Scales to high-dimensional sensors (camera images, dense LIDAR).

### 8.5 Task Definition API

Users define what "success" means for their robot.

- [ ] **YAML task definitions** — Specify tasks declaratively:
  ```yaml
  tasks:
    walk_forward:
      type: locomotion
      objective: maximize_x_velocity
      constraints:
        min_height: 0.3          # Don't fall over
        max_torque_cost: 0.001   # Energy efficiency
        max_joint_velocity: 10.0 # Smooth motion
      terrain: flat
      time_limit: 1000
      fitness: weighted_sum       # or pareto, lexicographic
  ```
- [ ] **Composite fitness functions** — Multi-objective: speed × stability × energy_efficiency. Pareto-front evolution for trade-off exploration. Users weight objectives in config.
- [ ] **Task curriculum** — Automatic difficulty progression: flat → mild slope → stairs → rubble → gaps. Advance when population fitness exceeds threshold. Configurable in YAML.
- [ ] **Task library** — Built-in tasks: `stand_still`, `walk_forward`, `walk_fast`, `turn_in_place`, `climb_stairs`, `cross_gap`, `push_object`, `follow_path`, `recover_from_push`.
- [ ] **Custom reward plugins** — Python function that receives (obs, action, next_obs, morphology) and returns float reward. Drop a `.py` file in `tasks/` folder and reference in YAML.

### 8.6 Sim-to-Real Transfer Pipeline

Bridge the reality gap so evolved controllers work on physical hardware.

- [ ] **Domain randomization** — During training, randomize physics parameters each episode: mass ±20%, friction ±30%, motor strength ±15%, sensor noise ±10%, terrain roughness ±25%. Config specifies randomization ranges. Produces robust controllers.
- [ ] **System identification** — Given real robot telemetry (joint positions over time), fine-tune world model / physics parameters to match reality. Minimizes sim-real trajectory gap.
- [ ] **Weight export formats** — Export learned genome weights as:
  - `.npz` (NumPy — current format, works for Python deployment)
  - `.onnx` (ONNX — runs on any framework, edge devices, microcontrollers)
  - `.tflite` (TensorFlow Lite — for mobile/embedded)
  - ROS2 parameter file (direct loading into ROS2 controller node)
- [ ] **Inference-only runtime** — Minimal Python/C++ module that loads exported weights and runs the GRU forward pass at control frequency (100-1000 Hz). No training dependencies. Deployable on Raspberry Pi, Jetson, ESP32 (via ONNX → C).
- [ ] **Reality validation loop** — Run physical robot → record trajectory → compare to simulation prediction → update world model → retrain. Closes the sim-to-real loop iteratively.
- [ ] **Transfer confidence score** — Use world model ensemble disagreement on real-world observations to estimate how far physical conditions are from training distribution. Warns user when the robot encounters situations not seen in simulation.

### 8.7 Platform & User Experience

Make it accessible to robotics researchers and hobbyists.

- [ ] **Web UI** — Browser-based interface (Streamlit or FastAPI+React) for: uploading URDF, configuring tasks, launching training, monitoring progress (live fitness curves), downloading weights. No CLI knowledge needed.
- [ ] **Training dashboard** — Real-time charts: population fitness, best agent replay, world model loss, training throughput (agents/sec). Built on the existing analysis pipeline.
- [ ] **Robot preview** — 3D viewer (three.js or PyBullet GUI) showing the robot morphology animated with the best evolved controller. Replay individual episodes.
- [ ] **Experiment tracking** — MLflow or Weights & Biases integration for logging hyperparameters, fitness curves, model checkpoints. Compare runs across different morphologies/tasks.
- [ ] **Multi-user server mode** — Multiple users submit training jobs. Queue-based execution. GPU allocation. Results stored per user. REST API for programmatic access.
- [ ] **Documentation & tutorials** — Step-by-step guides: "Train a biped to walk in 30 minutes", "Export to ROS2", "Design a custom robot morphology", "Create a new task".

---

## Part 9: Entertainment & Online World Platform

> **Vision:** An online multiplayer simulated world where users watch, interact with, and compete through evolving AI creatures — blending artificial life, gaming, and social sandbox mechanics.

### 9.1 User-Controlled Agents & Spectator Mode

- [ ] **Spectator mode** — Browser-based live viewer of an always-running world. Users watch AI agents evolve, farm, compete in real time. Chat overlay. Camera follows interesting events (births, deaths, calamities). Low bandwidth — server sends tile diffs, not frames.
- [ ] **Adopt-an-agent** — Users claim an AI agent lineage. They can name it, track its family tree, see stats (lifespan, offspring count, fitness records). Leaderboard of longest-surviving lineages.
- [ ] **Player avatar mode** — User controls one agent directly (WASD + action keys) in the same world as AI agents. Compete for resources, cooperate, or just explore. Player actions logged for imitation learning (AI learns from human play).
- [ ] **Creature designer** — Web UI where users design creature appearance (cosmetic skins over the brain/genome), choose starting instinct presets, and set evolution preferences (aggressive, cooperative, explorer). Cosmetics are NFT-free — just fun customization.
- [ ] **Photo mode** — Isometric renderer screenshot tool with filters, annotations, and share-to-social buttons. Frame-worthy shots of emergent behavior.

### 9.2 Competitive Game Modes

- [ ] **Arena mode** — Users submit trained genomes. 16-agent bracket tournament in a sealed arena (no reproduction). Last agent alive wins. Ranked ladder with ELO ratings. Genome upload API.
- [ ] **King of the Hill** — Central resource pile. Agents score points per tick they occupy the area. User-submitted genomes vs. wild-type AI. Round-robin seasons.
- [ ] **Speed farming challenge** — Fixed-seed world, 500-tick time limit. Highest food production wins. Tests evolved farming efficiency. Leaderboard.
- [ ] **Survival gauntlet** — Progressively harder environments (calamity frequency increases, resources shrink). How long can your genome last? Public high scores.
- [ ] **Co-op boss mode** — Multiple user-submitted genomes must cooperate to survive an extreme calamity wave. Shared fitness score. Encourages evolving cooperative strategies.
- [ ] **Genome marketplace** — Users can share/trade successful genomes. Download top-performing genomes from the arena leaderboard. Fork and mutate other users' creatures.

### 9.3 Persistent Online World

- [ ] **Always-on server** — Centralized world that runs 24/7. Agents evolve continuously. Users log in to check on their lineages, inject new genomes, or observe. World state persisted to database.
- [ ] **World events** — Scheduled and random: meteor impacts, drought seasons, resource booms, migration waves. Announced in advance ("Calamity in 2 hours!"). Users tune in to watch how populations adapt.
- [ ] **Biome servers** — Multiple parallel worlds with different configurations: Jungle (high resources, fast growth), Desert (scarce, hostile), Ocean (water-dominated, island survival), Volcanic (fire + rock). Users choose which world to play in.
- [ ] **Historical timeline** — Record world milestones: "First farming behavior evolved at tick 50,000", "Population crash from drought at tick 120,000", "Agent #4521 reached fitness 1000". Browse history like a timeline.
- [ ] **Ecosystem economy** — Agent actions produce byproducts (fertilizer, seeds) that affect the global economy. Resource scarcity creates natural supply-demand dynamics visible to users.

### 9.4 Social & Community Features

- [ ] **Live chat** — In-world chat tied to spectator view. Users comment on events, share strategies, react to emergent behaviors.
- [ ] **Replays & highlights** — Auto-detect interesting moments (rare behaviors, record fitness, mass extinction events) and save as shareable replay clips (15-30 seconds, GIF or video).
- [ ] **Creature wiki** — Auto-generated pages for notable agents: lineage tree, behavioral profile (action distribution), territory map, lifespan, offspring. Community can add notes.
- [ ] **Clan system** — Users group their lineages into clans. Clan-vs-clan arena competitions. Shared breeding pool (crossover between clan members' genomes).
- [ ] **Achievement system** — Unlock badges: "First 1000-tick survivor", "Planted 100 seeds", "10-generation lineage", "Arena champion", "Survived a calamity". Displayed on user profile.
- [ ] **Streaming integration** — OBS/Twitch overlay showing live stats, agent POV camera, leaderboard ticker. Built-in streaming-friendly UI mode.

### 9.5 Monetization & Sustainability (if needed)

- [ ] **Cosmetic creature skins** — Visual customizations for agent appearance in the renderer. No gameplay advantage.
- [ ] **Premium biome access** — Exclusive world configs with unique terrain/object combos. Base worlds always free.
- [ ] **Compute tier** — Free users evolve on shared servers (time-sliced). Paid tier gets dedicated GPU for faster evolution training.
- [ ] **API access** — Free: read-only stats, genome download. Paid: bulk training jobs, custom world configs, webhook notifications on events.
- [ ] **Tournament entry** — Major seasonal tournaments with prizes (cash, hardware, or community recognition). Fee-based entry for prize pools, free spectating.

### 9.6 Technical Infrastructure for Online Play

- [ ] **WebSocket world streaming** — Server pushes world state diffs to connected clients at configurable frequency (1-10 Hz). Client renders locally. Bandwidth-efficient (delta encoding).
- [ ] **REST API** — Endpoints: `/worlds` (list), `/worlds/{id}/state` (current state), `/genomes` (upload/download), `/arena/submit`, `/leaderboard`, `/user/lineages`. OpenAPI spec.
- [ ] **Database backend** — PostgreSQL for user accounts, genome storage, leaderboards, world history. Redis for real-time state caching and pub/sub (live events).
- [ ] **Horizontal scaling** — Each biome world runs as an independent process. Load balancer routes users to the correct world server. Add worlds by spinning up new containers.
- [ ] **Anti-cheat** — Server-authoritative simulation. Submitted genomes are validated (correct weight count, no NaN/Inf). Arena runs server-side only (no client-side execution).
- [ ] **Mobile client** — Responsive web viewer (touch-friendly spectator mode). Lightweight — receives rendered tiles, not raw simulation state. Progressive Web App for home screen install.

---

## 🎯 Ultimate Project Goal

> **Build an open-source platform with two faces:**
> 1. **Robotics Track** — Design a robot, evolve controllers in simulation via neuroevolution + world models, export trained brains to physical hardware (sim-to-real).
> 2. **Entertainment Track** — An online living world where users spectate, compete, adopt, and breed evolving AI creatures — part artificial life experiment, part social game.
>
> **Both tracks share the same engine:** neuroevolution, GRU brains, Lamarckian inheritance, world models, YAML-driven configuration, and the evolution loop proven in Phase 0.

The current Emergent World-Model Sandbox is **Phase 0** — it proves the core loop works:
- Neuroevolution produces capable agents (11K+ tick survival, emergent farming)
- GRU brains are compact enough for embedded deployment (~8K parameters) and lightweight enough for 1000+ concurrent agents on a single server
- Lamarckian inheritance accelerates evolution
- The YAML config system makes the platform extensible without code changes
- The dual renderer (Pygame + GPU isometric) provides both development and presentation quality visuals

**Why both tracks together?** The entertainment platform generates massive evolutionary runs (millions of agent-ticks from always-on servers + user competitions), producing training data and genome diversity that directly feeds the robotics world model. Users playing the game are unknowingly contributing to better robot controllers.

---

## Suggested Progression Order

A recommended path through these suggestions, balancing impact and effort:

### Phase A: Quick Wins (1-2 days)
1. Tune WAIT back to target range
2. Configurable reward shaping in YAML
3. Per-generation metrics CSV
4. Equalize turn energy cost

### Phase B: UI Quick Wins (2-3 days)
5. Agent trail visualization (`T` key toggle)
6. Speed control (1×–20×)
7. Step mode (`.` key while paused)
8. Click-to-follow agent
9. Population graph panel
10. Screenshot hotkey

### Phase C: World Physics (1-2 weeks)
11. Day/night cycle (most impactful single addition)
12. Water flow / moisture diffusion
13. Temperature per tile
14. Multiple plant species
15. Nutrient diffusion on plant death

### Phase D: Smarter Agents (1-2 weeks)
16. GAE advantage estimation
17. Curriculum learning
18. Curiosity-driven exploration (ICM)
19. Attention over vision grid

### Phase E: Advanced Rendering (1-2 weeks)
20. Heatmap overlays (fertility / moisture / visits)
21. Smooth agent movement (isometric)
22. Day/night lighting (isometric shader)
23. Particle effects for events
24. Minimap

### Phase F: Scale & Research (ongoing)
25. World model (transition predictor)
26. Speciation / NEAT-style evolution
27. Distributed island-model simulation
28. GRU hidden state interpretability analysis

### Phase G: Online World Foundation (2-4 weeks) — *Entertainment Track Entry Point*
29. WebSocket world streaming (server → browser client)
30. REST API (worlds, genomes, leaderboard)
31. Spectator mode (browser-based live world viewer)
32. Adopt-an-agent system (claim, name, track lineages)
33. Database backend (PostgreSQL + Redis)

### Phase H: Game Modes & Social (2-4 weeks)
34. Arena mode (genome-vs-genome tournaments, ELO)
35. Speed farming challenge (leaderboard)
36. Genome marketplace (share/download/fork)
37. Achievement system (badges, milestones)
38. Live chat + replay highlights

### Phase I: Persistent World (2-4 weeks)
39. Always-on server (24/7 evolving world)
40. World events system (scheduled calamities, booms)
41. Biome servers (jungle, desert, ocean, volcanic)
42. Historical timeline (milestones, records)
43. Player avatar mode (direct control alongside AI)

### Phase J: Physics Foundation (2-4 weeks) — *Robotics Track Entry Point*
44. Box2D physics sandbox (2D continuous physics alongside grid world)
45. Continuous policy head (Gaussian μ,σ outputs for joint torques)
46. YAML morphology format (define 2D robot bodies)
47. Terrain generator (slopes, stairs, gaps)
48. Sensor simulation (IMU, contact, distance)

### Phase K: World Model Core (2-4 weeks)
49. Transition predictor (MLP: obs+action → next_obs+reward)
50. Data collection pipeline (physics → world model training data)
51. World model training loop (offline supervised)
52. Dream-based evolution (evolve agents inside learned model)
53. Model ensemble for uncertainty

### Phase L: Task & Transfer System (2-4 weeks)
54. YAML task definitions (objective, constraints, terrain, time limit)
55. Task curriculum (auto difficulty progression)
56. Domain randomization (randomize physics params per episode)
57. Weight export (.npz, .onnx, .tflite)
58. Inference-only runtime (minimal forward-pass module)

### Phase M: 3D Physics & Full Platform (1-3 months)
59. PyBullet/MuJoCo backend for full 3D physics
60. URDF/SDF parser for robot loading
61. Morphology library (biped, quadruped, hexapod, arm)
62. Web UI for training management
63. Reality validation loop (physical robot → sim update → retrain)
64. Mobile client (PWA spectator)
65. Streaming integration (Twitch/OBS overlays)

---

*This document is a living checklist. Check off items as they're implemented and add new ideas as they emerge.*

*🎯 Every item above serves the twin goals: **a living online world for entertainment** and **a sim-to-real robotics learning platform** — both powered by the same neuroevolution + world model engine.*
