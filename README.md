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
- 🎮 **Dual Renderer**: Pygame 2D GUI **or** a GPU-accelerated isometric 2.5D renderer via ModernGL
- 🎯 **Fading Instincts**: Survival-bootstrapping biases (PICK_UP, EAT, USE, turn-toward-food, hunger-eat) that genuinely fade to zero with agent age — adults act purely on their learned network
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

See [CLI_GUIDE.md](CLI_GUIDE.md) for the full command-line reference.

## What's New — Brain v3 Upgrade (Phases 1–3)

**In one sentence:** the brain's internals were reorganised so it can grow,
the survival "training wheels" now genuinely come off as agents mature, and an
opt-in attention-based brain architecture is available alongside the legacy one.

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

Full design rationale and the roadmap for the remaining phases (full-network
lifetime learning, learned world models) are in
[BRAIN_V3_PROPOSAL.md](BRAIN_V3_PROPOSAL.md); change details are in
[CHANGELOG.md](CHANGELOG.md).

## Architecture

```
emergent-world-model/
├── world/           # Tiles, objects, systems, object registry, tile-effect engine
├── agents/          # Agent lifecycle, Actor-Critic learning, evolution, genome
│   └── brain/       # GRU brain package: spec (genome/observation layouts),
│                    #   modules (pure NN functions), instincts (fading biases)
├── simulation/      # Simulation management
├── utils/
│   ├── agents/      # Perception (72-dim obs), action execution, reward shaping
│   ├── data/        # AgentLogger, WorldModelLogger (async + persistent handles)
│   └── ui/          # Pygame renderer, GPU isometric renderer (ModernGL)
├── config/          # YAML configs + custom object definitions
├── tests/           # 240+ unit tests
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

Why v3 (full rationale in [BRAIN_V3_PROPOSAL.md](BRAIN_V3_PROPOSAL.md)):
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
  version: 3        # 2 (default) = legacy baseline, 3 = attention brain
  v3:
    embed_dim: 8
    state_dim: 40
    gru_hidden_size: 48
    value_hidden: 16
```

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

Key simulation parameters in `config/default.yaml`:
- **Evolution mode** (`evolution.mode`): `"rl"` or `"neuroevolution"`
- World size, terrain generation (soil, rock, water, sand ratios)
- Population size, genetic parameters
- Resource availability and growth rates
- Learning scheduler (interval, budget, adaptive mode)
- Visualisation and data collection settings

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
python analyze_v3_1.py
```
Reports action distribution, success rates, food consumption, and compares against
survival targets (WAIT ≥ 32–35 %, EAT success 100 %, survival > 1 500 ticks).

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

1. Follow the development guidelines in `guideline.md`
2. All functions must have proper docstrings
3. Focus on emergent behaviours, never hardcode high-level strategies
4. Maintain separation between world physics and agent logic
5. Write comprehensive tests — run `pytest` before opening a PR (240+ tests, all must pass)

## Future Extensions

- **Curiosity-driven Learning**: Intrinsic motivation for exploration (ICM / RND)
- **Communication Evolution**: Emergent language and signalling via SIGNAL action
- **Tool Construction**: Building and using complex tools
- **Environmental Dynamics**: Day/night cycles, seasons, temperature, and weather
- **Speciation (NEAT-style)**: Species-level diversity protection
- **World Models**: Dreamer-style imagination and model-based planning

See [SUGGESTIONS.md](SUGGESTIONS.md) for the full 80+ item roadmap.

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
