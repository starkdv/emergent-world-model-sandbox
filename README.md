# Emergent World-Model Sandbox

**Author:** Karan Vasa  
**License:** MIT  

A simulation sandbox with evolving agents that learn survival strategies and exhibit emergent behaviors through evolution in a physically consistent, resource-based 2D world.

## Overview

This project implements a **2D grid world** where:

- **Agents** move around, consume resources, and manipulate objects using only primitive actions
- **Evolution** drives the emergence of complex behaviors like farming, cooperation, and communication
- **World physics** governs resource transformation, plant growth, and environmental dynamics
- **Learning** — each agent trains an Actor-Critic GRU brain online using its own experience
- **No hardcoded behaviors** — all complex strategies emerge through natural selection and reinforcement learning

## Key Features

- 🌱 **Emergent Agriculture**: Agents discover seed planting and farming through evolution
- 🏜️ **Environmental Hazards**: Sand terrain spreads and degrades soil unless trees block it
- 🧩 **Custom Object System**: Define new objects via YAML — foods, plants, terrain effects, structures
- 🤝 **Cooperation**: Group behaviors emerge without explicit programming
- 🧠 **Neural Evolution + Online Learning**: Agents use evolved GRU Actor-Critic networks that learn in real time
- 🎮 **Dual Renderer**: Pygame 2D GUI **or** a GPU-accelerated isometric 2.5D renderer via ModernGL
- 🎯 **Contextual Instincts**: Survival-bootstrapping biases (PICK_UP, EAT, USE, turn-toward-food) that fade as learned weights strengthen
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

## Architecture

```
emergent-world-model/
├── world/           # Tiles, objects, systems, object registry, tile-effect engine
├── agents/          # Agent lifecycle, GRU brain, Actor-Critic learning, evolution, genome
├── simulation/      # Simulation management
├── utils/
│   ├── agents/      # Perception (72-dim obs), action execution, reward shaping
│   ├── data/        # AgentLogger, WorldModelLogger (async + persistent handles)
│   └── ui/          # Pygame renderer, GPU isometric renderer (ModernGL)
├── config/          # YAML configs + custom object definitions
├── tests/           # 205 unit tests
└── data/            # Logs, exported data, saved weights
```

## Neural Architecture

Each agent runs a compact **Actor-Critic GRU** network (≈8 873 parameters):

```
Observation (72) → Encoder [FC→tanh×2] (32) → GRU (32) → Policy head (8 actions)
                                                         → Value head  (1)
```

**Observation layout (72 features):**

| Range | Feature group | Description |
|-------|--------------|-------------|
| 0–7 | Agent state | Energy, age, direction one-hot, inventory space, metabolism |
| 8–57 | Vision 5×5×2 | Type + value encoding per tile, terrain-layer aware |
| 58–65 | Stimulus | food_on_tile, seed_on_tile, food_ahead, resource_ahead, nearest_food_prox, food_dir_match, energy_urgency, can_interact |
| 66–71 | Inventory | Fullness, has_food, has_seed, has_fertilizer, total_calories, count |

**Contextual instinct biases** (bootstrap survival; fade as learned logits strengthen):

| Condition | Action biased | Logit boost |
|-----------|--------------|------------|
| PICK_UP valid | PICK_UP | +1.5 |
| EAT valid | EAT | +1.0 |
| USE valid | USE | +0.5 |
| Food nearby but not ahead | TURN_LEFT / TURN_RIGHT | +0.8 × proximity |

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

### Emergent Behaviors
Complex behaviors emerge from combinations of primitive actions:
- **Farming**: Agents learn to pick up seeds, plant them, and return to harvest
- **Tool Use**: Discovery and utilisation of beneficial objects
- **Cooperation**: Coordinated group activities
- **Communication**: Evolved signalling between agents

### Evolution System
- **Neural Networks**: Each agent has an evolved GRU Actor-Critic brain
- **Genetic Algorithm**: Tournament selection, crossover, and mutation
- **Fitness-based Selection**: Survival and reproduction based on energy accumulation
- **Trait Inheritance**: Both neural weights and behavioural traits evolve

## Configuration

Key simulation parameters in `config/default.yaml`:
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
5. Write comprehensive tests — run `pytest` before opening a PR (205 tests, all must pass)

## Future Extensions

- **Curiosity-driven Learning**: Intrinsic motivation for exploration
- **Communication Evolution**: Emergent language and signalling
- **Tool Construction**: Building and using complex tools
- **Environmental Dynamics**: Seasons, climate, and resource cycles

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


## Overview

This project implements a **2D grid world** where:

- **Agents** move around, consume resources, and manipulate objects using only primitive actions
- **Evolution** drives the emergence of complex behaviors like farming, cooperation, and communication
- **World physics** governs resource transformation, plant growth, and environmental dynamics
- **No hardcoded behaviors** - all complex strategies emerge through natural selection

## Key Features

- 🌱 **Emergent Agriculture**: Agents discover seed planting and farming through evolution
- 🏜️ **Environmental Hazards**: Sand terrain spreads and degrades soil unless trees block it
- 🧩 **Custom Object System**: Define new objects via YAML — foods, plants, terrain effects, structures
- 🤝 **Cooperation**: Group behaviors emerge without explicit programming
- 🧠 **Neural Evolution**: Agents use evolved neural networks to make decisions
- 🔬 **Scientific Analysis**: Comprehensive data collection and visualization tools
- 🎮 **Real-time Visualization**: Interactive GUI for monitoring simulations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/starkdv/emergent-world-model-sandbox.git
cd emergent-world-model-sandbox

# Create and activate virtual environment (REQUIRED)
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Simulations

```bash
# Run with console demo (ASCII visualization)
python main.py --demo --seed 42

# Run with GUI visualization (Pygame)
python main.py --gui --seed 42

# Run with custom configuration
python main.py --config config/custom.yaml --gui

# Run with custom objects (add superfoods, mushrooms, etc.)
python main.py --gui --objects config/custom_objects.yaml

# Run with CSV logging for analysis
python main.py --gui --log --log-frequency 10

# Run headless for data collection
python main.py --no-viz --generations 1000 --log
```

### GUI Controls

When running with `--gui`:
- **SPACE**: Pause/Resume simulation
- **G**: Toggle grid overlay
- **R**: Reset camera to center
- **WASD/Arrow Keys**: Pan camera
- **Mouse Wheel**: Zoom in/out
- **Left Click + Drag**: Pan camera
- **Hover over tiles**: View detailed tile and object information
  - See terrain properties (fertility, moisture)
  - Inspect object components (calories, age, growth status, etc.)
  - View up to 3 objects per tile with full details
- **ESC**: Exit simulation

> 💡 **Tip**: Press SPACE to pause, then hover over objects for easier inspection!

See [INSPECTION_GUIDE.md](INSPECTION_GUIDE.md) for detailed object inspection instructions.

## Architecture

```
emergent_world_model/
├── world/          # World physics, tiles, objects, systems, object registry
├── agents/         # Agent behavior, brain, evolution, learning
├── simulation/     # Simulation management
├── utils/          # Utilities, rendering, data collection
├── config/         # Configuration files & custom object definitions
├── tests/          # 198+ unit tests
└── data/           # Exported data, logs, weights
```

## Core Concepts

### Primitive Actions Only
Agents can only perform basic actions:
- `MOVE` - Move in a direction
- `TURN` - Change facing direction  
- `PICK_UP` - Pick up objects
- `DROP` - Drop objects from inventory
- `EAT` - Consume edible objects

### Emergent Behaviors
Complex behaviors emerge from combinations of primitive actions:
- **Farming**: Agents learn to plant seeds and return to harvest
- **Tool Use**: Discovery and utilization of beneficial objects
- **Cooperation**: Coordinated group activities
- **Communication**: Evolved signaling between agents

### Evolution System
- **Neural Networks**: Each agent has an evolved brain
- **Genetic Algorithm**: Tournament selection, crossover, and mutation
- **Fitness-based Selection**: Survival and reproduction based on energy accumulation
- **Trait Inheritance**: Both neural weights and behavioral traits evolve

## Configuration

Key simulation parameters:
- World size and terrain generation (soil, rock, water, sand ratios)
- Population size and genetic parameters
- Resource availability and growth rates
- Visualization and data collection settings

See `config/default.yaml` for full configuration options.

## Custom Objects

Create new world objects entirely from YAML — no code changes required.
See `config/custom_objects.yaml` for a full reference with 7 example objects.

### Quick Example

Add an `objects:` section to your config **or** use a separate file with `--objects`:

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

| Component     | Purpose                                      |
|---------------|----------------------------------------------|
| `edible`      | Food with calories, toxicity, freshness decay |
| `seed`        | Plantable seed that grows into a plant       |
| `plant`       | Grows, matures, produces resources           |
| `fertilizer`  | Boosts soil fertility in a radius            |
| `interaction` | Controls pickable, usable, passable, blocks_growth |
| `tile_effect` | Environmental multipliers, spreading, terrain conversion |
| `physics`     | Decay, decomposition, nutrient return        |
| `observation` | How agents perceive the object               |

### Tile Effect System

Objects with `tile_effect` modify the environment around them:

- **Growth/germination multipliers** — sand uses 0.1 (10x slower)
- **Spawn rate multiplier** — reduce food production on affected tiles
- **Spreading** — converts neighbouring soil tiles over time (tracked per-tick)
- **Blocked by** — plants can prevent spread (e.g. trees block sand)
- **Terrain conversion** — permanently changes tile type (sand, rock, etc.)
- **Fertility/moisture override** — clamps tile values

Built-in terrain hazard: **Sand** (5% of default map)
- 10x harder germination, 10x slower growth, 70% less food
- Spreads to adjacent soil every 200 ticks if no plant is nearby
- Shows as sandy beige `:` in console, warm colour in GUI

## Data Analysis

The system provides comprehensive data collection:

### CSV Logging System ✨ NEW!
Track all agent actions and states throughout the simulation:
```bash
# Enable logging
python main.py --gui --log

# Custom log directory and frequency
python main.py --gui --log --log-dir analysis/run1 --log-frequency 10
```

**Generated Files:**
- `agent_actions_*.csv`: Every action taken by each agent (18 columns)
- `agent_states_*.csv`: Agent state snapshots (20 columns)

**Analysis Examples:**
```python
import pandas as pd

# Load and analyze actions
actions = pd.read_csv("data/logs/agent_actions_20251114.csv")
print(actions['action'].value_counts())  # Most common actions

# Track energy over time
states = pd.read_csv("data/logs/agent_states_20251114.csv")
states.groupby('agent_id')['energy'].plot()
```

See [AGENT_LOGGING.md](AGENT_LOGGING.md) for complete usage guide and analysis examples.

### Other Data Collection
- Agent behavior sequences and decision patterns
- Population dynamics and genetic diversity
- Resource distribution and consumption metrics
- Emergent behavior occurrence tracking

Export formats: CSV, JSON, Parquet for analysis with pandas/numpy/scipy.

## Visualization

- **Real-time GUI**: Live simulation viewing with agent status panels
- **Interactive Controls**: Play/pause, speed adjustment, agent inspection
- **Data Charts**: Population graphs, fitness distributions, behavior heatmaps
- **Replay System**: Review and analyze past simulation runs

## Research Applications

This sandbox is designed for studying:
- **Artificial Life**: How complex behaviors emerge from simple rules
- **Evolutionary Computation**: Natural selection in neural network populations
- **Multi-agent Systems**: Coordination and communication without central control
- **Emergent Intelligence**: The transition from reactive to planning-based behavior

## Contributing

1. Follow the development guidelines in `guideline.md`
2. All functions must have proper docstrings
3. Focus on emergent behaviors, never hardcode high-level strategies
4. Maintain separation between world physics and agent logic
5. Write comprehensive tests for new features

## Future Extensions

- **Curiosity-driven Learning**: Intrinsic motivation for exploration
- **Communication Evolution**: Emergent language and signaling
- **Tool Construction**: Building and using complex tools
- **Environmental Dynamics**: Seasons, climate, and resource cycles

## License

MIT License - see LICENSE file for details.

## Citation

If you use this work in research, please cite:

```
Vasa, K. (2025). Emergent World-Model Sandbox: Evolution of Complex Behaviors 
in Artificial Life Simulations. GitHub repository.
```

---

*This project demonstrates how complex, intelligent behaviors can emerge from simple rules and evolutionary pressure, without any explicit programming of high-level strategies.*
