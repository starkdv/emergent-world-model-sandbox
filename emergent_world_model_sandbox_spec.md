# Emergent World-Model Sandbox – Design Document

> **Goal:** Build a **simulation sandbox** with evolving agents that learn survival strategies and eventually exhibit **emergent behaviours** (e.g., proto-farming, cooperation, communication) using only **primitive actions** and a physically consistent, resource-based world.

This document is meant for **GitHub Copilot–assisted coding**. It describes:

- World structure and update rules  
- Agent structure, brain, genome, and evolution  
- Objects, components, and resource transformations  
- Action space (no high-level “farm/build” actions)  
- Future extensions: curiosity, communication, and learned world models  

The initial implementation should be in **Python**, but the design is language-agnostic.

---

## 1️⃣ High-Level Concept

We want a **2D grid world** where:

- Agents:
  - Move around.
  - Consume resources ("food") to maintain energy.
  - Can manipulate objects (e.g., seeds).
  - Can, through evolution, discover behaviours like **planting seeds**, revisiting areas, proto-farming, etc.
- The World:
  - Has **terrain tiles** with properties (soil, fertility, moisture).
  - Contains **objects** (seeds, plants, food piles, tools, etc.).
  - Evolves via **simple physics-like rules** (growth, decay, resource transformation).
- Evolution:
  - Agents have neural-net **brains** and **genomes**.
  - Mating and mutation create new generations.
  - No explicit “high-level” actions; behaviours emerge from low-level motor-like actions and world physics.

We start with a **minimal but extensible** system. The design should support adding:

- Curiosity-driven intrinsic rewards  
- Emergent communication between agents  
- Internal world models for agents (learned dynamics)  

But Phase 1 will focus on the **core simulation + evolving agents**.

---

## 2️⃣ Core Architecture Overview

### 2.1 Main Modules

Suggested Python module structure:

- `world/`
  - `world.py` – `World` class & simulation loop.
  - `tiles.py` – tile representation & terrain types.
  - `objects.py` – `WorldObject` and components.
  - `systems.py` – world update systems (growth, decay, resource spawn).
- `agents/`
  - `agent.py` – `Agent` class and basic logic.
  - `brain.py` – neural network policy implementation.
  - `genome.py` – genome representation, mating, mutation.
  - `observation.py` – building observation vectors.
- `simulation/`
  - `runner.py` – top-level run loop, generation management.
  - `config.py` – simulation configuration.
- `utils/`
  - `render.py` – simple console/ASCII renderer (or later GUI).
  - `random_utils.py` – RNG helpers.
- `tests/`
  - Unit tests for key modules.

This document defines the **logic & structure**, not specific library choices (but `numpy` and possibly `torch` are expected).

---

## 3️⃣ World Design

### 3.1 World Representation

Use a **2D grid**:

```python
class World:
    width: int
    height: int
    tiles: list[list["Tile"]]
    objects: dict[int, "WorldObject"]  # id → object
    agents: dict[int, "Agent"]        # id → agent
    tick: int
```

### 3.2 Tile Representation

Each tile holds terrain + environmental properties and references to objects on it.

```python
class Tile:
    x: int
    y: int
    terrain_type: "TerrainType"   # enum: SOIL, ROCK, WATER, etc.
    fertility: float              # 0.0 – 1.0
    moisture: float               # 0.0 – 1.0
    object_ids: list[int]         # IDs of WorldObjects on this tile
```

**TerrainType** examples:

- `SOIL` – plantable.
- `ROCK` – impassable, non-plantable.
- `WATER` – affects moisture nearby (later).

### 3.3 World Systems (per-tick updates)

These are **pure-ish** systems that run every tick:

1. **Plant Growth System**
   - For each object with `PlantComponent`:
     - Increase age.
     - When `age >= mature_age`, periodically spawn seeds/fruit.
     - When `age > max_age`, plant dies. Optionally spawn fertilizer.

2. **Seed Germination System**
   - For objects with `SeedComponent`:
     - Check tile fertility & moisture.
     - If conditions met and enough time passes, transform Seed → Plant.

3. **Decay System**
   - For `Edible` objects:
     - Decrease freshness.
     - When freshness <= 0 → remove or turn into compost/fertilizer object.

4. **Fertilizer System**
   - Tiles with `FertilizerComponent` objects:
     - Temporarily boost fertility for nearby tiles.

5. **Resource Spawn System (optional safety net)**
   - At low frequency, spawn a few seeds/food items randomly to avoid total extinction.

World tick function:

```python
def update_world(world: World) -> None:
    apply_plant_growth(world)
    apply_seed_germination(world)
    apply_decay(world)
    apply_fertilizer_effects(world)
    spawn_resources_if_needed(world)
    world.tick += 1
```

---

## 4️⃣ Objects and Components

### 4.1 WorldObject

`WorldObject` is an **entity** holding a position and a set of components.

```python
class WorldObject:
    id: int
    x: int
    y: int
    components: dict[str, "Component"]
```

`components` is a mapping from component name to instance. Components describe behaviour and data; systems operate on them.

### 4.2 Core Components

#### 4.2.1 EdibleComponent

Represents something that can be eaten.

```python
class EdibleComponent:
    calories: float
    toxicity: float       # 0.0 = safe; > 0 may cause negative effects
    freshness: float      # 0.0 – 1.0, decays over time
```

#### 4.2.2 SeedComponent

Represents a plantable seed.

```python
class SeedComponent:
    plant_type: str       # key to look up Plant archetype
    grow_time: int        # ticks needed to germinate
    time_in_soil: int     # counter
    required_fertility: float
    required_moisture: float
```

#### 4.2.3 PlantComponent

Represents a plant in the world.

```python
class PlantComponent:
    age: int
    mature_age: int
    max_age: int
    spawn_resource_type: str   # e.g. "BerrySeed"
    spawn_rate: float          # probability per tick to spawn resource
```

#### 4.2.4 FertilizerComponent

Represents something that boosts tile fertility.

```python
class FertilizerComponent:
    fertility_boost: float
    duration: int             # ticks
```

#### 4.2.5 ToolComponent (future use)

Represents tools that modify interactions.

```python
class ToolComponent:
    effect_type: str          # e.g. "DIG", "HARVEST_BOOST"
    efficiency: float         # multiplier for effect strength
```

---

## 5️⃣ Agents

### 5.1 Agent Structure

```python
class Agent:
    id: int
    x: int
    y: int
    direction: tuple[int, int]   # e.g. (0, 1) for facing up
    energy: float
    age: int
    alive: bool

    inventory: list[int]         # object IDs in inventory

    # Evolutionary data
    genome: "Genome"
    brain: "Brain"               # wraps a neural net model
    traits: dict[str, float]     # e.g. {"metabolism_rate": 0.1, "vision_radius": 3}

    # Future (not required in v1)
    world_model: "WorldModel | None"
    curiosity_state: dict | None
    comm_state: dict | None

    # Logging / metrics
    fitness: float
```

###


---

## 16️⃣ Expanded Evolution & Mutation Logic

### 16.1 Goals of Evolution Module

The evolution system should:
- Maintain a population across generations.
- Select agents based on **fitness**.
- Create offspring using **mating (crossover)** and **mutation**.
- Support **genetic drift**, **diversity preservation**, and **lineage tracking**.

### 16.2 Selection Strategies

**Recommended default: Tournament Selection**
- Randomly pick `k` agents.
- Parent = highest fitness among them.
- Repeat for second parent.

Alternatives (optional):
- Roulette wheel (fitness proportionate).
- Rank-based selection.
- Elitism (preserve top N agents directly).

### 16.3 Genome Structure (Detailed)

```python
class Genome:
    weights: np.ndarray      # flattened neural network parameters
    traits: dict[str, float] # metabolism_rate, vision_radius, etc.
    lineage_id: int          # lineage tracking
```

### 16.4 Mating / Crossover (Detailed)

Options:

#### A. Uniform Crossover (recommended for v1)
Each gene is taken randomly from either parent.

```python
child_weights[i] = parent_a[i] if random() < 0.5 else parent_b[i]
```

#### B. One-point or Two-point Crossover (optional)
Split genome into segments and swap.

#### C. Trait-Level Crossover
Traits combine via:
- 50% inheritance
- averaging
- or averaging + noise

Example:
```python
child.traits[t] = (pa.traits[t] + pb.traits[t]) / 2 + noise
```

### 16.5 Mutation (Expanded)

Each gene (weight) mutates with small probability `p_mut`:

```python
if random() < p_mut:
    weights[i] += normal(0, mutation_std)
```

Traits mutate separately:
```python
traits[t] += normal(0, trait_mutation_std)
```

### 16.6 Speciation / Diversity Preservation (Optional for v1)

To avoid all agents collapsing to a single behaviour, add:
- **Novelty rewards**
- **Diversity penalties**
- **Local species** based on genetic distance

This can come later.

---

## 17️⃣ Curiosity Module (Intrinsic Motivation)

Curiosity encourages agents to explore the world and discover behaviours like planting, tool use, or cooperation.

### 17.1 Core Idea
Agents build a **world model** attempting to predict:

```python
next_obs = world_model(obs, action)
```

Curiosity = **prediction error**:
```python
error = ||next_obs_pred - next_obs_actual||
```

Add this to fitness:
```python
agent.fitness += curiosity_weight * error
```

### 17.2 World Model Structure
A small MLP:
- Input: `obs_t + action_t`
- Output: predicted `obs_{t+1}`

Train online:
```python
loss = mse(predicted_next_obs, actual_next_obs)
```

### 17.3 Benefits
- Agents explore unseen states.
- Agents discover long-term interactions (e.g., seed planting).
- Opens path to **model-based planning**.

---

## 18️⃣ Communication System (Emergent Language)

### 18.1 Goal
Agents evolve to:
- Signal nearby agents.
- Warn about threats.
- Announce discoveries.
- Coordinate resource harvesting.
- Eventually form **proto-language**.

### 18.2 Communication Action
Add one new primitive action:

```python
EMIT_SIGNAL
```

With a continuous or discrete payload:

```python
signal_value = brain.output_signal_head(obs)
```

### 18.3 Signal Propagation
A simple system:
- Signal exists for 1–3 ticks.
- Within radius R.
- All agents in area receive it:

```python
agent.last_signal = signal_value
```

### 18.4 Observation Integration
Include `last_signal` in the observation vector.

### 18.5 No Predefined Semantics
No meaning is assigned.

Evolution discovers:
- Cooperation signals.
- Warnings.
- Food-location broadcasts.
- Territorial markers.

### 18.6 Fitness Influence (Optional)
Encourage meaningful communication:
- Reward agents who help others survive.
- Reward food-sharing groups.
- Reward coordinated behaviours.

---

## 19️⃣ Planned Extensions

### 19.1 Tools & Construction
Introduce tools with components:
- DIG
- HARVEST_BOOST
- CARRY_BOOST

Agents learn tool-use behaviour.

### 19.2 Structures
- Nests
- Farms
- Storage structures

Constructed by combinations of low-level interactions.

### 19.3 Climate / Seasons
- Affects fertility, moisture, food growth.
- Forces long-term planning.

---

## 20️⃣ Final Notes for Implementation

### Copilot Guidelines
- Implement modular, testable systems.
- Keep world physics and agent logic separate.
- Document assumptions per module.
- Add debug visualizations early.

### Expected Emergent Behaviours (Long-Term)
- Hoarding
- Seed planting
- Garden tending
- Tool discovery
- Group movement
- Proto-communication
- Territoriality
- Basic cooperation

The design now supports all of this.

---

End of extended specification.

