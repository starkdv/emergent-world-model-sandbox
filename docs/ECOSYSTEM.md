# Ecosystem Simulation - Complete Technical Reference

**Author:** Karan Vasa  
**Date:** November 14, 2025  
**Updated:** February 21, 2026  
**Status:** ✅ Production Ready

> **Note (June 2026):** the *world physics* sections of this document are
> current, but the **brain and learning sections describe the pre-Brain-v3
> era** (single GRU-MLP brain, heads-only A2C, hardcoded auto-eat). For the
> current agent architecture see ../README.md (Neural Architecture),
> [BRAIN_V2_V3_COMPARISON.md](BRAIN_V2_V3_COMPARISON.md) (full math), and
> [BRAIN_V3_PROPOSAL.md](BRAIN_V3_PROPOSAL.md) (phase status): brain v2/v3
> versions, fading instincts, PPO full-network learning, and the learned
> world model (curiosity + latent planning).
>
> **World upgrade (June 2026):** the world is now being upgraded in phases
> (see [WORLD_UPGRADE_PROPOSAL.md](WORLD_UPGRADE_PROPOSAL.md)). **W0**
> hardened the object registry / custom-object authoring, and **W1** added
> an opt-in **environment engine** (day/night, seasons, weather) plus three
> verified dynamics-bug fixes (B1 moisture, B2 sand germination, B5 plant
> carrying capacity). Where this document's physics differs from the
> upgraded behaviour it is annotated inline, and the
> [World Upgrade (W0–W1)](#world-upgrade-w0w1--june-2026) section at the end
> is the authoritative summary.

---

## Table of Contents

1. [Overview](#overview)
2. [World Architecture](#world-architecture)
3. [Components & Objects](#components--objects)
4. [Agent System](#agent-system)
5. [Ecosystem Physics](#ecosystem-physics)
6. [System Dynamics](#system-dynamics)
7. [Population Control & Environmental Disasters](#population-control--environmental-disasters)
8. [Complete Lifecycle](#complete-lifecycle)
9. [Configuration Reference](#configuration-reference)
10. [Testing & Validation](#testing--validation)
11. [Reinforcement Learning System](#reinforcement-learning-system)
12. [World Upgrade (W0–W1)](#world-upgrade-w0w1--june-2026)
13. [Recent Updates](#recent-updates---november-17-2025)

---

## Overview

This is a fully self-sustaining ecosystem simulation featuring realistic physics, resource cycling, and emergent behavior. The world operates autonomously with complete nutrient cycles, water dynamics, and population control mechanisms.

### Key Features

✅ **Self-Sustaining** - Resources naturally cycle without intervention  
✅ **Realistic Physics** - Soil fertility, moisture, and decomposition  
✅ **Population Control** - Natural birth/death cycles + max population limits  
✅ **Environmental Disasters** - Periodic calamities create survival pressure  
✅ **Autonomous Agents** - Neural network brains making decisions  
✅ **Reinforcement Learning** - Agents learn survival strategies during lifetime  
✅ **Genetic Evolution** - Agents evolve behaviors over generations  
✅ **Reproduction System** - In-simulation asexual fission with mutation  
✅ **Configurable** - All parameters externalized in YAML config
✅ **Tested** - 219 tests passing (Feb 2026)  
✅ **Scalable** - Efficient ECS architecture supports 1000+ entities

### World Statistics

- **Grid Size:** 100x100 tiles (configurable)
- **Terrain Types:** Soil (70%), Rock (20%), Water (10%)
- **Update Systems:** 6 independent systems + calamity system
- **Resource Types:** Plants, Seeds, Berries, Fertilizer
- **Agent Population:** 20 initial agents (configurable, with max limits)
- **Physics Cycles:** Nutrient, Water, Reproduction, Decay
- **Neural Networks:** Recurrent Actor-Critic (GRU; parameters depend on config)
- **Learning System:** Online Actor-Critic with replay buffer (NumPy-first; optional Torch backend)
- **Reproduction:** Energy-based with mutation and cooldown
- **Environmental Pressure:** Periodic calamities (configurable)

---

## World Architecture

### Grid-Based Tile System

The world is composed of a 2D grid of tiles, each with unique properties:

```python
class Tile:
    terrain_type: TerrainType  # Soil, Rock, or Water
    fertility: float           # 0.0 to 1.0 (nutrients)
    moisture: float            # 0.0 to 1.0 (water content)
    x, y: int                  # Grid coordinates
```

#### Terrain Types

| Type | Ratio | Plantable | Properties |
|------|-------|-----------|------------|
| **Soil** | 65% | ✅ Yes | Variable fertility (0.3-1.0), moisture (0.2-0.8) |
| **Rock** | 20% | ❌ No | Impassable, zero fertility, zero moisture |
| **Water** | 10% | ❌ No | Zero fertility, maximum moisture (1.0); with the environment engine, adjacent soil recovers moisture (see W1) |
| **Sand** | 5% | ⚠️ Barely | Passable; a spreading terrain hazard that clamps fertility/moisture and slows growth/germination 10× (see W0/B2) |

> The default terrain mix is now soil 0.65 / rock 0.20 / water 0.10 /
> sand 0.05 (`terrain.*_ratio` in config). Sand is a built-in
> `TileEffectSpec` object, not a bare terrain enum.

### Entity-Component System (ECS)

Objects in the world are composed of modular components:

```
WorldObject (base entity)
    ├─ Position (x, y)
    └─ Components (composition)
        ├─ PlantComponent
        ├─ SeedComponent
        ├─ EdibleComponent
        └─ FertilizerComponent
```

---

## Components & Objects

### 1. PlantComponent

Represents living vegetation that ages, matures, and produces resources.

**Attributes:**
```python
age: int                    # Current age in ticks
mature_age: int            # Age when plant starts producing (100)
max_age: int               # Age when plant dies (500)
spawn_resource_type: str   # What resource to produce ("berry")
spawn_rate: float          # Production probability per tick (0.1 = 10%)
```

**Lifecycle:**
- **Young Plant** (0-99 ticks): Growing, no reproduction
- **Mature Plant** (100-499 ticks): Produces berries at 10% per tick
- **Death** (500+ ticks): Dies and returns nutrients to soil

**Physics Impact:**
- Consumes **0.001 fertility/tick** from soil
- Consumes **0.0005 moisture/tick** from soil
- Returns **15% fertility** to soil on death

### 2. SeedComponent

Dormant plant embryo that germinates under suitable conditions.

**Attributes:**
```python
plant_type: str           # Type of plant to grow into
grow_time: int           # Ticks needed in soil to germinate (50)
time_in_soil: int        # Ticks spent in suitable conditions
required_fertility: float # Minimum soil fertility (0.3)
required_moisture: float  # Minimum soil moisture (0.2)
max_age: int             # Maximum age before rot (200)
```

**Germination Requirements:**
1. On plantable soil (not rock/water)
2. `time_in_soil >= grow_time` (50 ticks)
3. `soil.fertility >= 0.3`
4. `soil.moisture >= 0.2`
5. Not blocked by an object with `blocks_growth=True`
6. **Neighbourhood not saturated**: fewer than `max_neighbor_plants` (default 3)
   plants within `neighbor_radius` (default 2 → a 5×5 window). This is the
   carrying-capacity check added by the **B5** fix; without it the plant
   population grew unbounded (see W1).
7. Pass 75% probability check (× temperature window when the environment engine is on)
8. `time_in_soil < max_age` (not rotted)

**Lifecycle:**
- **Fresh Seed** (0-49 ticks): Aging, waiting for conditions
- **Ready Seed** (50-200 ticks): Can germinate if conditions met
- **Rotted Seed** (200+ ticks): Disappears, removed from world

### 3. EdibleComponent

Food resource with caloric value and decay rate.

**Attributes:**
```python
calories: float    # Energy provided when consumed (20.0)
toxicity: float    # Poison level (0.0 = safe)
freshness: float   # Quality level, 1.0 = fresh, 0.0 = spoiled
```

**Decay Mechanics:**
- Freshness decreases by **0.01 per tick**
- Takes **100 ticks** to completely spoil (1.0 → 0.0)
- When `freshness <= 0.0`:
  - Berry is removed from world
  - **70% chance** to drop a seed at location
  - Returns **15% fertility** to soil

### 4. FertilizerComponent

Temporary soil enhancement that boosts fertility.

**Attributes:**
```python
fertility_boost: float  # Amount of fertility to add
radius: int            # Tiles affected (Manhattan distance)
duration: int          # Ticks remaining
```

**Mechanics:**
- Affects all tiles within radius (Manhattan distance)
- Applies gradual fertility increase per tick
- Expires after duration ticks
- Multiple fertilizers stack

---

## Agent System

### Overview

Autonomous agents inhabit the world, each with a neural network brain that evolves over generations. Agents must survive by gathering food, managing energy, and making intelligent decisions using only primitive actions.

**Key Features:**
- 🧠 **Neural Network Brains** - GRU Actor-Critic with evolved weights + lifetime learning
- 🧬 **Genetic Evolution** - Crossover, mutation, selection
- 👁️ **Perception System** - 72-feature observation vector (egocentric vision)
- ⚡ **Energy Metabolism** - Must eat to survive
- 🎒 **Inventory System** - Carry items (seeds, food)
- 🎯 **Primitive Actions** - No high-level behaviors hardcoded

### Agent Architecture

```python
class Agent:
    # Position & State
    x, y: int                  # World coordinates
    direction: tuple[int, int] # Facing direction (N/E/S/W)
    energy: float              # Current energy (0.0 - max_energy)
    age: int                   # Age in ticks
    alive: bool                # Living status
    
    # Evolution
    genome: Genome             # Genetic information
    brain: Brain               # Neural network policy
    traits: dict[str, float]   # Phenotypic traits
    fitness: float             # Fitness score for selection
    
    # Inventory
    inventory: list[int]       # Object IDs (max 5 items)
    inventory_size: int        # Maximum capacity
    
    # Metabolism
    metabolism_rate: float     # Energy consumed per tick
    max_energy: float          # Maximum energy capacity (100-200)
    max_age: int               # Maximum lifespan (1000 ticks)
```

### Action Space (8 Primitive Actions)

Agents can only perform low-level actions. Complex behaviors must emerge through evolution.

| Action | Energy Cost | Description |
|--------|-------------|-------------|
| **MOVE_FORWARD** | 2.0 | Move one tile in current direction |
| **TURN_LEFT** | 0.5 | Rotate 90° counter-clockwise |
| **TURN_RIGHT** | 0.5 | Rotate 90° clockwise |
| **PICK_UP** | 1.0 | Pick up object from current tile |
| **DROP** | 1.0 | Drop held object onto current tile |
| **EAT** | 1.0 | Consume edible item from inventory |
| **USE** | 2.0 | Plant seed or apply fertilizer |
| **WAIT** | 0.1 | Do nothing (conserve energy) |

**Design Philosophy:** No "FARM", "BUILD", or "HUNT" actions. All complex behaviors emerge from combining primitives.

### Neural Network Brain

**Architecture:**
```
Input (72-dim observation)
  ↓
Encoder MLP
  ↓
GRU (memory / hidden state)
  ↓
Policy Head (8 action logits) + Value Head (1 scalar)
```

**Key Properties:**
- Weights encoded in genome (and updated online via Actor-Critic learning)
- Stochastic policy (samples from a masked softmax distribution)
- Action masking prevents invalid actions from stealing probability mass
- NumPy-first; optional Torch backend for acceleration

**Note:** If `allow_stacking: false`, `DROP` is masked off unless there is a legal drop location.

### Observation System (72 Features)

Agents perceive their environment through a normalized feature vector:

#### 1. Agent Internal State (8 features)
```python
[0] Energy ratio (0-1)
[1] Age ratio (0-1)
[2-5] Direction one-hot [N, E, S, W]
[6] Has inventory space (0 or 1)
[7] Metabolism rate (normalized)
```

#### 2. Vision Grid (50 features)
5×5 grid around agent (2 tiles in each direction):
- Each tile encoded as 2 features:
  - **Type:** 0=rock, 0.25=water, 0.5=soil, 0.75=plant, 1.0=food
  - **Value:** fertility/moisture or food/plant properties

**Egocentric frame:** the 5×5 grid is rotated so the agent's facing direction is always "up" in observation space.

#### 3. Stimulus Features (8 features)
Pre-processed survival signals (explicit hints like local danger/need) appended after vision.

#### 4. Inventory Summary (6 features)
```python
[66] Inventory fullness (0-1)
[67] Has food (0 or 1)
[68] Has seed (0 or 1)
[69] Has fertilizer (0 or 1)
[70] Total food calories (normalized)
[71] Inventory count (normalized)
```

### Genome & Evolution

**Genome Structure:**
```python
class Genome:
    weights: np.ndarray        # 2,744 neural network weights
    traits: dict[str, float]   # Phenotypic traits
    lineage_id: int            # Family tree tracking
    generation: int            # Generation number
    parent_ids: tuple[int, int] # Parent lineages (currently single parent)
```

**Traits:**
- `metabolism_rate`: Energy consumption multiplier (0.5-2.0)
- `vision_radius`: How far agent can see (2.0-10.0)

**Current Reproduction System: Asexual Fission**

The current implementation uses **single-parent asexual reproduction**:
- Parent creates genetic copy of itself
- Offspring genome = parent genome + Gaussian mutation
- No crossover/recombination (requires 2 parents)
- Mutation provides genetic diversity
- Learning configuration transferred from parent

**Genetic Operators:**

1. **Asexual Reproduction (Current Implementation)**:
   - **Fission**: Single parent creates offspring
   - **Cloning**: Offspring is copy of parent genome
   - **Mutation**: Gaussian noise added to all weights (std=0.02)
   - **Trait Inheritance**: Metabolism and vision inherited with variation
   - **Learning Transfer**: Offspring inherits learning configuration
   - **Energy Cost**: Parent loses 20-40% of current energy

2. **Crossover** (for future sexual reproduction):
   - **Uniform**: Each gene from random parent (PLANNED)
   - **One-point**: Split at random position (PLANNED)
   - **Blend**: Weighted average of parents (PLANNED)
   - **Note:** Requires two-parent mating system (not yet implemented)

3. **Mutation** (Currently Active):
   - Weight mutation: Gaussian noise (std=0.02 from config)
   - Trait mutation: Gaussian noise (std=0.05)
   - Applied to all offspring during fission

4. **Selection** (for evolutionary generations - PLANNED):
   - Tournament selection (k=5)
   - Fitness-based ranking
   - Elitism (preserve top 2 agents)
   - **Note:** Currently reproduction is in-simulation, not generational

### Agent Lifecycle

#### Birth (Asexual Fission)
```
1. Parent checks reproduction conditions
   - Energy >= 60% of max (configurable)
   - Age >= 100 ticks (configurable)
   - Cooldown period elapsed (50-70 ticks)
   
2. Offspring genome created via cloning
   - Parent genome copied
   - Gaussian mutation applied (std=0.02)
   - Traits inherited with small variation
   
3. Brain initialized from mutated genome
   
4. Spawn near parent
   - Find empty adjacent tile (3x3 grid)
   - Random selection of valid positions
   - Must not be occupied by other agents
   
5. Energy transfer
   - Offspring starts with FULL energy (100% of max_energy)
   - Parent loses configured % (20-40% of current energy)
   
6. Learning inheritance
   - If parent has learning enabled, offspring inherits:
     - Learning rate
     - Discount factor
     - Batch size
     - Exploration rate (epsilon)
```

#### Life
```
Every tick:
1. Observe environment (72-feature vector)
2. Brain decides action (masked stochastic policy)
3. Execute action (interact with world)
4. Consume energy (metabolism + action cost)
5. Update fitness (reward for survival)
6. Age by 1 tick
```

#### Death Conditions
- Energy reaches 0 (starvation)
- Age reaches max_age (old age, 1000 ticks)
- On death: Drop all inventory items

#### Reproduction (Asexual Fission)
- **Type:** Asexual reproduction (single parent)
- **Energy:** Parent must have ≥60% of max energy (configurable)
- **Age:** Parent must be ≥100 ticks old (configurable)
- **Mechanism:** Parent creates genetic copy with mutations
- **Energy Transfer:** Parent loses configured % of energy (20-40%)
- **Offspring:** Spawns at full energy in nearby empty tile
- **Learning Transfer:** Offspring inherits parent's learning configuration
- **Mutation:** Offspring genome has Gaussian noise applied (std=0.02)

### Agent Statistics

**Initial Population:** 20 agents  
**Spawn Rate:** Random on passable tiles  
**Starting Energy:** 100.0 (max_energy)  
**Average Lifespan:** ~200 ticks (without food)  
**Energy from Berry:** 20 calories  
**Metabolism Cost:** 0.5 energy/tick (base rate)

### Visual Representation (GUI)

Agents rendered as triangles:
- **Color** indicates energy level:
  - 🟢 Green: >70% energy (healthy)
  - 🟠 Orange: 30-70% energy (hungry)
  - 🔴 Red: <30% energy (starving)
- **Direction** shown by triangle orientation
- **Hover** to see: energy, age, inventory, fitness

### Integration with World

Agents interact with the ecosystem through actions:

**Eating:**
```python
# Agent picks up berry
PICK_UP → berry in inventory
# Agent consumes berry
EAT → energy += 20, berry destroyed
```

**Planting:**
```python
# Agent picks up seed
PICK_UP → seed in inventory
# Agent plants seed
USE → seed placed on soil tile
# Seed germinates (probabilistic)
→ becomes plant after 50 ticks (if conditions met)
```

**Movement:**
```python
# Agent explores
MOVE_FORWARD → position changes
TURN_LEFT/RIGHT → direction changes
# Must avoid rocks and world boundaries
```

**Reproduction (Asexual Fission):**
```python
# Parent checks if can reproduce
if agent.can_reproduce(config):  # Energy >= 60%, Age >= 100
    # Create offspring via cloning + mutation
    offspring = agent.reproduce(world, config)
    
    # Process:
    # 1. Clone parent genome
    # 2. Apply Gaussian mutation (std=0.02)
    # 3. Offspring gets FULL energy (100% of max)
    # 4. Parent loses 20-40% of current energy
    # 5. Offspring spawns in adjacent empty tile
    # 6. Offspring inherits learning configuration
    # 7. Generation counter increments
```

### Future Enhancements

- [ ] **Sexual reproduction** (two-parent mating with crossover)
- [ ] **Tournament selection** (for generational evolution)
- [ ] **Generation management** (discrete generations)
- [ ] **Fitness evaluation functions** (multi-objective)
- [ ] **Communication system** (signal emission between agents)
- [ ] **Tool usage** (crafting and construction)
- [ ] **Curiosity-driven exploration** (intrinsic motivation)
- [ ] **Learned world models** (predictive models)

**Note:** Current reproduction is **asexual fission** - single parent creates mutated offspring. Sexual reproduction with crossover is planned for future versions.

---

## Ecosystem Physics

### Soil Fertility Dynamics

Fertility represents soil nutrients required for plant growth.

#### Consumption (by plants)
```python
# Per plant per tick
tile.fertility -= 0.001
```

#### Recovery (empty soil)
```python
# Per tick on unoccupied plantable tiles
tile.fertility += 0.0005
```

#### Nutrient Return (decomposition)
```python
# When plant dies or berry rots
tile.fertility += 0.15  # 15% returned
```

#### Balance Analysis

| Scenario | Consumption | Recovery | Net Change |
|----------|-------------|----------|------------|
| Empty soil | 0 | +0.0005 | +0.0005 ✅ |
| With plant | -0.001 | 0 | -0.001 ❌ |
| Plant death | 0 | +0.15 | +0.15 ✅ |

**Result:** Sustainable with natural death/decomposition cycle

### Soil Moisture Dynamics

Moisture represents water content needed for germination and growth.

#### Consumption (by plants)
```python
# Per plant per tick
tile.moisture -= 0.0005
```

#### Evaporation (natural loss)
```python
# Per tick on all plantable tiles
tile.moisture -= 0.0002
```

#### Recovery (rain/groundwater)
```python
# Per tick on all plantable tiles
tile.moisture += 0.0008
```

#### Balance Analysis (legacy, environment **disabled**)

| Scenario | Consumption | Evaporation | Recovery | Net Change |
|----------|-------------|-------------|----------|------------|
| Empty soil | 0 | -0.0002 | +0.0008 | +0.0006 ⚠️ |
| With plant | -0.0005 | -0.0002 | +0.0008 | +0.0001 ⚠️ |

> ⚠️ **This was bug B1, not a feature.** "Net positive moisture in all
> scenarios" means moisture only ever *climbs* — it saturates at 1.0 within
> ~1.5k ticks, so the moisture dimension carries no information and every
> germination check passes. This legacy arithmetic is preserved bit-for-bit
> only when `environment.enabled: false`.
>
> **With the environment engine on (W1)** moisture becomes a real,
> time-varying constraint: evaporation scales with temperature and light
> (and doubles during droughts), and recovery arrives **only** during rain
> events or on tiles adjacent to water. See the
> [World Upgrade](#world-upgrade-w0w1--june-2026) section.

### Probabilistic Germination

Not all seeds succeed, creating natural selection pressure.

```python
# After meeting all conditions:
if random.random() < 0.75:  # 75% success rate
    germinate_seed()
else:
    remove_seed()  # Failed to germinate
```

**Purpose:**
- Prevents overpopulation
- Creates realistic failure rates
- Adds strategic depth
- Mimics natural germination success

### Seed Aging & Rot

Seeds have limited viability, preventing infinite accumulation.

```python
# Every tick
seed.time_in_soil += 1

# Check for rot
if seed.time_in_soil >= seed.max_age:  # 200 ticks
    remove_seed()  # Rotted away
```

**Timeline:**
- **0-50 ticks:** Too young to germinate
- **50-200 ticks:** Can germinate if conditions met
- **200+ ticks:** Rots and disappears

---

## System Dynamics

The simulation runs the world systems each tick in a specific order. With
the **environment engine** (W1) the `EnvironmentSystem` runs *first*, before
any other system, so the multipliers it publishes (light, temperature,
weather) are seen by everything downstream that tick.

### System Execution Order

```
0. EnvironmentSystem      → (W1, opt-in) advances clock + weather, publishes
                            light/temperature/rain multipliers (no-op when
                            environment.enabled is false)
1. PlantGrowthSystem      → Ages plants (× growth multiplier), handles death
2. SeedGerminationSystem  → Converts seeds to plants (× germination multiplier,
                            carrying-capacity check)
3. DecaySystem            → Degrades berries (× decay multiplier), spawns seeds
4. FertilizerSystem       → Boosts nearby soil
5. SoilDynamicsSystem     → Updates fertility/moisture (evaporation/recovery
                            model depends on environment.enabled)
6. ResourceSpawnSystem    → Produces berries from mature plants (× light)
```

> Agent metabolism is also scaled by the environment temperature multiplier
> (in both the serial and parallel agent paths) — extremes cost more energy.

### 1. PlantGrowthSystem

**Purpose:** Manage plant lifecycle and aging

**Operations per tick:**
```python
for each plant:
    plant.age += 1
    
    if plant.age >= max_age (500):
        remove_plant()
        return 15% fertility to soil
```

**Impact:**
- Plants age naturally
- Old plants die and return nutrients
- Prevents immortal plants
- Creates space for new growth

### 2. SeedGerminationSystem

**Purpose:** Convert seeds into plants when conditions are met

**Operations per tick:**
```python
for each seed:
    seed.time_in_soil += 1
    
    # Check for rot
    if seed.time_in_soil >= max_age (200):
        remove_seed()
        continue
    
    # Check germination conditions
    if time_in_soil >= grow_time (50):
        if soil.fertility >= 0.3:
            if soil.moisture >= 0.2:
                if not blocks_growth(tile):
                    # B5 carrying capacity: competition for space/light
                    if plants_in_radius(x, y, neighbor_radius) < max_neighbor_plants:
                        if random() < 0.75 * temp_window:  # success check
                            convert_to_plant()
                        else:
                            remove_seed()  # Failed
                    # else: seed waits (and eventually rots) — no new plant
```

**Impact:**
- Seeds become plants
- Natural selection via probability
- Soil requirements create spatial patterns
- Seed rot prevents accumulation
- **Carrying capacity (B5):** local crowding caps plant density, so the
  population plateaus instead of tiling the world

### 3. DecaySystem

**Purpose:** Handle resource degradation and seed production

**Operations per tick:**
```python
for each berry:
    berry.freshness -= 0.01
    
    if berry.freshness <= 0:
        remove_berry()
        return 15% fertility to soil
        
        if random() < 0.7:  # 70% seed drop chance
            spawn_seed_at_location()
```

**Impact:**
- Creates urgency to harvest fresh berries
- Produces seeds from decomposed fruit
- Returns nutrients to soil
- Completes reproduction cycle

### 4. FertilizerSystem

**Purpose:** Apply temporary soil enhancements

**Operations per tick:**
```python
for each fertilizer:
    fertilizer.duration -= 1
    
    if fertilizer.duration <= 0:
        remove_fertilizer()
    else:
        for each tile in radius:
            tile.fertility += boost / 10.0  # Gradual increase
```

**Impact:**
- Boosts soil fertility in radius
- Temporary enhancement
- Enables strategic soil management
- Multiple fertilizers stack

### 5. SoilDynamicsSystem

**Purpose:** Manage soil resource dynamics

**Operations per tick:**
```python
# First pass: plant consumption
for each plant at (x, y):
    tile = get_tile(x, y)
    tile.fertility -= 0.001
    tile.moisture -= 0.0005

# Second pass: all soil tiles
for each plantable tile:
    # Moisture dynamics — TWO models, selected by environment.enabled:
    #   legacy (disabled): constant evaporation + constant unconditional
    #                      recovery (the B1 bug — net always positive)
    #   environment (W1):  evaporation scales with temperature/light (×2 in
    #                      drought); recovery only during rain or next to water
    tile.moisture -= evaporation
    tile.moisture += recovery        # rain event and/or water-adjacency only when enabled

    # Fertility recovery (empty tiles only)
    if no plant at tile:
        tile.fertility += 0.0005
```

**Impact:**
- Plants deplete soil resources
- Empty soil recovers naturally
- Legacy mode: moisture only rises (bug B1); environment mode: moisture is
  a real, weather-driven constraint that can fall to zero in a drought

### 6. ResourceSpawnSystem

**Purpose:** Produce berries from mature plants + safety spawning

**Operations per tick:**
```python
# Plant-based spawning
for each mature_plant:
    if random() < spawn_rate (0.1):
        spawn_berry_nearby()

# Safety net spawning
if edible_count < min_resources (10):
    if random() < safety_spawn_rate (0.01):
        spawn_berry_random_location()
```

**Impact:**
- Mature plants produce food
- Safety net prevents total depletion
- Creates resource distribution
- Maintains minimum viability

---

## Complete Lifecycle

### Full Resource Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUSTAINABLE ECOSYSTEM                         │
│         (All Components, Physics, and Cycles Integrated)         │
└─────────────────────────────────────────────────────────────────┘

1. SEED (SeedComponent)
   ├─ Age: 0-200 ticks (max_age before rot)
   ├─ Location: On soil tile
   ├─ Physics: No impact on soil
   └─ Conditions: Waiting for fertility >= 0.3, moisture >= 0.2
   
   ↓ time_in_soil >= 50 ticks + conditions met + 75% chance
   
2. YOUNG PLANT (PlantComponent, age < 100)
   ├─ Age: 0-99 ticks
   ├─ Physics:
   │   ├─ Consumes 0.001 fertility/tick from soil
   │   └─ Consumes 0.0005 moisture/tick from soil
   └─ Status: Growing, not yet producing
   
   ↓ age reaches 100 ticks
   
3. MATURE PLANT (PlantComponent, age 100-499)
   ├─ Age: 100-499 ticks
   ├─ Physics: Same consumption as young plant
   ├─ Production: 10% chance/tick to spawn berry
   └─ Status: Actively producing resources
   
   ↓ 10% chance per tick
   
4. BERRY SPAWNING (EdibleComponent created)
   ├─ Spawns adjacent to mature plant
   ├─ Initial: freshness = 1.0, calories = 20.0
   └─ Physics: No impact on soil
   
   ↓ freshness -= 0.01/tick (100 ticks total lifespan)
   
5. FRESH BERRY (EdibleComponent, freshness > 0.5)
   ├─ Freshness: 1.0 → 0.5 (50 ticks)
   ├─ Status: Good for harvesting
   └─ Physics: No impact
   
   ↓ continues aging
   
6. OLD BERRY (EdibleComponent, freshness 0.1-0.5)
   ├─ Freshness: 0.5 → 0.0 (50 ticks)
   ├─ Status: Still edible but degraded
   └─ Physics: No impact
   
   ↓ freshness reaches 0.0
   
7. DECOMPOSITION (Berry removed)
   ├─ Berry removed from world
   ├─ Returns 15% fertility to soil tile
   ├─ 70% chance: spawn seed at location
   └─ 30% chance: no seed (natural loss)
   
   ↓ if seed spawned
   
8. NEW SEED → CYCLE REPEATS (back to step 1)

PARALLEL PROCESSES:

SOIL FERTILITY CYCLE:
├─ Plant consumption: -0.001/tick
├─ Empty recovery: +0.0005/tick
├─ Death/decay bonus: +0.15 (one-time)
└─ Net result: Sustainable with natural turnover

SOIL MOISTURE CYCLE:
├─ Plant consumption: -0.0005/tick
├─ Evaporation: -0.0002/tick
├─ Rain/groundwater: +0.0008/tick
└─ Net result: +0.0001/tick (always positive!)

PLANT DEATH (age >= 500):
├─ Plant removed from world
├─ Returns 15% fertility to soil
└─ Opens space for new plants

SEED ROT (age >= 200):
├─ Seed removed if not germinated
└─ Prevents infinite accumulation

SAFETY SPAWNING:
├─ If resources < 10
├─ 1% chance/tick to spawn berry
└─ Prevents world depletion
```

### Population Dynamics

```
Birth Factors (Increase):
├─ Mature plants spawning berries (10%/tick)
├─ Berries decomposing into seeds (70% chance)
├─ Seeds germinating into plants (75% success)
└─ Safety spawning when depleted (1%/tick)

Death Factors (Decrease):
├─ Plants dying from old age (at 500 ticks)
├─ Seeds failing to germinate (25% failure)
├─ Seeds rotting from age (at 200 ticks)
├─ Berries decomposing without seeds (30% chance)
└─ Resource harvesting by agents

Equilibrium:
└─ System self-balances through birth/death rates
```

---

## Population Control & Environmental Disasters

### Maximum Population Limit

**Purpose:** Prevent unlimited exponential population growth

**Configuration:**
```yaml
reproduction:
  max_population: 50  # null = unlimited
```

**Behavior:**
```python
# Before allowing reproduction:
current_population = len(agents) + len(new_offspring)
if current_population >= max_population:
    skip_reproduction()  # Wait for population to decrease
```

**Console Output:**
```
[REPRODUCTION] Agent 12 (Gen 3, Age 150) reproduced! 
               Parent energy: 420.0 → 336.0 (lost 84.0)
               Offspring: Agent 28, Gen 4 
               pop: 48/50
```

**Key Features:**
- Reproduction stops when at/above limit
- Resumes automatically when population drops
- Natural deaths create space for new offspring
- Prevents resource exhaustion from overpopulation

### Calamity System

**Purpose:** Create periodic environmental disasters that destroy resources, simulating natural catastrophes like droughts, fires, floods, or disease outbreaks.

**Why Calamities Matter:**
1. **Survival Pressure** - Agents must survive resource scarcity
2. **Selection Pressure** - Only adaptable agents survive disasters
3. **Prevents Overpopulation** - Periodic culling maintains balance
4. **Strategic Depth** - Agents learn to stockpile and adapt
5. **Realistic Dynamics** - Mimics real-world environmental disasters

#### Configuration

```yaml
calamity:
  enabled: true              # Toggle disasters on/off
  interval: 500              # Ticks between disasters
  destruction_rate: 0.30     # 30% of resources destroyed
  affect_plants: true        # Destroy plants
  affect_food: true          # Destroy berries/food
  affect_seeds: false        # Preserve seeds for recovery
```

#### Implementation Details

**Timing:**
```python
# Check every tick
if tick - last_calamity_tick >= interval:
    trigger_calamity()
    last_calamity_tick = tick
```

**Destruction Process:**
```python
def _trigger_calamity():
    destroyed_counts = {"plants": 0, "food": 0, "seeds": 0}
    
    # Collect objects by type
    plants = [obj for obj in objects if has PlantComponent]
    food = [obj for obj in objects if has EdibleComponent]
    seeds = [obj for obj in objects if has SeedComponent]
    
    # Destroy based on configuration
    if affect_plants:
        for plant in plants:
            if random() < destruction_rate:
                # Return nutrients to soil
                tile.fertility += 0.15
                remove_object(plant)
                destroyed_counts["plants"] += 1
    
    if affect_food:
        for berry in food:
            if random() < destruction_rate:
                remove_object(berry)
                destroyed_counts["food"] += 1
    
    if affect_seeds:
        for seed in seeds:
            if random() < destruction_rate:
                remove_object(seed)
                destroyed_counts["seeds"] += 1
    
    # Print report
    print_calamity_report(destroyed_counts)
```

**Nutrient Cycling:**
- Destroyed plants return nutrients to soil (15% fertility)
- Maintains ecosystem balance
- Prevents soil depletion from mass destruction

#### Console Output

```
⚠️  [CALAMITY] Tick 500: Environmental disaster struck!
   Destroyed 12 objects (30.0% destruction rate)
   Plants destroyed: 7
   Food destroyed: 5
   Seeds destroyed: 0 (preserved)
   Remaining objects: 48
```

#### Survival Strategies

Agents must learn to:
1. **Stockpile Resources** - Keep food in inventory before disasters
2. **Energy Management** - Maintain high energy reserves
3. **Spatial Awareness** - Spread out to avoid localized impacts
4. **Quick Recovery** - Exploit preserved seeds for regrowth
5. **Adaptive Behavior** - Survive periodic resource scarcity

#### Configuration Presets

**Mild Disasters:**
```yaml
calamity:
  interval: 1000             # Every 1000 ticks
  destruction_rate: 0.15     # 15% destroyed
  affect_seeds: false        # Seeds survive
```

**Moderate Disasters:**
```yaml
calamity:
  interval: 500              # Every 500 ticks
  destruction_rate: 0.30     # 30% destroyed
  affect_seeds: false        # Seeds survive
```

**Severe Catastrophes:**
```yaml
calamity:
  interval: 300              # Every 300 ticks
  destruction_rate: 0.50     # 50% destroyed
  affect_seeds: true         # Everything destroyed
```

**Apocalyptic Events:**
```yaml
calamity:
  interval: 200              # Every 200 ticks
  destruction_rate: 0.75     # 75% destroyed
  affect_seeds: true         # Total devastation
```

#### Testing Results

**Test Setup:** 40 objects (20 plants, 10 food, 10 seeds)  
**Configuration:** 30% destruction rate, seeds preserved

**Results:**
```
✅ Calamity triggered at tick 500
✅ Plants destroyed: 7/20 (35%) - expected ~30%
✅ Food destroyed: 5/10 (25%) - expected ~30%
✅ Seeds preserved: 0/10 (0%) - expected 0%
✅ Remaining objects: 48 total
✅ Nutrients returned to soil: 7 × 0.15 = 1.05 total fertility
```

**Statistical Validation:**
- Destruction rates within expected variance (±5%)
- Seeds correctly preserved when configured
- Nutrient cycling functioning properly
- Console output accurate and informative

### Combined Population Dynamics

```
Population Growth:
├─ Agent reproduction (when below max_population)
├─ Energy threshold: 60% of max energy
├─ Minimum age: 100 ticks
├─ Cooldown: 50-70 ticks between reproductions
└─ Parent loses 20-40% of energy

Population Decline:
├─ Natural death (energy = 0 or age = max_age)
├─ Starvation (failed to find food)
├─ Calamity-induced resource scarcity
└─ Competition for limited resources

Equilibrium:
├─ Max population prevents unlimited growth
├─ Calamities create periodic culling
├─ Natural selection favors efficient survivors
└─ System self-balances through multiple pressures
```

---

## Configuration Reference

All ecosystem parameters are configurable via `config/default.yaml`:

### World Settings
```yaml
world:
  width: 100                    # World grid width
  height: 100                   # World grid height
  initial_resources: 50         # Starting objects
  resource_spawn_rate: 0.01     # Safety spawn probability
  seed: null                    # Random seed (null = random)
```

### Terrain Settings
```yaml
terrain:
  soil_ratio: 0.65             # 65% soil tiles
  rock_ratio: 0.2              # 20% rock tiles
  water_ratio: 0.1             # 10% water tiles
  sand_ratio: 0.05             # 5% sand tiles (spreading hazard)
  fertility_range: [0.3, 1.0]  # Initial soil fertility
  moisture_range: [0.2, 0.8]   # Initial soil moisture
```

### Plant Settings
```yaml
plants:
  growth_time: 50               # Seed germination time
  mature_age: 100               # Age when plants mature
  max_age: 500                  # Age when plants die
  seed_spawn_rate: 0.1          # Berry production rate
  required_fertility: 0.3       # Germination fertility minimum
  required_moisture: 0.2        # Germination moisture minimum
  germination_success_rate: 0.75 # Probability of germination
  fertility_consumption_per_tick: 0.001  # Plant fertility use
  moisture_consumption_per_tick: 0.0005  # Plant moisture use
  seed_max_age: 200             # Seed rot age
  max_neighbor_plants: 3        # B5 carrying capacity: max plants in the
                                # crowding window before germination is blocked
                                # (0 = disabled / legacy unbounded growth)
  neighbor_radius: 2            # Chebyshev radius of that window (2 = 5×5)
```

### Resource Settings
```yaml
resources:
  berry_calories: 20.0          # Energy from berries
  berry_freshness_decay: 0.01   # Decay rate per tick
  seed_drop_chance: 0.7         # Decomposition seed spawn rate
```

### Soil Dynamics Settings
```yaml
soil:
  fertility_recovery_rate: 0.0005      # Empty soil recovery
  fertility_return_on_death: 0.15      # Decomposition nutrient return
  moisture_evaporation_rate: 0.0002    # Natural moisture loss
  moisture_recovery_rate: 0.0008       # Rain/groundwater recovery
  max_fertility: 1.0                   # Maximum fertility cap
  min_fertility: 0.0                   # Minimum fertility floor
  max_moisture: 1.0                    # Maximum moisture cap
  min_moisture: 0.0                    # Minimum moisture floor
```

> When `environment.enabled: true`, `moisture_evaporation_rate` and
> `moisture_recovery_rate` above are **superseded** by the environment
> engine's weather-driven model (see below); the fertility settings still
> apply.

### Environment Engine Settings (W1)
```yaml
environment:
  enabled: false               # Master switch (false = legacy static climate)
  day_length: 200              # Ticks per full day/night cycle
  min_light: 0.25              # Light level at deepest night (1.0 = noon)
  season_length: 2000          # Ticks per full seasonal cycle
  season_temp_amplitude: 0.25  # Seasonal temperature swing around base
  base_temperature: 0.5        # Yearly mean (0.5 = centre of comfort band)
  daynight_temp_amplitude: 0.10  # Day-vs-night temperature swing
  metabolism_temp_coef: 0.5    # Extra agent energy drain at temp extremes
  base_evaporation: 0.0012     # Soil moisture loss/tick (× temperature & light)
  water_adjacency_recovery: 0.002  # Moisture gain/tick next to WATER tiles
  weather:
    rain_start_chance: 0.01    # Per-tick chance rain begins (clear weather)
    rain_duration: 60          # Ticks a rain event lasts
    rain_recovery: 0.004       # Moisture gain/tick while raining
    drought_start_chance: 0.002  # Per-tick chance a drought begins
    drought_duration: 150      # Ticks a drought lasts
    drought_evaporation_factor: 2.0  # Evaporation multiplier in drought
```

**What each multiplier feeds (all exactly 1.0 when disabled):**
- **light** = `min_light + (1−min_light)·½(1+sin(2π·t/day_length))` → plant
  growth, food production
- **temperature** = `base + seasonal sin + day/night offset` (clamped to
  [0,1]) → growth window, germination window, decay rate, agent metabolism
- **temperature_response** = 1.0 in [0.3, 0.7], linear falloff to 0 at
  0.1/0.9 (the "comfort band")
- **rain / drought** → the only moisture recovery (rain) and amplified
  evaporation (drought)

### Sand / Tile-Effect Settings (W0 / B2)
```yaml
sand:
  spread_interval: 200          # Ticks a neighbour must be unprotected before spread
  spread_chance: 0.05           # Per-tick spread probability once interval met
  spread_radius: 1              # Manhattan distance for spreading
  spread_blocked_by: [plant]    # Object categories that block spreading
  germination_multiplier: 0.1   # 10× harder to germinate on sand
  growth_multiplier: 0.1        # 10× slower growth on sand
  spawn_rate_multiplier: 0.3    # 70% less food production on sand
  fertility_override: 0.30      # Clamp fertility on sand (B2: AT the seed
                                # threshold so the ×0.1 multiplier is the difficulty)
  moisture_override: 0.20       # Clamp moisture on sand (B2: AT the seed threshold)
  reclaim_terrain: soil         # Terrain a plant on sand reclaims it back to
  reclaim_interval: 150         # Ticks a plant must sit on sand to reclaim it
```

> **B2 fix:** sand previously clamped fertility/moisture to 0.05 — strictly
> below the seed thresholds (0.3/0.2) — so its ×0.1 germination multiplier
> never even applied and germination on sand was *impossible*. The clamps
> now sit **at** the thresholds, so sand is genuinely 10× harder, not a
> dead zone.

### Agent Settings
```yaml
agents:
  initial_population: 20       # Starting agent count
  starting_energy: 100.0       # Initial energy
  max_energy: 200.0            # Maximum energy capacity
  metabolism_rate: 0.5         # Energy consumed per tick
  vision_radius: 5             # Tiles visible in each direction
  max_age: 1000                # Maximum lifespan in ticks
  inventory_size: 5            # Maximum items carried
```

### Learning Settings
```yaml
learning:
  enabled: false               # Enable reinforcement learning
  learning_rate: 0.01          # Policy gradient step size
  discount_factor: 0.95        # Future reward discount (gamma)
  batch_size: 16               # Experiences per training batch
  buffer_capacity: 1000        # Maximum replay buffer size
  epsilon: 0.15                # Exploration rate (random actions)
```

### Reproduction Settings
```yaml
reproduction:
  enabled: false               # Enable in-simulation reproduction
  energy_threshold: 0.60       # 60% of max_energy required
  min_age: 100                 # Minimum age in ticks
  energy_split: 0.40           # Parent loses this % of energy
  mutation_std: 0.02           # Mutation standard deviation
  cooldown_ticks: 50           # Ticks between reproductions
  max_population: 100          # Maximum total population (null = unlimited)
```

**Reproduction Mechanics (Asexual Fission):**
- **Type:** Single-parent asexual reproduction (not sexual/mating)
- **Energy Check:** `agent.energy >= energy_threshold × max_energy`
- **Age Check:** `agent.age >= min_age`
- **Cooldown Check:** `ticks_since_last_reproduction >= cooldown_ticks`
- **Population Check:** `current_population < max_population` (if set)
- **Energy Transfer:** Parent loses `energy_split` of current energy
- **Offspring Energy:** Offspring spawns with FULL energy (100% of max_energy)
- **Mutation:** Offspring genome mutated with Gaussian noise (std=`mutation_std`)
- **Inheritance:** Offspring inherits parent's traits with small variations
- **Learning Transfer:** If parent has learning enabled, offspring inherits configuration
- **Spawning:** Offspring placed in random adjacent empty tile (3x3 grid)

### Calamity Settings
```yaml
calamity:
  enabled: false               # Enable periodic calamities
  interval: 500                # Ticks between disasters
  destruction_rate: 0.30       # Percentage destroyed (0.30 = 30%)
  affect_plants: true          # Whether to destroy plants
  affect_food: true            # Whether to destroy berries/food
  affect_seeds: false          # Whether to destroy seeds (false = preserved)
```

**Calamity Mechanics:**
- **Timing:** Triggers every `interval` ticks
- **Target Selection:** Random objects of specified types
- **Destruction:** Each object has `destruction_rate` chance of removal
- **Nutrient Return:** Destroyed plants return 15% fertility to soil
- **Recovery:** Preserved seeds enable ecosystem regrowth
- **Statistics:** Detailed console output after each calamity

### Object Stacking Settings
```yaml
world:
  allow_stacking: false        # If false, one object per tile (strict mode)
```

**Stacking Behavior:**
- **Strict Mode** (`false`): One object per tile, overflow placed nearby
- **Legacy Mode** (`true`): Multiple objects can stack on same tile

### Tuning Guide

**To make the ecosystem harder:**
- Decrease `germination_success_rate` (more seed failures)
- Decrease `seed_drop_chance` (fewer seeds from decay)
- Increase `required_fertility` / `required_moisture` (stricter conditions)
- Decrease `moisture_recovery_rate` (drier world)
- Decrease `seed_max_age` (seeds rot faster)
- **Enable calamities** with high destruction rates
- **Lower max_population** to increase competition

**To make the ecosystem easier:**
- Increase `germination_success_rate` (more successful seeds)
- Increase `seed_drop_chance` (more seeds from decay)
- Decrease `required_fertility` / `required_moisture` (easier germination)
- Increase `moisture_recovery_rate` (wetter world)
- Increase `seed_max_age` (seeds live longer)
- **Disable calamities** or reduce destruction rates
- **Raise max_population** to reduce competition

**To adjust sustainability:**
- Increase `fertility_recovery_rate` (faster soil recovery)
- Decrease `fertility_consumption_per_tick` (plants use less)
- Increase `fertility_return_on_death` (more nutrients returned)

**To adjust population dynamics:**
- Lower `energy_threshold` (easier reproduction)
- Lower `min_age` (earlier reproduction)
- Lower `energy_split` (parent retains more energy)
- Raise `max_population` (allow more agents)
- Decrease `cooldown_ticks` (faster reproduction)

**To adjust disaster impact:**
- Increase `calamity.interval` (less frequent disasters)
- Decrease `destruction_rate` (milder disasters)
- Set `affect_seeds: false` (preserve recovery mechanism)
- Set `affect_plants: false` (only destroy food)

**To create challenging survival scenarios:**
```yaml
# "Harsh World" preset
reproduction:
  enabled: true
  energy_threshold: 0.80      # Need 80% energy
  max_population: 30          # Limited population
  
calamity:
  enabled: true
  interval: 300               # Frequent disasters
  destruction_rate: 0.50      # 50% destruction
  affect_seeds: false         # But seeds survive
  
agents:
  metabolism_rate: 0.5        # Higher energy consumption
  starting_energy: 100        # Lower starting energy
```

**To create easy learning scenarios:**
```yaml
# "Easy Mode" preset
reproduction:
  enabled: true
  energy_threshold: 0.40      # Need only 40% energy
  max_population: 100         # Large population allowed
  
calamity:
  enabled: false              # No disasters
  
agents:
  metabolism_rate: 0.015      # Very low consumption
  starting_energy: 700        # High starting energy
  max_energy: 1000            # Large energy capacity
```

---

## Testing & Validation

### Test Suite

**381 tests passing (June 2026)** — includes the W0 object-validation suite,
the W1 environment suite (`test_environment.py`), and the B5 carrying-capacity
regression tests in `test_systems.py`.

Representative system tests include:

#### System Tests (`test_systems.py`)
```
PlantGrowthSystem:
  ✅ test_plant_ages_each_tick
  ✅ test_old_plants_die
  ✅ test_multiple_plants

SeedGerminationSystem:
  ✅ test_seed_germinates_on_suitable_soil
  ✅ test_seed_needs_time_in_soil

DecaySystem:
  ✅ test_freshness_decreases
  ✅ test_spoiled_objects_removed
  ✅ test_spoiled_berries_drop_seeds

FertilizerSystem:
  ✅ test_fertilizer_boosts_nearby_tiles
  ✅ test_fertilizer_expires

ResourceSpawnSystem:
  ✅ test_mature_plants_spawn_berries
  ✅ test_safety_spawn_when_depleted

WorldSystemManager:
  ✅ test_world_update_runs_all_systems
  ✅ test_systems_run_in_order
```

#### World Tests (`test_world.py`)
```
World Creation & Management:
  ✅ 21 additional tests covering:
     - Tile creation and properties
     - Object management
     - Coordinate validation
     - Terrain generation
     - Component queries
```

### Performance Benchmarks

Tested on 100x100 world with various object counts:

| Objects | Ticks/Second | Memory (MB) | CPU Usage |
|---------|--------------|-------------|-----------|
| 100 | 1000+ | 25 | < 5% |
| 500 | 500+ | 35 | ~10% |
| 1000 | 250+ | 50 | ~20% |
| 5000 | 100+ | 120 | ~40% |

**Result:** Scales efficiently to thousands of entities

### Sustainability Test

**Long-term viability verified:**
- ✅ 10,000 tick simulation completed
- ✅ Resource count remained stable (40-60 range)
- ✅ No resource depletion observed
- ✅ Fertility/moisture levels sustainable
- ✅ Population naturally balanced

---

## Reinforcement Learning System

**Implementation Date:** November 16, 2025  
**Status:** ✅ Production Ready - Agents surviving 2500+ ticks

### Overview

Agents learn survival behaviors during their lifetime through policy gradient reinforcement learning. The system enables agents to improve their neural network weights based on environmental feedback, leading to emergent survival strategies.

### Learning Architecture

```python
AgentLearner
    ├─ ReplayBuffer (capacity: 1000 experiences)
    ├─ RewardShaper (calculates rewards from actions)
    └─ learn() - Policy gradient with full backpropagation
```

### Key Components

#### 1. Experience Replay

Stores transitions for batch learning:
```python
Experience(
  state: np.ndarray,      # 72-feature observation
    action: int,            # Action taken (0-7)
    reward: float,          # Reward received
    next_state: np.ndarray, # Resulting observation
    done: bool              # Episode termination
)
```

#### 2. Reward Shaping

**Success Rewards:**
- Action success: +0.1
- Eating food: +10.0 (primary survival goal)
- Picking up items: +0.5
- Planting seeds: +0.3
- High energy (>80%): +0.1

**Penalties:**
- Action failure: -0.01
- Failed EAT attempt: -0.5 (discourages spam)
- Low energy (<20%): -0.05
- Death: -1.0

#### 3. Policy Gradient Learning

**Algorithm:** REINFORCE with baseline
- **Update Frequency:** Every 3 ticks (fast adaptation)
- **Batch Size:** 16 experiences
- **Learning Rate:** 0.01
- **Discount Factor:** 0.95

**Full Backpropagation:**
```python
# Forward pass: collect activations through all layers
activations = [input_layer, hidden1, hidden2, output_layer]

# Backward pass: compute gradients and update ALL weights/biases
for layer in reversed(network):
    compute_gradients(layer, returns)
    update_weights(layer, gradients)
    update_biases(layer, gradients)
    backpropagate_to_previous_layer()
```

### Training Configuration

**Optimized for Survival:**
```yaml
agents:
  starting_energy: 700.0      # High starting buffer
  max_energy: 1000.0          # Large capacity
  metabolism_rate: 0.015      # Very slow decay
  max_age: 5000               # Extended lifespan

learning:
  learning_rate: 0.01         # Standard gradient step
  discount_factor: 0.95       # Future-oriented
  batch_size: 16              # Efficient batching
  buffer_capacity: 1000       # Long memory
```

### Performance Results

#### Evolution of Survival Time

| Configuration | Survival (ticks) | Avg Death Age | Notes |
|--------------|------------------|---------------|-------|
| Baseline (bugs present) | 554 | ~550 | Learning disabled |
| Bug fixes applied | 1000-1400 | 1281 | Real gradients |
| Identical genomes | 815 | 790 | No diversity ❌ |
| Random genomes | 923 | 1052 | +33% improvement |
| **Final optimized** | **2500+** | **5000** | **3-4 survivors ✅** |

#### Behavioral Metrics

- **EAT Success Rate:** 7.1% (learned to eat when food available)
- **Movement Success:** 28.3% (navigating obstacles)
- **Food Consumption:** 446 successful eats over 2500 ticks
- **Survival Rate:** 30-40% reach age limit (5000)
- **Energy Management:** Survivors maintain 500-700 energy at end

### Critical Bug Fixes

#### 1. Reward Normalization Bug ❌→✅
**Problem:** Normalized rewards → zero gradients → no learning
```python
# BROKEN: Kills learning signal
returns = (returns - returns.mean()) / (returns.std() + 1e-8)

# FIXED: Use raw rewards
returns = np.array(returns)  # Keep original values
```

#### 2. Incomplete Backpropagation ❌→✅
**Problem:** Only output biases updated → limited learning
```python
# BROKEN: Only updates output layer
brain.biases[-1] -= learning_rate * gradient

# FIXED: Full backpropagation through all layers
for layer in all_layers:
    update_weights(layer)
    update_biases(layer)
    backpropagate_gradients()
```

#### 3. Double-Update Bug ❌→✅
**Problem:** Agents updated twice per tick → age = 2×tick
```python
# BROKEN: Updates agents twice
world.update()           # Updates agents internally
agent.update(world)      # Updates again!

# FIXED: Single update
world.update()           # Only call once
```

#### 4. Genome Diversity Bug ❌→✅
**Problem:** All agents identical → monoculture failure
```python
# BROKEN: Clone genome for all agents
for agent in agents:
    agent.genome = best_genome.copy()  # All identical!

# FIXED: Random diverse genomes
for agent in agents:
    agent.genome = Genome.random()  # Each unique
```

### Key Insights

1. **Reward Design Matters**
   - Large eating reward (+10.0) drives food-seeking
   - Failed EAT penalty (-0.5) prevents spam
   - Death penalty (-1.0) incentivizes survival

2. **Training Frequency**
   - Every 3 ticks = fast adaptation
   - Batch size 16 = stable gradients
   - Buffer capacity 1000 = diverse experiences

3. **Resource Abundance**
   - 400 berries needed for 10 agents
   - 150 plants for renewable food
   - Low metabolism (0.015) gives learning time

4. **Diversity is Critical**
   - Random initialization > pre-training
   - Each agent explores different strategies
   - Population diversity prevents collective failure

### Usage

**Enable Learning:**
```bash
# Command line
python main.py --gui --learning --config config/training_easy.yaml

# Or in code
agent.enable_learning(
    learning_rate=0.01,
    discount_factor=0.95,
    batch_size=16,
    buffer_capacity=1000
)
```

**Monitor Learning:**
```python
# Check if learning is working
if agent.learner:
    buffer_size = len(agent.learner.replay_buffer)
    print(f"Agent {agent.id}: {buffer_size} experiences")
    
# Training logs appear automatically
# "Agent 5 trained at age 300: buffer=299, loss=0.4264"
```

**Analyze Results:**
```bash
# After simulation completes
python scripts/analyze_logs.py  # Action distribution
python analyze_energy_economics.py  # Energy flow
```

### Future Enhancements

- [ ] Multi-agent communication rewards
- [ ] Curiosity-driven exploration bonus
- [ ] Meta-learning across generations
- [ ] Transfer learning to new environments
- [ ] Hierarchical policy learning

---

## World Upgrade (W0–W1) — June 2026

The world is being modernised in config-gated phases so the brain stack has
a richer, non-stationary environment to evolve against. Full plan:
[WORLD_UPGRADE_PROPOSAL.md](WORLD_UPGRADE_PROPOSAL.md). Phases W0 and W1 are
shipped; W2+ are in progress.

### W0 — Registry hardening & custom-object UX

The YAML object registry is the cheapest extension surface in the codebase,
but defining a custom object used to fail silently (a typo'd section name
registered a useless object; a wrong field crashed with a context-free
error; a definition took ~60 lines of copying).

- **Schema validation** (`world/object_validation.py`): unknown sections and
  fields are now errors with *did-you-mean* suggestions, all collected and
  reported together with the offending `type_id`.
- **Cross-reference checking** at load time: `grows_into`, `produces`,
  `decompose_into`, `spread_type_id` must name real types.
- **`extends:` inheritance**: a new object deep-merges a parent definition
  (builtin or earlier in the same file), so a new food is ~8 lines, not ~60.
  Spawn counts are never inherited implicitly.
- **`vision_encoding: auto`** allocates a free value inside per-category
  bands; collisions (two types closer than 0.02 in encoding space) warn,
  because agents literally cannot tell them apart.
- **Respawn spec**: `spawn.respawn_rate` / `spawn.max_count` let custom
  standalone foods replenish like builtin berries (they used to vanish
  forever once eaten).
- **Tooling**: `python scripts/objects.py {validate,list,preview} <file>`
  and `docs/OBJECTS_GUIDE.md`. `main.py` refuses an invalid object file at
  startup with the full report instead of running a broken world.

### W1 — Environment engine (day/night, seasons, weather)

One `EnvironmentSystem` (`world/environment.py`) runs first each tick and
publishes global multipliers; every existing system consumes them. It is
**off by default** — when disabled, every multiplier is exactly 1.0 and the
legacy arithmetic is bit-reproducible.

| Field | Drives | Formula (enabled) |
|-------|--------|-------------------|
| light | growth, food production | `min_light + (1−min_light)·½(1+sin(2π t/day_length))` |
| temperature | growth/germination window, decay, metabolism | `base + season·sin + day/night offset`, clamped [0,1] |
| temperature_response | the comfort band | 1.0 in [0.3,0.7], linear → 0 at 0.1/0.9 |
| rain | the only moisture recovery | `rain_recovery` while a rain event is active |
| drought | amplified evaporation | evaporation × `drought_evaporation_factor` |

Consumption points: plant growth × light × temp window; germination × temp
window; food production × light; freshness decay × (0.5 + temperature);
agent metabolism × (1 + coef·2·|temp − 0.5|) in both serial and parallel
paths; soil moisture via the weather-driven evaporation/recovery model.

### Verified dynamics-bug fixes

- **B1 — moisture only ever rose.** The legacy soil model added a constant
  +0.0008/tick recovery against −0.0002/tick evaporation, so every tile
  saturated at 1.0 and moisture constrained nothing. *Fixed (when the
  environment is enabled):* evaporation scales with temperature/light (×2 in
  drought) and recovery arrives only during rain or next to water. The old
  arithmetic is preserved verbatim when disabled.
- **B2 — germination on sand was impossible.** Sand clamped fertility/
  moisture to 0.05, strictly below the seed requirements (0.3/0.2), so its
  ×0.1 germination multiplier never even applied. *Fixed:* clamps now sit at
  the thresholds (0.30/0.20) so the multiplier is what makes sand harder.
- **B5 — runaway plant/food accumulation.** Each mature plant produced ~20
  offspring over its life and germination never checked crowding, so plants
  (and the berries they spawn) tiled the world (a small world saturated at
  ~65% plant coverage). *Fixed:* density-dependent germination — a seed will
  not establish where its neighbourhood already holds `max_neighbor_plants`
  plants. Defaults (cap 3 in a 5×5 window) plateau coverage flat at ~24%
  with food still abundant; `max_neighbor_plants: 0` restores legacy growth.

*(B3 — water is cosmetic — is deferred to W2's elevation/water rework.
B4 — "inventory is a stasis field" — was investigated and retracted: carried
food does spoil.)*

---

## Conclusion

This ecosystem simulation demonstrates a **fully self-sustaining, physics-based world** with:

✅ Complete nutrient cycling (fertility)  
✅ Dynamic water cycle (moisture)  
✅ Realistic population dynamics (birth/death)  
✅ Natural resource management (reproduction)  
✅ Emergent spatial patterns (clustering)  
✅ Long-term sustainability (infinite runtime)  
✅ Full configurability (YAML parameters)  
✅ Comprehensive testing (35/35 passing)  
✅ Excellent performance (1000+ entities)  
✅ Clean architecture (ECS pattern)

**The world is truly alive and self-sustaining!** 🌱💧⚡

---

---

## Recent Updates - November 17, 2025

### Phase 1.9: Reproduction Configuration Fix ✅

**Status:** Fully Fixed and Verified

Fixed reproduction configuration not being passed from `main.py` to `World`, causing hardcoded defaults to be used instead of YAML settings.

#### Problem
- Reproduction config defined in YAML but ignored
- World was using internal defaults
- Parameters like `energy_threshold`, `energy_split`, etc. had no effect

#### Solution
```python
# main.py - lines 362-372
if 'reproduction' in config:
    world.reproduction_config = config['reproduction']
    print(f"\nReproduction enabled:")
    print(f"  Energy threshold: {config['reproduction'].get('energy_threshold', 0.6)*100:.0f}%")
    print(f"  Energy split: {config['reproduction'].get('energy_split', 0.4)*100:.0f}%")
    # ... more verification output
```

#### Verified Parameters
All reproduction settings now work correctly:
- ✅ `energy_threshold: 0.60` - Agent needs 60% of max energy
- ✅ `energy_split: 0.20` - Parent loses 20% of energy (keeps 80%)
- ✅ `min_age: 100` - Minimum age before reproduction
- ✅ `mutation_std: 0.02` - Mutation standard deviation
- ✅ `cooldown_ticks: 50-70` - Cooldown between reproductions
- ✅ `max_population: 50-100` - Maximum population limit

#### Console Output
```
Reproduction enabled:
  Energy threshold: 60% of max energy
  Energy split: 20% to offspring, 80% retained by parent
  Minimum age: 100 ticks
  Mutation rate: 2% std deviation
  Cooldown: 50 ticks between reproductions
  Max population: 50
```

### Phase 1.10: Population Control & Calamity System ✅

**Status:** Fully Implemented and Tested

Added two major features for population dynamics and environmental pressure.

#### Max Population Limit

**Purpose:** Prevent unlimited exponential growth

**Implementation:**
```python
# world/world.py - lines 407-428
max_population = self.reproduction_config.get('max_population', None)
if max_population is not None:
    current_population = len(self.agents) + len(new_offspring)
    if current_population >= max_population:
        continue  # Skip reproduction
```

**Configuration:**
```yaml
reproduction:
  max_population: 50  # null = unlimited
```

**Console Output:**
```
[REPRODUCTION] Agent 12 (Gen 3, Age 150) reproduced! 
               pop: 48/50
```

**Benefits:**
- ✅ Prevents infinite population growth
- ✅ Creates competition for resources
- ✅ Reproduction resumes when population drops
- ✅ Natural selection for efficient agents

#### Calamity System

**Purpose:** Periodic environmental disasters that destroy resources

**Implementation:**
```python
# world/world.py
def _check_calamity(self):
    if self.tick - self.last_calamity_tick >= interval:
        self._trigger_calamity()
        
def _trigger_calamity(self):
    # Randomly destroy objects based on destruction_rate
    # Separate handling for plants, food, seeds
    # Return nutrients to soil from destroyed plants
    # Print comprehensive statistics
```

**Configuration:**
```yaml
calamity:
  enabled: true              # Toggle disasters
  interval: 500              # Ticks between disasters
  destruction_rate: 0.30     # 30% of resources destroyed
  affect_plants: true        # Destroy plants
  affect_food: true          # Destroy berries
  affect_seeds: false        # Preserve seeds for recovery
```

**Console Output:**
```
⚠️  [CALAMITY] Tick 500: Environmental disaster struck!
   Destroyed 12 objects (30.0% destruction rate)
   Plants destroyed: 7
   Food destroyed: 5
   Seeds destroyed: 0 (preserved)
   Remaining objects: 48
```

**Benefits:**
- ✅ Creates survival pressure
- ✅ Prevents overpopulation through scarcity
- ✅ Selects for adaptable agents
- ✅ Configurable frequency and severity
- ✅ Seeds can be preserved for recovery
- ✅ Destroyed plants return nutrients to soil

**Test Results:**
```
Test Setup: 40 objects (20 plants, 10 food, 10 seeds)
Config: 30% destruction rate, seeds preserved

✅ Plants destroyed: 7/20 (35%) - expected ~30%
✅ Food destroyed: 5/10 (25%) - expected ~30%
✅ Seeds preserved: 0/10 (0%) - expected 0%
✅ Nutrients returned: 7 × 0.15 = 1.05 total fertility
```

#### Files Modified
- `world/world.py` - Added calamity system and max population check
- `main.py` - Added config setup and console output
- `config/default.yaml` - Added reproduction and calamity sections
- `config/training_easy.yaml` - Updated with new features

#### Testing
All tests pass (100% success rate):
- ✅ `test_reproduction_config.py` - Config reading verification
- ✅ `test_max_population.py` - Population limit enforcement
- ✅ `test_calamity.py` - Disaster system verification

### Object Stacking Configuration System ✅

**Status:** Fully Implemented and Tested (Phase 1.8)

Added configurable `allow_stacking` option to control whether multiple objects can occupy the same tile.

#### Configuration
```yaml
world:
  allow_stacking: false  # Controls object stacking behavior
```

#### Modes

**Strict Mode (default: `false`):**
- ✅ Enforces one object per tile
- ✅ Automatic nearby placement when tiles occupied
- ✅ Realistic spatial constraints
- ✅ Better visualization (no overlapping objects)
- ✅ Strategic gameplay (agents manage space)

**Legacy Mode (`true`):**
- ✅ Allows multiple objects per tile
- ✅ Backward compatibility with old simulations
- ✅ Higher density in smaller worlds
- ✅ Simpler object placement logic

#### Implementation Details

**Files Modified:**
- `world/world.py` - Added `allow_stacking` parameter and logic
- `agents/agent.py` - Updated DROP, USE (planting), and DIE methods
- `config/default.yaml` - Added configuration setting
- `config/training_easy.yaml` - Added configuration setting
- `main.py` - Pass config to World constructor

**Key Behaviors:**
1. **World.add_object()** - Checks stacking config, tries 8 nearby tiles if occupied
2. **Agent._drop()** - Respects config, returns item to inventory if no space
3. **Agent._use()** (planting) - Checks config, tries nearby plantable tiles
4. **Agent.die()** - Checks config, scatters items or removes if no space

#### Testing
All tests pass (100% success rate):
- ✅ Strict Mode - Objects properly distributed to nearby tiles
- ✅ Legacy Mode - Multiple objects stack on same tile
- ✅ Agent Actions - Correctly respect configuration

**Test Suite:** `test_stacking_config.py`

#### Documentation
- `STACKING_CONFIG_FEATURE.md` - Complete feature documentation
- `STACKING_IMPLEMENTATION_SUMMARY.md` - Implementation details and test results

#### Benefits
- **Realism:** Physical space constraints enforced
- **Flexibility:** Easy switch between modes via config
- **Compatibility:** Preserves backward compatibility
- **Performance:** Negligible overhead (all O(1) operations)

---

**Version:** 1.2.0  
**Last Updated:** November 17, 2025  
**Author:** Karan Vasa  
**License:** MIT
