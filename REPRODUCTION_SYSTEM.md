# In-Simulation Reproduction System

## Overview

Successfully implemented **in-simulation reproduction** via fission-based cloning with Lamarckian inheritance. Agents can now reproduce during their lifetime when they meet specific conditions, creating offspring that inherit their trained neural network weights.

## Implementation Date
November 16, 2025

## Key Features

### 1. Reproduction Conditions
Agents can reproduce when they meet ALL of the following criteria:
- **Alive**: Must be a living agent
- **Energy Threshold**: ≥ 60% of max_energy (600 out of 1000 in easy config)
- **Age Threshold**: ≥ 100 ticks (maturity requirement)

### 2. Reproduction Mechanism (`Agent.reproduce()`)
```python
def reproduce(self, world: 'World') -> Optional['Agent']:
    """
    Reproduce via fission, creating an offspring.
    
    Process:
    1. Check if agent can_reproduce()
    2. Use clone_agent() from evolution.py
    3. Split energy: parent keeps 40%, offspring gets 60%
    4. Apply small Gaussian mutation (std=0.02) to offspring
    5. Find empty adjacent position for offspring
    6. Transfer learning capability to offspring
    7. Add offspring to world
    """
```

### 3. Lamarckian Inheritance
Offspring inherit the **trained weights** from their parent's brain at the time of reproduction:
- Parent's neural network weights → Offspring's initial weights
- Offspring inherits learned behaviors from parent
- Small mutations provide variation
- Learning continues in offspring's lifetime

### 4. Energy Economics
- **Parent Cost**: Loses 60% of current energy
- **Offspring Benefit**: Starts with 60% of parent's pre-split energy
- **Total Energy**: Conserved (100% → 40% parent + 60% offspring)
- **Strategic Trade-off**: Parent must balance survival vs reproduction

### 5. Spatial Placement
- Offspring placed in adjacent cell (8 possible positions)
- Random order search for empty position
- Fails if no empty adjacent cells available
- Energy refunded if reproduction fails

## Integration Points

### World Update Loop (`world/world.py`)
```python
def _update_agents(self) -> None:
    """Update all agents in the world."""
    new_offspring = []
    
    for agent in list(self.agents.values()):
        if agent.alive:
            agent.update(self)
            
            # Check for reproduction after update
            if agent.can_reproduce():
                offspring = agent.reproduce(self)
                if offspring is not None:
                    new_offspring.append(offspring)
                    print(f"🐣 Agent {agent.id} reproduced!")
    
    # Add all offspring to world
    for offspring in new_offspring:
        self.add_agent(offspring)
```

### Evolution Module (`agents/evolution.py`)
- `clone_agent(parent, mutate=True, mutation_std=0.02)` - Creates offspring
- Copies parent's brain weights (Lamarckian inheritance)
- Applies Gaussian mutation to weights and biases
- Resets age, energy, and fitness for new life

## Test Results

### Test: `test_reproduction.py`
**Configuration:**
- Initial population: 5 agents
- Max energy: 1000
- Starting energy: 700
- Reproduction threshold: 600 (60%)
- Min age: 100 ticks
- Environment: 800 berries

**Results:**
```
Tick 99: All 5 original agents reproduced!
  🐣 Agent 0 → Agent 5 (energy: 259.8)
  🐣 Agent 1 → Agent 6 (energy: 258.5)
  🐣 Agent 2 → Agent 7 (energy: 258.7)
  🐣 Agent 3 → Agent 8 (energy: 245.7)
  🐣 Agent 4 → Agent 9 (energy: 249.6)

Population: 5 → 10 agents (100% growth)
```

**Energy Dynamics:**
- Tick 0: Avg 699.4, Max 699.5
- Tick 99: Avg 318.1, Max 389.7 (after reproduction)
- Tick 200: Avg 245.9, Max 337.3
- Tick 300: Avg 175.0, Max 285.1

**Observations:**
- ✅ Reproduction working correctly
- ✅ Offspring inherit learning capability
- ✅ Population successfully doubled
- ⚠️ Energy declining over time (need better foraging)
- ⚠️ No second-generation reproduction (energy too low)

## Usage

### Enable Reproduction in Simulations

**Option 1: Using main.py**
```bash
python main.py --learning --config config/training_easy.yaml --gui
```

**Option 2: Using test scripts**
```bash
python test_reproduction.py
python test_v2_with_easy_config.py
```

**Option 3: Custom Code**
```python
from world.world import World
from agents import Agent, Genome, create_default_trait_config

# Create world
world = World(width=100, height=100)

# Create agents with learning
for i in range(5):
    genome = Genome.random(weight_count=2744, trait_config=create_default_trait_config())
    agent = Agent(x=50, y=50, genome=genome, max_energy=1000)
    agent.enable_learning()
    world.add_agent(agent)

# Run simulation
for tick in range(1000):
    world.update()  # Reproduction happens automatically
```

## Configuration Parameters

### Adjustable in Code (`agents/agent.py`)
```python
# Energy threshold (currently 60%)
energy_threshold = self.max_energy * 0.6

# Minimum age (currently 100 ticks)
min_age = 100

# Energy split (currently 60% to offspring)
energy_cost = self.energy * 0.6

# Mutation rate (currently std=0.02)
mutation_std = 0.02
```

### Recommended Adjustments
- **Lower energy threshold (50%)**: More reproductions, faster population growth
- **Higher energy threshold (70-80%)**: Fewer reproductions, select for better foragers
- **Lower min age (50)**: Earlier reproduction, faster generations
- **Higher mutation (0.03-0.05)**: More variation, faster evolution
- **Lower mutation (0.01)**: Preserve learned behaviors better

## Advantages Over Multi-Generation System

| Feature | Multi-Generation (`test_evolution.py`) | In-Simulation Reproduction |
|---------|----------------------------------------|----------------------------|
| **When** | Between separate simulation runs | During continuous simulation |
| **Selection** | Manual fitness-based | Natural (survival to reproduce) |
| **Population** | Fixed size per generation | Dynamic, grows/shrinks naturally |
| **Time Scale** | Discrete generations | Continuous overlapping generations |
| **Realism** | Artificial | More realistic/emergent |
| **Use Case** | Training, benchmarking | Ecosystem dynamics, long runs |

## Integration with Multi-Generation Evolution

Both systems can be used together:
1. **In-simulation reproduction**: For continuous ecosystem dynamics
2. **Multi-generation evolution**: For benchmarking and long-term evolution studies

```python
# Hybrid approach
population = create_initial_population(10)

for generation in range(50):
    # Run one generation with in-simulation reproduction
    world = World(...)
    for agent in population:
        world.add_agent(agent)
    
    for tick in range(5000):
        world.update()  # Reproduction happens naturally
    
    # Select survivors for next generation
    survivors = [a for a in world.agents.values() if a.alive]
    population = next_generation(survivors, config)
```

## Future Enhancements

### Planned
- [ ] **Mating**: Combine weights from two parents (crossover)
- [ ] **Variable mutation**: Adapt mutation rate based on population diversity
- [ ] **Reproduction cooldown**: Prevent immediate re-reproduction
- [ ] **Age-based fertility**: Reproduction window (e.g., age 100-500 only)
- [ ] **Resource-based**: Require food items in inventory to reproduce

### Advanced
- [ ] **Sexual selection**: Parents choose mates based on fitness
- [ ] **Parental investment**: Parents can transfer resources to offspring
- [ ] **K-selection vs r-selection**: Different strategies (few strong offspring vs many weak)
- [ ] **Speciation**: Different agent types that can't interbreed

## Performance Considerations

- **Memory**: Each offspring creates new Agent, Brain, Genome objects
- **CPU**: clone_agent() copies weight matrices (can be expensive)
- **Population growth**: Exponential growth possible if energy plentiful
- **Recommended limits**: Consider max population cap for long simulations

## Troubleshooting

### No Reproduction Occurring
1. **Check energy levels**: Are agents reaching 60% threshold?
2. **Check age**: Are agents surviving to 100 ticks?
3. **Check resources**: Enough food in environment?
4. **Check metabolism**: Rate too high for energy accumulation?

### Population Explosion
1. **Increase metabolism rate**: More energy drain
2. **Decrease food availability**: Limit berry count
3. **Increase energy threshold**: Require 70-80% for reproduction
4. **Add max population cap**: Stop reproduction at limit

### No Offspring Survival
1. **Check spawn positions**: Are adjacent cells blocked?
2. **Check starting energy**: Is 60% enough to survive?
3. **Check environment**: Enough resources near parents?

## Related Files

- `agents/agent.py` - Agent.can_reproduce(), Agent.reproduce()
- `agents/evolution.py` - clone_agent(), mutate_weights()
- `world/world.py` - World._update_agents() integration
- `test_reproduction.py` - Test script for reproduction system
- `EVOLUTION_ENHANCEMENTS.md` - Diversity tracking and adaptive mutation

## Conclusion

The in-simulation reproduction system successfully enables:
✅ **Dynamic population growth** based on survival success  
✅ **Lamarckian evolution** passing learned behaviors to offspring  
✅ **Natural selection** via reproduction requirements  
✅ **Emergent dynamics** of population ecology  

This system, combined with reinforcement learning and multi-generation evolution, creates a complete evolutionary learning framework for the emergent world simulation.

---
**Status**: ✅ **FULLY OPERATIONAL**  
**Test Coverage**: ✅ Verified with test_reproduction.py  
**Integration**: ✅ Integrated into main simulation loop  
**Documentation**: ✅ Complete
