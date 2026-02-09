# World Model Training Data Logging Format

## Overview

The `WorldModelLogger` class captures complete transition data for training neural network world models. It generates three CSV files optimized for different training needs.

## Files Generated

### 1. `transitions_{timestamp}.csv` - Main Training Data

Complete (state, action, reward, next_state, done) tuples for each agent action.

| Column | Type | Description |
|--------|------|-------------|
| **Identifiers** |||
| tick | int | Simulation tick |
| agent_id | int | Unique agent ID |
| episode_step | int | Step number within agent's episode |
| **Action Information** |||
| action | str | Action name (MOVE_FORWARD, TURN_LEFT, etc.) |
| action_value | int | Action enum value (0-7) |
| success | bool | Whether action succeeded |
| energy_cost | float | Energy cost of action |
| **Position & Direction** |||
| x, y | int | Position before action |
| direction_x, direction_y | int | Facing direction before |
| x_next, y_next | int | Position after action |
| direction_x_next, direction_y_next | int | Facing direction after |
| **Agent State** |||
| energy | float | Energy before action |
| energy_next | float | Energy after action |
| energy_pct | float | Energy % before (0-1) |
| energy_pct_next | float | Energy % after (0-1) |
| age | int | Agent age in ticks |
| inventory_count | int | Items in inventory before |
| inventory_count_next | int | Items in inventory after |
| fitness | float | Fitness before action |
| fitness_next | float | Fitness after action |
| **Learning Signals** |||
| reward | float | Reward signal |
| done | bool | Episode terminated (agent died) |
| death_reason | str | Cause of death if done |
| **Tile Information** |||
| tile_terrain | int | Terrain type (0=rock, 1=soil, 2=water) |
| tile_fertility | float | Soil fertility (0-1) |
| tile_moisture | float | Soil moisture (0-1) |
| tile_has_food | bool | Food present on tile |
| tile_has_plant | bool | Plant present on tile |
| tile_has_seed | bool | Seed present on tile |
| tile_food_calories | float | Calories of food on tile |
| **World Context** |||
| total_food_count | int | Total food in world |
| total_plant_count | int | Total plants in world |
| alive_agents | int | Number of alive agents |
| **Agent Traits** |||
| metabolism_rate | float | Agent's metabolism rate |
| vision_radius | float | Agent's vision radius |
| **Observation Vectors** |||
| obs_0 to obs_63 | float | 64-dim observation before action |
| obs_next_0 to obs_next_63 | float | 64-dim observation after action |

**Total columns: ~160**

### 2. `episodes_{timestamp}.csv` - Episode Summaries

One row per agent episode (from spawn to death).

| Column | Type | Description |
|--------|------|-------------|
| agent_id | int | Unique agent ID |
| generation | int | Evolutionary generation |
| lineage_id | int | Genetic lineage |
| start_tick | int | Tick when agent spawned |
| end_tick | int | Tick when agent died |
| duration | int | Lifespan in ticks |
| total_reward | float | Sum of all rewards |
| total_actions | int | Number of actions taken |
| successful_eats | int | Number of successful EAT actions |
| successful_pickups | int | Number of successful PICK_UP actions |
| tiles_explored | int | Number of unique tiles visited |
| final_fitness | float | Fitness at death |
| death_reason | str | Cause of death |
| final_energy | float | Energy at death |
| max_energy_reached | float | Maximum energy during episode |
| avg_energy | float | Average energy over episode |
| metabolism_rate | float | Agent's metabolism rate |
| vision_radius | float | Agent's vision radius |

### 3. `world_states_{timestamp}.csv` - Global State Snapshots

Periodic snapshots of global world state.

| Column | Type | Description |
|--------|------|-------------|
| tick | int | Simulation tick |
| alive_agents | int | Number of alive agents |
| total_agents | int | Total agents (alive + dead) |
| total_food | int | Food items in world |
| total_plants | int | Plants in world |
| total_seeds | int | Seeds in world |
| avg_agent_energy | float | Average energy of alive agents |
| min_agent_energy | float | Minimum energy |
| max_agent_energy | float | Maximum energy |
| avg_agent_age | float | Average age of alive agents |
| max_agent_age | int | Maximum age |
| avg_fertility | float | Average soil fertility |
| avg_moisture | float | Average soil moisture |
| total_fitness | float | Sum of all agent fitness |
| avg_fitness | float | Average fitness |

## Usage

### Enable Logging

```bash
# Enable world model logging
python main.py --gui --config config/training_easy.yaml --world-model-log

# With custom log directory
python main.py --gui --config config/training_easy.yaml --world-model-log --log-dir data/training_data

# With reduced logging frequency (every 10 ticks for world states)
python main.py --gui --config config/training_easy.yaml --world-model-log --log-frequency 10
```

### Load Training Data (Python)

```python
import pandas as pd
import numpy as np

# Load transitions
transitions = pd.read_csv('data/logs/transitions_20251129_123456.csv')

# Extract observation vectors
obs_cols = [f'obs_{i}' for i in range(64)]
obs_next_cols = [f'obs_next_{i}' for i in range(64)]

states = transitions[obs_cols].values
next_states = transitions[obs_next_cols].values
actions = transitions['action_value'].values
rewards = transitions['reward'].values
dones = transitions['done'].values

# Create training dataset
X = np.concatenate([states, actions.reshape(-1, 1)], axis=1)
y = next_states  # Predict next state from (state, action)

# For reward prediction
y_reward = rewards

# For done prediction
y_done = dones
```

### Observation Vector Format (64 dimensions)

The observation vector encodes what the agent perceives:

| Indices | Features | Description |
|---------|----------|-------------|
| 0-1 | Agent energy & age | Normalized to 0-1 |
| 2-5 | Direction one-hot | N, E, S, W |
| 6-7 | Inventory & metabolism | Has space, metabolism rate |
| 8-57 | 5x5 vision grid | 25 tiles × 2 features each |
| 58-63 | Inventory state | Fullness, has_food, has_seed, etc. |

## Training Tips

1. **Filter by success**: For learning dynamics, focus on `success=True` transitions
2. **Episode boundaries**: Use `done=True` rows to mark episode ends
3. **Normalize rewards**: Rewards range from ~-1 to ~25 (eating at critical energy)
4. **Balance dataset**: Death events are rare, may need oversampling
5. **Temporal features**: Add delta features (energy_change = energy_next - energy)

## Compatibility with World Models

This format supports training:
- **Transition models**: P(s' | s, a)
- **Reward models**: R(s, a, s')
- **Termination models**: P(done | s, a)
- **Dreamer-style models**: Latent dynamics + reward + value heads
- **MuZero-style models**: Dynamics + prediction + representation networks
