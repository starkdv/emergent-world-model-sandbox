# Emergent World-Model Sandbox

**Author:** Karan Vasa  
**License:** MIT  

A simulation sandbox with evolving agents that learn survival strategies and exhibit emergent behaviors through evolution in a physically consistent, resource-based 2D world.

## Overview

This project implements a **2D grid world** where:

- **Agents** move around, consume resources, and manipulate objects using only primitive actions
- **Evolution** drives the emergence of complex behaviors like farming, cooperation, and communication
- **World physics** governs resource transformation, plant growth, and environmental dynamics
- **No hardcoded behaviors** - all complex strategies emerge through natural selection

## Key Features

- 🌱 **Emergent Agriculture**: Agents discover seed planting and farming through evolution
- 🤝 **Cooperation**: Group behaviors emerge without explicit programming
- 🧠 **Neural Evolution**: Agents use evolved neural networks to make decisions
- 🔬 **Scientific Analysis**: Comprehensive data collection and visualization tools
- 🎮 **Real-time Visualization**: Interactive GUI for monitoring simulations

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/emergent-world-model-sandbox.git
cd emergent-world-model-sandbox

# Install dependencies
pip install -r requirements.txt

# Run a basic simulation
python -m simulation.runner --config config/default.yaml
```

## Architecture

```
emergent_world_model/
├── world/          # World physics and environment
├── agents/         # Agent behavior and evolution
├── simulation/     # Simulation management
├── utils/          # Utilities and visualization
├── config/         # Configuration files
└── data/           # Exported data and logs
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
- World size and terrain generation
- Population size and genetic parameters
- Resource availability and growth rates
- Visualization and data collection settings

See `config/default.yaml` for full configuration options.

## Data Analysis

The system automatically collects:
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
