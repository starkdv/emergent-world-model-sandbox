# Emergent World-Model Sandbox — CLI Guide

How to set up, run, analyze, and test the project from the command line.
For *what each mode does and its prerequisites*, see the
[Modes & Feature Toggles reference](../README.md#modes--feature-toggles--complete-reference)
in the README — this guide covers the mechanics of invoking things.

## 1. Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package installer)
- PyTorch is required only for the PPO learner, the world-model training,
  and dream evolution (everything else runs on NumPy)

### Setup
1. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Running Simulations — `main.py`

### Basic usage
```bash
python main.py              # default config (config/default.yaml), RL mode
python main.py --no-viz     # headless (fastest)
python main.py --gui        # Pygame 2D visualization
python main.py --gui --gpu  # ModernGL isometric 2.5D renderer
python main.py --demo       # one-shot console render of the world
```

### Full flag reference

| Flag | Effect |
|---|---|
| `--config PATH` | Configuration file (default: `config/default.yaml`) |
| `--mode {rl,neuroevolution}` | Evolution mode; overrides config. `rl` = lifetime learning + Lamarckian inheritance, `neuroevolution` = pure evolution, no gradients |
| `--learning` | Legacy alias for `--mode rl` |
| `--learning-rate F` | Learning rate for the A2C learner (PPO uses `learning.ppo.learning_rate`) |
| `--seed N` | World-generation seed. Note: with `simulation.parallel: true` (default) runs are not bit-reproducible; set `parallel: false` for determinism |
| `--generations N` | Number of generations (× `evolution.generation_length` ticks each) |
| `--no-viz` / `--gui` / `--gpu` | Headless / Pygame / GPU isometric rendering |
| `--log` | Write per-action + per-tick agent-state CSVs (`agent_actions_*.csv`, `agent_states_*.csv`) |
| `--log-dir PATH` | Directory for CSV logs (default: `data/logs`) |
| `--log-frequency N` | Log agent states every N ticks |
| `--world-model-log` | Write transition/episode/world-state CSVs — the training data for dream evolution |
| `--load-weights F.npz` | Seed agents from saved weights. **Length must match the configured brain** (version + world-model setting) |
| `--save-weights` | Save the best agents' weights at the end of the run |
| `--objects F.yaml` | Load custom object definitions |
| `--verbose` | Debug logging |

Architecture and learning behaviour (brain version, v3 sizes, instincts,
world model, curiosity, planner, PPO) are configured in the YAML — every
key is commented in `config/default.yaml`.

### Common recipes
```bash
# Controlled comparison: same seed, both evolution modes
python main.py --no-viz --seed 42 --generations 1 --mode rl
python main.py --no-viz --seed 42 --generations 1 --mode neuroevolution

# Collect world-model training data headlessly
python main.py --no-viz --world-model-log --seed 42

# Resume from saved champions
python main.py --gui --load-weights data/weights/best_weights.npz
```

## 3. Analysis Scripts — `scripts/`

### Log analysis — `scripts/analyze_logs.py`
```bash
python scripts/analyze_logs.py                     # latest log in data/logs
python scripts/analyze_logs.py --file path/to.csv  # a specific log
python scripts/analyze_logs.py --log-dir other/dir
python scripts/analyze_logs.py --fade-age 150      # match brain.instincts.fade_age
```
Works on both log schemas (`agent_actions_*.csv` and `transitions_*.csv`).
Reports action distribution and success rates, energy economy, population
dynamics, lifespans, spatial coverage, farming pipeline, behavioural
diversity, action n-grams, temporal phases, **instinct-fade phases**
(juvenile vs adult behaviour) and **death analysis** (reasons,
age-at-death).

### Observation sensitivity — `scripts/analyze_observation_sensitivity.py`
```bash
python scripts/analyze_observation_sensitivity.py                   # v2 brain
python scripts/analyze_observation_sensitivity.py --brain 3         # attention brain
python scripts/analyze_observation_sensitivity.py --brain 3 --world-model
python scripts/analyze_observation_sensitivity.py --epsilon 0.1 --top 30
```
Perturbs each observation feature and ranks policy/value impact in three
views: raw network, runtime juvenile (mask + instincts), runtime adult
(instincts faded).

### Dream-based evolution — `scripts/dream_evolve.py`
```bash
# 1. Collect real experience
python main.py --no-viz --world-model-log --seed 42

# 2. Train a population world model and evolve genomes inside it
python scripts/dream_evolve.py --transitions "data/logs/transitions_*.csv" \
    --generations 20 --population 32

# 3. GROUND the champions in the real environment (mandatory)
python main.py --load-weights data/weights/dream_best.npz --no-viz
```
Key flags: `--config` (must match the brain that will load the result),
`--epochs`, `--generations`, `--population`, `--episodes`, `--steps`,
`--mutation-std`, `--seed-weights` (warm-start from real champions),
`--out`, `--save-model`. Requires PyTorch.

## 4. Running Tests

```bash
python -m pytest tests/            # the whole suite (~2 min, all green)
python -m pytest tests/test_brain_spec.py -v   # one file
python -m pytest tests/ -k "instinct"          # by keyword
```

## 5. Troubleshooting

- **Module Not Found**: run from the project root with the virtual
  environment activated (the `scripts/` entries bootstrap the path
  themselves, so they also work from anywhere).
- **Visualization issues**: if Pygame fails to initialize, use `--no-viz`.
- **`--load-weights` fails with a shape/length error**: the saved weights
  were produced under a different brain configuration (version, v3 sizes,
  or world-model setting) — genome lengths must match.
- **PPO/world-model features silently fall back**: PyTorch isn't
  installed; the console prints a fallback notice.

## 6. Directory Structure
- `agents/` — agent logic, the brain package (v2/v3, spec, instincts),
  learners (A2C/PPO), curiosity, planner, dream evolution
- `world/` — world simulation, tiles, objects, systems
- `config/` — YAML configuration files (heavily commented)
- `scripts/` — analysis tools and the dream-evolution CLI
- `docs/` — all project documentation (proposals, comparisons, guides)
- `data/` — logs and saved weights (gitignored)
- `tests/` — unit, integration, and scenario tests
- `utils/` — perception, parallel engine, loggers, renderers
