# Emergent World Model Sandbox - CLI Guide

This guide provides instructions on how to set up, run, and test the Emergent World Model Sandbox project using the command-line interface.

## 1. Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Running Simulations

The main entry point for the simulation is `main.py`.

### Basic Usage
Run with default configuration:
```bash
python main.py
```

### Visualization Modes
- **Headless Mode** (faster, no graphics):
  ```bash
  python main.py --no-viz
  ```
- **GUI Mode** (Pygame visualization):
  ```bash
  python main.py --gui
  ```
- **GPU Isometric Mode** (ModernGL 2.5D accelerated renderer):
  ```bash
  python main.py --gui --gpu
  ```
- **Demo Mode** (Console visualization):
  ```bash
  python main.py --demo
  ```

### Configuration
- **Custom Configuration File**:
  ```bash
  python main.py --config config/custom.yaml
  ```
- **Set Random Seed** (for reproducibility):
  ```bash
  python main.py --seed 42
  ```
- **Set Number of Generations**:
  ```bash
  python main.py --generations 100
  ```

### Advanced Features
- **Enable Logging**:
  ```bash
  python main.py --log --log-dir data/logs --log-frequency 10
  ```
- **Evolution Mode** (select RL or pure neuroevolution):
  ```bash
  python main.py --mode rl              # RL + Lamarckian inheritance (agents learn via gradients)
  python main.py --mode neuroevolution  # Pure evolution, no gradient learning
  ```
  Can also be set in `config/default.yaml` under `evolution.mode`. The `--mode` flag overrides config.
- **Enable Agent Learning** (legacy, equivalent to `--mode rl`):
  ```bash
  python main.py --learning --learning-rate 0.01
  ```
- **Save Best Weights**:
  ```bash
  python main.py --save-weights
  ```
- **Load Pre-trained Weights**:
  ```bash
  python main.py --load-weights data/weights/best_weights.npz
  ```
- **Enable World Model Logging**:
  ```bash
  python main.py --world-model-log
  ```

## 3. Running Tests

All tests are located in the `tests/` directory.

### Run All Tests
Using `pytest`:
```bash
python -m pytest tests/
```

### Run Specific Tests
Run a specific test file:
```bash
python -m pytest tests/test_agents.py
```

Run a standalone test script (e.g., learning visualization):
```bash
python tests/test_learning.py
```

## 4. Troubleshooting

- **Module Not Found Error**: Ensure you are running from the project root directory and your virtual environment is activated.
- **Visualization Issues**: If Pygame fails to initialize, try running in headless mode with `--no-viz`.

## 5. Directory Structure
- `agents/`: Agent logic, brains, and evolution
- `world/`: World simulation, tiles, and objects
- `config/`: YAML configuration files
- `data/`: Logs and saved weights
- `tests/`: Unit and integration tests
- `utils/`: Helper functions and visualization tools
