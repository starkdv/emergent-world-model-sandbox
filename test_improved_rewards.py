"""
Quick test of improved learning rewards.
Run a short simulation and check if agents are more active.
"""

import subprocess
import sys

print("=" * 60)
print("TESTING IMPROVED REWARD SHAPING")
print("=" * 60)
print("\nRunning simulation with improved rewards...")
print("Expected improvements:")
print("  - More MOVE_FORWARD actions")
print("  - Higher EAT success rate")
print("  - Agents seeking food actively")
print("\nStarting simulation...\n")

# Run the simulation with learning enabled
result = subprocess.run([
    sys.executable, "main.py",
    "--config", "config/training_easy.yaml",
    "--learning",
    "--learning-rate", "0.01",
    "--log"
], capture_output=False)

print("\n" + "=" * 60)
print("Simulation complete! Check logs in data/logs/")
print("=" * 60)
print("\nTo analyze results, run:")
print("  python analyze_food_from_logs.py")
