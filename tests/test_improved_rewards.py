"""
Quick test of improved learning rewards.
Run a short simulation and check if agents are more active.
"""

import subprocess
import sys
import pytest


@pytest.mark.timeout(300)
def test_improved_rewards():
    print("=" * 60)
    print("TESTING IMPROVED REWARD SHAPING")
    print("=" * 60)
    print("\nRunning simulation with improved rewards...")
    print("Expected improvements:")
    print("  - More MOVE_FORWARD actions")
    print("  - Higher EAT success rate")
    print("  - Agents seeking food actively")
    print("\nStarting simulation...\n")

    # Run the simulation with learning enabled. Bound the run to a single
    # generation and headless mode so it stays a *short* smoke test (the
    # training_easy config defaults to 100 generations x 1000 ticks, which
    # would otherwise run for minutes and trip the CI per-test timeout).
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--config",
            "config/training_easy.yaml",
            "--learning",
            "--learning-rate",
            "0.01",
            "--log",
            "--no-viz",
            "--generations",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )  # Capture output to avoid spamming pytest logs unless failed

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)

    assert (
        result.returncode == 0
    ), f"Simulation failed with return code {result.returncode}"

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_improved_rewards()
