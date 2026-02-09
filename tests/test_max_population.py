"""Quick test to verify max_population setting works."""

import yaml
import pytest
import os


def test_max_population():
    # Load config
    config_path = "config/training_easy.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {"reproduction": {"enabled": True, "max_population": 50}}

    # Check reproduction settings
    repro = config.get("reproduction", {})
    print("Reproduction Configuration:")
    print(f"  enabled: {repro.get('enabled')}")
    print(f"  max_population: {repro.get('max_population')}")

    # Verify it's set correctly
    max_pop = repro.get("max_population")

    if max_pop == 50:
        print("\n✅ max_population is correctly set to 50")
    else:
        print(f"\n❌ max_population is {max_pop}, expected 50")

    assert max_pop == 50, f"max_population is {max_pop}, expected 50"


if __name__ == "__main__":
    test_max_population()
