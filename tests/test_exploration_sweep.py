"""
Exploration Bonus Sweep Test

Tests multiple exploration bonus values to find optimal balance.
Tests: 0.0 (V2 baseline), 0.05, 0.08, 0.10, 0.15 (current)

This will help us find the sweet spot quickly.
"""

import random
import pandas as pd
from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent, PlantComponent
from agents import Agent, Brain, Genome, create_default_trait_config
from utils.data.agent_logger import AgentLogger
import os
from datetime import datetime


def populate_world_with_resources(world, berry_count=30, plant_count=20, seed_count=15):
    """Add initial resources to the world."""
    # Add berries
    for _ in range(berry_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        if world.is_valid_position(x, y):
            berry = WorldObject(x, y)
            berry.add_component(EdibleComponent(calories=25.0))
            world.add_object(berry)

    # Add plants
    for _ in range(plant_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        tile = world.get_tile(x, y)
        if tile and tile.is_plantable():
            plant = WorldObject(x, y)
            plant.add_component(
                PlantComponent(mature_age=30, max_age=200, spawn_rate=0.1)
            )
            world.add_object(plant)

    # Add seeds
    for _ in range(seed_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        if world.is_valid_position(x, y):
            seed = WorldObject(x, y)
            seed.add_component(SeedComponent(plant_type="berry_plant", grow_time=30))
            world.add_object(seed)


def set_exploration_bonus(bonus_value):
    """Temporarily modify the exploration bonus in learning.py"""
    file_path = "agents/learning.py"

    # Read current file
    with open(file_path, "r") as f:
        content = f.read()

    # Find and replace the bonus line
    import re

    pattern = r"reward \+= ([\d.]+)  # .*exploration.*"

    # Search for the line
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        old_line = match.group(0)
        new_line = f"reward += {bonus_value}  # Exploration bonus (test sweep)"
        content = content.replace(old_line, new_line)

        # Write back
        with open(file_path, "w") as f:
            f.write(content)
        return True
    return False


def run_test(bonus_value, max_ticks=1000):
    """Run a single test with given exploration bonus."""
    print(f"\n{'='*70}")
    print(f"TESTING: Exploration Bonus = +{bonus_value}")
    print(f"{'='*70}")

    # Set the bonus value
    if not set_exploration_bonus(bonus_value):
        print("❌ Failed to set exploration bonus!")
        return None

    # Create world
    random.seed(42)  # Fixed seed for reproducibility
    world = World(width=30, height=30)
    populate_world_with_resources(world)

    # Create agents
    agents = []
    trait_config = create_default_trait_config()
    weight_count = Brain.calculate_weight_count()

    for i in range(10):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        genome = Genome.random(weight_count=weight_count, trait_config=trait_config)
        agent = Agent(x, y, genome=genome)
        agent.learning_enabled = True
        agents.append(agent)
        world.add_agent(agent)

    # Run simulation (no logging to save time)
    tick = 0
    action_log = []

    while tick < max_ticks:
        world.update()

        alive_agents = [a for a in agents if a.alive]
        if len(alive_agents) == 0:
            break

        # Collect action stats
        for agent in alive_agents:
            if hasattr(agent, "last_action_result"):
                action_log.append(
                    {
                        "tick": tick,
                        "agent_id": agent.id,
                        "action": (
                            str(agent.last_action_result.action)
                            if hasattr(agent.last_action_result, "action")
                            else "UNKNOWN"
                        ),
                        "success": agent.last_action_result.success,
                    }
                )

        tick += 1

    # Analyze results
    if not action_log:
        return None

    df = pd.DataFrame(action_log)
    total_actions = len(df)

    # Calculate metrics
    wait_pct = (
        (len(df[df["action"].str.contains("WAIT", na=False)]) / total_actions * 100)
        if total_actions > 0
        else 0
    )
    eat_actions = df[df["action"].str.contains("EAT", na=False)]
    eat_success_rate = (
        (eat_actions["success"].sum() / len(eat_actions) * 100)
        if len(eat_actions) > 0
        else 0
    )
    survival_time = tick

    results = {
        "bonus": bonus_value,
        "wait_pct": wait_pct,
        "eat_success_rate": eat_success_rate,
        "survival_time": survival_time,
        "total_actions": total_actions,
    }

    print(f"  Survival time: {survival_time} ticks")
    print(f"  WAIT actions: {wait_pct:.1f}%")
    print(f"  EAT success: {eat_success_rate:.1f}%")

    return results


def main():
    print("=" * 70)
    print("EXPLORATION BONUS SWEEP TEST")
    print("=" * 70)
    print("\nTesting multiple bonus values to find optimal balance...")
    print("This may take several minutes...\n")

    # Test values
    test_values = [0.0, 0.05, 0.08, 0.10]

    results = []
    for bonus in test_values:
        result = run_test(bonus, max_ticks=1000)
        if result:
            results.append(result)

    # Display comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Bonus':>8} | {'Survival':>9} | {'WAIT %':>7} | {'EAT %':>7} | Score")
    print("-" * 70)

    for r in results:
        # Calculate composite score
        # Ideal: survival=1500+, wait=32-35%, eat=4-5%
        survival_score = min(r["survival_time"] / 1500, 1.0) * 40
        wait_score = (
            30
            if 32 <= r["wait_pct"] <= 35
            else max(0, 30 - abs(r["wait_pct"] - 33.5) * 2)
        )
        eat_score = (
            30
            if 4 <= r["eat_success_rate"] <= 6
            else max(0, 30 - abs(r["eat_success_rate"] - 5) * 3)
        )
        total_score = survival_score + wait_score + eat_score

        print(
            f"{r['bonus']:>8.2f} | {r['survival_time']:>9d} | {r['wait_pct']:>6.1f}% | {r['eat_success_rate']:>6.1f}% | {total_score:>5.1f}"
        )

    # Find best
    best = max(
        results,
        key=lambda r: min(r["survival_time"] / 1500, 1.0) * 40
        + (
            30
            if 32 <= r["wait_pct"] <= 35
            else max(0, 30 - abs(r["wait_pct"] - 33.5) * 2)
        )
        + (
            30
            if 4 <= r["eat_success_rate"] <= 6
            else max(0, 30 - abs(r["eat_success_rate"] - 5) * 3)
        ),
    )

    print("\n" + "=" * 70)
    print(f"🏆 BEST BONUS: +{best['bonus']:.2f}")
    print("=" * 70)
    print(f"  Survival: {best['survival_time']} ticks")
    print(f"  WAIT: {best['wait_pct']:.1f}%")
    print(f"  EAT success: {best['eat_success_rate']:.1f}%")

    print("\n💡 RECOMMENDATION:")
    if best["bonus"] == 0.0:
        print("  No exploration bonus needed - V2 baseline is optimal")
        print("  The 10x reward improvements alone are sufficient")
    else:
        print(f"  Set exploration bonus to +{best['bonus']:.2f}")
        print(f"  Update agents/learning.py line ~141")

    print("\n" + "=" * 70)

    # Restore original value (0.15)
    set_exploration_bonus(0.15)


if __name__ == "__main__":
    main()
