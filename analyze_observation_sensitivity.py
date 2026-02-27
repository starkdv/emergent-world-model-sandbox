"""Per-feature observation sensitivity analysis for agent policy decisions.

This script perturbs each observation feature and measures how much the
policy distribution/value estimate changes.

It reports two views:
1) Raw network sensitivity (no action mask, no contextual instinct boosts)
2) Runtime sensitivity (with action mask, including contextual instincts)
"""

from __future__ import annotations

import argparse
import numpy as np

from agents import Agent, Brain, Genome, create_default_trait_config
from agents.actions import Action
from utils.agents import build_observation
from world.object_registry import ObjectRegistry, register_builtin_objects
from world.world import World


def _feature_names() -> list[str]:
    names: list[str] = []

    names.extend([
        "state.energy_ratio",
        "state.age_ratio",
        "state.dir_N",
        "state.dir_E",
        "state.dir_S",
        "state.dir_W",
        "state.has_inventory_space",
        "state.metabolism_norm",
    ])

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            names.append(f"vision.t[{dx:+d},{dy:+d}].type")
            names.append(f"vision.t[{dx:+d},{dy:+d}].value")

    names.extend([
        "stim.food_on_tile",
        "stim.seed_on_tile",
        "stim.food_ahead",
        "stim.resource_ahead",
        "stim.nearest_food_prox",
        "stim.food_dir_match",
        "stim.energy_urgency",
        "stim.can_interact",
    ])

    names.extend([
        "inv.fullness",
        "inv.has_food",
        "inv.has_seed",
        "inv.has_fertilizer",
        "inv.total_calories_norm",
        "inv.count_norm",
    ])

    return names


def _create_probe_world(seed: int = 42) -> tuple[World, Agent]:
    register_builtin_objects()

    world = World(width=30, height=30, seed=seed)

    weight_count = Brain.calculate_weight_count()
    genome = Genome.random(weight_count=weight_count, trait_config=create_default_trait_config())

    agent = Agent(x=15, y=15, genome=genome)
    world.add_agent(agent)

    # Construct a mildly informative local scene around the agent
    # so stimulus/vision channels are non-trivial.
    berry_ahead = ObjectRegistry.create("berry", 15, 13)
    berry_right = ObjectRegistry.create("berry", 17, 15)
    seed_left = ObjectRegistry.create("berry_seed", 14, 15)
    plant_diag = ObjectRegistry.create("berry_plant", 16, 14)

    world.add_object(berry_ahead)
    world.add_object(berry_right)
    world.add_object(seed_left)
    world.add_object(plant_diag)

    return world, agent


def _measure(
    brain: Brain,
    obs: np.ndarray,
    h: np.ndarray,
    action_mask: np.ndarray | None,
    epsilon: float,
) -> list[tuple[int, float, float, float, float]]:
    """Return tuples of (idx, p_l1_delta, value_delta, move_delta, wait_delta)."""
    base_probs, base_value, _ = brain.forward(obs, h, action_mask=action_mask, temperature=1.0)

    rows: list[tuple[int, float, float, float, float]] = []
    for idx in range(obs.shape[0]):
        perturbed = obs.copy()
        perturbed[idx] = float(np.clip(perturbed[idx] + epsilon, 0.0, 1.0))

        probs, value, _ = brain.forward(perturbed, h, action_mask=action_mask, temperature=1.0)

        p_l1_delta = float(np.abs(probs - base_probs).sum())
        value_delta = float(abs(value - base_value))
        move_delta = float(probs[Action.MOVE_FORWARD.value] - base_probs[Action.MOVE_FORWARD.value])
        wait_delta = float(probs[Action.WAIT.value] - base_probs[Action.WAIT.value])

        rows.append((idx, p_l1_delta, value_delta, move_delta, wait_delta))

    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def _print_top(title: str, rows: list[tuple[int, float, float, float, float]], names: list[str], top_n: int) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print("rank  idx  feature                          p_l1Δ     valueΔ    ΔP(move)  ΔP(wait)")
    for rank, (idx, p_l1, v_d, m_d, w_d) in enumerate(rows[:top_n], start=1):
        print(f"{rank:>4d}  {idx:>3d}  {names[idx]:<32s}  {p_l1:>8.5f}  {v_d:>8.5f}  {m_d:>8.5f}  {w_d:>8.5f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Observation feature sensitivity analysis")
    parser.add_argument("--seed", type=int, default=42, help="Probe world random seed")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Per-feature perturbation magnitude")
    parser.add_argument("--top", type=int, default=20, help="How many top features to print")
    args = parser.parse_args()

    names = _feature_names()
    world, agent = _create_probe_world(seed=args.seed)

    obs = build_observation(agent, world)
    h = agent.brain.initial_state()
    mask = agent.get_action_mask(world)

    if len(obs) != len(names):
        print(f"Observation/name length mismatch: obs={len(obs)} names={len(names)}")
        return 1

    print("Observation sensitivity report")
    print(f"obs_size={len(obs)}, epsilon={args.epsilon}")
    print(f"agent_pos=({agent.x},{agent.y}) dir={agent.direction}")

    raw_rows = _measure(agent.brain, obs, h, action_mask=None, epsilon=args.epsilon)
    runtime_rows = _measure(agent.brain, obs, h, action_mask=mask, epsilon=args.epsilon)

    _print_top("Top features (RAW network: no mask/instinct)", raw_rows, names, args.top)
    _print_top("Top features (RUNTIME: action mask + instincts)", runtime_rows, names, args.top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
