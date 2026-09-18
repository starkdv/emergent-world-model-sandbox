"""
Per-action shaped-reward probe.

Wraps ``Agent.compute_reward`` for the duration of a headless run and reports,
per action, how often it was taken, the shaped reward the learner assigned to
it, and what it cost in energy. This is the measurement behind §2.6 of
docs/BRAIN_V4_PROPOSAL.md: the reward landscape the policy is actually
climbing, as opposed to the one the config file describes.

    python docs/sample_v35_emergence_baseline/reward_probe.py \
        docs/sample_v35_emergence_baseline/configs/B_social_v35.yaml 1 2

Arguments are (config path, seed, generations). Output is JSON on stdout;
the run's own console output goes to stdout too, so the JSON object is the
final block.

Author: Karan Vasa
Date: September 2026
"""

import collections
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np  # noqa: E402
from agents.agent import Agent  # noqa: E402

RECORDS = collections.defaultdict(list)
_original_compute_reward = Agent.compute_reward


def _probed_compute_reward(self, action, result, energy_before, obs_after, world):
    """Record (reward, energy cost, success) per action, then defer."""
    reward = _original_compute_reward(
        self, action, result, energy_before, obs_after, world
    )
    RECORDS[action.name].append(
        (float(reward), float(result.energy_cost), bool(result.success))
    )
    return reward


def main(argv) -> int:
    """Run the simulation with the probe attached and print the summary."""
    if len(argv) < 4:
        print("usage: reward_probe.py CONFIG SEED GENERATIONS")
        return 1
    config_path, seed, generations = argv[1], argv[2], argv[3]

    Agent.compute_reward = _probed_compute_reward
    import main as sim  # noqa: E402 — imported after the patch is installed

    sys.argv = [
        "main.py",
        "--no-viz",
        "--config",
        config_path,
        "--mode",
        "rl",
        "--seed",
        seed,
        "--generations",
        generations,
    ]
    try:
        sim.main()
    except SystemExit:
        pass

    total = sum(len(v) for v in RECORDS.values()) or 1
    summary = {}
    for action, samples in RECORDS.items():
        rewards = np.array([s[0] for s in samples])
        costs = np.array([s[1] for s in samples])
        summary[action] = {
            "n": len(samples),
            "share_pct": round(100.0 * len(samples) / total, 2),
            "reward_mean": round(float(rewards.mean()), 4),
            "reward_median": round(float(np.median(rewards)), 4),
            "reward_p90": round(float(np.percentile(rewards, 90)), 4),
            "energy_cost_mean": round(float(costs.mean()), 4),
        }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
