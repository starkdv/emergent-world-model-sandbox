"""
Headless brain-cohort competition run.

Builds the config-driven world with two competing brain cohorts (the
``competition`` block in the config: ~old_fraction on the old brain, the rest
on the new one), runs it for N ticks with per-action logging + a per-generation
metrics CSV, then prints the analyzer report (which includes the
⚔️ COHORT COMPARISON section).

Usage:
    python scripts/competition_run.py --ticks 4000 --out data/competition
    python scripts/competition_run.py --config config/default.yaml --ticks 4000

The action log carries a ``cohort`` column, so analysis can compare the two
populations head-to-head. Logs land in --out (default data/competition/).

Author: Karan Vasa
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Headless brain-cohort competition run")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--ticks", type=int, default=4000)
    p.add_argument("--out", default="data/competition")
    p.add_argument("--generation-length", type=int, default=1000)
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    from agents.agent import Agent
    from render.sim_session import session_from_config
    from utils.data.agent_logger import AgentLogger
    from utils.agents.metrics import MetricsWriter

    session = session_from_config(args.config)
    world = session.world

    logger = AgentLogger(output_dir=args.out)
    Agent.logger = logger  # agents auto-log each action (incl. their cohort)
    metrics_path = os.path.join(args.out, "metrics.csv")
    metrics = MetricsWriter(metrics_path)

    from collections import Counter

    init = Counter(getattr(a, "cohort", "default") for a in world.agents.values())
    print(f"Starting competition: {dict(init)} on a {world.width}x{world.height} world")

    try:
        for _ in range(args.ticks):
            session.step()
            if world.tick % args.generation_length == 0:
                metrics.record(world, generation=world.tick // args.generation_length)
            if not world.agents:
                print(f"Population extinct at tick {world.tick}")
                break
    finally:
        logger.close()
        metrics.close()

    final = Counter(getattr(a, "cohort", "default") for a in world.agents.values())
    print(f"Final population: {dict(final)} at tick {world.tick}")
    print(f"Logs: {logger.action_file}\nMetrics: {metrics_path}")

    # Run the analyzer on the action log and save the report next to the logs.
    import subprocess

    report = os.path.join(args.out, "analysis.txt")
    with open(report, "w", encoding="utf-8") as f:
        subprocess.run(
            [sys.executable, "scripts/analyze_logs.py", "--file", logger.action_file],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"Analysis: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
