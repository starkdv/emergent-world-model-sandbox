"""
CSV loggers for agent actions, states, and world-model training data.

This module was referenced throughout the codebase (``--log`` flag,
``Agent.logger``, six test modules) but was never committed to the
repository. It is reconstructed here from its complete usage surface:

- ``Agent.execute_action`` calls
  ``log_action(tick, agent, action, result, x_before, y_before, energy_before)``
- ``World.update`` calls ``log_all_states(tick, agents)`` where ``agents``
  is either the live ``{id: agent}`` dict (serial path) or a snapshot
  *list* of agents (parallel path, via the I/O pool — so writes must be
  lock-guarded)
- ``main.py --log`` constructs ``AgentLogger(log_dir, log_frequency)``
  and calls ``close()``
- ``WorldModelLogger`` shares the transition-logging interface with
  ``AsyncWorldModelLogger`` and is implemented as a thin subclass so the
  CSV schema has a single source of truth.

Author: Karan Vasa
Date: June 2026
"""

import csv
import os
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, Union

from utils.data.async_logger import AsyncWorldModelLogger

if TYPE_CHECKING:
    from agents.actions import Action, ActionResult
    from agents.agent import Agent


class AgentLogger:
    """
    CSV logger for per-action events and per-tick agent state snapshots.

    Writes two files with persistent open handles (flushed periodically):
        actions_{timestamp}.csv — one row per executed action
        states_{timestamp}.csv  — one row per agent per logged tick

    Thread safety: ``log_all_states`` may run on the simulation's I/O
    pool while ``log_action`` runs on the main thread, so all writes are
    serialised through a lock.
    """

    FLUSH_EVERY = 50  # rows between explicit flushes

    def __init__(self, output_dir: str = "data/logs", log_every_n_ticks: int = 1):
        """
        Initialize the logger and write CSV headers.

        Args:
            output_dir: Directory for the CSV files
            log_every_n_ticks: Log agent states every N ticks
                (actions are always logged)
        """
        self.output_dir = output_dir
        self.log_every_n_ticks = max(1, int(log_every_n_ticks))

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.action_file = os.path.join(output_dir, f"agent_actions_{timestamp}.csv")
        self.state_file = os.path.join(output_dir, f"agent_states_{timestamp}.csv")

        self._lock = threading.Lock()
        self._writes_since_flush = 0
        self._closed = False

        # Persistent handles (see ECOSYSTEM.md performance notes)
        self._action_fh = open(self.action_file, "w", newline="", encoding="utf-8")
        self._action_writer = csv.writer(self._action_fh)
        self._action_writer.writerow(
            [
                "tick",
                "agent_id",
                "action",
                "success",
                "energy_cost",
                "message",
                # Structured interaction fields straight from ActionResult —
                # analysis no longer needs to regex-infer them from message
                "object_id",
                "object_type",
                "interaction_kind",
                "target_x",
                "target_y",
                "x_before",
                "y_before",
                "x_after",
                "y_after",
                "energy_before",
                "energy_after",
                "age",
                "fitness",
                "inventory_count",
            ]
        )

        self._state_fh = open(self.state_file, "w", newline="", encoding="utf-8")
        self._state_writer = csv.writer(self._state_fh)
        self._state_writer.writerow(
            [
                "tick",
                "agent_id",
                "x",
                "y",
                "direction_x",
                "direction_y",
                "energy",
                "energy_pct",
                "age",
                "alive",
                "fitness",
                "inventory_count",
                "metabolism_rate",
                "vision_radius",
                "generation",
                "lineage_id",
            ]
        )
        # Headers must be on disk immediately — consumers (tests, live
        # analysis) read the files while the simulation is still running
        self._action_fh.flush()
        self._state_fh.flush()

        print(
            f"AgentLogger initialized (state frequency: every "
            f"{self.log_every_n_ticks} tick(s)):"
        )
        print(f"  Actions: {self.action_file}")
        print(f"  States:  {self.state_file}")

    def _maybe_flush(self) -> None:
        """Flush both handles every FLUSH_EVERY writes (lock held)."""
        self._writes_since_flush += 1
        if self._writes_since_flush >= self.FLUSH_EVERY:
            self._action_fh.flush()
            self._state_fh.flush()
            self._writes_since_flush = 0

    def log_action(
        self,
        tick: int,
        agent: "Agent",
        action: "Action",
        result: "ActionResult",
        x_before: int,
        y_before: int,
        energy_before: float,
    ) -> None:
        """
        Log one executed action.

        Args:
            tick: Simulation tick
            agent: The acting agent (post-action state)
            action: Action taken
            result: Execution result
            x_before: X position before the action
            y_before: Y position before the action
            energy_before: Energy before the action
        """
        if self._closed:
            return
        with self._lock:
            self._action_writer.writerow(
                [
                    tick,
                    agent.id,
                    action.name,
                    result.success,
                    round(float(result.energy_cost), 4),
                    result.message,
                    result.object_id,
                    result.object_type,
                    result.interaction_kind,
                    result.target_x,
                    result.target_y,
                    x_before,
                    y_before,
                    agent.x,
                    agent.y,
                    round(float(energy_before), 3),
                    round(float(agent.energy), 3),
                    agent.age,
                    round(float(agent.fitness), 3),
                    len(agent.inventory),
                ]
            )
            self._maybe_flush()

    def log_all_states(self, tick: int, agents: Union[dict, Iterable["Agent"]]) -> None:
        """
        Log a state snapshot for every agent.

        Args:
            tick: Simulation tick
            agents: Live ``{id: agent}`` dict (serial path) or a snapshot
                iterable of agents (parallel I/O-pool path)
        """
        if self._closed or tick % self.log_every_n_ticks != 0:
            return

        agent_iter = agents.values() if hasattr(agents, "values") else agents
        with self._lock:
            for agent in agent_iter:
                self._state_writer.writerow(
                    [
                        tick,
                        agent.id,
                        agent.x,
                        agent.y,
                        agent.direction[0],
                        agent.direction[1],
                        round(float(agent.energy), 3),
                        round(float(agent.energy / agent.max_energy), 4),
                        agent.age,
                        agent.alive,
                        round(float(agent.fitness), 3),
                        len(agent.inventory),
                        round(float(agent.metabolism_rate), 4),
                        agent.vision_radius,
                        agent.genome.generation,
                        agent.genome.lineage_id,
                    ]
                )
            # State snapshots are low-volume and read while the run is
            # live — flush every call (actions stay batched)
            self._state_fh.flush()

    def close(self) -> None:
        """Flush and close both CSV files (idempotent)."""
        if self._closed:
            return
        with self._lock:
            self._closed = True
            for fh in (self._action_fh, self._state_fh):
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
        print(f"AgentLogger closed ({self.action_file})")


class WorldModelLogger(AsyncWorldModelLogger):
    """
    World-model transition logger (synchronous-flush variant).

    Identical schema and interface to AsyncWorldModelLogger — the CSV
    layout has a single source of truth there — retained as a distinct
    name because main.py, tests, and docs refer to both. ``close()``
    flushes all buffered rows before returning, so files are complete
    immediately after closing.
    """

    pass
