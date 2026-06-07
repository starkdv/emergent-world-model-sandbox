"""
Synchronous CSV loggers for agent behaviour and world-model training data.

This module provides two loggers used across the simulation:

* :class:`AgentLogger` – lightweight per-action and per-tick state logging
  (``agent_actions_*.csv`` and ``agent_states_*.csv``). Used by ``--log``.
* :class:`WorldModelLogger` – complete (state, action, reward, next_state,
  done) transition logging plus per-episode summaries and world-state
  snapshots (``transitions_*.csv``, ``episodes_*.csv``, ``world_states_*.csv``).
  See ``WORLD_MODEL_LOGGING_FORMAT.md`` for the column reference.

Both keep their file handles open for the lifetime of the run (avoiding
per-write open/close overhead) and flush periodically; an asynchronous,
batched variant is available as
:class:`utils.data.async_logger.AsyncWorldModelLogger`.

Author: Karan Vasa
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agents.agent import Agent
    from agents.actions import Action, ActionResult
    from world.world import World

# Flush handles to disk every N writes to bound data loss without paying the
# cost of a flush on every single row.
_FLUSH_EVERY = 50


# ---------------------------------------------------------------------------
# Agent action / state logger
# ---------------------------------------------------------------------------


class AgentLogger:
    """
    Logs per-action records and per-tick agent-state snapshots to CSV.

    Two files are produced in ``output_dir``:

    * ``agent_actions_{timestamp}.csv`` – one row per action taken, with the
      action, its result, energy, and position.
    * ``agent_states_{timestamp}.csv`` – one row per alive agent every
      ``log_every_n_ticks`` ticks.

    Attributes:
        action_file: Path to the actions CSV.
        state_file: Path to the states CSV.
    """

    def __init__(self, output_dir: str = "data/logs", log_every_n_ticks: int = 1):
        """
        Initialise the logger and open both CSV files with headers.

        Args:
            output_dir: Directory to write CSV files into (created if missing).
            log_every_n_ticks: Write state snapshots every N ticks.
        """
        self.output_dir = output_dir
        self.log_every_n_ticks = max(1, int(log_every_n_ticks))
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.action_file = os.path.join(output_dir, f"agent_actions_{timestamp}.csv")
        self.state_file = os.path.join(output_dir, f"agent_states_{timestamp}.csv")

        self._closed = False
        self._action_writes = 0
        self._state_writes = 0

        # Persistent file handles + writers.
        self._action_fh = open(self.action_file, "w", newline="", encoding="utf-8")
        self._state_fh = open(self.state_file, "w", newline="", encoding="utf-8")
        self._action_writer = csv.writer(self._action_fh)
        self._state_writer = csv.writer(self._state_fh)

        self._action_writer.writerow(
            [
                "tick",
                "agent_id",
                "action",
                "action_value",
                "success",
                "energy_cost",
                "message",
                "interaction_kind",
                "object_id",
                "object_type",
                "target_x",
                "target_y",
                "x",
                "y",
                "direction_x",
                "direction_y",
                "energy_before",
                "energy",
                "age",
                "inventory_count",
                "fitness",
            ]
        )
        self._state_writer.writerow(
            [
                "tick",
                "agent_id",
                "x",
                "y",
                "direction_x",
                "direction_y",
                "energy",
                "max_energy",
                "energy_pct",
                "age",
                "max_age",
                "generation",
                "fitness",
                "inventory_count",
                "metabolism_rate",
                "vision_radius",
            ]
        )
        self._action_fh.flush()
        self._state_fh.flush()

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
        Append a single action record.

        Args:
            tick: Current simulation tick.
            agent: Agent that performed the action.
            action: The action taken.
            result: Result of the action.
            x_before: Agent X before the action.
            y_before: Agent Y before the action.
            energy_before: Agent energy before the action.
        """
        if self._closed:
            return
        self._action_writer.writerow(
            [
                tick,
                agent.id,
                action.name,
                int(action.value),
                int(result.success),
                round(result.energy_cost, 3),
                result.message,
                result.interaction_kind,
                int(result.object_id),
                result.object_type,
                int(result.target_x),
                int(result.target_y),
                x_before,
                y_before,
                agent.direction[0],
                agent.direction[1],
                round(energy_before, 2),
                round(agent.energy, 2),
                agent.age,
                len(agent.inventory),
                round(agent.fitness, 2),
            ]
        )
        self._action_writes += 1
        if self._action_writes % _FLUSH_EVERY == 0:
            self._action_fh.flush()

    def log_all_states(self, tick: int, agents: Iterable["Agent"]) -> None:
        """
        Append a state snapshot for every alive agent (subject to frequency).

        Args:
            tick: Current simulation tick.
            agents: Iterable or dict of agents. A dict's values are used.
        """
        if self._closed:
            return
        if tick % self.log_every_n_ticks != 0:
            return

        if isinstance(agents, dict):
            agent_iter: Iterable["Agent"] = agents.values()
        else:
            agent_iter = agents

        wrote = False
        for agent in agent_iter:
            if not getattr(agent, "alive", False):
                continue
            max_energy = max(1e-9, agent.max_energy)
            self._state_writer.writerow(
                [
                    tick,
                    agent.id,
                    agent.x,
                    agent.y,
                    agent.direction[0],
                    agent.direction[1],
                    round(agent.energy, 2),
                    agent.max_energy,
                    round(agent.energy / max_energy, 3),
                    agent.age,
                    agent.max_age,
                    agent.genome.generation,
                    round(agent.fitness, 2),
                    len(agent.inventory),
                    round(agent.metabolism_rate, 4),
                    round(agent.genome.traits.get("vision_radius", 5), 2),
                ]
            )
            wrote = True
        if wrote:
            self._state_writes += 1
            # State snapshots happen at most once per tick, so flushing each
            # time is cheap and keeps the file readable mid-run.
            self._state_fh.flush()

    def close(self) -> None:
        """Flush and close both CSV files (idempotent)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._action_fh.flush()
            self._action_fh.close()
        except Exception:
            pass
        try:
            self._state_fh.flush()
            self._state_fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# World-model transition logger (synchronous)
# ---------------------------------------------------------------------------


def _transitions_header(obs_size: int) -> List[str]:
    """Build the transitions CSV header for an observation of ``obs_size``."""
    header = [
        "tick",
        "agent_id",
        "episode_step",
        "action",
        "action_value",
        "success",
        "energy_cost",
        "result_message",
        "interaction_kind",
        "object_id",
        "object_type",
        "target_x",
        "target_y",
        "x",
        "y",
        "direction_x",
        "direction_y",
        "x_next",
        "y_next",
        "direction_x_next",
        "direction_y_next",
        "energy",
        "energy_next",
        "energy_pct",
        "energy_pct_next",
        "age",
        "inventory_count",
        "inventory_count_next",
        "fitness",
        "fitness_next",
        "reward",
        "done",
        "death_reason",
        "tile_terrain",
        "tile_fertility",
        "tile_moisture",
        "tile_has_food",
        "tile_has_plant",
        "tile_has_seed",
        "tile_food_calories",
        "total_food_count",
        "total_plant_count",
        "alive_agents",
        "metabolism_rate",
        "vision_radius",
    ]
    header += [f"obs_{i}" for i in range(obs_size)]
    header += [f"obs_next_{i}" for i in range(obs_size)]
    return header


class WorldModelLogger:
    """
    Synchronous logger for world-model training data.

    Produces three CSV files (see ``WORLD_MODEL_LOGGING_FORMAT.md``):

    * ``transitions_{timestamp}.csv`` – full transition tuples + observations.
    * ``episodes_{timestamp}.csv`` – per-agent episode summaries.
    * ``world_states_{timestamp}.csv`` – per-tick world snapshots.

    Attributes:
        transitions_file: Path to the transitions CSV.
        episodes_file: Path to the episodes CSV.
        world_states_file: Path to the world-states CSV.
    """

    def __init__(self, output_dir: str = "data/logs", log_every_n_ticks: int = 1):
        """
        Initialise the logger and open all three CSV files.

        The transitions header is written lazily on the first transition so the
        observation-vector width matches the live observation size.

        Args:
            output_dir: Directory to write CSV files into (created if missing).
            log_every_n_ticks: Write world-state snapshots every N ticks.
        """
        self.output_dir = output_dir
        self.log_every_n_ticks = max(1, int(log_every_n_ticks))
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transitions_file = os.path.join(output_dir, f"transitions_{timestamp}.csv")
        self.episodes_file = os.path.join(output_dir, f"episodes_{timestamp}.csv")
        self.world_states_file = os.path.join(
            output_dir, f"world_states_{timestamp}.csv"
        )

        self._closed = False
        self._transitions_header_written = False
        self._transition_writes = 0
        self._world_state_writes = 0

        # Per-agent episode accumulators.
        self.episode_data: Dict[int, Dict[str, Any]] = {}

        # Persistent handles.
        self._trans_fh = open(self.transitions_file, "w", newline="", encoding="utf-8")
        self._epi_fh = open(self.episodes_file, "w", newline="", encoding="utf-8")
        self._world_fh = open(self.world_states_file, "w", newline="", encoding="utf-8")
        self._trans_writer = csv.writer(self._trans_fh)
        self._epi_writer = csv.writer(self._epi_fh)
        self._world_writer = csv.writer(self._world_fh)

        self._epi_writer.writerow(
            [
                "agent_id",
                "generation",
                "lineage_id",
                "start_tick",
                "end_tick",
                "duration",
                "total_reward",
                "total_actions",
                "successful_eats",
                "successful_pickups",
                "tiles_explored",
                "final_fitness",
                "death_reason",
                "final_energy",
                "max_energy_reached",
                "avg_energy",
                "metabolism_rate",
                "vision_radius",
            ]
        )
        self._world_writer.writerow(
            [
                "tick",
                "alive_agents",
                "total_agents",
                "total_food",
                "total_plants",
                "total_seeds",
                "avg_agent_energy",
                "min_agent_energy",
                "max_agent_energy",
                "avg_agent_age",
                "max_agent_age",
                "avg_fertility",
                "avg_moisture",
                "total_fitness",
                "avg_fitness",
            ]
        )
        self._epi_fh.flush()
        self._world_fh.flush()

    def log_transition(
        self,
        tick: int,
        agent: "Agent",
        action: "Action",
        result: "ActionResult",
        reward: float,
        obs_before: np.ndarray,
        obs_after: np.ndarray,
        world: "World",
        x_before: int,
        y_before: int,
        energy_before: float,
        done: bool = False,
        death_reason: str = "",
    ) -> None:
        """
        Write a single transition row (and an episode row when ``done``).

        Args mirror :meth:`AsyncWorldModelLogger.log_transition`.
        """
        if self._closed:
            return

        from world.objects import EdibleComponent, PlantComponent, SeedComponent

        # Lazily write the transitions header sized to the observation vector.
        if not self._transitions_header_written:
            self._trans_writer.writerow(_transitions_header(len(obs_before)))
            self._transitions_header_written = True

        # Episode accumulation.
        if agent.id not in self.episode_data:
            self.episode_data[agent.id] = {
                "start_tick": tick,
                "total_reward": 0.0,
                "total_actions": 0,
                "successful_eats": 0,
                "successful_pickups": 0,
                "tiles_visited": set(),
                "max_energy": energy_before,
                "energy_sum": 0.0,
                "energy_count": 0,
            }
        ep = self.episode_data[agent.id]
        ep["total_reward"] += reward
        ep["total_actions"] += 1
        ep["tiles_visited"].add((agent.x, agent.y))
        ep["max_energy"] = max(ep["max_energy"], agent.energy)
        ep["energy_sum"] += agent.energy
        ep["energy_count"] += 1
        if result.success:
            if action.name == "EAT":
                ep["successful_eats"] += 1
            elif action.name == "PICK_UP":
                ep["successful_pickups"] += 1

        # Tile context.
        tile = world.get_tile(agent.x, agent.y)
        tile_terrain = tile.terrain_type.value if tile else 0
        tile_fertility = tile.fertility if tile else 0.0
        tile_moisture = tile.moisture if tile else 0.0
        tile_has_food = 0
        tile_has_plant = 0
        tile_has_seed = 0
        tile_food_calories = 0.0
        if tile and tile.object_ids:
            for obj_id in tile.object_ids:
                obj = world.objects.get(obj_id)
                if obj is None:
                    continue
                edible = obj.get_component(EdibleComponent)
                if edible:
                    tile_has_food = 1
                    tile_food_calories = edible.calories * edible.freshness
                if obj.get_component(PlantComponent):
                    tile_has_plant = 1
                if obj.get_component(SeedComponent):
                    tile_has_seed = 1

        counts = world.get_cached_object_counts()
        max_energy = max(1e-9, agent.max_energy)

        row = [
            tick,
            agent.id,
            ep["total_actions"],
            action.name,
            int(action.value),
            int(result.success),
            round(result.energy_cost, 3),
            result.message,
            result.interaction_kind,
            int(result.object_id),
            result.object_type,
            int(result.target_x),
            int(result.target_y),
            x_before,
            y_before,
            agent.direction[0],
            agent.direction[1],
            agent.x,
            agent.y,
            agent.direction[0],
            agent.direction[1],
            round(energy_before, 2),
            round(agent.energy, 2),
            round(energy_before / max_energy, 3),
            round(agent.energy / max_energy, 3),
            agent.age,
            len(agent.inventory)
            - (1 if action.name == "PICK_UP" and result.success else 0),
            len(agent.inventory),
            round(agent.fitness - 0.1, 2),
            round(agent.fitness, 2),
            round(reward, 4),
            int(done),
            death_reason,
            tile_terrain,
            round(tile_fertility, 3),
            round(tile_moisture, 3),
            tile_has_food,
            tile_has_plant,
            tile_has_seed,
            round(tile_food_calories, 2),
            counts["total_food"],
            counts["total_plants"],
            counts["alive_agents"],
            round(agent.metabolism_rate, 4),
            round(agent.genome.traits.get("vision_radius", 5), 2),
        ]
        row.extend(round(float(v), 5) for v in obs_before)
        row.extend(round(float(v), 5) for v in obs_after)

        self._trans_writer.writerow(row)
        self._transition_writes += 1
        if self._transition_writes % _FLUSH_EVERY == 0:
            self._trans_fh.flush()

        if done:
            self._log_episode_end(agent, tick, death_reason)

    def _log_episode_end(
        self, agent: "Agent", end_tick: int, death_reason: str
    ) -> None:
        """Write an episode-summary row and drop the agent's accumulator."""
        ep = self.episode_data.pop(agent.id, None)
        if ep is None:
            return
        duration = end_tick - ep["start_tick"]
        avg_energy = (
            ep["energy_sum"] / ep["energy_count"] if ep["energy_count"] > 0 else 0.0
        )
        self._epi_writer.writerow(
            [
                agent.id,
                agent.genome.generation,
                getattr(agent.genome, "lineage_id", agent.id),
                ep["start_tick"],
                end_tick,
                duration,
                round(ep["total_reward"], 2),
                ep["total_actions"],
                ep["successful_eats"],
                ep["successful_pickups"],
                len(ep["tiles_visited"]),
                round(agent.fitness, 2),
                death_reason,
                round(agent.energy, 2),
                round(ep["max_energy"], 2),
                round(avg_energy, 2),
                round(agent.metabolism_rate, 4),
                round(agent.genome.traits.get("vision_radius", 5), 2),
            ]
        )
        self._epi_fh.flush()

    def log_world_state(self, tick: int, world: "World") -> None:
        """Write a world-state snapshot row (subject to frequency)."""
        if self._closed:
            return
        if tick % self.log_every_n_ticks != 0:
            return

        alive_agents = [a for a in world.agents.values() if a.alive]
        total_agents = len(world.agents)
        counts = world.get_cached_object_counts()

        if alive_agents:
            energies = [a.energy for a in alive_agents]
            ages = [a.age for a in alive_agents]
            fitnesses = [a.fitness for a in alive_agents]
            n = len(alive_agents)
            avg_energy = sum(energies) / n
            min_energy = min(energies)
            max_energy = max(energies)
            avg_age = sum(ages) / n
            max_age = max(ages)
            total_fitness = sum(fitnesses)
            avg_fitness = total_fitness / n
        else:
            avg_energy = min_energy = max_energy = 0.0
            avg_age = max_age = 0
            total_fitness = avg_fitness = 0.0

        avg_fertility, avg_moisture = world.get_cached_soil_stats()

        self._world_writer.writerow(
            [
                tick,
                len(alive_agents),
                total_agents,
                counts["total_food"],
                counts["total_plants"],
                counts["total_seeds"],
                round(avg_energy, 2),
                round(min_energy, 2),
                round(max_energy, 2),
                round(avg_age, 2),
                max_age,
                round(avg_fertility, 4),
                round(avg_moisture, 4),
                round(total_fitness, 2),
                round(avg_fitness, 2),
            ]
        )
        self._world_state_writes += 1
        if self._world_state_writes % _FLUSH_EVERY == 0:
            self._world_fh.flush()

    def close(self) -> None:
        """Flush and close all three CSV files (idempotent)."""
        if self._closed:
            return
        self._closed = True
        for fh in (
            getattr(self, "_trans_fh", None),
            getattr(self, "_epi_fh", None),
            getattr(self, "_world_fh", None),
        ):
            if fh is None:
                continue
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass
