"""
Live simulation session for the 3D frontend (phase F3a).

``SimSession`` wraps a running ``World`` and the F0 ``StateTracker`` into the
two calls a streaming server needs:

  * ``snapshot()`` — the full opening frame (terrain + entities + sky)
  * ``step()``     — advance one tick and return the delta since the last call

It is pure Python and transport-agnostic (the SSE server in ``render/server.py``
is one consumer; tests are another), so the live-streaming logic is exercised
without any sockets.

``build_demo_world`` / ``session_from_checkpoint`` are convenience factories:
the first spins up a fresh heightmap world with a day/night sky and a few agents
(so there is always something to look at), the second resumes a W6b checkpoint —
"fly around a saved run" falls straight out of the W6b serialization.

Author: Karan Vasa
"""

from __future__ import annotations

from typing import Optional

from render.state_bridge import StateTracker, world_snapshot


class SimSession:
    """A world + delta tracker exposed as snapshot()/step() for streaming."""

    def __init__(self, world):
        self.world = world
        self._tracker = StateTracker()

    def snapshot(self) -> dict:
        """Full opening frame; also (re)primes the delta tracker."""
        self._tracker.reset()
        snap = world_snapshot(self.world)
        # Prime the tracker against the current state so the first step() only
        # reports what changed *after* this snapshot.
        self._tracker.delta(self.world)
        return snap

    def step(self, n: int = 1) -> dict:
        """Advance ``n`` ticks (>=1) and return the delta since the last call."""
        for _ in range(max(1, n)):
            self.world.update()
        return self._tracker.delta(self.world)

    @property
    def tick(self) -> int:
        return int(self.world.tick)

    @property
    def alive_agents(self) -> int:
        return sum(1 for a in self.world.agents.values() if getattr(a, "alive", True))


def build_demo_world(
    width: int = 64,
    height: int = 64,
    n_agents: int = 12,
    seed: int = 7,
    brain_version=2,
):
    """
    Build a fresh heightmap world (elevation + biomes), a day/night sky, signals
    on, and a handful of agents — a self-contained scene for the viewer.
    """
    import numpy as np

    from agents.agent import Agent
    from agents.genome import Genome
    from agents.brain import calculate_weight_count_for_config
    from world.world import World
    from world.object_registry import ObjectRegistry, register_builtin_objects

    if not ObjectRegistry._definitions:
        register_builtin_objects()

    world = World(
        width,
        height,
        seed=seed,
        terrain_generator="heightmap",
        environment_config={"enabled": True, "day_length": 200},
        signal_config={"enabled": True},
        performance_config={"spatial_index": True},
        parallel=False,
    )

    brain_cfg = {"version": brain_version}
    prev_cfg = getattr(Agent, "brain_config", None)
    Agent.brain_config = brain_cfg
    try:
        rng = np.random.RandomState(seed)
        weight_count = calculate_weight_count_for_config(brain_cfg)
        placed = 0
        attempts = 0
        while placed < n_agents and attempts < n_agents * 50:
            attempts += 1
            x = int(rng.randint(0, width))
            y = int(rng.randint(0, height))
            if not world.tiles[y][x].is_passable():
                continue
            g = Genome.random(weight_count, {})
            world.add_agent(Agent(x=x, y=y, genome=g))
            placed += 1
    finally:
        Agent.brain_config = prev_cfg if prev_cfg is not None else brain_cfg

    return SimSession(world)


def session_from_checkpoint(path: str, config: Optional[dict] = None) -> SimSession:
    """Resume a W6b checkpoint as a streamable session."""
    from world.checkpoint import load_state

    world = load_state(path, config=config)
    return SimSession(world)
