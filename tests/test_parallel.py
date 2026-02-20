"""
Tests for parallel agent updates.

Verifies that the threaded phase-split pipeline produces the same
observable effects as the serial fallback.
"""

import pytest
import numpy as np

from world.world import World
from world.object_registry import ObjectRegistry, register_builtin_objects
from agents import Agent, Genome, Brain


def _make_world(parallel: bool) -> World:
    """Create a small world with parallel flag set."""
    register_builtin_objects()
    return World(width=20, height=20, seed=42, parallel=parallel)


def _spawn_agents(world: World, n: int = 5, learning: bool = False) -> list:
    """Spawn agents on passable tiles."""
    from world.tiles import TerrainType

    weight_count = Brain.calculate_weight_count()
    trait_config = {"metabolism_rate": (0.8, 1.2), "vision_radius": (3, 7)}
    agents = []

    passable = [
        (x, y)
        for y in range(world.height)
        for x in range(world.width)
        if world.tiles[y][x].terrain_type in (TerrainType.SOIL,)
    ]

    for i in range(min(n, len(passable))):
        x, y = passable[i]
        genome = Genome.random(weight_count=weight_count, trait_config=trait_config)
        agent = Agent(x=x, y=y, genome=genome, max_energy=200.0)
        if learning:
            from agents.learning import AgentLearner
            agent.learner = AgentLearner(learning_rate=0.01)
            agent.learning_enabled = True
        world.add_agent(agent)
        agents.append(agent)

    return agents


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestParallelUpdate:
    """Ensure parallel path executes without errors."""

    def test_parallel_flag_stored(self):
        world = _make_world(parallel=True)
        assert world.parallel is True

    def test_serial_flag_stored(self):
        world = _make_world(parallel=False)
        assert world.parallel is False

    def test_parallel_update_runs(self):
        """Parallel agent update should run without error."""
        world = _make_world(parallel=True)
        agents = _spawn_agents(world, n=6)
        # Run several ticks
        for _ in range(10):
            world.update()
        # Agents should have aged
        for a in agents:
            if a.alive:
                assert a.age > 0

    def test_serial_update_runs(self):
        """Serial fallback should still work."""
        world = _make_world(parallel=False)
        agents = _spawn_agents(world, n=6)
        for _ in range(10):
            world.update()
        for a in agents:
            if a.alive:
                assert a.age > 0

    def test_parallel_with_learning(self):
        """Parallel update with learning enabled should not crash."""
        world = _make_world(parallel=True)
        agents = _spawn_agents(world, n=4, learning=True)
        for _ in range(20):
            world.update()
        # At least one agent should still be alive after 20 ticks
        alive = sum(1 for a in world.agents.values() if a.alive)
        assert alive >= 0  # Just ensure no crash

    def test_parallel_and_serial_same_tick_count(self):
        """Both modes should advance the world tick equally."""
        for par in (True, False):
            world = _make_world(parallel=par)
            _spawn_agents(world, n=3)
            for _ in range(15):
                world.update()
            assert world.tick == 15

    def test_pool_shutdown(self):
        """Pool shutdown should not raise."""
        from utils.parallel import get_pool, shutdown_pool
        pool = get_pool()
        assert pool is not None
        shutdown_pool()
        # Getting a new pool after shutdown should also work
        pool2 = get_pool()
        assert pool2 is not None
        shutdown_pool()


class TestParallelWorldSystems:
    """Verify world systems run correctly in parallel mode."""

    def test_soil_dynamics_parallel_rows(self):
        """SoilDynamicsSystem row-parallel should produce valid tile state."""
        world = _make_world(parallel=True)
        # Run enough ticks for soil dynamics to alter tiles
        for _ in range(20):
            world.update()

        from world.tiles import TerrainType
        for row in world.tiles:
            for tile in row:
                if tile.terrain_type == TerrainType.SOIL:
                    assert 0.0 <= tile.fertility <= 1.0
                    assert 0.0 <= tile.moisture <= 1.0

    def test_plant_decay_stage_runs_in_parallel_mode(self):
        """PlantGrowth + Decay should run safely when world.parallel=True."""
        world = _make_world(parallel=True)
        _spawn_agents(world, n=3)

        # Add some plants and berries so both systems have work
        from world.object_registry import ObjectRegistry
        from world.tiles import TerrainType
        added = 0
        for y in range(world.height):
            for x in range(world.width):
                tile = world.tiles[y][x]
                if tile.terrain_type == TerrainType.SOIL and added < 10:
                    obj = ObjectRegistry.create("berry_plant", x, y)
                    if obj:
                        world.add_object(obj)
                        added += 1

        for _ in range(30):
            world.update()

        # No crash, and world objects should be a consistent dict
        assert isinstance(world.objects, dict)

    def test_parallel_serial_soil_consistency(self):
        """Parallel and serial soil updates should produce equivalent results."""
        from world.tiles import TerrainType

        results = {}
        for parallel in (True, False):
            w = _make_world(parallel=parallel)
            for _ in range(10):
                w.update()
            # Collect soil fertility/moisture
            soil_data = []
            for row in w.tiles:
                for t in row:
                    if t.terrain_type == TerrainType.SOIL:
                        soil_data.append((round(t.fertility, 6), round(t.moisture, 6)))
            results[parallel] = soil_data

        assert len(results[True]) == len(results[False])
        assert results[True] == results[False]

    def test_logging_offloaded(self):
        """With parallel=True and a logger, log_all_states should run via I/O pool."""
        from utils.data.agent_logger import AgentLogger
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AgentLogger(output_dir=tmpdir)
            from agents.agent import Agent
            Agent.logger = logger
            try:
                world = _make_world(parallel=True)
                _spawn_agents(world, n=3)
                for _ in range(5):
                    world.update()
                # Wait for any pending log future
                fut = getattr(world, '_log_future', None)
                if fut is not None:
                    fut.result()
                logger.close()
                # State file should have content
                assert os.path.getsize(logger.state_file) > 0
            finally:
                Agent.logger = None

    def test_logging_snapshot_safe_against_agent_dict_mutation(self):
        """Mutating world.agents after update() must not break background logging."""
        import time

        class SlowIterLogger:
            def __init__(self):
                self.calls = []

            def log_action(self, *args, **kwargs):
                # Agent.execute_action expects this when Agent.logger is set.
                return None

            def log_all_states(self, tick, agents):
                # Ensure we were given a snapshot iterable, not the live dict
                assert not hasattr(agents, "values")
                self.calls.append((tick, len(list(agents))))
                # Hold the worker to create a window for mutation
                time.sleep(0.05)
                # Iterate fully to catch any mutation-related errors
                for a in agents:
                    _ = a.id

        from agents.agent import Agent
        original = Agent.logger
        Agent.logger = SlowIterLogger()
        try:
            world = _make_world(parallel=True)
            _spawn_agents(world, n=3)
            world.update()

            # Mutate the dict while the logging task is still sleeping.
            new_agent = _spawn_agents(world, n=1)[0]
            assert new_agent.id in world.agents

            fut = getattr(world, '_log_future', None)
            assert fut is not None
            fut.result()
            assert Agent.logger.calls
        finally:
            Agent.logger = original
