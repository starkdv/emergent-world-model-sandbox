"""
World checkpointing — save/resume the full simulation state (World upgrade W6b).

Long runs (and the persistent-world track) need to stop and resume *exactly*
where they left off. ``save_state`` writes a single pickle capturing everything
that determines the future of a serial run; ``load_state`` rebuilds a World and
its agents from it. Resuming in serial mode (``simulation.parallel: false``)
reproduces a bit-identical trajectory — the W6b acceptance criterion.

What is captured:
  * world scalars + feature flags + the config used to build it
  * the tile grid (terrain, fertility, moisture, elevation, occupancy)
  * every WorldObject and its components (plain Python — pickled directly)
  * the pheromone field and the environment-engine state
  * each agent's genome, physical state, and GRU hidden state
  * the id counters (so freshly-spawned ids never collide with restored ones)
  * BOTH RNG streams: Python ``random`` and NumPy global (decisions sample via
    ``np.random.choice``; placement/spawn shuffles use ``random``)

What is NOT captured (rebuilt on load): the brain/learner/planner objects (made
fresh from the genome + config), the spatial index (rebuilt from objects), and
live-only attachments like the logger. Rebuilding a brain consumes no RNG; the
agent constructor draws one ``np.random.randint`` for facing, so RNG state is
restored LAST, after all agents exist, to keep the resumed stream identical.

Author: Karan Vasa
"""

import pickle
import random
from typing import Optional

import numpy as np

CHECKPOINT_VERSION = 1


def _agent_state(agent) -> dict:
    """Serialise the parts of an agent needed to reconstruct it identically."""
    g = agent.genome
    return {
        "id": agent.id,
        "x": agent.x,
        "y": agent.y,
        "direction": tuple(agent.direction),
        "energy": agent.energy,
        "max_energy": agent.max_energy,
        "age": agent.age,
        "max_age": agent.max_age,
        "alive": agent.alive,
        "inventory": list(agent.inventory),
        "inventory_size": agent.inventory_size,
        "fitness": agent.fitness,
        "metabolism_rate": agent.metabolism_rate,
        "vision_radius": getattr(agent, "vision_radius", 5),
        "temperature": getattr(agent, "temperature", 1.0),
        # Action-streak counters drive the anti-spin energy shaping, so they
        # must round-trip or the next action's energy cost diverges.
        "previous_action": (
            int(agent._previous_action)
            if getattr(agent, "_previous_action", None) is not None
            else None
        ),
        "consecutive_turns": getattr(agent, "_consecutive_turns", 0),
        "consecutive_waits": getattr(agent, "_consecutive_waits", 0),
        # GRU hidden state (memory) — restored so the policy continues mid-thought
        "h": np.asarray(agent.h).copy() if agent.h is not None else None,
        "last_observation": (
            None
            if getattr(agent, "last_observation", None) is None
            else np.asarray(agent.last_observation).copy()
        ),
        "last_hidden_state": (
            None
            if getattr(agent, "last_hidden_state", None) is None
            else np.asarray(agent.last_hidden_state).copy()
        ),
        # Genome
        "weights": g.weights.copy(),
        "traits": dict(g.traits),
        "lineage_id": g.lineage_id,
        "generation": g.generation,
        "parent_ids": tuple(g.parent_ids),
    }


def save_state(world, path: str, *, config: Optional[dict] = None) -> str:
    """
    Write a full checkpoint of ``world`` (and its agents) to ``path``.

    Args:
        world: The live World (its ``agents`` dict is captured too).
        path: Destination file path.
        config: The config dict the world was built from. Stored so
            ``load_state`` can rebuild the World with identical parameters.
            Falls back to ``world.config`` if the World carries one.

    Returns:
        The path written.
    """
    from agents.agent import Agent
    from agents.genome import Genome
    from world.objects import WorldObject

    cfg = config if config is not None else getattr(world, "config", None)

    env = world.environment
    state = {
        "version": CHECKPOINT_VERSION,
        "config": cfg,
        "world": {
            "width": world.width,
            "height": world.height,
            "tick": world.tick,
            "seed": world.seed,
            "allow_stacking": world.allow_stacking,
            "agents_visible": getattr(world, "agents_visible", False),
            "agent_collision": getattr(world, "agent_collision", False),
            "signal_enabled": getattr(world, "signal_enabled", False),
            "signal_strength": getattr(world, "signal_strength", 1.0),
            "signal_decay": getattr(world, "signal_decay", 0.9),
            "signal_diffuse": getattr(world, "signal_diffuse", 0.0),
            "transfer_enabled": getattr(world, "transfer_enabled", False),
        },
        # Tiles and objects are plain Python objects (no back-references to the
        # World or to brains), so they pickle directly and losslessly.
        "tiles": world.tiles,
        "objects": world.objects,
        "tile_effect_object_ids": set(world._tile_effect_object_ids),
        "pheromones": (
            None if world.pheromones is None else np.asarray(world.pheromones).copy()
        ),
        "environment": dict(env.__dict__),
        "agents": [_agent_state(a) for a in world.agents.values()],
        "counters": {
            "object_next_id": WorldObject._next_id,
            "agent_next_id": Agent._next_id,
            "genome_next_lineage_id": Genome._next_lineage_id,
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
    }

    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_state(path: str, *, config: Optional[dict] = None):
    """
    Rebuild a World and its agents from a checkpoint written by ``save_state``.

    Args:
        path: Checkpoint file path.
        config: Optional config override. If None, the config stored in the
            checkpoint is used. The brain section drives how agent brains are
            reconstructed, so it must match the saved genomes' layout.

    Returns:
        The restored World (with ``world.agents`` populated).

    Raises:
        ValueError: On an unknown checkpoint version.
    """
    from agents.agent import Agent
    from agents.actions import Action
    from agents.genome import Genome
    from world.objects import WorldObject
    from world.world import World

    with open(path, "rb") as f:
        state = pickle.load(f)

    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {state.get('version')!r} "
            f"(this build writes v{CHECKPOINT_VERSION})"
        )

    cfg = config if config is not None else state.get("config") or {}
    w = state["world"]

    # Build a World shell with the saved parameters. Its __init__ generates
    # terrain and consumes RNG, but we overwrite the grid below and restore the
    # RNG streams last, so that churn is harmless.
    terrain_cfg = cfg.get("terrain", {}) if cfg else {}
    world = World(
        w["width"],
        w["height"],
        seed=w["seed"],
        soil_ratio=terrain_cfg.get("soil_ratio", 0.695),
        rock_ratio=terrain_cfg.get("rock_ratio", 0.2),
        water_ratio=terrain_cfg.get("water_ratio", 0.1),
        sand_ratio=terrain_cfg.get("sand_ratio", 0.005),
        allow_stacking=w.get("allow_stacking", False),
        parallel=cfg.get("simulation", {}).get("parallel", False) if cfg else False,
        environment_config=cfg.get("environment", None) if cfg else None,
        fire_config=cfg.get("fire", None) if cfg else None,
        agents_visible=w.get("agents_visible", False),
        agent_collision=w.get("agent_collision", False),
        signal_config=cfg.get("signal", None) if cfg else None,
        social_config=cfg.get("social", None) if cfg else None,
        performance_config=cfg.get("performance", None) if cfg else None,
    )

    # Overwrite the generated state with the saved state.
    world.tick = w["tick"]
    world.tiles = state["tiles"]
    world.objects = state["objects"]
    world._tile_effect_object_ids = set(state["tile_effect_object_ids"])
    world.pheromones = (
        None if state["pheromones"] is None else np.asarray(state["pheromones"]).copy()
    )
    world.environment.__dict__.update(state["environment"])

    # Rebuild the spatial index (W6a) from the restored objects.
    if world.food_index is not None:
        world.food_index.clear()
        for obj in world.objects.values():
            world._index_add(obj)

    # Restore the object/genome counters now (agent construction touches
    # neither: WorldObjects aren't made here, and Genomes are built with an
    # explicit lineage_id so the counter isn't drawn). The Agent counter is
    # restored AFTER the loop, because each Agent() constructor bumps it.
    counters = state["counters"]
    WorldObject._next_id = counters["object_next_id"]
    Genome._next_lineage_id = counters["genome_next_lineage_id"]

    # Configure the brain factory the same way the run did, then rebuild agents.
    brain_cfg = cfg.get("brain", None) if cfg else None
    prev_brain_config = getattr(Agent, "brain_config", None)
    if brain_cfg is not None:
        Agent.brain_config = brain_cfg
    try:
        for a in state["agents"]:
            genome = Genome(
                a["weights"],
                a["traits"],
                lineage_id=a["lineage_id"],
                generation=a["generation"],
                parent_ids=a["parent_ids"],
            )
            agent = Agent(
                x=a["x"],
                y=a["y"],
                genome=genome,
                max_energy=a["max_energy"],
                max_age=a["max_age"],
                inventory_size=a["inventory_size"],
                metabolism_rate=a.get("metabolism_rate", 0.5),
            )
            # Overwrite the constructor's defaults/randomised fields.
            agent.id = a["id"]
            agent.direction = tuple(a["direction"])
            agent.energy = a["energy"]
            agent.age = a["age"]
            agent.alive = a["alive"]
            agent.inventory = list(a["inventory"])
            agent.fitness = a["fitness"]
            agent.metabolism_rate = a.get("metabolism_rate", agent.metabolism_rate)
            agent.vision_radius = a.get("vision_radius", agent.vision_radius)
            agent.temperature = a.get("temperature", 1.0)
            if a["h"] is not None:
                agent.h = np.asarray(a["h"]).copy()
            # Anti-spin counters + learner carry-over state
            pa = a.get("previous_action")
            agent._previous_action = None if pa is None else Action(pa)
            agent._consecutive_turns = a.get("consecutive_turns", 0)
            agent._consecutive_waits = a.get("consecutive_waits", 0)
            if a.get("last_observation") is not None:
                agent.last_observation = np.asarray(a["last_observation"]).copy()
            if a.get("last_hidden_state") is not None:
                agent.last_hidden_state = np.asarray(a["last_hidden_state"]).copy()
            world.agents[agent.id] = agent
    finally:
        # Restore the Agent id counter to its saved value (the loop's
        # constructors bumped it once per agent).
        Agent._next_id = counters["agent_next_id"]
        _ = prev_brain_config  # kept for clarity; resumed run uses brain_cfg

    # Restore BOTH RNG streams LAST — after every agent has been constructed
    # (each draws one np.random.randint for facing) — so the resumed stream is
    # byte-identical to the original.
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])

    return world
