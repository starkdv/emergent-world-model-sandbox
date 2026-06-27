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
    width: int = 96,
    height: int = 96,
    n_agents: int = 24,
    seed: int = 7,
    brain_version=3,
):
    """
    Build a fresh, *populated* heightmap world for the viewer: elevation +
    biomes, a day/night sky, signalling on, a scatter of plants ("trees"),
    berries, and seeds (so agents have something to forage and the scene isn't
    barren), and a handful of agents. Resource spawning is on so food
    regenerates over a long watch.
    """
    import numpy as np

    from agents.agent import Agent
    from agents.genome import Genome
    from agents.brain import calculate_weight_count_for_config
    from world.world import World
    from world.objects import PlantComponent
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
        safety_spawn_rate=0.02,
        min_resources=max(20, (width * height) // 200),
        parallel=False,
    )

    rng = np.random.RandomState(seed)
    area = width * height

    def _scatter(type_id, count, mature=False):
        placed = 0
        attempts = 0
        while placed < count and attempts < count * 40:
            attempts += 1
            x = int(rng.randint(0, width))
            y = int(rng.randint(0, height))
            tile = world.tiles[y][x]
            if not tile.can_support_plant():
                continue
            obj = ObjectRegistry.create(type_id, x, y)
            if mature:
                pc = obj.get_component(PlantComponent)
                if pc is not None:
                    pc.age = pc.mature_age  # immediately fruit-bearing
            if world.add_object(obj):
                placed += 1
        return placed

    # "trees" (berry plants), half already mature; berries; a few seeds.
    _scatter("berry_plant", int(area * 0.012), mature=False)
    _scatter("berry_plant", int(area * 0.012), mature=True)
    _scatter("berry", int(area * 0.02))
    _scatter("berry_seed", int(area * 0.008))

    brain_cfg = {"version": brain_version}
    prev_cfg = getattr(Agent, "brain_config", None)
    Agent.brain_config = brain_cfg
    try:
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


def session_from_config(
    config_path: str = "config/default.yaml",
    *,
    learning: bool = True,
    sustain: bool = True,
    load_weights: Optional[str] = None,
) -> SimSession:
    """
    Build a streamable session straight from a project config file.

    This is the config-driven world the viewer should normally show: the
    configured size, terrain generator (heightmap → biomes), climate, signal
    and social settings, then the same initial resource population and agent
    spawn the headless runner uses. Unlike ``build_demo_world`` (a fixed
    self-contained scene), everything here follows ``config_path``.

    Args:
        config_path: YAML config (defaults to config/default.yaml).
        learning: enable lifetime RL learning on spawned agents so they
            actually learn to forage instead of dying off (recommended for a
            watchable world).
        sustain: force reproduction on so the population is self-sustaining
            for a long watch, even though the file may ship it disabled.
        load_weights: optional .npz of pre-trained genome weights (from
            ``main.py --save-weights`` or ``scripts/dream_evolve.py``) to seed
            every agent of the configured brain with, instead of random genomes
            — so the live view starts with trained behaviour. Migrated onto the
            configured brain when the layout differs.
    """
    import random
    import numpy as np
    import yaml

    from agents.agent import Agent
    from agents.genome import Genome
    from agents.brain import calculate_weight_count_for_config, _is_v35
    from agents.brain.instincts import InstinctModule
    from agents.brain.spec import set_observation_version
    from utils.agents.learning_utils import RewardConfig, set_active_reward_config
    from world.world import World
    from world.object_registry import ObjectRegistry, register_builtin_objects

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ObjectRegistry._definitions.clear()
    register_builtin_objects()

    wcfg = config.get("world", {})
    tcfg = config.get("terrain", {})
    pcfg = config.get("plants", {})
    scfg = config.get("soil", {})
    rcfg = config.get("resources", {})
    acfg = config.get("agents", {})
    brain_cfg = config.get("brain", {"version": 3})

    # Activate observation layout + reward diet from config (as main.py does).
    set_observation_version(2 if _is_v35(brain_cfg.get("version", 2)) else 1)
    set_active_reward_config(RewardConfig.from_dict(config.get("reward", None)))

    world = World(
        width=wcfg.get("width", 100),
        height=wcfg.get("height", 100),
        seed=wcfg.get("seed", None),
        soil_ratio=tcfg.get("soil_ratio", 0.695),
        rock_ratio=tcfg.get("rock_ratio", 0.2),
        water_ratio=tcfg.get("water_ratio", 0.1),
        sand_ratio=tcfg.get("sand_ratio", 0.005),
        fertility_range=tuple(tcfg.get("fertility_range", [0.3, 1.0])),
        moisture_range=tuple(tcfg.get("moisture_range", [0.2, 0.8])),
        terrain_generator=tcfg.get("generator", "legacy"),
        heightmap_config=tcfg.get("heightmap", None),
        environment_config=config.get("environment", None),
        fire_config=config.get("fire", None),
        signal_config=config.get("signal", None),
        social_config=config.get("social", None),
        performance_config=config.get("performance", None),
        agents_visible=wcfg.get("agents_visible", False),
        agent_collision=wcfg.get("agent_collision", False),
        allow_stacking=wcfg.get("allow_stacking", False),
        plant_mature_age=pcfg.get("mature_age", 100),
        plant_max_age=pcfg.get("max_age", 500),
        germination_success_rate=pcfg.get("germination_success_rate", 0.75),
        seed_max_age=pcfg.get("seed_max_age", 200),
        max_neighbor_plants=pcfg.get("max_neighbor_plants", 3),
        neighbor_radius=pcfg.get("neighbor_radius", 2),
        fertility_recovery_rate=scfg.get("fertility_recovery_rate", 0.0005),
        moisture_evaporation_rate=scfg.get("moisture_evaporation_rate", 0.0002),
        moisture_recovery_rate=scfg.get("moisture_recovery_rate", 0.0008),
        berry_calories=rcfg.get("berry_calories", 20.0),
        safety_spawn_rate=wcfg.get("resource_spawn_rate", 0.01),
        min_resources=rcfg.get("min_resources", 20),
        parallel=False,
    )

    # --- initial resources (mirrors main.py) ---
    initial = wcfg.get("initial_resources", 50)
    occupied = set()
    rng = random.Random(world.seed)

    def _spawn(type_id, count, plantable_only):
        placed = 0
        attempts = 0
        while placed < count and attempts < count * 20:
            attempts += 1
            x = rng.randint(0, world.width - 1)
            y = rng.randint(0, world.height - 1)
            if (x, y) in occupied:
                continue
            tile = world.get_tile(x, y)
            if tile is None:
                continue
            if plantable_only and not tile.is_plantable():
                continue
            if not plantable_only and not world.is_valid_position(x, y):
                continue
            obj = ObjectRegistry.create(type_id, x, y)
            if world.add_object(obj):
                occupied.add((x, y))
                placed += 1

    _spawn("berry_plant", max(initial // 2, 1), True)
    _spawn("berry", max(initial // 3, 1), False)
    _spawn("berry_seed", max(initial // 4, 1), True)

    # --- agents (optionally two competing brain cohorts) ---
    learning_cfg = config.get("learning", {})
    use_rl = learning and config.get("evolution", {}).get("mode", "rl") == "rl"
    instinct_cfg = brain_cfg.get("instincts", None)

    comp = config.get("competition", {}) or {}
    comp_on = bool(comp.get("enabled", False))
    old_brain_cfg = {"version": comp.get("old_brain_version", 2)}
    old_label = comp.get("old_label", f"v{old_brain_cfg['version']}-old")
    new_label = comp.get("new_label", f"v{brain_cfg.get('version', 3)}-new")
    n = acfg.get("initial_population", 20)
    n_old = int(round(n * float(comp.get("old_fraction", 0.15)))) if comp_on else 0

    # Optional: seed the configured (new) brain with pre-trained genome weights.
    loaded_w = None
    if load_weights:
        from agents.brain import adapt_loaded_genome
        from utils.agents import BestAgentTracker

        raw = BestAgentTracker.load_best_weights(load_weights)
        if raw is not None:
            loaded_w = adapt_loaded_genome(raw, brain_cfg)
            if loaded_w is not None:
                print(f"Seeding agents with weights from {load_weights}")
            else:
                print(f"⚠️  {load_weights} doesn't fit the configured brain — random")

    def _spawn_agent(x, y, cfg, cohort):
        Agent.brain_config = cfg
        Agent.instinct_config = instinct_cfg
        count = calculate_weight_count_for_config(cfg)
        if loaded_w is not None and cfg is brain_cfg and len(loaded_w) == count:
            g = Genome(np.asarray(loaded_w, dtype=np.float32).copy(), {})
        else:
            g = Genome.random(count, {})
        agent = Agent(
            x=x,
            y=y,
            genome=g,
            max_energy=acfg.get("max_energy", 200.0),
            max_age=acfg.get("max_age", 1000),
            inventory_size=acfg.get("inventory_size", 5),
            metabolism_rate=acfg.get("metabolism_rate", 0.5),
        )
        agent.cohort = cohort
        if use_rl:
            agent.enable_learning(
                algorithm=learning_cfg.get("algorithm", "a2c"),
                compute_backend=learning_cfg.get("compute_backend", "auto"),
                compute_device=learning_cfg.get("compute_device", "auto"),
            )
        world.add_agent(agent)

    try:
        placed = 0
        attempts = 0
        while placed < n and attempts < n * 60:
            attempts += 1
            x = rng.randint(0, world.width - 1)
            y = rng.randint(0, world.height - 1)
            if not world.tiles[y][x].is_passable():
                continue
            if placed < n_old:
                _spawn_agent(x, y, old_brain_cfg, old_label)
            else:
                _spawn_agent(x, y, brain_cfg, new_label)
            placed += 1
    finally:
        # leave the class default on the new (configured) brain for any later
        # offspring without a recorded config
        Agent.brain_config = brain_cfg

    # Reproduction / calamity from config; force reproduction on for a
    # self-sustaining population if requested (viewers watch for a long time).
    repro = dict(config.get("reproduction", {}) or {})
    if sustain:
        repro["enabled"] = True
    world.reproduction_config = repro
    if "calamity" in config:
        world.calamity_config = config["calamity"]

    # InstinctModule import kept to validate config early (mirrors main.py).
    InstinctModule.from_config(Agent.instinct_config)

    return SimSession(world)
