"""
Serialization helpers for the Three.js web renderer.

These functions translate the live Python ``World`` (tiles, objects, agents,
and the :class:`ObjectRegistry`) into plain JSON-serialisable dictionaries
that the browser client consumes:

* :func:`build_meta`      – static data sent once (world size, palette, every
  registered object definition, action list, observation layout).
* :func:`build_state`     – per-frame dynamic snapshot (tick, counts, objects,
  agents) streamed to the client at the simulation rate.
* :func:`build_terrain`   – flat terrain-type grid (sent on change only).
* :func:`inspect_tile`    – full detail for one tile (hover / click inspector).
* :func:`inspect_agent`   – full detail for one agent (inspector panel).
* :func:`inspect_object`  – full detail for one world object.

The module has no third-party dependencies so it can be imported in headless
environments and unit-tested without a browser.

Author: Karan Vasa
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.actions import Action, DIRECTIONS
from world.objects import (
    EdibleComponent,
    SeedComponent,
    PlantComponent,
    FertilizerComponent,
    ToolComponent,
)
from world.object_registry import ObjectRegistry

# Numeric terrain codes shared with the JS client (see web/static/js/world3d.js).
TERRAIN_CODES: Dict[str, int] = {
    "soil": 0,
    "rock": 1,
    "water": 2,
    "sand": 3,
}

# Fallback terrain palette (RGB 0-255).  The sand colour is pulled from the
# registry when available so custom tuning is reflected in the browser.
TERRAIN_PALETTE: Dict[str, List[int]] = {
    "soil": [101, 67, 33],
    "rock": [105, 105, 105],
    "water": [30, 144, 255],
    "sand": [210, 180, 120],
}


# ---------------------------------------------------------------------------
# Object-definition serialisation (drives the "UI for every object" panel)
# ---------------------------------------------------------------------------


def definition_to_dict(defn) -> Dict[str, Any]:
    """
    Serialise an :class:`ObjectDefinition` into a UI-friendly dictionary.

    Every component spec and cross-cutting property is included so the
    browser can render a complete inspector card for the object type.

    Args:
        defn: The ObjectDefinition to serialise.

    Returns:
        A JSON-serialisable dictionary describing the object type.
    """
    out: Dict[str, Any] = {
        "type_id": defn.type_id,
        "display_name": defn.display_name,
        "category": defn.category,
        "color": list(defn.render.color),
        "char": defn.render.char,
        "components": [],
    }

    if defn.edible is not None:
        out["components"].append("edible")
        out["edible"] = {
            "calories": defn.edible.calories,
            "toxicity": defn.edible.toxicity,
            "freshness": defn.edible.freshness,
        }
    if defn.seed is not None:
        out["components"].append("seed")
        out["seed"] = {
            "grows_into": defn.seed.grows_into,
            "grow_time": defn.seed.grow_time,
            "required_fertility": defn.seed.required_fertility,
            "required_moisture": defn.seed.required_moisture,
            "max_age": defn.seed.max_age,
        }
    if defn.plant is not None:
        out["components"].append("plant")
        out["plant"] = {
            "mature_age": defn.plant.mature_age,
            "max_age": defn.plant.max_age,
            "produces": defn.plant.produces,
            "spawn_rate": defn.plant.spawn_rate,
        }
    if defn.fertilizer is not None:
        out["components"].append("fertilizer")
        out["fertilizer"] = {
            "fertility_boost": defn.fertilizer.fertility_boost,
            "duration": defn.fertilizer.duration,
            "radius": defn.fertilizer.radius,
        }
    if defn.tool is not None:
        out["components"].append("tool")
        out["tool"] = {
            "effect_type": defn.tool.effect_type,
            "efficiency": defn.tool.efficiency,
        }

    out["physics"] = {
        "decay_rate": defn.physics.decay_rate,
        "decompose_into": defn.physics.decompose_into,
        "decompose_chance": defn.physics.decompose_chance,
        "nutrient_return": defn.physics.nutrient_return,
    }
    out["interaction"] = {
        "pickable": defn.interaction.pickable,
        "usable": defn.interaction.usable,
        "passable": defn.interaction.passable,
        "blocks_growth": defn.interaction.blocks_growth,
    }
    out["observation"] = {
        "vision_encoding": defn.observation.vision_encoding,
        "value_source": defn.observation.value_source,
    }
    if defn.tile_effect is not None:
        out["components"].append("tile_effect")
        te = defn.tile_effect
        out["tile_effect"] = {
            "germination_multiplier": te.germination_multiplier,
            "growth_multiplier": te.growth_multiplier,
            "spawn_rate_multiplier": te.spawn_rate_multiplier,
            "spread_type_id": te.spread_type_id,
            "spread_radius": te.spread_radius,
            "spread_interval": te.spread_interval,
            "spread_blocked_by": list(te.spread_blocked_by),
            "spread_chance": te.spread_chance,
            "converts_terrain": te.converts_terrain,
            "fertility_override": te.fertility_override,
            "moisture_override": te.moisture_override,
            "reclaim_terrain": te.reclaim_terrain,
            "reclaim_interval": te.reclaim_interval,
        }
    if defn.spawn.initial_count > 0:
        out["spawn"] = {
            "initial_count": defn.spawn.initial_count,
            "terrain": defn.spawn.terrain,
        }
    return out


def build_meta(world, config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Build the static metadata payload sent to the client once at startup.

    Args:
        world: The live World instance.
        config: Optional simulation config dict (for a config summary).

    Returns:
        Dictionary with world dimensions, terrain palette, every registered
        object definition, the action list and the observation layout.
    """
    palette = dict(TERRAIN_PALETTE)
    sand_defn = ObjectRegistry.get("sand")
    if sand_defn is not None:
        palette["sand"] = list(sand_defn.render.color)

    object_types = {
        tid: definition_to_dict(defn)
        for tid, defn in ObjectRegistry.all_definitions().items()
    }

    actions = [a.name for a in Action]

    # Static observation layout (mirrors README documentation).
    observation_layout = [
        {
            "range": "0-7",
            "group": "Agent state",
            "desc": "energy, age, facing, inventory",
        },
        {"range": "8-57", "group": "Vision 5x5x2", "desc": "egocentric type + value"},
        {"range": "58-65", "group": "Stimulus", "desc": "food/seed cues, urgency"},
        {
            "range": "66-71",
            "group": "Inventory",
            "desc": "fullness, contents, calories",
        },
    ]

    config_summary: Dict[str, Any] = {}
    if config:
        evo = config.get("evolution", {})
        config_summary = {
            "evolution_mode": evo.get("mode", "neuroevolution"),
            "initial_population": config.get("agents", {}).get("initial_population"),
            "max_generations": config.get("simulation", {}).get("max_generations"),
            "ticks_per_second": config.get("simulation", {}).get(
                "ticks_per_second", 10
            ),
            "allow_stacking": config.get("world", {}).get("allow_stacking", False),
            "reproduction": bool(config.get("reproduction", {}).get("enabled", False)),
            "calamity": bool(config.get("calamity", {}).get("enabled", False)),
        }

    return {
        "world": {"width": world.width, "height": world.height, "seed": world.seed},
        "terrain_palette": palette,
        "terrain_codes": TERRAIN_CODES,
        "object_types": object_types,
        "actions": actions,
        "directions": {f"{k[0]},{k[1]}": v for k, v in DIRECTIONS.items()},
        "observation_layout": observation_layout,
        "config": config_summary,
    }


# ---------------------------------------------------------------------------
# Per-frame state
# ---------------------------------------------------------------------------


def _object_render_record(obj) -> Optional[Dict[str, Any]]:
    """
    Build a compact render record for a single world object.

    Returns None for objects that should not be drawn directly here (none
    currently, but kept for forward compatibility).
    """
    cat = ObjectRegistry.get_category(obj)
    rec: Dict[str, Any] = {
        "id": obj.id,
        "t": getattr(obj, "type_id", ""),
        "x": obj.x,
        "y": obj.y,
        "cat": cat,
    }

    seed = obj.get_component(SeedComponent)
    if seed is not None:
        rec["growth"] = min(1.0, seed.time_in_soil / max(1, seed.grow_time))
        rec["planted"] = bool(getattr(obj, "planted_by_agent", False))

    plant = obj.get_component(PlantComponent)
    if plant is not None:
        rec["mature"] = plant.is_mature()
        rec["growth"] = min(1.0, plant.age / max(1, plant.mature_age))

    edible = obj.get_component(EdibleComponent)
    if edible is not None:
        rec["fresh"] = edible.freshness

    return rec


def build_state(
    world,
    paused: bool,
    speed: float,
    sim_tps: float,
) -> Dict[str, Any]:
    """
    Build the per-frame dynamic state snapshot streamed to the client.

    Objects currently held in an agent's inventory are excluded (they render
    inside the inspector instead of on the ground).

    Args:
        world: The live World instance.
        paused: Whether the simulation is paused.
        speed: Current speed multiplier.
        sim_tps: Measured simulation ticks-per-second.

    Returns:
        JSON-serialisable dictionary describing the current frame.
    """
    inv_ids: set = set()
    for agent in world.agents.values():
        inv_ids.update(agent.inventory)

    objects: List[Dict[str, Any]] = []
    for oid, obj in world.objects.items():
        if oid in inv_ids:
            continue
        rec = _object_render_record(obj)
        if rec is not None:
            objects.append(rec)

    agents: List[Dict[str, Any]] = []
    max_generation = 0
    total_energy = 0.0
    max_fitness = float("-inf")
    alive = 0
    for agent in world.agents.values():
        if not agent.alive:
            continue
        alive += 1
        dx, dy = agent.direction
        gen = agent.genome.generation
        max_generation = max(max_generation, gen)
        total_energy += agent.energy
        max_fitness = max(max_fitness, agent.fitness)
        agents.append(
            {
                "id": agent.id,
                "x": agent.x,
                "y": agent.y,
                "dx": dx,
                "dy": dy,
                "e": round(agent.energy, 2),
                "me": agent.max_energy,
                "age": agent.age,
                "gen": gen,
                "inv": len(agent.inventory),
                "fit": round(agent.fitness, 2),
            }
        )

    counts = world.get_cached_object_counts()
    avg_energy = (total_energy / alive) if alive else 0.0

    return {
        "tick": world.tick,
        "paused": paused,
        "speed": speed,
        "sim_tps": round(sim_tps, 1),
        "object_count": len(world.objects),
        "agent_count": alive,
        "counts": counts,
        "stats": {
            "max_generation": max_generation,
            "avg_energy": round(avg_energy, 1),
            "max_fitness": round(max_fitness, 2) if alive else 0.0,
        },
        "objects": objects,
        "agents": agents,
    }


def build_terrain(world) -> Dict[str, Any]:
    """
    Build a flat terrain-type grid for the client (row-major, y * width + x).

    Args:
        world: The live World instance.

    Returns:
        Dictionary with width, height and a flat list of terrain codes.
    """
    codes: List[int] = []
    for row in world.tiles:
        for tile in row:
            codes.append(TERRAIN_CODES.get(tile.terrain_type.value, 0))
    return {"width": world.width, "height": world.height, "types": codes}


def terrain_signature(world) -> bytes:
    """
    Cheap signature of terrain types used to detect terrain changes.

    Args:
        world: The live World instance.

    Returns:
        A bytes object that changes when any tile's terrain type changes.
    """
    return bytes(
        TERRAIN_CODES.get(tile.terrain_type.value, 0)
        for row in world.tiles
        for tile in row
    )


# ---------------------------------------------------------------------------
# Inspection (hover / click detail panels)
# ---------------------------------------------------------------------------


def _object_detail(obj) -> Dict[str, Any]:
    """Build a full detail dictionary for a single world object."""
    tid = getattr(obj, "type_id", "")
    defn = ObjectRegistry.get(tid) if tid else None
    detail: Dict[str, Any] = {
        "id": obj.id,
        "type_id": tid,
        "name": defn.display_name if defn else "Object",
        "category": ObjectRegistry.get_category(obj),
        "x": obj.x,
        "y": obj.y,
        "color": list(defn.render.color) if defn else [200, 200, 200],
        "components": {},
    }

    edible = obj.get_component(EdibleComponent)
    if edible is not None:
        detail["components"]["edible"] = {
            "calories": round(edible.calories, 2),
            "toxicity": round(edible.toxicity, 3),
            "freshness": round(edible.freshness, 3),
            "max_freshness": round(edible.max_freshness, 3),
        }
    seed = obj.get_component(SeedComponent)
    if seed is not None:
        detail["components"]["seed"] = {
            "plant_type": seed.plant_type,
            "grow_time": seed.grow_time,
            "time_in_soil": seed.time_in_soil,
            "required_fertility": seed.required_fertility,
            "required_moisture": seed.required_moisture,
            "max_age": seed.max_age,
            "planted_by_agent": bool(getattr(obj, "planted_by_agent", False)),
        }
    plant = obj.get_component(PlantComponent)
    if plant is not None:
        detail["components"]["plant"] = {
            "age": plant.age,
            "mature_age": plant.mature_age,
            "max_age": plant.max_age,
            "is_mature": plant.is_mature(),
            "produces": plant.spawn_resource_type,
            "spawn_rate": plant.spawn_rate,
        }
    fert = obj.get_component(FertilizerComponent)
    if fert is not None:
        detail["components"]["fertilizer"] = {
            "fertility_boost": fert.fertility_boost,
            "duration": fert.duration,
            "max_duration": fert.max_duration,
            "radius": fert.radius,
        }
    tool = obj.get_component(ToolComponent)
    if tool is not None:
        detail["components"]["tool"] = {
            "effect_type": tool.effect_type,
            "efficiency": tool.efficiency,
        }
    return detail


def inspect_object(world, object_id: int) -> Optional[Dict[str, Any]]:
    """
    Return full detail for one world object, or None if it does not exist.

    Args:
        world: The live World instance.
        object_id: The object id to inspect.
    """
    obj = world.objects.get(object_id)
    if obj is None:
        return None
    return _object_detail(obj)


def inspect_agent(world, agent_id: int) -> Optional[Dict[str, Any]]:
    """
    Return full detail for one agent, or None if it does not exist.

    Args:
        world: The live World instance.
        agent_id: The agent id to inspect.
    """
    agent = world.agents.get(agent_id)
    if agent is None:
        return None

    dx, dy = agent.direction
    direction_name = DIRECTIONS.get((dx, dy), "Unknown")

    inventory: List[Dict[str, Any]] = []
    for inv_id in agent.inventory:
        inv_obj = world.objects.get(inv_id)
        if inv_obj is None:
            continue
        tid = getattr(inv_obj, "type_id", "")
        defn = ObjectRegistry.get(tid) if tid else None
        inventory.append(
            {
                "id": inv_id,
                "type_id": tid,
                "name": defn.display_name if defn else "item",
                "color": list(defn.render.color) if defn else [200, 200, 200],
                "category": ObjectRegistry.get_category(inv_obj),
            }
        )

    traits = {k: round(float(v), 3) for k, v in agent.genome.traits.items()}

    return {
        "id": agent.id,
        "x": agent.x,
        "y": agent.y,
        "alive": agent.alive,
        "energy": round(agent.energy, 2),
        "max_energy": agent.max_energy,
        "energy_pct": round(agent.energy / max(1e-9, agent.max_energy) * 100, 1),
        "age": agent.age,
        "max_age": agent.max_age,
        "generation": agent.genome.generation,
        "fitness": round(agent.fitness, 3),
        "facing": direction_name,
        "inventory_size": agent.inventory_size,
        "inventory": inventory,
        "metabolism_rate": round(agent.metabolism_rate, 3),
        "vision_radius": agent.vision_radius,
        "learning_enabled": bool(getattr(agent, "learning_enabled", False))
        and getattr(agent, "learner", None) is not None,
        "traits": traits,
    }


def inspect_tile(world, x: int, y: int) -> Optional[Dict[str, Any]]:
    """
    Return full detail for one tile including its objects and agents.

    Args:
        world: The live World instance.
        x: Tile X-coordinate.
        y: Tile Y-coordinate.
    """
    tile = world.get_tile(x, y)
    if tile is None:
        return None

    objects = [_object_detail(o) for o in world.get_objects_at(x, y)]
    agents = [
        {
            "id": a.id,
            "energy": round(a.energy, 1),
            "max_energy": a.max_energy,
            "age": a.age,
            "generation": a.genome.generation,
        }
        for a in world.agents.values()
        if a.alive and a.x == x and a.y == y
    ]

    return {
        "x": x,
        "y": y,
        "terrain": tile.terrain_type.value,
        "fertility": round(tile.fertility, 3),
        "moisture": round(tile.moisture, 3),
        "passable": tile.is_passable(),
        "plantable": tile.is_plantable(),
        "objects": objects,
        "agents": agents,
    }
