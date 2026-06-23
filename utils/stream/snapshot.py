"""
Serialization of world state into JSON-friendly snapshots for streaming.

Snapshots are intentionally plain ``dict`` / ``list`` / scalar structures so the
WebSocket server can hand them straight to ``json.dumps``. Two shapes exist:

- ``build_init(world)``   -- sent once when a client connects. Carries the
  static-ish scene description: world dimensions and the full terrain grid.
- ``build_frame(world)``  -- sent every (Nth) tick. Carries the dynamic state:
  the current tick, all alive agents, and all world objects.

IMPORTANT: both functions iterate ``world.agents`` / ``world.objects`` and must
therefore be called from the simulation thread, immediately after
``world.update()``, where the state is consistent and nothing else is mutating
those dictionaries. Hand the returned dict to ``StreamServer.publish`` -- never
let the server thread walk the live world dictionaries (see world.py:541 for the
same snapshot-then-handoff discipline used by the logger).

Author: Vinchenzo98
"""

from typing import TYPE_CHECKING, Dict

from world.objects import (
    EdibleComponent,
    SeedComponent,
    PlantComponent,
    FertilizerComponent,
)
from world.object_registry import ObjectRegistry

if TYPE_CHECKING:
    from world.world import World

# Object render palette (0-255). Mirrors OBJECT_COLORS in
# utils/ui/gpu_renderer.py so streamed colours match the isometric GPU view.
OBJECT_COLORS: Dict[str, tuple] = {
    "plant": (34, 139, 34),
    "berry": (220, 20, 60),
    "seed": (205, 170, 125),
    "seed_planted": (255, 200, 0),
    "seed_sprouting": (120, 200, 80),
    "fertilizer": (100, 200, 100),
    "default": (200, 200, 200),
}


def _object_category(obj) -> str:
    """
    Resolve a coarse render category for a world object.

    Prefers the registry-assigned ``type_id`` (e.g. "berry", "sand"). Falls back
    to a component-derived category for legacy objects created without a
    ``type_id`` so the client always has something to colour by.

    Args:
        obj: WorldObject to classify.

    Returns:
        A short category string for the client to map to a colour / mesh.
    """
    if getattr(obj, "type_id", ""):
        return obj.type_id
    if obj.has_component(PlantComponent):
        return "plant"
    if obj.has_component(SeedComponent):
        return "seed"
    if obj.has_component(FertilizerComponent):
        return "fertilizer"
    if obj.has_component(EdibleComponent):
        return "food"
    return "object"


def _object_render(obj) -> Dict:
    """
    Resolve render attributes (colour / radius / glow / shape) for an object.

    Mirrors ``IsometricRenderer._build_object_instances`` in gpu_renderer.py so
    the WebSocket client can draw objects identically to the GPU view: seeds as
    growth-shaded diamonds, edibles/plants with the right radius and glow, etc.

    Args:
        obj: WorldObject to classify.

    Returns:
        Dict with ``color`` ([r, g, b] 0-255), ``radius``, ``glow`` and
        ``shape`` ("circle" | "diamond" | "triangle").
    """
    r, g, b = OBJECT_COLORS["default"]
    radius = 0.8
    glow = 0.0
    shape = "circle"

    is_terrain = getattr(obj, "is_terrain", False)

    # Registry colour (overridden below for component-specific shading).
    tid = getattr(obj, "type_id", "")
    if tid:
        defn = ObjectRegistry.get(tid)
        if defn is not None:
            rc = defn.render.color
            r, g, b = rc[0], rc[1], rc[2]

    seed_comp = obj.get_component(SeedComponent)
    if seed_comp is not None:
        shape = "diamond"
        agent_planted = getattr(obj, "planted_by_agent", False)
        grow_ratio = min(1.0, seed_comp.time_in_soil / max(1, seed_comp.grow_time))
        if agent_planted:
            sr, sg, sb = OBJECT_COLORS["seed_planted"]
            glow = 0.8
        else:
            sr, sg, sb = OBJECT_COLORS["seed"]
        er, eg, eb = OBJECT_COLORS["seed_sprouting"]
        t = grow_ratio
        r = sr + (er - sr) * t
        g = sg + (eg - sg) * t
        b = sb + (eb - sb) * t
        radius = 0.6

    elif obj.has_component(EdibleComponent):
        if not tid:
            r, g, b = OBJECT_COLORS["berry"]
        glow = 0.3
        radius = 0.7

    elif obj.has_component(PlantComponent):
        plant = obj.get_component(PlantComponent)
        if not tid:
            r, g, b = OBJECT_COLORS["plant"]
        if plant.is_mature():
            glow = 0.15
            radius = 1.0
        else:
            radius = 0.5 + 0.5 * min(1.0, plant.age / max(1, plant.mature_age))

    elif is_terrain:
        radius = 1.1
        shape = "diamond"

    return {
        "color": [round(r), round(g), round(b)],
        "radius": round(radius, 3),
        "glow": round(glow, 3),
        "shape": shape,
    }


def build_init(world: "World") -> Dict:
    """
    Build the one-time scene-description message for a newly connected client.

    Args:
        world: World instance (read in the simulation thread).

    Returns:
        Dict with ``type="init"``, world dimensions, and the full terrain grid
        as ``height``-row x ``width``-column arrays. ``terrain`` holds terrain
        type strings; ``fertility`` / ``moisture`` hold rounded scalars so the
        client can shade the ground plane.
    """
    terrain = []
    fertility = []
    moisture = []
    for row in world.tiles:
        terrain.append([t.terrain_type.value for t in row])
        fertility.append([round(t.fertility, 3) for t in row])
        moisture.append([round(t.moisture, 3) for t in row])

    return {
        "type": "init",
        "width": world.width,
        "height": world.height,
        "tick": world.tick,
        "terrain": terrain,
        "fertility": fertility,
        "moisture": moisture,
    }


def build_frame(world: "World") -> Dict:
    """
    Build a per-tick dynamic-state message.

    Sends the full set of alive agents and all world objects each call (simple
    and correct for small/medium worlds). For large worlds, switch to deltas
    keyed off ``world.tick`` -- the same dirty-flag trigger the GPU renderer
    uses to rebuild its instance lists.

    Args:
        world: World instance (read in the simulation thread).

    Returns:
        Dict with ``type="frame"``, the current tick, agents, and objects.
        ``agents[i].dir`` is the agent's ``(dx, dy)`` facing as a 2-element list.
    """
    agents = []
    for a in world.agents.values():
        if not a.alive:
            continue
        agents.append(
            {
                "id": a.id,
                "x": a.x,
                "y": a.y,
                "dir": list(a.direction),
                "e": round(a.energy, 1),
                "max_energy": round(a.max_energy, 1),
                "age": a.age,
                "inventory_size": a.inventory_size,
                "inventory": a.inventory,
                # Tool used since the last frame (seed/fertilizer) or None;
                # drives the "using {tool}" floating label on the client.
                "using_tool": getattr(a, "last_tool_use", None),
            }
        )
        # Clear-on-stream so each tool use is reported exactly once, even when
        # frames are sampled every Nth tick. build_frame runs in the sim thread
        # (see module docstring), so this mutation is safe.
        a.last_tool_use = None

    # Objects carried in an agent's inventory aren't drawn on the map (they ride
    # with the agent), mirroring the GPU renderer's inv_ids cull.
    inventory_ids = set()
    for a in world.agents.values():
        inventory_ids.update(a.inventory)

    objects = [
        {
            "id": o.id,
            "x": o.x,
            "y": o.y,
            "name": _object_category(o),
            **_object_render(o),
        }
        for o in world.objects.values()
        if o.id not in inventory_ids
    ]

    # Build the terrain grid fresh each call. (Previously appended to a
    # module-level list, which grew without bound and resent stale rows.)
    terrain = [
        [
            {
                "terrain_type": t.terrain_type.value,
                "x": t.x,
                "y": t.y,
                "fertility": round(t.fertility, 3),
                "moisture": round(t.moisture, 3),
            }
            for t in row
        ]
        for row in world.tiles
    ]

    world_options = [
        {
            "height": world.height,
            "width": world.width,
        }
    ]

    return {
        "type": "frame",
        "world_options": world_options,
        "tick": world.tick,
        "agents": agents,
        "objects": objects,
        "terrain": terrain
    }
