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

Author: streaming integration
"""

from typing import TYPE_CHECKING, Dict, List

from world.objects import (
    EdibleComponent,
    SeedComponent,
    PlantComponent,
    FertilizerComponent,
)

if TYPE_CHECKING:
    from world.world import World


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
    terrain: List[List[str]] = []
    fertility: List[List[float]] = []
    moisture: List[List[float]] = []
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
    agents = [
        {
            "id": a.id,
            "x": a.x,
            "y": a.y,
            "dir": list(a.direction),
            "e": round(a.energy, 1),
            "age": a.age,
        }
        for a in world.agents.values()
        if a.alive
    ]

    objects = [
        {
            "id": o.id,
            "x": o.x,
            "y": o.y,
            "t": _object_category(o),
        }
        for o in world.objects.values()
    ]

    return {
        "type": "frame",
        "tick": world.tick,
        "agents": agents,
        "objects": objects,
    }
