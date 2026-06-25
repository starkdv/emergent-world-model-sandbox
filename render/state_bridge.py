"""
World → render state bridge (Frontend 3D, phase F0).

A **read-only** view of the simulation for any 3D/voxel front end. It never
mutates the world, so it cannot affect determinism or the science — it only
*reads* `World` and emits compact, JSON-serializable state:

  * ``world_snapshot(world)`` — a full frame: the (mostly static) terrain grid
    packed as base64 byte arrays, plus the current objects, agents, and sky.
    Sent once on connect, and whenever a client needs a resync.
  * ``StateTracker`` — diffs successive ticks into ``delta(world)`` payloads:
    only the objects/agents that moved/spawned/were-removed, the pheromone
    cells that changed, and the scalar sky state. Terrain is sent in the
    snapshot and only re-sent for the cells that mutate (fire, sand, rivers).

Design notes:
  * Grids are packed as raw little-endian bytes + base64 so the wire payload is
    small (a 100×100 uint8 grid is 10 KB raw, far less gzipped) and the client
    can blit them straight into typed arrays. ``decode_grid`` is the inverse,
    used by the tests and by any Python consumer.
  * Coordinates and ids are plain ints; ratios are quantized to bytes (0–255)
    for the grids but kept as floats for the per-entity fields the camera needs
    to be smooth (agent energy, sky scalars).
  * Elevation is the one field a 3D client cannot do without — it is the W2
    heightmap, already on every tile — so it leads the terrain pack.

Author: Karan Vasa
"""

from __future__ import annotations

import base64
from typing import Dict, List

import numpy as np

# Schema version so a client can reject an incompatible bridge.
BRIDGE_VERSION = 1

# Terrain enum → small int code (stable wire encoding; the client maps codes to
# block palettes). Kept here so the bridge owns the contract, not the renderer.
_TERRAIN_CODE = {"soil": 0, "rock": 1, "water": 2, "sand": 3}


# ---------------------------------------------------------------------------
# Grid packing helpers
# ---------------------------------------------------------------------------


def _pack_grid(arr: np.ndarray) -> dict:
    """Pack a 2-D numpy array into a base64 payload with shape + dtype."""
    arr = np.ascontiguousarray(arr)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_grid(packed: dict) -> np.ndarray:
    """Inverse of :func:`_pack_grid` (used by tests / Python consumers)."""
    raw = base64.b64decode(packed["b64"])
    arr = np.frombuffer(raw, dtype=np.dtype(packed["dtype"]))
    return arr.reshape(tuple(packed["shape"]))


# ---------------------------------------------------------------------------
# Per-entity views
# ---------------------------------------------------------------------------


def _object_category(world, obj) -> str:
    """Coarse render category for an object (drives the client's model choice)."""
    try:
        from world.object_registry import ObjectRegistry

        cat = ObjectRegistry.get_category(obj)
        if cat:
            return cat
    except Exception:
        pass
    return "object"


def _object_value(world, obj) -> float:
    """
    A 0–1 'state' scalar for an object so the view can show growth/decay:
    food → freshness, plant → maturity, seed → viability. 0 when N/A.
    """
    try:
        from world.objects import EdibleComponent, PlantComponent, SeedComponent

        e = obj.get_component(EdibleComponent)
        if e is not None:
            return round(max(0.0, min(1.0, float(getattr(e, "freshness", 0.0)))), 3)
        p = obj.get_component(PlantComponent)
        if p is not None and getattr(p, "mature_age", 0):
            return round(max(0.0, min(1.0, p.age / p.mature_age)), 3)
        s = obj.get_component(SeedComponent)
        if s is not None and getattr(s, "max_age", 0):
            return round(max(0.0, min(1.0, 1.0 - s.time_in_soil / s.max_age)), 3)
    except Exception:
        pass
    return 0.0


def _object_view(world, obj) -> dict:
    """Render fields for one WorldObject. ``type_id`` selects the model."""
    return {
        "id": int(obj.id),
        "x": int(obj.x),
        "y": int(obj.y),
        "type_id": getattr(obj, "type_id", "") or "",
        "category": _object_category(world, obj),
        "terrain": bool(getattr(obj, "is_terrain", False)),
        "value": _object_value(world, obj),  # freshness / maturity / viability
        "planted": bool(getattr(obj, "planted_by_agent", False)),
    }


def _agent_view(agent, world=None) -> dict:
    """
    Render fields for one agent — a faithful slice of its real state so the 3D
    view mirrors the simulation: position/facing, energy, age, what it's
    carrying, its last action, and lineage. ``lineage`` lets the client tint
    families. All read-only.
    """
    max_e = getattr(agent, "max_energy", 0.0) or 1.0
    max_a = getattr(agent, "max_age", 0) or 1
    g = getattr(agent, "genome", None)

    # last action (the brain's most recent decision), name if available
    pa = getattr(agent, "_previous_action", None)
    action = getattr(pa, "name", None)

    # inventory composition (read-only component lookups)
    inv = list(getattr(agent, "inventory", []) or [])
    has_food = has_seed = False
    if world is not None and inv:
        try:
            from world.objects import EdibleComponent, SeedComponent

            for oid in inv:
                o = world.objects.get(oid)
                if o is None:
                    continue
                if o.get_component(EdibleComponent) is not None:
                    has_food = True
                if o.get_component(SeedComponent) is not None:
                    has_seed = True
        except Exception:
            pass

    return {
        "id": int(agent.id),
        "x": int(agent.x),
        "y": int(agent.y),
        "dir": [int(agent.direction[0]), int(agent.direction[1])],
        "energy": round(float(agent.energy) / float(max_e), 4),
        "age": round(float(getattr(agent, "age", 0)) / float(max_a), 4),
        "alive": bool(getattr(agent, "alive", True)),
        "action": action,
        "inv": len(inv),
        "has_food": has_food,
        "has_seed": has_seed,
        "lineage": int(getattr(g, "lineage_id", -1)) if g is not None else -1,
        "generation": int(getattr(g, "generation", 0)) if g is not None else 0,
    }


def _burning_ids(world) -> list:
    """
    Object ids currently on fire (W3 FireSystem), read-only.

    The fire system owns the burning set; we only read it, so this adds no
    state to the world and changes no dynamics. Empty when fire is off.
    """
    systems = getattr(world, "systems", None)
    fire = getattr(systems, "fire", None) if systems is not None else None
    if fire is None or not getattr(fire, "enabled", False):
        return []
    return [int(oid) for oid in getattr(fire, "burning", {})]


def _sky_view(world) -> dict:
    """Scalar sky/climate state for the skybox + lighting (W1 environment)."""
    env = getattr(world, "environment", None)
    if env is None:
        return {
            "time_of_day": 0.0,
            "light": 1.0,
            "temperature": 0.5,
            "season": 0.0,
            "raining": False,
            "drought": False,
            "enabled": False,
        }
    return {
        "time_of_day": round(float(getattr(env, "time_of_day", 0.0)), 4),
        "light": round(float(getattr(env, "light", 1.0)), 4),
        "temperature": round(float(getattr(env, "temperature", 0.5)), 4),
        "season": round(float(getattr(env, "season_phase", 0.0)), 4),
        "raining": bool(getattr(env, "raining", False)),
        "drought": bool(getattr(env, "drought", False)),
        "enabled": bool(getattr(env, "enabled", False)),
    }


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def _terrain_pack(world) -> dict:
    """
    Pack the terrain grid the voxel client builds columns from.

    Four byte-grids (height × width): elevation, terrain code, fertility,
    moisture — each quantized to 0–255. Elevation is the column height source;
    terrain code picks the block palette; fertility/moisture tint it.
    """
    h, w = world.height, world.width
    elevation = np.zeros((h, w), dtype=np.uint8)
    terrain = np.zeros((h, w), dtype=np.uint8)
    fertility = np.zeros((h, w), dtype=np.uint8)
    moisture = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        row = world.tiles[y]
        for x in range(w):
            t = row[x]
            elevation[y, x] = int(round(max(0.0, min(1.0, t.elevation)) * 255))
            terrain[y, x] = _TERRAIN_CODE.get(t.terrain_type.value, 0)
            fertility[y, x] = int(round(max(0.0, min(1.0, t.fertility)) * 255))
            moisture[y, x] = int(round(max(0.0, min(1.0, t.moisture)) * 255))

    return {
        "width": w,
        "height": h,
        "terrain_codes": dict(_TERRAIN_CODE),
        "elevation": _pack_grid(elevation),
        "terrain": _pack_grid(terrain),
        "fertility": _pack_grid(fertility),
        "moisture": _pack_grid(moisture),
    }


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def world_snapshot(world) -> dict:
    """
    A full render frame: terrain + all objects + all agents + sky.

    JSON-serializable. Sent on connect and on resync; per-tick updates use
    :class:`StateTracker` deltas instead.
    """
    objects = [
        _object_view(world, o)
        for o in world.objects.values()
        if not getattr(o, "is_terrain", False)
    ]
    agents = [_agent_view(a, world) for a in world.agents.values()]
    pher = getattr(world, "pheromones", None)
    # brain architecture of the running agents (true to the sim): the class
    # name separates v2 (Brain) from v3/v3.5 (BrainV3), output_size 9 = v3.5.
    brain_out = None
    brain_cls = None
    for a in world.agents.values():
        b = getattr(a, "brain", None)
        brain_out = getattr(b, "output_size", None)
        brain_cls = type(b).__name__ if b is not None else None
        break
    return {
        "type": "snapshot",
        "version": BRIDGE_VERSION,
        "tick": int(world.tick),
        "terrain": _terrain_pack(world),
        "objects": objects,
        "agents": agents,
        "sky": _sky_view(world),
        "burning": _burning_ids(world),
        "has_pheromones": pher is not None,
        "signal_enabled": bool(getattr(world, "signal_enabled", False)),
        "transfer_enabled": bool(getattr(world, "transfer_enabled", False)),
        "brain_output_size": brain_out,
        "brain_class": brain_cls,
    }


# ---------------------------------------------------------------------------
# Delta tracking
# ---------------------------------------------------------------------------


class StateTracker:
    """
    Diffs successive world states into compact per-tick deltas.

    Holds the last-emitted object positions and agent states so ``delta(world)``
    can report only what changed. The first call returns everything as
    "added"/"changed"; later calls return just the diffs. A client applies a
    delta on top of the last snapshot/delta it holds.
    """

    def __init__(self):
        self._obj_pos: Dict[int, tuple] = {}
        self._agents: Dict[int, dict] = {}

    def delta(self, world) -> dict:
        # --- objects: added / moved / removed (non-terrain only) ---
        seen_obj = set()
        obj_upserts: List[dict] = []
        for o in world.objects.values():
            if getattr(o, "is_terrain", False):
                continue
            oid = int(o.id)
            seen_obj.add(oid)
            pos = (int(o.x), int(o.y))
            if self._obj_pos.get(oid) != pos:
                obj_upserts.append(_object_view(world, o))
                self._obj_pos[oid] = pos
        removed_obj = [oid for oid in self._obj_pos if oid not in seen_obj]
        for oid in removed_obj:
            del self._obj_pos[oid]

        # --- agents: added / changed / removed ---
        seen_ag = set()
        agent_upserts: List[dict] = []
        for a in world.agents.values():
            aid = int(a.id)
            seen_ag.add(aid)
            view = _agent_view(a, world)
            if self._agents.get(aid) != view:
                agent_upserts.append(view)
                self._agents[aid] = view
        removed_ag = [aid for aid in self._agents if aid not in seen_ag]
        for aid in removed_ag:
            del self._agents[aid]

        # --- pheromones: send the nonzero cells (sparse; field decays to 0) ---
        signals: List[list] = []
        pher = getattr(world, "pheromones", None)
        if pher is not None:
            ys, xs = np.nonzero(pher)
            for y, x in zip(ys.tolist(), xs.tolist()):
                signals.append([int(x), int(y), round(float(pher[y, x]), 4)])

        return {
            "type": "delta",
            "version": BRIDGE_VERSION,
            "tick": int(world.tick),
            "objects": obj_upserts,
            "removed_objects": removed_obj,
            "agents": agent_upserts,
            "removed_agents": removed_ag,
            "signals": signals,
            "burning": _burning_ids(world),
            "sky": _sky_view(world),
        }

    def reset(self) -> None:
        self._obj_pos.clear()
        self._agents.clear()
