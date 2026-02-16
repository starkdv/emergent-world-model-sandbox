"""
Observation vector construction for agent perception.

Builds a normalized feature vector representing what the agent
can perceive about its environment. The observation includes:
- Agent internal state (energy, age, direction, inventory capacity)
- Vision grid (5×5 tiles around agent, terrain-layer aware)
- Stimulus features (pre-processed survival signals)
- Inventory summary

Observation layout (72 features total):
  [0..7]   Agent state        (8)
  [8..57]  Vision grid 5×5×2  (50)
  [58..65] Stimulus features   (8)
  [66..71] Inventory summary   (6)

Author: Karan Vasa
Date: November 14, 2025
Updated: February 2026 — terrain-layer awareness, registry encoding, stimulus features
"""

import math
import numpy as np
from typing import TYPE_CHECKING

from world.tiles import TerrainType
from world.objects import EdibleComponent, PlantComponent, SeedComponent

if TYPE_CHECKING:
    from agents.agent import Agent
    from world.world import World


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_observation(agent: 'Agent', world: 'World') -> np.ndarray:
    """
    Build observation vector for agent.

    The observation vector contains:
    - Agent internal state   (8 features)
    - Vision grid 5×5×2      (50 features)
    - Stimulus features       (8 features)  ← NEW
    - Inventory summary       (6 features)

    Total: 72 features (normalised to [0, 1])

    Args:
        agent: The observing agent
        world: The world being observed

    Returns:
        Normalised observation vector (72 dimensions)
    """
    obs: list[float] = []

    # 1. Agent internal state (8 features)
    obs.extend(_encode_agent_state(agent))

    # 2. Vision grid (50 features)
    obs.extend(_encode_vision(agent, world))

    # 3. Stimulus features — explicit survival signals (8 features)
    obs.extend(_encode_stimulus(agent, world))

    # 4. Inventory state (6 features)
    obs.extend(_encode_inventory(agent, world))

    return np.array(obs, dtype=np.float32)


def get_observation_size() -> int:
    """Return the size of the observation vector."""
    return 72


# ---------------------------------------------------------------------------
# 1. Agent state  (8 features)
# ---------------------------------------------------------------------------

def _encode_agent_state(agent: 'Agent') -> list[float]:
    """
    Encode agent's internal state.

    Features (8):
      [0] Energy ratio (0–1)
      [1] Age ratio (0–1)
      [2-5] Direction one-hot (N, E, S, W)
      [6] Has inventory space (0/1)
      [7] Metabolism rate (normalised)
    """
    features: list[float] = []

    features.append(agent.energy / agent.max_energy)
    features.append(agent.age / agent.max_age)

    direction_map = {
        (0, -1): [1, 0, 0, 0],   # North
        (1, 0):  [0, 1, 0, 0],   # East
        (0, 1):  [0, 0, 1, 0],   # South
        (-1, 0): [0, 0, 0, 1],   # West
    }
    features.extend(direction_map.get(agent.direction, [0, 0, 0, 0]))

    features.append(1.0 if len(agent.inventory) < agent.inventory_size else 0.0)
    features.append(min(1.0, agent.metabolism_rate / 2.0))

    return features


# ---------------------------------------------------------------------------
# 2. Vision grid  (50 features — 5×5 tiles × 2 features each)
# ---------------------------------------------------------------------------

def _encode_vision(agent: 'Agent', world: 'World') -> list[float]:
    """
    Encode the 5×5 vision grid around the agent.

    Each tile produces (type_encoding, value_encoding).
    Terrain-layer objects (e.g. sand) are transparent — the
    agent perceives the real resource underneath.
    """
    features: list[float] = []
    vision_radius = 2  # 5×5

    for dy in range(-vision_radius, vision_radius + 1):
        for dx in range(-vision_radius, vision_radius + 1):
            wx = agent.x + dx
            wy = agent.y + dy

            if not (0 <= wx < world.width and 0 <= wy < world.height):
                features.append(0.0)   # Rock (out of bounds)
                features.append(0.0)
                continue

            tile = world.tiles[wy][wx]
            t_enc, v_enc = _encode_tile(tile, world)
            features.append(t_enc)
            features.append(v_enc)

    return features


def _encode_tile(tile, world: 'World') -> tuple[float, float]:
    """
    Encode a single tile — terrain-layer aware, registry-first.

    Skips terrain-layer objects (sand, etc.) so agents perceive
    real resources (berries, plants, seeds) on the same tile.

    Returns:
        (type_encoding, value_encoding)
    """
    from world.object_registry import ObjectRegistry

    if tile.object_ids:
        # Separate terrain-layer objects from real objects
        render_obj = None
        terrain_obj = None
        for oid in tile.object_ids:
            o = world.objects.get(oid)
            if o is None:
                continue
            if ObjectRegistry.is_terrain_layer(o):
                terrain_obj = o
            else:
                render_obj = o
                break

        obj = render_obj if render_obj is not None else terrain_obj

        if obj is not None:
            # Registry-based encoding (preferred)
            tid = getattr(obj, "type_id", "")
            if tid:
                defn = ObjectRegistry.get(tid)
                if defn is not None:
                    return (
                        defn.observation.vision_encoding,
                        _compute_observation_value(obj, defn.observation.value_source),
                    )

            # Fallback: component-based encoding
            edible = obj.get_component(EdibleComponent)
            if edible is not None:
                return 1.0, min(1.0, edible.calories * edible.freshness / 50.0)

            plant = obj.get_component(PlantComponent)
            if plant is not None:
                mat = min(1.0, plant.age / plant.mature_age) if plant.mature_age > 0 else 0.0
                return 0.75, mat

            seed = obj.get_component(SeedComponent)
            if seed is not None:
                via = 1.0 - min(1.0, seed.time_in_soil / seed.max_age) if seed.max_age > 0 else 0.0
                return 0.6, via

    # No (real) objects — encode terrain
    if tile.terrain_type == TerrainType.ROCK:
        return 0.0, 0.0
    elif tile.terrain_type == TerrainType.WATER:
        return 0.25, tile.moisture
    elif tile.terrain_type == TerrainType.SAND:
        return 0.15, (tile.fertility + tile.moisture) / 2.0
    elif tile.terrain_type == TerrainType.SOIL:
        return 0.5, (tile.fertility + tile.moisture) / 2.0
    else:
        return 0.0, 0.0


def _compute_observation_value(obj, value_source: str) -> float:
    """Compute observation value from a registry value_source descriptor."""
    if value_source == "freshness":
        edible = obj.get_component(EdibleComponent)
        if edible is not None:
            return min(1.0, edible.calories * edible.freshness / 50.0)
    elif value_source == "maturity":
        plant = obj.get_component(PlantComponent)
        if plant is not None:
            return min(1.0, plant.age / plant.mature_age) if plant.mature_age > 0 else 0.0
    elif value_source == "viability":
        seed = obj.get_component(SeedComponent)
        if seed is not None:
            return 1.0 - min(1.0, seed.time_in_soil / seed.max_age) if seed.max_age > 0 else 0.0
    elif value_source == "duration":
        from world.objects import FertilizerComponent
        fert = obj.get_component(FertilizerComponent)
        if fert is not None:
            return fert.duration / fert.max_duration if fert.max_duration > 0 else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# 3. Stimulus features  (8 features — explicit survival signals)
# ---------------------------------------------------------------------------

def _encode_stimulus(agent: 'Agent', world: 'World') -> list[float]:
    """
    Pre-processed survival signals that give the brain direct,
    actionable information without requiring spatial learning.

    Features (8):
      [0] food_on_tile      — 1.0 if pickable food on agent's tile
      [1] seed_on_tile      — 1.0 if pickable seed on agent's tile
      [2] food_ahead        — 1.0 if food within 3 tiles in facing direction
      [3] resource_ahead    — 1.0 if any pickable within 3 tiles ahead
      [4] nearest_food_prox — proximity to nearest food (1.0=on it, 0=far/none)
      [5] food_dir_match    — cosine of angle between facing dir and food dir
      [6] energy_urgency    — non-linear urgency signal (high when low energy)
      [7] can_interact      — 1.0 if PICK_UP, EAT, or USE is possible now
    """
    from world.object_registry import ObjectRegistry

    features: list[float] = [0.0] * 8

    # --- Current tile analysis ---
    tile = world.tiles[agent.y][agent.x]
    for oid in tile.object_ids:
        o = world.objects.get(oid)
        if o is None or ObjectRegistry.is_terrain_layer(o):
            continue
        if o.get_component(EdibleComponent) is not None:
            features[0] = 1.0  # food_on_tile
        if o.get_component(SeedComponent) is not None:
            features[1] = 1.0  # seed_on_tile

    # --- Look-ahead (3 tiles in facing direction) ---
    dx, dy = agent.direction
    for step in range(1, 4):
        lx = agent.x + dx * step
        ly = agent.y + dy * step
        if not (0 <= lx < world.width and 0 <= ly < world.height):
            break
        ahead_tile = world.tiles[ly][lx]
        for oid in ahead_tile.object_ids:
            o = world.objects.get(oid)
            if o is None or ObjectRegistry.is_terrain_layer(o):
                continue
            if o.get_component(EdibleComponent) is not None:
                features[2] = 1.0  # food_ahead
                features[3] = 1.0  # resource_ahead
            elif (o.get_component(SeedComponent) is not None or
                  o.get_component(PlantComponent) is not None):
                features[3] = 1.0  # resource_ahead

    # --- Nearest food scan (within vision radius 5) ---
    best_dist = float('inf')
    best_fx, best_fy = 0, 0
    scan_r = 5
    for sy in range(max(0, agent.y - scan_r), min(world.height, agent.y + scan_r + 1)):
        for sx in range(max(0, agent.x - scan_r), min(world.width, agent.x + scan_r + 1)):
            stile = world.tiles[sy][sx]
            for oid in stile.object_ids:
                o = world.objects.get(oid)
                if o is None:
                    continue
                if ObjectRegistry.is_terrain_layer(o):
                    continue
                if o.get_component(EdibleComponent) is not None:
                    d = abs(sx - agent.x) + abs(sy - agent.y)  # Manhattan
                    if d < best_dist:
                        best_dist = d
                        best_fx, best_fy = sx, sy

    if best_dist < float('inf'):
        # Proximity: 1.0 when on the food, ~0 at scan_r distance
        features[4] = max(0.0, 1.0 - best_dist / scan_r)

        # Direction match: dot product of facing direction and food direction
        diff_x = best_fx - agent.x
        diff_y = best_fy - agent.y
        mag = math.sqrt(diff_x * diff_x + diff_y * diff_y)
        if mag > 0:
            ndx, ndy = diff_x / mag, diff_y / mag
            dot = dx * ndx + dy * ndy  # range [-1, 1]
            features[5] = (dot + 1.0) / 2.0  # remap to [0, 1]

    # --- Energy urgency (non-linear: spikes when energy is very low) ---
    energy_ratio = agent.energy / agent.max_energy
    if energy_ratio < 0.25:
        features[6] = 1.0
    elif energy_ratio < 0.50:
        features[6] = 0.6
    elif energy_ratio < 0.75:
        features[6] = 0.2
    else:
        features[6] = 0.0

    # --- Can interact: any of PICK_UP / EAT / USE is possible ---
    has_inv_space = len(agent.inventory) < agent.inventory_size
    has_food_inv = any(
        world.objects.get(oid) is not None and
        world.objects.get(oid).get_component(EdibleComponent) is not None
        for oid in agent.inventory
    )
    has_seed_inv = any(
        world.objects.get(oid) is not None and
        world.objects.get(oid).get_component(SeedComponent) is not None
        for oid in agent.inventory
    )
    # Can pick up if food/seed on tile and inventory space
    can_pick = (features[0] > 0 or features[1] > 0) and has_inv_space
    # Can eat if has food in inventory
    can_eat = has_food_inv
    # Can use (plant) if has seed and on plantable soil
    can_use = has_seed_inv and tile.terrain_type in (TerrainType.SOIL, TerrainType.SAND)
    features[7] = 1.0 if (can_pick or can_eat or can_use) else 0.0

    return features


# ---------------------------------------------------------------------------
# 4. Inventory  (6 features)
# ---------------------------------------------------------------------------

def _encode_inventory(agent: 'Agent', world: 'World') -> list[float]:
    """
    Encode agent's inventory.

    Features (6):
      [0] Inventory fullness (0–1)
      [1] Has food (0/1)
      [2] Has seed (0/1)
      [3] Has fertilizer (0/1)
      [4] Total food calories (normalised)
      [5] Inventory count (normalised)
    """
    features: list[float] = []

    fullness = len(agent.inventory) / agent.inventory_size
    features.append(fullness)

    has_food = 0.0
    has_seed = 0.0
    has_fertilizer = 0.0
    total_calories = 0.0

    for obj_id in agent.inventory:
        obj = world.objects.get(obj_id)
        if obj is None:
            continue

        edible = obj.get_component(EdibleComponent)
        if edible is not None:
            has_food = 1.0
            total_calories += edible.calories * edible.freshness

        if obj.get_component(SeedComponent) is not None:
            has_seed = 1.0

        from world.objects import FertilizerComponent
        if obj.get_component(FertilizerComponent) is not None:
            has_fertilizer = 1.0

    features.append(has_food)
    features.append(has_seed)
    features.append(has_fertilizer)
    features.append(min(1.0, total_calories / 200.0))
    features.append(len(agent.inventory) / agent.inventory_size)

    return features
