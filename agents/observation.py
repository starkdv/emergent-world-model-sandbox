"""
Observation vector construction for agent perception.

Builds a normalized feature vector representing what the agent
can perceive about its environment. The observation includes:
- Local terrain information
- Nearby objects (plants, food, seeds)
- Agent's internal state (energy, inventory)
- Directional information

Author: Karan Vasa
Date: November 14, 2025
"""

import numpy as np
from typing import TYPE_CHECKING

from world.tiles import TerrainType
from world.objects import EdibleComponent, PlantComponent, SeedComponent

if TYPE_CHECKING:
    from agents.agent import Agent
    from world.world import World


def build_observation(agent: 'Agent', world: 'World') -> np.ndarray:
    """
    Build observation vector for agent.
    
    The observation vector contains:
    - Agent internal state (8 features)
    - Vision grid (5x5 = 25 tiles × 2 features = 50 features)
    - Inventory summary (6 features)
    
    Total: 64 features (normalized to [-1, 1] or [0, 1])
    
    Args:
        agent: The observing agent
        world: The world being observed
        
    Returns:
        Normalized observation vector (64 dimensions)
    """
    obs = []
    
    # 1. Agent internal state (8 features)
    obs.extend(_encode_agent_state(agent))
    
    # 2. Vision grid (50 features)
    obs.extend(_encode_vision(agent, world))
    
    # 3. Inventory state (6 features)
    obs.extend(_encode_inventory(agent, world))
    
    return np.array(obs, dtype=np.float32)


def _encode_agent_state(agent: 'Agent') -> list[float]:
    """
    Encode agent's internal state.
    
    Features (8 total):
    - Energy ratio (0-1)
    - Age ratio (0-1)
    - Direction (4 one-hot: N, E, S, W)
    - Has inventory space (0 or 1)
    - Metabolism rate (normalized)
    
    Args:
        agent: The agent to encode
        
    Returns:
        List of 8 features
    """
    features = []
    
    # Energy (0-1)
    features.append(agent.energy / agent.max_energy)
    
    # Age (0-1)
    features.append(agent.age / agent.max_age)
    
    # Direction (one-hot)
    direction_map = {
        (0, -1): [1, 0, 0, 0],  # North
        (1, 0): [0, 1, 0, 0],   # East
        (0, 1): [0, 0, 1, 0],   # South
        (-1, 0): [0, 0, 0, 1],  # West
    }
    features.extend(direction_map.get(agent.direction, [0, 0, 0, 0]))
    
    # Has inventory space (0 or 1)
    features.append(1.0 if len(agent.inventory) < agent.inventory_size else 0.0)
    
    # Metabolism rate (normalized to ~0-1)
    features.append(min(1.0, agent.metabolism_rate / 2.0))
    
    return features


def _encode_vision(agent: 'Agent', world: 'World') -> list[float]:
    """
    Encode 5x5 vision grid around agent.
    
    For each tile (25 tiles):
    - Terrain/object type (1 feature: combined encoding)
    - Resource value (1 feature: food calories or plant age)
    
    Total: 50 features
    
    Args:
        agent: The observing agent
        world: The world being observed
        
    Returns:
        List of 50 features
    """
    features = []
    vision_radius = 2  # 5x5 grid (2 tiles in each direction)
    
    for dy in range(-vision_radius, vision_radius + 1):
        for dx in range(-vision_radius, vision_radius + 1):
            # Calculate world position
            wx = agent.x + dx
            wy = agent.y + dy
            
            # Check bounds
            if not (0 <= wx < world.width and 0 <= wy < world.height):
                # Out of bounds: encode as rock
                features.append(0.0)  # Rock
                features.append(0.0)  # No value
                continue
            
            tile = world.tiles[wy][wx]
            
            # Encode terrain/object type (0-1)
            tile_encoding, tile_value = _encode_tile(tile, world)
            features.append(tile_encoding)
            features.append(tile_value)
    
    return features


def _encode_tile(tile, world: 'World') -> tuple[float, float]:
    """
    Encode a single tile.
    
    Returns:
        Tuple of (type_encoding, value_encoding)
        - type_encoding: 0=rock, 0.25=water, 0.5=empty_soil, 0.75=plant, 1.0=food
        - value_encoding: fertility/moisture or food/plant value (0-1)
    """
    # Check for objects on tile
    if tile.object_ids:
        obj_id = tile.object_ids[0]
        obj = world.objects.get(obj_id)
        
        if obj is not None:
            # Check if it's food
            edible = obj.get_component(EdibleComponent)
            if edible is not None:
                type_encoding = 1.0  # Food
                # Value is calories (normalized to 0-1, assuming max ~50 calories)
                value_encoding = min(1.0, edible.calories * edible.freshness / 50.0)
                return type_encoding, value_encoding
            
            # Check if it's a plant
            plant = obj.get_component(PlantComponent)
            if plant is not None:
                type_encoding = 0.75  # Plant
                # Value is maturity (0-1)
                value_encoding = min(1.0, plant.age / plant.mature_age)
                return type_encoding, value_encoding
            
            # Check if it's a seed
            seed = obj.get_component(SeedComponent)
            if seed is not None:
                type_encoding = 0.6  # Seed
                # Value is age/viability
                value_encoding = 1.0 - min(1.0, seed.time_in_soil / seed.max_age)
                return type_encoding, value_encoding
    
    # No objects, encode terrain
    if tile.terrain_type == TerrainType.ROCK:
        return 0.0, 0.0
    elif tile.terrain_type == TerrainType.WATER:
        return 0.25, tile.moisture
    elif tile.terrain_type == TerrainType.SOIL:
        return 0.5, (tile.fertility + tile.moisture) / 2.0
    else:
        return 0.0, 0.0


def _encode_inventory(agent: 'Agent', world: 'World') -> list[float]:
    """
    Encode agent's inventory.
    
    Features (6 total):
    - Inventory fullness (0-1)
    - Has food (0 or 1)
    - Has seed (0 or 1)
    - Has fertilizer (0 or 1)
    - Total food calories (normalized)
    - Inventory count (normalized)
    
    Args:
        agent: The agent to encode
        world: The world
        
    Returns:
        List of 6 features
    """
    features = []
    
    # Inventory fullness
    fullness = len(agent.inventory) / agent.inventory_size
    features.append(fullness)
    
    # Check what's in inventory
    has_food = 0.0
    has_seed = 0.0
    has_fertilizer = 0.0
    total_calories = 0.0
    
    for obj_id in agent.inventory:
        obj = world.objects.get(obj_id)
        if obj is None:
            continue
        
        if obj.get_component(EdibleComponent) is not None:
            has_food = 1.0
            edible = obj.get_component(EdibleComponent)
            total_calories += edible.calories * edible.freshness
        
        if obj.get_component(SeedComponent) is not None:
            has_seed = 1.0
        
        from world.objects import FertilizerComponent
        if obj.get_component(FertilizerComponent) is not None:
            has_fertilizer = 1.0
    
    features.append(has_food)
    features.append(has_seed)
    features.append(has_fertilizer)
    
    # Total calories (normalized, assuming max ~200)
    features.append(min(1.0, total_calories / 200.0))
    
    # Inventory count (normalized)
    features.append(len(agent.inventory) / agent.inventory_size)
    
    return features


def get_observation_size() -> int:
    """
    Get the size of the observation vector.
    
    Returns:
        Observation vector size (64)
    """
    return 64
