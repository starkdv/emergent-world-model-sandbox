"""
Utility functions for agent action execution and validation.

Extracts action implementation logic from the Agent class
to keep the main agent file clean and focused on structure.

Author: Karan Vasa
Date: February 14, 2026
"""

import numpy as np
import random
from typing import TYPE_CHECKING

from agents.actions import Action, ActionResult

if TYPE_CHECKING:
    from agents.agent import Agent
    from world.world import World


def _get_object_type(obj) -> str:
    """Infer semantic object type label from components."""
    from world.objects import EdibleComponent, SeedComponent, PlantComponent, FertilizerComponent

    if obj is None:
        return ""
    if obj.has_component(EdibleComponent):
        return "food"
    if obj.has_component(SeedComponent):
        return "seed"
    if obj.has_component(FertilizerComponent):
        return "fertilizer"
    if obj.has_component(PlantComponent):
        return "plant"
    return "object"


def get_action_mask(agent: 'Agent', world: 'World') -> np.ndarray:
    """
    Create binary mask over actions (1 = valid, 0 = invalid).
    
    This prevents the agent from wasting probability mass on
    actions that cannot be executed in the current state.
    
    Args:
        agent: The agent to check
        world: The world the agent is in
        
    Returns:
        Binary mask array of shape (num_actions,)
    """
    from world.objects import EdibleComponent, SeedComponent, FertilizerComponent
    
    mask = np.ones(len(Action), dtype=np.float32)

    # MOVE_FORWARD: check bounds and passability
    nx = agent.x + agent.direction[0]
    ny = agent.y + agent.direction[1]
    if not world.is_valid_position(nx, ny):
        mask[Action.MOVE_FORWARD.value] = 0.0
    else:
        if not world.tiles[ny][nx].is_passable():
            mask[Action.MOVE_FORWARD.value] = 0.0

    # PICK_UP: check if tile has objects and inventory has space
    tile = world.tiles[agent.y][agent.x]
    if not tile.object_ids or len(agent.inventory) >= agent.inventory_size:
        mask[Action.PICK_UP.value] = 0.0

    # DROP: check if inventory has items
    if not agent.inventory:
        mask[Action.DROP.value] = 0.0

    # EAT: check if inventory has edible items
    has_food = False
    for obj_id in agent.inventory:
        obj = world.objects.get(obj_id)
        if obj and obj.has_component(EdibleComponent):
            has_food = True
            break
    if not has_food:
        mask[Action.EAT.value] = 0.0

    # USE: check if inventory has usable items (seed or fertilizer)
    can_use = False
    tile = world.tiles[agent.y][agent.x]
    tile_can_plant_here = tile.can_support_plant()

    # If stacking is off and current tile already has objects, planting here will fail
    if not world.allow_stacking and tile.object_ids:
        tile_can_plant_here = False

    for obj_id in agent.inventory:
        obj = world.objects.get(obj_id)
        if obj is None:
            continue

        # Seed use: require plantable tile
        if obj.get_component(SeedComponent) is not None:
            if tile_can_plant_here and (world.allow_stacking or not tile.object_ids):
                can_use = True
                break

            # Check nearby tiles if current tile is blocked
            found_spot = False
            if not world.allow_stacking and tile.object_ids:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = agent.x + dx, agent.y + dy
                        if 0 <= nx < world.width and 0 <= ny < world.height:
                            t2 = world.tiles[ny][nx]
                            if t2.can_support_plant() and (world.allow_stacking or not t2.object_ids):
                                found_spot = True
                                break
                    if found_spot:
                        break

            if found_spot:
                can_use = True
                break

        # Fertilizer use: allow if tile can be improved
        if obj.get_component(FertilizerComponent) is not None:
            if tile.can_support_plant() and tile.fertility < 0.85:
                can_use = True
                break

    if not can_use:
        mask[Action.USE.value] = 0.0

    # TURN_LEFT, TURN_RIGHT, WAIT always valid
    return mask


def execute_move_forward(agent: 'Agent', world: 'World') -> ActionResult:
    """Move one tile in current direction."""
    new_x = agent.x + agent.direction[0]
    new_y = agent.y + agent.direction[1]
    
    # Check bounds
    if not (0 <= new_x < world.width and 0 <= new_y < world.height):
        return ActionResult(False, 0.22, "Out of bounds")
    
    # Check if tile is passable
    tile = world.tiles[new_y][new_x]
    if not tile.is_passable():
        return ActionResult(False, 0.22, "Tile blocked")
    
    # Move agent
    agent.x = new_x
    agent.y = new_y
    return ActionResult(True, 0.20, "Moved forward")


def execute_turn_left(agent: 'Agent') -> ActionResult:
    """Rotate direction 90° counter-clockwise."""
    dx, dy = agent.direction
    agent.direction = (dy, -dx)  # Rotate left
    return ActionResult(True, 0.24, "Turned left")


def execute_turn_right(agent: 'Agent') -> ActionResult:
    """Rotate direction 90° clockwise."""
    dx, dy = agent.direction
    agent.direction = (-dy, dx)  # Rotate right
    return ActionResult(True, 0.24, "Turned right")


def execute_pick_up(agent: 'Agent', world: 'World') -> ActionResult:
    """Pick up object from current tile, prioritizing food."""
    from world.objects import EdibleComponent, SeedComponent
    
    if len(agent.inventory) >= agent.inventory_size:
        return ActionResult(False, 0.1, "Inventory full")
    
    tile = world.tiles[agent.y][agent.x]
    if not tile.object_ids:
        return ActionResult(False, 0.1, "No objects here")
    
    # Prioritize edible items first
    obj_id_to_pick = None
    for obj_id in tile.object_ids:
        obj = world.objects.get(obj_id)
        if obj and obj.has_component(EdibleComponent):
            obj_id_to_pick = obj_id
            break
    
    # If no food, pick up seeds (useful for planting)
    if obj_id_to_pick is None:
        for obj_id in tile.object_ids:
            obj = world.objects.get(obj_id)
            if obj and obj.has_component(SeedComponent):
                obj_id_to_pick = obj_id
                break
    
    # If still nothing, pick first object
    if obj_id_to_pick is None:
        obj_id_to_pick = tile.object_ids[0]
    
    obj = world.objects.get(obj_id_to_pick)
    if obj is None:
        return ActionResult(False, 0.1, "Object not found")
    
    # Add to inventory and remove from world tile
    agent.inventory.append(obj_id_to_pick)
    tile.object_ids.remove(obj_id_to_pick)
    
    obj_type = _get_object_type(obj)
    return ActionResult(
        True,
        0.2,
        f"Picked up {obj_type} {obj_id_to_pick}",
        object_id=obj_id_to_pick,
        object_type=obj_type,
        target_x=agent.x,
        target_y=agent.y,
        interaction_kind="pickup"
    )


def execute_drop(agent: 'Agent', world: 'World') -> ActionResult:
    """Drop held object onto current tile or nearby if occupied."""
    if not agent.inventory:
        return ActionResult(False, 0.05, "Inventory empty")
    
    # Drop last object
    obj_id = agent.inventory.pop()
    obj = world.objects.get(obj_id)
    
    if obj is None:
        return ActionResult(False, 0.05, "Object not found")
    
    # Check stacking configuration
    tile = world.tiles[agent.y][agent.x]
    
    obj_type = _get_object_type(obj)

    if world.allow_stacking or not tile.object_ids:
        # Stacking allowed OR tile is empty - drop here
        tile.object_ids.append(obj_id)
        obj.x = agent.x
        obj.y = agent.y
        return ActionResult(
            True,
            0.1,
            f"Dropped {obj_type} {obj_id}",
            object_id=obj_id,
            object_type=obj_type,
            target_x=agent.x,
            target_y=agent.y,
            interaction_kind="drop_here"
        )
    
    # Stacking disabled and tile occupied - try nearby tiles
    nearby_positions = [
        (agent.x + dx, agent.y + dy)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        if (dx != 0 or dy != 0)
    ]
    
    for nx, ny in nearby_positions:
        if world.is_valid_position(nx, ny):
            nearby_tile = world.get_tile(nx, ny)
            if nearby_tile and not nearby_tile.object_ids:
                # Found empty spot
                nearby_tile.object_ids.append(obj_id)
                obj.x = nx
                obj.y = ny
                return ActionResult(
                    True,
                    0.1,
                    f"Dropped {obj_type} {obj_id} nearby at ({nx}, {ny})",
                    object_id=obj_id,
                    object_type=obj_type,
                    target_x=nx,
                    target_y=ny,
                    interaction_kind="drop_nearby"
                )
    
    # No empty spots - put back in inventory
    agent.inventory.append(obj_id)
    return ActionResult(False, 0.05, "No space to drop (tile occupied)")


def execute_eat(agent: 'Agent', world: 'World') -> ActionResult:
    """Consume edible object from inventory."""
    from world.objects import EdibleComponent
    
    if not agent.inventory:
        return ActionResult(False, 0.05, "Nothing to eat")
    
    # Find edible item
    for obj_id in agent.inventory:
        obj = world.objects.get(obj_id)
        if obj is None:
            continue
        
        edible = obj.get_component(EdibleComponent)
        if edible is not None:
            # Consume the food
            energy_gained = edible.calories * edible.freshness
            agent.energy = min(agent.max_energy, agent.energy + energy_gained)
            
            # Remove from inventory and world
            agent.inventory.remove(obj_id)
            world.remove_object(obj_id)
            
            # Fitness reward for eating
            agent.fitness += energy_gained * 0.1
            
            return ActionResult(
                True,
                0.1,
                f"Ate food {obj_id}, gained {energy_gained:.1f} energy",
                object_id=obj_id,
                object_type="food",
                target_x=agent.x,
                target_y=agent.y,
                interaction_kind="eat"
            )
    
    return ActionResult(False, 0.05, "No edible items")


def execute_use(agent: 'Agent', world: 'World') -> ActionResult:
    """Use/plant object (e.g., plant seed, apply fertilizer)."""
    from world.objects import SeedComponent, FertilizerComponent
    
    if not agent.inventory:
        return ActionResult(False, 0.05, "Nothing to use")
    
    tile = world.tiles[agent.y][agent.x]
    
    # Pick the first usable item (seed or fertilizer) from inventory
    obj_id = None
    obj = None

    for cand_id in list(agent.inventory):
        cand = world.objects.get(cand_id)
        if cand is None:
            continue
        if (cand.get_component(SeedComponent) is not None or
            cand.get_component(FertilizerComponent) is not None):
            obj_id = cand_id
            obj = cand
            break

    if obj_id is None or obj is None:
        return ActionResult(False, 0.1, "No usable item")
    
    # Check if it's a seed
    seed = obj.get_component(SeedComponent)
    if seed is not None:
        # Plant the seed
        if tile.can_support_plant():
            # Check stacking configuration
            if world.allow_stacking or not tile.object_ids:
                # Stacking allowed OR tile is empty - plant here
                agent.inventory.remove(obj_id)
                tile.object_ids.append(obj_id)
                obj.x = agent.x
                obj.y = agent.y
                agent.fitness += 1.0
                return ActionResult(
                    True,
                    0.5,
                    f"Planted seed {obj_id}",
                    object_id=obj_id,
                    object_type="seed",
                    target_x=agent.x,
                    target_y=agent.y,
                    interaction_kind="plant_seed"
                )
            else:
                # Stacking disabled and tile occupied - try nearby tiles
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                            (-1, -1), (-1, 1), (1, -1), (1, 1)]
                nearby_positions = [
                    (agent.x + dx, agent.y + dy) 
                    for dx, dy in directions
                ]
                random.shuffle(nearby_positions)
                
                for nx, ny in nearby_positions:
                    if 0 <= nx < world.width and 0 <= ny < world.height:
                        nearby_tile = world.tiles[ny][nx]
                        if nearby_tile.can_support_plant() and not nearby_tile.object_ids:
                            # Found empty plantable spot
                            agent.inventory.remove(obj_id)
                            nearby_tile.object_ids.append(obj_id)
                            obj.x = nx
                            obj.y = ny
                            agent.fitness += 1.0
                            return ActionResult(
                                True,
                                0.5,
                                f"Planted seed {obj_id} nearby at ({nx}, {ny})",
                                object_id=obj_id,
                                object_type="seed",
                                target_x=nx,
                                target_y=ny,
                                interaction_kind="plant_seed_nearby"
                            )
                
                # No empty tiles nearby - keep in inventory
                return ActionResult(False, 0.1, "Cannot plant - tile occupied and no space nearby")
        else:
            return ActionResult(False, 0.1, "Cannot plant here")
    
    # Check if it's fertilizer
    fertilizer = obj.get_component(FertilizerComponent)
    if fertilizer is not None:
        # Apply fertilizer to tile
        tile.fertility = min(1.0, tile.fertility + fertilizer.fertility_boost)
        
        # Remove from inventory and world
        agent.inventory.remove(obj_id)
        world.remove_object(obj_id)
        
        return ActionResult(
            True,
            0.5,
            f"Applied fertilizer {obj_id}",
            object_id=obj_id,
            object_type="fertilizer",
            target_x=agent.x,
            target_y=agent.y,
            interaction_kind="apply_fertilizer"
        )
    
    return ActionResult(False, 0.1, "Cannot use this object")


def execute_wait(agent: 'Agent') -> ActionResult:
    """Do nothing (conserve energy)."""
    return ActionResult(True, 0.18, "Waiting")
