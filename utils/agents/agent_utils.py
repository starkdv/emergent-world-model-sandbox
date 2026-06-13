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
    """Infer semantic object type label from registry or components."""
    from world.object_registry import ObjectRegistry

    return ObjectRegistry.get_category(obj)


def _tile_contact_damage(world: "World", x: int, y: int) -> float:
    """Total contact damage from hazard objects on a tile (W3, e.g. thorns)."""
    from world.object_registry import ObjectRegistry

    tile = world.tiles[y][x]
    if not tile.object_ids:
        return 0.0
    total = 0.0
    for obj_id in tile.object_ids:
        obj = world.objects.get(obj_id)
        if obj is None:
            continue
        defn = ObjectRegistry.get(getattr(obj, "type_id", ""))
        if defn is not None and defn.tile_effect is not None:
            total += max(0.0, defn.tile_effect.contact_damage)
    return total


def get_action_mask(agent: "Agent", world: "World") -> np.ndarray:
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
    from world.object_registry import ObjectRegistry

    mask = np.ones(len(Action), dtype=np.float32)

    # MOVE_FORWARD: check bounds and passability
    nx = agent.x + agent.direction[0]
    ny = agent.y + agent.direction[1]
    if not world.is_valid_position(nx, ny):
        mask[Action.MOVE_FORWARD.value] = 0.0
    else:
        if not world.tiles[ny][nx].is_passable():
            mask[Action.MOVE_FORWARD.value] = 0.0

    # PICK_UP: check if tile has *pickable* objects and inventory has space
    tile = world.tiles[agent.y][agent.x]
    has_pickable = False
    if tile.object_ids and len(agent.inventory) < agent.inventory_size:
        for obj_id in tile.object_ids:
            obj = world.objects.get(obj_id)
            if obj and ObjectRegistry.is_pickable(obj):
                has_pickable = True
                break
    if not has_pickable:
        mask[Action.PICK_UP.value] = 0.0

    # DROP: valid only if we have something to drop AND there is space.
    # This mirrors execute_drop: in no-stacking mode, dropping onto an
    # occupied tile requires an empty neighboring tile.
    if not agent.inventory:
        mask[Action.DROP.value] = 0.0
    else:
        if world.allow_stacking:
            mask[Action.DROP.value] = 1.0
        else:
            if not tile.object_ids:
                mask[Action.DROP.value] = 1.0
            else:
                has_empty_neighbor = False
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        tx = agent.x + dx
                        ty = agent.y + dy
                        if 0 <= tx < world.width and 0 <= ty < world.height:
                            if not world.tiles[ty][tx].object_ids:
                                has_empty_neighbor = True
                                break
                    if has_empty_neighbor:
                        break

                if not has_empty_neighbor:
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

    # Helper: count non-terrain-layer objects on a tile
    def _has_real_objects(t):
        for oid in t.object_ids:
            o = world.objects.get(oid)
            if o and not getattr(o, "is_terrain", False):
                return True
        return False

    # USE: check if inventory has usable items (seed or fertilizer)
    can_use = False
    tile_can_plant_here = tile.can_support_plant()

    # If stacking is off and current tile has real objects, planting here fails
    if not world.allow_stacking and _has_real_objects(tile):
        tile_can_plant_here = False

    for obj_id in agent.inventory:
        obj = world.objects.get(obj_id)
        if obj is None:
            continue

        # Seed use: require plantable tile
        if obj.get_component(SeedComponent) is not None:
            if tile_can_plant_here and (
                world.allow_stacking or not _has_real_objects(tile)
            ):
                can_use = True
                break

            # Check nearby tiles if current tile is blocked
            found_spot = False
            if not world.allow_stacking and _has_real_objects(tile):
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = agent.x + dx, agent.y + dy
                        if 0 <= nx < world.width and 0 <= ny < world.height:
                            t2 = world.tiles[ny][nx]
                            if t2.can_support_plant() and (
                                world.allow_stacking or not _has_real_objects(t2)
                            ):
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


# Energy added per unit of uphill elevation gain when moving (W2). With the
# legacy flat generator every elevation is 0.0, so this is always 0 and
# movement cost is unchanged (bit-compatible). Downhill is never cheaper than
# the base cost — gravity helps, but you still spend the step.
SLOPE_CLIMB_COST = 0.6


def execute_move_forward(agent: "Agent", world: "World") -> ActionResult:
    """Move one tile in current direction."""
    new_x = agent.x + agent.direction[0]
    new_y = agent.y + agent.direction[1]

    # Check bounds
    if not (0 <= new_x < world.width and 0 <= new_y < world.height):
        return ActionResult(False, 0.22, "Out of bounds")

    # Check if tile is passable
    dest = world.tiles[new_y][new_x]
    if not dest.is_passable():
        return ActionResult(False, 0.22, "Tile blocked")

    # Tile exclusivity (W4 sub-step b): when agent collision is on, another
    # living agent blocks the tile, so space itself becomes contested. Off by
    # default → agents may overlap exactly as before.
    if getattr(world, "agent_collision", False):
        for other in world.agents.values():
            if (
                other is not agent
                and getattr(other, "alive", True)
                and other.x == new_x
                and other.y == new_y
            ):
                return ActionResult(False, 0.22, "Tile occupied")

    # Slope cost (W2): climbing uphill costs extra energy proportional to the
    # elevation gained. Flat terrain (legacy) → no change.
    src = world.tiles[agent.y][agent.x]
    climb = dest.elevation - src.elevation
    energy_cost = 0.20
    if climb > 0.0:
        energy_cost += SLOPE_CLIMB_COST * climb

    # Move agent
    agent.x = new_x
    agent.y = new_y

    # Contact hazard (W3): stepping onto a tile with a damaging object (e.g.
    # thorns) costs extra energy. Folded into the move's energy_cost so it
    # flows through the normal energy/reward path. Harmless tiles → 0.
    damage = _tile_contact_damage(world, new_x, new_y)
    if damage > 0.0:
        energy_cost += damage
        return ActionResult(
            True,
            round(energy_cost, 3),
            f"Moved onto hazard (−{damage:.1f} energy)",
        )

    return ActionResult(True, round(energy_cost, 3), "Moved forward")


def execute_turn_left(agent: "Agent") -> ActionResult:
    """Rotate direction 90° counter-clockwise."""
    dx, dy = agent.direction
    agent.direction = (dy, -dx)  # Rotate left
    return ActionResult(True, 0.24, "Turned left")


def execute_turn_right(agent: "Agent") -> ActionResult:
    """Rotate direction 90° clockwise."""
    dx, dy = agent.direction
    agent.direction = (-dy, dx)  # Rotate right
    return ActionResult(True, 0.24, "Turned right")


def execute_pick_up(agent: "Agent", world: "World") -> ActionResult:
    """Pick up object from current tile, prioritizing food. Respects pickable flag."""
    from world.objects import EdibleComponent, SeedComponent
    from world.object_registry import ObjectRegistry

    if len(agent.inventory) >= agent.inventory_size:
        return ActionResult(False, 0.1, "Inventory full")

    tile = world.tiles[agent.y][agent.x]
    if not tile.object_ids:
        return ActionResult(False, 0.1, "No objects here")

    # Prioritize edible items first (must be pickable)
    obj_id_to_pick = None
    for obj_id in tile.object_ids:
        obj = world.objects.get(obj_id)
        if (
            obj
            and ObjectRegistry.is_pickable(obj)
            and obj.has_component(EdibleComponent)
        ):
            obj_id_to_pick = obj_id
            break

    # If no food, pick up seeds (must be pickable)
    if obj_id_to_pick is None:
        for obj_id in tile.object_ids:
            obj = world.objects.get(obj_id)
            if (
                obj
                and ObjectRegistry.is_pickable(obj)
                and obj.has_component(SeedComponent)
            ):
                obj_id_to_pick = obj_id
                break

    # If still nothing, pick first pickable object
    if obj_id_to_pick is None:
        for obj_id in tile.object_ids:
            obj = world.objects.get(obj_id)
            if obj and ObjectRegistry.is_pickable(obj):
                obj_id_to_pick = obj_id
                break

    if obj_id_to_pick is None:
        return ActionResult(False, 0.1, "No pickable objects here")

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
        interaction_kind="pickup",
    )


def execute_drop(agent: "Agent", world: "World") -> ActionResult:
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
        tile.object_ids.add(obj_id)
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
            interaction_kind="drop_here",
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
                nearby_tile.object_ids.add(obj_id)
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
                    interaction_kind="drop_nearby",
                )

    # No empty spots - put back in inventory
    agent.inventory.append(obj_id)
    return ActionResult(False, 0.05, "No space to drop (tile occupied)")


# Energy lost when eating, per unit of (toxicity × freshness). Tuned so a
# fully-toxic, fully-fresh food (toxicity 1.0) inflicts a large loss — enough
# that a low-calorie poison is net-negative and a discrimination pressure
# emerges (W3). Non-toxic food (toxicity 0.0) is unaffected → bit-compatible.
TOXICITY_DAMAGE = 30.0


def execute_eat(agent: "Agent", world: "World") -> ActionResult:
    """Consume edible object from inventory.

    Net energy = calories × freshness − toxicity × freshness × TOXICITY_DAMAGE
    (W3): the dormant toxicity field is now a real physical consequence, so
    poisonous food can cost more energy than it gives. Nothing labels a food
    "good" or "bad" — the agent must discover it through the energy/survival
    signal (guideline §8). ``object_type`` records the eaten *species* so the
    analyzer can break consumption down per species.
    """
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
            # Net energy: calories minus a toxicity penalty (W3)
            gain = edible.calories * edible.freshness
            toxic_loss = edible.toxicity * edible.freshness * TOXICITY_DAMAGE
            energy_delta = gain - toxic_loss
            # Cap at max energy; net loss can drive energy below 0 (the death
            # check handles fatal poisoning on the next tick).
            agent.energy = min(agent.max_energy, agent.energy + energy_delta)

            # Remove from inventory and world
            agent.inventory.remove(obj_id)
            world.remove_object(obj_id)

            # Fitness tracks the realised energy outcome (poison hurts)
            agent.fitness += energy_delta * 0.1

            species = getattr(obj, "type_id", "") or "food"
            if toxic_loss > 0:
                msg = (
                    f"Ate {species} {obj_id}: +{gain:.1f} −{toxic_loss:.1f} "
                    f"toxic = {energy_delta:+.1f} energy"
                )
            else:
                msg = f"Ate {species} {obj_id}, gained {energy_delta:.1f} energy"

            return ActionResult(
                True,
                0.1,
                msg,
                object_id=obj_id,
                object_type=species,
                target_x=agent.x,
                target_y=agent.y,
                interaction_kind="eat",
            )

    return ActionResult(False, 0.05, "No edible items")


def execute_use(agent: "Agent", world: "World") -> ActionResult:
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
        if (
            cand.get_component(SeedComponent) is not None
            or cand.get_component(FertilizerComponent) is not None
        ):
            obj_id = cand_id
            obj = cand
            break

    if obj_id is None or obj is None:
        return ActionResult(False, 0.1, "No usable item")

    # Check if it's a seed
    seed = obj.get_component(SeedComponent)
    if seed is not None:

        def _has_real_objects(t):
            for oid in t.object_ids:
                o = world.objects.get(oid)
                if o and not getattr(o, "is_terrain", False):
                    return True
            return False

        # Plant the seed
        if tile.can_support_plant():
            # Check stacking configuration
            if world.allow_stacking or not _has_real_objects(tile):
                # Stacking allowed OR tile is empty - plant here
                agent.inventory.remove(obj_id)
                tile.object_ids.add(obj_id)
                obj.x = agent.x
                obj.y = agent.y
                obj.planted_by_agent = True  # Tag for golden rendering
                agent.fitness += 1.0
                return ActionResult(
                    True,
                    0.5,
                    f"Planted seed {obj_id}",
                    object_id=obj_id,
                    object_type="seed",
                    target_x=agent.x,
                    target_y=agent.y,
                    interaction_kind="plant_seed",
                )
            else:
                # Stacking disabled and tile occupied - try nearby tiles
                directions = [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ]
                nearby_positions = [
                    (agent.x + dx, agent.y + dy) for dx, dy in directions
                ]
                random.shuffle(nearby_positions)

                for nx, ny in nearby_positions:
                    if 0 <= nx < world.width and 0 <= ny < world.height:
                        nearby_tile = world.tiles[ny][nx]
                        if nearby_tile.can_support_plant() and not _has_real_objects(
                            nearby_tile
                        ):
                            # Found empty plantable spot
                            agent.inventory.remove(obj_id)
                            nearby_tile.object_ids.add(obj_id)
                            obj.x = nx
                            obj.y = ny
                            obj.planted_by_agent = True  # Tag for golden rendering
                            agent.fitness += 1.0
                            return ActionResult(
                                True,
                                0.5,
                                f"Planted seed {obj_id} nearby at ({nx}, {ny})",
                                object_id=obj_id,
                                object_type="seed",
                                target_x=nx,
                                target_y=ny,
                                interaction_kind="plant_seed_nearby",
                            )

                # No empty tiles nearby - keep in inventory
                return ActionResult(
                    False, 0.1, "Cannot plant - tile occupied and no space nearby"
                )
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
            interaction_kind="apply_fertilizer",
        )

    return ActionResult(False, 0.1, "Cannot use this object")


def execute_wait(agent: "Agent") -> ActionResult:
    """Do nothing (conserve energy)."""
    return ActionResult(True, 0.18, "Waiting")
