"""
Action space definitions for agents.

This module defines the primitive actions available to agents.
No high-level actions (like "FARM" or "BUILD") are allowed - 
all complex behaviors must emerge from these primitives.

Author: Karan Vasa
Date: November 14, 2025
"""

from enum import IntEnum
from typing import NamedTuple


class Action(IntEnum):
    """Primitive actions available to agents."""
    
    # Movement actions
    MOVE_FORWARD = 0    # Move one tile in current direction
    TURN_LEFT = 1       # Rotate direction 90° counter-clockwise
    TURN_RIGHT = 2      # Rotate direction 90° clockwise
    
    # Interaction actions
    PICK_UP = 3         # Pick up object from current tile (into inventory)
    DROP = 4            # Drop held object onto current tile
    EAT = 5             # Consume edible object from inventory
    USE = 6             # Use/plant object (e.g., plant seed, apply fertilizer)
    
    # Passive action
    WAIT = 7            # Do nothing (save energy)


# Direction vectors (for movement and orientation)
DIRECTIONS = {
    (0, -1): "NORTH",   # up
    (1, 0): "EAST",     # right
    (0, 1): "SOUTH",    # down
    (-1, 0): "WEST",    # left
}

DIRECTION_NAMES = {v: k for k, v in DIRECTIONS.items()}


class ActionResult(NamedTuple):
    """Result of an action execution."""
    success: bool
    energy_cost: float
    message: str = ""
    object_id: int = -1
    object_type: str = ""
    target_x: int = -1
    target_y: int = -1
    interaction_kind: str = ""
