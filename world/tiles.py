"""
Tile representation and terrain types for the world grid.

This module defines the Tile class and TerrainType enum for representing
the world's grid cells with their properties.

Author: Karan Vasa
"""

from enum import Enum
from typing import Set


class TerrainType(Enum):
    """
    Enumeration of terrain types in the world.
    
    Attributes:
        SOIL: Plantable terrain with variable fertility
        ROCK: Impassable, non-plantable terrain
        WATER: Water terrain, affects moisture of nearby tiles
        SAND: Low-fertility terrain, passable but poor for growth
    """
    SOIL = "soil"
    ROCK = "rock"
    WATER = "water"
    SAND = "sand"


class Tile:
    """
    Represents a single tile in the world grid.
    
    Each tile has terrain properties, environmental conditions,
    and can contain multiple world objects.
    
    Attributes:
        x: X-coordinate of the tile in the grid
        y: Y-coordinate of the tile in the grid
        terrain_type: Type of terrain (SOIL, ROCK, WATER)
        fertility: Soil fertility level (0.0-1.0), affects plant growth
        moisture: Moisture level (0.0-1.0), affects plant growth
        object_ids: List of IDs of WorldObjects on this tile
    
    Author: Karan Vasa
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        terrain_type: TerrainType,
        fertility: float = 0.5,
        moisture: float = 0.5
    ):
        """
        Initialize a tile with position and properties.
        
        Args:
            x: X-coordinate in the grid
            y: Y-coordinate in the grid
            terrain_type: Type of terrain for this tile
            fertility: Initial fertility level (0.0-1.0)
            moisture: Initial moisture level (0.0-1.0)
            
        Raises:
            ValueError: If fertility or moisture is outside [0.0, 1.0] range
        """
        if not 0.0 <= fertility <= 1.0:
            raise ValueError(f"Fertility must be between 0.0 and 1.0, got {fertility}")
        if not 0.0 <= moisture <= 1.0:
            raise ValueError(f"Moisture must be between 0.0 and 1.0, got {moisture}")
        
        self.x = x
        self.y = y
        self.terrain_type = terrain_type
        self.fertility = fertility
        self.moisture = moisture
        self.object_ids: Set[int] = set()
    
    def add_object(self, object_id: int) -> None:
        """
        Add a world object ID to this tile.
        
        Args:
            object_id: ID of the WorldObject to add
        """
        self.object_ids.add(object_id)
    
    def remove_object(self, object_id: int) -> bool:
        """
        Remove a world object ID from this tile.
        
        Args:
            object_id: ID of the WorldObject to remove
            
        Returns:
            True if object was removed, False if not found
        """
        if object_id in self.object_ids:
            self.object_ids.discard(object_id)
            return True
        return False
    
    def is_passable(self) -> bool:
        """
        Check if agents can move through this tile.
        
        Returns:
            True if tile is passable, False otherwise
        """
        return self.terrain_type not in (TerrainType.ROCK,)
    
    def is_plantable(self) -> bool:
        """
        Check if seeds can be planted on this tile.
        
        Sand terrain *technically* allows planting, but with heavy
        penalties applied by the TileEffectSpec system.
        
        Returns:
            True if tile can support plants, False otherwise
        """
        return self.terrain_type in (TerrainType.SOIL, TerrainType.SAND)
    
    def can_support_plant(self) -> bool:
        """
        Check if this tile can support plant growth.
        
        Alias for is_plantable() for backward compatibility.
        
        Returns:
            True if tile can support plants, False otherwise
        """
        return self.is_plantable()
    
    def __repr__(self) -> str:
        """String representation of the tile."""
        return (f"Tile(x={self.x}, y={self.y}, terrain={self.terrain_type.value}, "
                f"fertility={self.fertility:.2f}, moisture={self.moisture:.2f})")
