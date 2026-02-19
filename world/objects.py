"""
Component system for world objects.

This module implements the Entity-Component-System (ECS) pattern for
world objects. Objects are entities with attached components that define
their properties and behaviors.

Author: Karan Vasa
"""

from typing import Dict, Any, Optional


class Component:
    """
    Base class for all components.
    
    Components are pure data structures that define properties and behaviors
    of world objects.
    
    Author: Karan Vasa
    """
    pass


class EdibleComponent(Component):
    """
    Component for objects that can be eaten by agents.
    
    Attributes:
        calories: Energy provided when consumed
        toxicity: Toxicity level (0.0 = safe, >0 may cause harm)
        freshness: Current freshness level (0.0-1.0), decays over time
        max_freshness: Maximum freshness value
    
    Author: Karan Vasa
    """
    
    def __init__(self, calories: float, toxicity: float = 0.0, freshness: float = 1.0):
        """
        Initialize an EdibleComponent.
        
        Args:
            calories: Energy value when consumed
            toxicity: Toxicity level (0.0 = safe)
            freshness: Initial freshness (0.0-1.0)
            
        Raises:
            ValueError: If freshness is outside [0.0, 1.0] range
        """
        if not 0.0 <= freshness <= 1.0:
            raise ValueError(f"Freshness must be between 0.0 and 1.0, got {freshness}")
        
        self.calories = calories
        self.toxicity = toxicity
        self.freshness = freshness
        self.max_freshness = freshness


class SeedComponent(Component):
    """
    Component for seeds that can be planted to grow plants.
    
    Attributes:
        plant_type: Key to identify which plant type this grows into
        grow_time: Ticks required for germination
        time_in_soil: Counter for ticks spent in suitable soil
        required_fertility: Minimum fertility needed to germinate
        required_moisture: Minimum moisture needed to germinate
        max_age: Maximum age before seed rots and disappears
    
    Author: Karan Vasa
    """
    
    def __init__(
        self,
        plant_type: str,
        grow_time: int = 50,
        required_fertility: float = 0.3,
        required_moisture: float = 0.2,
        max_age: int = 200
    ):
        """
        Initialize a SeedComponent.
        
        Args:
            plant_type: Type of plant this seed grows into
            grow_time: Ticks needed for germination
            required_fertility: Minimum fertility requirement
            required_moisture: Minimum moisture requirement
            max_age: Maximum age before seed rots (defaults to 200 ticks)
        """
        self.plant_type = plant_type
        self.grow_time = grow_time
        self.time_in_soil = 0
        self.required_fertility = required_fertility
        self.required_moisture = required_moisture
        self.max_age = max_age


class PlantComponent(Component):
    """
    Component for plants in the world.
    
    Attributes:
        age: Current age in ticks
        mature_age: Age at which plant can produce resources
        max_age: Maximum age before plant dies
        spawn_resource_type: Type of resource spawned when mature
        spawn_rate: Probability per tick of spawning resource when mature
    
    Author: Karan Vasa
    """
    
    def __init__(
        self,
        mature_age: int = 100,
        max_age: int = 500,
        spawn_resource_type: str = "berry",
        spawn_rate: float = 0.1
    ):
        """
        Initialize a PlantComponent.
        
        Args:
            mature_age: Age when plant becomes mature
            max_age: Maximum lifespan
            spawn_resource_type: Type of resource to spawn
            spawn_rate: Probability of spawning per tick when mature
        """
        self.age = 0
        self.mature_age = mature_age
        self.max_age = max_age
        self.spawn_resource_type = spawn_resource_type
        self.spawn_rate = spawn_rate
    
    def is_mature(self) -> bool:
        """
        Check if plant is mature enough to produce resources.
        
        Returns:
            True if plant age >= mature_age
        """
        return self.age >= self.mature_age
    
    def is_alive(self) -> bool:
        """
        Check if plant is still alive.
        
        Returns:
            True if plant age < max_age
        """
        return self.age < self.max_age


class FertilizerComponent(Component):
    """
    Component for objects that boost tile fertility.
    
    Attributes:
        fertility_boost: Amount to increase fertility of nearby tiles
        duration: Remaining ticks before effect expires
        radius: Tiles within this radius are affected
    
    Author: Karan Vasa
    """
    
    def __init__(self, fertility_boost: float = 0.2, duration: int = 100, radius: int = 2):
        """
        Initialize a FertilizerComponent.
        
        Args:
            fertility_boost: Fertility increase amount
            duration: Ticks before expiring
            radius: Effect radius in tiles
        """
        self.fertility_boost = fertility_boost
        self.duration = duration
        self.max_duration = duration
        self.radius = radius


class ToolComponent(Component):
    """
    Component for tools that modify agent interactions.
    
    This is a placeholder for future tool system implementation.
    
    Attributes:
        effect_type: Type of effect (e.g., "DIG", "HARVEST_BOOST")
        efficiency: Multiplier for effect strength
    
    Author: Karan Vasa
    """
    
    def __init__(self, effect_type: str, efficiency: float = 1.0):
        """
        Initialize a ToolComponent.
        
        Args:
            effect_type: Type of tool effect
            efficiency: Effect multiplier
        """
        self.effect_type = effect_type
        self.efficiency = efficiency


class WorldObject:
    """
    Entity in the world with component-based properties.
    
    WorldObjects use the Entity-Component-System pattern where
    objects are containers for components that define their behavior.
    
    Attributes:
        id: Unique identifier for this object
        x: X-coordinate in world
        y: Y-coordinate in world
        components: Dictionary of component name to component instance
    
    Author: Karan Vasa
    """
    
    _next_id = 0
    
    def __init__(self, x: int, y: int):
        """
        Initialize a WorldObject at the given position.
        
        Args:
            x: X-coordinate in world grid
            y: Y-coordinate in world grid
        """
        self.id = WorldObject._next_id
        WorldObject._next_id += 1
        self.x = x
        self.y = y
        self.type_id: str = ""  # Set by ObjectRegistry.create() for fast lookup
        self.is_terrain: bool = False  # Set by ObjectRegistry.create(); True if has TileEffectSpec
        self.components: Dict[str, Component] = {}
    
    def add_component(self, component: Component) -> None:
        """
        Add a component to this object.
        
        Args:
            component: Component instance to add
        """
        component_name = type(component).__name__
        self.components[component_name] = component
    
    def get_component(self, component_type: type) -> Optional[Component]:
        """
        Get a component by its type.
        
        Args:
            component_type: Type of component to retrieve
            
        Returns:
            Component instance if found, None otherwise
        """
        component_name = component_type.__name__
        return self.components.get(component_name)
    
    def has_component(self, component_type: type) -> bool:
        """
        Check if object has a component of given type.
        
        Args:
            component_type: Type of component to check
            
        Returns:
            True if component exists, False otherwise
        """
        component_name = component_type.__name__
        return component_name in self.components
    
    def remove_component(self, component_type: type) -> bool:
        """
        Remove a component from this object.
        
        Args:
            component_type: Type of component to remove
            
        Returns:
            True if component was removed, False if not found
        """
        component_name = component_type.__name__
        if component_name in self.components:
            del self.components[component_name]
            return True
        return False
    
    def __repr__(self) -> str:
        """String representation of the object."""
        component_names = ", ".join(self.components.keys())
        tid = f", type={self.type_id!r}" if self.type_id else ""
        return f"WorldObject(id={self.id}, pos=({self.x}, {self.y}){tid}, components=[{component_names}])"
