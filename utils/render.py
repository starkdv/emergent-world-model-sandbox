"""
Simple ASCII/Console renderer for the world simulation.

Provides basic text-based visualization for debugging and monitoring.

Author: Karan Vasa
"""

from typing import Optional
from world.world import World
from world.tiles import TerrainType
from world.objects import WorldObject, PlantComponent, EdibleComponent, SeedComponent


class ConsoleRenderer:
    """
    ASCII/Console renderer for world visualization.

    Renders the world grid with terrain, objects, and agents in the console.

    Attributes:
        world: World instance to render
        terrain_chars: Mapping of terrain types to ASCII characters
        show_objects: Whether to show objects in the render

    Author: Karan Vasa
    """

    # Default character mappings
    TERRAIN_CHARS = {
        TerrainType.SOIL: ".",
        TerrainType.ROCK: "#",
        TerrainType.WATER: "~",
        TerrainType.SAND: ":",
    }

    OBJECT_CHARS = {"plant": "P", "seed": "s", "berry": "o", "agent": "A"}

    def __init__(self, world: World, show_objects: bool = True):
        """
        Initialize the console renderer.

        Args:
            world: World instance to render
            show_objects: Whether to display objects on tiles
        """
        self.world = world
        self.show_objects = show_objects

    def render(self) -> str:
        """
        Render the world as an ASCII string.

        Returns:
            Multi-line string representation of the world
        """
        lines = []
        lines.append(f"=== World: Tick {self.world.tick} ===")
        lines.append(
            f"Size: {self.world.width}x{self.world.height}, Objects: {len(self.world.objects)}, Agents: {len(self.world.agents)}"
        )
        lines.append("")

        # Render grid
        for y in range(self.world.height):
            row = []
            for x in range(self.world.width):
                char = self._get_char_at(x, y)
                row.append(char)
            lines.append(" ".join(row))

        lines.append("")
        lines.append("Legend: . = Soil, # = Rock, ~ = Water, : = Sand")
        if self.show_objects:
            lines.append("        P = Plant, s = Seed, o = Berry, A = Agent")

        return "\n".join(lines)

    def _get_char_at(self, x: int, y: int) -> str:
        """
        Get the character to display at a given position.

        Args:
            x: X-coordinate
            y: Y-coordinate

        Returns:
            Character to display
        """
        tile = self.world.get_tile(x, y)
        if not tile:
            return " "

        # Check for agents first (highest priority)
        for agent_id in self.world.agents:
            agent = self.world.agents[agent_id]
            if agent.x == x and agent.y == y:
                return self.OBJECT_CHARS["agent"]

        # Check for objects if enabled
        if self.show_objects:
            objects = self.world.get_objects_at(x, y)
            if objects:
                from world.object_registry import ObjectRegistry

                # Skip terrain-layer objects when real objects exist
                render_obj = None
                terrain_obj = None
                for obj in objects:
                    if ObjectRegistry.is_terrain_layer(obj):
                        terrain_obj = obj
                    else:
                        render_obj = obj
                        break
                if render_obj is None:
                    render_obj = terrain_obj

                if render_obj is not None:
                    # Try registry-based char first
                    tid = getattr(render_obj, "type_id", "")
                    if tid:
                        defn = ObjectRegistry.get(tid)
                        if defn is not None and defn.render.char != "?":
                            return defn.render.char
                    # Fallback to component-based chars
                    if render_obj.has_component(PlantComponent):
                        return self.OBJECT_CHARS["plant"]
                    if render_obj.has_component(EdibleComponent):
                        return self.OBJECT_CHARS["berry"]
                    if render_obj.has_component(SeedComponent):
                        return self.OBJECT_CHARS["seed"]

        # Show terrain
        return self.TERRAIN_CHARS.get(tile.terrain_type, "?")

    def print(self) -> None:
        """Print the rendered world to console."""
        print(self.render())

    def render_tile_info(self, x: int, y: int) -> str:
        """
        Get detailed information about a specific tile.

        Args:
            x: X-coordinate
            y: Y-coordinate

        Returns:
            String with detailed tile information
        """
        tile = self.world.get_tile(x, y)
        if not tile:
            return f"Invalid position: ({x}, {y})"

        lines = []
        lines.append(f"=== Tile ({x}, {y}) ===")
        lines.append(f"Terrain: {tile.terrain_type.value}")
        lines.append(f"Fertility: {tile.fertility:.2f}")
        lines.append(f"Moisture: {tile.moisture:.2f}")
        lines.append(f"Passable: {tile.is_passable()}")
        lines.append(f"Plantable: {tile.is_plantable()}")

        objects = self.world.get_objects_at(x, y)
        if objects:
            lines.append(f"\nObjects ({len(objects)}):")
            for obj in objects:
                component_names = ", ".join(obj.components.keys())
                lines.append(f"  - Object {obj.id}: {component_names}")

        return "\n".join(lines)


def create_sample_world() -> World:
    """
    Create a sample world with some objects for testing visualization.

    Returns:
        World instance with sample objects

    Author: Karan Vasa
    """
    from world.objects import EdibleComponent, PlantComponent, SeedComponent

    world = World(width=20, height=20, seed=42)

    # Add some plants
    for i in range(5):
        x, y = 5 + i * 2, 5
        if world.get_tile(x, y) and world.get_tile(x, y).is_plantable():
            plant = WorldObject(x, y)
            plant.add_component(PlantComponent(mature_age=50, max_age=500))
            world.add_object(plant)

    # Add some berries
    for i in range(3):
        x, y = 10, 10 + i * 2
        berry = WorldObject(x, y)
        berry.add_component(EdibleComponent(calories=20.0))
        world.add_object(berry)

    # Add some seeds
    for i in range(4):
        x, y = 15 + i, 15
        seed = WorldObject(x, y)
        seed.add_component(SeedComponent(plant_type="berry_plant"))
        world.add_object(seed)

    return world


if __name__ == "__main__":
    # Demo the renderer
    print("Creating sample world...")
    world = create_sample_world()

    renderer = ConsoleRenderer(world)
    renderer.print()

    print("\n" + "=" * 50 + "\n")
    print(renderer.render_tile_info(5, 5))
    print("\n" + "=" * 50 + "\n")
    print(renderer.render_tile_info(10, 10))
