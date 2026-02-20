"""
Pygame-based renderer for the Emergent World-Model Sandbox.

Provides real-time graphical visualization of the world, agents, and objects
using Pygame.

Author: Karan Vasa
"""

import pygame
import sys
from typing import Optional, Tuple, Dict
from world.world import World
from world.tiles import TerrainType
from world.objects import (
    WorldObject, 
    PlantComponent, 
    EdibleComponent, 
    SeedComponent,
    FertilizerComponent,
    ToolComponent
)


# Color definitions (RGB) - Modern Dark Theme
COLORS = {
    # Terrain colors - Rich, natural tones
    'soil': (101, 67, 33),            # Rich brown
    'rock': (105, 105, 105),          # Dim gray
    'water': (30, 144, 255),          # Dodger blue
    'sand': (210, 180, 120),          # Sandy beige
    
    # Object colors - Vibrant, distinct
    'plant': (34, 139, 34),           # Forest green
    'plant_mature': (50, 205, 50),    # Lime green
    'seed': (205, 170, 125),          # Warm tan
    'seed_sprouting': (120, 200, 80), # Light green (near germination)
    'seed_agent_planted': (255, 200, 0),  # Golden — agent-planted seed
    'seed_agent_glow': (255, 235, 100),   # Bright gold glow ring
    'berry': (220, 20, 60),           # Crimson
    'berry_fresh': (255, 99, 71),     # Tomato
    
    # Agent colors - Energy gradient
    'agent': (50, 255, 50),           # Bright green
    'agent_high': (100, 220, 50),     # Yellow-green
    'agent_mid': (255, 200, 0),       # Gold
    'agent_low': (255, 100, 0),       # Orange
    'agent_critical': (255, 50, 50),  # Red
    'agent_selected': (255, 255, 0),  # Yellow
    
    # UI colors - Modern dark theme
    'background': (18, 18, 24),       # Near black
    'grid': (40, 40, 50),             # Subtle grid
    'text': (240, 240, 245),          # Off-white
    'text_dim': (140, 140, 155),      # Dimmed text
    'text_accent': (100, 200, 255),   # Cyan accent
    'panel_bg': (35, 35, 45),         # Dark panel
    'panel_border': (70, 70, 90),     # Subtle border
    'panel_header': (45, 45, 60),     # Header bg
    
    # Status colors
    'status_running': (50, 255, 100), # Bright green
    'status_paused': (255, 180, 50),  # Amber
    
    # Health bar colors
    'health_high': (80, 220, 80),     # Green
    'health_mid': (255, 190, 0),      # Yellow
    'health_low': (255, 80, 80),      # Red
    'health_bg': (40, 40, 40),        # Dark bg
}


class Camera:
    """
    Camera for panning and zooming the world view.
    
    Attributes:
        x: Camera X position in world coordinates
        y: Camera Y position in world coordinates
        zoom: Zoom level (1.0 = normal, >1.0 = zoomed in)
        
    Author: Karan Vasa
    """
    
    def __init__(self, x: float = 0, y: float = 0, zoom: float = 1.0):
        """
        Initialize the camera.
        
        Args:
            x: Initial X position
            y: Initial Y position
            zoom: Initial zoom level
        """
        self.x = x
        self.y = y
        self.zoom = zoom
        self.min_zoom = 0.25
        self.max_zoom = 4.0
    
    def move(self, dx: float, dy: float) -> None:
        """
        Move the camera by a delta.
        
        Args:
            dx: Change in X position
            dy: Change in Y position
        """
        self.x += dx / self.zoom
        self.y += dy / self.zoom
    
    def zoom_in(self, factor: float = 1.1) -> None:
        """
        Zoom in by a factor.
        
        Args:
            factor: Zoom multiplier (>1.0)
        """
        self.zoom = min(self.zoom * factor, self.max_zoom)
    
    def zoom_out(self, factor: float = 1.1) -> None:
        """
        Zoom out by a factor.
        
        Args:
            factor: Zoom divisor (>1.0)
        """
        self.zoom = max(self.zoom / factor, self.min_zoom)
    
    def world_to_screen(self, world_x: float, world_y: float, tile_size: int) -> Tuple[int, int]:
        """
        Convert world coordinates to screen coordinates.
        
        Args:
            world_x: World X coordinate
            world_y: World Y coordinate
            tile_size: Size of one tile in pixels
            
        Returns:
            Tuple of (screen_x, screen_y)
        """
        scaled_tile_size = tile_size * self.zoom
        screen_x = (world_x - self.x) * scaled_tile_size
        screen_y = (world_y - self.y) * scaled_tile_size
        return int(screen_x), int(screen_y)
    
    def screen_to_world(self, screen_x: int, screen_y: int, tile_size: int) -> Tuple[int, int]:
        """
        Convert screen coordinates to world coordinates.
        
        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate
            tile_size: Size of one tile in pixels
            
        Returns:
            Tuple of (world_x, world_y)
        """
        scaled_tile_size = tile_size * self.zoom
        world_x = int(screen_x / scaled_tile_size + self.x)
        world_y = int(screen_y / scaled_tile_size + self.y)
        return world_x, world_y


class PygameRenderer:
    """
    Main Pygame renderer for the simulation.
    
    Handles window creation, rendering, and user input.
    
    Attributes:
        world: World instance to render
        screen: Pygame display surface
        clock: Pygame clock for FPS management
        camera: Camera for view control
        running: Whether the renderer is active
        paused: Whether simulation is paused
        show_grid: Whether to show grid lines
        
    Author: Karan Vasa
    """
    def __init__(
        self,
        world: World,
        window_width: int = 1280,
        window_height: int = 800,
        tile_size: int = 24,
        target_fps: int = 60
    ):
        """
        Initialize the Pygame renderer with modern UI.
        
        Args:
            world: World instance to render
            window_width: Window width in pixels
            window_height: Window height in pixels
            tile_size: Size of each tile in pixels
            target_fps: Target frames per second
        """
        pygame.init()
        pygame.display.set_caption("🌍 Emergent World-Model Sandbox")
        
        self.world = world
        self.window_width = window_width
        self.window_height = window_height
        self.tile_size = tile_size
        self.target_fps = target_fps
        
        # Create window with double buffering
        self.screen = pygame.display.set_mode(
            (window_width, window_height),
            pygame.DOUBLEBUF | pygame.HWSURFACE
        )
        
        # Initialize clock
        self.clock = pygame.time.Clock()
        
        # Initialize camera centered on world
        self.camera = Camera(
            x=world.width / 2 - (window_width / tile_size) / 2,
            y=world.height / 2 - (window_height / tile_size) / 2,
            zoom=1.0
        )
          # State
        self.running = True
        self.paused = True
        self.show_grid = False  # Grid off by default for cleaner look
        self.selected_agent_id: Optional[int] = None
        self.hovered_tile: Optional[Tuple[int, int]] = None
        self.show_help = True  # Show help panel
        
        # Input state
        self.keys_pressed = set()
        self.mouse_dragging = False
        self.last_mouse_pos: Optional[Tuple[int, int]] = None
        
        # Fonts - try nicer system fonts
        try:
            self.font_small = pygame.font.SysFont('Segoe UI', 14)
            self.font_medium = pygame.font.SysFont('Segoe UI', 18)
            self.font_large = pygame.font.SysFont('Segoe UI Semibold', 24)
        except:
            self.font_small = pygame.font.Font(None, 18)
            self.font_medium = pygame.font.Font(None, 22)
            self.font_large = pygame.font.Font(None, 28)
        
        # FPS tracking
        self.fps_values = []
        self.tick_count = 0
    
    def handle_events(self) -> None:
        """
        Handle pygame events (keyboard, mouse, etc.).
        
        Author: Karan Vasa
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
                
                # Space - pause/unpause
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                
                # G - toggle grid
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                
                # H - toggle help
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                
                # R - reset camera
                elif event.key == pygame.K_r:
                    self.camera = Camera(
                        x=self.world.width / 2 - (self.window_width / self.tile_size) / 2,
                        y=self.world.height / 2 - (self.window_height / self.tile_size) / 2,
                        zoom=1.0
                    )
                
                # Escape - quit
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
            
            elif event.type == pygame.KEYUP:
                self.keys_pressed.discard(event.key)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
                    
                    # Check if clicking on agent
                    world_x, world_y = self.camera.screen_to_world(
                        event.pos[0], event.pos[1], self.tile_size
                    )
                    # TODO: Check for agent selection when agents are implemented
                
                elif event.button == 4:  # Scroll up
                    self.camera.zoom_in()
                
                elif event.button == 5:  # Scroll down
                    self.camera.zoom_out()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click release
                    self.mouse_dragging = False
                    self.last_mouse_pos = None
            
            elif event.type == pygame.MOUSEMOTION:
                # Update hovered tile
                world_x, world_y = self.camera.screen_to_world(
                    event.pos[0], event.pos[1], self.tile_size
                )
                if self.world.is_valid_position(world_x, world_y):
                    self.hovered_tile = (world_x, world_y)
                else:
                    self.hovered_tile = None
                
                # Handle camera dragging
                if self.mouse_dragging and self.last_mouse_pos:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.camera.move(-dx, -dy)
                    self.last_mouse_pos = event.pos
        
        # Handle continuous key presses
        if pygame.K_LEFT in self.keys_pressed or pygame.K_a in self.keys_pressed:
            self.camera.move(-5, 0)
        if pygame.K_RIGHT in self.keys_pressed or pygame.K_d in self.keys_pressed:
            self.camera.move(5, 0)
        if pygame.K_UP in self.keys_pressed or pygame.K_w in self.keys_pressed:
            self.camera.move(0, -5)
        if pygame.K_DOWN in self.keys_pressed or pygame.K_s in self.keys_pressed:
            self.camera.move(0, 5)
    
    def render_world(self) -> None:
        """
        Render the world grid with terrain and objects.
        
        Optimized for performance with spatial indexing.
        
        Author: Karan Vasa
        """
        scaled_tile_size = int(self.tile_size * self.camera.zoom)
        
        # Calculate visible tile range
        start_x = max(0, int(self.camera.x))
        start_y = max(0, int(self.camera.y))
        end_x = min(self.world.width, int(self.camera.x + self.window_width / scaled_tile_size) + 2)
        end_y = min(self.world.height, int(self.camera.y + self.window_height / scaled_tile_size) + 2)
        
        # Pre-build agent position lookup for VISIBLE area only (optimization)
        agent_positions = {}
        for agent in self.world.agents.values():
            if agent.alive and start_x <= agent.x < end_x and start_y <= agent.y < end_y:
                pos = (agent.x, agent.y)
                if pos not in agent_positions:
                    agent_positions[pos] = []
                agent_positions[pos].append(agent)
        
        # Render visible tiles
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                tile = self.world.get_tile(x, y)
                if not tile:
                    continue
                
                screen_x, screen_y = self.camera.world_to_screen(x, y, self.tile_size)
                
                # Render terrain
                terrain_color = COLORS.get(tile.terrain_type.value, COLORS['soil'])
                pygame.draw.rect(
                    self.screen,
                    terrain_color,
                    (screen_x, screen_y, scaled_tile_size, scaled_tile_size)
                )
                
                # Render objects on tile (hot-path)
                # Avoid per-tile list allocations by iterating tile.object_ids directly.
                if tile.object_ids:
                    from world.object_registry import ObjectRegistry
                    render_obj = None
                    terrain_obj = None
                    for oid in tile.object_ids:
                        o = self.world.objects.get(oid)
                        if o is None:
                            continue
                        if ObjectRegistry.is_terrain_layer(o):
                            terrain_obj = o
                        else:
                            render_obj = o
                            break
                    if render_obj is None:
                        render_obj = terrain_obj

                    if render_obj is not None:
                        obj_color = None
                        is_seed_in_soil = False
                        seed_growth_ratio = 0.0

                        # Try registry-based color first
                        tid = getattr(render_obj, 'type_id', '')
                        if tid:
                            defn = ObjectRegistry.get(tid)
                            if defn is not None:
                                obj_color = tuple(defn.render.color)

                        # Check if this is a planted seed (for special rendering)
                        # A seed is "in soil" if it's on a tile (not in an agent's
                        # inventory).  Even at time_in_soil==0 (freshly planted) we
                        # want the diamond shape so the player can see it.
                        seed_comp = render_obj.get_component(SeedComponent)
                        agent_planted = getattr(render_obj, 'planted_by_agent', False)
                        if seed_comp is not None:
                            is_seed_in_soil = True
                            seed_growth_ratio = min(1.0, seed_comp.time_in_soil / max(1, seed_comp.grow_time))
                            if agent_planted:
                                # Agent-planted seed: golden → bright sprouting green
                                sr, sg, sb = COLORS['seed_agent_planted']
                                er, eg, eb = COLORS['seed_sprouting']
                            else:
                                # Natural seed: tan → sprouting green
                                sr, sg, sb = COLORS['seed']
                                er, eg, eb = COLORS['seed_sprouting']
                            t = seed_growth_ratio
                            obj_color = (
                                int(sr + (er - sr) * t),
                                int(sg + (eg - sg) * t),
                                int(sb + (eb - sb) * t),
                            )

                        # Fallback to component-based coloring
                        if obj_color is None:
                            if render_obj.has_component(PlantComponent):
                                obj_color = COLORS['plant']
                            elif render_obj.has_component(EdibleComponent):
                                obj_color = COLORS['berry']
                            elif seed_comp is not None:
                                obj_color = COLORS['seed']
                        
                        if obj_color:
                            center_x = screen_x + scaled_tile_size // 2
                            center_y = screen_y + scaled_tile_size // 2

                            if is_seed_in_soil:
                                # Draw planted seed as DIAMOND with growth ring
                                half = max(3, scaled_tile_size // 3)
                                diamond_pts = [
                                    (center_x, center_y - half),  # top
                                    (center_x + half, center_y),  # right
                                    (center_x, center_y + half),  # bottom
                                    (center_x - half, center_y),  # left
                                ]

                                if agent_planted:
                                    # Golden glow ring behind the diamond
                                    glow_r = max(4, half + 2)
                                    pygame.draw.circle(
                                        self.screen,
                                        COLORS['seed_agent_glow'],
                                        (center_x, center_y),
                                        glow_r,
                                    )

                                pygame.draw.polygon(self.screen, obj_color, diamond_pts)
                                # Outline: gold for agent-planted, white for natural
                                outline_color = COLORS['seed_agent_planted'] if agent_planted else (255, 255, 255)
                                pygame.draw.polygon(self.screen, outline_color, diamond_pts, 1)
                                # Growth indicator: small inner dot turns green near maturity
                                if seed_growth_ratio > 0.5:
                                    inner_r = max(1, half // 3)
                                    pygame.draw.circle(
                                        self.screen,
                                        COLORS['seed_sprouting'],
                                        (center_x, center_y),
                                        inner_r,
                                    )
                            else:
                                # Regular objects: circle
                                radius = max(2, scaled_tile_size // 3)
                                pygame.draw.circle(self.screen, obj_color, (center_x, center_y), radius)
                
                # Render agents on tile (use pre-built lookup - O(1) instead of O(n))
                agents_here = agent_positions.get((x, y), [])
                if agents_here:
                    agent = agents_here[0]  # Render first agent
                    
                    # Draw agent as triangle pointing in direction
                    center_x = screen_x + scaled_tile_size // 2
                    center_y = screen_y + scaled_tile_size // 2
                    size = max(4, scaled_tile_size // 2)
                    
                    # Calculate triangle points based on direction
                    dx, dy = agent.direction
                    if dx == 0 and dy == -1:  # North
                        points = [
                            (center_x, center_y - size),  # Top
                            (center_x - size//2, center_y + size//2),  # Bottom left
                            (center_x + size//2, center_y + size//2)   # Bottom right
                        ]
                    elif dx == 1 and dy == 0:  # East
                        points = [
                            (center_x + size, center_y),  # Right
                            (center_x - size//2, center_y - size//2),  # Top left
                            (center_x - size//2, center_y + size//2)   # Bottom left
                        ]
                    elif dx == 0 and dy == 1:  # South
                        points = [
                            (center_x, center_y + size),  # Bottom
                            (center_x - size//2, center_y - size//2),  # Top left
                            (center_x + size//2, center_y - size//2)   # Top right
                        ]
                    else:  # West
                        points = [
                            (center_x - size, center_y),  # Left
                            (center_x + size//2, center_y - size//2),  # Top right
                            (center_x + size//2, center_y + size//2)   # Bottom right
                        ]
                    
                    # Color based on energy
                    if agent.energy > agent.max_energy * 0.7:
                        agent_color = COLORS['health_high']
                    elif agent.energy > agent.max_energy * 0.3:
                        agent_color = COLORS['health_mid']
                    else:
                        agent_color = COLORS['health_low']
                    
                    pygame.draw.polygon(self.screen, agent_color, points)
                    
                    # Draw outline
                    pygame.draw.polygon(self.screen, (255, 255, 255), points, 1)
                
                # Render grid if enabled
                if self.show_grid and scaled_tile_size > 4:
                    pygame.draw.rect(
                        self.screen,
                        COLORS['grid'],
                        (screen_x, screen_y, scaled_tile_size, scaled_tile_size),
                        1
                    )
                
                # Highlight hovered tile
                if self.hovered_tile and self.hovered_tile == (x, y):
                    pygame.draw.rect(
                        self.screen,
                        (255, 255, 255),
                        (screen_x, screen_y, scaled_tile_size, scaled_tile_size),
                        2
                    )
    def render_hud(self) -> None:
        """
        Render the modern heads-up display with stats and info.
        
        Author: Karan Vasa
        """
        padding = 15
        
        # ═══════════════════════════════════════════════════════════════════
        # TOP-LEFT: Main Stats Panel
        # ═══════════════════════════════════════════════════════════════════
        panel_width = 200
        panel_height = 160
        
        # Draw panel background
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((35, 35, 45, 230))
        self.screen.blit(panel_surface, (padding, padding))
        
        # Panel border
        pygame.draw.rect(
            self.screen,
            COLORS['panel_border'],
            (padding, padding, panel_width, panel_height),
            1
        )
        
        # Header bar
        pygame.draw.rect(
            self.screen,
            COLORS['panel_header'],
            (padding, padding, panel_width, 28)
        )
        
        title = self.font_medium.render("World Status", True, COLORS['text'])
        self.screen.blit(title, (padding + 10, padding + 4))
        
        # Stats
        y_offset = padding + 38
        line_height = 24
        
        # Tick counter
        tick_text = self.font_small.render(f"Tick: {self.world.tick:,}", True, COLORS['text'])
        self.screen.blit(tick_text, (padding + 15, y_offset))
        y_offset += line_height
        
        # Status with color indicator
        if self.paused:
            status_color = COLORS['status_paused']
            status_text = "PAUSED"
        else:
            status_color = COLORS['status_running']
            status_text = "RUNNING"
        
        pygame.draw.circle(self.screen, status_color, (padding + 22, y_offset + 7), 5)
        text = self.font_small.render(status_text, True, status_color)
        self.screen.blit(text, (padding + 35, y_offset))
        y_offset += line_height
        
        # Object count
        obj_text = self.font_small.render(f"Objects: {len(self.world.objects)}", True, COLORS['text'])
        self.screen.blit(obj_text, (padding + 15, y_offset))
        y_offset += line_height
        
        # Agent count
        agent_text = self.font_small.render(f"Agents: {len(self.world.agents)}", True, COLORS['text'])
        self.screen.blit(agent_text, (padding + 15, y_offset))
        y_offset += line_height
        
        # FPS with color coding
        fps = int(self.clock.get_fps())
        if fps >= 50:
            fps_color = COLORS['health_high']
        elif fps >= 30:
            fps_color = COLORS['health_mid']
        else:
            fps_color = COLORS['health_low']
        fps_text = self.font_small.render(f"FPS: {fps}", True, fps_color)
        self.screen.blit(fps_text, (padding + 15, y_offset))
        
        # ═══════════════════════════════════════════════════════════════════
        # BOTTOM-LEFT: Controls Help (toggleable with H key)
        # ═══════════════════════════════════════════════════════════════════
        if self.show_help:
            help_panel_width = 180
            help_panel_height = 130
            help_x = padding
            help_y = self.window_height - help_panel_height - padding
            
            # Semi-transparent background
            help_surface = pygame.Surface((help_panel_width, help_panel_height), pygame.SRCALPHA)
            help_surface.fill((30, 30, 40, 200))
            self.screen.blit(help_surface, (help_x, help_y))
            
            # Border
            pygame.draw.rect(
                self.screen, COLORS['panel_border'],
                (help_x, help_y, help_panel_width, help_panel_height), 1
            )
            
            # Header
            help_title = self.font_small.render("Controls", True, COLORS['text_accent'])
            self.screen.blit(help_title, (help_x + 10, help_y + 8))
            
            # Control hints
            controls = [
                ("SPACE", "Pause/Resume"),
                ("G", "Toggle Grid"),
                ("H", "Toggle Help"),
                ("R", "Reset Camera"),
                ("WASD", "Pan"),
                ("Scroll", "Zoom"),
            ]
            
            for i, (key, action) in enumerate(controls):
                y = help_y + 28 + i * 16
                key_text = self.font_small.render(key, True, COLORS['text_accent'])
                self.screen.blit(key_text, (help_x + 10, y))
                action_text = self.font_small.render(action, True, COLORS['text_dim'])
                self.screen.blit(action_text, (help_x + 70, y))
        else:
            # Just show hint
            hint = self.font_small.render("Press H for help", True, COLORS['text_dim'])
            self.screen.blit(hint, (padding, self.window_height - 25))
        
        # ═══════════════════════════════════════════════════════════════════
        # TOP-RIGHT: Zoom indicator
        # ═══════════════════════════════════════════════════════════════════
        zoom_text = f"Zoom: {self.camera.zoom:.1f}x"
        zoom_surface = self.font_small.render(zoom_text, True, COLORS['text_dim'])
        self.screen.blit(zoom_surface, (self.window_width - 85, padding))
    
    def render_tile_info(self) -> None:
        """
        Render information about the hovered tile and its objects.
        
        Author: Karan Vasa
        """
        if not self.hovered_tile:
            return
        
        x, y = self.hovered_tile
        tile = self.world.get_tile(x, y)
        if not tile:
            return
        
        # Position info panel at mouse
        mouse_x, mouse_y = pygame.mouse.get_pos()
        panel_x = mouse_x + 15
        panel_y = mouse_y + 15
        
        # Create info text
        info_lines = [
            f"Pos: ({x}, {y})",
            f"Terrain: {tile.terrain_type.value}",
            f"Fertility: {tile.fertility:.2f}",
            f"Moisture: {tile.moisture:.2f}",
        ]        # Add agent details
        agents_here = [agent for agent in self.world.agents.values() 
                      if agent.x == x and agent.y == y and agent.alive]
        if agents_here:
            info_lines.append("")  # Empty line for spacing
            info_lines.append(f"--- Agents ({len(agents_here)}) ---")
            
            for idx, agent in enumerate(agents_here[:2]):  # Show up to 2 agents
                info_lines.append("")
                info_lines.append(f"Agent #{agent.id}:")
                info_lines.append(f"  Energy: {agent.energy:.1f}/{agent.max_energy:.1f} ({agent.energy/agent.max_energy*100:.0f}%)")
                info_lines.append(f"  Age: {agent.age}/{agent.max_age}")
                info_lines.append(f"  Generation: {agent.genome.generation}")
                
                # Direction
                dx, dy = agent.direction
                direction_name = {(0, -1): "North", (1, 0): "East", 
                                 (0, 1): "South", (-1, 0): "West"}.get((dx, dy), "Unknown")
                info_lines.append(f"  Facing: {direction_name}")
                
                # Inventory
                info_lines.append(f"  Inventory: {len(agent.inventory)}/{agent.inventory_size}")
                
                # Fitness
                info_lines.append(f"  Fitness: {agent.fitness:.2f}")
                
                # Traits
                info_lines.append(f"  Metabolism: {agent.genome.traits['metabolism_rate']:.2f}")
                info_lines.append(f"  Vision: {agent.genome.traits['vision_radius']:.1f}")
                info_lines.append(f"  Speed: {agent.genome.traits['movement_speed']:.2f}")
            
            if len(agents_here) > 2:
                info_lines.append("")
                info_lines.append(f"  ... and {len(agents_here) - 2} more")
        
        # Add object details
        objects = self.world.get_objects_at(x, y)
        if objects:
            info_lines.append("")  # Empty line for spacing
            info_lines.append(f"--- Objects ({len(objects)}) ---")
            
            for idx, obj in enumerate(objects[:3]):  # Show up to 3 objects
                info_lines.append("")
                info_lines.append(f"Object #{idx + 1}:")
                
                # Check for EdibleComponent
                if obj.has_component(EdibleComponent):
                    edible = obj.get_component(EdibleComponent)
                    info_lines.append(f"  Type: Edible")
                    info_lines.append(f"  Calories: {edible.calories:.1f}")
                    info_lines.append(f"  Freshness: {edible.freshness:.2f}")
                    if edible.toxicity > 0:
                        info_lines.append(f"  Toxicity: {edible.toxicity:.2f}")
                
                # Check for PlantComponent
                if obj.has_component(PlantComponent):
                    plant = obj.get_component(PlantComponent)
                    info_lines.append(f"  Type: Plant")
                    info_lines.append(f"  Age: {plant.age}/{plant.max_age}")
                    info_lines.append(f"  Mature: {'Yes' if plant.is_mature() else 'No'}")
                    if plant.is_mature():
                        info_lines.append(f"  Spawns: {plant.spawn_resource_type}")
                        info_lines.append(f"  Spawn Rate: {plant.spawn_rate:.1%}")
                
                # Check for SeedComponent
                if obj.has_component(SeedComponent):
                    seed = obj.get_component(SeedComponent)
                    info_lines.append(f"  Type: Seed")
                    info_lines.append(f"  Plant Type: {seed.plant_type}")
                    info_lines.append(f"  Grow Time: {seed.grow_time} ticks")
                    info_lines.append(f"  Time in Soil: {seed.time_in_soil}")
                    info_lines.append(f"  Req. Fertility: {seed.required_fertility:.2f}")
                    info_lines.append(f"  Req. Moisture: {seed.required_moisture:.2f}")
                
                # Check for FertilizerComponent
                if obj.has_component(FertilizerComponent):
                    fert = obj.get_component(FertilizerComponent)
                    info_lines.append(f"  Type: Fertilizer")
                    info_lines.append(f"  Boost: +{fert.fertility_boost:.2f}")
                    info_lines.append(f"  Duration: {fert.duration}/{fert.max_duration}")
                    info_lines.append(f"  Radius: {fert.radius} tiles")
                
                # Check for ToolComponent
                if obj.has_component(ToolComponent):
                    tool = obj.get_component(ToolComponent)
                    info_lines.append(f"  Type: Tool")
                    info_lines.append(f"  Effect: {tool.effect_type}")
                    info_lines.append(f"  Efficiency: {tool.efficiency:.2f}")
            
            # Show if there are more objects
            if len(objects) > 3:
                info_lines.append("")
                info_lines.append(f"  ... and {len(objects) - 3} more")
        
        # Calculate panel size
        max_width = max(self.font_small.size(line)[0] for line in info_lines)
        panel_width = max_width + 20
        panel_height = len(info_lines) * 18 + 10
        
        # Keep panel on screen
        if panel_x + panel_width > self.window_width:
            panel_x = mouse_x - panel_width - 15
        if panel_y + panel_height > self.window_height:
            panel_y = mouse_y - panel_height - 15
        
        # Draw panel
        pygame.draw.rect(
            self.screen,
            COLORS['panel_bg'],
            (panel_x, panel_y, panel_width, panel_height)
        )
        pygame.draw.rect(
            self.screen,
            COLORS['panel_border'],
            (panel_x, panel_y, panel_width, panel_height),
            1
        )
        
        # Draw text
        for i, line in enumerate(info_lines):
            text = self.font_small.render(line, True, COLORS['text'])
            self.screen.blit(text, (panel_x + 10, panel_y + 5 + i * 18))
    
    def render(self) -> None:
        """
        Main render function - renders entire frame.
        
        Author: Karan Vasa
        """
        # Clear screen
        self.screen.fill(COLORS['background'])
        
        # Render world
        self.render_world()
        
        # Render UI overlays
        self.render_hud()
        self.render_tile_info()
        
        # Update display
        pygame.display.flip()
    
    def update(self) -> None:
        """
        Update simulation state (if not paused).
        
        Author: Karan Vasa
        """
        if not self.paused:
            self.world.update()
            self.tick_count += 1
            
            # Log world state for world model training (if logger enabled)
            from agents.agent import Agent
            if Agent.world_model_logger is not None:
                Agent.world_model_logger.log_world_state(self.world.tick, self.world)
    
    def run(self) -> None:
        """
        Main render loop.
        
        Author: Karan Vasa
        """
        print("Pygame renderer started!")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  G - Toggle Grid")
        print("  R - Reset Camera")
        print("  Arrow Keys/WASD - Pan Camera")
        print("  Mouse Wheel - Zoom")
        print("  Left Click + Drag - Pan Camera")
        print("  ESC - Quit")
        
        while self.running:
            # Handle input
            self.handle_events()
            
            # Update simulation
            self.update()
            
            # Render
            self.render()
            
            # Maintain target FPS
            frame_time_ms = self.clock.tick(self.target_fps)

            # Adaptive training budget control (if world supports it)
            if hasattr(self.world, 'adapt_learning_budget'):
                self.world.adapt_learning_budget(frame_time_ms, self.target_fps)
        
        # Cleanup
        pygame.quit()
        print(f"\nSimulation ended at tick {self.world.tick}")
