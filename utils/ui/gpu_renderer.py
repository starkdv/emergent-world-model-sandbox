"""
GPU-Accelerated Isometric Renderer for the Emergent World-Model Sandbox.

Uses ModernGL (OpenGL 3.3 core) for GPU-instanced rendering of an
isometric 2.5D world view.  Tiles, objects, and agents are rendered
via instanced draw calls — one per category — keeping CPU overhead
minimal even on large worlds.

Features:
    * Isometric diamond-tile terrain with depth-sorted rendering
    * GPU-instanced tiles / objects / agents (one draw call each)
    * Object glow effects (bloom ring for special items)
    * Agent hover tooltip showing inventory as coloured item icons
    * Smooth camera pan (WASD / drag) + zoom (scroll)
    * HUD overlay rendered via pygame text → GL texture blit
    * Performance-first: targets 60 fps on GTX 960M class hardware

Usage:
    renderer = IsometricRenderer(world, window_width=1280, ...)
    renderer.run()       # enters main loop (same API as PygameRenderer)

Author: Karan Vasa
Date: February 2026
"""

from __future__ import annotations

import math
import sys
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pygame
import moderngl

from world.world import World
from world.tiles import TerrainType
from world.objects import (
    WorldObject,
    EdibleComponent,
    PlantComponent,
    SeedComponent,
    FertilizerComponent,
    ToolComponent,
)
from world.object_registry import ObjectRegistry

# ═══════════════════════════════════════════════════════════════════════════
# Colour palette (matches the pygame renderer)
# ═══════════════════════════════════════════════════════════════════════════

TERRAIN_COLORS: Dict[str, Tuple[float, float, float]] = {
    "soil": (101 / 255, 67 / 255, 33 / 255),
    "rock": (105 / 255, 105 / 255, 105 / 255),
    "water": (30 / 255, 144 / 255, 255 / 255),
    "sand": (210 / 255, 180 / 255, 120 / 255),
}

OBJECT_COLORS: Dict[str, Tuple[float, float, float]] = {
    "plant": (34 / 255, 139 / 255, 34 / 255),
    "berry": (220 / 255, 20 / 255, 60 / 255),
    "seed": (205 / 255, 170 / 255, 125 / 255),
    "seed_planted": (255 / 255, 200 / 255, 0 / 255),
    "seed_sprouting": (120 / 255, 200 / 255, 80 / 255),
    "fertilizer": (100 / 255, 200 / 255, 100 / 255),
    "default": (200 / 255, 200 / 255, 200 / 255),
}

UI_COLORS = {
    "background": (18, 18, 24),
    "panel_bg": (35, 35, 45, 230),
    "panel_border": (70, 70, 90),
    "panel_header": (45, 45, 60),
    "text": (240, 240, 245),
    "text_dim": (140, 140, 155),
    "text_accent": (100, 200, 255),
    "text_good": (80, 220, 80),
    "text_warn": (255, 190, 0),
    "text_bad": (255, 80, 80),
    "status_running": (50, 255, 100),
    "status_paused": (255, 180, 50),
}

# ═══════════════════════════════════════════════════════════════════════════
# GLSL Shaders
# ═══════════════════════════════════════════════════════════════════════════

TILE_VERTEX_SHADER = """
#version 330 core

// Per-vertex (unit diamond quad — 6 verts)
in vec2 in_vert;          // local offset

// Per-instance
in vec2 in_world_pos;     // world x, y
in vec3 in_color;         // terrain RGB
in float in_height;       // slight height offset for depth feel

// Uniforms
uniform float u_tile_hw;   // tile half-width (pixels)
uniform float u_tile_hh;   // tile half-height (pixels)
uniform vec2  u_camera;    // camera world pos
uniform float u_zoom;      // camera zoom
uniform vec2  u_screen;    // screen dimensions

out vec3 v_color;
out vec2 v_uv;

void main() {
    // Isometric projection: world → screen
    float wx = in_world_pos.x - u_camera.x;
    float wy = in_world_pos.y - u_camera.y;
    float iso_cx = (wx - wy) * u_tile_hw * u_zoom;
    float iso_cy = (wx + wy) * u_tile_hh * u_zoom - in_height * u_zoom;

    // Apply per-vertex offset (diamond shape)
    float vx = iso_cx + in_vert.x * u_tile_hw * u_zoom;
    float vy = iso_cy + in_vert.y * u_tile_hh * u_zoom;

    // Centre on screen
    vx += u_screen.x * 0.5;
    vy += u_screen.y * 0.5;

    // To NDC
    float ndc_x = (vx / u_screen.x) * 2.0 - 1.0;
    float ndc_y = 1.0 - (vy / u_screen.y) * 2.0;

    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);

    v_color = in_color;
    v_uv = in_vert;
}
"""

TILE_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_color;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    // Subtle lighting gradient (top-left lit)
    float light = 0.85 + 0.15 * (1.0 - v_uv.y);
    vec3 lit = v_color * light;

    // Edge highlight (thin bright edge on top-left faces)
    float edge = smoothstep(0.92, 1.0, abs(v_uv.x) + abs(v_uv.y));
    lit = mix(lit, lit * 1.35, edge * 0.3);

    fragColor = vec4(lit, 1.0);
}
"""

SPRITE_VERTEX_SHADER = """
#version 330 core

in vec2 in_vert;          // unit circle quad verts

// Per-instance
in vec2 in_world_pos;     // world x, y
in vec3 in_color;         // RGB
in float in_radius;       // radius multiplier
in float in_glow;         // glow intensity (0 = none)
in float in_shape;        // 0 = circle, 1 = diamond, 2 = triangle

uniform float u_tile_hw;
uniform float u_tile_hh;
uniform vec2  u_camera;
uniform float u_zoom;
uniform vec2  u_screen;

out vec3 v_color;
out vec2 v_uv;
out float v_glow;
out float v_shape;

void main() {
    float wx = in_world_pos.x - u_camera.x;
    float wy = in_world_pos.y - u_camera.y;
    float iso_cx = (wx - wy) * u_tile_hw * u_zoom;
    float iso_cy = (wx + wy) * u_tile_hh * u_zoom;

    // Offset to tile centre, then apply vertex
    float size = in_radius * u_tile_hh * u_zoom * 0.7;
    float vx = iso_cx + in_vert.x * size + u_screen.x * 0.5;
    float vy = iso_cy + in_vert.y * size + u_screen.y * 0.5;

    float ndc_x = (vx / u_screen.x) * 2.0 - 1.0;
    float ndc_y = 1.0 - (vy / u_screen.y) * 2.0;

    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);

    v_color = in_color;
    v_uv = in_vert;
    v_glow = in_glow;
    v_shape = in_shape;
}
"""

SPRITE_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_color;
in vec2 v_uv;
in float v_glow;
in float v_shape;
out vec4 fragColor;

void main() {
    float dist = length(v_uv);

    float alpha;
    if (v_shape < 0.5) {
        // Circle
        alpha = 1.0 - smoothstep(0.7, 0.85, dist);
    } else if (v_shape < 1.5) {
        // Diamond
        float d = abs(v_uv.x) + abs(v_uv.y);
        alpha = 1.0 - smoothstep(0.7, 0.85, d);
    } else {
        // Triangle (point up)
        float tri = v_uv.y + abs(v_uv.x) * 1.2;
        alpha = 1.0 - smoothstep(0.3, 0.45, tri);
        alpha *= step(-0.5, -v_uv.y);  // cut bottom
    }

    if (alpha < 0.05) discard;

    // Glow ring
    vec3 col = v_color;
    if (v_glow > 0.0) {
        float glow_ring = smoothstep(0.5, 0.85, dist) * (1.0 - smoothstep(0.85, 1.0, dist));
        vec3 glow_col = v_color * 1.5 + vec3(0.2, 0.15, 0.0);
        col = mix(col, glow_col, glow_ring * v_glow);
        // Also brighten the core
        col *= 1.0 + v_glow * 0.3;
    }

    fragColor = vec4(col, alpha);
}
"""

# Fullscreen quad for UI texture overlay
OVERLAY_VERTEX_SHADER = """
#version 330 core
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_uv;
}
"""

OVERLAY_FRAGMENT_SHADER = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_texture;
void main() {
    fragColor = texture(u_texture, v_uv);
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Renderer
# ═══════════════════════════════════════════════════════════════════════════


class IsometricRenderer:
    """
    GPU-accelerated isometric renderer using ModernGL.

    Provides the same external API as PygameRenderer (run(), update(), etc.)
    so it can be swapped in via a CLI flag.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        world: World,
        window_width: int = 1280,
        window_height: int = 800,
        tile_size: int = 24,
        target_fps: int = 60,
    ):
        self.world = world
        self.window_width = window_width
        self.window_height = window_height
        self.tile_size = tile_size
        self.target_fps = target_fps

        # Isometric half-sizes
        self.tile_hw = tile_size * 0.75  # half-width of diamond
        self.tile_hh = tile_size * 0.375  # half-height of diamond

        # ── Pygame window (with OpenGL context) ──
        pygame.init()
        pygame.display.set_caption("🌍 Emergent World — Isometric GPU View")
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        self.screen = pygame.display.set_mode(
            (window_width, window_height),
            pygame.OPENGL | pygame.DOUBLEBUF,
        )
        self.clock = pygame.time.Clock()

        # ── ModernGL context ──
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )

        # ── Camera ──
        self.cam_x: float = world.width / 2.0
        self.cam_y: float = world.height / 2.0
        self.cam_zoom: float = 1.0
        self.cam_zoom_min = 0.3
        self.cam_zoom_max = 5.0

        # ── State ──
        self.running = True
        self.paused = True
        self.show_help = True
        self.tick_count = 0
        self.hovered_tile: Optional[Tuple[int, int]] = None
        self.mouse_dragging = False
        self.last_mouse_pos: Optional[Tuple[int, int]] = None
        self.keys_pressed: set = set()

        # ── Fonts (rendered via pygame → texture) ──
        try:
            self.font_small = pygame.font.SysFont("Segoe UI", 14)
            self.font_medium = pygame.font.SysFont("Segoe UI", 18)
            self.font_large = pygame.font.SysFont("Segoe UI Semibold", 24)
        except Exception:
            self.font_small = pygame.font.Font(None, 18)
            self.font_medium = pygame.font.Font(None, 22)
            self.font_large = pygame.font.Font(None, 28)

        # ── Compile shaders ──
        self.tile_prog = self.ctx.program(
            vertex_shader=TILE_VERTEX_SHADER,
            fragment_shader=TILE_FRAGMENT_SHADER,
        )
        self.sprite_prog = self.ctx.program(
            vertex_shader=SPRITE_VERTEX_SHADER,
            fragment_shader=SPRITE_FRAGMENT_SHADER,
        )
        self.overlay_prog = self.ctx.program(
            vertex_shader=OVERLAY_VERTEX_SHADER,
            fragment_shader=OVERLAY_FRAGMENT_SHADER,
        )

        # ── Geometry ── (unit shapes — positions applied via instancing)
        self._build_tile_geometry()
        self._build_sprite_geometry()
        self._build_overlay_geometry()

        # ── Cached tile data (terrain rarely changes) ──
        self._tile_cache: Optional[np.ndarray] = None
        self._tile_cache_tick: int = -1
        self._tile_cache_range: Tuple[int, int, int, int] = (-1, -1, -1, -1)
        self._tile_dirty: bool = True

        # ── Cached object/agent instance data (only changes per tick) ──
        self._obj_cache: Optional[np.ndarray] = None
        self._obj_cache_tick: int = -1
        self._agent_cache: Optional[np.ndarray] = None
        self._agent_cache_tick: int = -1

        # ── Reusable GPU buffers (orphan+write, never alloc/free per frame) ──
        max_tiles = world.width * world.height
        self._tile_buf = self.ctx.buffer(reserve=max_tiles * 6 * 4)  # 6 floats
        self._tile_vao = self.ctx.vertex_array(
            self.tile_prog,
            [
                (self.tile_vbo, "2f", "in_vert"),
                (self._tile_buf, "2f 3f 1f/i", "in_world_pos", "in_color", "in_height"),
            ],
        )

        max_sprites = max(2000, len(world.objects) + len(world.agents) + 100)
        self._obj_buf = self.ctx.buffer(reserve=max_sprites * 8 * 4)
        self._obj_vao = self.ctx.vertex_array(
            self.sprite_prog,
            [
                (self.sprite_vbo, "2f", "in_vert"),
                (
                    self._obj_buf,
                    "2f 3f 1f 1f 1f/i",
                    "in_world_pos",
                    "in_color",
                    "in_radius",
                    "in_glow",
                    "in_shape",
                ),
            ],
        )

        self._agent_buf = self.ctx.buffer(reserve=max_sprites * 8 * 4)
        self._agent_vao = self.ctx.vertex_array(
            self.sprite_prog,
            [
                (self.sprite_vbo, "2f", "in_vert"),
                (
                    self._agent_buf,
                    "2f 3f 1f 1f 1f/i",
                    "in_world_pos",
                    "in_color",
                    "in_radius",
                    "in_glow",
                    "in_shape",
                ),
            ],
        )

        # ── UI overlay texture (cached, only re-upload when dirty) ──
        self.ui_surface = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        self.ui_texture: Optional[moderngl.Texture] = None
        self._ui_dirty: bool = True
        self._last_ui_tick: int = -1
        self._last_ui_paused: bool = self.paused
        self._last_ui_hovered: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Geometry builders
    # ------------------------------------------------------------------

    def _build_tile_geometry(self):
        """Unit diamond quad: 6 verts forming 2 triangles."""
        # Diamond: top=(0,-1), right=(1,0), bottom=(0,1), left=(-1,0)
        verts = np.array(
            [
                # tri 1
                0.0,
                -1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                # tri 2
                0.0,
                -1.0,
                0.0,
                1.0,
                -1.0,
                0.0,
            ],
            dtype="f4",
        )
        self.tile_vbo = self.ctx.buffer(verts)

    def _build_sprite_geometry(self):
        """Unit quad for sprite instancing (circle/diamond/triangle via shader)."""
        verts = np.array(
            [
                -1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
            ],
            dtype="f4",
        )
        self.sprite_vbo = self.ctx.buffer(verts)

    def _build_overlay_geometry(self):
        """Fullscreen quad for HUD texture."""
        verts = np.array(
            [
                # pos        uv
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype="f4",
        )
        buf = self.ctx.buffer(verts)
        self.overlay_vao = self.ctx.vertex_array(
            self.overlay_prog,
            [(buf, "2f 2f", "in_vert", "in_uv")],
        )

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        """World coords → pixel screen coords (for pygame UI overlay)."""
        rx = wx - self.cam_x
        ry = wy - self.cam_y
        sx = (rx - ry) * self.tile_hw * self.cam_zoom + self.window_width * 0.5
        sy = (rx + ry) * self.tile_hh * self.cam_zoom + self.window_height * 0.5
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> Tuple[int, int]:
        """Pixel screen coords → world tile coords (for mouse picking)."""
        # Invert the isometric projection
        rx = sx - self.window_width * 0.5
        ry = sy - self.window_height * 0.5
        thw = self.tile_hw * self.cam_zoom
        thh = self.tile_hh * self.cam_zoom
        if thw == 0 or thh == 0:
            return 0, 0
        # iso_x = (rx/thw + ry/thh) / 2 + cam
        # iso_y = (ry/thh - rx/thw) / 2 + cam
        wx = (rx / thw + ry / thh) / 2.0 + self.cam_x
        wy = (ry / thh - rx / thw) / 2.0 + self.cam_y
        return int(round(wx)), int(round(wy))

    # ------------------------------------------------------------------
    # Instance-data builders (called each frame)
    # ------------------------------------------------------------------

    def _get_visible_range(self) -> Tuple[int, int, int, int]:
        """Return (x_min, x_max, y_min, y_max) of tiles visible on screen."""
        # Sample the four screen corners → world coords, then pad
        corners = [
            self.screen_to_world(0, 0),
            self.screen_to_world(self.window_width, 0),
            self.screen_to_world(0, self.window_height),
            self.screen_to_world(self.window_width, self.window_height),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        pad = 2
        x_min = max(0, min(xs) - pad)
        x_max = min(self.world.width - 1, max(xs) + pad)
        y_min = max(0, min(ys) - pad)
        y_max = min(self.world.height - 1, max(ys) + pad)
        return x_min, x_max, y_min, y_max

    def _build_tile_instances(self) -> np.ndarray:
        """Build per-tile instance data with frustum culling + caching."""
        # Only visible tiles
        x0, x1, y0, y1 = self._get_visible_range()
        vw = x1 - x0 + 1
        vh = y1 - y0 + 1
        count = vw * vh
        if count <= 0:
            return np.zeros((0, 6), dtype="f4")

        # Use cache if terrain hasn't changed (check every 50 ticks)
        cur_tick = self.world.tick
        if (
            self._tile_cache is not None
            and not self._tile_dirty
            and self._tile_cache_range == (x0, x1, y0, y1)
        ):
            return self._tile_cache

        data = np.zeros((count, 6), dtype="f4")

        # Pre-build lookup arrays for speed
        soil_rgb = TERRAIN_COLORS.get("soil", (0.4, 0.26, 0.13))
        color_lut = {
            "soil": TERRAIN_COLORS["soil"],
            "rock": TERRAIN_COLORS["rock"],
            "water": TERRAIN_COLORS["water"],
            "sand": TERRAIN_COLORS["sand"],
        }
        height_lut = {"rock": 2.0, "water": -1.0}

        idx = 0
        tiles = self.world.tiles
        for y in range(y0, y1 + 1):
            row = tiles[y]
            for x in range(x0, x1 + 1):
                tile = row[x]
                ttype = tile.terrain_type.value
                r, g, b = color_lut.get(ttype, soil_rgb)
                fert = tile.fertility
                moist = tile.moisture
                data[idx, 0] = x
                data[idx, 1] = y
                data[idx, 2] = min(1.0, r + fert * 0.05)
                data[idx, 3] = min(1.0, g + fert * 0.08)
                data[idx, 4] = min(1.0, b + moist * 0.06)
                data[idx, 5] = height_lut.get(ttype, 0.0)
                idx += 1

        data = data[:idx]
        self._tile_cache = data
        self._tile_cache_range = (x0, x1, y0, y1)
        self._tile_dirty = False
        self._tile_cache_tick = cur_tick
        return data

    def _build_object_instances(self) -> np.ndarray:
        """Build per-object instance data: (wx, wy, r, g, b, radius, glow, shape)."""
        # Cache per-tick: objects only change between ticks, not between frames
        cur_tick = self.world.tick
        if self._obj_cache is not None and self._obj_cache_tick == cur_tick:
            return self._obj_cache

        # O(agents) once to build inventory set, then O(1) per object
        inv_ids: set = set()
        for agent in self.world.agents.values():
            inv_ids.update(agent.inventory)

        # Visible range for frustum culling
        x0, x1, y0, y1 = self._get_visible_range()

        objs: List[WorldObject] = []
        for oid, obj in self.world.objects.items():
            if oid in inv_ids:
                continue
            # Cull off-screen objects
            if obj.x < x0 or obj.x > x1 or obj.y < y0 or obj.y > y1:
                continue
            objs.append(obj)

        if not objs:
            return np.zeros((0, 8), dtype="f4")

        data = np.zeros((len(objs), 8), dtype="f4")
        for i, obj in enumerate(objs):
            r, g, b = OBJECT_COLORS["default"]
            radius = 0.8
            glow = 0.0
            shape = 0.0  # circle

            # Check terrain layer (use cached flag — set at creation)
            is_terrain = getattr(obj, "is_terrain", False)

            # Registry colour
            tid = getattr(obj, "type_id", "")
            if tid:
                defn = ObjectRegistry.get(tid)
                if defn is not None:
                    rc = defn.render.color
                    r, g, b = rc[0] / 255, rc[1] / 255, rc[2] / 255

            # Seed special rendering
            seed_comp = obj.get_component(SeedComponent)
            if seed_comp is not None:
                shape = 1.0  # diamond
                agent_planted = getattr(obj, "planted_by_agent", False)
                grow_ratio = min(
                    1.0, seed_comp.time_in_soil / max(1, seed_comp.grow_time)
                )
                if agent_planted:
                    sr, sg, sb = OBJECT_COLORS["seed_planted"]
                    er, eg, eb = OBJECT_COLORS["seed_sprouting"]
                    glow = 0.8
                else:
                    sr, sg, sb = OBJECT_COLORS["seed"]
                    er, eg, eb = OBJECT_COLORS["seed_sprouting"]
                t = grow_ratio
                r = sr + (er - sr) * t
                g = sg + (eg - sg) * t
                b = sb + (eb - sb) * t
                radius = 0.6

            elif obj.has_component(EdibleComponent):
                if not tid:
                    r, g, b = OBJECT_COLORS["berry"]
                glow = 0.3
                radius = 0.7

            elif obj.has_component(PlantComponent):
                plant = obj.get_component(PlantComponent)
                if not tid:
                    r, g, b = OBJECT_COLORS["plant"]
                if plant.is_mature():
                    glow = 0.15
                    radius = 1.0
                else:
                    radius = 0.5 + 0.5 * min(1.0, plant.age / max(1, plant.mature_age))

            elif is_terrain:
                radius = 1.1  # fill tile
                shape = 1.0  # diamond
                glow = 0.0

            data[i] = (obj.x, obj.y, r, g, b, radius, glow, shape)

        self._obj_cache = data
        self._obj_cache_tick = cur_tick
        return data

    def _build_agent_instances(self) -> np.ndarray:
        """Build per-agent instance data: (wx, wy, r, g, b, radius, glow, shape)."""
        # Cache per-tick: agents only change between ticks
        cur_tick = self.world.tick
        if self._agent_cache is not None and self._agent_cache_tick == cur_tick:
            return self._agent_cache

        agents = [a for a in self.world.agents.values() if a.alive]
        if not agents:
            return np.zeros((0, 8), dtype="f4")

        data = np.zeros((len(agents), 8), dtype="f4")
        for i, agent in enumerate(agents):
            energy_ratio = agent.energy / agent.max_energy
            if energy_ratio > 0.7:
                r, g, b = 80 / 255, 220 / 255, 80 / 255
            elif energy_ratio > 0.4:
                r, g, b = 255 / 255, 200 / 255, 0 / 255
            elif energy_ratio > 0.2:
                r, g, b = 255 / 255, 100 / 255, 0 / 255
            else:
                r, g, b = 255 / 255, 50 / 255, 50 / 255

            glow = 0.2 if energy_ratio > 0.5 else 0.0
            data[i] = (agent.x, agent.y, r, g, b, 0.85, glow, 2.0)

        self._agent_cache = data
        self._agent_cache_tick = cur_tick
        return data

    # ------------------------------------------------------------------
    # Render passes (reuse persistent GPU buffers via orphan+write)
    # ------------------------------------------------------------------

    def _set_uniforms(self, prog: moderngl.Program):
        """Set common uniforms on a shader program."""
        if "u_tile_hw" in prog:
            prog["u_tile_hw"].value = self.tile_hw
        if "u_tile_hh" in prog:
            prog["u_tile_hh"].value = self.tile_hh
        if "u_camera" in prog:
            prog["u_camera"].value = (self.cam_x, self.cam_y)
        if "u_zoom" in prog:
            prog["u_zoom"].value = self.cam_zoom
        if "u_screen" in prog:
            prog["u_screen"].value = (
                float(self.window_width),
                float(self.window_height),
            )

    def _render_tiles(self, tile_data: np.ndarray):
        if tile_data.shape[0] == 0:
            return
        self._set_uniforms(self.tile_prog)
        raw = tile_data.tobytes()
        # Orphan the buffer (driver can reuse old memory) then write new data
        self._tile_buf.orphan(len(raw))
        self._tile_buf.write(raw)
        self._tile_vao.render(moderngl.TRIANGLES, instances=tile_data.shape[0])

    def _render_objects(self, instance_data: np.ndarray):
        if instance_data.shape[0] == 0:
            return
        self._set_uniforms(self.sprite_prog)
        raw = instance_data.tobytes()
        self._obj_buf.orphan(len(raw))
        self._obj_buf.write(raw)
        self._obj_vao.render(moderngl.TRIANGLES, instances=instance_data.shape[0])

    def _render_agents(self, instance_data: np.ndarray):
        if instance_data.shape[0] == 0:
            return
        self._set_uniforms(self.sprite_prog)
        raw = instance_data.tobytes()
        self._agent_buf.orphan(len(raw))
        self._agent_buf.write(raw)
        self._agent_vao.render(moderngl.TRIANGLES, instances=instance_data.shape[0])

    def _render_ui_overlay(self):
        """Blit the pygame UI surface → GL texture → fullscreen quad.

        Only re-uploads the texture when UI content has actually changed.
        """
        # Check if UI state changed
        cur_tick = self.world.tick
        ui_changed = (
            self._ui_dirty
            or cur_tick != self._last_ui_tick
            or self.paused != self._last_ui_paused
            or self.hovered_tile != self._last_ui_hovered
        )

        if ui_changed:
            self.ui_surface.fill((0, 0, 0, 0))
            self._paint_hud()
            self._paint_tooltip()

            raw = pygame.image.tostring(self.ui_surface, "RGBA", True)
            if self.ui_texture is None:
                self.ui_texture = self.ctx.texture(
                    (self.window_width, self.window_height), 4, raw
                )
                self.ui_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            else:
                self.ui_texture.write(raw)

            self._ui_dirty = False
            self._last_ui_tick = cur_tick
            self._last_ui_paused = self.paused
            self._last_ui_hovered = self.hovered_tile

        if self.ui_texture is not None:
            self.ui_texture.use(0)
            if "u_texture" in self.overlay_prog:
                self.overlay_prog["u_texture"].value = 0
            self.overlay_vao.render(moderngl.TRIANGLES)

    # ------------------------------------------------------------------
    # UI painting (pygame surface → later blitted as GL texture)
    # ------------------------------------------------------------------

    def _paint_hud(self):
        """Draw HUD onto self.ui_surface."""
        pad = 15

        # -- Top-left: World Status --
        pw, ph = 210, 170
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill(UI_COLORS["panel_bg"])
        self.ui_surface.blit(panel, (pad, pad))
        pygame.draw.rect(
            self.ui_surface,
            UI_COLORS["panel_border"],
            (pad, pad, pw, ph),
            1,
        )
        # Header
        pygame.draw.rect(
            self.ui_surface,
            UI_COLORS["panel_header"],
            (pad, pad, pw, 28),
        )
        title = self.font_medium.render("World Status", True, UI_COLORS["text"])
        self.ui_surface.blit(title, (pad + 10, pad + 4))

        y = pad + 38
        lh = 24
        self._text(f"Tick: {self.world.tick:,}", pad + 15, y, "text")
        y += lh
        # Status dot
        sc = UI_COLORS["status_paused"] if self.paused else UI_COLORS["status_running"]
        st = "PAUSED" if self.paused else "RUNNING"
        pygame.draw.circle(self.ui_surface, sc, (pad + 22, y + 7), 5)
        self._text(st, pad + 35, y, color_key=None, color=sc)
        y += lh
        self._text(f"Objects: {len(self.world.objects)}", pad + 15, y, "text")
        y += lh
        alive = self.world.get_cached_object_counts().get("alive_agents", 0)
        self._text(f"Agents: {alive}", pad + 15, y, "text")
        y += lh
        fps = int(self.clock.get_fps())
        fk = "text_good" if fps >= 50 else ("text_warn" if fps >= 30 else "text_bad")
        self._text(f"FPS: {fps}", pad + 15, y, fk)

        # -- Bottom-left: Controls help --
        if self.show_help:
            hpw, hph = 190, 145
            hx, hy = pad, self.window_height - hph - pad
            hp = pygame.Surface((hpw, hph), pygame.SRCALPHA)
            hp.fill((30, 30, 40, 200))
            self.ui_surface.blit(hp, (hx, hy))
            pygame.draw.rect(
                self.ui_surface,
                UI_COLORS["panel_border"],
                (hx, hy, hpw, hph),
                1,
            )
            self._text("Controls", hx + 10, hy + 8, "text_accent")
            ctrls = [
                ("SPACE", "Pause/Resume"),
                ("G", "Toggle Grid (n/a)"),
                ("H", "Toggle Help"),
                ("R", "Reset Camera"),
                ("WASD", "Pan"),
                ("Scroll", "Zoom"),
                ("Hover", "Inspect tile"),
            ]
            for i, (k, a) in enumerate(ctrls):
                cy = hy + 28 + i * 16
                self._text(k, hx + 10, cy, "text_accent", font=self.font_small)
                self._text(a, hx + 75, cy, "text_dim", font=self.font_small)
        else:
            self._text(
                "Press H for help",
                pad,
                self.window_height - 25,
                "text_dim",
            )

        # -- Top-right: zoom --
        zt = f"Zoom: {self.cam_zoom:.1f}x"
        self._text(zt, self.window_width - 100, pad, "text_dim")

    def _paint_tooltip(self):
        """Draw hover tooltip with agent inventory icons + tile info."""
        if not self.hovered_tile:
            return

        x, y = self.hovered_tile
        tile = self.world.get_tile(x, y)
        if not tile:
            return

        mx, my = pygame.mouse.get_pos()
        lines: List[Tuple[str, str]] = []  # (text, colour_key)

        lines.append((f"Pos: ({x}, {y})", "text"))
        lines.append((f"Terrain: {tile.terrain_type.value}", "text"))
        lines.append((f"Fertility: {tile.fertility:.2f}", "text"))
        lines.append((f"Moisture: {tile.moisture:.2f}", "text"))

        # Objects on tile
        objects = self.world.get_objects_at(x, y)
        if objects:
            lines.append(("", "text"))
            lines.append((f"— Objects ({len(objects)}) —", "text_accent"))
            for obj in objects[:3]:
                tid = getattr(obj, "type_id", "")
                defn = ObjectRegistry.get(tid) if tid else None
                name = defn.display_name if defn else "Object"
                lines.append((f"  {name}", "text"))
                if obj.has_component(EdibleComponent):
                    e = obj.get_component(EdibleComponent)
                    lines.append(
                        (
                            f"    Cal: {e.calories:.0f}  Fresh: {e.freshness:.0%}",
                            "text_dim",
                        )
                    )
                if obj.has_component(PlantComponent):
                    p = obj.get_component(PlantComponent)
                    lines.append(
                        (
                            f"    Age: {p.age}/{p.max_age}  Mature: {'✓' if p.is_mature() else '✗'}",
                            "text_dim",
                        )
                    )
                if obj.has_component(SeedComponent):
                    s = obj.get_component(SeedComponent)
                    lines.append(
                        (f"    Soil: {s.time_in_soil}/{s.grow_time}", "text_dim")
                    )

        # Agents on tile
        agents_here = [
            a for a in self.world.agents.values() if a.x == x and a.y == y and a.alive
        ]
        if agents_here:
            lines.append(("", "text"))
            lines.append((f"— Agents ({len(agents_here)}) —", "text_accent"))
            for agent in agents_here[:2]:
                ep = agent.energy / agent.max_energy
                ek = (
                    "text_good"
                    if ep > 0.5
                    else ("text_warn" if ep > 0.2 else "text_bad")
                )
                lines.append((f"  Agent #{agent.id}", "text"))
                lines.append(
                    (
                        f"    Energy: {agent.energy:.0f}/{agent.max_energy:.0f} ({ep:.0%})",
                        ek,
                    )
                )
                lines.append(
                    (
                        f"    Age: {agent.age}  Gen: {agent.genome.generation}",
                        "text_dim",
                    )
                )
                # -- Inventory as coloured item icons --
                if agent.inventory:
                    inv_text = f"    Inventory ({len(agent.inventory)}/{agent.inventory_size}):"
                    lines.append((inv_text, "text"))
                    for inv_id in agent.inventory:
                        inv_obj = self.world.objects.get(inv_id)
                        if inv_obj is None:
                            continue
                        inv_tid = getattr(inv_obj, "type_id", "")
                        inv_defn = ObjectRegistry.get(inv_tid) if inv_tid else None
                        inv_name = inv_defn.display_name if inv_defn else "item"
                        lines.append((f"      ● {inv_name}", "text"))
                        # We'll draw a coloured dot before the name below
                else:
                    lines.append((f"    Inventory: empty", "text_dim"))

        # Measure & render panel
        max_w = max((self.font_small.size(t)[0] for t, _ in lines), default=100)
        pw = max_w + 30
        phl = len(lines) * 18 + 12
        px = mx + 15
        py = my + 15
        if px + pw > self.window_width:
            px = mx - pw - 15
        if py + phl > self.window_height:
            py = my - phl - 15

        panel = pygame.Surface((pw, phl), pygame.SRCALPHA)
        panel.fill(UI_COLORS["panel_bg"])
        self.ui_surface.blit(panel, (px, py))
        pygame.draw.rect(
            self.ui_surface,
            UI_COLORS["panel_border"],
            (px, py, pw, phl),
            1,
        )

        # Draw lines with inventory item colour dots
        for i, (text, ck) in enumerate(lines):
            ty = py + 6 + i * 18
            col = UI_COLORS.get(ck, UI_COLORS["text"])
            # Coloured dot for inventory items
            if text.strip().startswith("●"):
                # Find the object colour
                item_name = text.strip()[2:]
                dot_color = self._lookup_item_color(item_name)
                pygame.draw.circle(
                    self.ui_surface,
                    dot_color,
                    (px + 52, ty + 7),
                    5,
                )
                # Render the name after the dot
                rendered = self.font_small.render(f"  {item_name}", True, col)
                self.ui_surface.blit(rendered, (px + 58, ty))
            else:
                rendered = self.font_small.render(text, True, col)
                self.ui_surface.blit(rendered, (px + 10, ty))

    def _lookup_item_color(self, display_name: str) -> Tuple[int, int, int]:
        """Get the render colour for an item by display name."""
        for defn in ObjectRegistry.all_definitions().values():
            if defn.display_name == display_name:
                return tuple(defn.render.color)
        # Fallback colour categories
        name_lower = display_name.lower()
        if "berry" in name_lower or "food" in name_lower:
            return (220, 20, 60)
        if "seed" in name_lower or "acorn" in name_lower:
            return (205, 170, 125)
        if "plant" in name_lower or "tree" in name_lower:
            return (34, 139, 34)
        return (200, 200, 200)

    def _text(
        self,
        text: str,
        x: int,
        y: int,
        color_key: Optional[str] = "text",
        color: Optional[Tuple[int, ...]] = None,
        font: Optional[pygame.font.Font] = None,
    ):
        """Helper — render text onto ui_surface."""
        if font is None:
            font = self.font_small
        c = color if color else UI_COLORS.get(color_key, UI_COLORS["text"])
        surf = font.render(text, True, c)
        self.ui_surface.blit(surf, (x, y))

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_r:
                    self.cam_x = self.world.width / 2.0
                    self.cam_y = self.world.height / 2.0
                    self.cam_zoom = 1.0
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

            elif event.type == pygame.KEYUP:
                self.keys_pressed.discard(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
                elif event.button == 4:
                    self.cam_zoom = min(self.cam_zoom * 1.1, self.cam_zoom_max)
                    self._tile_dirty = True
                elif event.button == 5:
                    self.cam_zoom = max(self.cam_zoom / 1.1, self.cam_zoom_min)
                    self._tile_dirty = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
                    self.last_mouse_pos = None

            elif event.type == pygame.MOUSEMOTION:
                wx, wy = self.screen_to_world(event.pos[0], event.pos[1])
                if self.world.is_valid_position(wx, wy):
                    self.hovered_tile = (wx, wy)
                else:
                    self.hovered_tile = None

                if self.mouse_dragging and self.last_mouse_pos:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    # Convert pixel delta → world delta (invert isometric)
                    thw = self.tile_hw * self.cam_zoom
                    thh = self.tile_hh * self.cam_zoom
                    if thw > 0 and thh > 0:
                        dwx = -(dx / thw + dy / thh) / 2.0
                        dwy = -(dy / thh - dx / thw) / 2.0
                        self.cam_x += dwx
                        self.cam_y += dwy
                        self._tile_dirty = True
                    self.last_mouse_pos = event.pos

        # Continuous key presses → camera pan
        pan_speed = 0.15 / max(self.cam_zoom, 0.3)
        moved = False
        if pygame.K_LEFT in self.keys_pressed or pygame.K_a in self.keys_pressed:
            self.cam_x -= pan_speed
            moved = True
        if pygame.K_RIGHT in self.keys_pressed or pygame.K_d in self.keys_pressed:
            self.cam_x += pan_speed
            moved = True
        if pygame.K_UP in self.keys_pressed or pygame.K_w in self.keys_pressed:
            self.cam_y -= pan_speed
            moved = True
        if pygame.K_DOWN in self.keys_pressed or pygame.K_s in self.keys_pressed:
            self.cam_y += pan_speed
            moved = True
        if moved:
            self._tile_dirty = True

    # ------------------------------------------------------------------
    # Update + Render
    # ------------------------------------------------------------------

    def update(self):
        """Advance simulation by one tick (if not paused)."""
        if not self.paused:
            self.world.update()
            self.tick_count += 1
            # Invalidate tile cache every tick (terrain may change)
            self._tile_dirty = True
            self._ui_dirty = True

            from agents.agent import Agent

            if Agent.world_model_logger is not None:
                Agent.world_model_logger.log_world_state(self.world.tick, self.world)

    def render(self):
        """Full render frame: tiles → objects → agents → UI overlay."""
        bg = UI_COLORS["background"]
        self.ctx.clear(bg[0] / 255, bg[1] / 255, bg[2] / 255, 1.0)

        # Build instance data (cached tiles, culled objects)
        tile_data = self._build_tile_instances()
        obj_data = self._build_object_instances()
        agent_data = self._build_agent_instances()

        # GPU passes — reuse persistent buffers
        self._render_tiles(tile_data)
        self._render_objects(obj_data)
        self._render_agents(agent_data)

        # UI overlay (only re-uploads texture when content changed)
        self._render_ui_overlay()

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Main loop — same API as PygameRenderer.run()."""
        print("Isometric GPU renderer started!")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  H     - Toggle Help")
        print("  R     - Reset Camera")
        print("  WASD / Arrows - Pan Camera")
        print("  Mouse Wheel   - Zoom")
        print("  Left Drag     - Pan Camera")
        print("  Hover         - Inspect tile + agent inventory")
        print("  ESC   - Quit")

        while self.running:
            self.handle_events()
            self.update()
            self.render()
            frame_time_ms = self.clock.tick(self.target_fps)

            if hasattr(self.world, "adapt_learning_budget"):
                self.world.adapt_learning_budget(frame_time_ms, self.target_fps)

        pygame.quit()
        print(f"\nSimulation ended at tick {self.world.tick}")
