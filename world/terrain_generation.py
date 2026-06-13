"""
Heightmap terrain generation — living terrain for the World upgrade (W2).

The legacy generator places terrain by a uniform random shuffle: no spatial
coherence, no elevation, no rivers, no biomes (proposal §1, P1/P2). This
module replaces that with an **elevation-first** generator built on pure
NumPy value-noise — no new dependencies:

    elevation(x, y)  smoothed value-noise surface in [0, 1]
        │
        ├─ high  →  ROCK ridges (mountains, impassable)
        ├─ low   →  WATER basins, and rivers that FLOW DOWNHILL from peaks
        │            into them (steepest-descent tracing)
        └─ mid   →  SOIL / SAND, chosen by a moisture field derived from
                     elevation + distance-to-water

Moisture and fertility are then seeded from the geography so that
**river corridors are fertile** (the W2 acceptance criterion) and deserts
(SAND) fall where the land is high and dry. Elevation is stored on every
tile so the rest of the world can use it: water flows downhill, slopes cost
movement energy (W2), and a future 3D track can read the surface directly
(proposal §12). The elevation field is *not* added to the agent observation
yet — that genome break is reserved for W4 (Observation v2).

Determinism: everything is driven by a single seeded ``numpy.random.Generator``,
so a given (width, height, seed) always produces the same world.

Author: Karan Vasa
Date: June 2026
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from world.tiles import TerrainType


@dataclass
class HeightmapConfig:
    """Tunables for the heightmap generator (all have sane defaults)."""

    feature_scale: int = 12  # coarse-noise cell size in tiles (bigger = smoother)
    octaves: int = 4  # layered noise detail levels
    persistence: float = 0.5  # amplitude falloff per octave
    rock_ratio: float = 0.20  # fraction of highest land that becomes mountain rock
    water_ratio: float = 0.10  # total water fraction (lakes + rivers)
    sand_ratio: float = 0.05  # fraction of the driest remaining land → desert sand
    river_sources: int = 6  # how many downhill rivers to trace from peaks
    fertility_range: Tuple[float, float] = (0.3, 1.0)
    moisture_range: Tuple[float, float] = (0.2, 0.8)


@dataclass
class GeneratedTerrain:
    """Per-tile grids produced by the generator (row-major, [y][x])."""

    terrain: List[List[TerrainType]]
    elevation: np.ndarray  # float32 [height, width] in [0, 1]
    fertility: np.ndarray  # float32 [height, width] in [0, 1]
    moisture: np.ndarray  # float32 [height, width] in [0, 1]
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Value-noise elevation field
# ---------------------------------------------------------------------------


def _smooth_grid(rng: np.random.Generator, h: int, w: int, scale: int) -> np.ndarray:
    """
    One octave of value noise: random values on a coarse lattice, bilinearly
    upsampled to (h, w). Pure NumPy.
    """
    scale = max(2, int(scale))
    ch = h // scale + 2
    cw = w // scale + 2
    coarse = rng.random((ch, cw), dtype=np.float64)

    ys = np.linspace(0, ch - 1, h)
    xs = np.linspace(0, cw - 1, w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, ch - 1)
    x1 = np.minimum(x0 + 1, cw - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]

    # Smoothstep weights for visually smoother interpolation
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)

    top = coarse[np.ix_(y0, x0)] * (1 - fx) + coarse[np.ix_(y0, x1)] * fx
    bot = coarse[np.ix_(y1, x0)] * (1 - fx) + coarse[np.ix_(y1, x1)] * fx
    return top * (1 - fy) + bot * fy


def generate_elevation(
    width: int, height: int, rng: np.random.Generator, cfg: HeightmapConfig
) -> np.ndarray:
    """
    Fractal value noise (summed octaves), normalised to [0, 1].
    """
    field_ = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    total = 0.0
    scale = cfg.feature_scale
    for _ in range(max(1, cfg.octaves)):
        field_ += amplitude * _smooth_grid(rng, height, width, scale)
        total += amplitude
        amplitude *= cfg.persistence
        scale = max(2, scale // 2)
    field_ /= total

    lo, hi = float(field_.min()), float(field_.max())
    if hi - lo < 1e-9:
        return np.full((height, width), 0.5, dtype=np.float32)
    return ((field_ - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# Rivers: steepest-descent tracing from high points
# ---------------------------------------------------------------------------


def carve_rivers(
    elevation: np.ndarray,
    is_water: np.ndarray,
    rng: np.random.Generator,
    n_sources: int,
    max_river_tiles: int,
) -> int:
    """
    Trace rivers downhill from high points until they reach existing water or
    a map edge, marking the path in ``is_water`` (mutated in place).

    Returns the number of river tiles carved. Stops once ``max_river_tiles``
    is reached so the total water budget is respected.
    """
    h, w = elevation.shape
    carved = 0

    # Candidate sources: the highest land tiles, sampled for spread
    flat = elevation.flatten()
    n_candidates = max(n_sources * 8, 1)
    high_idx = np.argpartition(flat, -n_candidates)[-n_candidates:]
    rng.shuffle(high_idx)
    sources = high_idx[:n_sources]

    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for src in sources:
        if carved >= max_river_tiles:
            break
        y, x = divmod(int(src), w)
        visited = set()
        steps = 0
        max_steps = h * w  # hard safety bound
        while steps < max_steps:
            steps += 1
            if (y, x) in visited:
                break
            visited.add((y, x))
            if not is_water[y, x]:
                is_water[y, x] = True
                carved += 1
                if carved >= max_river_tiles:
                    break

            # Reached an edge → river exits the map
            if x == 0 or y == 0 or x == w - 1 or y == h - 1:
                break

            # Step to the lowest neighbour (steepest descent)
            best = None
            best_elev = elevation[y, x]
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                    if is_water[ny, nx]:
                        best = (ny, nx)  # flow into existing water and stop
                        best_elev = -1.0
                        break
                    if elevation[ny, nx] < best_elev:
                        best_elev = elevation[ny, nx]
                        best = (ny, nx)
            if best is None:
                break  # local minimum (a lake forms here); stop
            if best_elev < 0.0:
                # flowed into water
                break
            y, x = best

    return carved


# ---------------------------------------------------------------------------
# Distance-to-water (multi-source BFS) → moisture field
# ---------------------------------------------------------------------------


def _distance_to_water(is_water: np.ndarray) -> np.ndarray:
    """Manhattan BFS distance from every tile to the nearest water tile."""
    h, w = is_water.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    q: deque = deque()
    ys, xs = np.where(is_water)
    for y, x in zip(ys.tolist(), xs.tolist()):
        dist[y, x] = 0
        q.append((y, x))
    while q:
        y, x = q.popleft()
        d = dist[y, x]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] < 0:
                dist[ny, nx] = d + 1
                q.append((ny, nx))
    if (dist < 0).any():  # no water at all → uniform "far"
        dist[dist < 0] = max(h, w)
    return dist


# ---------------------------------------------------------------------------
# Full generation
# ---------------------------------------------------------------------------


def generate_terrain(
    width: int, height: int, seed: int, cfg: HeightmapConfig
) -> GeneratedTerrain:
    """
    Generate a coherent heightmap world: elevation, terrain types (with
    mountains + downhill rivers + biomes), and geography-derived moisture
    and fertility fields.

    Ratios are honoured via elevation/moisture quantiles, so the requested
    rock/water/sand fractions are met regardless of the noise realisation.
    """
    rng = np.random.default_rng(seed)
    elevation = generate_elevation(width, height, rng, cfg)

    # --- Mountains: the highest `rock_ratio` of tiles become rock ---
    rock_q = np.quantile(elevation, 1.0 - cfg.rock_ratio) if cfg.rock_ratio > 0 else 2.0
    is_rock = elevation >= rock_q

    # --- Lakes: the lowest portion of the water budget settles in basins ---
    lake_fraction = cfg.water_ratio * 0.6
    is_water = np.zeros((height, width), dtype=bool)
    if lake_fraction > 0:
        water_q = np.quantile(elevation, lake_fraction)
        is_water = (elevation <= water_q) & (~is_rock)

    # --- Rivers: trace the remaining water budget downhill from peaks ---
    total_tiles = width * height
    water_budget = int(round(cfg.water_ratio * total_tiles))
    river_budget = max(0, water_budget - int(is_water.sum()))
    rivers_carved = 0
    if cfg.river_sources > 0 and river_budget > 0:
        river_only = np.zeros_like(is_water)
        rivers_carved = carve_rivers(
            elevation, river_only, rng, cfg.river_sources, river_budget
        )
        # Rivers never overwrite mountains
        river_only &= ~is_rock
        is_water |= river_only

    # --- Moisture field: wetter low + near water, drier high + far ---
    dist = _distance_to_water(is_water).astype(np.float32)
    dmax = float(dist.max()) or 1.0
    near_water = 1.0 - (dist / dmax)  # 1 at water, →0 far away
    mlo, mhi = cfg.moisture_range
    moisture = mlo + (mhi - mlo) * (0.55 * near_water + 0.45 * (1.0 - elevation))
    moisture = np.clip(moisture, 0.0, 1.0).astype(np.float32)
    moisture[is_water] = 1.0

    # --- Desert sand: the driest `sand_ratio` of the remaining land ---
    land = (~is_rock) & (~is_water)
    is_sand = np.zeros((height, width), dtype=bool)
    if cfg.sand_ratio > 0 and land.any():
        land_moist = moisture[land]
        sand_q = np.quantile(land_moist, cfg.sand_ratio)
        is_sand = land & (moisture <= sand_q)

    # --- Fertility: rich in river corridors, poor when dry/sandy ---
    flo, fhi = cfg.fertility_range
    fertility = flo + (fhi - flo) * (0.6 * near_water + 0.4 * moisture)
    fertility = np.clip(fertility, 0.0, 1.0).astype(np.float32)
    fertility[is_rock] = 0.0
    fertility[is_water] = 0.0
    # Sand keeps low fertility/moisture (the sand TileEffect clamps it anyway)
    rng_sand = rng.random((height, width)).astype(np.float32)
    fertility = np.where(is_sand, rng_sand * 0.05, fertility)
    moisture = np.where(is_sand & (~is_water), rng_sand * 0.05, moisture)

    # --- Assemble the terrain-type grid ---
    terrain: List[List[TerrainType]] = []
    for y in range(height):
        row = []
        for x in range(width):
            if is_rock[y, x]:
                row.append(TerrainType.ROCK)
            elif is_water[y, x]:
                row.append(TerrainType.WATER)
            elif is_sand[y, x]:
                row.append(TerrainType.SAND)
            else:
                row.append(TerrainType.SOIL)
        terrain.append(row)

    stats = {
        "rock": int(is_rock.sum()),
        "water": int(is_water.sum()),
        "sand": int(is_sand.sum()),
        "soil": int(total_tiles - is_rock.sum() - is_water.sum() - is_sand.sum()),
        "rivers_carved": int(rivers_carved),
        "elevation_min": float(elevation.min()),
        "elevation_max": float(elevation.max()),
    }

    return GeneratedTerrain(
        terrain=terrain,
        elevation=elevation,
        fertility=fertility,
        moisture=moisture,
        stats=stats,
    )
