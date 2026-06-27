"""
Offline voxel export (Frontend phase F1).

Turn a render snapshot — from a live world or a W6b checkpoint — into a colored
voxel **mesh** written as ASCII **PLY**. PLY loads in Blender, MeshLab, and most
online 3D viewers with no plugins, so this is the cheap "fly around a saved
frame with zero networking" milestone, and it validates the same
elevation→column mapping the live client uses.

The mesh is built from the F0 snapshot: each terrain tile becomes a column box
(height = quantized elevation), each agent/object a small colored cube on the
surface. Faces between two solid terrain columns are culled (only exposed faces
are emitted), so the file stays small.

CLI::

    python -m render.voxel_export --out world.ply                 # fresh demo world
    python -m render.voxel_export --checkpoint run.pkl --out w.ply  # a saved run

Author: Karan Vasa
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from render.state_bridge import decode_grid

MAX_H = 18  # matches the web client's column height for elevation 1.0

# Terrain code → RGB (0–255), aligned with the web client's palette.
_TERRAIN_RGB = {
    0: (79, 157, 58),  # soil / grass
    1: (138, 141, 146),  # rock
    2: (47, 111, 176),  # water
    3: (215, 196, 137),  # sand
}

# Unit-cube corner offsets and the six faces (as quads → two triangles each).
_CORNERS = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
]
# face name -> (corner indices ccw, neighbor delta in (x,y,z))
_FACES = {
    "nx": ((0, 3, 7, 4), (-1, 0, 0)),
    "px": ((1, 5, 6, 2), (1, 0, 0)),
    "ny": ((0, 4, 5, 1), (0, -1, 0)),
    "py": ((3, 2, 6, 7), (0, 1, 0)),
    "nz": ((0, 1, 2, 3), (0, 0, -1)),
    "pz": ((4, 7, 6, 5), (0, 0, 1)),
}


class _MeshBuilder:
    """Accumulates colored triangles, deduplicating nothing (simple + fast)."""

    def __init__(self):
        self.verts: List[Tuple[float, float, float, int, int, int]] = []
        self.faces: List[Tuple[int, int, int]] = []

    def add_quad(self, p, corners, rgb):
        # p = (x,y,z) min corner; corners = 4 corner indices of the face
        base = len(self.verts)
        for ci in corners:
            cx, cy, cz = _CORNERS[ci]
            self.verts.append((p[0] + cx, p[1] + cy, p[2] + cz, rgb[0], rgb[1], rgb[2]))
        # two triangles: (0,1,2) and (0,2,3)
        self.faces.append((base + 0, base + 1, base + 2))
        self.faces.append((base + 0, base + 2, base + 3))

    def add_cube(self, p, size, rgb, faces=None):
        # a scaled cube at min-corner p with edge `size`; emit the named faces
        names = faces if faces is not None else list(_FACES.keys())
        for name in names:
            corners, _ = _FACES[name]
            base = len(self.verts)
            for ci in corners:
                cx, cy, cz = _CORNERS[ci]
                self.verts.append(
                    (
                        p[0] + cx * size,
                        p[1] + cy * size,
                        p[2] + cz * size,
                        rgb[0],
                        rgb[1],
                        rgb[2],
                    )
                )
            self.faces.append((base + 0, base + 1, base + 2))
            self.faces.append((base + 0, base + 2, base + 3))

    def to_ply(self) -> str:
        lines = [
            "ply",
            "format ascii 1.0",
            "comment Emergent World voxel export (F1)",
            f"element vertex {len(self.verts)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"element face {len(self.faces)}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        for x, y, z, r, g, b in self.verts:
            lines.append(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}")
        for a, b, c in self.faces:
            lines.append(f"3 {a} {b} {c}")
        return "\n".join(lines) + "\n"


def _column_heights(elevation: np.ndarray) -> np.ndarray:
    """Quantized elevation → integer column heights (>=1)."""
    h = np.maximum(1, np.rint(elevation.astype(np.float32) / 255.0 * MAX_H)).astype(int)
    return h


def snapshot_to_mesh(snapshot: dict) -> _MeshBuilder:
    """Build the colored voxel mesh from an F0 snapshot."""
    terr = snapshot["terrain"]
    w, h = terr["width"], terr["height"]
    elevation = decode_grid(terr["elevation"])
    terrain = decode_grid(terr["terrain"])
    heights = _column_heights(elevation)

    mb = _MeshBuilder()

    # Terrain columns with exposed-face culling against neighbor column heights.
    # We treat a column as solid from z=0..height; a side face at height level is
    # exposed if the neighbor column is shorter. Top is always exposed; bottom
    # never (sits on z=0). To keep it simple and robust we emit, per column:
    #   - the top quad (pz) at z=height
    #   - side quads only up to the exposed portion vs each neighbor
    for y in range(h):
        for x in range(w):
            ch = int(heights[y, x])
            code = int(terrain[y, x])
            rgb = _TERRAIN_RGB.get(code, _TERRAIN_RGB[0])
            # top face (one quad at z=ch)
            mb.add_quad((x, y, ch), _FACES["pz"][0], rgb)
            # side faces: emit a stacked quad for each exposed unit on each side
            for name, (corners, (dx, dy, dz)) in _FACES.items():
                if name in ("pz", "nz"):
                    continue
                nx, ny = x + dx, y + dy
                neigh_h = 0
                if 0 <= nx < w and 0 <= ny < h:
                    neigh_h = int(heights[ny, nx])
                # expose levels neigh_h..ch-1 (the part of this column taller
                # than the neighbor)
                for z in range(neigh_h, ch):
                    base = len(mb.verts)
                    for ci in corners:
                        cx, cy, cz = _CORNERS[ci]
                        mb.verts.append(
                            (x + cx, y + cy, z + cz, rgb[0], rgb[1], rgb[2])
                        )
                    mb.faces.append((base + 0, base + 1, base + 2))
                    mb.faces.append((base + 0, base + 2, base + 3))

    # Entities as small cubes on the surface.
    def surf(tx, ty):
        if 0 <= tx < w and 0 <= ty < h:
            return int(heights[ty, tx])
        return 1

    for o in snapshot.get("objects", []):
        rgb = _object_rgb(o)
        s = 0.4
        ox, oy = int(o["x"]), int(o["y"])
        mb.add_cube((ox + 0.3, surf(ox, oy), oy + 0.3), s, rgb)

    for a in snapshot.get("agents", []):
        if a.get("alive") is False:
            continue
        rgb = _lineage_rgb(int(a.get("lineage", -1)))
        s = 0.6
        ax, ay = int(a["x"]), int(a["y"])
        mb.add_cube((ax + 0.2, surf(ax, ay), ay + 0.2), s, rgb)

    return mb


def _object_rgb(o: dict):
    cat = (o.get("category") or "").lower()
    tid = (o.get("type_id") or "").lower()
    if "night" in tid or "toxic" in tid:
        return (155, 48, 192)
    if "food" in cat or "berry" in tid or "fruit" in tid:
        return (226, 59, 59)
    if "plant" in cat or "tree" in tid or "shrub" in tid:
        return (47, 143, 58)
    if "seed" in cat:
        return (205, 176, 121)
    if "hazard" in cat or "thorn" in tid:
        return (58, 42, 42)
    return (176, 176, 176)


def _lineage_rgb(lineage: int):
    if lineage < 0:
        return (230, 230, 230)
    import colorsys

    hue = ((lineage * 47) % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.55)
    return (int(r * 255), int(g * 255), int(b * 255))


def export_ply(snapshot: dict, path: str) -> str:
    """Write the snapshot's voxel mesh to ``path`` as ASCII PLY. Returns path."""
    mesh = snapshot_to_mesh(snapshot)
    with open(path, "w", encoding="utf-8") as f:
        f.write(mesh.to_ply())
    return path


def main(argv=None):
    import argparse

    from render.state_bridge import world_snapshot

    p = argparse.ArgumentParser(description="Export a world frame to PLY (F1)")
    p.add_argument("--out", default="world.ply")
    p.add_argument("--checkpoint", default=None, help="W6b checkpoint to export")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--agents", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    if args.checkpoint:
        from world.checkpoint import load_state

        world = load_state(args.checkpoint, config={"brain": {"version": 2}})
        snap = world_snapshot(world)
    else:
        from render.sim_session import build_demo_world

        session = build_demo_world(
            width=args.width, height=args.height, n_agents=args.agents, seed=args.seed
        )
        snap = session.snapshot()

    path = export_ply(snap, args.out)
    mesh = snapshot_to_mesh(snap)
    print(f"Wrote {path}: {len(mesh.verts)} vertices, {len(mesh.faces)} faces")


if __name__ == "__main__":
    main()
