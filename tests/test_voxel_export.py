"""
Tests for the F1 offline voxel export (render/voxel_export.py).

Validates the snapshot → colored PLY mesh: a well-formed header with matching
counts, exposed-face culling (a flat world emits fewer faces than a naive
6-faces-per-column upper bound), entity cubes are added, and the file parses
back to the declared vertex/face counts.

Author: Karan Vasa
"""

import os

import numpy as np
import pytest

from render.state_bridge import world_snapshot
from render.voxel_export import export_ply, snapshot_to_mesh
from world.world import World
from world.tiles import TerrainType
from world.objects import WorldObject, EdibleComponent
from world.object_registry import ObjectRegistry, register_builtin_objects


@pytest.fixture(autouse=True)
def _registry():
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    ObjectRegistry._definitions.clear()


def _flat_world(n=6, elev=0.5):
    w = World(
        n,
        n,
        seed=1,
        soil_ratio=1.0,
        rock_ratio=0.0,
        water_ratio=0.0,
        sand_ratio=0.0,
        parallel=False,
    )
    for y in range(n):
        for x in range(n):
            t = w.get_tile(x, y)
            t.terrain_type = TerrainType.SOIL
            t.elevation = elev
            t.fertility = 0.5
            t.moisture = 0.5
    return w


def _parse_ply_counts(text):
    nv = nf = None
    for line in text.splitlines():
        if line.startswith("element vertex"):
            nv = int(line.split()[-1])
        elif line.startswith("element face"):
            nf = int(line.split()[-1])
        elif line == "end_header":
            break
    return nv, nf


def test_ply_header_counts_match_body(tmp_path):
    w = _flat_world()
    snap = world_snapshot(w)
    path = os.path.join(str(tmp_path), "w.ply")
    export_ply(snap, path)
    text = open(path).read()
    assert text.startswith("ply\nformat ascii 1.0")
    nv, nf = _parse_ply_counts(text)
    # count the actual body lines after end_header
    body = text.split("end_header\n", 1)[1].splitlines()
    vert_lines = body[:nv]
    face_lines = body[nv : nv + nf]
    assert len(vert_lines) == nv
    assert len(face_lines) == nf
    # every face line is a triangle ("3 a b c")
    assert all(ln.startswith("3 ") for ln in face_lines)


def test_face_culling_on_flat_world(tmp_path):
    # A flat world: interior columns share side faces with equal-height
    # neighbors, so those sides are culled. The mesh should be far smaller
    # than the naive 6 faces/column * 2 tris.
    from render.voxel_export import MAX_H

    n = 6
    w = _flat_world(n=n, elev=0.5)
    snap = world_snapshot(w)
    mesh = snapshot_to_mesh(snap)
    # No-cull upper bound: every column emits top + all 4 sides at full height.
    naive_upper = n * n * (2 + 4 * MAX_H * 2)
    assert len(mesh.faces) < naive_upper  # interior side faces are culled
    # top faces alone = 36 columns * 2 tris = 72
    assert len(mesh.faces) >= n * n * 2


def test_taller_column_exposes_sides(tmp_path):
    # One tall column among short ones must emit side faces (cliffs).
    w = _flat_world(n=5, elev=0.1)
    w.get_tile(2, 2).elevation = 1.0  # a spike
    snap = world_snapshot(w)
    mesh_spike = snapshot_to_mesh(snap)

    w2 = _flat_world(n=5, elev=0.1)
    snap2 = world_snapshot(w2)
    mesh_flat = snapshot_to_mesh(snap2)

    # the spike world has strictly more faces (exposed cliff sides)
    assert len(mesh_spike.faces) > len(mesh_flat.faces)


def test_entities_add_geometry(tmp_path):
    w = _flat_world()
    base = snapshot_to_mesh(world_snapshot(w))

    o = WorldObject(x=2, y=2)
    o.type_id = "berry"
    o.add_component(EdibleComponent(calories=20.0, freshness=1.0))
    w.add_object(o)
    with_obj = snapshot_to_mesh(world_snapshot(w))

    # a berry cube adds 6 faces * 2 tris = 12 triangles
    assert len(with_obj.faces) == len(base.faces) + 12


def test_vertices_carry_color(tmp_path):
    w = _flat_world()
    snap = world_snapshot(w)
    mesh = snapshot_to_mesh(snap)
    # each vertex is (x,y,z,r,g,b) with bytes in range
    for v in mesh.verts[:50]:
        assert len(v) == 6
        assert all(0 <= c <= 255 for c in v[3:])
