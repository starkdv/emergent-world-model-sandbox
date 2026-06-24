"""
Coarse-cell spatial index for fast radius queries (World upgrade W6a).

The reward shaper and perception repeatedly ask "where is the nearest edible
object to (x, y)?" The naive answer scans every tile in a window — up to
21×21 = 441 tiles per call, several calls per agent per tick — and most of
those tiles are empty. That O(R²) cost is the single biggest tick-rate sink
once a world has many agents (SUGGESTIONS §4.2 / WORLD_UPGRADE_PROPOSAL W6a).

``SpatialIndex`` buckets tracked object ids into coarse square cells keyed by
``(x // cell_size, y // cell_size)``. A radius query then visits only the few
cells overlapping the query box and the objects inside them — O(objects
nearby) instead of O(tiles in window). The index stores *only* objects whose
membership is stable while they sit in the world (edible items: a berry stays
edible until it is removed or picked up), so maintenance is a handful of
add/remove/move hooks rather than a per-tile mirror.

The index is an *acceleration structure*, never the source of truth: callers
still resolve each candidate against the live world (object still present,
still on its tile) so results are identical to a direct tile scan. Disabling
it (``performance.spatial_index: false``) falls back to the tile scan.

Author: Karan Vasa
"""

from typing import Dict, Iterator, Set, Tuple


class SpatialIndex:
    """Bucket object ids into coarse cells for O(nearby) radius queries."""

    def __init__(self, width: int, height: int, cell_size: int = 8):
        """
        Args:
            width: World width in tiles (for bounds; queries are clamped).
            height: World height in tiles.
            cell_size: Square cell edge in tiles. Larger = fewer, fatter
                buckets (cheaper to maintain, more candidates per query);
                smaller = more buckets (tighter queries). 8 is a good default
                for the 5–10 tile scan radii the callers use.
        """
        self.width = int(width)
        self.height = int(height)
        self.cell_size = max(1, int(cell_size))
        self._cells: Dict[Tuple[int, int], Set[int]] = {}
        self._pos: Dict[int, Tuple[int, int]] = {}

    # -- maintenance --------------------------------------------------------

    def _ckey(self, x: int, y: int) -> Tuple[int, int]:
        return (x // self.cell_size, y // self.cell_size)

    def add(self, obj_id: int, x: int, y: int) -> None:
        """Track ``obj_id`` at (x, y) (idempotent — re-adds move the entry)."""
        if obj_id in self._pos:
            self.remove(obj_id)
        key = self._ckey(x, y)
        self._cells.setdefault(key, set()).add(obj_id)
        self._pos[obj_id] = (x, y)

    def remove(self, obj_id: int) -> None:
        """Stop tracking ``obj_id`` (no-op if not tracked)."""
        pos = self._pos.pop(obj_id, None)
        if pos is None:
            return
        key = self._ckey(pos[0], pos[1])
        bucket = self._cells.get(key)
        if bucket is not None:
            bucket.discard(obj_id)
            if not bucket:
                del self._cells[key]

    def move(self, obj_id: int, x: int, y: int) -> None:
        """Update a tracked object's position (cheap if it stays in-cell)."""
        old = self._pos.get(obj_id)
        if old is not None and self._ckey(*old) == self._ckey(x, y):
            self._pos[obj_id] = (x, y)
            return
        self.add(obj_id, x, y)

    def clear(self) -> None:
        self._cells.clear()
        self._pos.clear()

    # -- queries ------------------------------------------------------------

    def query_box(self, x0: int, y0: int, x1: int, y1: int) -> Iterator[int]:
        """
        Yield every tracked object id whose cell overlaps the inclusive box
        [x0, x1] × [y0, y1]. May yield ids just outside the box (cell
        granularity) — callers filter by exact position.
        """
        cs = self.cell_size
        cx0, cy0 = x0 // cs, y0 // cs
        cx1, cy1 = x1 // cs, y1 // cs
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                bucket = self._cells.get((cx, cy))
                if bucket:
                    # Copy so callers may mutate the world mid-iteration.
                    yield from tuple(bucket)

    def position_of(self, obj_id: int):
        """Return the tracked (x, y) for ``obj_id`` or None."""
        return self._pos.get(obj_id)

    def __len__(self) -> int:
        return len(self._pos)

    def __contains__(self, obj_id: int) -> bool:
        return obj_id in self._pos
