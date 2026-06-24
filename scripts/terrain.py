"""
Terrain preview — visualise the heightmap generator as ASCII (World W2).

See, in one second, what a seed produces before running a full simulation:

    python scripts/terrain.py preview --seed 42
    python scripts/terrain.py preview --width 80 --height 40 --water 0.12
    python scripts/terrain.py preview --config config/default.yaml

Glyphs:  ^ rock/mountain   ~ water (lakes & rivers)   : sand   . soil
Use --elevation to print the raw height field (0–9) instead of terrain.

Author: Karan Vasa
Date: June 2026
"""

import argparse
import os
import sys

# Run from anywhere: put the repo root on sys.path (this file lives in scripts/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.terrain_generation import HeightmapConfig, generate_terrain  # noqa: E402
from world.tiles import TerrainType  # noqa: E402

GLYPH = {
    TerrainType.ROCK: "^",
    TerrainType.WATER: "~",
    TerrainType.SAND: ":",
    TerrainType.SOIL: ".",
}


def _load_terrain_cfg(path):
    """Pull the terrain block from a YAML config (best-effort)."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("terrain", {}) or {}


def cmd_preview(args) -> int:
    terr_cfg = _load_terrain_cfg(args.config) if args.config else {}
    hm = terr_cfg.get("heightmap", {}) or {}

    cfg = HeightmapConfig(
        feature_scale=int(hm.get("feature_scale", 12)),
        octaves=int(hm.get("octaves", 4)),
        persistence=float(hm.get("persistence", 0.5)),
        rock_ratio=(
            args.rock if args.rock is not None else terr_cfg.get("rock_ratio", 0.20)
        ),
        water_ratio=(
            args.water if args.water is not None else terr_cfg.get("water_ratio", 0.10)
        ),
        sand_ratio=(
            args.sand if args.sand is not None else terr_cfg.get("sand_ratio", 0.05)
        ),
        river_sources=int(hm.get("river_sources", 6)),
    )

    res = generate_terrain(args.width, args.height, args.seed, cfg)
    total = args.width * args.height
    s = res.stats
    print(
        f"seed={args.seed}  {args.width}x{args.height}  "
        f"rock={s['rock']/total:.1%}  water={s['water']/total:.1%}  "
        f"sand={s['sand']/total:.1%}  soil={s['soil']/total:.1%}  "
        f"rivers_carved={s['rivers_carved']}"
    )
    print()

    if args.elevation:
        for y in range(args.height):
            print(
                "".join(
                    str(min(9, int(res.elevation[y, x] * 10)))
                    for x in range(args.width)
                )
            )
    else:
        for row in res.terrain:
            print("".join(GLYPH[t] for t in row))
        print("\nlegend:  ^ rock   ~ water   : sand   . soil")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview the heightmap terrain generator as ASCII"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preview", help="Render an ASCII map for a seed")
    p.add_argument("--width", type=int, default=80)
    p.add_argument("--height", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rock", type=float, default=None, help="rock ratio override")
    p.add_argument("--water", type=float, default=None, help="water ratio override")
    p.add_argument("--sand", type=float, default=None, help="sand ratio override")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="read terrain.heightmap from a YAML config",
    )
    p.add_argument(
        "--elevation", action="store_true", help="print the height field (0-9) instead"
    )

    args = parser.parse_args()
    if args.command == "preview":
        return cmd_preview(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
