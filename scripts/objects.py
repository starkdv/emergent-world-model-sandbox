"""
Custom-object toolbox — validate, list, and preview object definitions.

Turns the custom-object authoring loop from "run a full simulation and
discover at tick 800 that nothing spawned" into "validate in one second":

    python scripts/objects.py validate config/custom_objects.yaml
    python scripts/objects.py list [config/custom_objects.yaml]
    python scripts/objects.py preview config/custom_objects.yaml

See docs/OBJECTS_GUIDE.md for the schema reference and cookbook.

Author: Karan Vasa
Date: June 2026
"""

import argparse
import os
import sys
import warnings

import yaml

# Allow running from anywhere: put the repo root on sys.path so the
# agents/world/utils packages resolve (this file lives in scripts/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.object_registry import ObjectRegistry, register_builtin_objects
from world.object_validation import ObjectValidationError


def _load_yaml_objects(path: str) -> dict:
    """Load the ``objects:`` mapping from a YAML file (with clear errors)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "objects" not in data:
        print(f"Error: {path} has no top-level 'objects:' section", file=sys.stderr)
        raise SystemExit(1)
    return data["objects"]


def _components_string(defn) -> str:
    """Compact one-letter component summary, e.g. 'E..P' style flags."""
    parts = []
    if defn.edible:
        parts.append("edible")
    if defn.seed:
        parts.append("seed")
    if defn.plant:
        parts.append("plant")
    if defn.fertilizer:
        parts.append("fertilizer")
    if defn.tile_effect:
        parts.append("tile_effect")
    return "+".join(parts) if parts else "-"


def cmd_validate(path: str) -> int:
    """Validate a file: schema, cross-refs, encodings, spawn dry-run."""
    register_builtin_objects()
    objects = _load_yaml_objects(path)

    collected_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            count = ObjectRegistry.load_from_config(objects)
        except ObjectValidationError as exc:
            print(f"❌ {path}: INVALID\n")
            for err in exc.errors:
                print(f"  ✗ {err}")
            return 1
        collected_warnings = [str(w.message) for w in caught]

    print(f"✅ {path}: {count} definition(s) valid\n")

    for msg in collected_warnings:
        print(f"  ⚠ {msg}")

    # Spawn dry-run: will each type ever appear in a world?
    produced_by = set()
    for defn in ObjectRegistry.all_definitions().values():
        if defn.plant and defn.plant.produces:
            produced_by.add(defn.plant.produces)
        if defn.physics.decompose_into:
            produced_by.add(defn.physics.decompose_into)
        if defn.seed and defn.seed.grows_into:
            produced_by.add(defn.seed.grows_into)

    for type_id in objects:
        defn = ObjectRegistry.get(type_id)
        sp = defn.spawn
        if (
            sp.initial_count <= 0
            and sp.respawn_rate <= 0
            and type_id not in produced_by
        ):
            print(
                f"  ⚠ '{type_id}' will NEVER appear: no spawn.initial_count, "
                f"no spawn.respawn_rate, and nothing produces/decomposes/"
                f"grows into it"
            )
    return 0


def cmd_list(path: str | None) -> int:
    """Print a table of all registered types (builtins + optional file)."""
    register_builtin_objects()
    if path:
        try:
            ObjectRegistry.load_from_config(_load_yaml_objects(path))
        except ObjectValidationError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    rows = []
    for tid, d in sorted(ObjectRegistry.all_definitions().items()):
        spawn = "-"
        if d.spawn.initial_count > 0 or d.spawn.respawn_rate > 0:
            cap = d.spawn.max_count or d.spawn.initial_count
            spawn = f"{d.spawn.initial_count}@{d.spawn.terrain}" + (
                f", respawn {d.spawn.respawn_rate}/t cap {cap}"
                if d.spawn.respawn_rate > 0
                else ""
            )
        rows.append(
            (
                tid,
                d.category,
                _components_string(d),
                f"{d.observation.vision_encoding:.2f}",
                spawn,
            )
        )

    widths = [
        max(
            len(r[i])
            for r in rows + [("type_id", "category", "components", "enc", "spawn")]
        )
        for i in range(5)
    ]
    header = ("type_id", "category", "components", "enc", "spawn")
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    return 0


def cmd_preview(path: str) -> int:
    """Explain, per type, exactly how the simulation will treat it."""
    register_builtin_objects()
    try:
        objects = _load_yaml_objects(path)
        ObjectRegistry.load_from_config(objects)
    except ObjectValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for type_id in objects:
        d = ObjectRegistry.get(type_id)
        print(f"\n━━ {d.display_name} ({type_id}) — category '{d.category}'")
        print(
            f"  agents see: vision_encoding={d.observation.vision_encoding}, "
            f"value channel='{d.observation.value_source}'"
        )
        if d.edible:
            life = (
                int(d.edible.freshness / d.physics.decay_rate)
                if d.physics.decay_rate > 0
                else "∞ (no decay)"
            )
            print(
                f"  EAT: +{d.edible.calories} energy × freshness"
                + (f", toxicity {d.edible.toxicity}" if d.edible.toxicity else "")
                + f" | spoils in ~{life} ticks"
            )
            if d.physics.decompose_into:
                print(
                    f"  on full decay: {d.physics.decompose_chance:.0%} chance "
                    f"→ '{d.physics.decompose_into}', returns "
                    f"{d.physics.nutrient_return} fertility"
                )
        if d.seed:
            print(
                f"  plant via USE: germinates after {d.seed.grow_time} ticks "
                f"on tiles with fertility ≥ {d.seed.required_fertility} and "
                f"moisture ≥ {d.seed.required_moisture} → "
                f"'{d.seed.grows_into}'; rots after {d.seed.max_age} ticks"
            )
        if d.plant:
            print(
                f"  plant: matures at {d.plant.mature_age} ticks, lives "
                f"{d.plant.max_age}; produces '{d.plant.produces}' at "
                f"{d.plant.spawn_rate}/tick when mature"
            )
        if d.tile_effect:
            te = d.tile_effect
            print(
                f"  tile effect: growth ×{te.growth_multiplier}, "
                f"germination ×{te.germination_multiplier}, "
                f"production ×{te.spawn_rate_multiplier}"
                + (
                    f"; spreads (p={te.spread_chance}/tick after "
                    f"{te.spread_interval} ticks)"
                    if te.spread_type_id
                    else ""
                )
            )
        ia = d.interaction
        print(
            f"  interaction: pickable={ia.pickable}, usable={ia.usable}, "
            f"passable={ia.passable}, blocks_growth={ia.blocks_growth}"
        )
        sp = d.spawn
        if sp.initial_count > 0 or sp.respawn_rate > 0:
            print(
                f"  spawn: {sp.initial_count} at start on '{sp.terrain}' tiles"
                + (
                    f"; respawns at {sp.respawn_rate}/tick up to "
                    f"{sp.max_count or sp.initial_count}"
                    if sp.respawn_rate > 0
                    else ""
                )
            )
        else:
            print("  spawn: ⚠ never placed directly (must be produced by a chain)")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate, list, and preview custom object definitions"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Validate a YAML objects file")
    p_val.add_argument("file")
    p_list = sub.add_parser("list", help="Table of all registered types")
    p_list.add_argument("file", nargs="?", default=None)
    p_prev = sub.add_parser("preview", help="Explain how each type will behave")
    p_prev.add_argument("file")

    args = parser.parse_args()
    if args.command == "validate":
        return cmd_validate(args.file)
    if args.command == "list":
        return cmd_list(args.file)
    return cmd_preview(args.file)


if __name__ == "__main__":
    sys.exit(main())
