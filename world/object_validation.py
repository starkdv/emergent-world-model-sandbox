"""
Validation, inheritance, and encoding management for object definitions.

Fixes the custom-object failure modes documented in
docs/WORLD_UPGRADE_PROPOSAL.md §8:

- O1: a typo'd section name (``edibel:``) silently registered an object
  with no food component → unknown sections are now errors with
  "did you mean" suggestions.
- O2: a typo'd field name crashed with a context-free TypeError →
  unknown fields are now errors naming the type_id, section, and field.
- O3: dangling cross-references (``grows_into: NONEXISTENT``) were
  accepted at load and exploded mid-simulation → checked at load time.
- O5: hand-picked ``vision_encoding`` floats silently collided, making
  types indistinguishable to every brain → collisions warn, and
  ``vision_encoding: auto`` allocates within per-category bands.
- O7: no reuse → ``extends: <type_id>`` deep-merges a parent definition
  (registered builtins or earlier entries in the same file), so a new
  food is ~8 lines instead of ~60.

All errors are collected and reported together, with the offending
type_id in every message.

Author: Karan Vasa
Date: June 2026
"""

import difflib
import warnings
from dataclasses import fields as dataclass_fields
from typing import Callable, Optional

from world.object_registry import (
    EdibleSpec,
    FertilizerSpec,
    InteractionSpec,
    ObservationSpec,
    PhysicsSpec,
    PlantSpec,
    RenderSpec,
    SeedSpec,
    SpawnSpec,
    TileEffectSpec,
    ToolSpec,
)

# Section name → spec dataclass (single source of truth for the schema)
SECTION_SPECS = {
    "edible": EdibleSpec,
    "seed": SeedSpec,
    "plant": PlantSpec,
    "fertilizer": FertilizerSpec,
    "tool": ToolSpec,
    "physics": PhysicsSpec,
    "interaction": InteractionSpec,
    "tile_effect": TileEffectSpec,
    "observation": ObservationSpec,
    "render": RenderSpec,
    "spawn": SpawnSpec,
}

# Scalar top-level keys (everything else must be a known section)
SCALAR_KEYS = {"display_name", "category", "extends"}

# Fields that must reference a registered type_id
CROSS_REFERENCE_FIELDS = [
    ("seed", "grows_into"),
    ("plant", "produces"),
    ("physics", "decompose_into"),
    ("tile_effect", "spread_type_id"),
]

# vision_encoding bands reserved per category, used by `auto` allocation
# and collision diagnostics. Builtins: berry=1.0, plant=0.75, seed=0.6,
# fertilizer=0.4, sand=0.15 — each sits inside its band.
CATEGORY_ENCODING_BANDS = {
    "food": (0.85, 1.00),
    "plant": (0.65, 0.84),
    "seed": (0.50, 0.64),
    "fertilizer": (0.35, 0.49),
    "object": (0.20, 0.34),
    "structure": (0.20, 0.34),
    "terrain": (0.05, 0.19),
}

# Two types closer than this in encoding space are effectively
# indistinguishable in the agents' vision channel
ENCODING_COLLISION_EPSILON = 0.02


class ObjectValidationError(ValueError):
    """All collected definition errors, raised as one actionable report."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        lines = "\n".join(f"  - {e}" for e in errors)
        super().__init__(f"Invalid object definition(s):\n{lines}")


def _suggest(name: str, options) -> str:
    """Return a ' — did you mean X?' suffix when a close match exists."""
    matches = difflib.get_close_matches(name, list(options), n=1, cutoff=0.6)
    return f" — did you mean '{matches[0]}'?" if matches else ""


def validate_definition_dict(type_id: str, data: dict) -> list[str]:
    """
    Validate one raw definition dict against the schema.

    Args:
        type_id: The object's type_id (for error context)
        data: Raw definition dict (post-extends resolution)

    Returns:
        List of error strings (empty = valid)
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"'{type_id}': definition must be a mapping, got {type(data).__name__}"]

    allowed_top = SCALAR_KEYS | set(SECTION_SPECS)
    for key, value in data.items():
        if key in SCALAR_KEYS:
            continue
        if key not in SECTION_SPECS:
            errors.append(
                f"'{type_id}': unknown section '{key}'{_suggest(key, allowed_top)}"
            )
            continue
        if not isinstance(value, dict):
            errors.append(
                f"'{type_id}.{key}': expected a mapping of fields, "
                f"got {type(value).__name__}"
            )
            continue

        spec_cls = SECTION_SPECS[key]
        allowed_fields = {f.name for f in dataclass_fields(spec_cls)}
        for field_name in value:
            if field_name not in allowed_fields:
                errors.append(
                    f"'{type_id}.{key}': unknown field '{field_name}'"
                    f"{_suggest(field_name, allowed_fields)}"
                )

    # vision_encoding must be a number in [0,1] or the literal "auto"
    obs = data.get("observation", {})
    if isinstance(obs, dict) and "vision_encoding" in obs:
        enc = obs["vision_encoding"]
        if enc != "auto" and not (
            isinstance(enc, (int, float)) and 0.0 <= float(enc) <= 1.0
        ):
            errors.append(
                f"'{type_id}.observation.vision_encoding': must be a number "
                f"in [0, 1] or 'auto', got {enc!r}"
            )

    return errors


def deep_merge(base: dict, override: dict) -> dict:
    """
    Merge ``override`` into ``base`` (nested dicts merged key-wise,
    everything else replaced). Neither input is mutated.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_definitions(
    objects_config: dict,
    get_registered: Callable[[str], Optional[dict]],
) -> tuple[dict, list[str]]:
    """
    Resolve ``extends:`` inheritance for a whole ``objects:`` mapping.

    Parents may be already-registered types (builtins) or earlier entries
    in the same file. The resulting dicts have ``extends`` stripped.

    Args:
        objects_config: type_id → raw definition dict
        get_registered: type_id → registered definition as dict, or None

    Returns:
        (resolved type_id → dict in input order, list of error strings)
    """
    resolved: dict = {}
    errors: list[str] = []

    for type_id, data in objects_config.items():
        if not isinstance(data, dict):
            errors.append(
                f"'{type_id}': definition must be a mapping, "
                f"got {type(data).__name__}"
            )
            continue
        parent_id = data.get("extends")
        if parent_id is None:
            resolved[type_id] = dict(data)
            continue

        if parent_id in resolved:
            base = resolved[parent_id]
        else:
            base = get_registered(parent_id)
        if base is None:
            errors.append(
                f"'{type_id}': extends unknown type '{parent_id}' "
                f"(not a builtin and not defined earlier in this file)"
            )
            resolved[type_id] = {k: v for k, v in data.items() if k != "extends"}
            continue

        child = {k: v for k, v in data.items() if k != "extends"}
        merged = deep_merge(base, child)
        # A child is a NEW type: never inherit the parent's spawn counts
        # unless explicitly given (otherwise extending 'berry' would
        # silently double berry spawns under a different name)
        if "spawn" not in child:
            merged.pop("spawn", None)
        resolved[type_id] = merged

    return resolved, errors


def validate_cross_references(definitions: dict, known_ids: set[str]) -> list[str]:
    """
    Check that every type-reference field points at a known type_id.

    Args:
        definitions: type_id → resolved definition dict (the new batch)
        known_ids: All type_ids that will exist after this batch loads

    Returns:
        List of error strings
    """
    errors: list[str] = []
    for type_id, data in definitions.items():
        for section, fld in CROSS_REFERENCE_FIELDS:
            ref = (
                data.get(section, {}).get(fld, "")
                if isinstance(data.get(section, {}), dict)
                else ""
            )
            if ref and ref not in known_ids:
                errors.append(
                    f"'{type_id}.{section}.{fld}': references unknown type "
                    f"'{ref}'{_suggest(ref, known_ids)}"
                )
    return errors


def allocate_auto_encodings(definitions: dict, taken: dict[str, float]) -> list[str]:
    """
    Replace ``vision_encoding: auto`` with a free value inside the
    type's category band (mutates the definition dicts in place).

    Args:
        definitions: type_id → resolved definition dict
        taken: type_id → encoding already in use (registered types)

    Returns:
        List of error strings (band exhausted / unknown category)
    """
    errors: list[str] = []
    used = list(taken.values())

    for type_id, data in definitions.items():
        obs = data.get("observation")
        if not (isinstance(obs, dict) and obs.get("vision_encoding") == "auto"):
            enc = (obs or {}).get("vision_encoding") if isinstance(obs, dict) else None
            if isinstance(enc, (int, float)):
                used.append(float(enc))
            continue

        category = data.get("category", "object")
        band = CATEGORY_ENCODING_BANDS.get(category)
        if band is None:
            errors.append(
                f"'{type_id}': vision_encoding 'auto' needs a known category "
                f"(got '{category}'){_suggest(category, CATEGORY_ENCODING_BANDS)}"
            )
            continue

        lo, hi = band
        candidate = None
        step = ENCODING_COLLISION_EPSILON
        value = hi  # allocate downward from the top of the band
        while value >= lo - 1e-9:
            if all(abs(value - u) >= step for u in used):
                candidate = round(value, 3)
                break
            value -= step
        if candidate is None:
            errors.append(
                f"'{type_id}': no free vision_encoding left in the "
                f"'{category}' band [{lo}, {hi}] — set one explicitly"
            )
            continue
        obs["vision_encoding"] = candidate
        used.append(candidate)

    return errors


def warn_encoding_collisions(encodings: dict[str, float]) -> list[str]:
    """
    Emit warnings for type pairs whose vision encodings are closer than
    ENCODING_COLLISION_EPSILON — such types are indistinguishable in the
    agents' vision channel.

    Args:
        encodings: type_id → vision_encoding for ALL registered types

    Returns:
        The warning strings (also raised via warnings.warn)
    """
    messages: list[str] = []
    items = sorted(encodings.items(), key=lambda kv: kv[1])
    for (id_a, enc_a), (id_b, enc_b) in zip(items, items[1:]):
        if abs(enc_a - enc_b) < ENCODING_COLLISION_EPSILON:
            msg = (
                f"vision_encoding collision: '{id_a}' ({enc_a}) and "
                f"'{id_b}' ({enc_b}) differ by less than "
                f"{ENCODING_COLLISION_EPSILON} — agents cannot tell them apart"
            )
            messages.append(msg)
            warnings.warn(msg, stacklevel=3)
    return messages
