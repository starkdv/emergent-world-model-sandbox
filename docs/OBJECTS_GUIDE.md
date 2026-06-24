# Custom Objects Guide

Define new world objects — foods, poisons, seeds, plants, fertilizers,
terrain effects — in YAML, without touching code.

## The 60-second loop

```bash
# 1. Write (or edit) a YAML file — minimal example below
# 2. Validate instantly (schema, references, encodings, "will it spawn?")
python scripts/objects.py validate config/my_objects.yaml
# 3. See exactly how the simulation will treat each type
python scripts/objects.py preview config/my_objects.yaml
# 4. Run it
python main.py --gui --objects config/my_objects.yaml
```

`main.py --objects` runs the same validation and **refuses to start** on
errors, so a broken file can no longer half-load silently.

## Minimal example — `extends` does the boilerplate

Every type can inherit a builtin (`berry`, `berry_seed`, `berry_plant`,
`fertilizer`, `sand`) or an earlier entry in the same file, then override
only what differs:

```yaml
objects:
  golden_apple:
    extends: berry                      # food behaviour inherited
    edible: { calories: 60.0 }          # 3x a berry
    observation: { vision_encoding: 0.95 }
    spawn: { initial_count: 10, respawn_rate: 0.005, max_count: 10 }
```

Notes on `extends`:
- Nested sections deep-merge: you can override one field of `physics`
  and keep the rest.
- `spawn` is **never inherited** — a new type must say how it appears
  (otherwise extending `berry` would silently double berry spawns).

## The three rules people trip on

1. **Spawn or it never exists.** Give `spawn.initial_count` and/or
   `spawn.respawn_rate`, or make the type part of a chain (something
   `produces`/`decompose_into`/`grows_into` it). The validator warns when
   none of these hold.
2. **Distinct `vision_encoding` or agents can't see the difference.**
   Two types closer than 0.02 are indistinguishable in every brain's
   vision channel (warning emitted). Use `vision_encoding: auto` to get
   a free slot in your category's reserved band:
   food 0.85–1.0 · plant 0.65–0.84 · seed 0.50–0.64 ·
   fertilizer 0.35–0.49 · object/structure 0.20–0.34 · terrain 0.05–0.19.
3. **Chains must close.** `grows_into`, `produces`, `decompose_into`, and
   `spread_type_id` must name existing types — dangling references are
   load-time errors now, not tick-800 crashes.

## Schema reference

Top-level keys per type: `display_name`, `category`
(`food|seed|plant|fertilizer|terrain|structure|object`), `extends`, and
the sections below. Everything is optional and has working defaults;
unknown sections/fields are errors with "did you mean" hints.

| Section | Fields (defaults) |
|---|---|
| `edible` | `calories` (20.0) · `toxicity` (0.0 — wiring into EAT lands in phase W3) · `freshness` (1.0) |
| `seed` | `grows_into` ("") · `grow_time` (50) · `required_fertility` (0.3) · `required_moisture` (0.2) · `max_age` (200) |
| `plant` | `mature_age` (100) · `max_age` (500) · `produces` ("") · `spawn_rate` (0.1) |
| `fertilizer` | `fertility_boost` (0.2) · `duration` (100) · `radius` (2) |
| `physics` | `decay_rate` (0.0) · `decompose_into` ("") · `decompose_chance` (0.0) · `nutrient_return` (0.0) |
| `interaction` | `pickable` (true) · `usable` (false) · `passable` (true) · `blocks_growth` (false) |
| `tile_effect` | `germination_multiplier`/`growth_multiplier`/`spawn_rate_multiplier` (1.0) · spreading: `spread_type_id` ("") · `spread_radius` (1) · `spread_interval` (200) · `spread_chance` (0.05) · `spread_blocked_by` ([]) · `converts_terrain` ("") · clamps: `fertility_override`/`moisture_override` (−1 = off) · reclamation: `reclaim_terrain` ("") · `reclaim_interval` (0) |
| `observation` | `vision_encoding` (0.5 or `"auto"`) · `value_source` (`freshness\|maturity\|viability\|duration\|none`) |
| `render` | `char` ("?") · `color` ([200,200,200]) |
| `spawn` | `initial_count` (0) · `terrain` (`soil\|sand\|plantable\|any`) · `respawn_rate` (0.0 per tick) · `max_count` (0 = use initial_count) |

The schema's source of truth is the spec dataclasses in
`world/object_registry.py`; the validator derives its field lists from
them, so this table cannot silently drift from the code.

## Cookbook

**Poisonous look-alike** (discrimination pressure):
```yaml
  mushroom:
    extends: berry
    edible: { calories: -20.0, toxicity: 0.8 }
    observation: { vision_encoding: 0.87 }   # close to food, not identical
    spawn: { initial_count: 6, respawn_rate: 0.01, max_count: 8 }
```

**A full plant chain** (seed ⇄ plant must reference each other):
```yaml
  acorn:
    extends: berry_seed
    seed: { grows_into: "oak_tree", grow_time: 120 }
    observation: { vision_encoding: 0.55 }
    spawn: { initial_count: 5, terrain: "plantable" }
  oak_tree:
    extends: berry_plant
    plant: { mature_age: 300, max_age: 2000, produces: "acorn", spawn_rate: 0.03 }
    observation: { vision_encoding: 0.70 }
    spawn: { initial_count: 3, terrain: "plantable" }
```

**Terrain patch** (boosts everything growing on its tile):
```yaml
  oasis:
    category: "terrain"
    interaction: { pickable: false }
    tile_effect: { germination_multiplier: 2.5, growth_multiplier: 2.0,
                   spawn_rate_multiplier: 1.5, spread_chance: 0.0 }
    observation: { vision_encoding: 0.20 }
    spawn: { initial_count: 4 }
```

A complete worked file ships at `config/custom_objects.yaml`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "unknown section/field … did you mean" | Typo — the message names the type and the closest valid name |
| "references unknown type" | A chain field points at a type that doesn't exist (check spelling, define the target, or order it earlier when using `extends`) |
| "will NEVER appear" warning | Add `spawn.initial_count`, `spawn.respawn_rate`, or make something produce it |
| vision_encoding collision warning | Pick a different value or use `auto` |
| Object appeared once, then vanished forever | Set `spawn.respawn_rate` (+ `max_count`) — initial copies don't regenerate by themselves |
