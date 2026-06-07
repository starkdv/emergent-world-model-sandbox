# Drop-in 3D model assets (`web/static/assets/models/`)

The 3D web renderer is **already wired** to load real glTF/GLB models for world
objects from this folder — you just need to add the files. Anything missing
falls back automatically to the shipped SVG sprite icons, so the world always
renders.

## Pre-wired slots (just drop these files in)

[`../manifest.json`](../manifest.json) already maps every object **category** to
a canonical filename here. Add any of these and it renders immediately — no code
or config changes:

| Drop this file in `models/` | Used for objects of category | Built-in examples |
|-----------------------------|------------------------------|-------------------|
| `food.glb`                  | `food`                       | Berry |
| `seed.glb`                  | `seed`                       | Berry Seed |
| `plant.glb`                 | `plant`                      | Berry Plant (scales up as it matures) |
| `fertilizer.glb`            | `fertilizer`                 | Fertilizer |
| `tool.glb`                  | `tool`                       | (custom tools) |

Want a *specific* type to differ from its category? Add it under
`objects.by_type` in the manifest (it overrides `by_category`), e.g.
`"berry": { "model": "models/berry.glb", "scale": 0.5 }`.

## Recommended CC0 source models (used in real game dev)

Download a CC0 pack, then copy + **rename** a model to the canonical filename
above. All sources are public-domain (CC0); no attribution required.

| Target file       | Suggested CC0 source model |
|-------------------|----------------------------|
| `food.glb`        | Kenney **Food Kit** (e.g. `apple`, `tomato`) or Quaternius **Survival** berries |
| `seed.glb`        | Quaternius **Ultimate Nature** seed/acorn, or Kenney **Nature Kit** `mushroom`/small prop |
| `plant.glb`       | Kenney **Nature Kit** `plant_bush` / `tree_default`, or Quaternius **Ultimate Nature** bush |
| `fertilizer.glb`  | Kenney **Survival Kit** sack/barrel, or any small crate/bag prop |
| `tool.glb`        | Kenney **Tools** / **Survival Kit** axe/hammer |

Sources: **Kenney** (https://kenney.nl/assets) · **Quaternius**
(https://quaternius.com/) · **Poly Pizza** (https://poly.pizza/, CC0 filter).

## Per-entry manifest fields

`model` (required), `scale` (default 1), `yOffset` (lift off the ground,
default 0), and for animated/rigged models `skinned: true` + `animation: "<clip
name>"`. Tune `scale`/`yOffset` so the model sits nicely on a 1×1 tile.

> The sandbox that generated this project cannot download these packs (its
> network is locked down) and won't commit copyrighted art, so the model files
> are not bundled — only the wiring is. Add the CC0 packs you want here.

See [`../CREDITS.md`](../CREDITS.md) for licensing of assets shipped in-repo.
