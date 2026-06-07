# Drop-in 3D model assets (`web/static/assets/models/`)

Put real game-development **glTF / GLB** model files here to give world objects
(and agents) authentic 3D art. The renderer loads them through a manifest, so
**no code changes are needed** — and anything missing falls back automatically
to the shipped SVG sprite icons.

## How it works

1. Drop a `.glb` (preferred) or `.gltf` model into this folder, e.g.
   `berry.glb`, `plant.glb`, `seed.glb`, `fertilizer.glb`.
2. Reference it in [`../manifest.json`](../manifest.json) under
   `objects.by_type` (exact `type_id`) or `objects.by_category` (a whole
   category). Example:

   ```json
   "objects": {
     "by_type": {
       "berry":       { "model": "models/berry.glb",       "scale": 0.5 },
       "berry_plant": { "model": "models/plant.glb",       "scale": 0.7, "yOffset": 0.0 },
       "berry_seed":  { "model": "models/seed.glb",        "scale": 0.4 },
       "fertilizer":  { "model": "models/fertilizer.glb",  "scale": 0.5 }
     },
     "by_category": {
       "plant": { "model": "models/generic_plant.glb", "scale": 0.6 }
     }
   }
   ```

3. Reload the page. `by_type` wins over `by_category`; if neither matches (or a
   model fails to load), the object renders with its SVG sprite icon.

Per-entry fields: `model` (required), `scale` (default 1), `yOffset` (lift off
the ground, default 0), `skinned` + `animation` (for animated/rigged models).

## Where to get free, license-clean assets (CC0 — used in real game dev)

These are the packs indie studios and game jams actually use. All **CC0 /
public-domain** (no attribution required, though it's appreciated):

- **Kenney** — https://kenney.nl/assets  (e.g. *Nature Kit*, *Food Kit*,
  *Survival Kit*, *Farm pack*). glTF + textures, CC0.
- **Quaternius** — https://quaternius.com/  (*Ultimate Nature*, *Animated
  Animals*, *Survival* packs). glTF, CC0.
- **Poly Pizza** — https://poly.pizza/  (filter by CC0).
- **three.js bundled models** — already used here for the agent
  (`RobotExpressive`, CC0).

> The sandbox that generated this project cannot download these packs (its
> network is locked down) and won't commit copyrighted art, so the files are not
> bundled. Add the CC0 packs you want here and update the manifest — everything
> wires up automatically.

See [`../CREDITS.md`](../CREDITS.md) for licensing of assets shipped in-repo.
