# User Guide — modes, training, the dream world, planning, and the 3D viewer

A practical, task-oriented guide: **which knob to turn, why, and what happens.**
For the deep design see `BRAIN_V3_PROPOSAL.md`, `WORLD_UPGRADE_PROPOSAL.md`,
`ECOSYSTEM.md`, and `FRONTEND_3D_PROPOSAL.md`. For the full CLI flag list see
`CLI_GUIDE.md`.

Everything is configured in `config/default.yaml` (or a copy passed with
`--config`). The shipped defaults are a **large heightmap world with biomes,
day/night + weather, wildfire, a small starting population that breeds, and the
v3 attention brain with RL learning** — i.e. the "full" world.

---

## 1. Pick an evolution mode — `--mode`

| Mode | What it does | Pick it when |
|---|---|---|
| `rl` (default) | Agents **learn within their lifetime** (Actor-Critic / PPO) *and* evolve; learned weights are inherited (Lamarckian). | You want agents that visibly improve at foraging during a run. |
| `neuroevolution` | **No gradient learning** — only mutation + selection across generations. | A pure-evolution control, or to isolate "did learning help?". |

```bash
python main.py --no-viz --mode rl            # learn + evolve
python main.py --no-viz --mode neuroevolution
```

What happens: in `rl`, each agent runs a learner every few ticks (the `[LEARN]`
console lines are its loss). In `neuroevolution`, agents act purely on their
evolved network and only the genetic algorithm changes them between generations.

## 2. Pick a brain — `brain.version`

| Version | Architecture | Pick it when |
|---|---|---|
| `2` | Legacy GRU-MLP (~8.9k params). | A small, fast control baseline. |
| `3` (default) | Attention perception + `[z,h]` value head (~17k). | The recommended brain — better perception, scales to bigger vision. |
| `3.5` | v3 **+ social senses + the SIGNAL action** (78-dim obs, pheromone field). | Studying communication / social behaviour. Pair with `signal.enabled: true`. |

```yaml
brain:
  version: 3            # 2 | 3 | 3.5
```

> **Next (designed, not built):** v3.6 adds a `nearest_agent_kin` sense for kin
> selection — see `BRAIN_V3_PROPOSAL.md` §9.

## 3. Fading instincts — `brain.instincts`

Small additive biases (pick-up food, eat when hungry, turn toward food) that
**fade to zero by `fade_age`** so adult behaviour is purely learned. On by
default. Turn off (`enabled: false`) for "hard mode" (pure network from birth)
to test the emergence-first claim.

## 4. Learning algorithm — `learning.algorithm`

| Value | What | Notes |
|---|---|---|
| `a2c` (default) | Heads-only Actor-Critic, NumPy. | Light; encoder/GRU shaped by evolution only. |
| `ppo` | Full-network backprop, sequence replay, GAE(λ), clipping. | Needs PyTorch; stronger but heavier. |

## 5. Reward diet — `reward.preset`

| Preset | Reward signal | Pick it when |
|---|---|---|
| `legacy` (default) | Full dense shaping (exploration, anti-loop, eat bonuses…). | Fastest learning. |
| `minimal` | **eat / death / energy-delta only.** | Test whether the world's own pressures (+ curiosity) are enough — the world-side analogue of fading instincts. |

---

## 6. The world: biomes, weather, fire, desertification

The default world uses the **heightmap** generator (`terrain.generator:
heightmap`) → real biomes from elevation × moisture:

- **rock** = mountains (high elevation, impassable),
- **water** = seas/rivers in basins (impassable — agents stay on land),
- **sand** = arid lowlands/beaches; **soil/grass** = everything else.

Tune the mix with `terrain.{soil,rock,water,sand}_ratio` and the
`terrain.heightmap.*` knobs (bigger `feature_scale` = broader, smoother land).

- **Day/night + seasons + weather** — `environment.enabled: true`. Drives light,
  temperature, rain, and drought (which feed growth, metabolism, and fire).
- **Wildfire** — `fire.enabled: true`. Plants on hot, dry tiles can ignite, fire
  spreads to neighbours and self-extinguishes at wet boundaries.
- **Desertification** — the `sand:` block: sand spreads to adjacent unprotected
  soil over time and can be reclaimed by plants. Raise `terrain.sand_ratio` to
  seed more deserts; tune `sand.spread_*`.
- **Reproduction** — `reproduction.enabled: true` lets a small starting
  population grow by breeding (the default starts with few agents).

---

## 7. Learned world model + planning (model-based "imagination")

The brain can carry a **latent dynamics head** that predicts *(next perception,
reward)* from *(hidden state, action)*. It powers two things:

1. **Curiosity** — prediction error becomes intrinsic reward (rewards surprise).
2. **Planning** — short imagined rollouts pick the next action.

```yaml
brain:
  world_model:
    enabled: true        # adds the dynamics head to the genome
    hidden: 32
    planner:
      enabled: true      # model-based action selection (costlier per tick)
      depth: 3           # imagination horizon
      samples: 16        # candidate rollouts per decision
      gamma: 0.95
learning:
  algorithm: ppo         # trains the dynamics head (a2c/neuroevolution only evolve it)
  curiosity:
    enabled: true        # intrinsic reward from world-model error
    weight: 0.1
```

Run the full stack:

```bash
python main.py --no-viz --mode rl   # with the YAML above
```

What happens: with `world_model.enabled` the genome grows a dynamics head; with
`planner.enabled` each decision simulates a few futures and picks the best; with
`curiosity.enabled` agents are rewarded for exploring the unpredictable.

## 8. Dream-based evolution (offline world-model training)

"Dreaming" trains a **population world model** offline from logged transitions,
then evolves agents inside that learned model (no live sim needed) and seeds the
winners back into the real run:

```bash
# 1) Run with world-model logging to collect transitions
python main.py --no-viz --mode rl --learning --world-model-log

# 2) Train the dream model + evolve inside it
python scripts/dream_evolve.py --transitions data/logs/transitions_*.csv

# 3) Seed the next real run from the dream winners
python main.py --load-weights data/weights/dream_best.npz --mode rl
```

See `WORLD_MODEL_LOGGING_FORMAT.md` for the transition schema and
`dream_evolve.py --help` for options.

---

## 9. Watching it in 3D (the voxel viewer)

The viewer is the **real simulation** streamed read-only — not a demo.

```bash
python -m render.server                 # builds the world from config/default.yaml
#   open http://127.0.0.1:8000
python -m render.server --config config/training_easy.yaml
python -m render.server --checkpoint data/states/run.pkl   # fly a saved run
python -m render.server --demo          # fixed self-contained scene
```

**Controls:** drag = orbit · scroll = zoom · WASD/QE = fly · **F** = follow next
agent · **R** = reset · **click** an agent / tree / tile to inspect it.

Performance: terrain and all objects are drawn with instanced meshes, so large,
full worlds stay smooth. See `web/README.md` for the full fidelity map
(every sim mechanic → how it appears in 3D).

**Building your own frontend (Unity, etc.)?** The live stream protocol — SSE
framing, snapshot/delta JSON schemas, terrain decoding, coordinates, and a
Unity C# client — is documented in [`UNITY_STREAM.md`](UNITY_STREAM.md).

## 10. Download / export the world design

- **Checkpoint (resumable, exact):** `--save-state run.pkl` writes the full
  world + agents + RNG; resume with `--load-state run.pkl` (bit-identical in
  serial mode). Also loadable by the viewer via `--checkpoint`.
- **Voxel mesh (PLY) for Blender/MeshLab/any 3D tool:**

  ```bash
  python -m render.voxel_export --out world.ply                 # fresh world
  python -m render.voxel_export --checkpoint run.pkl --out run.ply
  ```

  The `.ply` is a colored voxel mesh (terrain columns + entities) that opens
  directly in Blender (`File → Import → Stanford (.ply)`), MeshLab, or online
  glTF/PLY viewers.
- **Recording (replay):** `python -m render.recorder --out run.jsonl --ticks N`
  records the render stream; replay it with `python -m render.server --replay
  run.jsonl`.

## 11. Bringing a Blender design *into* the world

The Python side is the source of truth (terrain is the W2 heightmap; objects are
the ECS registry), so a Blender model is used as a **render asset**, not as sim
geometry. Two supported directions:

1. **Author terrain heightmaps in Blender → drive the sim.** Bake a heightmap
   (a grayscale image of elevation) from your Blender sculpt, then feed it as the
   elevation field. (Use the `heightmap` generator's parameters to match, or
   load a custom elevation array — see `world/terrain_generation.py`.)
2. **Swap the 3D models the viewer uses.** The web client (`web/main.js`)
   currently builds primitive geometries per object category. To use Blender
   art, export your models as **glTF (.glb)**, drop them in `web/`, and load
   them with Three.js `GLTFLoader`, keying by object `category`/`type_id` from
   the bridge. (This is a client-only change; the sim is unaffected.)

Round-trip: export the live world to PLY (§10), refine it in Blender, and either
re-import a heightmap (path 1) or use the polished models as viewer assets
(path 2).

---

## 12. Analysis & logs

```bash
python main.py --no-viz --mode rl --log --metrics-csv data/metrics.csv
python scripts/analyze_logs.py            # action mix, survival, society/roles, …
```

`--log` writes per-action + per-tick CSVs to `data/logs/`; `--metrics-csv` writes
one aggregate row per generation. The analyzer reports action distribution,
success rates, energy economy, per-species consumption, the 🛰️ SOCIAL/SIGNAL
and 🧬 SOCIETY/ROLES sections, behavioural diversity, and an instinct-fade split.

## 13. Tests

The suite (512 tests) is documented file-by-file with pass criteria in
**[TESTING.md](TESTING.md)**. Run it with `pytest tests/ -q`; the CI lint gate
is `black --check . && flake8 .`.
