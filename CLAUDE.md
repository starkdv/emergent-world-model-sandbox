# CLAUDE.md — project memory for emergent-world-model-sandbox

## What this is

A multi-agent evolutionary sandbox: agents with recurrent attention
brains forage/plant/reproduce in a procedural ecology, learn within
their lifetime by PPO (full-network backprop), and pass trained weights
to offspring (Lamarckian). Public, MIT. The findings are written up in
a DOI'd technical report — `paper/emergent_world_model.pdf`,
DOI 10.5281/zenodo.21133600, mirrored at `docs/PAPER.md` with **all
per-run data inline**. Treat the paper as the source of truth for what
has been measured.

## Architecture map (where things live)

- `agents/brain/` — Brain v2 (dense) and BrainV3 (tokenised vision +
  attention → GRU; value on [z,h]; optional latent dynamics head
  g(h,a)→(ẑ′,r̂)). `spec.py` holds the frozen, injectable
  `ObservationSpec` + append-only genome ParamSpec + migration.
  v3.5 = v3 + Observation-v2 EXTRA block + SIGNAL action (output 9).
- `agents/ppo.py` — PPOSequenceLearner (sequence chunks, GAE, clipped
  PPO, torch mirror, Lamarckian sync) + world-model auxiliary loss +
  the model-quality toolkit (see below) + Dreamer-style imagination
  (P3, default off).
- `agents/planner.py` — LatentPlanner MPC: `shooting` (legacy default),
  `policy_shooting` (P1), `cem` (P2); warmup scheduling by ticks and
  readiness gating by measured error.
- `world/` — the ecology (terrain gen, plants/soil/weather/fire,
  checkpointing, spatial index). `render/` — state bridge + live
  2-D/isometric/browser-voxel viewers. `main.py` — headless/viz CLI.
- Headless experiment pattern: `python main.py --no-viz --config X
  --learning --mode rl --seed S --generations N --log --metrics-csv m.csv`
  then `scripts/analyze_logs.py`. NOTE: keep `simulation.parallel: false`
  in experiment configs (parallel+torch ≈ 50× slowdown).

## Model-quality toolkit (M1–M3) — the project's key instrument

- M1: `learning.ppo.rollout_metric_k` — k-step OPEN-LOOP rollout error
  measured every learn(); `learner.wm_rollout_error_ema`; metrics CSV
  column `wm_rollout_error`.
- M2: `planner.warmup_error_threshold` (+ `imagination.warmup_error`) —
  switch strategies on MEASURED error, latched, `warmup_ticks` as
  deadline.
- M3: `learning.ppo.world_model_multistep {k, coef}` — train the
  dynamics head on open-loop horizons 2..k.

## Measured conclusions (do not re-litigate without new multi-seed data)

1. **Single-seed effect sizes routinely evaporate** — every claim needs
   ≥4 seeds. Three separate reversals are documented in the paper
   (P2 +32%, P3 +25%, and a 2/2-seed warmup win all died under
   replication).
2. **P1 (policy_shooting) is the only planner upgrade that survived
   replication.** Plain `shooting` remains the recommended default;
   CEM + imagination lost under every schedule, fixed or error-gated.
3. **A weak model is useful for cheap things** (curiosity + blunt
   search: real behavioural phase change) **and harmful for expensive
   ones** (sharper search amplifies compounding open-loop error —
   ecology wm_err plateaus ~2.9).
4. **M3 reduces open-loop error exactly where the model is worst
   (early)**; no downstream fitness change in the ecology.

## Sibling project

**worldmodel-robotics** (PRIVATE repo): a differential-drive car with a
ray-cast camera navigating a room, driven by THIS repo's brain/PPO/
planner unchanged (rays = vision tokens via an injected ObservationSpec,
output_size 5, instincts disabled). It consumes this repo read-only at
a pinned ref on PYTHONPATH — core changes it needs get PR'd here, never
forked there. Its open question: does planning pay when the model is
good (room wm_err ~0.002 vs ecology ~2.9)?

## Conventions

- CI (must be green to merge; main is ruleset-protected, squash-only):
  black --check, flake8, pytest on 3.11 + 3.12. Run `black .` before
  committing — flake8 alone is not enough (a CI break happened exactly
  this way).
- Experiment results are committed under `docs/sample_*/` with README +
  ALL raw per-run data; corrections are made by banner + new study, not
  by deleting history. Honest negative results are house style.
- `.github/rulesets/` holds importable branch/tag protection JSON.
- Release tags `v*` are protected; the DOI'd paper pins to the repo, so
  never rewrite published history.
