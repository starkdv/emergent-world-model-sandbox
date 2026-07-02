# Testing Guide

Comprehensive reference for the test suite: **what each test file verifies and
its pass criteria.** The suite has **512 tests** across 48 files plus 3 scenario
files. CI runs lint (black + flake8, blocking) and the test matrix (Python
3.11 + 3.12).

## Running

```bash
pytest tests/ -q                       # whole suite
pytest tests/test_brain_v35.py -v      # one file, verbose
pytest tests/ -k checkpoint            # by keyword
pytest tests/ -x --tb=short            # stop at first failure
black --check . && flake8 .            # the CI lint gate (must pass to merge)
```

Conventions: each file resets shared singletons it touches (e.g.
`ObjectRegistry._definitions`, the active observation/reward specs) in an autouse
fixture, so tests are order-independent. Sims under test use `parallel=False`
for determinism.

> One known-flaky test: `test_dream_evolution.py::...::test_learns_action_
> conditional_dynamics` is a *stochastic learning-convergence* check; it can dip
> below threshold in a crowded full-suite run but passes in isolation. Re-run it
> alone to confirm.

---

## 1. Brain, genome & perception

| File | Tests | Verifies / pass criteria |
|---|---|---|
| `test_brain_spec.py` | 22 | The `ParamSpec`/`ObservationSpec` single-source-of-truth. **Pass:** weight counts, zero-copy unpack/pack round-trip exactly, observation slices/indices are correct, and `migrate_genome` top-left-copies an old genome into a larger spec. |
| `test_brain_v2.py` | 9 | Legacy GRU-MLP brain. **Pass:** forward pass shapes, GRU state update, deterministic action given fixed weights + RNG, genome↔weights sync. |
| `test_brain_v3.py` | 15 | v3 attention brain (tile embedding, attention pool, `[z,h]` value MLP). **Pass:** correct shapes/param count, attention is position-equivariant, value head reads `[z,h]`, batched forward matches per-step. |
| `test_brain_v35.py` | 19 | Brain v3.5 (Observation-v2 + SIGNAL). **Pass:** obs grows 72→78 with the 6 EXTRA senses; weight delta = **+289**; a v3 genome migrates to v3.5 with **bit-identical** logits/value on the original 8 actions (float tol); SIGNAL is masked unless `signal.enabled`; pheromone deposit/decay works; end-to-end v3.5 world runs. |
| `test_genome.py` | 10 | Genome mutation, crossover, trait inheritance, lineage/generation bookkeeping. **Pass:** offspring weights/traits within mutation bounds; lineage ids/generation increment correctly. |
| `test_observation_sanity.py` | 2 | Observation vector is well-formed. **Pass:** length matches the active spec; all values finite and in range. |
| `test_instinct_fading.py` | 13 | Bootstrap instincts that fade with age. **Pass:** instinct bias is full at birth, **linearly fades to exactly 0 at `fade_age`**, `null` fade reproduces legacy behaviour; the fade scales by the energy-urgency stimulus. |

Related design: `BRAIN_V3_PROPOSAL.md`, `BRAIN_V2_V3_COMPARISON.md`.

## 2. World, terrain & tiles

| File | Tests | Verifies / pass criteria |
|---|---|---|
| `test_world.py` | 21 | Core `World`: tile passability/plantability, object add/remove/move, neighbours, bounds. **Pass:** soil/sand passable, **rock + water impassable**, object lifecycle keeps `world.objects` and tile sets consistent. |
| `test_terrain_generation.py` | 15 | W2 heightmap generator. **Pass:** elevation in [0,1], requested rock/water/sand ratios honoured via quantiles, rivers flow downhill, biomes derive from elevation×moisture, output is seed-reproducible. |
| `test_systems.py` | 19 | The per-tick system pipeline (plant growth, germination, decay, soil dynamics, resource spawn). **Pass:** plants mature/produce, seeds germinate under fertility/moisture thresholds, food decays, soil fertility/moisture follow their rules. |
| `test_environment.py` | 34 | W1 environment engine (day/night, seasons, weather). **Pass:** `time_of_day`/`light`/`season`/`temperature` cycle correctly; rain/drought start/stop on their chances/durations; growth/germination/decay/metabolism consume the right climate multipliers; the **B1 moisture fix** (evaporation scales with temp/light, recovery only via rain or water-adjacency) holds; disabled = all multipliers exactly 1.0 (bit-compatible). |
| `test_agents_in_world.py` | 10 | W4 agents-in-world toggles. **Pass:** genome migration helper; `agents_visible` injects the 0.40 vision encoding; `agent_collision` blocks a tile; all off by default (bit-compatible). |
| `test_max_population.py` | 1 | Reproduction respects `max_population`. **Pass:** population never exceeds the cap. |
| `test_reproduction.py` / `test_reproduction_config.py` | 1 / 2 | In-sim breeding. **Pass:** offspring spawn only when energy/age/cooldown thresholds are met; parent loses the configured energy split; config flags honoured. |

Related design: `WORLD_UPGRADE_PROPOSAL.md`, `ECOSYSTEM.md`.

## 3. Objects, ecology & tile effects

| File | Tests | Verifies / pass criteria |
|---|---|---|
| `test_object_registry.py` | 76 | The ECS object registry (the largest suite). **Pass:** definitions register/lookup, `create()` builds components from spec + overrides, categories/pickability/terrain-layer classification, builtins (berry/plant/seed/fertilizer/sand/thorns) are correct. |
| `test_object_validation.py` | 19 | W0 validating loader. **Pass:** unknown sections/fields raise helpful errors (typo `edibel` → suggestion), dangling cross-references (`grows_into: NONEXISTENT`) are caught at load, all errors collected with the type_id. |
| `test_tile_effects.py` | 47 | Tile-effect engine (sand spread/reclaim, thorns contact damage). **Pass:** sand spreads to unprotected neighbours on the configured interval/chance, plants block/reclaim it (desertification dynamics), thorns deal contact damage; clamps apply. |
| `test_ecology.py` | 14 | W3 ecology (multi-species food, toxicity, species pack). **Pass:** eating applies `net = calories×freshness − toxicity×…`; toxic food is net-negative; per-species definitions load from YAML. |
| `test_calamity.py` | 1 | Periodic calamity. **Pass:** at the configured interval the configured fraction of resources is destroyed (respecting affect_plants/food/seeds). |
| `test_resource_spawn_stacking.py` / `test_stacking_config.py` | 2 / 3 | Object stacking rules. **Pass:** with stacking off, one real object per tile (overflow placed on an empty neighbour); spawn respects this. |

Related design: `OBJECTS_GUIDE.md`, `ECOSYSTEM.md`.

## 4. Actions, movement & reward shaping

| File | Tests | Verifies / pass criteria |
|---|---|---|
| `test_action_energy_shaping.py` | 6 | Per-action energy costs + anti-spin shaping. **Pass:** action costs match config; escalating turn/wait penalties apply; movement into impassable tiles fails. |
| `test_drop_action_mask.py` | 2 | DROP masking. **Pass:** DROP is masked when inventory empty or no space (mirrors `execute_drop`). |
| `test_turn_balance.py` | 5 | Turn-direction balance regularizer. **Pass:** heavily skewed L/R turning is penalised; balanced turning isn't. |
| `test_anti_spin.py` | 1 | Anti-spin penalty. **Pass:** tight turn-only cycles are penalised. |
| `test_improved_rewards.py` | 1 | Dense reward sanity. **Pass:** moving toward food / eating yields positive reward. |
| `test_loop_reward_shaping.py` | 2 | Path-loop penalties. **Pass:** A→B→A backtracks and tight revisit cycles are penalised. |
| `test_w6c_reward_diet.py` | 11 | W6c reward diet + metrics CSV. **Pass:** `legacy` default unchanged; `minimal` rewards **eat / death / energy-delta only** (no exploration/anti-spam terms — exact values asserted: e.g. eat 5.0 + gain×0.2); `RewardConfig.from_dict` falls back to legacy on unknown preset; `MetricsWriter` writes the fixed header + one row per generation. |

## 5. Learning (A2C / PPO)

| File | Tests | Verifies / pass criteria |
|---|---|---|
| `test_ppo_learner.py` | 13 | PPO learner. **Pass:** sequence replay + GAE(λ) advantages, clipped ratio update reduces loss on a fixed batch, gradient clipping, value/entropy coefficients applied, falls back gracefully without torch. |
| `test_energy_sustainability.py` | 1 | Long-run viability. **Pass:** a learning population survives past the survival-time target rather than collapsing. |
| `test_parallel.py` | 12 | Threaded agent updates. **Pass:** parallel observe/decide/learn produces the same per-agent results as serial (no races on shared state); pools shut down cleanly. |

## 6. World model, curiosity, planning & dreams

| File | Tests | Verifies / pass criteria |
|---|---|---|
| `test_world_model.py` | 19 | Latent dynamics head (predicts next-latent + reward). **Pass:** head shapes/param count, it only extends the genome (prefix stays valid), curiosity = clipped z-scored prediction error, planner picks actions from imagined rollouts. |
| `test_dream_evolution.py` | 10 | Population world model + dream-based evolution. **Pass:** the dream model trains from logged transitions and agents evolved inside it transfer back. *(The action-conditional-dynamics convergence test is stochastic — see the note above.)* |
| `test_world_model_logger.py` | 1 | Transition logging. **Pass:** transitions/episodes/world-state CSVs are written with the documented schema (`WORLD_MODEL_LOGGING_FORMAT.md`). |
| `test_planner.py` | 16 | Latent planner upgrades (`docs/PLANNING_PROPOSAL.md` P1+P2). **Pass:** `shooting`/`policy_shooting`/`cem` strategies each pick the best first action and return valid actions, action-mask is honoured, reward/value normalization runs, the `commit` control-horizon queues and replays, TD(λ) `_score` matches the closed form at λ∈{0,1}, and `from_config` defaults reproduce the legacy controller. |
| `test_imagination.py` | 5 | Dreamer-style imagination actor-critic (P3). **Pass:** `TorchBrainMirror.imagine_loss` returns a finite scalar with gradients reaching the policy/value heads (horizon ≥ 1), and the PPO learner's `imagination` config is off by default and read when enabled. *(Skipped without torch.)* |

## 7. Substrate: spatial index & checkpointing (W6)

| File | Tests | Verifies / pass criteria |
|---|---|---|
| `test_spatial_index.py` | 11 | W6a edible-object spatial index. **Pass:** add/remove/move/query_box correct; **`nearest_edible` with the index ON equals the legacy tile scan OFF across 20 random worlds** (the contract); row-major tie-break; index stays consistent through add/remove/move/pickup/drop; a world runs identically with the index on and off. |
| `test_checkpoint.py` | 4 | W6b save/resume. **Pass:** save→continue vs save→load→run is **bit-identical in serial mode** (the acceptance criterion); tick/population/pheromones/spatial-index restored; immediate-resume matches (RNG captured at the right moment); a bad version is rejected. |

## 8. Social dynamics & analyzers (W5)

| File | Tests | Verifies / pass criteria |
|---|---|---|
| `test_w5_society.py` | 13 | W5 trade + society metrics. **Pass:** USE transfers the first item to a facing agent when `transfer_enabled` (recipient-full rejects; gate off → legacy plant behaviour; mask widens when only a trade is possible); analyzer role-entropy (high when roles differ, 0 when shared), behavioural-novelty JS (0 identical, >0 disjoint), territory bbox/overlap, and give-counting. |
| `test_signal_analyzer.py` | 6 | W4 SOCIAL/SIGNAL analyzer. **Pass:** SIGNAL count/rate, normalised signal entropy (≈1 shared evenly, <0.5 concentrated), `interaction_kind="signal"` detection, agent-proximity buckets, graceful handling of a missing action column. |

The analyzer itself is `scripts/analyze_logs.py`; see also `docs/sample_run/`
and the brain-cohort competition sample in `docs/sample_competition/` (its
⚔️ COHORT COMPARISON section is covered by `TestCohortComparison` in
`tests/test_w5_society.py`).

## 9. Frontend / 3D viewer (render)

| File | Tests | Verifies / pass criteria |
|---|---|---|
| `test_state_bridge.py` | 12 | F0 world→render bridge (read-only). **Pass:** snapshot is JSON-serializable and round-trips the packed terrain grids; objects/agents listed with full state (age, last action, inventory, lineage; object value/planted); feature flags + burning ids reported; **the bridge never mutates the world**; deltas report only what changed (move/add/remove/signals/dead-agent). |
| `test_sim_session.py` | 5 | F3a streaming core. **Pass:** the demo/config world has terrain + agents; `step()` advances the tick and returns a delta; `snapshot()` re-primes the tracker; checkpoint resume works through the session. |
| `test_recorder.py` | 6 | F5 record/replay. **Pass:** `record()` writes a JSONL snapshot + one delta per tick; `Recording.load` round-trips; `ReplaySession` replays deltas in order and loops with a snapshot resync; bad recordings rejected; it satisfies the server's session interface. |
| `test_voxel_export.py` | 5 | F1 offline PLY export. **Pass:** PLY header counts match the body; exposed-face culling shrinks a flat world; a tall column exposes cliff sides; entity cubes add geometry; vertices carry in-range colors. |

Related design: `FRONTEND_3D_PROPOSAL.md`, `web/README.md`.

## 10. Infrastructure & logging

| File | Tests | Verifies / pass criteria |
|---|---|---|
| `test_async_logger.py` | 5 | Async agent logger. **Pass:** rows are written off-thread, flushed/closed cleanly, schema stable. |
| `test_logging.py` | 1 | AgentLogger smoke. **Pass:** a run with `--log` produces a readable CSV. |

## 11. Scenario / acceptance (`tests/scenarios/`)

| File | Verifies / pass criteria |
|---|---|
| `test_baseline_easy_config.py` | An "easy" config keeps a population alive for the target horizon. |
| `test_baseline_verification.py` | A baseline run hits the documented survival/feeding targets. |
| `test_exploration_bonus.py` | The exploration bonus measurably increases coverage vs. off. |

---

## 12. Non-pytest helper modules

`tests/test_evolution.py` and `tests/test_exploration_sweep.py` contain **no
`test_` functions** — they are runnable analysis/sweep scripts kept alongside
the suite (run directly with `python`), not collected by pytest. They are
excluded from CI's test count.

---

## Adding tests

- Put unit tests in `tests/test_<area>.py`; reset any global singleton you touch
  in an autouse fixture.
- Prefer deterministic worlds (`parallel=False`, fixed `seed`).
- State the **pass criterion** in the test docstring/name (what behaviour, what
  bound). For "acceleration" or "diet"-style features, assert **equivalence to a
  reference** (e.g. index-on == index-off) rather than absolute magic numbers
  where possible.
- Keep new code lint-clean (`black`, `flake8`) — the lint job blocks merges.

See also: `USER_GUIDE.md` (running the project), `CLI_GUIDE.md` (flags),
`CHANGELOG.md` (what each phase added and its test count).
