# Brain v3 → v3.5 → v3.6 — Architecture Proposal

**Status:**
- **v3 — ALL phases implemented**, including dream-based evolution
  (see §5 and ../CHANGELOG.md). Math-level architecture/learner comparison:
  BRAIN_V2_V3_COMPARISON.md
- **v3.5 — IMPLEMENTED** (the World-upgrade **W4** social/observation break).
  Part 1 shipped the genome-migration tool, agents-in-vision, and tile
  collision; **Part 2 shipped the Observation-v2 input block + the SIGNAL
  action / pheromone field** (`brain.version: 3.5`). See ../CHANGELOG.md
  ("Phase W4") and §8 below, which now documents the *as-built* design.
- **v3.6 — DESIGNED, not yet built** (the kin-similarity sense / Observation
  v3, deferred from World-upgrade **W5**). A single append-only input feature
  — `nearest_agent_kin` — bundled as the next batched genome bump rather than
  slipped in mid-cycle. Full design in **§9**; tracked in
  `WORLD_UPGRADE_PROPOSAL.md` (W5 "Deferred").

**Scope:** `agents/brain.py`, `utils/agents/brain_utils.py`, `agents/learning.py`,
`utils/agents/perception.py`, plus new modules under `agents/brain/`
**Inputs reviewed:** current codebase, `PROJECT_OVERVIEW_TECHNICAL.md`,
`WORLD_MODEL_IMPLEMENTATION_GUIDE.md`, `SUGGESTIONS.md` §2, `guideline.md` §8,
`ECOSYSTEM.md`, `WORLD_UPGRADE_PROPOSAL.md` (W4), `todo.md`

> **What is v3.5?** A *minor* version bump of the v3 attention brain: the
> architecture shape is unchanged (tokenised-vision attention, GRU memory,
> `[z, h]` value MLP, optional latent dynamics head), but its **I/O grows so
> agents can live in each other's world** — six new observation inputs
> (time-of-day, tile temperature, nearest-agent proximity & signal, on-hazard)
> and one new action output (**SIGNAL**, with a decaying pheromone field). It
> is the brain that completes World-upgrade phase W4. Because the v3 genome
> layout is append-only, an existing v3 genome migrates into v3.5 with
> **bit-identical behaviour** on the original actions (see §8.4). Full design,
> diagram, and as-built notes are in **§8**.

---

## 1. Why upgrade — what the current brain actually is

Brain v2 (`agents/brain.py`) is:

```
obs (72) → Encoder MLP [32] → GRU (32) → Policy head (8, masked) + Value head (1)
```

≈ 8,873 parameters, stored as one flat `genome.weights` vector. It works — the core
loop is proven — but four structural problems cap where it can go:

### P1 — Hardcoded behaviour inside the brain (emergence-first violation)

`Brain.forward()` (`agents/brain.py:136-216`) contains constant additive logit biases:
`+1.5` PICK_UP, `+1.0` EAT, `+0.5` USE, and a full **direction-aware "turn toward
food" routine** that scans the vision grid and biases the correct turn. Separately,
`Agent.update()` (`agents/agent.py:222-232`) **forces** EAT when energy < 50% and food
is held.

- `guideline.md` §8 says: *never hardcode high-level behaviours*. "Turn toward the
  nearest food" is a foraging policy, not a reflex.
- `PROJECT_OVERVIEW_TECHNICAL.md` §3 describes these as *"fading instinct biases"* —
  but in code they are **constant for the agent's entire life and every generation**.
  They never fade; they just hope to be out-shouted by learned logits.
- They are wired to **magic observation indices** (`obs[60]`, `obs[62]`, `obs[63]`,
  vision layout `8 + (row*5+col)*2`), so any observation change silently breaks them.

This is the single biggest credibility risk for the research claims ("foraging-to-
cultivation behaviour emerges without being programmed") and the first thing a
reviewer will find.

### P2 — Lifetime learning only touches the last layer

`AgentLearner._learn_vectorized_numpy/_torch` (`agents/learning.py:296-477`) computes
gradients **only for the policy and value heads**. The encoder and GRU receive no
gradient — they evolve by mutation only. So "online RL + Lamarckian inheritance"
currently means: evolution shapes the representation, RL tunes an 8×32 readout.
That weakens the central learning-×-evolution experiment the project is built around.

Additional issues in the same loop:

- **Recurrent replay mismatch**: random single transitions are sampled with *stored*
  hidden states that go stale as weights change; there is no sequence replay, no
  importance correction, no recomputed hidden states.
- Raw TD(0) advantage (high variance); no GAE, no update clipping — both already on
  the roadmap (`SUGGESTIONS.md` §2.2).
- The torch backend re-wraps every NumPy weight matrix into tensors on **every**
  `learn()` call.

### P3 — The weight layout is defined in three places

The flat-vector layout is hand-maintained in `brain_utils.calculate_weight_count`,
`brain_utils.unpack_weights`, **and** `AgentLearner._sync_genome_weights`. Any
architecture change (bigger GRU, attention, a new head) is three-file surgery with
silent-corruption failure modes. This is the practical reason the brain hasn't grown.

### P4 — No seat for the world model

Phase 3 (`WORLD_MODEL_IMPLEMENTATION_GUIDE.md`) plans forward/inverse dynamics,
curiosity, and planning. The guide's forward model (72 → [128,128,64] → obs,
≈34k params) is ~4× the entire brain and predicts in raw observation space. The
current brain offers no shared representation for it to use, and no genome story for
it (per-agent 34k evolving params is not viable).

---

## 2. Design goals for v3

Pulled directly from the aspiration docs:

| Goal | Source |
|---|---|
| Restore emergence-first integrity (instincts must actually fade, live outside the brain) | `guideline.md` §8, overview §3 |
| Attention over the vision grid; larger / stacked GRU; separate value pathway | `SUGGESTIONS.md` §2.1 |
| GAE + PPO-style clipped updates; full-network lifetime learning | `SUGGESTIONS.md` §2.2 |
| World model: forward/inverse dynamics → curiosity → planning → dream evolution | `WORLD_MODEL_IMPLEMENTATION_GUIDE.md`, overview §6.1 |
| Stay compact: thousands of concurrent agents, embedded-deployable, evolvable genome | overview §3 |
| Keep the dual-mode (pure neuroevolution vs RL+Lamarckian) toggle intact | overview §4 |
| Future-proof for continuous (Gaussian) action head and a signalling/communication head | overview §6.2, §6.4 |

---

## 3. Proposed architecture

### 3.1 Overview

```
                        ObservationSpec (named fields, replaces magic indices)
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
     vision 5×5×2 (50)        state+stimulus (16)            inventory (6)
            │                           │                           │
   per-tile embed (2→E)          state encoder MLP ──────────┬──────┘
   25 tile tokens + pos enc             │                    │
            │                           └── query ─┐         │
   single-head attention pool  ◄────────────────────┘        │
            │                                                │
            └────────────── concat → latent z_t (Z) ◄────────┘
                                        │
                          Memory core: GRU (H, 1–2 layers)
                                        │ h_t
        ┌───────────────┬───────────────┼────────────────────┬─────────────┐
        │               │               │                    │             │
   Policy head     Value head      Dynamics head        (future)      (future)
   h→8 logits,     [z,h]→V       (h,a)→ ẑ_{t+1}, r̂      Gaussian       message
   masked          small MLP     latent world model     action head    head
```

**Key decisions, with rationale:**

1. **Tokenised vision + tiny attention pool** (`SUGGESTIONS.md` §2.1). The 25 tiles
   become 25 tokens of (type_enc, value_enc) + a fixed 2-D positional encoding,
   embedded by one shared 2→E linear (E≈8). A single attention query derived from the
   agent-state encoding pools them. Cost: a few hundred params (the shared embed is
   tiny because weights are shared across tiles — unlike the current dense 50→32 layer
   which spends 1,600 params memorising tile positions). Benefit: position-equivariant
   perception that scales to larger vision radii (the `vision_radius` trait finally
   becomes usable: more tokens, **same weights**) and later to camera-like sensors
   (Track A).

2. **Latent dynamics head instead of a separate 34k observation-space model.** The
   forward model predicts the *next latent* `ẑ_{t+1}` (and reward) from `(h_t, a_t)`,
   sharing the perception encoder. This is the Dreamer-lite version of the guide's
   Phase 3 plan: ~Z×(H+8+Z) params instead of 34k, it lives naturally in the genome,
   prediction error in latent space is a cleaner curiosity signal (no wasted capacity
   predicting uncontrollable pixels — the role the guide assigns to the inverse
   model), and planning rollouts never need to decode back to raw observations.
   The guide's standalone observation-space `ForwardDynamicsModel` remains valid as a
   *population-level shared model* trained offline from the existing transition logs
   (`WORLD_MODEL_LOGGING_FORMAT.md`) — see §5 Phase 4.

3. **Value head reads `[z_t, h_t]`, not just `h_t`** (`SUGGESTIONS.md` §2.1 "separate
   value head"). The critic gets a direct, non-recurrent view of the current state, so
   value learning stops being hostage to whatever the GRU chose to remember. Full
   separation (independent MLP from raw obs) is a config option, but sharing the
   encoder keeps the genome compact.

4. **Instincts move out of the brain into `InstinctModule` — and actually fade.**
   The brain becomes a pure function of (obs, h, params). Instinct biases are computed
   outside, from *named* observation fields, and applied as
   `logits += instinct_bias * strength(age)` where
   `strength(age) = max(0, 1 - age / fade_age)` (fade_age ≈ 150 ticks, config).
   The auto-eat override in `Agent.update()` is replaced by a (fading) EAT instinct
   bias — a strong prior, not a forced action. Net effect: bootstrapping is preserved,
   but every adult behaviour is genuinely produced by the network, and the docs'
   "fading instinct" claim becomes true. Config-gated so the ablation (instincts
   on/off, fade fast/slow) becomes a reportable experiment.

5. **Memory core stays a GRU**, default H=48–64, optional 2-layer stack
   (`SUGGESTIONS.md` §2.1). No transformer memory: per-tick incremental state with
   O(1) step cost is exactly right for thousands of always-on agents and for embedded
   deployment.

### 3.2 Parameter budget

| Config | E / Z / H | Approx params | Use |
|---|---|---|---|
| `v3-small` | 8 / 32 / 32 | ≈ 9k | parity with v2; migration validation |
| `v3-base` (default) | 8 / 48 / 48 | ≈ 18k | attention + dynamics head |
| `v3-large` | 12 / 64 / 64 | ≈ 32k | capacity study |

All still trivially evolvable and embedded-friendly. Whether mutation-based search
degrades at 32k vs 9k params is itself one of the experiments the dual-mode toggle
exists to answer.

---

## 4. Proposed code structure

The enabling refactor matters more than the network itself. Three new abstractions
kill problems P1/P3 permanently:

```
agents/
  brain/
    __init__.py       # Brain class (same public API as v2: initial_state,
                      #   forward, decide, calculate_weight_count)
    spec.py           # ParamSpec + ObservationSpec (single sources of truth)
    modules.py        # pure functions: tile_embed, attention_pool, gru_step,
                      #   heads — NumPy core, torch mirror for training
    instincts.py      # InstinctModule: fading, config-gated, name-based
  learning.py         # upgraded learner (GAE, clipping, full backprop, seq replay)
  world_model.py      # Phase 4: population-level model + dream rollouts (per guide)
  planner.py          # Phase 4: rollout planner (per guide)
```

**`ParamSpec`** — a declarative, ordered list of named tensors:

```python
SPEC_V3 = ParamSpec([
    ("tile_embed.W",  (2, E)),  ("tile_embed.b",  (E,)),
    ("attn.Wq",       (S, E)),  ("attn.Wk", (E, E)), ("attn.Wv", (E, E)),
    ("state_enc.W",   (16, S)), ("state_enc.b",   (S,)),
    ("gru.Wr_in", (Z, H)), ("gru.Wr_h", (H, H)), ("gru.br", (H,)),
    # ... z/h gates ...
    ("policy.W",  (H, A)),      ("policy.b",  (A,)),
    ("value.W1",  (Z + H, 16)), ("value.W2", (16, 1)), ...
    ("dyn.W1",    (H + A, Z)),  ...
])
```

`pack()`, `unpack()`, `count()`, and `slices_for(prefix)` are all **derived** from
the spec — replacing the three hand-maintained copies in `brain_utils` and
`learning._sync_genome_weights`. Genome ↔ params sync becomes one line; adding a head
is one spec entry. The genome gains a `spec_version` tag so populations saved under
v2 are detected and migrated (v2 weights copied into matching v3 slices, new slices
randomly initialised — Lamarckian continuity across the upgrade).

**`ObservationSpec`** — named slices over the 72-vector
(`agent_state[0:8]`, `vision[8:58]`, `stimulus[58:66]`, `inventory[66:72]`, with named
fields like `stimulus.nearest_food_prox`). `perception.py` builds from it; the brain,
instincts, action-mask logic, and the world-model logger all read through it. Magic
indices disappear; observation changes become single-point edits.

**`InstinctModule`** — see §3.1 item 4.

### 4.1 Learning upgrade (P2)

- **Full-network gradients** via the torch path: keep one persistent torch parameter
  mirror per brain (built once, synced from/to the flat genome through `ParamSpec`),
  use autograd end-to-end, drop the per-call `as_tensor` copies. The NumPy fallback
  keeps the current heads-only update (it remains correct, just weaker) so the
  no-torch install still runs everywhere.
- **Sequence replay**: store short chunks (length ~8) instead of single transitions;
  initialise the GRU from the stored first hidden state ("burn-in lite"). Fixes the
  stale-hidden-state problem honestly.
- **GAE(λ)** advantage + **PPO-style clipped** objective with the stored behaviour
  log-prob (both explicitly on the roadmap, `SUGGESTIONS.md` §2.2). With clipping,
  replayed slightly-off-policy chunks stop being able to destroy the policy.
- Everything stays Lamarckian: `learn()` ends with `ParamSpec.pack(params) → genome`.

### 4.2 What deliberately does NOT change

- The 8-action discrete space, action masking, and `Agent`/`World`/evolution APIs.
- The 72-feature observation content (only how it is *addressed*).
- The dual-mode toggle — v3 must run gradient-free identically to v2.
- Reproduction/mutation machinery (`genome.py` operates on flat vectors regardless
  of length).
- The transition-logging pipeline (it feeds Phase 4).

---

## 5. Phased delivery plan

Each phase is independently shippable, behind config, with tests, and leaves `main`
green — matching `guideline.md` test/commit conventions.

**Phase 1 — Enabling refactor (no behaviour change). ✅ DONE**
`ParamSpec` + `ObservationSpec`; port v2 brain onto them; delete the three duplicate
layout definitions; extract instincts into `instincts.py` *with current constant
strengths* so the fixed-seed regression bands in `tests/` still hold.
Acceptance: identical action distributions on a fixed seed; all existing tests pass.
*Shipped: `agents/brain/` package, `tests/test_brain_spec.py` (layout equivalence
proven against a frozen legacy unpacker), NumPy 2.x fix.*

**Phase 2 — Emergence-integrity fixes. ✅ DONE**
Turn on instinct fading; replace forced auto-eat with the fading EAT bias.
Acceptance: baseline survival scenarios (`tests/scenarios/`) still pass with
recalibrated bands; an A/B config demonstrates adults act purely from learned logits.
*This phase is what makes the paper's emergence claim defensible.*
*Shipped: linear fade (default `fade_age: 150`), hunger-scaled EAT prior
(`hunger_eat_bias: 3.0 × energy_urgency`) replacing the forced auto-eat,
`brain.instincts` config in both YAML configs, `tests/test_instinct_fading.py`.
A/B survival validated on 1000-tick headless runs in both evolution modes.*

**Phase 3a — Capacity (config `brain.version: 3`). ✅ DONE**
Attention perception, GRU 48, `[z,h]` value head. v2 remains the default and
selectable for controlled comparison.
*Shipped: `agents/brain/v3.py` (BrainV3, ≈17.3k params: shared tile embedding
+ positional encoding, state-driven attention pool, GRU(48), value MLP over
[z,h]), `create_brain` factory + `Agent.brain_config`, `Brain.rebind` for
architecture-agnostic genome rebinds, dedicated v3 learner path,
`tests/test_brain_v3.py`. 1000-tick v3 runs viable in both modes (RL: 100
alive; neuroevolution: 48 — the documented capacity/evolvability trade-off).*

**Phase 3b — Learning upgrade. ✅ DONE**
Torch full-backprop learner (persistent parameter mirror, autograd end-to-end)
with sequence replay, GAE(λ), and PPO-style clipped updates, for both brain
versions. Acceptance: learning-curve comparison v2/v3, heads-only vs full
backprop, logged under identical conditions.
*Shipped: `agents/ppo.py` (PPOSequenceLearner + TorchBrainMirror), opt-in via
`learning.algorithm: ppo`, A2C kept as the default control;
`Brain.decide_with_logprob`; parallel update path brought back in line with
`Agent.update` (it had retained the auto-eat override and non-fading
instincts); `tests/test_ppo_learner.py`; math documented in
`BRAIN_V2_V3_COMPARISON.md`. The systematic learning-curve sweep remains as
the measurement step.*

**Phase 4 — World model. ✅ DONE (per-agent core)**
Per-agent latent dynamics head → curiosity reward (prediction error, normalised,
replacing hand-crafted exploration shaping) → rollout planner over latents.
*Shipped: `brain.world_model` dynamics head in the genome (both versions,
appended to the layout so the prefix stays migration-compatible), PPO
auxiliary training with stop-gradient latent targets, `agents/curiosity.py`
(z-scored, clipped, decaying intrinsic reward), `agents/planner.py`
(random-shooting latent rollouts with critic bootstrap),
`tests/test_world_model.py`.*

**Dream-based evolution. ✅ DONE**
Population-level shared model trained offline from the transition logs
(WORLD_MODEL_LOGGING_FORMAT.md), used to evaluate mutated genomes *inside*
the learned model with periodic grounding in the real environment.
*Shipped: `agents/dream.py` (PopulationWorldModel: obs-space, policy-agnostic
Δobs/reward/done predictor; dream rollouts; (μ+λ) dream evolution) and the
`dream_evolve.py` CLI — log → train → dream → grounding-ready `.npz` for
`main.py --load-weights`. Validated end-to-end; `tests/test_dream_evolution.py`.*

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Removing constant instincts collapses early survival | Fade-from-full-strength keeps tick-0 behaviour identical; fade rate is a config dial; Phase 2 ships only after scenario tests are recalibrated |
| Bigger genome slows neuroevolution convergence | `v3-small` parity config; capacity is an experiment, not a forced default |
| Full backprop + Lamarckian sync destabilises evolution | PPO clipping bounds update size; mode toggle isolates the effect; keep heads-only NumPy path as control |
| Genome incompatibility with saved populations | `spec_version` tag + slice-wise migration (§4) |
| Per-tick attention cost × thousands of agents | Single head, 25 tokens, E=8 → a few flops more than the dense layer it replaces; NumPy-vectorised over tokens |

---

## 7. Decision summary

1. **Refactor first** (`ParamSpec`/`ObservationSpec`) — everything else is cheap after it.
2. **Move instincts out of the brain and make them fade** — highest scientific value
   per line of code in the whole proposal.
3. **Grow the brain modestly** (attention + GRU 48/64 + `[z,h]` value head), behind a
   version switch, keeping v2 as the controlled baseline.
4. **Teach the whole network during a lifetime** (torch autograd, GAE, clipped
   updates, sequence replay) — makes the Lamarckian experiment real.
5. **Build the world model as a latent head sharing the encoder**, with the guide's
   observation-space model reserved for the population-level/offline role.

---

## 8. Brain v3.5 — the social / Observation-v2 upgrade (World phase W4)

### 8.1 Why a v3.5 (and why now)

Phases W1–W3 of the world upgrade gave the world a *climate*, *terrain*, and
an *ecology*. The one structural gap left (proposal `WORLD_UPGRADE_PROPOSAL.md`
P3) is that **agents cannot perceive or affect each other** — the simulation
is effectively N independent single-agent worlds sharing a food budget. Every
social research direction (kin selection, territory, communication,
cooperation) is blocked behind this.

Closing that gap needs the brain's **I/O to grow**: new senses (other agents,
time of day, local hazard) and a new act (emit a signal). That is exactly the
"single batched genome break" W4 was reserved for. Rather than mint a whole
new architecture, **v3.5 keeps every v3 component and only widens the input
and output layers** — the smallest change that unlocks the social layer while
preserving v3's learned/evolved weights via migration (§8.4).

v3 stays the controlled baseline (obs-v1, 8 actions); v3.5 is the brain you
select when running the W4 social world.

| Brain | Perception | Inputs | Actions | Role |
|---|---|---|---|---|
| v2 | dense vision MLP | 72 | 8 | legacy control |
| v3 | tokenised-vision attention | 72 | 8 | attention baseline |
| **v3.5** | **same attention + agent-aware vision** | **78** | **9 (+SIGNAL)** | **the social world (W4)** |

### 8.2 Architecture diagram (v3.5)

```
        Observation v2 (78 = 72 v1 prefix ⧺ 6 new)  ── ObservationSpec(version=2)
        │
        ├─ vision 5×5×2 (50)  ── now AGENT-AWARE: a tile with another living
        │        │               agent encodes (0.40, its energy)  [shipped W4·1]
        │        ▼
        │   per-tile embed (4→E, shared)  → 25 tile tokens (+ fixed pos-enc)
        │        │
        │        ├──────────────── keys / values ──────────┐
        │                                                   │
        ├─ STATE block (28)  = agent_state(8) ⧺ stimulus(8) ⧺ inventory(6)
        │        │                              ⧺ EXTRA(6)  ◄── NEW v3.5 inputs
        │        │      EXTRA = [ time_of_day_sin, time_of_day_cos,
        │        │                tile_temperature, nearest_agent_proximity,
        │        │                nearest_agent_signal, on_hazard ]
        │        ▼
        │   state encoder (28→S)  ── s ──► attention query ──► softmax pool ─► e (E)
        │        │                                                            │
        └────────┴──────────────── concat → latent  z = [s ‖ e]  (Z = S+E) ──┘
                                            │
                              Memory core:  GRU (H)
                                            │ h
            ┌──────────────────┬────────────┼─────────────────┬───────────────┐
            ▼                  ▼             ▼                 ▼               ▼
       Policy head        Value head    Dynamics head      (future)        SIGNAL
       h → 9 logits       [z,h]→V       (h, onehot a9)     Gaussian        is action #8
       (8 + SIGNAL),      MLP→1         → ẑ_{t+1}, r̂       action head     in the policy
       masked                           latent WM                          head ↑

   SIGNAL execution → writes a float onto the agent's tile → PheromoneField
   decays each tick → re-sensed next tick as EXTRA[nearest_agent_signal].
   Signals carry NO built-in meaning (emergence-first); any protocol must evolve.
```

Everything inside the dashed interior (tile embedding, attention, GRU, value
MLP, dynamics head) is **byte-for-byte the v3 design**. v3.5 only:
- widens the **state encoder input** 22 → 28 (the EXTRA block), and
- widens the **policy head output** 8 → 9 (the SIGNAL column).

### 8.3 The six new observation features (EXTRA block, indices 72–77)

Appended *after* the v1 layout so the 0–71 prefix is unchanged. All in [0, 1].

| Idx | Field | Meaning / source |
|----|-------|------------------|
| 72 | `time_of_day_sin` | `sin(2π · environment.time_of_day)` — phase of the day cycle (W1) |
| 73 | `time_of_day_cos` | `cos(2π · environment.time_of_day)` — together a smooth clock |
| 74 | `tile_temperature` | `environment.temperature` (W1); the agent feels the climate |
| 75 | `nearest_agent_proximity` | `1 − dist/​R` to the nearest *other* agent in vision range (0 = none) |
| 76 | `nearest_agent_signal` | strongest pheromone/SIGNAL value sensed on/around the tile |
| 77 | `on_hazard` | `1.0` if the agent's tile carries `contact_damage` (W3 thorns) |

Rationale: these are precisely the channels W1–W4 created but that the brain
had no input wires for — climate (so day/night and seasons become learnable
rather than just lethal), other agents (the P3 unblock, the spatial half of
which already shows up in vision), the communication channel, and a direct
danger sense. The vision grid already gained agent-awareness in W4 part 1;
`nearest_agent_proximity` adds a compact scalar the attention query can use
without spatial decoding.

### 8.4 Genome layout & migration (already-built mechanism)

v3.5 reuses `build_brain_v3_param_spec` with `state_inputs=28, output_size=9`.
Only three tensors change shape, and each *grows by appending* — the old one
sits in the new one's top-left corner:

| Tensor | v3 shape | v3.5 shape | Δ params (base S=40,H=48) |
|--------|----------|-----------|---------------------------|
| `state_enc.W` | (22, S) | (28, S) | +6·S = **+240** |
| `policy.W` | (H, 8) | (H, 9) | +H = **+48** |
| `policy.b` | (8,) | (9,) | **+1** |
| `dyn.W1` (if WM on) | (Z+8, hid) | (Z+9, hid) | +hid |

So **v3.5-base ≈ 17,626 params** (v3-base 17,337 + 289; +hid more with the
world model). v3.5-small / v3.5-large scale identically. The cost of the whole
social upgrade is <2% more weights.

Migration is the **already-shipped** `migrate_genome(old_flat, old_spec,
new_spec)` (W4 part 1, `agents/brain/spec.py`): a generic top-left copy. New
`state_enc.W` rows (the EXTRA features) and the new `policy.W`/`b` SIGNAL column
are zero, which means a migrated v3 genome produces the **same logits for
actions 0–7 and the same value to floating-point tolerance** (~1e-6: the
copied weights are exact; the only difference is summation order in the wider
state-encoder dot product). The EXTRA inputs and SIGNAL do nothing until
mutation/learning fills those weights in. This is the W4
"old populations load and behave identically" acceptance criterion, and its
unit test for the migration mechanism is already green.

### 8.5 SIGNAL action & the pheromone field

- **Action 8 = SIGNAL.** Executing it writes a fixed (or graded) float onto
  the agent's current tile in a new `PheromoneField` (a `float` grid).
- The field **decays geometrically each tick** (`signal_decay`, e.g. ×0.9) and
  optionally diffuses to neighbours — a stigmergic medium, like ant trails.
- Agents sense it through `EXTRA[nearest_agent_signal]`. Signals have **no
  built-in semantics**: whether they come to mean "food here", "danger", or
  "kin" must *emerge* (guideline §8). DROP-able zero-cost marker objects (W0
  registry) give a complementary, persistent channel for free.
- SIGNAL is gated by `signal.enabled` (default off). When off it is always
  masked, so a migrated genome's sampled behaviour is **exactly** v3's
  (softmax over the original 8). When on, the zero-init policy column starts
  neutral and can be learned/evolved.

### 8.6 Implementation status (v3.5 / W4 part 2)

Part 1 (✅): `migrate_genome`, agents-in-vision (`world.agents_visible`),
tile collision (`world.agent_collision`). Part 2 (✅, this is now built):

1. ✅ **ObservationSpec v2** — `build_observation_spec(version=2)` adds the
   `extra` slice (72:78) + named field indices; `OBSERVATION_SPEC_V2`; a
   module-level active spec (`get/set_active_observation_spec`,
   `set_observation_version`) that perception and the brain both read.
   `get_observation_size()` follows the active spec.
2. ✅ **Perception** — `_encode_extra` appends the six features (clock,
   temperature, nearest-agent proximity, nearest signal, on-hazard) under v2;
   the v1 path is untouched.
3. ✅ **Action space** — `Action.SIGNAL = 8`; `get_action_mask` sizes by the
   brain's `output_size` and masks SIGNAL unless `signal.enabled`. The
   `action_probabilities` helper was guarded for 8-wide policies; the dream
   model / planner / PPO already size by `output_size`; the logger stores the
   raw `action_value` (no fixed one-hot), so nothing else needed widening.
4. ✅ **BrainV3 in v3.5 mode** — resolves the active spec, derives
   `state_inputs` (28), includes the `extra` slice in the state path;
   `create_brain`/`calculate_weight_count` select v3.5 via `brain.version: 3.5`
   with `output_size` fixed at 9.
5. ✅ **PheromoneField + SIGNAL execution** — `world.pheromones` float grid with
   per-tick decay (and optional diffusion); `World.emit_signal`;
   `execute_signal`; `signal.{enabled,strength,decay,diffuse}` config.
6. ✅ **Migration on load** — `adapt_loaded_genome` migrates a v3 genome into
   v3.5 via `migrate_genome`; wired into `main.py --load-weights`.
7. ✅ **Learner parity** — the A2C-v3 (`_forward_batch_numpy_v3`) and
   PPO-torch batched forwards now include the EXTRA slice in `state_feats`
   (without this, v3.5 RL runs crash on a 28-vs-22 matmul).
8. ✅ **Config + tests** — `brain.version: 3.5` documented, `signal:` block;
   `tests/test_brain_v35.py` (19 tests: spec, perception, weight count,
   migration bit-identity, SIGNAL masking, pheromone decay, end-to-end);
   A2C + PPO end-to-end runs verified; full suite green.

9. ✅ **Analyzer** — `scripts/analyze_logs.py` gained a **SOCIAL / SIGNAL**
   section: SIGNAL usage rate, **signal entropy** (normalised Shannon entropy
   of emissions across signalling agents — shared behaviour vs specialists),
   and an **agent-proximity response** breakdown (action mix and SIGNAL rate
   bucketed by `nearest_agent_proximity`, plus mean proximity when signalling
   vs overall). This is the W4 "signal entropy and agent-proximity response
   measurable" acceptance criterion. (`tests/test_signal_analyzer.py`.)

All eight original checklist items and the analyzer follow-up are now done;
the W4 acceptance criteria (migration identity; signal/proximity measurable)
are both met.

### 8.7 Risks specific to v3.5

| Risk | Mitigation |
|---|---|
| Action-count change silently corrupts a learning path (mask/onehot width) | Single source of truth = `brain.output_size`; add shape asserts at the mask/replay/logger boundaries; SIGNAL gated off by default so every existing path keeps running at 8 until explicitly enabled |
| New EXTRA inputs destabilise evolved foragers | Migration zero-inits their rows → bit-identical start; they only matter once selection finds them useful; `observation.version` is an ablation switch |
| Pheromone field cost at thousands of agents | One `float` grid, vectorised decay; sensing is an O(1) tile read (or a tiny local max); diffusion optional |
| "Two genome breaks" if obs-v2 and SIGNAL ship separately | They are deliberately bundled into v3.5 as a *single* break, exactly as W4 intended |

---

## 9. Brain v3.6 — the kin-similarity sense (Observation v3, World phase W5)

> **Status: DESIGNED, not yet built.** This section is a *proposal* in the same
> sense §3 was a proposal before v3 shipped — it documents the next batched
> genome bump so it is on record, reviewable, and ready to build, but no code
> for it exists yet. W5 shipped its non-genome half (inventory transfer + the
> SOCIETY/ROLES analyzer); the genome-touching half — a kin sense — is parked
> here for Brain v3.6.

### 9.1 Why a v3.6 (and why it was deferred)

W5 opened **division of labor and trade**. The capability (give via USE) and
the instruments (role-entropy, behavioural novelty, territory) shipped without
touching the genome. But the deepest social question — **kin selection** (do
agents treat genetic relatives differently from strangers?) — needs the brain
to *perceive relatedness*, which it currently cannot. Everything an agent
senses about another agent today is phenotypic (their position, energy, signal);
nothing tells it "this neighbour shares my lineage."

Adding that perception is, again, an **observation break**: a new input wire.
Per the W4 lesson, we do not slip input changes in mid-cycle — we batch them
into a named version with a migration. So rather than tack `nearest_agent_kin`
onto v3.5 (orphaning every v3.5 genome the same way an unplanned obs change
would), we reserve it for **v3.6 / Observation v3**. Until then, kin dynamics
remain *observable from outside* (lineage logs + the W5 territory/overlap
metrics) even though agents can't yet *act on* kinship — research priority, not
a blocker (`WORLD_UPGRADE_PROPOSAL.md`, W5 "Deferred").

| Brain | Perception | Inputs | Actions | Role |
|---|---|---|---|---|
| v3 | tokenised-vision attention | 72 | 8 | attention baseline |
| v3.5 | + agent-aware vision + EXTRA block | 78 | 9 (+SIGNAL) | the social world (W4) |
| **v3.6** | **+ kin-similarity scalar** | **79** | **9** | **kin selection (W5)** |

### 9.2 Architecture diagram (v3.6)

```
        Observation v3 (79 = 78 v2 prefix ⧺ 1 new)  ── ObservationSpec(version=3)
        │
        ├─ vision 5×5×2 (50)  ── agent-aware (unchanged from v3.5)
        │        ▼
        │   per-tile embed → 25 tile tokens → attention pool → e (E)
        │
        ├─ STATE block (29)  = agent_state(8) ⧺ stimulus(8) ⧺ inventory(6)
        │        │                              ⧺ EXTRA(6) ⧺ KIN(1)  ◄── NEW v3.6
        │        │      KIN = [ nearest_agent_kin ]   (idx 78)
        │        ▼
        │   state encoder (29→S)  ──► attention query ──► pool ─► e
        │        │                                              │
        └────────┴──────────── concat → latent z = [s ‖ e] ────┘
                                        │
                          Memory core:  GRU (H) ─► policy(9) / value / dynamics
```

Identical to v3.5 except the **state encoder input grows 28 → 29**. The policy
head, value MLP, attention, GRU, and dynamics head are byte-for-byte v3.5. No
new action — kin perception changes *how* agents use the actions they already
have (give to kin, signal to kin, contest strangers), not the action set.

### 9.3 The new observation feature (KIN block, index 78)

Appended after the v2 layout so the 0–77 prefix is unchanged. In [0, 1].

| Idx | Field | Meaning / source |
|----|-------|------------------|
| 78 | `nearest_agent_kin` | genetic similarity (0 = unrelated/none, 1 = clone) to the **same** nearest other agent already used by `nearest_agent_proximity` (idx 75) |

Pairing it with the existing proximity feature is deliberate: the brain gets
"someone is *this* close (75) and *this* related (78)" as two scalars about one
neighbour, so it can condition behaviour on kinship without spatial decoding.

### 9.4 How kin similarity is computed (the fingerprint trick)

Comparing two full genomes per agent per tick (thousands of weights × N²
neighbours) is far too expensive. The design uses a **birth-time genetic
fingerprint**: a small fixed-dim vector `f ∈ R^k` (k≈8) computed **once** when
a genome is created, as a deterministic random projection of the weight vector
(seeded by a fixed matrix, so it is stable across the run):

```
f = normalize(P · weights)      # P: fixed (k × W) projection, computed once
                                # f cached on the genome; recomputed only on
                                # mutation/crossover (i.e. at birth)
kin(a, b) = clip( (f_a · f_b + 1) / 2 , 0, 1 )   # cosine → [0,1]
```

Per-tick cost is then **one k-dim dot product** against the already-located
nearest neighbour — O(k), not O(W). Properties:
- **Clones / un-mutated offspring** share `f` exactly → kin ≈ 1.
- **Distant lineages** project to near-orthogonal fingerprints → kin ≈ 0.5
  (random) and below; the value is a smooth genetic-distance proxy, not a
  hard "same lineage" bit, which is what kin-selection theory actually wants.
- A cheaper **lineage-only** fallback (`parent_ids`/`generation` overlap) is
  available as an ablation, but the projection captures *graded* relatedness
  that lineage IDs cannot.

The fingerprint lives on `Genome` (alongside `lineage_id`); perception reads
`world`'s already-computed nearest-other-agent (shared with idx 75) and does
the single dot product in `_encode_extra`.

### 9.5 Genome layout & migration

v3.6 reuses `build_brain_v3_param_spec` with `state_inputs=29` (everything else
unchanged from v3.5). Exactly one tensor changes shape, append-only:

| Tensor | v3.5 shape | v3.6 shape | Δ params (base S=40) |
|--------|-----------|-----------|----------------------|
| `state_enc.W` | (28, S) | (29, S) | +1·S = **+40** |

So **v3.6-base ≈ 17,666 params** (v3.5-base 17,626 + 40; +0.2%). Migration is
the *same* shipped `migrate_genome` top-left copy: the new `state_enc.W` row is
zero, so a migrated v3.5 (or v3) genome produces **bit-identical behaviour to
float tolerance** until selection/learning fills the kin row in. The
fingerprint matrix `P` is a fixed constant (not part of the genome), so it adds
no weights and needs no migration.

### 9.6 Implementation plan (as-planned checklist)

When built, v3.6 follows the v3.5 recipe exactly — this is why it was made a
*minor* bump:

1. ⏳ **ObservationSpec v3** — `build_observation_spec(version=3)` extends the
   `extra` slice to 72:79 with a `nearest_agent_kin` index (78);
   `OBSERVATION_SPEC_V3`; `set_observation_version(3)`.
2. ⏳ **Genome fingerprint** — `Genome.fingerprint` (k-dim), computed at
   `__init__`/`offspring`/mutation via a module-level fixed projection `P`.
3. ⏳ **Perception** — `_encode_extra` appends `nearest_agent_kin` under v3,
   reusing the nearest-other-agent already found for idx 75.
4. ⏳ **BrainV3 in v3.6 mode** — `state_inputs=29` via the active spec;
   `create_brain`/`calculate_weight_count` select v3.6 via `brain.version: 3.6`.
5. ⏳ **Migration on load** — `adapt_loaded_genome` already generic; add the
   v3.5→v3.6 spec pair so loaded weights migrate.
6. ⏳ **Learner parity** — the A2C-v3 and PPO batched forwards already slice
   `obs[:, spec.extra]`; widening the EXTRA slice to 7 is automatic once the
   spec changes (verify, no new code expected).
7. ⏳ **Config + tests** — `brain.version: 3.6` docs; `tests/test_brain_v36.py`
   (spec v3 layout, fingerprint determinism + clone=1/unrelated<1, migration
   bit-identity v3.5→v3.6, end-to-end).
8. ⏳ **Analyzer** — a **kin-conditioned behaviour** view: give-rate and
   signal-rate bucketed by `nearest_agent_kin` (do agents favour relatives?) —
   the W5 kin-selection acceptance criterion, made measurable.

### 9.7 Risks specific to v3.6

| Risk | Mitigation |
|---|---|
| Fingerprint projection too lossy → kin signal is noise | k is tunable; validate clones→1 and unrelated→~0.5 in a unit test; lineage-only fallback for ablation |
| Per-tick cost at high agent counts | O(k) dot product against the *one* already-located nearest neighbour; no extra neighbour scan (shares idx 75's) |
| "Yet another genome break" | Same discipline as v3.5: one batched, append-only, migration-safe break; v3.5 genomes load bit-identically |
| Kin sense hard-codes nepotism | Like SIGNAL, the feature is a *capability*, not an incentive — favouring kin must emerge; `observation.version` ablates it |
