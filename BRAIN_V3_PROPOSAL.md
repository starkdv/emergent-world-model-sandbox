# Brain v3 — Architecture Proposal

**Status:** Phases 1–3a implemented (see §5 and CHANGELOG.md); Phase 3b
(learning upgrade) and Phase 4 (world model) proposed
**Scope:** `agents/brain.py`, `utils/agents/brain_utils.py`, `agents/learning.py`,
`utils/agents/perception.py`, plus new modules under `agents/brain/`
**Inputs reviewed:** current codebase, `PROJECT_OVERVIEW_TECHNICAL.md`,
`WORLD_MODEL_IMPLEMENTATION_GUIDE.md`, `SUGGESTIONS.md` §2, `guideline.md` §8,
`ECOSYSTEM.md`, `todo.md`

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

**Phase 3b — Learning upgrade.**
Torch full-backprop learner (persistent parameter mirror, autograd end-to-end)
with sequence replay, GAE(λ), and PPO-style clipped updates, for both brain
versions. Acceptance: learning-curve comparison v2/v3, heads-only vs full
backprop, logged under identical conditions.

**Phase 4 — World model (the guide's Phase 3.x, re-based on v3).**
Per-agent latent dynamics head → curiosity reward (prediction error, normalised,
replacing hand-crafted exploration shaping) → rollout planner over latents →
population-level shared model trained offline from transition logs for
**dream-based evolution** (guide §6.1: evolve in the model, ground periodically in
the real world).

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
