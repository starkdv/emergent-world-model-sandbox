# Changelog

## [Unreleased] — Brain v3, Phases 1–3

### Phase 3 — Attention brain architecture (opt-in, `brain.version: 3`)

**In simple terms:** there is now a second, smarter brain design you can switch
on in the config. Instead of looking at all 25 vision tiles with equal, fixed
importance, the new brain *attends*: a tiny attention mechanism, steered by the
agent's internal state (hunger, inventory, …), decides which tiles matter right
now. Its memory is larger, and its value estimate ("how good is my situation?")
can see the current situation directly instead of only remembered context. The
old brain remains the default, so the two can be compared scientifically under
identical conditions.

**Technical detail:**

- New `agents/brain/v3.py` — `BrainV3(Brain)`, ≈17,337 parameters
  (vs ≈8,873 for v2):
  - **Shared tile embedding**: each vision tile becomes a token
    `[type, value, pos_row, pos_col]` embedded by one 4×8 matrix shared
    across all tiles (position-equivariant; scales to larger vision radii
    with the same weights). Fixed positional encoding, not learned.
  - **Single-head attention pool**: query derived from the state encoding
    (22 non-vision features → 40), keys/values from tile embeddings;
    softmax(k·q/√E) pools the tiles.
  - **Latent z = [state 40 | pooled vision 8]** feeds a **GRU(48)**.
  - **Value MLP reads [z, h]** (96 → 16 → 1) — the critic gets a direct
    view of the present state ("separate value head" roadmap item).
- `Brain` was refactored into overridable `_encode` / `_value` /
  `_build_nested` template methods; v2 behaviour is unchanged (verified by
  the existing equivalence tests).
- New `Brain.rebind(genome)` re-binds parameter views after genome
  replacement; `clone_agent`, `inherit_knowledge`, and pretrained-weight
  loading now use it (architecture-agnostic, replaces hand-rolled
  reconstruction).
- Factory `agents.brain.create_brain(genome, brain_config, instincts)` and
  `calculate_weight_count_for_config(brain_config)` map the YAML `brain`
  section to the right class; `Agent.brain_config` (class-level, set in
  `main.py`) makes offspring inherit the architecture.
- Learner support: `AgentLearner.learn` routes version-3 brains to a
  dedicated NumPy batch path (attention forward + policy-head and
  value-MLP-output updates — same heads-only update depth as v2; the
  full-backprop learning upgrade is Phase 3b).
- Config: `brain.version` (default **2**) and a `brain.v3` size block
  (`embed_dim`, `state_dim`, `gru_hidden_size`, `value_hidden`).
- Validation: 1000-tick headless v3 runs — RL seed 42: 100 alive;
  neuroevolution seed 42: 48 alive (vs 100 for v2 — the expected
  capacity-vs-evolvability trade-off of mutating 2× the parameters, and
  the reason v2 remains the default baseline).
- New tests: `tests/test_brain_v3.py` (15 tests — spec count, factory
  dispatch, masking, attention/positional sensitivity, rebind, learner
  updates + Lamarckian sync, evolution integration).

### Phase 2 — Fading instincts & auto-eat removal (emergence integrity)

**In simple terms:** survival "training wheels" now come off as agents grow up.
Newborns are still nudged toward picking up food, eating when hungry, and turning
toward food — but the nudges weaken every tick and disappear entirely at age 150
(configurable). The old hidden rule that *forced* hungry agents to eat was removed.
Result: every behaviour you observe in an adult agent is genuinely produced by its
evolved/learned network, which is what makes the project's "emergence-first" claim
honest and testable.

**Technical detail:**

- `agents/brain/instincts.py`: `InstinctModule.strength_at(age)` now drives a
  linear fade (1.0 at birth → 0.0 at `fade_age`, default 150 ticks). All instinct
  biases — PICK_UP +1.5, EAT +1.0, USE +0.5, turn-toward-food +0.8×proximity —
  are multiplied by this strength.
- **Auto-eat override removed** (`agents/agent.py`): the hardcoded
  "force EAT below 50% energy while holding food" rule is gone. Replaced by a
  hunger-scaled EAT prior inside the instinct module:
  `+hunger_eat_bias × energy_urgency × fade_strength` (default bias 3.0). The
  policy can override it, and it fades with age like every other instinct.
- New config section `brain.instincts` (`enabled`, `fade_age`, `hunger_eat_bias`)
  in `config/default.yaml` and `config/training_easy.yaml`; wired class-level via
  `Agent.instinct_config` in `main.py` so offspring inherit the configuration.
  Setting `fade_age: null` reproduces the legacy (never-fading) behaviour;
  `enabled: false` gives a pure network from birth — both useful as ablations.
- Instinct modules are preserved across all brain rebuilds (offspring cloning in
  `agents/evolution.py`, knowledge inheritance in `agents/agent.py`, pretrained
  weight loading in `utils/agents/learning_utils.py`).
- Validation: 1000-tick headless A/B runs remain population-stable in both
  evolution modes (RL seed 42: 94 alive vs 100 before; neuroevolution seed 42:
  100 vs 92; harsh seed 7 RL: 23 with fade vs 30 without — a modest, intended
  increase in selection pressure, no collapse).
- New tests: `tests/test_instinct_fading.py` (config plumbing, hunger bias
  scaling/gating, fade-through-agent path, adult policy == pure network,
  instinct survival across clone/inherit, end-to-end "hungry agent still eats").

### Phase 1 — Spec-driven brain refactor (enabling groundwork)

**In simple terms:** the brain's wiring diagram used to be written out by hand in
three different files that all had to agree byte-for-byte; now it is declared once
and everything else is derived from it. This makes upcoming architecture upgrades
(bigger memory, attention, world-model heads) one-line changes instead of
error-prone three-file surgery. No behaviour changed in this phase.

**Technical detail:**

- `agents/brain.py` became the `agents/brain/` package:
  - `spec.py` — `ParamSpec` (declarative genome layout: weight counting,
    zero-copy unpacking, packing, version tag for future genome migration) and
    `ObservationSpec` (named observation fields replacing magic indices).
  - `modules.py` — pure neural functions (sigmoid, softmax, GRU step).
  - `instincts.py` — instinct biases extracted from `Brain.forward`; the network
    is now a pure function of (observation, hidden state, parameters).
- `AgentLearner._sync_genome_weights` (Lamarckian inheritance) reduced from ~50
  hand-maintained lines to a single `ParamSpec.pack` call.
- `utils/agents/brain_utils.py` kept as a thin backward-compatibility shim.
- Fixed NumPy ≥ 2.0 incompatibility in the value head
  (`float()` on a shape-`(1,)` array), which broke 8 tests on fresh installs.
- New tests: `tests/test_brain_spec.py`, including byte-for-byte layout
  equivalence against a frozen copy of the legacy unpacker.

### Earlier

#### Added
- `agents/agent.py`: Improved agent lifecycle and GRU hidden-state integration.
- `tests/test_evolution.py`: New tests covering mating, selection, and lineage tracking.

#### Changed
- `agents/learning.py`: Refactored actor-critic update loop for numerical stability and batch handling.

#### Notes
Consolidated migration to Brain v2 (GRU + Actor-Critic), reorganized utilities into
`utils/agents/`, and updated tests to match new APIs.
