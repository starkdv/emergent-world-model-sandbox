# Changelog

## [Unreleased] — Brain v3, Phases 1–4

### Phase 4 — Learned world model: dynamics head, curiosity, latent planning (opt-in)

**In simple terms:** agents can now carry a tiny *imagination*: a learned model
that predicts "if I do this, what will I perceive next, and what reward will I
get?". Three things fall out of it. (1) **Curiosity** — when reality differs
from the prediction, the surprise itself becomes a reward, so agents explore
without any hand-coded exploration bonus. (2) **Planning** — before acting, an
agent can imagine a few action sequences in its head and pick the most
promising first move. (3) The model lives in the genome, so good imaginations
are inherited and evolved. Everything is off by default and works with both
brain versions.

**Technical detail:**

- **Latent dynamics head** (`brain.world_model.enabled`): appended to the
  genome layout (prefix unchanged → migration-friendly), for both brain
  versions: `d = tanh([h ‖ onehot(a)]·W1+b1)`, `ẑ' = d·Wz+bz`,
  `r̂ = d·Wr+br`. v2: +2,401 params (8,873 → 11,274); v3: +3,441
  (17,337 → 20,778). New `Brain.encode`, `Brain.predict_next_latent`,
  `Brain.has_world_model`.
- **PPO training** (`learning.ppo.world_model_coef`): auxiliary loss
  `‖ẑ_{t+1} − sg(z_{t+1})‖² + (r̂ − r)²` over sequence chunks, with
  stop-gradient latent targets (the policy/value losses anchor the
  representation; the dynamics head chases it — prevents latent collapse).
  Under a2c or pure neuroevolution the head evolves only.
- **Curiosity** (`agents/curiosity.py`, `learning.curiosity`): intrinsic
  reward = `weight · clip(zscore⁺(prediction error), 0, clip)` with Welford
  running statistics, warmup, and optional weight decay; only above-average
  surprise is rewarded. Attached via `Agent.enable_learning`; inherited by
  offspring.
- **Latent rollout planner** (`agents/planner.py`,
  `brain.world_model.planner`): random-shooting over imagined rollouts —
  predict ẑ', advance the GRU on the imagined latent, accumulate discounted
  r̂, bootstrap with the critic at the horizon; first action of the best
  rollout wins. First-step actions respect the action mask.
- **Decision/reward logic unified**: new `Agent.choose_action` and
  `Agent.compute_reward` are the single sources used by BOTH the serial
  path and `utils/parallel.py` — the duplication that caused the Phase-2
  drift bug is gone structurally.
- Config: documented `brain.world_model` + `learning.curiosity` blocks;
  startup banner shows world-model/planner status.
- New tests: `tests/test_world_model.py` (20 tests — spec counts and
  prefix stability, prediction shapes, curiosity warmup/normalisation/
  decay/clipping, planner mask-respect and preference for model-predicted
  reward, PPO gradient reach into dyn.*, prediction error decreasing on a
  learnable toy world, agent/offspring integration).

### Documentation cleanup

- Deleted `WORLD_MODEL_IMPLEMENTATION_GUIDE.md` — its plan is superseded by
  the implemented Phase 4 (latent-space model sharing the policy encoder,
  rather than the guide's standalone observation-space MLP). The surviving
  idea from it — a population-level offline model trained from transition
  logs for dream-based evolution — is tracked in BRAIN_V3_PROPOSAL.md §5
  and SUGGESTIONS.md §5.1.
- Deleted `todo.md` — a historical phase-completion checklist (Nov 2025)
  fully superseded by SUGGESTIONS.md (forward roadmap), CHANGELOG.md
  (history), and BRAIN_V3_PROPOSAL.md (phase status).
- Updated SUGGESTIONS.md checkboxes (attention, GRU size, value head,
  curiosity, GAE, PPO, world model items now DONE with pointers),
  PROJECT_OVERVIEW_TECHNICAL.md status (world model now trained, not just
  logged), ECOSYSTEM.md (status banner pointing to the current docs).

### Phase 3b — PPO learner: real lifetime learning (opt-in, `learning.algorithm: ppo`)

**In simple terms:** until now, "learning during a lifetime" only adjusted the
brain's final output layer — everything underneath (perception, memory) could
only change across generations by mutation. The new PPO learner trains the
**whole network** while the agent lives: it replays short episodes in order
(so memory is trained too), uses a better estimate of "was that action good?"
(GAE), and clips every update so a single bad batch can't wreck the policy.
Learned weights are still written back into the genome, so offspring inherit
them. The old learner remains the default for controlled comparisons.

**Technical detail:**

- New `agents/ppo.py` — `PPOSequenceLearner` + `TorchBrainMirror`:
  - Persistent torch parameter mirror per agent (one `nn.Parameter` per
    ParamSpec tensor) with a persistent Adam optimizer; functional forward
    re-expressed in torch for **both** brain versions, so autograd reaches
    encoder/attention, GRU (through time), and both heads.
  - **Sequence replay**: time-ordered chunks of `seq_len: 8` steps with the
    chunk-start hidden state; the GRU is re-run over each chunk during
    learning (truncated BPTT; stored-state strategy à la R2D2).
  - **GAE(λ)** advantages (`compute_gae`, default λ=0.95) with per-batch
    advantage normalisation; value targets R = Â + V.
  - **PPO clipped surrogate** (ε=0.2) against the recorded behaviour
    log-probs (`Brain.decide_with_logprob`), + value loss (0.5) and entropy
    bonus (0.01); gradient-norm clipping (0.5).
  - Lamarckian sync after every update via `ParamSpec.pack`.
- `Agent.enable_learning(algorithm="ppo", ppo_config=...)`; offspring inherit
  the algorithm; graceful fallback to a2c when torch is unavailable.
- Config: `learning.algorithm` + documented `learning.ppo` block
  (`learning_rate`, `seq_len`, `batch_size`, `gae_lambda`, `clip_epsilon`,
  `value_coef`, `entropy_coef`, `epochs`, `grad_clip`, `chunk_buffer`).
- **Fix: the parallel agent-update path (`utils/parallel.py`, the default
  `simulation.parallel: true` pipeline) had drifted from `Agent.update`** —
  it still contained the removed auto-eat override and non-fading instinct
  calls, so Phase 2 behaviour was not active in default headless runs (the
  serial path and the test suite were correct). The parallel path now mirrors
  `Agent.update` exactly (fading instincts, no auto-eat, PPO step storage and
  terminal handling). Refreshed baselines with fading genuinely active:
  RL a2c seed 42: 64 alive @1000; neuroevolution: 16–33 across seeds — harder,
  viable, no extinctions.
- New docs: **`BRAIN_V2_V3_COMPARISON.md`** — full math-level comparison of
  both architectures (GRU gates, attention scaling, parameter/FLOP budgets)
  and both learners (policy-gradient algebra, GAE derivation, clipped
  surrogate), written for a CS-student audience.
- New tests: `tests/test_ppo_learner.py` (14 tests — GAE vs hand-computed
  values, chunking/padding/terminal handling, full-network gradient reach for
  v2 AND v3, Lamarckian sync, clip-bounds-off-policy, agent/offspring
  integration, no-torch fallback).

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
