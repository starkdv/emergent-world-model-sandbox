# Planning — current design vs. improvements (proposal)

**Status:** Planner is **IMPLEMENTED** (`agents/planner.py`,
`brain.world_model.planner`). The staged upgrade path is **P1 → P3**.
- **P1 — IMPLEMENTED** (`strategy: policy_shooting`, `first_action`,
  `normalize`, `commit`). The legacy `strategy: shooting` remains the default.
- **P2 — PARTIAL** (`strategy: cem`, `lam` built; dynamics-ensemble uncertainty
  deferred — needs a genome change).
- **P3 — EXPERIMENTAL** (`learning.ppo.imagination`, default off).
- **Multi-seed replication (`docs/sample_planning_multiseed/`, 4 seeds):** only
  **P1 looks reliably good** (modestly higher, much *steadier* fitness at equal
  cost). The single-seed CEM (P2) and imagination (P3) wins **did not replicate**
  at this scale; the legacy `shooting` controller stays the default and P1 is the
  recommended upgrade. The per-phase "MEASURED" notes below are single-seed and
  superseded by the replication.

**Scope:** `agents/planner.py`, `agents/agent.py` (`choose_action`),
`agents/brain/__init__.py` + `agents/brain/v3.py` (latent dynamics head),
`agents/ppo.py` (imagination training, P3), `config/*.yaml`
(`brain.world_model.planner`).

**Inputs reviewed:** the current code (below), the measured A/B in
`docs/sample_planning_curiosity/`, `agents/dream.py` (population world model),
`docs/BRAIN_V3_PROPOSAL.md` §3/§5, and the model-based RL literature
(random shooting / PETS, CEM, MPPI, PlaNet, Dreamer V1–V3, MuZero, MBPO).

---

## 1. What the planner is today (as-built)

Each agent with `brain.world_model.enabled` carries a **latent dynamics head**
in its genome (trained online by PPO). When `…planner.enabled`, the agent
selects actions with a one-step-lookahead **random-shooting controller**
(`agents/planner.py`):

```
plan(brain, h, action_mask):                       # called EVERY tick
    best = None
    repeat `samples` times:
        a0 = uniform pick from VALID first actions
        score = rollout(brain, h, a0)
        track best
    return a0 of the best rollout                   # only the FIRST action is used

rollout(brain, h, a0):
    h_sim = h ; score = 0 ; discount = 1
    a = a0
    for k in range(depth):
        z', r̂ = brain.predict_next_latent(h_sim, a) #  f(h,a) → (ẑ', r̂)
        score += discount * r̂ ; discount *= gamma
        h_sim  = brain._gru_step(z', h_sim)
        a = uniform random action                    #  ⚠ continuation is RANDOM
    score += discount * brain._value(z', h_sim)      #  critic bootstrap at horizon
    return score
```

Key facts, with code references:

- **Receding-horizon MPC, not plan-following.** `choose_action`
  (`agents/agent.py:369-385`) calls `planner.plan(...)` **every tick**, executes
  **only the first action**, and discards the rest of the imagined sequence.
  Next tick it replans from scratch on the fresh observation. (See the prior
  finding: "the agent moves one step and plans again.")
- **Continuations are uniform-random** (`planner.py:121`,
  `a = np.random.randint(output_size)`). Only the *first* action of each rollout
  is the candidate being scored; everything after it is a random walk.
- **First action is uniform over the valid set** (`planner.py:100`) — random
  shooting, *not* policy-guided. The policy `probs` is computed (the GRU memory
  must advance) but used only for the PPO importance-ratio log-prob
  (`agent.py:384`), not to bias the search.
- **The model is on-policy-trained, used off-policy.** The dynamics head is fit
  by PPO as an auxiliary loss — `z_pred → stop-grad(z_target)` + reward MSE,
  weighted by `world_model_coef` (`agents/ppo.py:631-649`) — on the *policy's
  own* transitions. The planner then queries it under *uniform-random* actions,
  which are out-of-distribution, so deep rollouts drift.
- **Config:** `brain.world_model.planner: {enabled, depth, samples, gamma}`
  (`config/planning_curiosity_v35.yaml`). Built in `agent.py:141-145`.

### 1.1 It already helps — and that's the point

The committed A/B (`docs/sample_planning_curiosity/`, same seed/world/3000 ticks,
v3.5 + PPO) shows the *current* planner + curiosity already changes behaviour
substantially vs. the head-off baseline:

| | Baseline | + planning & curiosity |
|---|---|---|
| Peak fitness | 38.7 | **57.8** (+49%) |
| Tiles explored / agent | 27.7 | **54.9** (+98%) |
| EAT attempts | 185 | **1,196** (×6.5) |
| Turning share of actions | 51% | 36% |

That a *random-shooting, random-continuation, 6×2* planner already lifts fitness
~50% is strong evidence the learned latent model carries useful signal — and
that a better **search** and better **rollout policy** should compound the gain.

### 1.2 Cost

Random shooting is `samples × depth` extra latent forward passes **per agent per
tick**. Measured on the 64×64 / pop-30 v3.5 world: the no-planner run ran
~8 ticks/s; `planner depth 2, samples 6` dropped it to ~1 tick/s (~8× slower).
Cost scales linearly with `samples × depth`, and the marginal value of extra
random rollouts is low (see §3).

---

## 2. Why this is weak (diagnosis)

1. **Random continuations destroy the signal.** Scoring "action `a0` followed by
   a random walk" is a high-variance, low-information estimate of `a0`'s value.
   With 9 actions and `depth-1` random steps, two rollouts of the *same* `a0`
   can score wildly differently, so `argmax` over a handful of samples is mostly
   noise past depth 1. This is the single biggest defect.
2. **Random shooting under-covers a discrete action space.** Uniformly sampling
   the first action means good and bad first actions get equal search budget;
   with `samples=6` and 9 actions, some actions may not be tried at all.
3. **Off-distribution model queries.** The dynamics head is trained on policy
   actions; uniform-random rollout actions push it off-distribution where its
   predictions are least reliable — exactly where the planner trusts it.
4. **No model-error awareness.** The score trusts `r̂` and `V` fully. Model-based
   control is known to *exploit* model errors (the dream-evolution code already
   warns about this); there's no uncertainty penalty or horizon discipline.
5. **Scale mismatch / no normalization.** The score sums per-step `r̂` (reward
   MSE target) and a bootstrapped `V`; their scales aren't reconciled, so the
   horizon term can dominate or vanish arbitrarily.
6. **All cost at inference, none distilled.** Planning runs every tick forever
   and never improves the *policy* — the plan overrides the action but the
   actor isn't trained to imitate the planner, so inference stays expensive and
   the policy alone (planner off) stays weak.
7. **Pure MPC, no commitment.** Replanning from scratch each tick is the most
   expensive possible schedule and yields jittery, incoherent behaviour
   (no temporal abstraction).

---

## 3. What the literature does (survey)

| Family | Idea | Pros | Cons / fit here |
|---|---|---|---|
| **Random shooting / PETS** (Chua et al. 2018) | sample action sequences, pick best; PETS adds model ensembles | dead simple; current code | high variance; needs many samples; **our continuations are random, worse than PETS** |
| **CEM / MPPI** (Williams 2017; PlaNet, Hafner 2019) | iteratively refit the action distribution toward high-return samples | far better sample-efficiency than shooting; standard in PlaNet | a few iterations × samples; discrete needs a categorical CEM |
| **Policy-prior rollouts** (Dreamer actor in imagination) | roll out the *policy* in latent space, not random actions | cheap, huge quality win, in-distribution | needs the policy usable inside imagination (we have it) |
| **MCTS over a learned model** (MuZero, Schrittwieser 2020) | value+policy-guided tree search in latent space | strongest decision-time planner | heavy; tree bookkeeping per agent per tick — too costly for 100s of agents |
| **Dreamer V1–V3** (Hafner et al.) | *learn* actor & critic by backprop/λ-returns through imagined latent rollouts; at runtime just run the actor | **no per-tick planning cost**, model distilled into policy, SOTA sample-efficiency | training-time complexity; needs differentiable imagination (our brain is numpy) |
| **MBPO** (Janner 2019) | use short model rollouts to generate extra training data for a model-free learner | improves the policy with the model without decision-time planning | adds a replay/dyna loop |

**Takeaways that fit this codebase (numpy genome brains + per-agent PPO, 100s of
agents, emergence-first):**
- The cheapest, highest-leverage fix is **replace random continuations with
  policy-guided rollouts** and **bias the first action by the policy** (Dreamer's
  imagination rollout, used as an MPC scorer). Same cost, far less variance.
- **Categorical CEM/MPPI** over the first-action (or short-sequence) distribution
  is the natural "better search" upgrade once rollouts are informative.
- The **right long-term answer** is **Dreamer-style imagination learning**: train
  the actor/critic in imagination so the *policy itself* plans, and decision-time
  cost goes back to a single forward pass. This matches the project's existing
  "train the world model, then act" philosophy (`agents/dream.py`).
- **MuZero-style MCTS** is powerful but the wrong cost profile for a world with
  hundreds of concurrent agents.

---

## 4. Proposal — staged upgrade

All phases are **config-gated and default-off / behaviour-preserving**; the
current random-shooting planner remains available
(`planner.strategy: shooting`).

### Phase P1 — make the existing rollouts informative (cheap, do first)

**Status: IMPLEMENTED & MEASURED.** Built in `agents/planner.py`
(`strategy: policy_shooting`, `first_action`, `topk`, `normalize`, `commit`),
unit-tested in `tests/test_planner.py`, and A/B-measured in
`docs/sample_planning_p1/`. **Result:** policy-guided *continuations* beat
random shooting by **+21% peak fitness / +26% planting at ≈equal cost** — *but
only when the first action stays exploratory*. Biasing the first action toward
the (immature, from-scratch) policy made agents passive (WAIT 22%→39%, fitness
−21%). **Lesson:** policy-guide the continuations, keep the first action
exploratory while the policy is still learning. Recommended config:
`config/planning_p1_v35.yaml`. The legacy `shooting` remains the default.

Small, local changes in `agents/planner.py` (+ a few config knobs). No genome or
training changes. Expected: large quality gain at **equal or lower** cost.

1. **Policy-guided continuations.** Replace the uniform `randint` at
   `planner.py:121` with a sample from the brain's own policy at the imagined
   state. The policy head reads **only the GRU hidden state**
   (`logits = h_next @ policy_head.W + b`, `agents/brain/__init__.py:208`), so
   after `h_sim = brain._gru_step(z', h_sim)` we can take `probs` straight from
   `h_sim` — no `z` needed. This turns each rollout into "what happens if I take
   `a0` then act like myself" — an in-distribution, low-variance value estimate
   (Dreamer's imagination rollout). Needs only a tiny
   `brain.policy_from_hidden(h, action_mask=None)` helper exposing the existing
   masked softmax over `h`.
2. **Policy-biased first action.** Draw the `samples` first actions from the
   policy (optionally top-k) instead of uniform, so search budget concentrates
   on plausible actions (`planner.py:100`).
3. **Reward + value normalization.** Track running mean/std of `r̂` and `V`
   (reuse the Welford helper from `agents/curiosity.py`) and z-score them before
   summing, so the horizon bootstrap and per-step rewards are commensurable.
4. **Plan commitment (control horizon).** Add `commit: k` — execute the best
   rollout's first `k` actions before replanning. Cache the remaining actions on
   the agent; replan early if an interaction action fails or energy crosses a
   threshold. Cuts planning cost ~`k×` and yields more coherent movement.
5. **Antithetic / common-random-number scoring.** Score each candidate first
   action with the *same* set of continuation seeds to reduce comparison
   variance (cheap variance reduction).

Config (additions under `brain.world_model.planner`):
```yaml
planner:
  enabled: true
  strategy: policy_shooting   # shooting (legacy) | policy_shooting | cem
  depth: 5                    # can grow once rollouts are informative
  samples: 12
  gamma: 0.95
  commit: 1                   # control horizon (k≥1)
  first_action: policy_topk   # uniform | policy | policy_topk
  topk: 3
  normalize: true             # z-score r̂ and V before summing
```

**Expected effect:** lower-variance value estimates → the planner actually
prefers good actions several steps out; with `commit>1`, **faster** than today.
**Risk:** low (local, reversible, gated). **Effort:** ~half a day + an A/B.

### Phase P2 — better search + model-error discipline (medium)

**Status: PARTIALLY IMPLEMENTED & MEASURED.** Categorical CEM (`strategy: cem`)
and TD(λ) returns (`lam`) are built (`agents/planner.py`), unit-tested
(`tests/test_planner.py`), and A/B-measured (`docs/sample_planning_p2/`):
**CEM is the strongest controller — peak fitness +32% over baseline and +8% over
P1** (eating +59%, planting +72%) at a ~20% throughput cost. The third item,
**model-error discipline via a dynamics ensemble, is DEFERRED** — it needs an
append-only genome-length change + migration, out of scope for the cheap P2 pass.
Recommended config: `config/planning_p2_v35.yaml`.

1. **Categorical CEM/MPPI** (`strategy: cem`): maintain a per-step categorical
   action distribution; each iteration sample `N` sequences, keep the top-`e`
   elites by return, refit the distribution; after `I` iterations act on the
   first action of the elite mean. 2–3 iterations of CEM typically beat
   shooting with the same total samples (PlaNet).
2. **Uncertainty / horizon discipline.** Penalize the rollout score by a proxy
   for model error so the planner stops trusting long rollouts:
   - cheap: a per-agent EMA of the dynamics head's *training* latent-MSE
     (already computed in `ppo.py:644`) as a confidence scalar that shrinks
     `gamma` or caps `depth`;
   - stronger: a **small dynamics ensemble** (2–3 heads in the genome) and
     penalize by inter-head disagreement (PETS/MBPO). Adds genome length;
     batched cheaply in `ppo.py`.
3. **λ-returns in imagination** (DreamerV2): replace the single end-of-horizon
   bootstrap with a TD(λ) mixture over the imagined trajectory for a
   lower-bias/variance trade-off.

**Expected effect:** robust gains on harder/longer-horizon tasks; avoids the
"model exploitation" failure. **Risk:** medium (ensemble = genome change →
migration, like the v3→v3.5 append-only bump). **Effort:** ~2–3 days.

### Phase P3 — learn the policy *in imagination* (the real fix, large)

**Status: IMPLEMENTED (EXPERIMENTAL) & MEASURED.** `TorchBrainMirror.imagine_loss`
in `agents/ppo.py` adds a Dreamer-style actor-critic-in-imagination auxiliary
loss (config `learning.ppo.imagination`, default **off**), unit-tested in
`tests/test_imagination.py` and A/B-measured in `docs/sample_planning_p3/`.
**Result:** with the planner OFF in both arms, imagination training lifted peak
fitness **+25%** (and lifespan +11%, planting +38%, idle WAIT 31%→15%) — nearly
matching the best decision-time planner (CEM, +32%) **without its per-tick cost**
(the slowdown is one-time training, not per-decision). Recommended config:
`config/planning_p3_v35.yaml`. Trains the critic on model-predicted returns, so
it pairs with the deferred P2 model-error discipline; kept experimental/off.

Adopt the **Dreamer** objective in `agents/ppo.py`: periodically roll the actor
forward in the **latent** world model for `H` steps, compute λ-returns from the
critic, and update the actor to maximize imagined return (and the critic to
predict it). At runtime the **planner can be turned off** — the policy already
embodies the plan, so decision-time cost is one forward pass again.

- Fits the existing split perfectly: PPO already trains the dynamics head and
  critic; this adds an imagination-rollout loss for the actor.
- Removes the per-tick `samples×depth` tax (the ~8× slowdown) while *keeping*
  the behavioural benefit — planning amortized into the weights.
- Pairs naturally with the offline `PopulationWorldModel` / `agents/dream.py`
  (same "evolve/learn inside the model, then act" idea, now for the lifetime
  policy).

**Expected effect:** the headline upgrade — better policies *and* cheap
inference. **Risk:** higher (new training loop; imagination can diverge without
the P2 error discipline). **Effort:** ~1–2 weeks; do P1+P2 first.

---

### Warmup scheduling — model-readiness gating (IMPLEMENTED)

The multi-seed replication showed P2/P3 do not help from a cold start, and the
likely cause is timing: **at tick 0 the world model is untrained, so a
model-heavy planner (CEM) or imagination training "shoots off" from a garbage
model.** Fix: gate them on model readiness. The planner runs the cheap
exploratory `warmup_strategy` (default `policy_shooting`) until
`planner.warmup_ticks`, *then* switches to the configured `strategy` (e.g. CEM);
imagination is likewise gated by `learning.ppo.imagination.warmup_ticks`. By the
switch the world model has trained on thousands of diverse transitions. The
agent passes the live world tick to both (see `agents/agent.py`); unit-tested in
`tests/test_planner.py` / `tests/test_imagination.py`. Recommended schedule lives
in `config/planning_scheduled_v35.yaml` (P1 for 5k ticks → CEM + imagination).

## 5. Recommendation & sequencing

1. **P1 now.** Policy-guided rollouts + policy-biased first action +
   normalization + `commit`. Biggest bang-for-buck; local; reversible.
2. **P2 next** if longer horizons are wanted: categorical CEM + a model-error
   penalty (start with the cheap EMA proxy before an ensemble).
3. **P3** as the strategic direction: imagination-trained actor so planning is
   *learned*, not paid for every tick.

Keep `strategy: shooting` as the documented baseline so every step is a clean
A/B against the current committed result.

## 6. Evaluation plan

Reuse the existing harness (`docs/sample_planning_curiosity/` is the template):
matched runs (same `--seed`, world, tick budget) across
`strategy ∈ {off, shooting, policy_shooting, cem}`, scored with
`scripts/analyze_logs.py`. Primary metrics:

- **Competence:** mean/peak fitness, avg lifespan, EAT success, seeds planted.
- **Exploration:** unique tiles/agent, strategy entropy.
- **Decision quality:** action mix (turning% down, MOVE/interaction% up).
- **Cost:** ticks/s (the planner tax) — P1 with `commit>1` and P3 should
  *improve* this vs. today.
- **Ablations:** depth sweep, `samples` sweep, `commit` sweep, normalization
  on/off; for P2, error-penalty on/off and ensemble size.

A change ships only if it beats `strategy: shooting` on competence **without**
a disproportionate ticks/s regression (P1/P3 should be Pareto-better).

## 7. Risks & mitigations

- **Model exploitation** (planner finds high-`r̂` fantasies): short horizons,
  reward/value normalization (P1), uncertainty penalty + λ-returns (P2).
- **Out-of-distribution rollouts:** policy-guided continuations (P1) keep
  imagination near the data the model was trained on.
- **Cost blowup:** `commit` (P1) and imagination-trained actor (P3) both reduce
  per-tick cost; all planning stays config-gated and off by default.
- **Genome/migration churn** (P2 ensemble, any new head): follow the established
  append-only genome bump + `adapt_loaded_genome` migration used for v3→v3.5.

## 8. Backwards compatibility

- All new behaviour is under `brain.world_model.planner` with safe defaults;
  `strategy: shooting` reproduces today's planner exactly.
- P1/P2 touch only `agents/planner.py` (+ a small brain helper for the
  latent-only policy path) — no observation or action-space change, no genome
  change except the optional P2 ensemble (migratable).
- P3 adds a training loss in `agents/ppo.py`, gated by a config flag; with it
  off, learning is unchanged.

---

### Appendix A — P1 policy-guided rollout (pseudo-code)

```python
def _rollout(self, brain, h, first_action, rng):
    h_sim, score, discount = h, 0.0, 1.0
    a = first_action
    z = None
    for k in range(self.depth):
        z, r = brain.predict_next_latent(h_sim, a)
        score += discount * _znorm_r(r)          # normalized reward
        discount *= self.gamma
        h_sim = brain._gru_step(z, h_sim)
        probs = brain.policy_from_hidden(h_sim)      # NEW: in-distribution
        a = int(rng.choice(brain.output_size, p=probs))
    score += discount * _znorm_v(brain._value(z, h_sim))
    return score
```

The only new brain surface is `policy_from_hidden(h)` — the masked policy softmax
as a function of an imagined `h` (the exact matmul `forward` already runs at
`agents/brain/__init__.py:208`), so it is a few lines on `Brain` (inherited by
`BrainV3`).
