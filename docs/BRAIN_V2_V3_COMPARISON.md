# Brain v2 vs Brain v3 — A Complete Technical Comparison

**Audience:** a computer-science student comfortable with linear algebra and
basic probability. Every formula used by the actual code is derived or
explained here; file references point at the implementation.

> **Note:** **Brain v3.5** (implemented; `brain.version: 3.5`) keeps the v3
> maths below unchanged and only *widens its I/O* for the World-upgrade W4
> social layer — the observation vector grows from 72 to 78 (six new senses)
> and the action set from 8 to 9 (a SIGNAL action with a pheromone field).
> Every derivation here still holds for v3.5 with **x** ∈ ℝ⁷⁸ and 9 action
> logits; only the state encoder input (22→28) and the policy head output
> (8→9) change size. Full design + as-built notes: `BRAIN_V3_PROPOSAL.md` §8.

---

## 1. The problem both brains solve

Each tick, an agent receives an observation vector **x** ∈ ℝ⁷² and must:

1. choose one of 8 discrete actions (a **policy** π(a|history)), and
2. estimate how good its situation is (a **value** V(history) ∈ ℝ),
   which the learning algorithms use for credit assignment.

The observation layout (`utils/agents/perception.py`, named slices in
`agents/brain/spec.py`):

| Slice | Content | Size |
|---|---|---|
| x[0:8] | agent state: energy ratio, age ratio, facing one-hot (4), inventory-space flag, metabolism | 8 |
| x[8:58] | egocentric 5×5 vision grid, 2 features per tile (type, value), rotated so "up" is always "ahead" | 50 |
| x[58:66] | stimulus signals: food_on_tile, seed_on_tile, food_ahead, resource_ahead, nearest_food_prox, food_dir_match, energy_urgency, can_interact | 8 |
| x[66:72] | inventory summary | 6 |

A single observation is not enough to act well (food seen two ticks ago may
be behind you now), so both brains are **recurrent**: they carry a hidden
state **h** between ticks. Both are also tiny on purpose — they must run for
hundreds of agents every tick and fit in an evolvable flat genome.

Both brains share the exact same outer loop (`agents/brain/__init__.py`):

```
z   = encode(x)              # differs between v2 and v3
h'  = GRU(z, h)              # same equations, different sizes
ℓ   = h'·W_pol + b_pol       # action logits, 8 numbers
ℓ   = mask(ℓ); ℓ += instincts(x)·fade(age)
π   = softmax(ℓ / τ)
V   = value(z, h')           # differs between v2 and v3
a   ~ π                      # sample an action
```

---

## 2. The shared core: GRU memory

Both versions use a Gated Recurrent Unit (`agents/brain/modules.py:gru_step`).
Given input **z** and previous hidden state **h** (size H):

```
r  = σ(z·W_r,in + h·W_r,hid + b_r)        reset gate    ∈ (0,1)^H
u  = σ(z·W_u,in + h·W_u,hid + b_u)        update gate   ∈ (0,1)^H
h̃  = tanh(z·W_h,in + (r ⊙ h)·W_h,hid + b_h)   candidate state
h' = (1 − u) ⊙ h + u ⊙ h̃                  new hidden state
```

where σ is the logistic sigmoid and ⊙ is element-wise multiplication.

Intuition: each of the H memory cells interpolates between *keeping* its old
value (uᵢ ≈ 0) and *overwriting* it with new information (uᵢ ≈ 1). The reset
gate r lets the candidate ignore stale memory when computing what to write.
This gating is what lets gradients (and evolved behaviours) span many ticks
without vanishing as fast as in a plain RNN.

Parameter count of a GRU with input Z and hidden H — three gates, each with
an input matrix, a hidden matrix, and a bias:

```
P_GRU(Z, H) = 3·(Z·H + H² + H)
```

---

## 3. Brain v2 — dense MLP perception (`version: 2`, default)

### 3.1 Architecture

```
x (72) ──→ tanh(x·W_e + b_e) ──→ z (32) ──→ GRU(32) ──→ h' (32)
                                                  ├─→ logits = h'·W_pol + b_pol   (8)
                                                  └─→ V      = h'·w_val + b_val   (1)
```

One dense ("fully connected") layer reads the **entire** observation at once:
every one of the 72 inputs has its own learned weight to every one of the 32
encoder units. Vision is not treated specially — tile (2,3)'s features are
just inputs #34 and #35.

### 3.2 Parameter count (must equal `Brain.calculate_weight_count()` = 8,873)

```
Encoder:      72·32 + 32                 = 2,336
GRU(32→32):   3·(32·32 + 32² + 32)       = 6,240
Policy head:  32·8 + 8                   =   264
Value head:   32·1 + 1                   =    33
                                   Total = 8,873
```

### 3.3 Properties — what this buys and what it costs

**Strengths**
- Maximum flexibility: any linear function of the full observation is
  representable in the first layer.
- Cheapest possible forward pass (one 72×32 matmul ≈ 2.3k multiply-adds for
  perception).
- Small genome → easy for Gaussian-mutation evolution to search.

**Weaknesses**
- **Position-bound vision.** The weight connecting "food at tile (2,3)" to
  any encoder unit is a *different parameter* from the one for "food at tile
  (2,4)". The network must learn what food means **25 separate times**, once
  per tile. Of the 2,304 encoder weights, 1,600 (50×32) exist only to
  memorise tile positions.
- **Fixed input size.** Change the vision radius (a trait that exists in the
  genome!) and the weight matrix no longer fits — the architecture cannot
  use it.
- **Uniform processing.** Every tile gets the same compute and the same
  static weighting regardless of context; the network cannot "look harder"
  at food when hungry except through downstream nonlinearity.
- **Myopic critic.** V is a *linear* readout of h' only. If the GRU didn't
  store some fact, the critic cannot see it, even when it is right there in
  the current observation.

---

## 4. Brain v3 — attention perception (`version: 3`, opt-in)

### 4.1 Architecture (`agents/brain/v3.py`)

```
                          ┌────────────── 25 tile tokens ──────────────┐
vision (50) ─ reshape ──→ │ t_i = [type_i, value_i, row_i, col_i] ∈ ℝ⁴ │
                          └──────── shared embed: e_i = tanh(t_i·W_E + b_E) ∈ ℝ⁸
x[0:8] ┐
x[58:66]├─ concat (22) ──→ s = tanh(f·W_S + b_S) ∈ ℝ⁴⁰        (state encoder)
x[66:72]┘                          │
                                   │ q = s·W_q ∈ ℝ⁸            (attention query)
                                   ▼
            α_i = softmax_i( (e_i·W_k)·q / √8 )                (attention weights)
            v_att = Σᵢ α_i (e_i·W_v) ∈ ℝ⁸                      (pooled vision)

z = [s ‖ v_att] ∈ ℝ⁴⁸  ──→  GRU(48)  ──→  h' ∈ ℝ⁴⁸
                                   ├─→ logits = h'·W_pol + b_pol            (8)
                                   └─→ V = tanh([z‖h']·W₁ + b₁)·W₂ + b₂     (1)
```

### 4.2 Step-by-step math

**Tokenisation with positional encoding.** Each tile i of the 5×5 grid
becomes a 4-vector: its two observation features plus its grid coordinates
normalised to [−1, 1] (`make_positional_encoding`). The coordinates are
**constants**, not learned. Why are they needed? Because the attention sum
below is *permutation-invariant* — without position features, "food on my
left" and "food on my right" would produce identical outputs. (Test
`test_positional_encoding_distinguishes_tiles` verifies this matters.)

**Shared embedding.** Every token goes through the *same* 4×8 matrix:

```
e_i = tanh(t_i·W_E + b_E),   i = 1…25,   W_E ∈ ℝ⁴ˣ⁸
```

This is the key inductive bias: **what food looks like is learned once**, in
40 parameters, instead of 25 times in 1,600. Formally, perception becomes
*position-equivariant*: moving an object to another tile changes only the
positional part of its token, not the machinery that interprets it. It also
means the same weights work for a 7×7 or 9×9 grid — only the number of
tokens changes — finally making the evolvable `vision_radius` trait usable,
and pointing toward camera-like inputs for robotics.

**State encoder and query.** The 22 non-vision features (agent state +
stimulus + inventory) become s = tanh(f·W_S + b_S) ∈ ℝ⁴⁰, and the attention
query is q = s·W_q. Because q depends on the agent's *internal condition*,
the network can learn context-dependent looking: a hungry agent (high
energy_urgency in f) can emit a query that matches food-like tile
embeddings; a full agent can attend to terrain.

**Scaled dot-product attention** (single head, single query):

```
k_i = e_i·W_k        (keys)
v_i = e_i·W_v        (values)
score_i = (k_i·q) / √8
α = softmax(score)                       Σ α_i = 1, α_i ≥ 0
v_att = Σᵢ α_i v_i
```

*Why divide by √E (E = 8)?* If the components of k and q are roughly
independent with variance 1, then Var(k·q) = E, i.e. dot products grow like
√E. Unscaled, scores would saturate the softmax (one α_i ≈ 1, the rest ≈ 0)
and gradients through the softmax would vanish. Dividing by √E keeps score
variance ≈ 1 regardless of embedding size — the same trick as in
Transformers.

The output v_att is a **content-addressed weighted average** of what the
agent sees, where the weighting is decided per-tick by the agent's own
state. Compare v2, where the mixing weights are frozen into W_e at birth.

**Value MLP over [z ‖ h'].** The critic input concatenates the *current*
latent z with the memory h', then applies a hidden layer:

```
V = tanh([z‖h']·W₁ + b₁)·W₂ + b₂,   W₁ ∈ ℝ⁹⁶ˣ¹⁶, W₂ ∈ ℝ¹⁶ˣ¹
```

Two upgrades over v2's `V = h'·w + b`: (1) the critic sees the present
state directly — it no longer depends on the GRU having memorised it; (2)
the tanh hidden layer makes V a *nonlinear* function of its inputs, so it
can represent things like "value is high only if food is near AND energy is
low", which a linear readout cannot.

### 4.3 Parameter count (must equal 17,337 — verified in `tests/test_brain_v3.py`)

```
State encoder:  22·40 + 40                   =   920
Tile embedding: 4·8 + 8                      =    40
Attention:      W_q 40·8 + W_k 8·8 + W_v 8·8 =   448
GRU(48→48):     3·(48·48 + 48² + 48)         = 13,968
Policy head:    48·8 + 8                     =   392
Value MLP:      96·16 + 16 + 16·1 + 1        = 1,569
                                       Total = 17,337
```

Note where the budget went: the *perception* shrank (2,336 → 1,408) while
becoming smarter; the extra capacity is almost all **memory** (GRU 6,240 →
13,968).

### 4.4 Compute cost per tick (approximate multiply-adds)

| Stage | v2 | v3 |
|---|---|---|
| Perception | 72·32 ≈ 2.3k | 22·40 + 25·4·8 + 40·8 + 25·8·8·2 + 25·8 ≈ 5.3k |
| GRU | 3·(32+32)·32 ≈ 6.1k | 3·(48+48)·48 ≈ 13.8k |
| Heads | ≈ 0.3k | ≈ 1.9k |
| **Total** | **≈ 8.7k** | **≈ 21k** |

v3 is ~2.4× the FLOPs — both are trivially cheap in absolute terms
(microseconds per agent in NumPy).

### 4.5 The honest trade-off: evolvability

Pure neuroevolution searches weight space by Gaussian mutation. The expected
behavioural effect of isotropic noise grows with dimension, and useful random
directions get rarer; empirically, mutation-based search degrades as the
genome grows. Measured in this codebase (1000 ticks, seed 42, default
config):

| Mode | v2 (8.9k params) | v3 (17.3k params) |
|---|---|---|
| RL + Lamarckian | comparable survival | comparable survival |
| Pure neuroevolution | healthy | noticeably weaker early populations |

This is *why v2 remains the default*: the v2-vs-v3 comparison under
identical conditions is itself one of the experiments the sandbox exists to
run.

---

## 5. The two learning algorithms

Architecture says what the network *can* compute; the learner decides what
it *does* learn during a lifetime. Both learners are Lamarckian: after every
update the weights are packed back into the genome
(`ParamSpec.pack`, `agents/brain/spec.py`), so offspring inherit them.

### 5.1 A2C learner (`learning.algorithm: a2c`, `agents/learning.py`) — legacy

**Data:** single transitions (x_t, h_t, a_t, r_t, x_{t+1}, h_{t+1}, done)
sampled *uniformly at random* from a replay buffer.

**Advantage (TD(0)):**

```
A_t = r_t + γ·V(x_{t+1}, h_{t+1})·(1 − done_t) − V(x_t, h_t)
```

This is the one-step "surprise": how much better the step turned out than
the critic expected. It is cheap and low-variance but biased by whatever
errors V currently has.

**Policy update.** From the policy-gradient theorem,
∇J = E[∇log π(a|s)·A]. For a softmax over logits ℓ = h'·W + b, the gradient
of −log π(a) with respect to the logits has the classic closed form:

```
∂(−log π(a)) / ∂ℓ_j = π(j) − 1[j = a]
```

so the code's update (`_learn_vectorized_numpy`) is literally:

```
g = (π − onehot(a)) · A · lr          # (8,)
W_pol ← W_pol − h'ᵀ·g                 # outer product
```

**Critic update** minimises ½(V − target)², again only through the output
layer: ∂/∂w = (V − target)·h'.

**The crucial limitation:** these manual gradients stop at the heads. The
encoder and GRU receive **no learning signal at all** — during a lifetime
only the last layer adapts; the representation underneath can only change
across generations, by mutation. "Learning" in v2-A2C is tuning an 8×32
readout on top of a frozen, evolved feature extractor.

Two further structural problems:
- *Stale hidden states:* h_t was recorded under old weights; after updates
  it no longer equals what the current network would have computed.
- *Silently off-policy:* uniformly replayed transitions came from an older
  policy, but the plain policy-gradient formula assumes on-policy data —
  there is nothing bounding the damage a misestimated step can do.

### 5.2 PPO sequence learner (`learning.algorithm: ppo`, `agents/ppo.py`) — Phase 3b

Four upgrades, each fixing one of the problems above.

**(1) Full-network backprop.** A persistent torch mirror holds every
ParamSpec tensor as a `torch.nn.Parameter`; the forward pass is re-expressed
in torch (`TorchBrainMirror`), so autograd differentiates the whole
computation graph — value MLP, policy head, GRU (through time), attention,
embeddings, encoders. Optimisation uses Adam with gradient-norm clipping.
*Lifetime learning finally reaches the representation.* The test
`test_all_parameter_groups_receive_gradients` asserts that encoder/attention
and GRU weights actually change.

**(2) Sequence replay.** Experience is stored as time-ordered chunks of
L = 8 steps together with the hidden state h₀ captured when the chunk began
(`SequenceChunk`). During learning the GRU is *re-run* over the chunk:

```
h_1 = GRU(encode(x_1), h_0), …, h_L = GRU(encode(x_L), h_{L−1})
```

so gradients flow backward through time across the chunk
(truncated BPTT, horizon L). The stored h₀ does go slightly stale as weights
change — the standard "stored state" compromise (cf. R2D2) — but inside the
chunk all states are recomputed fresh, which random single-transition replay
can never do.

**(3) GAE(λ) advantages** (`compute_gae`). Define the one-step TD error

```
δ_t = r_t + γ·V_{t+1}·(1 − done_t) − V_t .
```

Every k-step advantage estimator
Â(k) = δ_t + γδ_{t+1} + … + γ^{k−1}δ_{t+k−1} trades bias (small k → trusts
the critic) against variance (large k → trusts sampled rewards). GAE takes
the exponentially-weighted average of *all* of them, which telescopes to:

```
Â_t = Σ_{k≥0} (γλ)^k · δ_{t+k}
    = δ_t + γλ·(1 − done_t)·Â_{t+1}      (one backward pass, as coded)
```

λ = 0 recovers TD(0) (what A2C used); λ = 1 recovers Monte-Carlo returns.
The default λ = 0.95 keeps most of the variance reduction with little bias.
Value targets are R_t = Â_t + V_t, and advantages are normalised to zero
mean / unit variance across the batch.

**(4) PPO clipped objective.** Replayed data was generated by an older
policy π_old (whose log-probs were recorded at acting time —
`Brain.decide_with_logprob`). Importance sampling says reweight by

```
ρ_t = π_new(a_t|s_t) / π_old(a_t|s_t) = exp(log π_new − log π_old),
```

but raw ratios can explode. PPO instead maximises the **clipped surrogate**:

```
L_policy = −E[ min( ρ_t·Â_t,  clip(ρ_t, 1−ε, 1+ε)·Â_t ) ],   ε = 0.2
```

Read it case by case: if Â_t > 0 (action was good), the objective stops
rewarding increases of ρ beyond 1+ε — the gradient *vanishes* there, so the
policy cannot over-commit to one lucky sample. If Â_t < 0, decreases of ρ
below 1−ε similarly stop counting. The `min` makes the bound pessimistic
(clipping never helps the objective). Net effect: **every update is small
and trustworthy**, even on stale or instinct-influenced data — which is
exactly the situation here, since young agents act partly on fading instinct
biases that the learner's π_new does not model. The test
`test_clipping_bounds_off_policy_updates` corrupts stored log-probs to make
ρ astronomically large and asserts the policy barely moves.

**Total loss** (per valid step, padding masked out):

```
L = L_policy + c_v·½(V − R)² − c_e·H(π),   c_v = 0.5, c_e = 0.01
H(π) = −Σ_a π(a)·log π(a)                  (entropy bonus → exploration)
```

### 5.3 Learner comparison at a glance

| | A2C (legacy) | PPO (Phase 3b) |
|---|---|---|
| Trains | output heads only | **every parameter** |
| Data | random single transitions | time-ordered L=8 chunks |
| Recurrence in learning | none (stored, stale h) | BPTT through the chunk |
| Advantage | TD(0) | GAE(λ = 0.95) |
| Off-policy safety | none | clipped ratio (ε = 0.2) |
| Optimiser | manual SGD | Adam + grad-norm clip |
| Exploration term | entropy (reported only) | entropy in the loss |
| Backend | NumPy (torch fast-path) | torch (falls back to A2C without it) |
| Cost per update | ~µs | ~ms (CPU); use `compute_device: cuda` at scale |
| Works with | v2 + v3 | v2 + v3 |

### 5.4 Why this combination matters scientifically

The project's central question is the interaction of *lifetime learning*
with *generational evolution* (Lamarckian inheritance). Under A2C, "lifetime
learning" was confined to a last-layer readout, so the experiment was much
weaker than advertised. With PPO, learned changes reach the representation
itself, are inherited via the genome, and are then mutated and selected —
the full Baldwin/Lamarck loop. The clean toggles
(`evolution.mode`, `learning.algorithm`, `brain.version`) make the 2×2×2
grid of conditions directly comparable under identical environments.

---

## 6. Empirical snapshot (1000 ticks, seed 42, default config, max-pop 100)

| Configuration | Alive at tick 1000 |
|---|---|
| v2 + A2C (RL mode) | ~64 |
| v2 + PPO (RL mode) | ~30 |
| v2, pure neuroevolution | ~16–33 (across seeds/runs) |
| v3 + A2C (RL mode) | ~100 |
| v3 + PPO (RL mode, 200-tick run) | viable (~31, reproducing) |

Caveats a careful reader should apply: parallel-mode runs are
non-deterministic (thread scheduling shares one RNG), instinct fading makes
all of these *harder* than pre-Phase-2 numbers, and PPO's hyperparameters
are defaults, not tuned. These are smoke signals, not results — the rigorous
sweep is the next research step.

---

## 7. Choosing a configuration

| You want… | Use |
|---|---|
| The controlled baseline / fastest runs | `brain.version: 2`, `algorithm: a2c` |
| Smarter perception, richer critic | `brain.version: 3` |
| Real lifetime learning (representation included) | `algorithm: ppo` |
| The full Phase-3 stack | `version: 3` + `ppo` (budget CPU, or set `learning.compute_device: cuda`) |
| Pure-evolution science | `evolution.mode: neuroevolution` (learner unused) |

All switches live in `config/default.yaml` with inline documentation.
