# Brain v4 — architecture proposal for emergent behaviour

**Status: IMPLEMENTED.** §4 is built and shipping; §9 below documents what
changed between the design and the as-built code, and §10 reports the
validation campaign. Brain v3.6 (`BRAIN_V3_PROPOSAL.md` §9) is *absorbed*
into v4 (§4.7) rather than skipped, and finally exists in code.

- **V4.0** (scoring integrity) — `agents/scoring.py`, `config/v4_baseline.yaml`
- **V4.1-V4.4** (the architecture) — `agents/brain/v4.py`,
  `agents/brain/spec.py`, `agents/ppo.py`, `config/v4_full.yaml`
- Tests — `tests/test_scoring_v40.py`, `tests/test_brain_v4.py`,
  `tests/test_ppo_v4.py`

**Author:** Karan Vasa · **Date:** September 2026
**Inputs reviewed:** the codebase at `eaaf53d`, `docs/PAPER.md`,
`docs/BRAIN_V3_PROPOSAL.md`, `docs/WORLD_UPGRADE_PROPOSAL.md`,
`docs/PLANNING_PROPOSAL.md`, the sibling project `worldmodel-robotics`
(T1/T2/P0 campaigns), and a fresh 24-run baseline campaign (§2).

---

## 0. The one-paragraph version

The brain that ships today (v3.5) is a competent little actor-critic, and the
project's measured conclusions about *planning* are sound. But the system as a
whole cannot produce the emergent behaviour it was built for, for four reasons
that are arithmetic rather than aesthetic: **(a)** the reward function is a
420-line hand-written behaviour policy, so anything "emergent" is downstream
of a line that asked for it; **(b)** the learner's credit-assignment horizon is
8 ticks of BPTT and `0.95^k` discount, while the world's signature behaviour —
plant a seed, eat the fruit — pays out at ~160 ticks, so cultivation is
*provably* unlearnable by gradient; **(c)** there is no memory of place, so
"go back to the patch" is not merely hard to learn, it is not representable;
**(d)** the cheapest action in the game is `SIGNAL`, and §2.3 measures the
population discovering exactly that. v4 proposes five changes: strip the
hand-written objective, give the network a second recurrent state with
**evolved time constants**, a **dual-discount critic**, an **episodic place
memory with exact path integration**, and a **drive basis whose weights live
in the genome** — i.e. stop writing the reward function and start evolving it.

---

## 1. What exists today, precisely

### 1.1 Version map (and what "v3.6" actually is)

| Brain | Obs | Actions | State-encoder inputs | Params (no WM / +WM) | Status |
|---|---|---|---|---|---|
| v2 | 72 | 8 | — (dense MLP) | 8,873 | legacy baseline, `brain.version: 2` |
| v3 | 72 | 8 | 22 | 17,337 | `brain.version: 3` (default) |
| **v3.5** | **78** | **9 (+SIGNAL)** | **28** | **17,626 / 21,099** | **`brain.version: 3.5` — the newest brain that exists in code** |
| v3.6 | 79 | 9 | 29 | 17,666 | **designed, not built** — `BRAIN_V3_PROPOSAL.md` §9 |

`v3.6` is a one-line observation break (`nearest_agent_kin`, index 78) with a
birth-time genome fingerprint behind it. It has a complete written design and
zero lines of code: `grep -rn "3\.6" --include='*.py' .` returns nothing.
So "the current structure of brain v3.6" is, as built, **v3.5** — and v4
adopts v3.6 §9.4 verbatim as part of §4.7 rather than leaving it stranded.

### 1.2 The v3.5 forward pass, as written

`agents/brain/v3.py` (`BrainV3`), with the genome layout declared once in
`agents/brain/spec.py::build_brain_v3_param_spec`:

```
  observation (78)
    |
    +-- vision 5x5x2 (50), egocentric, rotated so "ahead" is the top row
    |     tokens_i = [ type_i , value_i , pos_row_i , pos_col_i ]   i = 1..25
    |     t_i      = tanh( tokens_i W_emb + b_emb )         in R^E    E = 8
    |
    +-- state block (28) = agent_state(8) || stimulus(8) || inventory(6) || EXTRA(6)
          s        = tanh( state W_s + b_s )                in R^S    S = 40

  attention (one head, query from the agent's own internal state):
          q     = s W_q                                     in R^E
          k_i   = t_i W_k ,   v_i = t_i W_v
          alpha = softmax_i( k_i . q / sqrt(E) )
          e     = sum_i alpha_i v_i                         in R^E

  latent:  z  = [ s || e ]                                  in R^48
  memory:  h_t = GRU( z_t , h_{t-1} )                       in R^H    H = 48
  policy:  logits = h_t W_pi + b_pi   (masked; + fading instinct biases)
  value :  V = tanh([z || h] W_1 + b_1) W_2 + b_2           in R^1
  world model (optional):
           d       = tanh( [ h || onehot(a) ] W_d + b_d )
           zhat'   = d W_z + b_z ,   rhat = d W_r + b_r
```

The shared 4->8 tile embedding (40 parameters, used 25 times) is the good idea
in this architecture: perception is position-equivariant and scales to a larger
vision radius with the *same* weights. The single state-conditioned query is
the second good idea. Neither is what limits the system.

### 1.3 The learner

`agents/ppo.py::PPOSequenceLearner`: sequence chunks of `seq_len = 8`, GAE(λ),
clipped PPO, full-network backprop through a torch mirror, a 1-step
world-model auxiliary loss (with a stop-gradient on the target latent), the
optional multi-step loss (M3) and imagination (P3), and a Lamarckian write-back
of trained weights into `genome.weights` so offspring inherit them.

### 1.4 Where behaviour actually comes from

This is the part that matters for an emergence claim. Four separate
hand-written sources shape what an agent does, and only one of them is the
network:

| Source | Location | What it encodes |
|---|---|---|
| Instinct logit biases | `agents/brain/instincts.py` | `PICK_UP +1.5`, `EAT +1.0`, `USE +0.5`, hunger-scaled `EAT +3.0`, and a direction-aware **turn-toward-food** routine. Fades linearly to 0 at `fade_age = 150`. |
| Dense reward shaping | `utils/agents/learning_utils.py::RewardShaper` (~420 lines) | movement bonus, new-tile bonus, anti-backtrack, anti-revisit, anti-spin, turn-toward-food, eat bonuses, wait penalties |
| "Behaviour economics" | `agents/agent.py::execute_action` | escalating per-turn and per-wait energy costs, a turn->move cost *discount* |
| `Agent.fitness` | `agents/agent.py:551-560` | `+0.1` for **every successful action**, `-0.05` for every failed one |

The project already has the right ablations on the shelf —
`brain.instincts.enabled: false` and `reward.preset: minimal` (W6c) — but the
headline configs use `legacy` shaping with instincts on. The `fitness` scalar
has no ablation at all.

### 1.5 How a run actually works

```
  per world tick, for every living agent:
     obs  = build_observation(agent, world)             utils/agents/perception.py
     mask = get_action_mask(agent, world)
     a, h', V, logp = brain.decide_with_logprob(obs, h, mask, instinct_strength)
     result = agent.execute_action(a, world)            + "behaviour economics"
     r      = reward_shaper.calculate_reward(...) + curiosity
     learner.store_step(obs, h, a, r, obs', done, logp, mask)
     every train_interval_ticks (staggered, adaptive budget):
         learner.learn(brain)     -> PPO update -> Lamarckian sync into genome
  world systems: plants, soil, weather/season, fire, pheromone decay, spawning
  reproduction: energy >= 45% of max AND age >= 50 AND cooldown -> split energy,
                child inherits mutated (already-trained) weights
```

Two things about this loop are load-bearing for §2:

- **Selection is energy-driven, in-world.** `agents/evolution.py::next_generation`
  is never called by `main.py`; who reproduces is decided by
  `Agent.can_reproduce` (energy threshold, min age, cooldown). `Agent.fitness`
  therefore does **not** drive reproduction — it drives `WeightManager`'s
  "best weights" ranking, the analyzer, and any future tournament selection.
  A scoring function that is wrong but unused is still a trap waiting to be
  stepped in.
- **`config/default.yaml` ships `learning.algorithm: a2c`**, which updates the
  policy and value **heads only** — the encoder, attention and GRU are shaped
  by evolution alone. Every baseline in §2 therefore uses `ppo` explicitly, so
  the diagnosis is made against the strongest learner the repo has, not the
  default one.

### 1.6 The three simulation types measured in §2

| Arm | Config | World | Brain | What it is for |
|---|---|---|---|---|
| **A** | `configs/A_default_v3.yaml` (from `config/default.yaml`) | 96x96 heightmap, 16 -> 30 agents, weather + fire on | v3, 8 actions, `algorithm: ppo` | the shipped showcase world, with the v2/v3 **cohort competition** on |
| **B** | `configs/B_social_v35.yaml` (from `config/worldmodel_v35.yaml`) | 64x64, 24 -> 30 agents | **v3.5**, 9 actions, `signal.enabled`, `agents_visible`, `agent_collision`, `social.transfer_enabled` all **on** | the most social configuration the repo can currently express |
| **C** | `configs/C_ecology_v35.yaml` = B + `--objects config/ecology.yaml` | as B, plus 3 food species and a toxic look-alike | v3.5 | the only real **discrimination** task: nothing labels `nightshade` as poison |

Each arm was run in both evolution modes — `rl` (PPO + Lamarckian) and
`neuroevolution` (no gradients at all) — for 5 generations x 1000 ticks, on
**4 seeds** each. 24 runs.

---

## 2. Baseline campaign — 3 simulation types x 2 evolution modes x 4 seeds

24 headless runs, 5,000 ticks each, all data in
`docs/sample_v35_emergence_baseline/`. Every number below is mean ± SD over
4 seeds. The instrument is `scripts/analyze_emergence.py` (new in this
branch), which pairs each behavioural claim with a null model.

### 2.1 Survival and population

| | A/neuro | A/rl | B/neuro | B/rl | C/neuro | C/rl |
|---|---|---|---|---|---|---|
| final population | 30 ± 0 | 30 ± 0 | 30 ± 0 | 30 ± 0 | 30 ± 0 | 30 ± 0 |
| mean final age | 514 ± 73 | 646 ± 25 | 609 ± 75 | 625 ± 22 | 608 ± 58 | 654 ± 23 |
| mean energy (end) | 113 ± 20 | 179 ± 10 | 114 ± 27 | 140 ± 18 | 122 ± 25 | 171 ± 7 |
| agents ever alive | 294 ± 38 | 231 ± 9 | 249 ± 30 | 239 ± 8 | 247 ± 22 | 228 ± 8 |

**Every arm pins at the population cap.** Survival is not a discriminating
measurement in this configuration — `reproduction.max_population` is binding
in all 24 runs, in both evolution modes, from early in the run. Any future
selection-pressure experiment has to raise the cap or lower carrying capacity
first, or it is measuring the cap.

### 2.2 What the agents actually do

Share of all logged actions (%):

| action | A/neuro | A/rl | B/neuro | B/rl | C/neuro | C/rl |
|---|---|---|---|---|---|---|
| MOVE_FORWARD | 20.6 ± 17 | 26.9 ± 1.1 | 18.1 ± 12 | 6.6 ± 3.0 | 7.5 ± 6.0 | 12.7 ± 4.1 |
| TURN_LEFT | 8.6 ± 2.3 | 18.0 ± 6.5 | 8.2 ± 1.9 | 18.9 ± 11 | 6.9 ± 4.3 | 10.5 ± 5.3 |
| TURN_RIGHT | 21.0 ± 17 | 13.2 ± 5.0 | 12.4 ± 5.4 | 15.4 ± 6.3 | 7.4 ± 4.6 | 21.7 ± 17 |
| WAIT | 41.9 ± 22 | 17.0 ± 5.8 | 27.9 ± 13 | 21.3 ± 5.1 | 19.8 ± 3.8 | 16.6 ± 6.4 |
| **SIGNAL** | — | — | **29.7 ± 18** | **31.0 ± 10** | **54.8 ± 9.5** | **26.0 ± 14** |
| EAT | 0.73 ± 0.71 | 4.31 ± 1.6 | 0.51 ± 0.72 | 0.99 ± 0.30 | 0.37 ± 0.47 | 1.99 ± 0.88 |
| USE (plant) | 0.87 ± 0.67 | 6.19 ± 1.7 | 0.61 ± 0.49 | 1.10 ± 1.1 | 0.39 ± 0.30 | 0.64 ± 0.61 |
| **always-valid share** | **92.1 ± 7.7** | **75.1 ± 2.6** | **96.3 ± 4.4** | **93.2 ± 5.1** | **96.3 ± 2.3** | **87.4 ± 8.6** |

Between **75% and 96% of everything an agent ever does is one of the five
actions that are always legal** (move / turn / turn / wait / signal). The four
actions that touch the world — pick up, drop, eat, plant — are the remaining
4-25%.

### 2.3 The finding: SIGNAL is the cheapest way to spend a tick, and the population found it

Energy cost of each action on success (`utils/agents/agent_utils.py`):

| action | cost | always valid? |
|---|---|---|
| **SIGNAL** | **0.12** | **yes** |
| WAIT | 0.18 | yes |
| MOVE_FORWARD | 0.20 (+0.6 x climb, +hazard) | yes |
| PICK_UP | 0.20 | no |
| TURN_LEFT / TURN_RIGHT | 0.24 (+ up to 0.20 escalating) | yes |
| EAT / DROP | 0.10 | no |
| USE (plant a seed) | 0.12 | no |

`SIGNAL` is **strictly the cheapest always-legal action** — 33% cheaper than
`WAIT`. Over a 600-tick life the gap against turning is
`(0.24 - 0.12) x 600 = 72` energy, i.e. 36% of `max_energy`. It always
succeeds, it never fails a mask, and the reward shaper has no term for it at
all.

So the population signals. Constantly. 26-55% of all behaviour, at **34-67
successful signal emissions per agent per 1000 ticks**, against **0.5-2.6
meals**.

The decisive detail is *where* it appears:

| arm | SIGNAL share by quarter of the run |
|---|---|
| B/neuroevolution | 30.5 -> 30.5 -> 28.1 -> 29.5 |
| B/rl | 34.8 -> 40.5 -> 28.0 -> 20.5 |
| **C/neuroevolution** | **41.3 -> 58.4 -> 60.7 -> 58.6** |
| C/rl | 26.5 -> 25.3 -> 28.1 -> 24.1 |

It is **strongest in the arm with no reward function at all**. Under pure
neuroevolution on the ecology pack, signal-spam *grows* from 41% to 59% of
all behaviour across the run. This is not a reward-shaping artefact and not a
PPO artefact: it is **selection finding that the cheapest way to stay alive is
to emit meaningless signals**, which is a fact about the world's action
costs, not about the brain.

It is also, precisely, emergent behaviour. Just not any we wanted. The v3.5
pheromone field — the repo's one communication channel — spends its entire
capacity carrying the population's idling.

Arms A/neuro and A/rl have no `SIGNAL` action (v3 has 8 outputs), and there
the idling lands on `WAIT` (41.9 ± 22 under neuroevolution) and on turning.
The attractor is structural, not specific to signalling.

### 2.4 Cultivation: planting happens, and it pays the planter nothing

E2 from §5, at the maturation latency (>= 160 ticks, the earliest a planted
seed can bear fruit under the default object pack):

| | A/neuro | A/rl | B/neuro | B/rl | C/neuro | C/rl |
|---|---|---|---|---|---|---|
| plantings logged | 1,290 ± 990 | 9,220 ± 2,500 | 646 ± 590 | 1,070 ± 1,100 | 442 ± 350 | 847 ± 850 |
| harvested by **anyone**, latency >= 160 | 0.49 ± 0.27 | 0.72 ± 0.05 | 0.39 ± 0.17 | 0.75 ± 0.05 | 0.45 ± 0.31 | 0.72 ± 0.08 |
| **matched spatial null** | 0.30 ± 0.27 | 0.57 ± 0.10 | 0.26 ± 0.23 | 0.59 ± 0.14 | 0.34 ± 0.25 | 0.63 ± 0.18 |
| harvested by **the planter** | **0.10 ± 0.09** | **0.14 ± 0.03** | **0.04 ± 0.02** | **0.21 ± 0.04** | **0.12 ± 0.13** | **0.13 ± 0.04** |
| median harvest latency (any) | 123 ± 67 | 59 ± 9 | 166 ± 74 | 66 ± 32 | 220 ± 180 | 73 ± 23 |

Read the third and fourth rows together. A tile the population merely *walked
on* is harvested near, at maturity latency, 26-63% of the time. A tile an
agent deliberately **planted** is harvested by **that agent** 4-21% of the
time. **Planting confers no measurable personal harvest advantage** — the
planter is less likely to benefit from its own seed than a random passer-by
is to eat near a random tile.

And the median harvest latency under the RL arms is 59-73 ticks — well short
of the ~160 ticks a seed needs to become fruit. The "harvests" that do happen
near planted tiles are mostly the *pre-existing* food that made the agent
stand there in the first place.

This is what L3 predicts arithmetically. It is the measurement that makes
the case for v4.1 and v4.3.

### 2.5 Individual differentiation is real; spatial structure is not

| | A/neuro | A/rl | B/neuro | B/rl | C/neuro | C/rl |
|---|---|---|---|---|---|---|
| `role_MI` (bits) | 0.41 ± 0.12 | 0.43 ± 0.03 | 0.45 ± 0.06 | 0.64 ± 0.23 | 0.45 ± 0.08 | 0.53 ± 0.09 |
| `role_MI` **null** (label-shuffled) | 0.08 ± 0.05 | 0.13 ± 0.03 | 0.06 ± 0.03 | 0.14 ± 0.02 | 0.07 ± 0.03 | 0.14 ± 0.04 |
| `H_pop` (bits) | 1.84 ± 0.15 | 2.64 ± 0.04 | 2.15 ± 0.35 | 2.37 ± 0.11 | 1.89 ± 0.30 | 2.47 ± 0.21 |
| trigram entropy | 4.42 ± 0.35 | 6.54 ± 0.10 | 5.35 ± 0.83 | 5.79 ± 0.60 | 4.56 ± 0.72 | 6.16 ± 0.40 |
| fraction of agents collapsed (modal action > 50%) | 0.77 ± 0.08 | 0.29 ± 0.01 | 0.55 ± 0.16 | 0.58 ± 0.15 | 0.73 ± 0.08 | 0.53 ± 0.12 |
| tiles visited per agent | 64 ± 46 | 102 ± 8 | 48 ± 30 | 22 ± 7 | 24 ± 21 | 45 ± 16 |
| territory overlap (Jaccard) | 0.019 | 0.017 | 0.022 | 0.010 | 0.009 | 0.012 |
| patch-return index (vs own-trajectory null) | 1.21 ± 0.13 | 1.26 ± 0.03 | 1.17 ± 0.05 | 1.30 ± 0.03 | 1.11 ± 0.18 | 1.39 ± 0.06 |

Three honest readings:

- **`role_MI` beats its null by 3-5x in every arm.** Agents really do differ
  from one another persistently — identity explains ~0.4-0.6 bits of what an
  agent does. That is a genuine positive result and it is the seed of
  division of labour. What we cannot yet say is whether the differences are
  *complementary* (roles) or merely *heterogeneous* (different degenerate
  attractors) — under neuroevolution, where 77% of agents are collapsed onto
  a single modal action, it is plainly the latter.
- **Neuroevolution collapses.** Without gradients, 55-77% of agents spend
  more than half their life on one action, trigram entropy drops ~2 bits, and
  meals fall 4-7x. PPO is doing real work; the "dual-mode" toggle is not a
  symmetric comparison, it is learner vs no-learner.
- **Territory overlap is ~1-2% everywhere and each agent sees 22-102 tiles of
  a 4,096-9,216 tile world.** Agents are not partitioning space; they are each
  stuck in a small blob. The patch-return index is 1.1-1.4 — barely above a
  null that is already saturated *because the blob is small*. There is no
  place memory to measure, because there is no place memory.

### 2.6 The reward landscape the learner actually sees

Measured by wrapping `Agent.compute_reward` over a 2,000-tick arm-B run
(PPO, seed 1, 59,228 scored actions):

| action | share of actions | mean shaped reward | median | mean energy cost |
|---|---|---|---|---|
| EAT | 0.22% | **+19.74** | +15.02 | 0.10 |
| DROP | 0.59% | +6.34 | +5.70 | 0.10 |
| USE (plant) | 2.40% | +2.99 | +2.69 | 0.21 |
| MOVE_FORWARD | 3.74% | +1.61 | +0.06 | 0.21 |
| PICK_UP | 1.47% | +0.50 | +0.80 | 0.20 |
| WAIT | 28.72% | +0.17 | −0.30 | 0.24 |
| **SIGNAL** | **40.69%** | **−0.07** | **−0.30** | **0.12** |
| TURN_LEFT | 9.98% | −0.29 | −0.37 | 0.28 |
| TURN_RIGHT | 12.19% | −0.43 | −0.62 | 0.28 |

The reward landscape is a **needle in a flat plain**: one action worth +19.7
that is legal on 0.2% of steps, and a plain of ±0.3 covering ~90% of steps.
Within the plain, `SIGNAL` is the cheapest tile to stand on — same median
reward as `WAIT`, half the energy of turning. The population is not being
irrational; it is optimising exactly what we wrote down.

Note also what the shaping is *for*: `MOVE_FORWARD` has mean +1.61 but median
+0.06 — the positive mean is carried by the explicit "moved toward food" and
"new tile" bonuses firing rarely. The dense shaping is doing the exploration
that the architecture cannot do for itself.

---

## 3. Diagnosis — seven structural limits on emergence

Each limit is stated as a claim, with the measurement or the arithmetic that
supports it, and the v4 component that attacks it.

### L1 — The objective is a hand-written behaviour policy (§2.6)

`RewardShaper.calculate_reward` is ~420 lines encoding movement bonuses,
new-tile bonuses, anti-backtrack and anti-revisit penalties, anti-spin
penalties, turn-toward-food bonuses and eat bonuses. `Agent.execute_action`
adds a second layer of escalating turn/wait costs and a turn->move discount.
`InstinctModule` adds `PICK_UP +1.5 / EAT +1.0 / USE +0.5 / hunger-EAT +3.0`
and a direction-aware turn-toward-food routine.

Anything that looks like foraging is downstream of a line of code that asked
for foraging. The `minimal` diet and `instincts.enabled: false` exist as
ablations and are not used by the headline configs.
**-> v4.0 (integrity) and v4.3 (evolved drive basis).**

### L2 — Action costs, not policy, decide most behaviour (§2.3)

`SIGNAL` at 0.12 is the cheapest always-legal action; the population spends
26-55% of its life on it, *including under pure neuroevolution where no
reward function exists at all*, and in the C/neuro arm the share grows
41% -> 59% across the run. Before any architecture work, the action-cost
table has to stop being a strategy.
**-> v4.0 (flat, published action costs; costly signalling in v4.4).**

### L3 — Delayed payoff has no gradient path (§2.4)

```
    BPTT horizon          seq_len = 8 ticks
    discount horizon      1/(1 - 0.95) = 20 ticks;  0.95^160 = 2.6e-4
    plant -> fruit         ~160 ticks
    day / season          200 / 2000 ticks
```

Measured consequence: a planted tile is harvested by its planter at
maturity-latency 4-21% of the time, **below** the 26-63% rate at which a
random walked-on tile is harvested by anyone. Foraging (10-15 ticks) sits
inside the discount horizon; cultivation, caching, and anything seasonal
does not.
**-> v4.1 (slow leaky state with evolved time constants, `seq_len` 32,
dual-discount critic).**

### L4 — There is no memory of place (§2.5)

Vision is a 5x5 egocentric window. Agents visit 22-102 tiles of a
4,096-9,216-tile world, with ~1-2% pairwise territory overlap. "Return to the
patch" and "return to where I planted" are not hard to learn — they are not
*representable*, because nothing in the state carries where anything was.
**-> v4.2 (episodic slots + exact path integration).**

### L5 — The communication channel cannot carry a protocol

One scalar, fixed strength, sensed as a 3x3 max, no addressing, no identity,
and (L2) cheaper than doing nothing. A protocol needs capacity, a cost, and
a receiver whose behaviour can depend on the symbol.
**-> v4.4 (C-channel vector field, cost proportional to amplitude,
mean+max sensing).**

### L6 — Every agent optimises the same objective, so there is nothing to specialise into

`role_MI` is 3-5x its shuffled null (§2.5) — differentiation is real — but
with one shared reward function and Lamarckian inheritance the population has
no mechanism that makes *being different* pay. Under neuroevolution the
"roles" are 55-77% collapsed single-action policies.
**-> v4.3 (per-genome drive weights `lambda`, so siblings can want different
things) and v4.4 (kin + identity, so others' wants are perceivable).**

### L7 — Kinship and identity are invisible

`BRAIN_V3_PROPOSAL.md` §9 designed the kin sense and it was never built. With
`social.transfer_enabled` on (arms B and C) an agent can give an item to a
neighbour, but it cannot tell a sibling from a stranger, so kin selection and
reciprocity have no perceptual basis.
**-> v4.4 (v3.6's fingerprint adopted verbatim + a 4-dim identity tag).**

### Not on the list: planning

The paper measured planning losing at every schedule with a weak model
(ecology `wm_err ~ 2.9`), and the robotics sibling measured it losing again
with a strong model (`wm_err ~ 0.0067`) because the dynamics head's reward
channel is hazard-blind. v4 deliberately does **not** add a planner. It gives
the dynamics head a different job (empowerment, §4.5b) that needs only the
one-step latent spread — the regime where the model is measurably good.
## 4. Brain v4 — the proposed architecture

> One sentence: **stop writing the objective and start evolving it, and give
> the network the two things it is missing to represent a delayed, spatial,
> social strategy — long memory and a place to put it.**

v4 is five changes. Four of them are genome changes (one batched, append-only
break, exactly the v3.5 discipline). One of them — the first — is not an
architecture change at all, and is the single highest-value item in the
document.

| Phase | Change | Designed in | Genome | Attacks |
|---|--------|---|--------|---------|
| **V4.0** | Scoring integrity: `fitness` := reproduction; `minimal` reward diet; instincts off; flat action costs | §4.5 (last part) | none | L1, L2 |
| **V4.1** | Multi-timescale memory core (slow leaky state, evolved time constants) + dual-discount critic + longer BPTT | §4.2, §4.3 | append-only | L3 |
| **V4.2** | Episodic place memory with exact path integration | §4.4 | append-only | L4 |
| **V4.3** | Evolved intrinsic motivation (drive basis in the genome) + empowerment | §4.5 | append-only | L1, L6 |
| **V4.4** | Vector communication channel with a cost, kin sense (v3.6 adopted), identity tag | §4.6, §4.7 | append-only | L5, L7 |

(The **phase** numbers V4.x and the **section** numbers §4.x are different
things — the table above maps between them. §4.1 and §4.8 are the
observation layout and the genome accounting that every phase shares.)

---

### 4.1 Observation v4 (91 dims)

Append-only over the v2 layout, so every v3.5 genome migrates.

```
  0..77   Observation v2  (unchanged — vision 50, agent_state 8,
                           stimulus 8, inventory 6, EXTRA 6)
  78      nearest_agent_kin           <- v3.6 §9.3, adopted verbatim
  79..82  comm_mean[0..3]             <- C=4 channel field, 3x3 mean
  83..86  comm_max[0..3]              <- C=4 channel field, 3x3 max
  87..90  neighbour_tag[0..3]         <- nearest agent's visible identity tag
```

`state_inputs` grows 28 -> 41. Everything else in the perception front end
(tile embedding, positional encoding, attention pool) is byte-for-byte v3.5.

---

### 4.2 The memory core: fast GRU + slow leaky state with **evolved** time constants

#### The problem, quantitatively

The current core is one GRU, `H = 48`, one tick per update:

```
    z_t = [ s_t || e_t ] in R^48 ,     h_t = GRU(z_t, h_{t-1}) in R^48
```

Two independent horizons throttle it:

1. **Truncated BPTT.** `learning.ppo.seq_len: 8`. Gradient does not exist
   beyond 8 ticks. Full stop.
2. **Discount.** `discount_factor: 0.95`. The effective horizon of a
   gamma-discounted return is `1/(1-gamma) = 20` ticks, and the weight on a
   payoff `k` ticks away is `0.95^k`.

Now the world's timescales, from `config/default.yaml`:

| Event | Delay from the decision that causes it |
|---|---|
| plant a seed -> germination (`plants.growth_time`) | 50 ticks |
| germination -> maturity (`plants.mature_age`) | +100 ticks |
| maturity -> first berry (`seed_spawn_rate 0.1`) | +~10 ticks |
| **USE(plant) -> the fruit exists** | **~160 ticks** |
| day/night cycle (`environment.day_length`) | 200 ticks |
| season (`environment.season_length`) | 2000 ticks |
| reproduction cooldown / min age | 30 / 50 ticks |

`0.95^160 = 2.6e-4`. With the 8-tick BPTT window, the gradient is *exactly*
zero. **Cultivation cannot be learned by this learner.** Whatever planting
behaviour the runs show is produced by `InstinctModule.USE_BIAS` and by the
reward shaper's explicit planting terms — not by credit assignment. This is
the single most important structural fact in this document, and it is
arithmetic, not opinion.

#### The fix: a second state with a learned leak

Keep the fast GRU. Add `H_s = 24` slow units that are a **leaky integrator**
whose per-unit rate is a genome parameter:

```
    alpha_i = sigmoid(rho_i)   in (0,1)      tau_i = 1 / alpha_i
    u_t     = tanh( W_f h^f_t + W_s h^s_{t-1} + b_s )
    h^s_t   = (1 - alpha) (.) h^s_{t-1}  +  alpha (.) u_t
```

`(.)` is elementwise. The Jacobian along the recurrence is

```
    d h^s_t / d h^s_{t-k}  =  diag( (1 - alpha)^k )  +  (terms through u)
                           ~  diag( exp(-k / tau) )
```

so a unit with `tau = 300` still carries `e^{-160/300} = 0.59` of the gradient
across a seed-to-fruit interval. The GRU at a typical update gate of 0.5
carries on the order of `2^{-160}` over the same span — and, at `seq_len: 8`,
exactly zero regardless.

**Initialisation matters and is part of the design.** Spread the ladder
geometrically over the world's timescales:

```
    tau_i = tau_min * (tau_max / tau_min)^( i / (H_s - 1) ) ,   i = 0..H_s-1
    tau_min = 2,  tau_max = 512
    rho_i   = logit(1 / tau_i) = -log(tau_i - 1)
```

With `H_s = 24` that places units at 2, 2.6, 3.4, ... , 392, 512 ticks —
straddling plant maturation (160), the day (200) and reaching toward the
season (2000).

#### Why `rho` in the genome is the interesting part

`rho` mutates and crosses over like every other weight, and it is selected on
real reproductive success. So the population **evolves its own memory
horizons**, and the evolved `tau` histogram is a measurement we can put
against the world's actual spectral peaks. That is a falsifiable emergence
claim of its own:

> **Prediction P1.** In a world where plant maturation is slow (`mature_age`
> 300, the `fruit_tree` of `config/ecology.yaml`), the evolved `tau`
> distribution shifts to longer time constants than in a fast-maturation
> world (`shrub`, 50) — measured as the median evolved `tau` after N
> generations, 4 seeds per arm.

If P1 fails, the slow core is decoration and we say so.

#### BPTT horizon

`seq_len: 8 -> 32`, `batch_size: 8 -> 4` (same number of transitions per
update, 4x the temporal reach, ~same cost per `learn()`). `h0` is already
stored per chunk, so the *value* of the carried state is already correct;
this only extends where gradient flows.

---

### 4.3 Dual-discount critic — the other half of the credit-assignment fix

Long memory is necessary but not sufficient: the *return* must also look far
enough ahead. Raising `gamma` to 0.999 alone makes PPO unusable, because the
variance of a gamma-return grows like `1/(1-gamma)^2` — 400x.

So: one critic, **two outputs**.

```
    V(z, h) in R^2 :   V^s  with gamma_s = 0.95      (local control)
                       V^l  with gamma_l = 0.999     (horizon 1000 ticks)
```

Each is trained on its own GAE target. The policy uses a **mixture
advantage**

```
    A_t  =  (1 - beta) * Ahat^{(gamma_s)}_t  +  beta * Ahat^{(gamma_l)}_t
    Ahat^{(gamma)}_t = sum_{l>=0} (gamma lambda)^l delta^{(gamma)}_{t+l}
    delta^{(gamma)}_t = r_t + gamma V^{gamma}(s_{t+1}) - V^{gamma}(s_t)
```

with **`beta` a genome scalar**: the population evolves its own planning
horizon. `beta = 0` reproduces v3.5 exactly.

Scale hygiene: `V^l` predicts returns ~20x larger than `V^s`. Train it on
returns normalised by a running percentile spread `S = clip(p95 - p5, 1, inf)`
(the Dreamer-V3 trick), and undo the scaling before forming `A`. Without this
the value loss term swamps the policy loss.

> **Prediction P2.** With V4.1 + V4.2 only (no reward change), the
> plant-to-self-harvest rate at latency >= 160 ticks rises above the
> spatially-matched null. If it does not, delayed credit assignment is not
> the binding constraint and we stop investing in it.

---

### 4.4 Episodic place memory with exact path integration

#### Why

Vision is a 5x5 egocentric window. A fruit tree 8 tiles away does not exist
for the agent. "Go back to the patch" and "go back to where I planted" are
not merely hard to learn — they are **not representable**. No amount of
training fixes an architecture that cannot express the target function.

#### The mechanism

`M = 8` slots. Slot `m` holds

```
    w_m in R^8      compressed latent at write time
    delta_m in R^2  displacement to that place, in the agent's CURRENT frame
    sigma_m in R    salience at write time
    a_m in R        age in ticks
```

**Write** (one parameter matrix, no gating network):

```
    sigma_t = |r_t| + eta * curiosity_t
    if sigma_t > min_m sigma_m :  overwrite argmin_m sigma_m with
        w   = tanh( W_c [ s_t || e_t ] + b_c )
        delta = (0, 0),   sigma = sigma_t,   a = 0
```

**Path integration** — every tick, for every slot, in the egocentric frame:

```
    move forward by one tile :   delta_m <- delta_m - (0, 1)
    turn left                :   delta_m <- R(+90) delta_m
    turn right               :   delta_m <- R(-90) delta_m
```

`R(+-90)` on a 4-heading grid is `(x, y) -> (-y, x)` / `(y, -x)`: **exact
integer arithmetic, zero parameters, zero drift.** This is literally what
hippocampal path integration does, in its simplest possible form. (Adding
proprioceptive noise later turns drift itself into an object of study.)

**Read** — the same single-head attention the vision path already uses, with a
second set of projections:

```
    x_m    = [ w_m || delta_m / (1 + ||delta_m||) || sigma_m || exp(-a_m / T) ]  in R^12
    t_m    = tanh( W_tok x_m + b_tok )                                  in R^E
    q^mem  = [ s_t || h^s_{t-1} ] W_q^mem                               in R^E
    r_t    = softmax_m( (t_m W_k^mem) . q^mem / sqrt(E) ) (t_m W_v^mem)  in R^E
```

and the latent grows

```
    z_t = [ s_t || e_t || r_t ]  in R^{40 + 8 + 8} = R^56
```

Note the bounded `delta / (1 + ||delta||)`: it keeps the token in the same
range as every other feature while preserving direction exactly. The network
gets "the thing I cared about is *that* way and *that* far", which is the
minimum content required for goal-directed return.

Cost: `M = 8` slots x a 12->8 embed + 3 tiny projections = **1,136 genome
parameters**, and 8 extra token rows per tick in the attention pool the brain
already runs.

> **Prediction P3.** The return-to-patch index (E1 in §5: revisit rate to a
> tile where this agent previously ate, over the rate for a tile drawn from
> that same agent's own trajectory) rises **materially above the v3.5
> baseline of 1.11-1.39** measured in §2.5, and `tiles_per_agent` rises with
> it — a memory of place is only useful to an agent that ranges further than
> 22-102 tiles. Ablation: zero the memory read and both must return to
> baseline. If they do not, the slots are not carrying place information and
> §4.4 is 1,136 wasted parameters.

---

### 4.5 Evolved intrinsic motivation — the reward function joins the genome

This is the part of the proposal that is actually about emergence.

Today, `RewardShaper.calculate_reward` is ~420 lines of hand-written
behavioural policy: exploration bonuses, new-tile bonuses, anti-backtrack
penalties, anti-spin penalties, turn-toward-food bonuses, eat bonuses.
`Agent.execute_action` adds a second, undocumented layer of "behaviour
economics" (escalating turn cost, escalating wait cost, a turn->move
discount). Whatever behaviour appears, a reviewer can point at the line that
asked for it.

The `reward.preset: minimal` diet (W6c) already exists and is the right
starting point. v4 goes one step further: **the agent's reward is a weighted
sum of a small basis of drives, and the weights are in the genome.**

```
    r_t  =  lambda_h r^homeo_t  +  lambda_e r^emp_t
          + lambda_c r^cur_t    +  lambda_s r^soc_t
    lambda in R^4  <- genome, mutated and crossed over like any weight
```

#### (a) Homeostasis — the only fixed term, and it is provably non-distorting

```
    d_t          = ( 1 - e_t / e_max )^2            energy deficit
    r^homeo_t    = d_{t-1} - d_t                    reward = reducing it
                 + (-1) on the tick the agent dies
```

`sum_t (d_{t-1} - d_t) = d_0 - d_T` telescopes: this is exactly a
**potential-based shaping function** `F(s,s') = Phi(s') - Phi(s)` with
`Phi = -d`, and by Ng, Harada & Russell (1999) potential-based shaping leaves
the optimal policy of the underlying MDP unchanged. It carries **no
information about how to get energy** — only that having it is good. That is
the difference between a drive and a strategy. Every current shaping term
fails this test.

#### (b) Empowerment — control over one's own future

The dynamics head `g_z(h, a) -> zhat'` already exists and, in the ecology,
currently does nothing behaviourally (measured: planning loses at every
setting; see the paper). Give it a job:

```
    zbar'      = (1/|A|) sum_a g_z(h_t, a)
    r^emp_t    = (1/|A|) sum_a || g_z(h_t, a) - zbar' ||^2  /  Z
```

the **variance across actions of the predicted next latent**: "how much does
my choice matter here". Justification as a bound: one-step empowerment is
`E = max_{p(a)} I(A ; Z')`. Under a fixed-noise Gaussian channel
`Z' = mu(a) + eps`, `eps ~ N(0, sigma^2 I)`,

```
    I(A; Z')  ~  (1/2) log( 1 + Var_a[ mu(a) ] / sigma^2 )
```

which is **monotone increasing in `Var_a[mu(a)]`**. So maximising the proxy
maximises a monotone transform of the true one-step empowerment, and the
proxy costs `|A| = 9` evaluations of a 2-layer MLP (~37k flops/tick —
negligible next to the 5x5 attention pass).

Empowerment is the best-documented intrinsic drive that produces *agent-like*
behaviour with no task: avoid death (dead = no control), avoid traps, stay
near resources, keep options open. It is a drive, not a strategy.

#### (c) Curiosity

`agents/curiosity.py`, unchanged: normalised, clipped dynamics prediction
error. Already implemented; it simply gets a genome weight instead of a YAML
constant.

#### (d) The social term — deliberately unsigned

```
    r^soc_t  =  sum over other agents within radius R=3 of ( e_{t} - e_{t-1} )
```

and `lambda_s` may evolve **positive (altruism), negative (spite), or zero
(indifference)**. We do not choose. Hamilton's rule says that under a kin
sense (4.7) the selected `lambda_s` should track relatedness `r` against the
cost/benefit ratio; that is a prediction, not a setting.

#### And the scoring-integrity half

`Agent.fitness` currently accrues `+0.1 for every successful action` and
`-0.05 for every failed one` (`agents/agent.py:551-560`). Selection in the
live simulation is energy-driven (reproduce above `energy_threshold`), so
this scalar does not feed reproduction — but it *is* what
`WeightManager._calculate_fitness` ranks agents by when saving "best"
weights, what the analyzer reports, and what tournament selection would use.
It rewards *acting*, not *living*. Replace it with:

```
    fitness  =  (number of offspring that themselves reached reproductive age)
             +  epsilon * lifespan
```

Nothing else. And make `reward.preset: minimal`, `brain.instincts.enabled:
false` the defaults for every v4 experiment config.

The third piece of V4.0 is the **action-cost table**, and §2.3 says why it
cannot wait for the reward change: the `SIGNAL` attractor is *strongest under
pure neuroevolution*, where there is no reward function at all. Removing the
shaping therefore cannot remove it — only the physics can. Two rules, applied
at once:

1. Delete the escalating turn/wait costs and the turn->move discount from
   `Agent.execute_action`. An action's cost becomes a constant of the action,
   declared in one table in one place.
2. **No always-legal action may be cheaper than `WAIT`.** `SIGNAL` moves from
   0.12 to `WAIT`'s cost plus the amplitude term of §4.6. "Do nothing" must be
   the cheapest way to do nothing.

> **Prediction P4.** V4.0 alone (no architecture change) removes the
> `SIGNAL`/`WAIT` cost inversion, so the always-valid-action share falls and
> the *idling* that remains lands on `WAIT` rather than on the communication
> channel. Survival gets **worse** once the dense shaping and the instincts
> come off — we expect a regression on every fitness-flavoured number. That is
> the point: V4.0's deliverable is an honest baseline, not a better one.
> Falsifier: if `pct_always_valid` does not move, the attractor is not about
> action costs and §4.6's cost term is unmotivated.
>
> **Prediction P5.** The evolved `lambda` distributions diverge between
> biomes / object packs (scarce vs abundant, `ecology.yaml` vs default) —
> 4 seeds per arm, measured as the between-arm effect size on each
> `lambda` component.

---

### 4.6 Communication with capacity, cost, and no semantics

Today: `SIGNAL` deposits **one scalar** with a fixed strength onto one tile;
receivers sense the **max over a 3x3 window**. A protocol cannot emerge
through that channel — it has ~1 symbol, no addressing, and (measured in
§2.3) it is the cheapest action in the game.

v4:

```
  emit:     u_t = tanh( W_u [ h^f_t || h^s_t ] + b_u )   in R^C,  C = 4
  cost:     energy  kappa_0 + kappa_1 * || u_t ||_1       (kappa_0 = 0.12,
                                                            kappa_1 = 0.05)
  field:    F_{t+1} = decay * (F_t + deposit)  per channel, optional diffusion
  sense:    mean and max of F over the 3x3 neighbourhood -> 2C = 8 inputs
```

The cost term is the point. A **costly signal** (Zahavi's handicap; Grafen
1990) is the standard precondition for honest signalling to be
evolutionarily stable. Today signalling is *cheaper than waiting*
(0.12 vs 0.18), which is the opposite of that condition, and §2.3 shows
exactly what the population does with it.

Emergence is then measured, not assumed:

```
    referential content :  I( U_t ; F_t )            F = discretised local
                                                         world feature
    uptake             :  I( U^recv_t ; A_{t+1} | own observation )
    ablation           :  fitness cost of shuffling the field
```

A protocol is claimed only if all three are non-zero.

---

### 4.7 Kin sense and identity tag

- **Kin.** Adopt Brain v3.6 §9.4 **verbatim** — birth-time fingerprint
  `f = normalize(P w)` with a fixed projection `P in R^{k x W}`, `k = 8`;
  `kin(a,b) = (f_a . f_b + 1) / 2`; one k-dim dot product per tick against the
  nearest neighbour already located for index 75. v3.6 is not superseded by
  v4; it is **absorbed into it**, which is also the cheapest way to finally
  ship the one designed-but-unbuilt item in the repo.
- **Identity tag.** A 4-dim vector `g` in the genome, mutated like weights,
  visible to other agents as observation 87..90 for the nearest neighbour.
  This is what makes reputation, reciprocity and green-beard dynamics
  *representable*. Whether any of them appear is the experiment.

---

### 4.8 Genome layout, parameter count, migration

All shape changes are **append-only**, so the shipped `migrate_genome`
top-left copy carries a v3.5 genome into v4 unchanged.

| Tensor | v3.5 (+wm) | v4 | Change |
|---|---|---|---|
| `state_enc.W` | (28, 40) | (41, 40) | +13 rows |
| `gru.W*_input` | (48, 48) | (56, 48) | +8 rows (memory read) |
| `policy.W` | (48, 9) | (72, 9) | +24 rows (slow state) |
| `value.W1` | (96, 16) | (128, 16) | +32 rows |
| `value.W2` | (16, 1) | (16, 2) | +1 column (long head) |
| `dyn.W1` | (57, 32) | (81, 32) | +24 rows |
| `dyn.Wz` | (32, 48) | (32, 56) | +8 columns |
| `mem.*`, `slow.*`, `comm.*`, `drive.*` | — | new | appended |

Totals (E=8, S=40, H_f=48, H_s=24, V=16, A=9, C=4, M=8, wm=32):

| Build | Params | vs v3.5+wm |
|---|---|---|
| v3.5 (no world model) | 17,626 | — |
| v3.5 + world model | 21,099 | baseline |
| **v4 full** | **27,756** | **+31.5%** |
| v4 lean (H_f=40, H_s=20) | 22,976 | +8.9% |

> These are the design-time estimates. The as-built numbers are in §9.7
> (27,689 with the world model); the small difference is §9.1's move of the
> memory compressor from the write side to the read side.

Group breakdown of v4 full: GRU 15,120 (54%), dynamics 4,505 (16%), value
2,098, state encoder 1,680, slow core 1,776, episodic memory 1,136, policy
657, comm 292, attention 448, tile embed 40, drives 4.

**Two tensors must not migrate to zero**, and this needs a small core change:

- `drive.lam` -> zero would make the reward identically zero. Default
  `(1, 0, 0, 0)` = homeostasis only.
- `slow.rho` -> zero gives `alpha = 0.5`, i.e. `tau = 2` for every slow unit,
  which is the one setting that makes the slow core pointless. Default: the
  geometric ladder of 4.2.

So `ParamSpec` entries gain an **optional `init` callable**
(`(name, shape) -> ndarray`), used by `Genome.random`, by `migrate_genome`
for tensors absent from the old spec, and by nothing else. ~20 lines, and it
is generally useful (every future architecture with structured priors needs
it).

---

## 5. The emergence instrument — what we will measure, decided in advance

The project's own rule ("single-seed effect sizes routinely evaporate") makes
this section the price of admission. Every metric below is computed by
`scripts/analyze_emergence.py` from the committed per-run CSVs, with >= 4
seeds per arm and a stated null model. **They are fixed before the runs.**

| Id | Emergent behaviour | Metric | Null model | Ablation that must kill it |
|----|---|---|---|---|
| E1 | Return to a known patch | `patch_return_index` — revisit rate to tiles where this agent previously ate, within its remaining life | the same agent's revisit rate to a tile drawn from its own trajectory | zero the episodic memory read |
| E2 | Cultivation with intent | `plant_harvest_mature_self_rate` — P(the planter harvests within 1 tile of its own `USE`, at latency >= 160 ticks) | `plant_harvest_mature_null` — the same question for a tile the population merely walked on | `beta = 0` (short discount only) |
| E3 | Division of labour | `role_MI = H(action) - E_agent[H(action \| agent)]` in bits | `role_MI_null` — agent labels shuffled within 100-tick blocks (finite-sample MI is positive under the null, so it must be reported) | — (report density dependence) |
| E4 | Referential signalling | `I(U_t ; local world feature)` and `I(U^recv ; A_{t+1} \| own obs)` | channel-shuffled field | shuffle the field, measure the fitness cost |
| E5 | Kin-biased behaviour | give-rate and signal-rate bucketed by `nearest_agent_kin`; fitted Hamilton slope | kin-shuffled control | `observation.version` rollback (no kin input) |
| E6 | Evolved motivation | distribution of `lambda_h, lambda_e, lambda_c, lambda_s, beta` and evolved `tau` over generations | initial (prior) distribution | — (between-biome divergence is the test) |
| E7 | Behavioural complexity | `trigram_H`, `H_cond` = `H(a_t \| a_{t-1})`, `frac_agents_collapsed`, `pct_always_valid` | uniform over the valid action set | — |

Rules of engagement, inherited from the paper:

- **>= 4 seeds or it does not exist.** Three reversals are already on the
  record (P2 +32%, P3 +25%, the 2/2-seed warmup win).
- **Every phase A/Bs against the previous phase, not against v3.5.** A
  five-change proposal evaluated only end-to-end teaches nothing about which
  change did the work.
- **Negative results ship** with the same care as wins, under
  `docs/sample_v4_*/` with all raw per-run data inline.
- An emergent behaviour is claimed only when the metric beats its null **and**
  the stated ablation removes it.

---

## 6. Implementation plan

Each phase is a PR, is independently revertible, and ends with a committed
4-seed study. Phases 1-4 are append-only genome bumps, so a v3.5 genome loads
into any of them.

### V4.0 — Scoring integrity (no genome change) — **do this first, alone**

1. `Agent.fitness` := offspring-that-reached-reproductive-age + `eps * lifespan`;
   delete the per-action `+0.1 / -0.05`.
2. Delete the "behaviour economics" block in `Agent.execute_action`
   (escalating turn/wait costs, turn->move discount). Action energy costs
   become flat constants of the action, published in one table, with **no
   always-legal action cheaper than `WAIT`** (this is what actually removes
   the §2.3 attractor — see §4.5).
3. New config `config/v4_baseline.yaml`: `reward.preset: minimal`,
   `brain.instincts.enabled: false`, `simulation.parallel: false`, and a
   `reproduction.max_population` high enough that the cap stops binding
   (§2.1).
4. E4-E6 added to `scripts/analyze_emergence.py` when their phases land
   (E1, E2, E3 and E7 already ship with this proposal) + tests.
5. **Study `docs/sample_v4_integrity/`**: 3 sim types x 2 modes x 4 seeds,
   v3.5 vs V4.0. Expect a broad regression. Publish it.

### V4.1 — Memory core

6. `ParamSpec` gains optional `init` callables (needed by `slow.rho`).
7. `slow.*` tensors + the leaky update in `BrainV3` (-> `BrainV4`), the torch
   mirror, and `forward_sequence`.
8. Policy and value read `[h^f || h^s]`; `value.W2` grows to 2 columns;
   `compute_gae` runs twice; `beta` mixing (config constant in V4.1, genome
   scalar in V4.3).
9. `seq_len: 32`, `batch_size: 4`; return-scale normaliser for `V^l`.
10. Tests: migration bit-identity v3.5 -> V4.1 with `beta = 0` and
    `slow.* = init`; `tau` ladder; gradient-reach test (a synthetic
    delayed-reward task where v3.5 provably cannot learn and V4.1 can).
11. **Study `docs/sample_v4_memory/`** — P2 is the headline.

### V4.2 — Episodic place memory

12. Slot state on the agent (not the genome), exact path integration, write
    rule, `mem.*` tensors, second attention pool, `z` grows to 56.
13. Tests: path-integration exactness over random walks (integer equality);
    memory-ablation equivalence to V4.1.
14. **Study `docs/sample_v4_place/`** — E1 is the headline.

### V4.3 — Evolved drives

15. `drive.lam` in the genome; `RewardShaper` gains a `drives` preset that is
    the four-term basis and nothing else; empowerment term on the dynamics
    head; `beta` becomes genomic.
16. Tests: potential-based-shaping invariance of `r^homeo`; empowerment
    proxy monotonicity on a synthetic channel; `lambda` inheritance.
17. **Study `docs/sample_v4_drives/`** — P5 and E6 are the headline.

### V4.4 — Society

18. Kin fingerprint (v3.6 §9.4, adopted), identity tag, C-channel field,
    costly `SIGNAL`, observation 91.
19. Tests: v3.6's own test list, plus channel-capacity and cost accounting.
20. **Study `docs/sample_v4_society/`** — E4 and E5 are the headline.

### Cost

Per-tick brain cost rises by the memory attention (8 extra tokens) and the
empowerment term (`|A|` dynamics evaluations): ~1.4x the current forward
pass, measured. Per-`learn()` cost is roughly flat (`seq_len` 4x,
`batch_size` 1/4x) plus the second GAE pass. The four arm-A runs in §2
cost ~4 min per 1000 ticks each at 4-way parallelism; budget ~1.5x that for
v4.

---

## 7. Risks, and what would falsify each claim

| Risk | Why it is real | Mitigation / falsifier |
|---|---|---|
| The whole thing is one big change and nothing is attributable | five components | each phase A/Bs against the previous phase; no phase lands without its own 4-seed study |
| Slow core is decoration | the GRU may already be enough within its window | **P1** (evolved `tau` tracks the world's timescale) and the synthetic delayed-reward test; if both fail, drop 4.2 |
| `gamma_l = 0.999` destabilises PPO | variance grows 400x | `beta` starts at 0; return-scale normaliser; the long head is an auxiliary before it is a policy input |
| Episodic memory is written with garbage | salience is a heuristic | ablation must remove E1; `M = 8` is tiny, so a useless memory costs 1.1k params and little else |
| Empowerment drives pathological "twitching" (maximise control by doing nothing irreversible) | documented failure mode of empowerment agents | `lambda_e` is evolved, not set — if twitching costs offspring, selection removes it. That is the design. |
| Evolved `lambda` collapses to a degenerate corner | selection is greedy | report the full distribution, not the mean; between-biome divergence (P5) is the test that it is doing work |
| Costly signalling just kills signalling | cost may exceed any benefit | `kappa_1` swept; the honest outcome "no protocol emerged at any cost" is a publishable result and would match the literature's base rate |
| Genome grows 31.5% | more params, fewer agents per second | the v4-lean sizing (+8.9%) is pre-computed; sizes are config, not code |
| We are measuring our own reward function again | the historical failure mode of this repo | V4.0 ships **first and alone**, and its expected outcome is a regression |

---

## 8. What this does not do

- It does not add a planner. The paper and the robotics sibling have now
  measured planning losing at **both** ends of the model-quality axis (weak
  model: compounding latent error; strong model: hazard-blind reward
  channel). v4 gives the dynamics head a behavioural job through
  *empowerment*, which needs only the one-step latent spread — the regime
  where the model is actually good — rather than an open-loop rollout.
- It does not change the world. Every claim above is measured in the existing
  ecology, with the existing object packs.
- It does not touch the DOI'd paper's numbers. v4 is a new architecture line;
  corrections are made by banner and new study, never by rewriting history.

---

## 9. As built — where the code differs from §4

§4 is the design as reviewed. Five things changed while building it; each is
a correction, not a compromise, and each is locked by a test.

### 9.1 A memory slot stores the raw latent, not a compressed one

§4.4 had a write-time compressor `w = tanh(W_c [s || e] + b_c)` into `R^8`.
The as-built slot stores the **uncompressed** `z_pre = [s || e]` (48 dims) and
the token embedding does the compression on the read side:
`mem.Wtok: (52, E)` over `[ z_pre | d_right | d_ahead | salience | age ]`.

Why: with a write-side compressor the only gradient path to `W_c` runs
through a slot written many ticks ago, which is either a very deep graph or
(if detached, as it must be for replay) no gradient at all. Moving the
projection to the read side means every parameter in the memory subsystem
gets gradient on every step from the slot contents it is reading *now*. Slot
*contents* remain a stop-gradient record — memory is what happened, not a
second path for the encoder's gradient.

Cost: the memory subsystem is 1,064 genome parameters, and a slot is 52
floats of runtime state.

### 9.2 Concatenation order is chosen for the migration, not for readability

The critic reads `[ z_pre | h_fast | memory_read | h_slow ]` and the dynamics
head reads `[ h_fast | onehot(a) | h_slow ]` — not the natural `[z | core]`
and `[core | onehot]`.

Why: `migrate_genome` is a **top-left copy**. If a block grows in the
*middle* of a concatenation, a v3.5 genome's rows land on the wrong inputs
and the migration is silently wrong rather than loudly broken. Both new
blocks (the memory read, the slow core) therefore sit at the end of every
concatenation they join. `test_v35_genome_migrates_to_v4_with_identical_behaviour`
is what caught this, and is what keeps it fixed.

### 9.3 A write needs a margin, or the memory is just a short GRU

§4.4's rule was "write when this step's salience beats the weakest slot".
Under the `drives` reward the homeostatic term is small but **never exactly
zero**, so that rule fires every single tick and the eight slots end up
holding the last eight ticks — which is what the GRU is already for. The
as-built rule requires `salience > 1.5 x weakest`, which (with the 0.999
per-tick salience decay) means the slots settle after the initial fill and
only a genuinely more salient event displaces one.

### 9.4 Path integration follows the outcome, so `moved` is plumbed through

A blocked `MOVE_FORWARD` must not shift every stored displacement by a tile.
That means the replay needs to know whether each logged move *succeeded*, so
`SequenceChunk` gains a `moved` array and `store_step` a `moved` argument.
`test_replay_of_a_chunk_without_memory_signals_would_diverge` is the control
that shows this matters: withhold `moved` and the replayed critic values
diverge from what the agent actually computed.

### 9.5 `beta` lives with the drive weights

§4.3 and §4.5 described `beta` and `lambda` as separate genome scalars. They
are one tensor, `drive.lam` of width 5: four drive weights plus `beta_raw`,
with `beta = sigmoid(beta_raw)`. The prior sets `beta_raw = -4`
(`beta ~ 0.018`), so a migrated genome behaves as v3.5 did until selection
finds a use for the long-horizon head. Drive weights are clipped to
`[-4, 4]` at read time: mutation is unbounded, and an agent whose curiosity
weight random-walked to 1e3 would not be exploring, it would be diverging.

### 9.6 Return scaling divides the loss, not the critic's output space

§4.3 said to "train it on returns normalised by a running percentile spread
and undo the scaling before forming `A`". The as-built code divides each
head's **squared error** by that head's spread instead, leaving the critic
predicting in raw units. Scaling the critic's output space would leave GAE
computing `delta = r + gamma V_scaled - V_scaled` — raw rewards against
scaled values — which is a bug that would have been invisible except as
slightly-wrong advantages. Same effect on the loss balance, no unit mismatch.

### 9.7 Final sizes

| Build | Params | vs v3.5 + world model |
|---|---|---|
| v3.5 | 17,626 | — |
| v3.5 + world model | 21,099 | baseline |
| v4 | 23,183 | +9.9% |
| **v4 + world model** | **27,689** | **+31.2%** |

Observation v4 is 91 dims (78 + kin + 2x4 comm + 4 tag); the state encoder
grows 28 → 41 inputs. The packed recurrent state is
`48 (fast) + 24 (slow) + 8 x 52 (slots) = 488` floats — runtime state, not
genome, so `brain.v4.memory_slots` can be changed or zeroed without touching
the genome length.

Measured cost: ~4 min per 1,000 ticks at 30 agents on one core, against
~50 s for v3.5 — roughly 5x, from `seq_len` 8 → 32, the second attention
pool, the empowerment term and the multi-step world-model loss.
