# Why nothing emerges — four defects, and what actually moves the needle

**Status: measured.** Data in `docs/sample_ecology/`. Instruments:
`scripts/analyze_emergence.py` (what agents *do*),
`scripts/analyze_evolution.py` (what they *are*).

This document is the result of stopping work on the brain.

`BRAIN_V4_PROPOSAL.md` assumed the bottleneck was architecture and proposed
five changes to fix it. The v4 ladder measured all five losing
(`docs/sample_v4_ladder/`). Rather than iterate on the architecture, this round
tested the assumption behind it — and the assumption is wrong. The bottleneck
is not in the brain and never was.

---

## 0. The one-paragraph version

Selection in this sandbox is alive and well: Crow's opportunity for selection
over completed lifetimes is **I = 0.52**, offspring count correlates with
foraging at **r = 0.50**, and it is heritable (**h² ≈ 0.55**). What was missing
is anything to select *into* and any reason to forage. Four pre-existing
defects, none of them in the brain: the phenotypic traits are decorative, a
third of them literally unread; reproduction **creates 164 energy from nothing
per birth**, so foraging covers 3.5% of the population's energy budget; a
genotype/phenotype confusion drives metabolism to ~0 within twenty generations;
and phenotypic traits **never mutate at all** on the reproduction path the
simulation actually uses, so trait variance is pinned at exactly zero after the
first selective sweep. Fixing the energetics and calibrating the world's energy
density produces the first genuinely emergent behaviour this project has
measured: **foraging rises 8.7× under pure natural selection**, with no
gradient learning anywhere.

---

## 1. The diagnostic that redirected everything

Before adding mechanism, measure whether the machinery that would use it works.
Crow's index is the standard: the **opportunity for selection** is the variance
in relative fitness,

```
    I = Var(w) / w̄²
```

which is the upper bound on the rate of adaptation — if it is ~0, nothing
downstream of evolution can matter, and every hour spent on brains is wasted.

Measured over 147 completed lifetimes on the committed v3.5 baseline
(`B_social_v35`, seed 1, 3,000 ticks):

| quantity | value | reading |
|---|---|---|
| Crow's `I = Var(w)/w̄²` | **0.52** | real selective opportunity (human populations sit at 0.2–1.5) |
| mean offspring `w̄` | 0.97 | at replacement, as expected at carrying capacity |
| fraction leaving no offspring | 0.22 | a fifth of the population is selected out |
| `corr(eats, offspring)` | **0.50** | selection is coupled to *behaviour*, not noise |
| `corr(lifespan, offspring)` | 0.17 | and only weakly to merely surviving |
| parent–offspring slope `b` | 0.27 | → `h² = 2b ≈ 0.55`, strongly heritable |

**Selection works.** So the failure is upstream of it: the population has
nothing worth becoming, and no reason to eat.

---

## 2. Defect 1 — the phenotypic traits are decorative

`create_default_trait_config` advertises three heritable traits. In the shipped
code:

| trait | what reads it | selective character |
|---|---|---|
| `movement_speed` | **only** `pygame_renderer.py`, to print a debug string | none — it does nothing |
| `vision_radius` | stored on the agent; perception **never reads it** (`vision_radius = 2` is hardcoded in `_encode_vision`) | none — it does nothing |
| `metabolism_rate` | multiplies basal burn | **free lunch** — lower is strictly better, no cost anywhere |

So the phenotype space is one monotone axis with no tradeoff. A fitness
landscape like that has a single peak, and populations on single-peaked
landscapes converge — that is what a peak *is*. No brain architecture changes
this, which is why the v3.6 kin sense and the whole of v4 could not have
helped.

---

## 3. Defect 2 — reproduction creates energy

`Agent.reproduce`:

```python
energy_cost = self.energy * energy_split   # parent gives up 40% of current
self.energy -= energy_cost
offspring.energy = offspring.max_energy    # child gets a FULL TANK regardless
```

At the shipped numbers a parent breeds at 90 energy, gives up 36, and a child
appears holding 200:

```
    net energy created per birth = 200 − 36 = +164     (82% of a full tank)
                                 = 8 berries, from nothing
```

Against the committed baseline's measured foraging rate:

```
    burn   = 30 agents × (metabolism 0.5 + action ~0.2) = 21.0 energy/tick
    income = 30 × 1.24 eats/1k × 20 cal                 =  0.74 energy/tick
    → foraging covers 3.5% of the population's energy budget
    → the ecology runs on a ~28× reproduction subsidy
```

This is the reason `EAT` is ~1% of all actions in every arm ever measured, and
the reason the population pins at `reproduction.max_population` rather than at
any ecological carrying capacity. **Foraging skill barely affects fitness,
because food is barely the source of energy.**

---

## 4. Defect 3 — genotype/phenotype confusion, compounding

`clone_agent` handed the child the parent's **phenotype** as its base:

```python
max_energy=parent.max_energy,              # already trait-multiplied
metabolism_rate=parent.metabolism_rate,    # already trait-multiplied
```

and `Agent.__init__` multiplied by the trait again. Traits are multipliers, so
re-applying one every generation compounds it geometrically. With the shipped
trait mean of ~0.64:

| generation | 0 | 5 | 10 | 20 | 48 (measured live) |
|---|---|---|---|---|---|
| metabolism | 0.500 | 0.053 | 0.0057 | 0.000064 | **0.0003** |

Measured in a live run at generation ~48: total population burn **0.02
energy/tick for 80 agents**. After roughly twenty generations an agent burns
about 1/1600th of the intended metabolism and can live essentially forever
without eating. That is a **second** path by which foraging stops mattering,
independent of the birth subsidy.

Fixed: `Agent` now keeps `base_metabolism_rate` / `base_max_energy` and
`clone_agent` passes the base. The same probe post-fix reads metabolism 0.43
and population burn 34.6 energy/tick — a **1700× correction**.

---

## 5. Defect 4 — phenotypic traits never mutate

`clone_agent` calls `mutate_genome_weights`, which touches `genome.weights`.
Traits ride along in the deep-copied genome **unchanged**. On the in-world
reproduction path — the only one `main.py` uses — phenotypic traits are drawn
once at founding and then frozen per lineage. The
`evolution.trait_mutation_std: 0.05` knob every shipped config carries feeds
only `Genome.mate`, which the live simulation never calls.

Measured consequence, from the first ecology campaign (20 runs):

```
    lineages  = 1        in every arm — a complete selective sweep
    body_size_sd = 0     exactly
    diet_sd      = 0     exactly
```

Once the sweep completes, trait variance is **exactly zero, forever**. Trait
evolution is structurally impossible. Every niche axis in §6 is an axis the
population cannot travel along, which is why the first campaign's answer to
"which mechanism wins" was an artifact and was withdrawn.

Fixed: `mutate_genome_traits` applies Gaussian mutation and re-clamps, driven
by `reproduction.trait_mutation_std` (default 0.0, so existing configs stay
bit-identical).

---

## 6. The mechanisms built, and their maths

All live in `agents/ecology.py`, all off by default; with the ecology disabled
every function is an identity.

### 6.1 Allometry — body size as a real niche axis

Heritable body size `S`, with the scaling laws real organisms obey:

```
    E_max(S) = E₀ · S           storage scales with mass
    B(S)     = b₀ · S^0.75      Kleiber's law (holds over ~27 orders of magnitude)
    c_move(S)= c₀ · S           transporting mass costs proportionally
```

The two consequences point in **opposite** directions, which is what makes it
an axis rather than a gradient:

```
    famine endurance = E_max/B = (E₀/b₀) · S^0.25      larger survives longer
    energy to breed  = thr · E_max ∝ S                 larger breeds slower
```

Small bodies win where food is dense and steady; large bodies win where it is
patchy or intermittent. The world already supplies that variability — seasons,
drought, rain, wildfire — and the biome generator already supplies rich river
corridors next to poor sand, so both regimes exist *at once, in different
places*.

### 6.2 Visual acuity — aperture cost

The observation grid stays 5×5 so the genome layout is untouched. A tile at
Chebyshev distance `r` is attenuated by a Butterworth-style rolloff, and acuity
is charged per tick:

```
    a(r) = 1 / (1 + (r/R)⁴)          visibility
    c(R) = k · R²                     upkeep — light gathering scales with area
```

The rolloff is smooth on purpose: the v4 ladder measured what a discontinuity
in the world model's prediction target costs (`wm_rollout_error` 4.02 vs 1.18).

### 6.3 Diet — assimilation kernel and the route to branching

Each food species carries a chemistry trait `s ∈ [0,1]`; an agent with diet `d`
assimilates it with Gaussian efficiency

```
    η(d,s) = floor + (1 − floor) · exp( −(d − s)² / 2σ² )
```

Species are ordered by a stable FNV-1a hash and then **spread evenly** across
[0,1], so the axis resolves whatever object pack is loaded — the shipped pack
hashes into 0.11–0.38, all inside one kernel width, which would select for
nothing. Chemistry is uncorrelated with calories, toxicity and vision encoding:
nothing labels a food.

Because eating depletes food *locally*, a shared diet grazes out its own
resource and rewards mutants at the kernel's edges. That is negative
frequency-dependent selection, the precondition for **evolutionary branching**
(Dieckmann & Doebeli 1999) — a unimodal trait distribution splitting in two,
which is emergent resource partitioning.

### 6.4 The subsidy dial — closing the fountain

```
    E_child = min( E_max, delivered + subsidy(t) · (E_max − delivered) )
    delivered = (1 − overhead) · transfer
```

`subsidy = 1` is exactly the shipped free tank; `subsidy = 0` is strict
conservation, and the population's energy is then sourced only by eating.
`subsidy(t)` anneals linearly so the cliff becomes a ramp.

### 6.5 World growth — frontier expansion

`World.grow(step)` extends the map on the +x/+y edges so every existing `(x,y)`
keeps its meaning: agents, objects and the pheromone/communication fields stay
valid. The frontier is seeded at the founding resource density, so growth
expands carrying capacity instead of diluting it. Triggered on agents-per-tile,
which is scale-free.

### 6.6 World energy density

`calorie_scale` multiplies food energy. The shipped numbers were never
calibrated against what the agents can actually forage, because the subsidy
covered the gap.

---

## 7. Results

### 7.1 Closing the fountain outright is an extinction event

Every arm with `energy_conservation` on and no compensating change died inside
1,000 ticks; the identical config without it is stable. A static subsidy sweep
locates a sharp cliff:

| subsidy | 1.00 | 0.90 | 0.75 | 0.50 | 0.25 | 0.00 |
|---|---|---|---|---|---|---|
| outcome | survives | survives | survives | **extinct** | **extinct** | **extinct** |

The arithmetic says why. A conserved newborn starts at 75 energy and must reach
150 to breed:

```
    lifetime burn      (0.43 + 0.20) × 250 = 157.5
    net gain to breed                      =  75.0
    TOTAL to eat                           = 232.5  = 11.6 berries per life
                                           = 46.5 eats per agent per 1,000 ticks
    measured baseline                      =  1.24
    GAP                                    =  38×
```

Annealing the subsidy to zero over 3,000 ticks (~50 generations) did **not**
rescue the population: 4/4 seeds extinct on every arm. Selection cannot close a
38× gap that fast.

### 7.2 The energy-density frontier, and the headline result

Sweeping `calorie_scale` with the subsidy annealed to zero:

| calorie_scale | 3× | 10× | 30× | 100× |
|---|---|---|---|---|
| outcome | extinct | **survives** | survives | survives |
| foraging rise over the run | — | **8.7×** | 5.0× | 1.4× |

At **10×**, with the subsidy annealed to zero by tick 3,000:

| tick | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---|---|---|---|---|---|
| birth subsidy | 0.67 | 0.33 | **0.00** | 0.00 | 0.00 | 0.00 |
| eats/agent/1k | 6.4 | 23.8 | 31.0 | 36.4 | 41.3 | **55.2** |
| income : burn | 1.8 | 6.7 | 8.6 | 10.1 | 11.0 | 15.0 |
| generations | 13 | 18 | 23 | 25 | 30 | 34 |

**Foraging rises 8.7× and keeps rising after the subsidy reaches zero.** This
run is in `neuroevolution` mode: there is no gradient learning anywhere, no
reward function in play, and no shaping. It is pure natural selection on the
genome, and it is the first genuinely emergent behaviour this project has
measured.

At 100× the rise is only 1.4×. **Selection strength on foraging is non-monotone
in world richness**: too little energy and the population dies before it can
adapt, too much and foraging stops mattering again. The subsidy was simply an
extreme case of "too much".

### 7.3 Niche axes: variation appears, branching does not

With `trait_mutation_std: 0.05` (defect 4 fixed) trait variance finally exists.
All arms: closed energy (subsidy annealed to 0 by tick 3,000), `calorie_scale
10`, 8,000 ticks (~50 generations), neuroevolution, 4 seeds.

| arm | eats/agent/1k | income:burn | body_size SD | diet SD | diet BC |
|---|---|---|---|---|---|
| rescue (no niche axes) | 34.4 ± 8.8 | 13.6 ± 8.1 | 0.196 | 0.107 | 0.453 |
| + allometry | 33.7 ± 13 | 12.2 ± 7.1 | 0.149 | 0.149 | 0.507 |
| + diet (σ=0.20) | 17.7 ± 7 | 5.5 ± 2.3 | 0.122 | 0.110 | 0.404 |
| + all three | 27.3 ± 16 | 9.2 ± 6.1 | 0.159 | 0.133 | 0.435 |
| + all three + world growth | **43.4 ± 21** | 11.3 ± 6.9 | 0.137 | 0.121 | 0.376 |

**Trait variance is now maintained** (SD ≈ 0.11–0.20 where it was exactly 0),
which is the fix working. But no bimodality coefficient reaches the 5/9 =
0.5556 branching threshold in any arm.

### 7.4 The diet kernel does not branch, at any width

Branching needs specialising to pay more than staying generalist. The kernel
width σ controls exactly that:

| σ | generalist midway between two species | specialist on its own |
|---|---|---|
| 0.20 | 0.840 on **each** neighbour | 1.000 |
| 0.08 | 0.366 on each | 1.000 |
| 0.05 | 0.140 on each | 1.000 |

At σ=0.20 being in the middle costs almost nothing, so there is no disruptive
pressure at all. Narrowing it:

| σ | eats/agent/1k | income:burn | seeds branched |
|---|---|---|---|
| 0.20 | 17.7 ± 7 | 5.5 ± 2.3 | **0 / 4** |
| 0.08 | 20.9 ± 9.2 | 5.7 ± 2.4 | **0 / 4** |
| 0.05 | 7.3 ± 8.6 | 1.0 ± 1.4 | **1 / 4** |
| 0.05, `calorie_scale` 30 | 41.1 ± 6 | 13.8 ± 3.4 | **0 / 4** |

The single branched seed at σ=0.05 is 1/4 and does not survive the project's
own ≥4-seed rule. It is also confounded: at σ=0.05 uncompensated the population
is at the edge of survival (income:burn 1.0), so that seed's bimodality is as
likely a bottleneck artifact as a branch.

The structural tension: **narrowing σ to make specialising pay simultaneously
makes the world poorer**, because mean assimilation efficiency falls with σ.
Compensating it (`calorie_scale` 30) produces the healthiest, most consistent
population in the whole campaign — eats 41.1 ± 6, the tightest variance of any
arm — and **still 0/4 branching**.

**Conclusion: no evolutionary branching was demonstrated.** 50 generations is
plausibly just too few — adaptive-dynamics branching is usually reported over
hundreds — but that is a hypothesis this campaign did not test, not a result.

---

## 8. What actually works, and the best combination

Ranked on the one emergent behaviour that was demonstrated (foraging rate under
pure selection):

| rank | arm | eats/agent/1k |
|---|---|---|
| 1 | all niche axes + world growth | 43.4 ± 21 |
| 2 | diet σ=0.05 + `calorie_scale` 30 | **41.1 ± 6** |
| 3 | rescue only (no niche axes) | 34.4 ± 8.8 |
| 4 | + allometry | 33.7 ± 13 |
| 5 | + all three | 27.3 ± 16 |
| 8 | + diet σ=0.20 | 17.7 ± 7 |
| 9 | + diet σ=0.05 uncompensated | 7.3 ± 8.6 |

Read with the standard deviations, **none of the niche axes reliably beats the
plain rescue arm at n=4.** The one clearly separated effect is negative: the
diet kernel is an energy tax (a generalist assimilates ~0.84, not 1.0) and
costs half the foraging rate unless the world's energy density is raised to
compensate.

### The recommended combination

**Load-bearing — these are what produced the result:**

1. Fix all four defects (§2–§5). Non-optional; three of them are bugs.
2. `ecology.energy_conservation: true` with `birth_subsidy` annealed 1 → 0.
   The anneal is required: a step change is an extinction event.
3. `ecology.calorie_scale: 10`. Below ~10 the closed-energy world is not
   viable at current foraging competence; far above it, selection on foraging
   weakens again (1.4× rise at 100× vs 8.7× at 10×).
4. `reproduction.trait_mutation_std: 0.05`, or traits cannot evolve at all.

**Safe to include, unproven at n=4:** allometry, acuity, world growth. No
measurable harm; world growth is in the top arm but well inside the noise.

**Avoid unless compensated:** the diet kernel. If used for branching work,
pair σ ≤ 0.05 with `calorie_scale` ≈ 30 — that combination is the healthiest
population measured, it simply does not branch.

---

## 9. What this says about the brain work

The v4 proposal's §3 listed seven structural limits on emergence and put six of
them in the brain. On this evidence that was the wrong place to look:
foraging — the behaviour whose absence motivated everything — moved **8.7×**
without touching the brain at all, in a mode with no gradient learning
whatsoever.

That does not make the v4 architecture wrong; it makes it **untested**. Every
brain measurement in `docs/sample_v35_emergence_baseline/` and
`docs/sample_v4_ladder/` was taken in a world where reproduction created 82% of
a tank of energy per birth, metabolism decayed to 3e-4 within twenty
generations, and phenotypic traits could not mutate. Those numbers describe a
broken ecology, not an architecture.

**The brain comparisons are worth re-running on the fixed world.** That is the
obvious next campaign, and it is the one that would finally say whether the
slow core, the dual-discount critic or the episodic memory are worth anything —
this time in a world where foraging pays.

---

## 10. Honest limits

- **Neuroevolution only.** Every result here is from `--mode neuroevolution`,
  chosen because it isolates selection and is ~5× cheaper. Nothing here says
  what PPO does in the fixed world.
- **~50 generations.** Enough to see foraging adapt, plausibly far too few for
  branching.
- **4 seeds, large SDs.** The niche-axis ranking in §8 is mostly within noise
  and is reported as such.
- **`calorie_scale` is a tuned parameter.** It was swept, not derived, and 10×
  is the frontier for *these* agents' competence — a better forager would need
  less.
- **The world-growth seam.** `World.grow` splices a new heightmap onto the old
  one; biomes are discontinuous at the join.
- **Two withdrawn results.** An early pilot appeared to show foraging rising
  10× as the subsidy fell; it was taken under the metabolism-decay bug and is
  void. The first niche campaign's arm ranking was taken before trait mutation
  existed and is void.
