# sample_ecology — closing the free-energy fountain

Raw data for **`docs/EMERGENCE_RESEARCH.md`**, which is the write-up. This
directory holds the configs and the per-generation metrics of every run.

All runs: `--mode neuroevolution` (no gradient learning anywhere, so behaviour
change is pure selection on the genome), `--objects config/ecology.yaml`,
64x64 world, generation-compressed (`max_age 250`, `min_age 20`), 4 seeds per
arm unless noted.

## Campaigns

| prefix | what it tests | ticks |
|---|---|---|
| `sub_*` | static reproduction-subsidy sweep — locates the extinction cliff | 6,000 |
| `c_*` | subsidy annealing at the shipped energy density — evolutionary rescue | 6,000 |
| `k_cal*` | world energy-density sweep at subsidy -> 0 — locates the viability frontier | 6,000 (1 seed) |
| `m_*` | first niche campaign — **void**, taken before trait mutation existed | 8,000 |
| `n_*` | niche campaign with `trait_mutation_std: 0.05` | 8,000 |
| `w_dw*` | diet kernel width sweep (sigma 0.08, 0.05) | 8,000 |
| `x_*` | compensated narrow kernel, and the best-combination candidate | 8,000 |

`m_*` is kept rather than deleted: it is the evidence for defect 4 (every arm
shows `body_size_sd = 0`, `diet_sd = 0`, `lineages = 1`).

## Headline numbers

**The energy budget of the shipped world.** Foraging income covers **3.5%** of
the population's burn (0.74 vs 21.0 energy/tick). Reproduction creates +164
energy per birth from nothing. The ecology runs on a ~28x subsidy.

**Static subsidy sweep** — sharp cliff between 0.75 and 0.50:

| subsidy | 1.00 | 0.90 | 0.75 | 0.50 | 0.25 | 0.00 |
|---|---|---|---|---|---|---|
| outcome | lives | lives | lives | extinct | extinct | extinct |

**Energy-density frontier** at subsidy -> 0: 3x extinct, 10x survives.

**The result.** At `calorie_scale 10` with the subsidy annealed to zero by
tick 3,000:

| tick | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---|---|---|---|---|---|
| birth subsidy | 0.67 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 |
| eats/agent/1k | 6.4 | 23.8 | 31.0 | 36.4 | 41.3 | **55.2** |
| income:burn | 1.8 | 6.7 | 8.6 | 10.1 | 11.0 | 15.0 |

Foraging rises **8.7x** and keeps rising after the subsidy reaches zero. At
100x energy density it rises only 1.4x — too much energy weakens the selection
that drives foraging.

**Branching: not demonstrated.** 0/4 seeds at every diet kernel width tested,
including the energy-compensated `sigma=0.05 + calorie_scale 30` arm, which is
the healthiest population in the campaign (eats 41.1 +- 6) and still does not
branch.

## Reproducing

```
python main.py --no-viz \
  --config docs/sample_ecology/configs/k_cal010.yaml \
  --objects config/ecology.yaml \
  --mode neuroevolution --seed 1 --generations 6 \
  --metrics-csv runs/k_cal010_s1/metrics.csv
```

The metrics CSV carries the evolution columns directly (`birth_subsidy`,
`eats_per_agent_per_1k`, `energy_income_per_tick`, `energy_burn_per_tick`,
`energy_ratio`, per-trait mean/SD/bimodality), so no per-tick agent logs are
needed — at 40 agents x 8,000 ticks those are millions of rows and make the run
I/O bound rather than compute bound.

## Caveats

Neuroevolution only; ~50 generations; 4 seeds with large SDs; `calorie_scale`
is swept, not derived. Two earlier results are withdrawn and recorded as such
in the write-up (§10).
