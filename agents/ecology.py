"""
Ecological niche physics — the tradeoffs that make being different pay.

Why this module exists
----------------------
The v3.5 baseline and the v4 ladder both found the population collapsing onto
one attractor with nothing interesting emerging. Two measurements explain it
without reference to the brain at all:

1. **Selection is alive.** Crow's opportunity for selection over completed
   lifetimes is ``I = Var(w)/mean(w)^2 = 0.52``, offspring count correlates
   with foraging (``r = 0.50``), and it is heritable (parent-offspring slope
   0.27, so ``h^2 ~ 0.55``). Evolution is working.
2. **There is nothing to select *into*.** The three "phenotypic traits" in
   ``create_default_trait_config`` are decorative: ``movement_speed`` is read
   only by the renderer's debug text, ``vision_radius`` is stored on the agent
   and never read by perception (the grid is hardcoded 5x5), and
   ``metabolism_rate`` is a **free lunch** — lower is strictly better, so it is
   under pure directional selection toward its floor with no cost anywhere.

A fitness landscape with one monotone axis and no tradeoffs has one peak.
Populations on single-peaked landscapes converge; they do not diversify. No
brain architecture fixes that.

This module adds three heritable traits that each carry a **real physical
cost**, turning the landscape multi-peaked and frequency-dependent:

``body_size`` (S) — allometric scaling
    Storage scales with mass and metabolism scales with mass to the 3/4 power
    (Kleiber's law, which holds empirically across ~27 orders of magnitude of
    body mass):

        E_max(S) = E0 * S
        B(S)     = b0 * S^0.75          basal burn per tick
        c_move(S)= c0 * S               cost of transporting mass

    The consequences point in opposite directions, which is the point:

        famine endurance  = E_max/B  = (E0/b0) * S^0.25     larger is better
        energy to breed   = thr * E_max ∝ S                 larger is slower

    So small bodies win where food is dense and steady, large bodies win where
    it is patchy or intermittent. This world already has seasons, drought,
    rain and wildfire, so both regimes occur — and the biome generator already
    makes rich river corridors and poor sand, so both occur *at once, in
    different places*.

``visual_acuity`` (R) — aperture cost
    The observation grid stays 5x5 so the genome layout is untouched, but a
    tile at Chebyshev distance r is attenuated by a Butterworth-style rolloff

        a(r) = 1 / (1 + (r/R)^4)

    and acuity is paid for per tick at

        c_vision(R) = k * R^2

    (optical light-gathering scales with aperture *area*). Sharp eyes see the
    whole window and pay for it; dim eyes are cheap and half-blind.

``energy_conservation`` — closing the free-energy fountain
    Not a trait, but the mechanism the other three need in order to matter.
    Reproduction currently *creates* energy: the parent gives up
    ``split * E_parent`` and the offspring is initialised to a **full tank**
    regardless. At the shipped numbers that is +164 energy per birth out of
    nothing, the equivalent of eight berries. Foraging skill is therefore
    almost irrelevant to fitness, which is why the population pins at the
    administrative cap and why ``EAT`` is ~1% of all actions.

    With conservation on, the offspring receives exactly what the parent gave
    up, less a gestation overhead:

        transfer = split * E_parent
        E_child  = min( E_max(child), (1 - overhead) * transfer )
        E_parent'= E_parent - transfer

    Energy in the population is then sourced only by eating. Carrying capacity
    becomes ecological rather than administrative, and offspring quality is
    tied to how well the parent actually foraged — which is what makes
    selection act on behaviour at all.

``diet`` (d) — assimilation kernel, and the route to evolutionary branching
    Every food species carries a chemistry trait ``s`` in [0,1] (a stable hash
    of its type id — nothing labels a food "good"), and an agent assimilates it
    with Gaussian efficiency

        eta(d, s) = floor + (1 - floor) * exp( -(d - s)^2 / (2 * sigma^2) )

    Energy gained is ``calories * freshness * eta``. Because eating depletes
    food *locally*, this is **negative frequency-dependent selection**: when
    everyone shares a diet the species they digest is grazed out, and mutants
    at the edges of the kernel do better. That is precisely the condition for
    **evolutionary branching** in adaptive dynamics (Dieckmann & Doebeli 1999)
    — a unimodal trait distribution splitting into two, which is emergent
    resource partitioning and the closest thing to speciation this sandbox can
    show.

Everything here is off by default (``ecology.enabled: false``), and with it off
every function is an identity, so existing runs are bit-unchanged.

Author: Karan Vasa
Date: September 2026
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Kleiber's exponent. Metabolic rate ~ mass^(3/4) across bacteria to whales.
KLEIBER_EXPONENT = 0.75

# Width of the diet assimilation kernel. Small = narrow specialists (strong
# frequency dependence, likely branching); large = generalists (no branching).
DEFAULT_DIET_WIDTH = 0.20

# Minimum assimilation efficiency, so a badly matched diet is expensive rather
# than instantly lethal — branching needs the population to survive its own
# transient.
DEFAULT_DIET_FLOOR = 0.10

# Per-tick cost coefficient of visual acuity (multiplies R^2).
DEFAULT_ACUITY_COST = 0.02

# Trait ranges. body_size is the interesting one; acuity is capped at the
# grid radius because seeing past the window is meaningless.
TRAIT_RANGES = {
    "body_size": (0.5, 2.0),
    "visual_acuity": (0.6, 2.0),
    "diet": (0.0, 1.0),
}


@dataclass(frozen=True)
class EcologyConfig:
    """
    Which niche axes are live, and how steep each tradeoff is.

    Attributes:
        enabled: Master switch. False makes every function below an identity.
        allometry: body_size drives storage, metabolism and movement cost
        acuity: visual_acuity attenuates distant tiles and costs energy
        diet: diet trait gates assimilation efficiency per food species
        kleiber_exponent: metabolic scaling exponent (0.75 = Kleiber)
        move_cost_exponent: movement cost scaling (1.0 = proportional to mass)
        acuity_cost: per-tick coefficient k in k*R^2
        calorie_scale: global multiplier on food energy — the world's energy
            density, which the shipped numbers never calibrated because the
            reproduction subsidy hid the deficit
        diet_width: sigma of the assimilation kernel
        diet_floor: minimum assimilation efficiency
        energy_conservation: offspring receive what the parent gives up,
            instead of a free full tank
        birth_overhead: fraction of the transfer lost to gestation
    """

    enabled: bool = False
    allometry: bool = True
    acuity: bool = True
    diet: bool = True
    calorie_scale: float = 1.0
    energy_conservation: bool = True
    birth_overhead: float = 0.0
    birth_subsidy: float = 1.0
    birth_subsidy_final: Optional[float] = None
    birth_subsidy_anneal_ticks: int = 20000
    kleiber_exponent: float = KLEIBER_EXPONENT
    move_cost_exponent: float = 1.0
    acuity_cost: float = DEFAULT_ACUITY_COST
    diet_width: float = DEFAULT_DIET_WIDTH
    diet_floor: float = DEFAULT_DIET_FLOOR

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "EcologyConfig":
        """
        Build from the ``ecology`` section of the YAML config.

        Args:
            config: Full config dict (or None for defaults)

        Returns:
            EcologyConfig
        """
        eco = (config or {}).get("ecology", {}) or {}
        return cls(
            enabled=bool(eco.get("enabled", False)),
            allometry=bool(eco.get("allometry", True)),
            acuity=bool(eco.get("acuity", True)),
            diet=bool(eco.get("diet", True)),
            kleiber_exponent=float(eco.get("kleiber_exponent", KLEIBER_EXPONENT)),
            move_cost_exponent=float(eco.get("move_cost_exponent", 1.0)),
            acuity_cost=float(eco.get("acuity_cost", DEFAULT_ACUITY_COST)),
            diet_width=float(eco.get("diet_width", DEFAULT_DIET_WIDTH)),
            diet_floor=float(eco.get("diet_floor", DEFAULT_DIET_FLOOR)),
            calorie_scale=float(eco.get("calorie_scale", 1.0)),
            energy_conservation=bool(eco.get("energy_conservation", True)),
            birth_overhead=float(eco.get("birth_overhead", 0.0)),
            birth_subsidy=float(eco.get("birth_subsidy", 1.0)),
            birth_subsidy_final=(
                None
                if eco.get("birth_subsidy_final", None) is None
                else float(eco["birth_subsidy_final"])
            ),
            birth_subsidy_anneal_ticks=int(
                eco.get("birth_subsidy_anneal_ticks", 20000)
            ),
        )


_ACTIVE_ECOLOGY = EcologyConfig()


def get_active_ecology() -> EcologyConfig:
    """Return the ecology rules the simulation is currently using."""
    return _ACTIVE_ECOLOGY


def set_active_ecology(config: EcologyConfig) -> None:
    """Set the ecology rules (call once at startup, before agents exist)."""
    global _ACTIVE_ECOLOGY
    _ACTIVE_ECOLOGY = config


# ---------------------------------------------------------------------------
# Allometry
# ---------------------------------------------------------------------------


def body_size(traits: dict) -> float:
    """Heritable body size S, clamped to its range (1.0 when absent)."""
    lo, hi = TRAIT_RANGES["body_size"]
    return float(np.clip(traits.get("body_size", 1.0), lo, hi))


def capacity_multiplier(traits: dict) -> float:
    """
    Energy storage multiplier ``E_max(S)/E0 = S``.

    Storage scales with mass. This is also what sets the energy an agent must
    accumulate to reproduce (the threshold is a fraction of max energy), so it
    is the "large bodies breed slowly" half of the tradeoff.
    """
    eco = get_active_ecology()
    if not (eco.enabled and eco.allometry):
        return 1.0
    return body_size(traits)


def metabolic_multiplier(traits: dict) -> float:
    """
    Basal burn multiplier ``B(S)/b0 = S^0.75`` — Kleiber's law.

    Paired with `capacity_multiplier`, famine endurance ``E_max/B`` scales as
    ``S^(1 - 0.75) = S^0.25``: larger bodies ride out a drought, which is the
    other half of the tradeoff.
    """
    eco = get_active_ecology()
    if not (eco.enabled and eco.allometry):
        return 1.0
    return float(body_size(traits) ** eco.kleiber_exponent)


def movement_cost_multiplier(traits: dict) -> float:
    """Cost of transporting mass one tile, ``c_move(S)/c0 = S``."""
    eco = get_active_ecology()
    if not (eco.enabled and eco.allometry):
        return 1.0
    return float(body_size(traits) ** eco.move_cost_exponent)


def famine_endurance(traits: dict) -> float:
    """
    Ticks of survival on a full tank with no intake, relative to S=1.

    ``E_max/B ∝ S^(1 - kleiber)``. Diagnostic only — nothing reads it.
    """
    eco = get_active_ecology()
    if not (eco.enabled and eco.allometry):
        return 1.0
    return float(body_size(traits) ** (1.0 - eco.kleiber_exponent))


# ---------------------------------------------------------------------------
# Visual acuity
# ---------------------------------------------------------------------------


def visual_acuity(traits: dict) -> float:
    """Heritable acuity R, clamped to its range (grid radius when absent)."""
    lo, hi = TRAIT_RANGES["visual_acuity"]
    return float(np.clip(traits.get("visual_acuity", hi), lo, hi))


def acuity_upkeep(traits: dict) -> float:
    """
    Per-tick energy cost of the eyes, ``k * R^2``.

    Light gathering scales with aperture area, so acuity is quadratic. The
    cost is charged every tick whether or not the agent looks at anything —
    eyes are expensive tissue, not a per-use fee.
    """
    eco = get_active_ecology()
    if not (eco.enabled and eco.acuity):
        return 0.0
    return float(eco.acuity_cost * visual_acuity(traits) ** 2)


def acuity_attenuation(radius: int, acuity: float) -> np.ndarray:
    """
    Per-tile visibility weights for a (2r+1)^2 egocentric grid.

    Butterworth-style rolloff in Chebyshev distance:

        a(r) = 1 / (1 + (r / R)^4)

    which is ~1 inside the acuity radius and falls off sharply beyond it,
    without the discontinuity a hard cutoff would put into the observation
    (and therefore into the world model's prediction target — the v4 ladder
    measured what discontinuities in that target cost).

    Args:
        radius: Grid radius (2 for the standard 5x5 window)
        acuity: The agent's R

    Returns:
        (side*side,) attenuation weights in row-major egocentric order
    """
    side = 2 * radius + 1
    rows = np.abs(np.arange(side) - radius)[:, None]
    cols = np.abs(np.arange(side) - radius)[None, :]
    dist = np.maximum(rows, cols).astype(np.float64)
    weights = 1.0 / (1.0 + (dist / max(acuity, 1e-6)) ** 4)
    return weights.ravel().astype(np.float32)


# ---------------------------------------------------------------------------
# Diet
# ---------------------------------------------------------------------------


def _stable_hash01(text: str) -> float:
    """FNV-1a hash of a string into [0, 1) — stable across processes/runs."""
    h = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return (h % 10_000) / 10_000.0


# Calibrated chemistry map, built once at startup from the registered edible
# species (see calibrate_species_chemistry). Empty = fall back to the raw hash.
_CHEMISTRY: dict[str, float] = {}


def calibrate_species_chemistry(type_ids) -> dict[str, float]:
    """
    Spread the registered food species evenly across the chemistry axis.

    The raw hash is arbitrary, which is what we want — but with only a handful
    of species it can leave them clustered (the shipped ecology pack hashes
    into 0.11..0.38), and a diet axis whose species all sit inside one kernel
    width cannot select for anything. So the hash is used only to *order* the
    species, and the order is then spread evenly:

        s_i = (rank_i + 0.5) / n

    Arbitrary (nothing about a food's calories, toxicity or appearance enters
    it) but guaranteed to resolve, whatever the object pack contains.

    Args:
        type_ids: Iterable of edible species type ids

    Returns:
        The calibrated ``type_id -> s`` map (also stored module-side)
    """
    global _CHEMISTRY
    ids = sorted(set(str(t) for t in type_ids if t))
    if not ids:
        _CHEMISTRY = {}
        return _CHEMISTRY
    ordered = sorted(ids, key=_stable_hash01)
    n = len(ordered)
    _CHEMISTRY = {t: (i + 0.5) / n for i, t in enumerate(ordered)}
    return _CHEMISTRY


def species_chemistry(type_id: str) -> float:
    """
    A food species' chemistry trait ``s`` in [0, 1].

    Deliberately uncorrelated with calories, toxicity or vision encoding, so
    nothing in the observation tells an agent whether a food matches its diet.
    It has to discover that from the energy it actually gets.

    Args:
        type_id: Object type id, e.g. "shrub_berry"

    Returns:
        Chemistry trait in [0, 1]
    """
    if not type_id:
        return 0.5
    if _CHEMISTRY:
        return _CHEMISTRY.get(type_id, _stable_hash01(type_id))
    return _stable_hash01(type_id)


def diet_trait(traits: dict) -> float:
    """Heritable diet trait d, clamped to [0, 1] (0.5 when absent)."""
    return float(np.clip(traits.get("diet", 0.5), 0.0, 1.0))


def assimilation_efficiency(traits: dict, type_id: str) -> float:
    """
    Fraction of a food item's calories this agent can actually extract.

        eta(d, s) = floor + (1 - floor) * exp( -(d - s)^2 / (2 sigma^2) )

    Because eating depletes food locally, a shared diet grazes out its own
    resource and rewards mutants at the kernel's edges — negative
    frequency-dependent selection, the precondition for evolutionary
    branching.

    Args:
        traits: The agent's trait dict
        type_id: The food species being eaten

    Returns:
        Efficiency in [floor, 1]
    """
    eco = get_active_ecology()
    if not (eco.enabled and eco.diet):
        return 1.0
    delta = diet_trait(traits) - species_chemistry(type_id)
    kernel = float(np.exp(-(delta**2) / (2.0 * max(eco.diet_width, 1e-6) ** 2)))
    return eco.diet_floor + (1.0 - eco.diet_floor) * kernel


# ---------------------------------------------------------------------------
# Trait plumbing
# ---------------------------------------------------------------------------


def extend_trait_config(trait_config: dict) -> dict:
    """
    Add the niche traits to a trait config, when the ecology is enabled.

    Args:
        trait_config: Existing name -> (min, max) mapping

    Returns:
        The same dict, extended in place
    """
    if not get_active_ecology().enabled:
        return trait_config
    for name, bounds in TRAIT_RANGES.items():
        trait_config.setdefault(name, bounds)
    return trait_config


def clamp_traits(traits: dict) -> dict:
    """Clamp the niche traits to their ranges after mutation/crossover."""
    for name, (lo, hi) in TRAIT_RANGES.items():
        if name in traits:
            traits[name] = float(np.clip(traits[name], lo, hi))
    return traits


# ---------------------------------------------------------------------------
# Reproduction energetics
# ---------------------------------------------------------------------------


def birth_subsidy_at(tick: int) -> float:
    """
    The reproduction subsidy in force at a given world tick.

    ``subsidy = 1`` is the shipped behaviour (the offspring is topped up to a
    full tank for free); ``subsidy = 0`` is strict conservation (it gets only
    what the parent gave up). Anything between is a partial top-up.

    Switching straight from 1 to 0 is an extinction event: measured on the
    committed baseline, foraging income covers **3.5%** of the population's
    energy burn, so the population has been running on a ~28x subsidy and
    cannot pay its own way the moment it is withdrawn. Annealing the subsidy
    down over ``birth_subsidy_anneal_ticks`` turns that cliff into a ramp and
    gives selection time to improve foraging as the free lunch is withdrawn —
    evolutionary rescue rather than mass extinction.

    Args:
        tick: Current world tick

    Returns:
        Subsidy fraction in [0, 1]
    """
    eco = get_active_ecology()
    start = eco.birth_subsidy
    end = eco.birth_subsidy_final
    if end is None or eco.birth_subsidy_anneal_ticks <= 0:
        return float(np.clip(start, 0.0, 1.0))
    frac = min(1.0, max(0.0, tick / float(eco.birth_subsidy_anneal_ticks)))
    return float(np.clip(start + (end - start) * frac, 0.0, 1.0))


def offspring_starting_energy(
    transfer: float,
    offspring_max_energy: float,
    default_full: float,
    tick: int = 0,
) -> float:
    """
    Energy an offspring starts life with.

        E_child = min( E_max, (1 - overhead) * transfer
                              + subsidy(t) * (E_max - (1 - overhead) * transfer) )

    At ``subsidy = 1`` this is exactly the shipped free full tank; at 0 it is
    strict conservation, and the population's energy is then sourced only by
    eating.

    Args:
        transfer: Energy the parent gave up
        offspring_max_energy: The offspring's own capacity
        default_full: What the legacy path would have used
        tick: Current world tick, for the subsidy anneal

    Returns:
        Starting energy for the offspring
    """
    eco = get_active_ecology()
    if not (eco.enabled and eco.energy_conservation):
        return default_full
    delivered = max(0.0, (1.0 - eco.birth_overhead) * transfer)
    subsidy = birth_subsidy_at(tick)
    topped = delivered + subsidy * max(0.0, offspring_max_energy - delivered)
    return float(min(offspring_max_energy, topped))


def calorie_multiplier() -> float:
    """
    Global multiplier on food energy — the world's energy density.

    The shipped numbers were never calibrated against what the agents can
    actually forage, because the reproduction subsidy covered the gap. With
    the subsidy withdrawn, a conserved newborn must eat ~11.6 berries in a
    250-tick life (46.5 eats per agent per 1000 ticks) while the measured
    baseline manages 1.24 — a 38x shortfall. This knob is the one-parameter
    rescaling that closes it, and sweeping it locates the frontier at which a
    closed-energy world is viable at all.

    Returns:
        Multiplier applied to a food item's calories (1.0 = shipped)
    """
    eco = get_active_ecology()
    return eco.calorie_scale if eco.enabled else 1.0
