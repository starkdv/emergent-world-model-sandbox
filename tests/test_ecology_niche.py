"""
Ecological niche physics — the tradeoffs, and the free-energy fountain.

What these lock down:
  * with the ecology off, every function is an identity (existing runs are
    bit-unchanged, which is why the other 589 tests still pass);
  * Kleiber scaling really does point the two halves of the body-size
    tradeoff in opposite directions;
  * the diet kernel resolves into separable niches whatever object pack is
    loaded;
  * the reproduction subsidy dial reproduces the shipped free-tank behaviour
    at 1.0 and strict conservation at 0.0, and anneals monotonically between.
"""

import numpy as np
import pytest

from agents import ecology
from agents.ecology import EcologyConfig


@pytest.fixture(autouse=True)
def _restore():
    previous = ecology.get_active_ecology()
    previous_chem = dict(ecology._CHEMISTRY)
    yield
    ecology.set_active_ecology(previous)
    ecology._CHEMISTRY = previous_chem


def _on(**kw):
    ecology.set_active_ecology(EcologyConfig(enabled=True, **kw))


# --- off is off -----------------------------------------------------------


def test_disabled_ecology_is_an_identity():
    ecology.set_active_ecology(EcologyConfig(enabled=False))
    traits = {"body_size": 2.0, "visual_acuity": 2.0, "diet": 0.0}
    assert ecology.capacity_multiplier(traits) == 1.0
    assert ecology.metabolic_multiplier(traits) == 1.0
    assert ecology.movement_cost_multiplier(traits) == 1.0
    assert ecology.acuity_upkeep(traits) == 0.0
    assert ecology.assimilation_efficiency(traits, "nightshade") == 1.0
    assert ecology.offspring_starting_energy(10.0, 200.0, 200.0) == 200.0
    assert ecology.extend_trait_config({}) == {}


# --- allometry ------------------------------------------------------------


def test_kleiber_scaling_is_three_quarter_power():
    _on(allometry=True)
    for size in (0.5, 1.0, 1.7, 2.0):
        traits = {"body_size": size}
        assert ecology.metabolic_multiplier(traits) == pytest.approx(size**0.75)
        assert ecology.capacity_multiplier(traits) == pytest.approx(size)


def test_body_size_tradeoff_points_both_ways():
    """Large = longer famine endurance, but more energy needed to breed."""
    _on(allometry=True)
    small, large = {"body_size": 0.5}, {"body_size": 2.0}

    # Famine endurance E_max/B scales as S^0.25 — large survives longer
    assert ecology.famine_endurance(large) > ecology.famine_endurance(small)
    # ...but capacity (and so the absolute energy to reach the reproduction
    # threshold, which is a fraction of capacity) scales as S — large is slower
    assert ecology.capacity_multiplier(large) > ecology.capacity_multiplier(small)
    # ...and moving the mass costs more
    assert ecology.movement_cost_multiplier(large) > ecology.movement_cost_multiplier(
        small
    )
    # The tradeoff is only real if endurance grows SLOWER than cost, or large
    # would simply dominate.
    endurance_gain = ecology.famine_endurance(large) / ecology.famine_endurance(small)
    cost_gain = ecology.capacity_multiplier(large) / ecology.capacity_multiplier(small)
    assert endurance_gain < cost_gain


# --- acuity ---------------------------------------------------------------


def test_acuity_cost_is_quadratic_in_aperture():
    _on(acuity=True, acuity_cost=0.02)
    assert ecology.acuity_upkeep({"visual_acuity": 1.0}) == pytest.approx(0.02)
    assert ecology.acuity_upkeep({"visual_acuity": 2.0}) == pytest.approx(0.08)


def test_acuity_attenuation_is_smooth_and_monotone():
    """A hard cutoff would put a step into the world model's target."""
    weights = ecology.acuity_attenuation(2, 1.0).reshape(5, 5)
    centre_row = weights[2]
    assert centre_row[2] == pytest.approx(1.0)
    # Monotone decreasing away from the centre, strictly
    assert centre_row[2] > centre_row[1] > centre_row[0]
    assert np.all(weights > 0.0), "attenuation must never hard-zero a tile"
    # A sharper eye sees more of the window than a dim one, everywhere
    sharp = ecology.acuity_attenuation(2, 2.0)
    dim = ecology.acuity_attenuation(2, 0.6)
    assert np.all(sharp >= dim - 1e-9)
    assert sharp.sum() > dim.sum()


# --- diet -----------------------------------------------------------------


def test_chemistry_calibration_spreads_species_across_the_axis():
    """The raw hash can cluster; the calibrated axis must always resolve."""
    ids = ["shrub_berry", "tree_fruit", "nightshade", "berry"]
    chem = ecology.calibrate_species_chemistry(ids)
    values = sorted(chem.values())
    assert len(values) == 4
    assert values[0] == pytest.approx(0.125)
    assert values[-1] == pytest.approx(0.875)
    gaps = np.diff(values)
    assert np.allclose(gaps, gaps[0]), "species must be evenly spread"


def test_chemistry_is_deterministic_and_content_free():
    a = ecology.calibrate_species_chemistry(["a", "b", "c"])
    b = ecology.calibrate_species_chemistry(["c", "b", "a"])
    assert a == b, "calibration must not depend on input order"
    # Two different packs give different assignments; nothing is hardcoded
    assert ecology._stable_hash01("berry") != ecology._stable_hash01("tree_fruit")


def test_diet_kernel_separates_specialists_from_generalists():
    _on(diet=True, diet_width=0.20, diet_floor=0.10)
    ecology.calibrate_species_chemistry(
        ["shrub_berry", "tree_fruit", "nightshade", "berry"]
    )
    order = sorted(ecology._CHEMISTRY, key=lambda k: ecology._CHEMISTRY[k])
    low, high = order[0], order[-1]

    specialist = {"diet": ecology._CHEMISTRY[low]}
    assert ecology.assimilation_efficiency(specialist, low) == pytest.approx(
        1.0, abs=1e-6
    )
    assert ecology.assimilation_efficiency(specialist, high) < 0.2

    generalist = {"diet": 0.5}
    # A generalist is mediocre everywhere and best nowhere — which is what
    # makes the middle of the axis invadable once it is grazed out.
    effs = [ecology.assimilation_efficiency(generalist, s) for s in order]
    assert max(effs) < 0.95
    assert min(effs) > ecology.get_active_ecology().diet_floor


def test_assimilation_never_falls_below_the_floor():
    _on(diet=True, diet_floor=0.1)
    ecology.calibrate_species_chemistry(["a", "b"])
    for d in np.linspace(0, 1, 21):
        for s in ("a", "b"):
            assert 0.1 - 1e-9 <= ecology.assimilation_efficiency({"diet": d}, s) <= 1.0


# --- the free-energy fountain --------------------------------------------


def test_subsidy_one_reproduces_the_shipped_free_tank():
    _on(energy_conservation=True, birth_subsidy=1.0)
    assert ecology.offspring_starting_energy(36.0, 200.0, 200.0) == pytest.approx(200.0)


def test_subsidy_zero_is_strict_conservation():
    _on(energy_conservation=True, birth_subsidy=0.0)
    assert ecology.offspring_starting_energy(36.0, 200.0, 200.0) == pytest.approx(36.0)


def test_subsidy_is_monotone_between_the_extremes():
    previous = -1.0
    for sub in (0.0, 0.25, 0.5, 0.75, 1.0):
        _on(energy_conservation=True, birth_subsidy=sub)
        value = ecology.offspring_starting_energy(36.0, 200.0, 200.0)
        assert value > previous
        previous = value


def test_subsidy_anneals_linearly_and_clamps():
    _on(
        energy_conservation=True,
        birth_subsidy=1.0,
        birth_subsidy_final=0.0,
        birth_subsidy_anneal_ticks=1000,
    )
    assert ecology.birth_subsidy_at(0) == pytest.approx(1.0)
    assert ecology.birth_subsidy_at(500) == pytest.approx(0.5)
    assert ecology.birth_subsidy_at(1000) == pytest.approx(0.0)
    assert ecology.birth_subsidy_at(99999) == pytest.approx(0.0), "must clamp"


def test_offspring_never_exceeds_its_own_capacity():
    _on(energy_conservation=True, birth_subsidy=0.0)
    # A huge transfer into a small body is capped by the body
    assert ecology.offspring_starting_energy(1e6, 50.0, 50.0) == pytest.approx(50.0)
