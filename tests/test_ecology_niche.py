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


# --- genotype / phenotype -------------------------------------------------


def test_trait_multipliers_do_not_compound_across_generations():
    """
    clone_agent must hand the child the BASE values, not the parent's
    phenotype. Re-applying a multiplier every generation compounds it: with
    the shipped trait mean of ~0.64 that drove metabolism from 0.5 to 3e-4
    within ~20 generations, i.e. agents that never need to eat.
    """
    from agents.agent import Agent
    from agents.brain import Brain
    from agents.evolution import clone_agent
    from agents.genome import Genome, create_default_trait_config

    ecology.set_active_ecology(EcologyConfig(enabled=False))
    np.random.seed(0)
    genome = Genome.random(
        weight_count=Brain.calculate_weight_count(),
        trait_config=create_default_trait_config(),
    )
    genome.traits["metabolism_rate"] = 0.6  # a below-1 multiplier
    agent = Agent(x=0, y=0, genome=genome, metabolism_rate=0.5, max_energy=200.0)
    assert agent.metabolism_rate == pytest.approx(0.30)
    assert agent.base_metabolism_rate == pytest.approx(0.50)

    # Twenty generations of cloning with the multiplier held fixed
    current = agent
    for _ in range(20):
        current = clone_agent(parent=current, mutate=False)
        current.genome.traits["metabolism_rate"] = 0.6
        current.traits["metabolism_rate"] = 0.6
        current.metabolism_rate = current.base_metabolism_rate * 0.6

    assert current.base_metabolism_rate == pytest.approx(0.50)
    assert current.metabolism_rate == pytest.approx(
        0.30
    ), "metabolism must not decay across generations"
    assert current.base_max_energy == pytest.approx(200.0)


def test_clone_passes_base_not_phenotype():
    """The direct regression: one clone must not shrink the base."""
    from agents.agent import Agent
    from agents.brain import Brain
    from agents.evolution import clone_agent
    from agents.genome import Genome, create_default_trait_config

    ecology.set_active_ecology(EcologyConfig(enabled=False))
    np.random.seed(1)
    genome = Genome.random(
        weight_count=Brain.calculate_weight_count(),
        trait_config=create_default_trait_config(),
    )
    genome.traits["metabolism_rate"] = 0.5
    parent = Agent(x=0, y=0, genome=genome, metabolism_rate=0.5, max_energy=200.0)
    child = clone_agent(parent=parent, mutate=False)
    assert child.base_metabolism_rate == pytest.approx(parent.base_metabolism_rate)
    assert child.base_max_energy == pytest.approx(parent.base_max_energy)


# --- world growth ---------------------------------------------------------


class TestWorldGrowth:
    """Frontier expansion must not move the ground under living agents."""

    def _world(self, **kw):
        from world.world import World

        return World(width=24, height=24, seed=7, **kw)

    def test_grow_preserves_every_existing_tile(self):
        world = self._world()
        before = [
            (t.terrain_type, t.fertility, t.moisture)
            for row in world.tiles
            for t in row
        ]
        assert world.grow(8)
        assert (world.width, world.height) == (32, 32)
        after = [
            (
                world.tiles[y][x].terrain_type,
                world.tiles[y][x].fertility,
                world.tiles[y][x].moisture,
            )
            for y in range(24)
            for x in range(24)
        ]
        assert before == after, "growth must not disturb the existing map"

    def test_grow_preserves_object_coordinates(self):
        from world.object_registry import ObjectRegistry, register_builtin_objects

        register_builtin_objects()
        world = self._world()
        placed = []
        for x, y in ((1, 1), (5, 9), (20, 3)):
            tile = world.get_tile(x, y)
            if tile is None or not tile.is_passable():
                continue
            obj = ObjectRegistry.create("berry", x, y)
            if obj is not None and world.add_object(obj):
                placed.append((obj.id, x, y))
        assert placed, "fixture needs at least one placed object"
        world.grow(8)
        for obj_id, x, y in placed:
            obj = world.objects.get(obj_id)
            assert obj is not None, "growth must not drop objects"
            assert (obj.x, obj.y) == (x, y)

    def test_grow_extends_the_signal_fields_with_zeros(self):
        world = self._world(
            signal_config={"enabled": True, "channels": 3, "strength": 1.0}
        )
        world.emit_signal(2, 2, vector=np.array([1.0, -1.0, 0.5]))
        before = float(world.pheromones[2, 2])
        before_vec = world.comm_field[2, 2].copy()
        world.grow(8)
        assert world.pheromones.shape == (32, 32)
        assert world.comm_field.shape == (32, 32, 3)
        assert float(world.pheromones[2, 2]) == pytest.approx(before)
        assert np.allclose(world.comm_field[2, 2], before_vec)
        # Frontier starts silent
        assert float(world.pheromones[30, 30]) == 0.0

    def test_growth_is_off_by_default_and_density_gated(self):
        world = self._world()
        assert world.growth_config == {}
        for _ in range(3):
            world._maybe_grow()
        assert (world.width, world.height) == (24, 24)

    def test_grow_rejects_nonpositive_steps(self):
        world = self._world()
        assert world.grow(0) is False
        assert (world.width, world.height) == (24, 24)


# --- trait heritability ---------------------------------------------------


class TestTraitMutation:
    """
    Traits must be able to change across generations, or every niche axis is
    inert. clone_agent mutates weights only; without mutate_genome_traits the
    in-world reproduction path freezes traits per lineage, and a selective
    sweep to one lineage then pins trait variance at exactly zero forever.
    """

    def _genome(self):
        from agents.genome import Genome, create_default_trait_config

        ecology.set_active_ecology(EcologyConfig(enabled=True))
        np.random.seed(3)
        return Genome.random(16, create_default_trait_config())

    def test_zero_std_is_a_no_op(self):
        from agents.evolution import mutate_genome_traits

        genome = self._genome()
        before = dict(genome.traits)
        mutate_genome_traits(genome, 0.0)
        assert genome.traits == before

    def test_mutation_moves_traits_and_respects_bounds(self):
        from agents.evolution import mutate_genome_traits

        genome = self._genome()
        before = dict(genome.traits)
        for _ in range(50):
            mutate_genome_traits(genome, 0.2)
        assert genome.traits != before
        for name, (lo, hi) in ecology.TRAIT_RANGES.items():
            assert lo <= genome.traits[name] <= hi, name
        assert 0.5 <= genome.traits["metabolism_rate"] <= 2.0

    def test_trait_variance_survives_a_lineage_sweep(self):
        """
        The measurement that motivated the fix: clone a single founder many
        times and check the descendants are not all identical.
        """
        from agents.agent import Agent
        from agents.brain import Brain
        from agents.evolution import clone_agent
        from agents.genome import Genome, create_default_trait_config

        ecology.set_active_ecology(EcologyConfig(enabled=True))
        np.random.seed(4)
        genome = Genome.random(
            weight_count=Brain.calculate_weight_count(),
            trait_config=create_default_trait_config(),
        )
        founder = Agent(x=0, y=0, genome=genome)

        frozen = [
            clone_agent(founder, mutate=True, mutation_std=0.02) for _ in range(20)
        ]
        assert (
            len({round(a.traits["diet"], 6) for a in frozen}) == 1
        ), "without trait mutation a lineage is phenotypically frozen"

        varied = [
            clone_agent(
                founder, mutate=True, mutation_std=0.02, trait_mutation_std=0.05
            )
            for _ in range(20)
        ]
        assert len({round(a.traits["diet"], 6) for a in varied}) > 10
        assert np.std([a.traits["body_size"] for a in varied]) > 0.0

    def test_agent_traits_track_the_mutated_genome(self):
        """The phenotype the agent uses must match its mutated genome."""
        from agents.agent import Agent
        from agents.brain import Brain
        from agents.evolution import clone_agent
        from agents.genome import Genome, create_default_trait_config

        ecology.set_active_ecology(EcologyConfig(enabled=True))
        np.random.seed(5)
        genome = Genome.random(
            weight_count=Brain.calculate_weight_count(),
            trait_config=create_default_trait_config(),
        )
        parent = Agent(x=0, y=0, genome=genome)
        child = clone_agent(
            parent, mutate=True, mutation_std=0.02, trait_mutation_std=0.1
        )
        assert child.traits == child.genome.traits
