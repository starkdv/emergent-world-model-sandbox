"""
Tests for fading instincts and the auto-eat replacement (Brain v3 Phase 2).

Covers:
- Instinct strength fading through the Agent decision path
- Hunger-scaled EAT bias (replacement for the forced auto-eat override)
- Config plumbing: Agent.instinct_config → InstinctModule.from_config
- Instinct module preservation across brain rebuilds (clone / inherit)
- Hungry agents holding food still eat (statistically) without the override
"""

import numpy as np
import pytest

from agents.agent import Agent
from agents.actions import Action
from agents.brain import Brain
from agents.brain.instincts import InstinctModule
from agents.brain.spec import DEFAULT_OBSERVATION_SPEC
from agents.genome import Genome, create_default_trait_config
from world.world import World
from world.object_registry import ObjectRegistry, register_builtin_objects


@pytest.fixture(autouse=True)
def _reset_instinct_config():
    """Keep the class-level config from leaking between tests."""
    saved = Agent.instinct_config
    Agent.instinct_config = None
    yield
    Agent.instinct_config = saved


def _make_agent(x=5, y=5):
    weight_count = Brain.calculate_weight_count()
    genome = Genome.random(weight_count, create_default_trait_config())
    return Agent(x=x, y=y, genome=genome)


def _hungry_obs(urgency=1.0):
    """Observation of a hungry agent with food in inventory."""
    spec = DEFAULT_OBSERVATION_SPEC
    obs = np.zeros(spec.size, dtype=np.float32)
    obs[spec.energy_urgency] = urgency
    return obs


class TestInstinctConfig:
    def test_from_config_defaults(self):
        instincts = InstinctModule.from_config(None)
        assert instincts.enabled
        assert instincts.fade_age == InstinctModule.DEFAULT_FADE_AGE
        assert instincts.hunger_eat_bias == InstinctModule.HUNGER_EAT_BIAS

    def test_from_config_overrides(self):
        instincts = InstinctModule.from_config(
            {"enabled": False, "fade_age": None, "hunger_eat_bias": 1.0}
        )
        assert not instincts.enabled
        assert instincts.fade_age is None
        assert instincts.hunger_eat_bias == 1.0

    def test_agent_picks_up_class_level_config(self):
        Agent.instinct_config = {"fade_age": 50}
        agent = _make_agent()
        assert agent.brain.instincts.fade_age == 50

    def test_agent_defaults_to_fading_instincts(self):
        agent = _make_agent()
        assert agent.brain.instincts.fade_age == InstinctModule.DEFAULT_FADE_AGE
        assert agent.brain.instincts.strength_at(0) == 1.0
        assert agent.brain.instincts.strength_at(10_000) == 0.0


class TestHungerEatBias:
    def test_hunger_scales_eat_bias(self):
        instincts = InstinctModule()
        mask = np.ones(8)

        for urgency in (0.0, 0.2, 0.6, 1.0):
            logits = np.zeros(8)
            instincts.apply(logits, _hungry_obs(urgency), mask, strength=1.0)
            expected = InstinctModule.EAT_BIAS + (
                InstinctModule.HUNGER_EAT_BIAS * urgency
            )
            assert logits[Action.EAT.value] == pytest.approx(expected)

    def test_hunger_bias_requires_valid_eat(self):
        instincts = InstinctModule()
        logits = np.zeros(8)
        mask = np.ones(8)
        mask[Action.EAT.value] = 0  # no food in inventory → EAT masked
        instincts.apply(logits, _hungry_obs(1.0), mask, strength=1.0)
        assert logits[Action.EAT.value] == 0.0

    def test_hunger_bias_fades_with_strength(self):
        instincts = InstinctModule()
        logits = np.zeros(8)
        instincts.apply(logits, _hungry_obs(1.0), np.ones(8), strength=0.25)
        expected = (InstinctModule.EAT_BIAS + InstinctModule.HUNGER_EAT_BIAS) * 0.25
        assert logits[Action.EAT.value] == pytest.approx(expected)

    def test_hungry_newborn_eats_with_high_probability(self):
        """The prior must make EAT dominant for a hungry random-weight agent
        — the behavioural guarantee that replaced the forced auto-eat.
        (Agents sample every tick, so a high mean P(EAT) means they eat
        within a couple of ticks even when it isn't the modal action.)"""
        np.random.seed(123)
        weight_count = Brain.calculate_weight_count()
        eat_probs = []
        trials = 50
        for _ in range(trials):
            genome = Genome.random(weight_count, create_default_trait_config())
            brain = Brain(genome)
            probs, _, _ = brain.forward(
                _hungry_obs(1.0),
                brain.initial_state(),
                action_mask=np.ones(8),
                instinct_strength=1.0,
            )
            eat_probs.append(probs[Action.EAT.value])

        eat_probs = np.array(eat_probs)
        # +4.0 logits: EAT should usually dominate and always be likely
        assert np.mean(eat_probs) > 0.5
        assert np.mean(eat_probs > 0.5) >= 0.6
        assert np.mean(eat_probs > 1.0 / 8.0) >= 0.95  # ≫ uniform chance


class TestFadingThroughAgentPath:
    def test_strength_follows_age(self):
        agent = _make_agent()
        fade = agent.brain.instincts.fade_age
        assert agent.brain.instincts.strength_at(agent.age) == 1.0
        agent.age = fade // 2
        assert agent.brain.instincts.strength_at(agent.age) == pytest.approx(0.5)
        agent.age = fade
        assert agent.brain.instincts.strength_at(agent.age) == 0.0

    def test_adult_probs_equal_pure_network(self):
        """Past fade_age the policy must be identical to an instinct-free
        brain — adults act purely on learned weights."""
        weight_count = Brain.calculate_weight_count()
        genome = Genome.random(weight_count, create_default_trait_config())
        faded = Brain(genome, instincts=InstinctModule(fade_age=150))
        pure = Brain(genome, instincts=InstinctModule(enabled=False))

        obs = _hungry_obs(1.0)
        h = faded.initial_state()
        mask = np.ones(8)
        probs_faded, _, _ = faded.forward(
            obs, h, action_mask=mask, instinct_strength=faded.instincts.strength_at(200)
        )
        probs_pure, _, _ = pure.forward(obs, h, action_mask=mask)
        assert np.allclose(probs_faded, probs_pure)

    def test_instincts_survive_clone(self):
        from agents.evolution import clone_agent

        Agent.instinct_config = {"fade_age": 77}
        parent = _make_agent()
        child = clone_agent(parent, mutate=True, mutation_std=0.02)
        assert child.brain.instincts.fade_age == 77

    def test_instincts_survive_inherit_knowledge(self):
        Agent.instinct_config = {"fade_age": 88}
        agent = _make_agent()
        agent.inherit_knowledge(agent.genome.weights.copy())
        assert agent.brain.instincts.fade_age == 88


class TestAutoEatRemoval:
    def test_hungry_agent_with_food_eats_within_a_few_ticks(self):
        """End-to-end: a hungry agent holding a berry must still eat soon
        after the forced override was removed (prior, not force)."""
        register_builtin_objects()
        world = World(width=10, height=10, seed=7)

        ate = 0
        trials = 10
        for t in range(trials):
            agent = _make_agent(x=5, y=5)
            agent.energy = agent.max_energy * 0.2  # very hungry
            berry = ObjectRegistry.create("berry", -1, -1)
            world.objects[berry.id] = berry
            agent.inventory.append(berry.id)
            world.add_agent(agent)

            for _ in range(10):
                energy_before = agent.energy
                agent.update(world)
                if agent.energy > energy_before:  # eating gains energy
                    ate += 1
                    break
            world.remove_agent(agent.id)

        assert ate >= trials * 0.8, f"only {ate}/{trials} hungry agents ate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
