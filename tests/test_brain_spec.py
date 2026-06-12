"""
Tests for the spec-driven brain refactor (Phase 1 of Brain v3).

Covers:
- ParamSpec weight counting, unpack/pack roundtrip, zero-copy views
- Exact layout equivalence with the legacy hand-written unpacker
- Genome sync (Lamarckian inheritance) through ParamSpec.pack
- ObservationSpec layout vs. perception module
- InstinctModule extraction: biases, gating, strength scaling/fading
"""

import numpy as np
import pytest

from agents.brain import Brain
from agents.brain.instincts import InstinctModule
from agents.brain.spec import (
    DEFAULT_OBSERVATION_SPEC,
    build_brain_param_spec,
    build_observation_spec,
)
from agents.actions import Action
from agents.genome import Genome, create_default_trait_config

# ---------------------------------------------------------------------------
# Legacy reference unpacker — frozen copy of the pre-refactor
# brain_utils.unpack_weights, used to prove layout equivalence.
# ---------------------------------------------------------------------------


def _legacy_unpack_weights(
    flat_weights, input_size, encoder_layers, gru_hidden_size, output_size
):
    params = {
        "encoder_weights": [],
        "encoder_biases": [],
        "gru": {},
        "policy_head": {},
        "value_head": {},
    }
    idx = 0
    encoder_sizes = [input_size] + encoder_layers
    for i in range(len(encoder_sizes) - 1):
        in_size, out_size = encoder_sizes[i], encoder_sizes[i + 1]
        w_size = in_size * out_size
        params["encoder_weights"].append(
            flat_weights[idx : idx + w_size].reshape(in_size, out_size)
        )
        idx += w_size
        params["encoder_biases"].append(flat_weights[idx : idx + out_size])
        idx += out_size

    encoder_out = encoder_layers[-1]
    for gate in ("r", "z", "h"):
        params["gru"][f"W{gate}_input"] = flat_weights[
            idx : idx + encoder_out * gru_hidden_size
        ].reshape(encoder_out, gru_hidden_size)
        idx += encoder_out * gru_hidden_size
        params["gru"][f"W{gate}_hidden"] = flat_weights[
            idx : idx + gru_hidden_size * gru_hidden_size
        ].reshape(gru_hidden_size, gru_hidden_size)
        idx += gru_hidden_size * gru_hidden_size
        params["gru"][f"b{gate}"] = flat_weights[idx : idx + gru_hidden_size]
        idx += gru_hidden_size

    params["policy_head"]["W"] = flat_weights[
        idx : idx + gru_hidden_size * output_size
    ].reshape(gru_hidden_size, output_size)
    idx += gru_hidden_size * output_size
    params["policy_head"]["b"] = flat_weights[idx : idx + output_size]
    idx += output_size

    params["value_head"]["W"] = flat_weights[idx : idx + gru_hidden_size].reshape(
        gru_hidden_size, 1
    )
    idx += gru_hidden_size
    params["value_head"]["b"] = flat_weights[idx : idx + 1]
    return params


def _make_brain(weights=None):
    weight_count = Brain.calculate_weight_count()
    if weights is None:
        genome = Genome.random(weight_count, create_default_trait_config())
    else:
        genome = Genome(weights, create_default_trait_config_values())
    return Brain(genome)


def create_default_trait_config_values():
    return {name: lo for name, (lo, hi) in create_default_trait_config().items()}


class TestParamSpec:
    def test_count_matches_legacy_formula(self):
        spec = build_brain_param_spec(72, [32], 32, 8)
        # Encoder 2336 + GRU 6240 + policy 264 + value 33
        assert spec.count() == 8873

    def test_count_for_non_default_architecture(self):
        spec = build_brain_param_spec(60, [48, 24], 16, 6)
        expected = (
            (60 * 48 + 48)
            + (48 * 24 + 24)
            + 3 * (24 * 16 + 16 * 16 + 16)
            + (16 * 6 + 6)
            + (16 + 1)
        )
        assert spec.count() == expected

    def test_pack_unpack_roundtrip(self):
        spec = build_brain_param_spec()
        flat = np.random.randn(spec.count()).astype(np.float32)
        named = spec.unpack(flat)
        repacked = spec.pack(named)
        assert repacked.dtype == np.float32
        assert np.array_equal(repacked, flat)

    def test_unpack_returns_views(self):
        spec = build_brain_param_spec()
        flat = np.zeros(spec.count(), dtype=np.float32)
        named = spec.unpack(flat)
        named["policy.b"][0] = 7.0
        # In-place update of a view must be visible in the flat vector
        assert 7.0 in flat

    def test_unpack_rejects_wrong_length(self):
        spec = build_brain_param_spec()
        with pytest.raises(ValueError):
            spec.unpack(np.zeros(spec.count() + 1))

    def test_layout_matches_legacy_unpacker(self):
        """Every tensor must occupy exactly the same genome slice as before."""
        spec = build_brain_param_spec(72, [32], 32, 8)
        flat = np.random.randn(spec.count())
        legacy = _legacy_unpack_weights(flat, 72, [32], 32, 8)

        brain = _make_brain()
        new = brain.spec.unpack(flat)

        assert np.array_equal(new["encoder.0.W"], legacy["encoder_weights"][0])
        assert np.array_equal(new["encoder.0.b"], legacy["encoder_biases"][0])
        for gate in ("r", "z", "h"):
            assert np.array_equal(
                new[f"gru.W{gate}_input"], legacy["gru"][f"W{gate}_input"]
            )
            assert np.array_equal(
                new[f"gru.W{gate}_hidden"], legacy["gru"][f"W{gate}_hidden"]
            )
            assert np.array_equal(new[f"gru.b{gate}"], legacy["gru"][f"b{gate}"])
        assert np.array_equal(new["policy.W"], legacy["policy_head"]["W"])
        assert np.array_equal(new["policy.b"], legacy["policy_head"]["b"])
        assert np.array_equal(new["value.W"], legacy["value_head"]["W"])
        assert np.array_equal(new["value.b"], legacy["value_head"]["b"])


class TestBrainSpecIntegration:
    def test_nested_params_share_memory_with_named(self):
        brain = _make_brain()
        brain.params["policy_head"]["W"][0, 0] = 42.0
        assert brain.named_params["policy.W"][0, 0] == 42.0

    def test_genome_sync_roundtrip_preserves_forward(self):
        """Lamarckian sync: pack(params) → genome → new brain must behave
        identically."""
        from agents.learning import AgentLearner

        brain = _make_brain()
        # Perturb parameters in place, as the learner does
        brain.params["policy_head"]["W"] += 0.05
        brain.params["value_head"]["b"] -= 0.1

        learner = AgentLearner(compute_backend="numpy")
        learner._sync_genome_weights(brain)

        rebuilt = Brain(brain.genome)
        obs = np.random.randn(72).astype(np.float32)
        h = brain.initial_state()
        probs_a, val_a, h_a = brain.forward(obs, h)
        probs_b, val_b, h_b = rebuilt.forward(obs, h)

        assert np.allclose(probs_a, probs_b, atol=1e-5)
        assert np.isclose(val_a, val_b, atol=1e-5)
        assert np.allclose(h_a, h_b, atol=1e-5)


class TestObservationSpec:
    def test_size_matches_perception(self):
        from utils.agents import get_observation_size

        assert DEFAULT_OBSERVATION_SPEC.size == get_observation_size() == 72

    def test_group_layout(self):
        spec = DEFAULT_OBSERVATION_SPEC
        assert spec.agent_state == slice(0, 8)
        assert spec.vision == slice(8, 58)
        assert spec.stimulus == slice(58, 66)
        assert spec.inventory == slice(66, 72)
        assert spec.vision_shape == (5, 5, 2)

    def test_stimulus_field_indices(self):
        """Field indices must match the documented perception layout."""
        spec = DEFAULT_OBSERVATION_SPEC
        assert spec.food_on_tile == 58
        assert spec.seed_on_tile == 59
        assert spec.food_ahead == 60
        assert spec.resource_ahead == 61
        assert spec.nearest_food_prox == 62
        assert spec.food_dir_match == 63
        assert spec.energy_urgency == 64
        assert spec.can_interact == 65

    def test_vision_grid_indexing(self):
        spec = DEFAULT_OBSERVATION_SPEC
        obs = np.zeros(spec.size, dtype=np.float32)
        # type_enc of tile (row=1, col=3) lives at 8 + (1*5+3)*2
        obs[8 + (1 * 5 + 3) * 2] = 0.9
        grid = spec.vision_grid(obs)
        assert grid[1, 3, 0] == pytest.approx(0.9)

    def test_scales_with_vision_radius(self):
        spec = build_observation_spec(vision_radius=3)
        assert spec.vision_shape == (7, 7, 2)
        assert spec.size == 8 + 7 * 7 * 2 + 8 + 6


class TestInstinctModule:
    def _obs(
        self, prox=0.0, dir_match=1.0, ahead=0.0, food_left=False, food_right=False
    ):
        spec = DEFAULT_OBSERVATION_SPEC
        obs = np.zeros(spec.size, dtype=np.float32)
        obs[spec.nearest_food_prox] = prox
        obs[spec.food_dir_match] = dir_match
        obs[spec.food_ahead] = ahead
        grid = obs[spec.vision].reshape(spec.vision_shape)
        if food_left:
            grid[2, 0, 0] = 1.0  # food-like type on agent's left
        if food_right:
            grid[2, 4, 0] = 1.0  # food-like type on agent's right
        return obs

    def test_interaction_biases_match_legacy_constants(self):
        instincts = InstinctModule()
        logits = np.zeros(8)
        mask = np.ones(8)
        instincts.apply(logits, self._obs(), mask, strength=1.0)
        assert logits[Action.PICK_UP.value] == pytest.approx(1.5)
        assert logits[Action.EAT.value] == pytest.approx(1.0)
        assert logits[Action.USE.value] == pytest.approx(0.5)
        assert logits[Action.MOVE_FORWARD.value] == 0.0
        assert logits[Action.WAIT.value] == 0.0

    def test_biases_respect_action_mask(self):
        instincts = InstinctModule()
        logits = np.zeros(8)
        mask = np.zeros(8)
        mask[Action.EAT.value] = 1
        instincts.apply(logits, self._obs(), mask, strength=1.0)
        assert logits[Action.PICK_UP.value] == 0.0
        assert logits[Action.EAT.value] == pytest.approx(1.0)

    def test_turn_toward_food_left(self):
        instincts = InstinctModule()
        logits = np.zeros(8)
        mask = np.ones(8)
        obs = self._obs(prox=0.5, dir_match=0.3, ahead=0.0, food_left=True)
        instincts.apply(logits, obs, mask, strength=1.0)
        expected = 0.8 * 0.5  # base_bias = TURN_TOWARD_FOOD_BIAS * prox
        assert logits[Action.TURN_LEFT.value] == pytest.approx(expected)
        assert logits[Action.TURN_RIGHT.value] == pytest.approx(expected * 0.2)

    def test_turn_toward_food_right(self):
        instincts = InstinctModule()
        logits = np.zeros(8)
        mask = np.ones(8)
        obs = self._obs(prox=0.5, dir_match=0.3, ahead=0.0, food_right=True)
        instincts.apply(logits, obs, mask, strength=1.0)
        expected = 0.8 * 0.5
        assert logits[Action.TURN_RIGHT.value] == pytest.approx(expected)
        assert logits[Action.TURN_LEFT.value] == pytest.approx(expected * 0.2)

    def test_turn_instinct_gated_when_facing_food(self):
        instincts = InstinctModule()
        logits = np.zeros(8)
        mask = np.ones(8)
        # Facing the food (dir_match high) — no turn bias
        obs = self._obs(prox=0.5, dir_match=0.9, ahead=0.0, food_left=True)
        instincts.apply(logits, obs, mask, strength=1.0)
        assert logits[Action.TURN_LEFT.value] == 0.0
        assert logits[Action.TURN_RIGHT.value] == 0.0

    def test_strength_scales_all_biases(self):
        instincts = InstinctModule()
        logits = np.zeros(8)
        mask = np.ones(8)
        obs = self._obs(prox=0.5, dir_match=0.3, ahead=0.0, food_left=True)
        instincts.apply(logits, obs, mask, strength=0.5)
        assert logits[Action.PICK_UP.value] == pytest.approx(1.5 * 0.5)
        assert logits[Action.TURN_LEFT.value] == pytest.approx(0.8 * 0.5 * 0.5)

    def test_zero_strength_and_disabled_are_inert(self):
        for instincts, strength in [
            (InstinctModule(), 0.0),
            (InstinctModule(enabled=False), 1.0),
        ]:
            logits = np.zeros(8)
            instincts.apply(logits, self._obs(prox=0.5), np.ones(8), strength)
            assert np.all(logits == 0.0)

    def test_strength_fades_with_age(self):
        instincts = InstinctModule(fade_age=150)
        assert instincts.strength_at(0) == pytest.approx(1.0)
        assert instincts.strength_at(75) == pytest.approx(0.5)
        assert instincts.strength_at(150) == 0.0
        assert instincts.strength_at(1000) == 0.0
        # Legacy behaviour: no fade configured → constant strength
        assert InstinctModule().strength_at(10_000) == 1.0

    def test_brain_with_instincts_prefers_pickup(self):
        """Same genome, instincts on vs off: PICK_UP probability must be
        strictly higher with instincts (the +1.5 logit bias)."""
        weight_count = Brain.calculate_weight_count()
        genome = Genome.random(weight_count, create_default_trait_config())
        with_instincts = Brain(genome)
        without = Brain(genome, instincts=InstinctModule(enabled=False))

        obs = np.zeros(72, dtype=np.float32)
        h = with_instincts.initial_state()
        mask = np.ones(8)
        probs_on, _, _ = with_instincts.forward(obs, h, action_mask=mask)
        probs_off, _, _ = without.forward(obs, h, action_mask=mask)
        assert probs_on[Action.PICK_UP.value] > probs_off[Action.PICK_UP.value]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
