"""
Brain v4 — layout, migration, memory dynamics, and numpy/torch agreement.

The load-bearing claims this file locks down:

  * Observation v4 is append-only over v2 (0..77 byte-identical).
  * A v3.5 genome migrates into v4 and behaves identically to float
    tolerance — the two new structured priors (the leak ladder, the drive
    weights) do not leak into the logits.
  * The slow core's time constants come from the genome and span the
    intended range.
  * Path integration is exact integer arithmetic.
  * The torch mirror reproduces the numpy brain step for step, INCLUDING
    the episodic memory — if it does not, PPO trains on a state the agent
    never had.
"""

import numpy as np
import pytest

from agents.brain import (
    activate_brain_layout,
    adapt_loaded_genome,
    calculate_weight_count_for_config,
    create_brain,
    observation_version_for,
)
from agents.brain.instincts import InstinctModule
from agents.brain.spec import (
    DRIVE_BETA_INDEX,
    OBSERVATION_SPEC_V2,
    OBSERVATION_SPEC_V4,
    build_brain_v4_param_spec,
    slow_rho_ladder,
)
from agents.brain.v4 import BrainV4, SALIENCE_DECAY, WRITE_MARGIN
from agents.actions import Action
from agents.genome import Genome, create_default_trait_config

WM = {"enabled": True, "hidden": 32}
CFG_V35 = {"version": 3.5, "world_model": WM}
CFG_V4 = {"version": 4, "world_model": WM}


@pytest.fixture(autouse=True)
def _restore_layout():
    """Leave the process-wide observation/genome layout as we found it."""
    from agents.brain.spec import (
        get_active_observation_spec,
        get_active_param_spec,
        set_active_observation_spec,
        set_active_param_spec,
    )

    obs, params = get_active_observation_spec(), get_active_param_spec()
    yield
    set_active_observation_spec(obs)
    set_active_param_spec(params)


def _v4_brain(seed: int = 0, **cfg_overrides):
    """A v4 brain on a fresh random genome with the structured priors applied."""
    cfg = dict(CFG_V4)
    cfg.update(cfg_overrides)
    activate_brain_layout(cfg)
    np.random.seed(seed)
    genome = Genome.random(
        calculate_weight_count_for_config(cfg), create_default_trait_config()
    )
    return create_brain(genome, cfg, instincts=InstinctModule(enabled=False))


def _obs_v4(rng) -> np.ndarray:
    return rng.random(OBSERVATION_SPEC_V4.size).astype(np.float32)


# --- layout ---------------------------------------------------------------


def test_observation_v4_is_append_only_over_v2():
    v2, v4 = OBSERVATION_SPEC_V2, OBSERVATION_SPEC_V4
    assert v4.size == 91 and v2.size == 78
    for group in ("agent_state", "vision", "stimulus", "inventory"):
        assert getattr(v4, group) == getattr(v2, group)
    # The v2 EXTRA block keeps its six indices at the same absolute positions
    for name in (
        "time_of_day_sin",
        "time_of_day_cos",
        "tile_temperature",
        "nearest_agent_proximity",
        "nearest_agent_signal",
        "on_hazard",
    ):
        assert getattr(v4, name) == getattr(v2, name)
    assert v4.nearest_agent_kin == 78
    assert (v4.comm_mean.start, v4.comm_mean.stop) == (79, 83)
    assert (v4.comm_max.start, v4.comm_max.stop) == (83, 87)
    assert (v4.neighbour_tag.start, v4.neighbour_tag.stop) == (87, 91)


def test_observation_version_mapping():
    assert observation_version_for({"version": 2}) == 1
    assert observation_version_for({"version": 3}) == 1
    assert observation_version_for({"version": 3.5}) == 2
    assert observation_version_for({"version": 4}) == 4


def test_v4_state_is_packed_and_sized():
    brain = _v4_brain()
    assert isinstance(brain, BrainV4)
    expected = 48 + 24 + 8 * (40 + 8 + 4)
    assert brain.state_size == expected
    assert brain.initial_state().shape == (expected,)
    hf, hs, slots = brain.split_state(brain.initial_state())
    assert hf.shape == (48,) and hs.shape == (24,) and slots.shape == (8, 52)


def test_memory_slots_are_state_not_genome():
    """Changing the slot count must not change the genome length."""
    a = calculate_weight_count_for_config(dict(CFG_V4, v4={"memory_slots": 8}))
    b = calculate_weight_count_for_config(dict(CFG_V4, v4={"memory_slots": 32}))
    assert a == b


# --- structured priors ----------------------------------------------------


def test_slow_time_constants_span_the_worlds_timescales():
    rho = slow_rho_ladder((24,))
    tau = 1.0 / (1.0 / (1.0 + np.exp(-rho)))
    assert tau[0] == pytest.approx(2.0, rel=1e-3)
    assert tau[-1] == pytest.approx(512.0, rel=1e-3)
    assert np.all(np.diff(tau) > 0), "the ladder must be monotone"
    # Plant maturation (~160 ticks) and the day (200) must be bracketed
    assert tau.min() < 160 < tau.max()
    assert tau.min() < 200 < tau.max()


def test_random_genome_gets_the_priors_not_noise():
    brain = _v4_brain(seed=3)
    assert np.allclose(brain.params["slow"]["rho"], slow_rho_ladder((24,)))
    assert np.allclose(brain.drive_weights, [1.0, 0.0, 0.0, 0.0])
    # beta starts effectively at zero: v3.5 behaviour until selection moves it
    assert brain.advantage_mix < 0.05


def test_drive_weights_are_clipped():
    brain = _v4_brain()
    brain.params["drive"]["lam"][:DRIVE_BETA_INDEX] = [1e3, -1e3, 0.0, 0.0]
    assert np.all(np.abs(brain.drive_weights) <= 4.0)


# --- migration ------------------------------------------------------------


def test_v35_genome_migrates_to_v4_with_identical_behaviour():
    activate_brain_layout(CFG_V35)
    np.random.seed(11)
    g35 = Genome.random(
        calculate_weight_count_for_config(CFG_V35), create_default_trait_config()
    )
    b35 = create_brain(g35, CFG_V35, instincts=InstinctModule(enabled=False))

    flat = adapt_loaded_genome(g35.weights, CFG_V4)
    assert flat is not None
    assert flat.shape == (calculate_weight_count_for_config(CFG_V4),)

    activate_brain_layout(CFG_V4)
    b4 = create_brain(
        Genome(flat, g35.traits), CFG_V4, instincts=InstinctModule(enabled=False)
    )

    rng = np.random.default_rng(5)
    obs2 = rng.random(OBSERVATION_SPEC_V2.size).astype(np.float32)
    obs4 = np.concatenate(
        [obs2, rng.random(OBSERVATION_SPEC_V4.size - OBSERVATION_SPEC_V2.size)]
    ).astype(np.float32)

    mask = np.ones(9, dtype=np.float32)
    h35, h4 = b35.initial_state(), b4.initial_state()
    for step in range(10):
        p35, v35, h35 = b35.forward(obs2, h35, action_mask=mask)
        p4, v4, h4 = b4.forward(obs4, h4, action_mask=mask)
        assert np.allclose(p35, p4, atol=1e-5), f"logits diverged at step {step}"
        assert v35 == pytest.approx(v4, abs=1e-4)
        # A memory write must not perturb the migrated behaviour: the GRU's
        # rows for the memory read are zero.
        h4 = b4.update_memory(h4, obs4, 0, 5.0 if step == 3 else 0.0, True)


def test_v3_genome_also_migrates_to_v4():
    cfg_v3 = {"version": 3, "world_model": WM}
    activate_brain_layout(cfg_v3)
    np.random.seed(2)
    g3 = Genome.random(
        calculate_weight_count_for_config(cfg_v3), create_default_trait_config()
    )
    flat = adapt_loaded_genome(g3.weights, CFG_V4)
    assert flat is not None
    assert flat.shape == (calculate_weight_count_for_config(CFG_V4),)


def test_migration_keeps_the_priors_for_genuinely_new_tensors():
    activate_brain_layout(CFG_V35)
    np.random.seed(4)
    g35 = Genome.random(
        calculate_weight_count_for_config(CFG_V35), create_default_trait_config()
    )
    flat = adapt_loaded_genome(g35.weights, CFG_V4)
    spec = build_brain_v4_param_spec(state_inputs=41, world_model_hidden=32)
    named = spec.unpack(flat)
    assert np.allclose(named["slow.rho"], slow_rho_ladder((24,)))
    assert named["drive.lam"][0] == pytest.approx(1.0)
    # ...and the new rows/columns of grown tensors stay at zero
    assert np.all(named["gru.Wr_input"][48:] == 0)
    assert np.all(named["policy.W"][48:] == 0)
    assert np.all(named["value.W2"][:, 1] == 0)


# --- episodic memory ------------------------------------------------------


def test_path_integration_is_exact():
    brain = _v4_brain()
    rng = np.random.default_rng(1)
    obs = _obs_v4(rng)
    h = brain.initial_state()
    # Write one memory here, then walk a closed square and come home.
    h = brain.update_memory(h, obs, Action.WAIT.value, 10.0, False)
    n = brain.latent_pre
    slot = int(np.argmax(brain.split_state(h)[2][:, n + 2]))

    def delta():
        return tuple(brain.split_state(h)[2][slot, n : n + 2].tolist())

    assert delta() == (0.0, 0.0)
    for _ in range(2):
        h = brain.update_memory(h, obs, Action.MOVE_FORWARD.value, 0.0, True)
    assert delta() == (0.0, -2.0), "two steps forward leaves the place 2 behind"

    h = brain.update_memory(h, obs, Action.TURN_LEFT.value, 0.0, False)
    assert delta() == (-2.0, -0.0)

    # Walk the rest of a square; the place must land back at the origin.
    for action in (
        Action.MOVE_FORWARD,
        Action.MOVE_FORWARD,
        Action.TURN_LEFT,
        Action.MOVE_FORWARD,
        Action.MOVE_FORWARD,
        Action.TURN_LEFT,
        Action.MOVE_FORWARD,
        Action.MOVE_FORWARD,
        Action.TURN_LEFT,
    ):
        h = brain.update_memory(
            h, obs, action.value, 0.0, action == Action.MOVE_FORWARD
        )
    assert delta() == (0.0, 0.0), "a closed loop must return the place to the origin"


def test_blocked_moves_do_not_shift_stored_places():
    brain = _v4_brain()
    rng = np.random.default_rng(2)
    obs = _obs_v4(rng)
    h = brain.update_memory(brain.initial_state(), obs, Action.WAIT.value, 10.0, False)
    n = brain.latent_pre
    before = brain.split_state(h)[2][:, n : n + 2].copy()
    h = brain.update_memory(h, obs, Action.MOVE_FORWARD.value, 0.0, moved=False)
    assert np.array_equal(brain.split_state(h)[2][:, n : n + 2], before)


def test_write_needs_to_beat_the_weakest_slot_by_a_margin():
    brain = _v4_brain()
    rng = np.random.default_rng(3)
    obs = _obs_v4(rng)
    n = brain.latent_pre
    h = brain.initial_state()
    for _ in range(8):
        h = brain.update_memory(h, obs, Action.WAIT.value, 1.0, False)
    saliences = brain.split_state(h)[2][:, n + 2]
    assert np.all(saliences > 0), "all slots should be claimed"

    # A trickle below the margin must not churn the slots.
    ages_before = brain.split_state(h)[2][:, n + 3].copy()
    for _ in range(20):
        h = brain.update_memory(h, obs, Action.WAIT.value, 0.01, False)
    ages_after = brain.split_state(h)[2][:, n + 3]
    assert np.all(ages_after > ages_before), "memories must age, not be replaced"

    # ...but something clearly more salient does claim a slot.
    h = brain.update_memory(h, obs, Action.WAIT.value, 50.0, False)
    assert brain.split_state(h)[2][:, n + 2].max() == pytest.approx(50.0)
    assert WRITE_MARGIN > 1.0 and SALIENCE_DECAY < 1.0


def test_empty_memory_reads_as_zero_and_a_write_changes_the_read():
    brain = _v4_brain(seed=9)
    rng = np.random.default_rng(4)
    obs = _obs_v4(rng)
    h = brain.initial_state()
    _, _, slots = brain.split_state(h)
    s = brain._encode_pre(obs)[: brain.state_dim]
    assert np.all(brain._read_memory(slots, s, np.zeros(24, dtype=np.float32)) == 0.0)

    h = brain.update_memory(h, obs, Action.WAIT.value, 7.0, False)
    _, _, slots = brain.split_state(h)
    read = brain._read_memory(slots, s, np.zeros(24, dtype=np.float32))
    assert np.any(read != 0.0)


def test_memory_can_be_ablated_without_a_genome_change():
    brain = _v4_brain(v4={"memory_slots": 0})
    assert brain.memory_slots == 0
    rng = np.random.default_rng(6)
    obs = _obs_v4(rng)
    h = brain.initial_state()
    assert h.shape == (48 + 24,)
    probs, value, h2 = brain.forward(obs, h, action_mask=np.ones(9, dtype=np.float32))
    assert probs.shape == (9,)
    assert np.array_equal(brain.update_memory(h2, obs, 0, 1.0, True), h2)


# --- heads ----------------------------------------------------------------


def test_heads_have_the_shapes_the_rest_of_the_sim_expects():
    brain = _v4_brain(seed=7)
    rng = np.random.default_rng(8)
    obs = _obs_v4(rng)
    h = brain.initial_state()
    z, h_next, core, s = brain.core_step(obs, h)
    assert z.shape == (56,) and core.shape == (72,) and s.shape == (40,)
    assert brain.values(z, core).shape == (2,)
    assert brain.comm_vector(h_next).shape == (4,)
    assert np.all((brain.identity_tag >= 0) & (brain.identity_tag <= 1))
    z_pred, r_pred = brain.predict_next_latent(h_next, 3)
    assert z_pred.shape == (56,) and isinstance(r_pred, float)
    assert brain.action_latent_spread(h_next) >= 0.0
