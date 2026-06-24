"""
Unit & integration tests for the environment engine (World upgrade W1).

Covers:
- EnvironmentSystem clock: day/night light cycle, seasons, temperature
- temperature_response curve shape
- Weather event lifecycle (rain / drought start, duration, end)
- Multipliers (growth, germination, spawn, decay, metabolism)
- Neutrality: environment disabled → every multiplier exactly 1.0
- B1 fix: with the environment enabled, soil moisture DECREASES without
  rain, recovers during rain / next to water, and is non-monotonic
- Legacy parity: disabled environment keeps the old (buggy-but-frozen)
  constant-drip moisture arithmetic
- B2 fix: sand clamps now sit at the germination thresholds, so seeds
  CAN germinate on sand (just 10× harder)

Author: Karan Vasa
"""

import math
import random

import pytest

from world.environment import EnvironmentSystem, temperature_response
from world.object_registry import ObjectRegistry, register_builtin_objects
from world.systems import SeedGerminationSystem, SoilDynamicsSystem
from world.tiles import TerrainType
from world.world import World

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure a clean registry for every test."""
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    ObjectRegistry._definitions.clear()


def make_world(environment_config=None, **kwargs):
    """A small all-soil world (no parallel) for deterministic tests."""
    defaults = dict(
        seed=42,
        soil_ratio=1.0,
        rock_ratio=0.0,
        water_ratio=0.0,
        sand_ratio=0.0,
        parallel=False,
    )
    defaults.update(kwargs)
    w = World(6, 6, environment_config=environment_config, **defaults)
    for y in range(6):
        for x in range(6):
            tile = w.get_tile(x, y)
            tile.terrain_type = TerrainType.SOIL
            tile.fertility = 0.8
            tile.moisture = 0.6
    return w


WEATHER_KEYS = {
    "rain_start_chance",
    "rain_duration",
    "rain_recovery",
    "drought_start_chance",
    "drought_duration",
    "drought_evaporation_factor",
}


def enabled_config(**overrides):
    """Environment config with weather randomness disabled by default."""
    cfg = {
        "enabled": True,
        "weather": {"rain_start_chance": 0.0, "drought_start_chance": 0.0},
    }
    for key, value in overrides.items():
        if key in WEATHER_KEYS:
            cfg["weather"][key] = value
        else:
            cfg[key] = value
    return cfg


class FakeWorld:
    """Minimal stand-in so EnvironmentSystem.update only needs a tick."""

    def __init__(self, tick=0):
        self.tick = tick


# ===================================================================
# temperature_response curve
# ===================================================================


class TestTemperatureResponse:
    def test_comfort_band_is_full_rate(self):
        for t in (0.3, 0.4, 0.5, 0.6, 0.7):
            assert temperature_response(t) == pytest.approx(1.0)

    def test_zero_at_extremes(self):
        assert temperature_response(0.0) == pytest.approx(0.0)
        assert temperature_response(0.1) == pytest.approx(0.0)
        assert temperature_response(0.9) == pytest.approx(0.0)
        assert temperature_response(1.0) == pytest.approx(0.0)

    def test_linear_falloff_midpoints(self):
        assert temperature_response(0.2) == pytest.approx(0.5)
        assert temperature_response(0.8) == pytest.approx(0.5)

    def test_always_in_unit_interval(self):
        for i in range(101):
            r = temperature_response(i / 100)
            assert 0.0 <= r <= 1.0


# ===================================================================
# Disabled environment = exact neutrality
# ===================================================================


class TestDisabledNeutrality:
    def test_default_is_disabled(self):
        env = EnvironmentSystem()
        assert env.enabled is False

    def test_all_multipliers_are_exactly_one(self):
        env = EnvironmentSystem({"enabled": False})
        # Even with extreme live state, disabled → neutral
        env.temperature = 0.95
        env.light = 0.1
        env.raining = True
        env.drought = True
        assert env.growth_multiplier == 1.0
        assert env.germination_multiplier == 1.0
        assert env.spawn_multiplier == 1.0
        assert env.decay_multiplier == 1.0
        assert env.metabolism_multiplier == 1.0

    def test_update_is_a_no_op_when_disabled(self):
        env = EnvironmentSystem({"enabled": False})
        before = (env.time_of_day, env.light, env.temperature, env.raining)
        env.update(FakeWorld(tick=137))
        assert (env.time_of_day, env.light, env.temperature, env.raining) == before


# ===================================================================
# Clock: day/night and seasons
# ===================================================================


class TestClock:
    def test_light_stays_in_range_over_full_day(self):
        env = EnvironmentSystem(enabled_config(day_length=100, min_light=0.25))
        for tick in range(100):
            env.update(FakeWorld(tick=tick))
            assert 0.25 - 1e-9 <= env.light <= 1.0 + 1e-9

    def test_light_is_periodic(self):
        env = EnvironmentSystem(enabled_config(day_length=100))
        env.update(FakeWorld(tick=17))
        light_a = env.light
        env.update(FakeWorld(tick=117))
        assert env.light == pytest.approx(light_a)

    def test_light_reaches_noon_and_night(self):
        env = EnvironmentSystem(enabled_config(day_length=100, min_light=0.25))
        # sin peaks at quarter-day, troughs at three-quarter-day
        env.update(FakeWorld(tick=25))
        assert env.light == pytest.approx(1.0)
        env.update(FakeWorld(tick=75))
        assert env.light == pytest.approx(0.25)

    def test_temperature_clamped_to_unit_interval(self):
        env = EnvironmentSystem(
            enabled_config(
                base_temperature=0.9,
                season_temp_amplitude=0.5,
                daynight_temp_amplitude=0.3,
            )
        )
        for tick in range(0, 4000, 37):
            env.update(FakeWorld(tick=tick))
            assert 0.0 <= env.temperature <= 1.0

    def test_season_modulates_temperature(self):
        env = EnvironmentSystem(
            enabled_config(
                season_length=2000,
                season_temp_amplitude=0.25,
                daynight_temp_amplitude=0.0,
                day_length=200,
            )
        )
        env.update(FakeWorld(tick=500))  # quarter cycle → +amplitude
        summer = env.temperature
        env.update(FakeWorld(tick=1500))  # three-quarter cycle → -amplitude
        winter = env.temperature
        assert summer == pytest.approx(0.75)
        assert winter == pytest.approx(0.25)


# ===================================================================
# Weather events
# ===================================================================


class TestWeather:
    def test_rain_starts_runs_and_ends(self):
        env = EnvironmentSystem(enabled_config(rain_start_chance=1.0, rain_duration=5))
        env.update(FakeWorld(tick=1))
        assert env.raining is True
        assert env.moisture_recovery_rate == pytest.approx(env.rain_recovery)
        env.rain_start_chance = 0.0  # don't restart the moment it ends
        # Rain counts down and ends after rain_duration updates
        for tick in range(2, 7):
            env.update(FakeWorld(tick=tick))
        assert env.raining is False
        assert env.moisture_recovery_rate == 0.0

    def test_drought_multiplies_evaporation(self):
        env = EnvironmentSystem(enabled_config(drought_evaporation_factor=2.0))
        env.update(FakeWorld(tick=0))
        base = env.evaporation_rate
        env.drought = True
        assert env.evaporation_rate == pytest.approx(base * 2.0)

    def test_drought_lifecycle(self):
        env = EnvironmentSystem(
            enabled_config(drought_start_chance=1.0, drought_duration=3)
        )
        env.update(FakeWorld(tick=1))
        assert env.drought is True
        env.drought_start_chance = 0.0  # don't restart the moment it ends
        for tick in range(2, 5):
            env.update(FakeWorld(tick=tick))
        assert env.drought is False

    def test_no_recovery_without_rain(self):
        env = EnvironmentSystem(enabled_config())
        env.update(FakeWorld(tick=10))
        assert env.raining is False
        assert env.moisture_recovery_rate == 0.0

    def test_evaporation_scales_with_temperature_and_light(self):
        env = EnvironmentSystem(enabled_config())
        env.temperature, env.light = 0.5, 1.0
        hot_noon = env.evaporation_rate
        env.temperature, env.light = 0.3, 0.25
        cool_night = env.evaporation_rate
        assert hot_noon > cool_night > 0.0


# ===================================================================
# Multipliers
# ===================================================================


class TestMultipliers:
    def test_growth_is_light_times_temperature_window(self):
        env = EnvironmentSystem(enabled_config())
        env.light, env.temperature = 0.8, 0.5
        assert env.growth_multiplier == pytest.approx(0.8)
        env.temperature = 0.9  # outside the window
        assert env.growth_multiplier == pytest.approx(0.0)

    def test_germination_uses_temperature_window_only(self):
        env = EnvironmentSystem(enabled_config())
        env.light, env.temperature = 0.25, 0.5
        assert env.germination_multiplier == pytest.approx(1.0)

    def test_spawn_follows_light(self):
        env = EnvironmentSystem(enabled_config())
        env.light = 0.4
        assert env.spawn_multiplier == pytest.approx(0.4)

    def test_decay_faster_when_hot(self):
        env = EnvironmentSystem(enabled_config())
        env.temperature = 0.5
        assert env.decay_multiplier == pytest.approx(1.0)
        env.temperature = 0.9
        assert env.decay_multiplier == pytest.approx(1.4)
        env.temperature = 0.1
        assert env.decay_multiplier == pytest.approx(0.6)

    def test_metabolism_rises_at_both_extremes(self):
        env = EnvironmentSystem(enabled_config(metabolism_temp_coef=0.5))
        env.temperature = 0.5
        assert env.metabolism_multiplier == pytest.approx(1.0)
        env.temperature = 1.0
        assert env.metabolism_multiplier == pytest.approx(1.5)
        env.temperature = 0.0
        assert env.metabolism_multiplier == pytest.approx(1.5)


# ===================================================================
# B1 fix: soil moisture dynamics
# ===================================================================


class TestMoistureDynamics:
    def test_b1_moisture_decreases_without_rain(self):
        """The headline B1 fix: no rain → moisture goes DOWN, not up."""
        world = make_world(environment_config=enabled_config())
        soil = SoilDynamicsSystem()
        world.environment.update(world)
        tile = world.get_tile(2, 2)
        start = tile.moisture
        for _ in range(50):
            soil.update(world)
        assert tile.moisture < start

    def test_moisture_recovers_during_rain(self):
        world = make_world(environment_config=enabled_config())
        world.environment.update(world)
        world.environment.raining = True
        soil = SoilDynamicsSystem()
        tile = world.get_tile(2, 2)
        tile.moisture = 0.3
        for _ in range(50):
            soil.update(world)
        # rain_recovery (0.004) outpaces evaporation (~0.002 max)
        assert tile.moisture > 0.3

    def test_moisture_is_non_monotonic_across_a_weather_cycle(self):
        world = make_world(environment_config=enabled_config())
        world.environment.update(world)
        soil = SoilDynamicsSystem()
        tile = world.get_tile(2, 2)

        for _ in range(30):  # dry spell
            soil.update(world)
        after_dry = tile.moisture
        world.environment.raining = True
        for _ in range(30):  # rain event
            soil.update(world)
        after_rain = tile.moisture

        assert after_dry < 0.6  # fell during the dry spell
        assert after_rain > after_dry  # rose again during rain

    def test_drought_dries_soil_faster(self):
        world_a = make_world(environment_config=enabled_config())
        world_b = make_world(environment_config=enabled_config())
        for w in (world_a, world_b):
            w.environment.update(w)
        world_b.environment.drought = True

        soil = SoilDynamicsSystem()
        for _ in range(50):
            soil.update(world_a)
            soil.update(world_b)
        assert world_b.get_tile(2, 2).moisture < world_a.get_tile(2, 2).moisture

    def test_water_adjacent_tiles_recover_without_rain(self):
        world = make_world(environment_config=enabled_config())
        world.get_tile(0, 0).terrain_type = TerrainType.WATER
        world._water_adjacent_cache = None  # rebuild after terrain edit
        world.environment.update(world)

        soil = SoilDynamicsSystem()
        beside_water = world.get_tile(1, 0)
        far_away = world.get_tile(4, 4)
        beside_water.moisture = far_away.moisture = 0.5
        for _ in range(50):
            soil.update(world)
        assert beside_water.moisture > far_away.moisture

    def test_legacy_parity_when_disabled(self):
        """Disabled environment keeps the old net-positive moisture drip."""
        world = make_world(environment_config=None)
        soil = SoilDynamicsSystem(
            moisture_evaporation_rate=0.0002, moisture_recovery_rate=0.0008
        )
        tile = world.get_tile(2, 2)
        start = tile.moisture
        for _ in range(10):
            soil.update(world)
        # Legacy behaviour (bug B1, intentionally frozen): +0.0006/tick
        assert tile.moisture == pytest.approx(start + 10 * 0.0006)


# ===================================================================
# B2 fix: sand germination is possible again
# ===================================================================


class TestSandGermination:
    def test_sand_clamps_meet_builtin_seed_requirements(self):
        sand = ObjectRegistry.get("sand").tile_effect
        seed = ObjectRegistry.get("berry_seed").seed
        assert sand.fertility_override >= seed.required_fertility
        assert sand.moisture_override >= seed.required_moisture

    def test_seed_can_germinate_on_sand(self):
        world = make_world()
        tile = world.get_tile(2, 2)
        tile.terrain_type = TerrainType.SAND
        # Sand clamps applied by TileEffectSystem; emulate the clamped state
        tile.fertility = 0.30
        tile.moisture = 0.20

        sand = ObjectRegistry.create("sand", 2, 2)
        world.add_object(sand)

        from world.objects import PlantComponent, SeedComponent

        def add_ready_seed():
            obj = ObjectRegistry.create("berry_seed", 2, 2)
            seed = obj.get_component(SeedComponent)
            seed.time_in_soil = seed.grow_time  # ready, but not rotted
            world.add_object(obj)

        add_ready_seed()
        random.seed(0)
        system = SeedGerminationSystem(germination_success_rate=1.0)
        germinated = False
        for _ in range(200):  # effective rate is 0.1/attempt on sand
            system.update(world)
            if any(o.has_component(PlantComponent) for o in world.objects.values()):
                germinated = True
                break
            # Re-arm a fresh ready seed if the previous one failed
            if not any(o.has_component(SeedComponent) for o in world.objects.values()):
                add_ready_seed()
        assert germinated, "seed never germinated on sand (B2 regression)"


# ===================================================================
# World integration
# ===================================================================


class TestWorldIntegration:
    def test_world_advances_environment_clock(self):
        world = make_world(environment_config=enabled_config(day_length=100))
        for _ in range(25):
            world.update()
        assert world.environment.time_of_day == pytest.approx(0.25)
        assert world.environment.light == pytest.approx(1.0)

    def test_disabled_world_environment_stays_inert(self):
        world = make_world(environment_config=None)
        for _ in range(10):
            world.update()
        env = world.environment
        assert env.enabled is False
        assert env.light == 1.0 and env.time_of_day == 0.0
        assert env.growth_multiplier == 1.0

    def test_enabled_world_runs_many_ticks(self):
        """Smoke: full world update loop with the environment on."""
        world = make_world(
            environment_config={
                "enabled": True,
                "day_length": 50,
                "season_length": 200,
                "weather": {"rain_start_chance": 0.05, "rain_duration": 10},
            }
        )
        for _ in range(300):
            world.update()
        assert world.tick == 300
        # The clock kept moving and stayed in bounds the whole way
        assert 0.0 <= world.environment.temperature <= 1.0
        assert world.environment.min_light <= world.environment.light <= 1.0

    def test_temperature_follows_sin_formula(self):
        env = EnvironmentSystem(
            enabled_config(
                day_length=100,
                season_length=1000,
                base_temperature=0.5,
                season_temp_amplitude=0.2,
                daynight_temp_amplitude=0.1,
            )
        )
        tick = 333
        env.update(FakeWorld(tick=tick))
        day_curve = 0.5 * (1.0 + math.sin(2.0 * math.pi * (tick % 100) / 100))
        expected = (
            0.5
            + 0.2 * math.sin(2.0 * math.pi * (tick % 1000) / 1000)
            + 0.1 * (day_curve - 0.5)
        )
        assert env.temperature == pytest.approx(min(1.0, max(0.0, expected)))
