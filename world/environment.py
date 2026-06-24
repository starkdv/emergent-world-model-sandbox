"""
Environment engine — day/night, seasons, and weather (World upgrade W1).

The single source of climate (proposal §3.1): one system computes the
global environmental state each tick, and every existing system consumes
plain multipliers from it. Nothing here scripts behaviour — these are
*pressures* the agents must discover (guideline §8).

    light(t)        day/night sinusoid  → plant growth, food production
    season(t)       slow sinusoid       → temperature baseline
    temperature(t)  season + day/night  → growth window, decay, metabolism
    rain / drought  stochastic events   → the ONLY moisture recovery

This is also the fix for verified bug B1 (moisture never decreases):
with the environment enabled, the old unconditional +0.0008/tick
"simulated rain" is replaced by evaporation that scales with temperature
and light, and recovery that arrives only during rain events — moisture
becomes a real, time-varying constraint instead of a saturated constant.

With ``environment.enabled: false`` (the default) every multiplier is
exactly 1.0 and SoilDynamics keeps its legacy arithmetic, so the
pre-upgrade baseline remains bit-reproducible.

Author: Karan Vasa
Date: June 2026
"""

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world.world import World


def temperature_response(temperature: float) -> float:
    """
    Biological response curve to temperature (proposal §3.1 / SUGGESTIONS
    §6.1): full rate in the comfortable band, linear falloff outside it,
    zero at the extremes.

        1.0 for t in [0.3, 0.7]
        linear → 0.0 at t = 0.1 and t = 0.9

    Args:
        temperature: Normalised temperature in [0, 1]

    Returns:
        Multiplier in [0, 1]
    """
    if 0.3 <= temperature <= 0.7:
        return 1.0
    if temperature < 0.3:
        return max(0.0, (temperature - 0.1) / 0.2)
    return max(0.0, (0.9 - temperature) / 0.2)


class EnvironmentSystem:
    """
    Global climate state and the multipliers other systems consume.

    Attributes (read by systems/agents each tick):
        enabled: Master switch (False = all multipliers neutral)
        time_of_day: Phase of the day cycle in [0, 1) (0 = dawn)
        light: Current light level in [min_light, 1.0]
        season_phase: Phase of the seasonal cycle in [0, 1)
        temperature: Global normalised temperature in [0, 1]
        raining / drought: Active weather event flags
    """

    def __init__(self, config: dict = None):
        """
        Initialize from the ``environment:`` config section.

        Args:
            config: Config dict (None/missing keys → documented defaults)
        """
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", False))

        # Day/night
        self.day_length: int = int(cfg.get("day_length", 200))
        self.min_light: float = float(cfg.get("min_light", 0.25))

        # Seasons (one full cycle = season_length ticks)
        self.season_length: int = int(cfg.get("season_length", 2000))
        self.season_temp_amplitude: float = float(
            cfg.get("season_temp_amplitude", 0.25)
        )
        self.base_temperature: float = float(cfg.get("base_temperature", 0.5))
        self.daynight_temp_amplitude: float = float(
            cfg.get("daynight_temp_amplitude", 0.10)
        )

        # Weather events
        weather = cfg.get("weather", {}) or {}
        self.rain_start_chance: float = float(weather.get("rain_start_chance", 0.01))
        self.rain_duration: int = int(weather.get("rain_duration", 60))
        self.rain_recovery: float = float(weather.get("rain_recovery", 0.004))
        self.drought_start_chance: float = float(
            weather.get("drought_start_chance", 0.002)
        )
        self.drought_duration: int = int(weather.get("drought_duration", 150))
        self.drought_evaporation_factor: float = float(
            weather.get("drought_evaporation_factor", 2.0)
        )

        # Soil-water coupling (replaces the legacy constant recovery)
        self.base_evaporation: float = float(cfg.get("base_evaporation", 0.0012))
        self.water_adjacency_recovery: float = float(
            cfg.get("water_adjacency_recovery", 0.002)
        )

        # Agent metabolism response to temperature extremes
        self.metabolism_temp_coef: float = float(cfg.get("metabolism_temp_coef", 0.5))

        # Live state
        self.time_of_day: float = 0.0
        self.light: float = 1.0
        self.season_phase: float = 0.0
        self.temperature: float = self.base_temperature
        self.raining: bool = False
        self.drought: bool = False
        self._rain_ticks_left: int = 0
        self._drought_ticks_left: int = 0

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------

    def update(self, world: "World") -> None:
        """
        Advance the clock and weather events (runs FIRST each world tick).

        Args:
            world: The world (used for tick and event announcements)
        """
        if not self.enabled:
            return

        tick = world.tick

        # Day/night: light follows a raised sinusoid over day_length
        self.time_of_day = (tick % self.day_length) / self.day_length
        day_curve = 0.5 * (1.0 + math.sin(2.0 * math.pi * self.time_of_day))
        self.light = self.min_light + (1.0 - self.min_light) * day_curve

        # Season: slow sinusoid modulating the temperature baseline
        self.season_phase = (tick % self.season_length) / self.season_length
        season_offset = self.season_temp_amplitude * math.sin(
            2.0 * math.pi * self.season_phase
        )
        day_offset = self.daynight_temp_amplitude * (day_curve - 0.5)
        self.temperature = min(
            1.0, max(0.0, self.base_temperature + season_offset + day_offset)
        )

        # Weather events (drought suppresses rain)
        if self.raining:
            self._rain_ticks_left -= 1
            if self._rain_ticks_left <= 0:
                self.raining = False
                print(f"🌤  Tick {tick}: rain ended")
        if self.drought:
            self._drought_ticks_left -= 1
            if self._drought_ticks_left <= 0:
                self.drought = False
                print(f"🌤  Tick {tick}: drought ended")

        if not self.raining and not self.drought:
            roll = random.random()
            if roll < self.drought_start_chance:
                self.drought = True
                self._drought_ticks_left = self.drought_duration
                print(
                    f"☀️  Tick {tick}: DROUGHT began "
                    f"({self.drought_duration} ticks)"
                )
            elif roll < self.drought_start_chance + self.rain_start_chance:
                self.raining = True
                self._rain_ticks_left = self.rain_duration
                print(f"🌧  Tick {tick}: rain began ({self.rain_duration} ticks)")

    # ------------------------------------------------------------------
    # Multipliers consumed by the existing systems
    # (all exactly 1.0 / legacy when disabled)
    # ------------------------------------------------------------------

    @property
    def growth_multiplier(self) -> float:
        """Plant maturation rate: needs light AND a livable temperature."""
        if not self.enabled:
            return 1.0
        return self.light * temperature_response(self.temperature)

    @property
    def germination_multiplier(self) -> float:
        """Seed germination success: temperature window only."""
        if not self.enabled:
            return 1.0
        return temperature_response(self.temperature)

    @property
    def spawn_multiplier(self) -> float:
        """Mature-plant food production scales with light."""
        if not self.enabled:
            return 1.0
        return self.light

    @property
    def decay_multiplier(self) -> float:
        """Freshness decay: faster when hot, slower when cold."""
        if not self.enabled:
            return 1.0
        return 0.5 + self.temperature

    @property
    def metabolism_multiplier(self) -> float:
        """
        Agent energy drain: mild at the comfortable middle, up to
        (1 + coef) at the temperature extremes — cold AND heat both cost.
        """
        if not self.enabled:
            return 1.0
        return 1.0 + self.metabolism_temp_coef * 2.0 * abs(self.temperature - 0.5)

    @property
    def evaporation_rate(self) -> float:
        """
        Per-tick soil moisture loss (replaces the legacy constant).
        Scales with temperature and light; droughts multiply it further.
        """
        rate = self.base_evaporation * (0.25 + self.temperature + 0.5 * self.light)
        if self.drought:
            rate *= self.drought_evaporation_factor
        return rate

    @property
    def moisture_recovery_rate(self) -> float:
        """
        Per-tick soil moisture recovery — ONLY during rain (the B1 fix:
        the legacy unconditional +0.0008/tick drip is gone when enabled).
        """
        return self.rain_recovery if self.raining else 0.0
