"""
Tests for W3 ecology & hazards: toxicity, contact damage, wildfire, species.

Covers:
- toxicity wired into EAT (net energy = calories − toxicity×freshness×scale)
- EAT records the eaten species in object_type
- reward shaper gives no survival bonus for a net-negative (poison) eat
- contact_damage: stepping onto thorns costs energy; harmless tiles don't
- FireSystem: ignition on hot/dry tiles, spread, self-extinguish at wet
  boundaries, nutrient return, and disabled = no-op
- the ecology.yaml species pack loads and validates

Author: Karan Vasa
"""

from types import SimpleNamespace

import pytest

import utils.agents.agent_utils as au
from utils.agents.agent_utils import TOXICITY_DAMAGE
from utils.agents.learning_utils import RewardShaper
from agents.actions import Action, ActionResult
from world.objects import WorldObject, EdibleComponent, PlantComponent
from world.object_registry import ObjectRegistry, register_builtin_objects
from world.systems import FireSystem
from world.tiles import TerrainType
from world.world import World


@pytest.fixture(autouse=True)
def _reset_registry():
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    ObjectRegistry._definitions.clear()


def _soil_world(n=8):
    w = World(
        n,
        n,
        seed=1,
        soil_ratio=1.0,
        rock_ratio=0.0,
        water_ratio=0.0,
        sand_ratio=0.0,
        parallel=False,
    )
    for y in range(n):
        for x in range(n):
            t = w.get_tile(x, y)
            t.terrain_type = TerrainType.SOIL
            t.fertility = 0.8
            t.moisture = 0.6
    return w


def _agent(world, x=0, y=0, energy=100.0):
    return SimpleNamespace(
        x=x,
        y=y,
        energy=energy,
        max_energy=200.0,
        fitness=0.0,
        inventory=[],
        direction=(1, 0),
    )


def _give(world, agent, calories, toxicity=0.0, freshness=1.0, type_id="berry"):
    obj = WorldObject(agent.x, agent.y)
    obj.type_id = type_id
    obj.add_component(EdibleComponent(calories, toxicity, freshness))
    world.objects[obj.id] = obj
    agent.inventory.append(obj.id)
    return obj


# ===================================================================
# Toxicity in EAT
# ===================================================================


class TestToxicEating:
    def test_nontoxic_food_gives_full_energy(self):
        w = _soil_world()
        a = _agent(w, energy=50.0)
        _give(w, a, calories=20.0, toxicity=0.0)
        res = au.execute_eat(a, w)
        assert res.success
        assert a.energy == pytest.approx(70.0)

    def test_toxic_food_is_net_negative(self):
        w = _soil_world()
        a = _agent(w, energy=100.0)
        # calories 12, toxicity 0.7 → 12 − 0.7×30 = −9
        _give(w, a, calories=12.0, toxicity=0.7, type_id="nightshade")
        res = au.execute_eat(a, w)
        assert res.success
        assert a.energy == pytest.approx(100.0 - 9.0)

    def test_freshness_scales_both_terms(self):
        w = _soil_world()
        a = _agent(w, energy=100.0)
        _give(w, a, calories=20.0, toxicity=0.5, freshness=0.5)
        # (20 − 0.5×30) × 0.5 = (20 − 15)×0.5 = 2.5
        res = au.execute_eat(a, w)
        assert res.success
        assert a.energy == pytest.approx(102.5)

    def test_eat_records_species(self):
        w = _soil_world()
        a = _agent(w)
        _give(w, a, calories=10.0, type_id="shrub_berry")
        res = au.execute_eat(a, w)
        assert res.object_type == "shrub_berry"
        assert res.interaction_kind == "eat"

    def test_strong_poison_can_drive_energy_negative(self):
        w = _soil_world()
        a = _agent(w, energy=10.0)
        _give(w, a, calories=5.0, toxicity=1.0)  # 5 − 30 = −25
        au.execute_eat(a, w)
        assert a.energy < 0  # fatal next tick


# ===================================================================
# Reward shaping respects the energy outcome (not food identity)
# ===================================================================


class TestPoisonReward:
    def _reward(self, action_result, energy_before, energy_after):
        # A fresh shaper per call so cross-call state doesn't bias the result.
        shaper = RewardShaper()
        agent = SimpleNamespace(
            x=0,
            y=0,
            energy=energy_after,
            max_energy=200.0,
            inventory=[],
            direction=(1, 0),
            alive=True,
            id=1,
        )
        return shaper.calculate_reward(
            action=Action.EAT,
            action_result=action_result,
            energy_before=energy_before,
            energy_after=energy_after,
            agent=agent,
            world=None,
        )

    def test_good_eat_rewarded_more_than_poison_eat(self):
        ok = ActionResult(True, 0.1, "ate", object_type="berry", interaction_kind="eat")
        good = self._reward(ok, energy_before=50.0, energy_after=70.0)
        poison = self._reward(ok, energy_before=50.0, energy_after=41.0)
        # The net-positive eat earns the big survival bonus; the poison eat
        # (net energy loss) does not — so it scores strictly lower.
        assert good > poison


# ===================================================================
# Contact damage (thorns)
# ===================================================================


class TestContactDamage:
    def test_thorns_cost_energy_to_enter(self):
        w = _soil_world()
        thorns = ObjectRegistry.create("thorns", 1, 0)
        w.add_object(thorns)
        a = _agent(w, x=0, y=0)
        a.direction = (1, 0)
        res = au.execute_move_forward(a, w)
        assert res.success
        # base move 0.20 + thorns contact_damage 8.0
        assert res.energy_cost == pytest.approx(0.20 + 8.0)

    def test_plain_tile_has_no_contact_damage(self):
        w = _soil_world()
        a = _agent(w, x=0, y=0)
        a.direction = (1, 0)
        res = au.execute_move_forward(a, w)
        assert res.energy_cost == pytest.approx(0.20)


# ===================================================================
# Wildfire
# ===================================================================


def _add_plant(world, x, y):
    p = WorldObject(x, y)
    p.type_id = "berry_plant"
    p.add_component(PlantComponent(mature_age=100, max_age=500, spawn_rate=0.1))
    world.add_object(p)
    return p


class TestFireSystem:
    def test_disabled_is_no_op(self):
        w = _soil_world()
        for x in range(3):
            _add_plant(w, x, 0)
            w.get_tile(x, 0).moisture = 0.05
        fire = FireSystem({"enabled": False})
        fire.burning[next(iter(w.objects))] = 5  # even if seeded, disabled skips
        before = len(w.objects)
        for _ in range(20):
            fire.update(w)
        assert len(w.objects) == before

    def test_fire_spreads_along_dry_plants(self):
        w = _soil_world()
        plants = [_add_plant(w, x, 0) for x in range(5)]
        for x in range(5):
            w.get_tile(x, 0).moisture = 0.05  # bone dry
        fire = FireSystem(
            {
                "enabled": True,
                "spread_chance": 1.0,
                "ignite_chance": 0.0,
                "burn_duration": 50,
                "moisture_threshold": 0.4,
            }
        )
        # Ignite one end, let it spread
        fire.burning[plants[0].id] = 50
        for _ in range(10):
            fire.update(w)
        # Spread should have ignited the whole dry row
        assert all(p.id in fire.burning for p in plants)

    def test_fire_self_extinguishes_at_wet_boundary(self):
        w = _soil_world()
        plants = [_add_plant(w, x, 0) for x in range(5)]
        for x in range(5):
            w.get_tile(x, 0).moisture = 0.05
        # A damp firebreak at x=2 (above the moisture threshold)
        w.get_tile(2, 0).moisture = 0.9
        fire = FireSystem(
            {
                "enabled": True,
                "spread_chance": 1.0,
                "ignite_chance": 0.0,
                "burn_duration": 50,
                "moisture_threshold": 0.4,
            }
        )
        fire.burning[plants[0].id] = 50
        for _ in range(20):
            fire.update(w)
        # Left of the wet break burns; the wet tile and beyond do not catch
        assert plants[0].id in fire.burning
        assert plants[1].id in fire.burning
        assert plants[2].id not in fire.burning  # the wet firebreak
        assert plants[3].id not in fire.burning  # protected beyond it
        assert plants[4].id not in fire.burning

    def test_burned_plant_returns_nutrients(self):
        w = _soil_world()
        p = _add_plant(w, 0, 0)
        tile = w.get_tile(0, 0)
        tile.moisture = 0.05
        tile.fertility = 0.2
        fire = FireSystem(
            {
                "enabled": True,
                "spread_chance": 0.0,
                "ignite_chance": 0.0,
                "burn_duration": 1,
                "moisture_threshold": 0.4,
                "nutrient_return": 0.25,
            }
        )
        fire.burning[p.id] = 1
        fire.update(w)
        assert p.id not in w.objects  # consumed by fire
        assert tile.fertility == pytest.approx(0.45)  # 0.2 + 0.25 ash
        assert fire.total_burned == 1

    def test_ignition_only_on_dry_tiles(self):
        w = _soil_world()
        dry = _add_plant(w, 0, 0)
        wet = _add_plant(w, 5, 5)
        w.get_tile(0, 0).moisture = 0.05
        w.get_tile(5, 5).moisture = 0.95
        fire = FireSystem(
            {
                "enabled": True,
                "ignite_chance": 1.0,
                "spread_chance": 0.0,
                "burn_duration": 50,
                "moisture_threshold": 0.4,
            }
        )
        fire.update(w)
        assert dry.id in fire.burning
        assert wet.id not in fire.burning


# ===================================================================
# Ecology species pack
# ===================================================================


class TestEcologyPack:
    def test_ecology_yaml_loads_and_validates(self):
        import yaml

        with open("config/ecology.yaml", "r", encoding="utf-8") as f:
            objects = yaml.safe_load(f)["objects"]
        count = ObjectRegistry.load_from_config(objects)
        assert count == 7
        # Distinct species with distinct encodings
        ns = ObjectRegistry.get("nightshade")
        assert ns.edible.toxicity > 0
        net = ns.edible.calories - ns.edible.toxicity * TOXICITY_DAMAGE
        assert net < 0  # the poison is genuinely net-negative
        tree = ObjectRegistry.get("tree_fruit")
        shrub = ObjectRegistry.get("shrub_berry")
        assert tree.edible.calories > shrub.edible.calories  # rich vs cheap
        # encodings all distinct
        encs = {
            ObjectRegistry.get(t).observation.vision_encoding
            for t in ("shrub_berry", "tree_fruit", "nightshade")
        }
        assert len(encs) == 3
