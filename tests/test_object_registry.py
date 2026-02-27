"""
Comprehensive tests for the Unified Object Definition & Registry system.

Tests cover:
- ObjectDefinition creation and serialisation
- ObjectRegistry CRUD (register, get, create, clear)
- Factory: create() builds correct components with defaults and overrides
- Built-in definitions (berry, berry_seed, berry_plant, fertilizer)
- Lifecycle chain via definitions (berry ↔ seed ↔ plant)
- Category lookup and backward compatibility
- Observation encoding via registry
- Physics spec lookup
- Config (YAML dict) loading
- Edge cases and error handling

Author: Karan Vasa
"""

import pytest
from world.objects import (
    WorldObject,
    EdibleComponent,
    SeedComponent,
    PlantComponent,
    FertilizerComponent,
    ToolComponent,
)
from world.object_registry import (
    EdibleSpec,
    SeedSpec,
    PlantSpec,
    FertilizerSpec,
    ToolSpec,
    PhysicsSpec,
    ObservationSpec,
    ObjectDefinition,
    ObjectRegistry,
    register_builtin_objects,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure every test starts with a clean registry and resets WorldObject IDs."""
    ObjectRegistry.clear()
    WorldObject._next_id = 0
    yield
    ObjectRegistry.clear()


@pytest.fixture
def builtins():
    """Register built-in definitions and return the registry for convenience."""
    register_builtin_objects()
    return ObjectRegistry


# ---------------------------------------------------------------------------
# Spec dataclass tests
# ---------------------------------------------------------------------------


class TestSpecs:
    """Tests for individual component specification dataclasses."""

    def test_edible_spec_defaults(self):
        spec = EdibleSpec()
        assert spec.calories == 20.0
        assert spec.toxicity == 0.0
        assert spec.freshness == 1.0

    def test_edible_spec_custom(self):
        spec = EdibleSpec(calories=50.0, toxicity=0.3, freshness=0.8)
        assert spec.calories == 50.0
        assert spec.toxicity == 0.3
        assert spec.freshness == 0.8

    def test_seed_spec_defaults(self):
        spec = SeedSpec()
        assert spec.grows_into == ""
        assert spec.grow_time == 50
        assert spec.max_age == 200

    def test_plant_spec_defaults(self):
        spec = PlantSpec()
        assert spec.mature_age == 100
        assert spec.max_age == 500
        assert spec.produces == ""
        assert spec.spawn_rate == 0.1

    def test_fertilizer_spec_defaults(self):
        spec = FertilizerSpec()
        assert spec.fertility_boost == 0.2
        assert spec.duration == 100
        assert spec.radius == 2

    def test_tool_spec_defaults(self):
        spec = ToolSpec()
        assert spec.effect_type == ""
        assert spec.efficiency == 1.0

    def test_physics_spec_defaults(self):
        spec = PhysicsSpec()
        assert spec.decay_rate == 0.0
        assert spec.decompose_into == ""
        assert spec.decompose_chance == 0.0
        assert spec.nutrient_return == 0.0

    def test_observation_spec_defaults(self):
        spec = ObservationSpec()
        assert spec.vision_encoding == 0.5
        assert spec.value_source == "none"


# ---------------------------------------------------------------------------
# ObjectDefinition tests
# ---------------------------------------------------------------------------


class TestObjectDefinition:
    """Tests for ObjectDefinition creation, from_dict, and to_dict."""

    def test_basic_definition(self):
        defn = ObjectDefinition(
            type_id="test_food",
            display_name="Test Food",
            category="food",
            edible=EdibleSpec(calories=30.0),
        )
        assert defn.type_id == "test_food"
        assert defn.display_name == "Test Food"
        assert defn.category == "food"
        assert defn.edible is not None
        assert defn.edible.calories == 30.0
        assert defn.seed is None
        assert defn.plant is None

    def test_multi_component_definition(self):
        """An object can theoretically have multiple components."""
        defn = ObjectDefinition(
            type_id="magic_fruit",
            display_name="Magic Fruit",
            category="food",
            edible=EdibleSpec(calories=100.0),
            seed=SeedSpec(grows_into="magic_plant"),
        )
        assert defn.edible is not None
        assert defn.seed is not None

    def test_from_dict_minimal(self):
        data = {
            "display_name": "Rock Candy",
            "category": "food",
            "edible": {"calories": 10.0},
        }
        defn = ObjectDefinition.from_dict("rock_candy", data)
        assert defn.type_id == "rock_candy"
        assert defn.display_name == "Rock Candy"
        assert defn.category == "food"
        assert defn.edible.calories == 10.0
        assert defn.seed is None
        assert defn.physics.decay_rate == 0.0  # default

    def test_from_dict_full(self):
        data = {
            "display_name": "Super Berry",
            "category": "food",
            "edible": {"calories": 50.0, "toxicity": 0.1, "freshness": 0.9},
            "physics": {
                "decay_rate": 0.02,
                "decompose_into": "super_seed",
                "decompose_chance": 0.5,
                "nutrient_return": 0.2,
            },
            "observation": {
                "vision_encoding": 0.95,
                "value_source": "freshness",
            },
        }
        defn = ObjectDefinition.from_dict("super_berry", data)
        assert defn.edible.toxicity == 0.1
        assert defn.physics.decay_rate == 0.02
        assert defn.physics.decompose_into == "super_seed"
        assert defn.observation.vision_encoding == 0.95

    def test_from_dict_seed_type(self):
        data = {
            "display_name": "Oak Acorn",
            "category": "seed",
            "seed": {
                "grows_into": "oak_tree",
                "grow_time": 100,
                "required_fertility": 0.5,
                "required_moisture": 0.4,
                "max_age": 300,
            },
        }
        defn = ObjectDefinition.from_dict("oak_acorn", data)
        assert defn.seed.grows_into == "oak_tree"
        assert defn.seed.grow_time == 100
        assert defn.seed.max_age == 300

    def test_to_dict_roundtrip(self):
        original = ObjectDefinition(
            type_id="berry",
            display_name="Berry",
            category="food",
            edible=EdibleSpec(calories=20.0),
            physics=PhysicsSpec(
                decay_rate=0.01, decompose_into="berry_seed", decompose_chance=0.7
            ),
            observation=ObservationSpec(vision_encoding=1.0, value_source="freshness"),
        )
        d = original.to_dict()
        restored = ObjectDefinition.from_dict("berry", d)
        assert restored.type_id == original.type_id
        assert restored.edible.calories == original.edible.calories
        assert restored.physics.decompose_into == original.physics.decompose_into
        assert (
            restored.observation.vision_encoding == original.observation.vision_encoding
        )

    def test_from_dict_defaults_when_missing(self):
        """from_dict should use defaults when category/display_name is missing."""
        data = {}
        defn = ObjectDefinition.from_dict("bare", data)
        assert defn.display_name == "bare"
        assert defn.category == "object"
        assert defn.edible is None


# ---------------------------------------------------------------------------
# ObjectRegistry CRUD tests
# ---------------------------------------------------------------------------


class TestRegistryCRUD:
    """Tests for registry register / get / clear / type_ids."""

    def test_register_and_get(self):
        defn = ObjectDefinition(type_id="foo", display_name="Foo", category="food")
        ObjectRegistry.register(defn)
        assert ObjectRegistry.get("foo") is defn

    def test_get_unknown_returns_none(self):
        assert ObjectRegistry.get("nonexistent") is None

    def test_clear(self):
        ObjectRegistry.register(
            ObjectDefinition(type_id="x", display_name="X", category="food")
        )
        assert len(ObjectRegistry.all_definitions()) == 1
        ObjectRegistry.clear()
        assert len(ObjectRegistry.all_definitions()) == 0

    def test_type_ids(self):
        ObjectRegistry.register(
            ObjectDefinition(type_id="a", display_name="A", category="food")
        )
        ObjectRegistry.register(
            ObjectDefinition(type_id="b", display_name="B", category="seed")
        )
        ids = ObjectRegistry.type_ids()
        assert set(ids) == {"a", "b"}

    def test_replace_definition(self):
        defn1 = ObjectDefinition(type_id="x", display_name="X1", category="food")
        defn2 = ObjectDefinition(type_id="x", display_name="X2", category="seed")
        ObjectRegistry.register(defn1)
        ObjectRegistry.register(defn2)
        assert ObjectRegistry.get("x").display_name == "X2"

    def test_all_definitions_returns_copy(self):
        ObjectRegistry.register(
            ObjectDefinition(type_id="t", display_name="T", category="food")
        )
        defs = ObjectRegistry.all_definitions()
        defs.pop("t")
        # Original should still have it
        assert ObjectRegistry.get("t") is not None


# ---------------------------------------------------------------------------
# Factory (create) tests
# ---------------------------------------------------------------------------


class TestRegistryCreate:
    """Tests for ObjectRegistry.create() factory method."""

    def test_create_berry(self, builtins):
        berry = builtins.create("berry", 5, 10)
        assert berry.x == 5
        assert berry.y == 10
        assert berry.type_id == "berry"
        assert berry.has_component(EdibleComponent)
        edible = berry.get_component(EdibleComponent)
        assert edible.calories == 20.0
        assert edible.freshness == 1.0

    def test_create_berry_seed(self, builtins):
        seed = builtins.create("berry_seed", 3, 7)
        assert seed.type_id == "berry_seed"
        assert seed.has_component(SeedComponent)
        sc = seed.get_component(SeedComponent)
        assert sc.plant_type == "berry_plant"
        assert sc.grow_time == 50
        assert sc.max_age == 200

    def test_create_berry_plant(self, builtins):
        plant = builtins.create("berry_plant", 2, 4)
        assert plant.type_id == "berry_plant"
        assert plant.has_component(PlantComponent)
        pc = plant.get_component(PlantComponent)
        assert pc.mature_age == 100
        assert pc.max_age == 500
        assert pc.spawn_resource_type == "berry"

    def test_create_fertilizer(self, builtins):
        fert = builtins.create("fertilizer", 0, 0)
        assert fert.type_id == "fertilizer"
        assert fert.has_component(FertilizerComponent)
        fc = fert.get_component(FertilizerComponent)
        assert fc.fertility_boost == 0.2
        assert fc.duration == 100
        assert fc.radius == 2

    def test_create_with_overrides(self, builtins):
        big_berry = builtins.create("berry", 1, 1, calories=50.0, freshness=0.5)
        edible = big_berry.get_component(EdibleComponent)
        assert edible.calories == 50.0
        assert edible.freshness == 0.5

    def test_create_seed_with_overrides(self, builtins):
        seed = builtins.create("berry_seed", 0, 0, grow_time=100, seed_max_age=500)
        sc = seed.get_component(SeedComponent)
        assert sc.grow_time == 100
        assert sc.max_age == 500

    def test_create_plant_with_overrides(self, builtins):
        plant = builtins.create("berry_plant", 0, 0, mature_age=50, plant_max_age=1000)
        pc = plant.get_component(PlantComponent)
        assert pc.mature_age == 50
        assert pc.max_age == 1000

    def test_create_unknown_type_raises(self):
        with pytest.raises(KeyError, match="Unknown object type"):
            ObjectRegistry.create("unicorn", 0, 0)

    def test_create_assigns_unique_ids(self, builtins):
        a = builtins.create("berry", 0, 0)
        b = builtins.create("berry", 1, 1)
        assert a.id != b.id

    def test_create_sets_type_id_on_object(self, builtins):
        obj = builtins.create("berry_plant", 0, 0)
        assert hasattr(obj, "type_id")
        assert obj.type_id == "berry_plant"


# ---------------------------------------------------------------------------
# Built-in definitions tests
# ---------------------------------------------------------------------------


class TestBuiltinDefinitions:
    """Tests for register_builtin_objects() and the 4 built-in types."""

    def test_register_builtin_objects_populates_registry(self, builtins):
        defs = builtins.all_definitions()
        assert "berry" in defs
        assert "berry_seed" in defs
        assert "berry_plant" in defs
        assert "fertilizer" in defs

    def test_berry_definition_properties(self, builtins):
        d = builtins.get("berry")
        assert d.category == "food"
        assert d.edible is not None
        assert d.edible.calories == 20.0
        assert d.physics.decay_rate == 0.01
        assert d.physics.decompose_into == "berry_seed"
        assert d.physics.decompose_chance == 0.7
        assert d.physics.nutrient_return == 0.15
        assert d.observation.vision_encoding == 1.0
        assert d.observation.value_source == "freshness"

    def test_berry_seed_definition_properties(self, builtins):
        d = builtins.get("berry_seed")
        assert d.category == "seed"
        assert d.seed is not None
        assert d.seed.grows_into == "berry_plant"
        assert d.seed.grow_time == 50
        assert d.observation.vision_encoding == 0.6
        assert d.observation.value_source == "viability"

    def test_berry_plant_definition_properties(self, builtins):
        d = builtins.get("berry_plant")
        assert d.category == "plant"
        assert d.plant is not None
        assert d.plant.produces == "berry"
        assert d.plant.spawn_rate == 0.1
        assert d.physics.nutrient_return == 0.15
        assert d.observation.vision_encoding == 0.75
        assert d.observation.value_source == "maturity"

    def test_fertilizer_definition_properties(self, builtins):
        d = builtins.get("fertilizer")
        assert d.category == "fertilizer"
        assert d.fertilizer is not None
        assert d.observation.vision_encoding == 0.4
        assert d.observation.value_source == "duration"

    def test_lifecycle_chain_references(self, builtins):
        """Verify the berry lifecycle chain is internally consistent."""
        berry = builtins.get("berry")
        seed = builtins.get("berry_seed")
        plant = builtins.get("berry_plant")

        # berry decomposes into seed
        assert berry.physics.decompose_into == seed.type_id
        # seed grows into plant
        assert seed.seed.grows_into == plant.type_id
        # plant produces berry
        assert plant.plant.produces == berry.type_id


# ---------------------------------------------------------------------------
# Category lookup tests
# ---------------------------------------------------------------------------


class TestCategoryLookup:
    """Tests for ObjectRegistry.get_category()."""

    def test_category_via_type_id(self, builtins):
        berry = builtins.create("berry", 0, 0)
        assert builtins.get_category(berry) == "food"

    def test_category_via_type_id_seed(self, builtins):
        seed = builtins.create("berry_seed", 0, 0)
        assert builtins.get_category(seed) == "seed"

    def test_category_via_type_id_plant(self, builtins):
        plant = builtins.create("berry_plant", 0, 0)
        assert builtins.get_category(plant) == "plant"

    def test_category_via_type_id_fertilizer(self, builtins):
        fert = builtins.create("fertilizer", 0, 0)
        assert builtins.get_category(fert) == "fertilizer"

    def test_category_fallback_no_type_id(self, builtins):
        """Objects created without registry should still get correct category."""
        obj = WorldObject(0, 0)
        obj.add_component(EdibleComponent(calories=10.0))
        assert builtins.get_category(obj) == "food"

    def test_category_fallback_seed_component(self):
        obj = WorldObject(0, 0)
        obj.add_component(SeedComponent(plant_type="test"))
        assert ObjectRegistry.get_category(obj) == "seed"

    def test_category_fallback_plant_component(self):
        obj = WorldObject(0, 0)
        obj.add_component(PlantComponent())
        assert ObjectRegistry.get_category(obj) == "plant"

    def test_category_fallback_fertilizer_component(self):
        obj = WorldObject(0, 0)
        obj.add_component(FertilizerComponent())
        assert ObjectRegistry.get_category(obj) == "fertilizer"

    def test_category_fallback_tool_component(self):
        obj = WorldObject(0, 0)
        obj.add_component(ToolComponent(effect_type="DIG"))
        assert ObjectRegistry.get_category(obj) == "tool"

    def test_category_fallback_empty_object(self):
        obj = WorldObject(0, 0)
        assert ObjectRegistry.get_category(obj) == "object"


# ---------------------------------------------------------------------------
# Observation encoding tests
# ---------------------------------------------------------------------------


class TestObservationEncoding:
    """Tests for ObjectRegistry.get_observation_encoding()."""

    def test_observation_encoding_berry(self, builtins):
        berry = builtins.create("berry", 0, 0)
        assert builtins.get_observation_encoding(berry) == 1.0

    def test_observation_encoding_seed(self, builtins):
        seed = builtins.create("berry_seed", 0, 0)
        assert builtins.get_observation_encoding(seed) == 0.6

    def test_observation_encoding_plant(self, builtins):
        plant = builtins.create("berry_plant", 0, 0)
        assert builtins.get_observation_encoding(plant) == 0.75

    def test_observation_encoding_fertilizer(self, builtins):
        fert = builtins.create("fertilizer", 0, 0)
        assert builtins.get_observation_encoding(fert) == 0.4

    def test_observation_encoding_no_type_id(self):
        obj = WorldObject(0, 0)
        assert ObjectRegistry.get_observation_encoding(obj) is None


# ---------------------------------------------------------------------------
# Physics spec tests
# ---------------------------------------------------------------------------


class TestPhysicsSpec:
    """Tests for ObjectRegistry.get_physics()."""

    def test_physics_berry(self, builtins):
        berry = builtins.create("berry", 0, 0)
        physics = builtins.get_physics(berry)
        assert physics is not None
        assert physics.decay_rate == 0.01
        assert physics.decompose_into == "berry_seed"
        assert physics.decompose_chance == 0.7
        assert physics.nutrient_return == 0.15

    def test_physics_plant(self, builtins):
        plant = builtins.create("berry_plant", 0, 0)
        physics = builtins.get_physics(plant)
        assert physics.nutrient_return == 0.15

    def test_physics_no_type_id(self):
        obj = WorldObject(0, 0)
        assert ObjectRegistry.get_physics(obj) is None


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Tests for ObjectRegistry.load_from_config()."""

    def test_load_single_type(self):
        config = {
            "mushroom": {
                "display_name": "Mushroom",
                "category": "food",
                "edible": {"calories": 15.0, "toxicity": 0.1, "freshness": 1.0},
                "physics": {"decay_rate": 0.02},
                "observation": {"vision_encoding": 0.85, "value_source": "freshness"},
            }
        }
        count = ObjectRegistry.load_from_config(config)
        assert count == 1
        defn = ObjectRegistry.get("mushroom")
        assert defn is not None
        assert defn.edible.calories == 15.0
        assert defn.edible.toxicity == 0.1
        assert defn.physics.decay_rate == 0.02

    def test_load_multiple_types(self):
        config = {
            "apple": {
                "display_name": "Apple",
                "category": "food",
                "edible": {"calories": 30.0},
            },
            "apple_seed": {
                "display_name": "Apple Seed",
                "category": "seed",
                "seed": {"grows_into": "apple_tree", "grow_time": 80},
            },
            "apple_tree": {
                "display_name": "Apple Tree",
                "category": "plant",
                "plant": {
                    "mature_age": 200,
                    "max_age": 1000,
                    "produces": "apple",
                    "spawn_rate": 0.05,
                },
            },
        }
        count = ObjectRegistry.load_from_config(config)
        assert count == 3
        assert ObjectRegistry.get("apple").edible.calories == 30.0
        assert ObjectRegistry.get("apple_seed").seed.grows_into == "apple_tree"
        assert ObjectRegistry.get("apple_tree").plant.produces == "apple"

    def test_load_and_create(self):
        config = {
            "golden_berry": {
                "display_name": "Golden Berry",
                "category": "food",
                "edible": {"calories": 100.0, "freshness": 0.9},
            }
        }
        ObjectRegistry.load_from_config(config)
        obj = ObjectRegistry.create("golden_berry", 5, 5)
        assert obj.type_id == "golden_berry"
        edible = obj.get_component(EdibleComponent)
        assert edible.calories == 100.0
        assert edible.freshness == 0.9

    def test_load_overwrites_builtins(self, builtins):
        """Config definitions should override built-in definitions."""
        config = {
            "berry": {
                "display_name": "Ultra Berry",
                "category": "food",
                "edible": {"calories": 999.0},
            }
        }
        ObjectRegistry.load_from_config(config)
        defn = ObjectRegistry.get("berry")
        assert defn.display_name == "Ultra Berry"
        assert defn.edible.calories == 999.0

    def test_load_empty_config(self):
        count = ObjectRegistry.load_from_config({})
        assert count == 0


# ---------------------------------------------------------------------------
# Tool component via registry tests
# ---------------------------------------------------------------------------


class TestToolDefinition:
    """Tests for objects with ToolComponent via the registry."""

    def test_create_tool_object(self):
        ObjectRegistry.register(
            ObjectDefinition(
                type_id="pickaxe",
                display_name="Pickaxe",
                category="tool",
                tool=ToolSpec(effect_type="DIG", efficiency=2.0),
            )
        )
        obj = ObjectRegistry.create("pickaxe", 0, 0)
        assert obj.type_id == "pickaxe"
        assert obj.has_component(ToolComponent)
        tc = obj.get_component(ToolComponent)
        assert tc.effect_type == "DIG"
        assert tc.efficiency == 2.0

    def test_tool_override(self):
        ObjectRegistry.register(
            ObjectDefinition(
                type_id="shovel",
                display_name="Shovel",
                category="tool",
                tool=ToolSpec(effect_type="DIG", efficiency=1.0),
            )
        )
        obj = ObjectRegistry.create("shovel", 0, 0, efficiency=3.0)
        tc = obj.get_component(ToolComponent)
        assert tc.efficiency == 3.0


# ---------------------------------------------------------------------------
# Custom object type integration tests
# ---------------------------------------------------------------------------


class TestCustomObjectTypes:
    """Tests demonstrating how easy it is to add new object types."""

    def test_define_mushroom_lifecycle(self):
        """Define a complete mushroom lifecycle: mushroom → spore → mycelium → mushroom."""
        ObjectRegistry.register(
            ObjectDefinition(
                type_id="mushroom",
                display_name="Mushroom",
                category="food",
                edible=EdibleSpec(calories=15.0, toxicity=0.05),
                physics=PhysicsSpec(
                    decay_rate=0.03,
                    decompose_into="spore",
                    decompose_chance=0.8,
                    nutrient_return=0.1,
                ),
                observation=ObservationSpec(
                    vision_encoding=0.9, value_source="freshness"
                ),
            )
        )
        ObjectRegistry.register(
            ObjectDefinition(
                type_id="spore",
                display_name="Spore",
                category="seed",
                seed=SeedSpec(
                    grows_into="mycelium",
                    grow_time=30,
                    required_fertility=0.2,
                    required_moisture=0.3,
                    max_age=150,
                ),
                observation=ObservationSpec(
                    vision_encoding=0.55, value_source="viability"
                ),
            )
        )
        ObjectRegistry.register(
            ObjectDefinition(
                type_id="mycelium",
                display_name="Mycelium",
                category="plant",
                plant=PlantSpec(
                    mature_age=60,
                    max_age=300,
                    produces="mushroom",
                    spawn_rate=0.15,
                ),
                physics=PhysicsSpec(nutrient_return=0.1),
                observation=ObservationSpec(
                    vision_encoding=0.7, value_source="maturity"
                ),
            )
        )

        # Verify lifecycle chain
        mushroom_def = ObjectRegistry.get("mushroom")
        spore_def = ObjectRegistry.get("spore")
        mycelium_def = ObjectRegistry.get("mycelium")

        assert mushroom_def.physics.decompose_into == "spore"
        assert spore_def.seed.grows_into == "mycelium"
        assert mycelium_def.plant.produces == "mushroom"

        # Create each type and verify
        m = ObjectRegistry.create("mushroom", 5, 5)
        assert m.has_component(EdibleComponent)
        assert m.get_component(EdibleComponent).calories == 15.0

        s = ObjectRegistry.create("spore", 5, 5)
        assert s.has_component(SeedComponent)
        assert s.get_component(SeedComponent).plant_type == "mycelium"

        p = ObjectRegistry.create("mycelium", 5, 5)
        assert p.has_component(PlantComponent)
        assert p.get_component(PlantComponent).spawn_resource_type == "mushroom"

    def test_define_healing_herb(self):
        """Define a simple non-lifecycle object: healing herb."""
        ObjectRegistry.register(
            ObjectDefinition(
                type_id="healing_herb",
                display_name="Healing Herb",
                category="food",
                edible=EdibleSpec(
                    calories=5.0, toxicity=-0.5
                ),  # negative toxicity = healing
                physics=PhysicsSpec(decay_rate=0.005),
                observation=ObservationSpec(
                    vision_encoding=0.88, value_source="freshness"
                ),
            )
        )

        herb = ObjectRegistry.create("healing_herb", 10, 20)
        assert herb.type_id == "healing_herb"
        edible = herb.get_component(EdibleComponent)
        assert edible.calories == 5.0
        assert edible.toxicity == -0.5

    def test_define_object_with_multiple_components(self):
        """A fruit that is both edible AND contains a seed (combo object)."""
        ObjectRegistry.register(
            ObjectDefinition(
                type_id="seeded_fruit",
                display_name="Seeded Fruit",
                category="food",
                edible=EdibleSpec(calories=25.0),
                seed=SeedSpec(grows_into="fruit_tree", grow_time=60),
            )
        )

        obj = ObjectRegistry.create("seeded_fruit", 0, 0)
        assert obj.has_component(EdibleComponent)
        assert obj.has_component(SeedComponent)
        assert obj.get_component(SeedComponent).plant_type == "fruit_tree"


# ---------------------------------------------------------------------------
# WorldObject.type_id backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """
    Tests that objects created the 'old' way (direct WorldObject + add_component)
    still work correctly with the registry helpers.
    """

    def test_old_style_object_has_empty_type_id(self):
        obj = WorldObject(0, 0)
        obj.add_component(EdibleComponent(calories=10.0))
        assert obj.type_id == ""

    def test_get_category_old_style(self, builtins):
        obj = WorldObject(0, 0)
        obj.add_component(SeedComponent(plant_type="test"))
        assert builtins.get_category(obj) == "seed"

    def test_get_physics_old_style_returns_none(self, builtins):
        obj = WorldObject(0, 0)
        obj.add_component(EdibleComponent(calories=10.0))
        assert builtins.get_physics(obj) is None

    def test_get_observation_encoding_old_style_returns_none(self, builtins):
        obj = WorldObject(0, 0)
        obj.add_component(PlantComponent())
        assert builtins.get_observation_encoding(obj) is None

    def test_world_object_repr_with_type_id(self, builtins):
        obj = builtins.create("berry", 5, 10)
        repr_str = repr(obj)
        assert "type='berry'" in repr_str
        assert "pos=(5, 10)" in repr_str

    def test_world_object_repr_without_type_id(self):
        obj = WorldObject(3, 7)
        repr_str = repr(obj)
        assert "type=" not in repr_str
        assert "pos=(3, 7)" in repr_str


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_register_builtin_twice_is_safe(self):
        register_builtin_objects()
        register_builtin_objects()
        assert (
            len(ObjectRegistry.all_definitions()) == 5
        )  # berry, berry_seed, berry_plant, fertilizer, sand

    def test_create_at_zero_zero(self, builtins):
        obj = builtins.create("berry", 0, 0)
        assert obj.x == 0 and obj.y == 0

    def test_create_at_large_coords(self, builtins):
        obj = builtins.create("berry", 9999, 9999)
        assert obj.x == 9999 and obj.y == 9999

    def test_definition_with_no_components(self):
        """An object with no components should still be creatable."""
        ObjectRegistry.register(
            ObjectDefinition(
                type_id="marker",
                display_name="Marker",
                category="object",
            )
        )
        obj = ObjectRegistry.create("marker", 0, 0)
        assert obj.type_id == "marker"
        assert len(obj.components) == 0

    def test_overrides_ignored_for_absent_components(self, builtins):
        """Overrides for components not in the definition should be harmless."""
        # berry has no PlantComponent, so mature_age override is ignored
        berry = builtins.create("berry", 0, 0, mature_age=999)
        assert not berry.has_component(PlantComponent)
        # Still has correct edible
        assert berry.get_component(EdibleComponent).calories == 20.0
