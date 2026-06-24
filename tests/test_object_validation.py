"""
Tests for object-definition validation, inheritance, and respawn (W0).

Covers the failure modes from docs/WORLD_UPGRADE_PROPOSAL.md §8:
- O1: unknown sections error with "did you mean" (no more silent typos)
- O2: unknown fields error with type_id + section context
- O3: dangling cross-references rejected at load time
- O4: spawn.respawn_rate regenerates depleted custom types
- O5: vision-encoding collisions warn; "auto" allocates in-category
- O7: extends inheritance (deep-merge, spawn never inherited)
Plus: the shipped example file validates, and legacy valid configs load.
"""

import warnings

import numpy as np
import pytest
import yaml

from world.object_registry import ObjectRegistry, register_builtin_objects
from world.object_validation import (
    ObjectValidationError,
    deep_merge,
)


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Each test starts from a clean builtin registry."""
    ObjectRegistry.clear()
    register_builtin_objects()
    yield
    ObjectRegistry.clear()
    register_builtin_objects()


class TestSchemaValidation:
    def test_unknown_section_errors_with_suggestion(self):
        with pytest.raises(ObjectValidationError) as exc:
            ObjectRegistry.load_from_config({"my_food": {"edibel": {"calories": 99}}})
        msg = str(exc.value)
        assert "my_food" in msg and "edibel" in msg
        assert "did you mean 'edible'" in msg

    def test_unknown_field_errors_with_context(self):
        with pytest.raises(ObjectValidationError) as exc:
            ObjectRegistry.load_from_config({"my_food": {"edible": {"callories": 99}}})
        msg = str(exc.value)
        assert "'my_food.edible'" in msg
        assert "did you mean 'calories'" in msg

    def test_all_errors_collected_in_one_report(self):
        with pytest.raises(ObjectValidationError) as exc:
            ObjectRegistry.load_from_config(
                {
                    "a": {"edibel": {}},
                    "b": {"seed": {"growtime": 1}},
                }
            )
        assert len(exc.value.errors) >= 2

    def test_nothing_registers_when_batch_invalid(self):
        with pytest.raises(ObjectValidationError):
            ObjectRegistry.load_from_config(
                {
                    "good": {"edible": {"calories": 5}},
                    "bad": {"edibel": {}},
                }
            )
        assert ObjectRegistry.get("good") is None

    def test_bad_vision_encoding_rejected(self):
        with pytest.raises(ObjectValidationError) as exc:
            ObjectRegistry.load_from_config(
                {"x": {"observation": {"vision_encoding": 5.0}}}
            )
        assert "vision_encoding" in str(exc.value)

    def test_valid_definition_loads(self):
        n = ObjectRegistry.load_from_config(
            {
                "snack": {
                    "edible": {"calories": 5.0},
                    "observation": {"vision_encoding": 0.9},
                }
            }
        )
        assert n == 1
        assert ObjectRegistry.get("snack").edible.calories == 5.0


class TestCrossReferences:
    def test_dangling_grows_into_rejected(self):
        with pytest.raises(ObjectValidationError) as exc:
            ObjectRegistry.load_from_config({"s": {"seed": {"grows_into": "NOPE"}}})
        assert "references unknown type 'NOPE'" in str(exc.value)

    def test_reference_to_builtin_ok(self):
        ObjectRegistry.load_from_config(
            {
                "s": {
                    "seed": {"grows_into": "berry_plant"},
                    "observation": {"vision_encoding": 0.52},
                }
            }
        )
        assert ObjectRegistry.get("s") is not None

    def test_reference_within_same_file_ok(self):
        ObjectRegistry.load_from_config(
            {
                "my_seed": {
                    "seed": {"grows_into": "my_plant"},
                    "observation": {"vision_encoding": 0.52},
                },
                "my_plant": {
                    "plant": {"produces": "berry"},
                    "observation": {"vision_encoding": 0.68},
                },
            }
        )
        assert ObjectRegistry.get("my_plant") is not None


class TestExtends:
    def test_inherits_parent_and_overrides(self):
        ObjectRegistry.load_from_config(
            {
                "apple": {
                    "extends": "berry",
                    "edible": {"calories": 60.0},
                    "observation": {"vision_encoding": 0.95},
                }
            }
        )
        apple = ObjectRegistry.get("apple")
        assert apple.edible.calories == 60.0  # overridden
        assert apple.edible.freshness == 1.0  # inherited from berry
        assert apple.physics.decay_rate == 0.01  # inherited deep section
        assert apple.interaction.pickable  # inherited

    def test_spawn_is_never_inherited(self):
        # berry has no spawn in registry, give the parent one explicitly
        ObjectRegistry.load_from_config(
            {
                "parent_food": {
                    "edible": {"calories": 5},
                    "observation": {"vision_encoding": 0.9},
                    "spawn": {"initial_count": 50},
                },
                "child_food": {
                    "extends": "parent_food",
                    "observation": {"vision_encoding": 0.93},
                },
            }
        )
        assert ObjectRegistry.get("child_food").spawn.initial_count == 0

    def test_extends_unknown_parent_errors(self):
        with pytest.raises(ObjectValidationError) as exc:
            ObjectRegistry.load_from_config({"x": {"extends": "ghost"}})
        assert "extends unknown type 'ghost'" in str(exc.value)

    def test_deep_merge_is_keywise(self):
        merged = deep_merge(
            {"a": {"x": 1, "y": 2}, "b": 1},
            {"a": {"y": 3}},
        )
        assert merged == {"a": {"x": 1, "y": 3}, "b": 1}


class TestEncodings:
    def test_collision_warns(self):
        with pytest.warns(UserWarning, match="vision_encoding collision"):
            ObjectRegistry.load_from_config(
                {
                    "fake_berry": {
                        "edible": {"calories": 1},
                        "observation": {"vision_encoding": 1.0},
                    }
                }
            )  # berry builtin is also 1.0

    def test_auto_allocates_in_category_band(self):
        ObjectRegistry.load_from_config(
            {
                "auto_food": {
                    "category": "food",
                    "edible": {"calories": 1},
                    "observation": {"vision_encoding": "auto"},
                }
            }
        )
        enc = ObjectRegistry.get("auto_food").observation.vision_encoding
        assert 0.85 <= enc <= 1.0
        assert abs(enc - 1.0) >= 0.02  # avoided the berry builtin

    def test_two_autos_get_distinct_values(self):
        ObjectRegistry.load_from_config(
            {
                "f1": {
                    "category": "food",
                    "edible": {"calories": 1},
                    "observation": {"vision_encoding": "auto"},
                },
                "f2": {
                    "category": "food",
                    "edible": {"calories": 2},
                    "observation": {"vision_encoding": "auto"},
                },
            }
        )
        e1 = ObjectRegistry.get("f1").observation.vision_encoding
        e2 = ObjectRegistry.get("f2").observation.vision_encoding
        assert abs(e1 - e2) >= 0.02


class TestRespawn:
    def test_respawn_replenishes_depleted_type(self):
        from world.world import World

        ObjectRegistry.load_from_config(
            {
                "manna": {
                    "edible": {"calories": 5},
                    "observation": {"vision_encoding": 0.9},
                    "spawn": {
                        "initial_count": 0,
                        "terrain": "any",
                        "respawn_rate": 1.0,
                        "max_count": 3,
                    },
                }
            }
        )
        world = World(width=20, height=20, seed=3, parallel=False)
        np.random.seed(0)
        for _ in range(30):
            world.systems.update(world)
        manna = [
            o for o in world.objects.values() if getattr(o, "type_id", "") == "manna"
        ]
        assert len(manna) == 3  # respawned up to cap, not beyond

    def test_builtins_unchanged_no_respawn_by_default(self):
        assert ObjectRegistry.get("berry").spawn.respawn_rate == 0.0


class TestShippedExamples:
    def test_custom_objects_yaml_is_valid(self):
        with open("config/custom_objects.yaml") as f:
            objects = yaml.safe_load(f)["objects"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # collisions would fail this
            n = ObjectRegistry.load_from_config(objects)
        assert n == 5
        # extends actually worked: golden_apple inherits berry behaviour
        apple = ObjectRegistry.get("golden_apple")
        assert apple.edible.calories == 60.0
        assert apple.interaction.pickable
        # chain closes
        assert ObjectRegistry.get("acorn").seed.grows_into == "oak_tree"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
