"""
Unified Object Definition & Registry system.

Provides a declarative way to define world object types with all their
properties (components, physics, observation encoding, reward hooks)
in one place. Objects are created via the registry factory instead of
manually constructing WorldObject + components.

Usage:
    # Register built-in types (called once at startup)
    register_builtin_objects()

    # Create an object
    berry = ObjectRegistry.create("berry", x=5, y=10)

    # Create with overrides
    big_berry = ObjectRegistry.create("berry", x=5, y=10, calories=50.0)

    # Look up a definition
    defn = ObjectRegistry.get("berry")
    print(defn.display_name)  # "Berry"

    # Add custom types from YAML config
    ObjectRegistry.load_from_config(config_dict)

Author: Karan Vasa
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

from world.objects import (
    WorldObject,
    EdibleComponent,
    SeedComponent,
    PlantComponent,
    FertilizerComponent,
    ToolComponent,
)

# ---------------------------------------------------------------------------
# Component specification dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EdibleSpec:
    """
    Specification for an edible component.

    Attributes:
        calories: Energy provided when consumed.
        toxicity: Toxicity level (0.0 = safe).
        freshness: Initial freshness (0.0-1.0).
    """

    calories: float = 20.0
    toxicity: float = 0.0
    freshness: float = 1.0


@dataclass
class SeedSpec:
    """
    Specification for a seed component.

    Attributes:
        grows_into: type_id of the plant this seed grows into.
        grow_time: Ticks required for germination.
        required_fertility: Minimum soil fertility to germinate.
        required_moisture: Minimum soil moisture to germinate.
        max_age: Maximum ticks before seed rots.
    """

    grows_into: str = ""
    grow_time: int = 50
    required_fertility: float = 0.3
    required_moisture: float = 0.2
    max_age: int = 200


@dataclass
class PlantSpec:
    """
    Specification for a plant component.

    Attributes:
        mature_age: Age (ticks) when plant can produce resources.
        max_age: Maximum lifespan in ticks.
        produces: type_id of the resource produced when mature.
        spawn_rate: Per-tick probability of producing a resource when mature.
    """

    mature_age: int = 100
    max_age: int = 500
    produces: str = ""
    spawn_rate: float = 0.1


@dataclass
class FertilizerSpec:
    """
    Specification for a fertilizer component.

    Attributes:
        fertility_boost: Amount to increase tile fertility.
        duration: Remaining ticks before effect expires.
        radius: Tiles within this Manhattan distance are affected.
    """

    fertility_boost: float = 0.2
    duration: int = 100
    radius: int = 2


@dataclass
class ToolSpec:
    """
    Specification for a tool component (placeholder for future).

    Attributes:
        effect_type: Type of tool effect (e.g. "DIG", "HARVEST_BOOST").
        efficiency: Multiplier for effect strength.
    """

    effect_type: str = ""
    efficiency: float = 1.0


# ---------------------------------------------------------------------------
# Cross-cutting specs (physics, observation, interaction, tile-effects)
# ---------------------------------------------------------------------------


@dataclass
class PhysicsSpec:
    """
    World-physics properties attached to an object definition.

    Attributes:
        decay_rate: Freshness loss per tick (0 = no decay, only for edibles).
        decompose_into: type_id of object spawned on full decomposition (empty = none).
        decompose_chance: Probability of spawning decompose_into on decomposition.
        nutrient_return: Fertility returned to soil tile on death / decomposition.
    """

    decay_rate: float = 0.0
    decompose_into: str = ""
    decompose_chance: float = 0.0
    nutrient_return: float = 0.0


@dataclass
class InteractionSpec:
    """
    Defines how agents can interact with this object type.

    Attributes:
        pickable: Whether agents can pick this object up.
        usable: Whether agents can USE this object (e.g. plant a seed).
        passable: Whether agents can walk through the tile this occupies.
        blocks_growth: If True, seeds cannot germinate on the same tile.
    """

    pickable: bool = True
    usable: bool = False
    passable: bool = True
    blocks_growth: bool = False


@dataclass
class TileEffectSpec:
    """
    Specifies environmental effects this object exerts on nearby tiles.

    Used for terrain-like objects (e.g. sand) that modify growth rates,
    germination chances, and can spread to adjacent tiles.

    Attributes:
        germination_multiplier: Factor applied to germination success rate
            on the same tile (1.0 = normal, 0.1 = 10x harder).
        growth_multiplier: Factor applied to plant maturation speed
            (1.0 = normal, 0.1 = 10x slower).
        spawn_rate_multiplier: Factor applied to mature plant resource
            spawn rate (1.0 = normal, 0.5 = half production).
        spread_type_id: type_id of object to place on converted tiles
            (empty = no spreading).  Typically the same type_id.
        spread_radius: Maximum Manhattan distance for spreading.
        spread_interval: Ticks a neighbouring soil tile must be without
            a blocking object before it is converted.
        spread_blocked_by: List of object categories that prevent
            spreading (e.g. ["plant"] means trees block sand spread).
        spread_chance: Per-tick probability of attempting to spread to
            each eligible neighbour once the interval is met.
        converts_terrain: TerrainType value string the tile is converted
            to when spread happens ("sand", "rock", etc.).  Empty = no
            terrain conversion (only places the object).
        fertility_override: If >= 0, tile fertility is clamped to this
            value when the effect object is present.
        moisture_override: If >= 0, tile moisture is clamped to this
            value when the effect object is present.
        reclaim_terrain: TerrainType value string the tile is converted
            back to when a blocking object sits on it long enough.
            Empty = no reclamation (default).  e.g. "soil" means a
            plant growing on sand can turn it back to soil.
        reclaim_interval: Ticks a blocking object must be present on
            the same tile before reclamation triggers (0 = disabled).
    """

    germination_multiplier: float = 1.0
    growth_multiplier: float = 1.0
    spawn_rate_multiplier: float = 1.0
    spread_type_id: str = ""
    spread_radius: int = 1
    spread_interval: int = 200
    spread_blocked_by: List[str] = field(default_factory=list)
    spread_chance: float = 0.05
    converts_terrain: str = ""
    fertility_override: float = -1.0
    moisture_override: float = -1.0
    reclaim_terrain: str = ""
    reclaim_interval: int = 0


@dataclass
class RenderSpec:
    """
    Visual appearance of this object in the GUI and console.

    Attributes:
        char: Single character for the console/ASCII renderer.
        color: RGB colour tuple ``[R, G, B]`` for the pygame renderer.
            Each channel is 0-255.
    """

    char: str = "?"
    color: List[int] = field(default_factory=lambda: [200, 200, 200])


@dataclass
class ObservationSpec:
    """
    How this object appears in the agent observation vector.

    Attributes:
        vision_encoding: Tile-type encoding value (0.0-1.0 scale).
        value_source: What drives the "value" feature:
            "freshness"  -> edible.freshness * edible.calories
            "maturity"   -> plant.age / plant.mature_age
            "viability"  -> 1 - seed.time_in_soil / seed.max_age
            "duration"   -> fertilizer.duration / fertilizer.max_duration
            "none"       -> 0.0
    """

    vision_encoding: float = 0.5
    value_source: str = "none"


@dataclass
class SpawnSpec:
    """
    Controls how many of this object to place at world initialisation.

    Attributes:
        initial_count: Number of instances to scatter on the map at start.
        terrain: Where to place them.  Options:
            "soil"     - only on SOIL tiles (default)
            "sand"     - only on SAND tiles
            "any"      - any walkable tile (SOIL, SAND, WATER)
            "plantable"- tiles that can support plants (is_plantable)
        respawn_rate: Per-tick probability of respawning one instance
            while fewer than the cap exist (0 = never respawn; fixes
            "my custom food vanished forever" — proposal issue O4).
        max_count: Population cap for respawning (0 = use initial_count).
    """

    initial_count: int = 0
    terrain: str = "soil"
    respawn_rate: float = 0.0
    max_count: int = 0


# ---------------------------------------------------------------------------
# Unified object definition
# ---------------------------------------------------------------------------


@dataclass
class ObjectDefinition:
    """
    Complete, declarative definition of a world object type.

    Groups *all* information needed to create, simulate, observe, and
    interact with this kind of object in one place:

    * **Component specs** – which ECS components the object carries and
      their default parameter values.
    * **Physics** – per-type decay rate, decomposition chain, nutrient return.
    * **Observation** – how the object is encoded in the agent vision vector.
    """

    # Identity
    type_id: str  # Unique machine key, e.g. "berry"
    display_name: str  # Human-readable name, e.g. "Berry"
    category: str  # Semantic tag: "food" | "seed" | "plant" | "fertilizer" | "tool"

    # Component specifications (None = component not present)
    edible: Optional[EdibleSpec] = None
    seed: Optional[SeedSpec] = None
    plant: Optional[PlantSpec] = None
    fertilizer: Optional[FertilizerSpec] = None
    tool: Optional[ToolSpec] = None

    # Cross-cutting properties
    physics: PhysicsSpec = field(default_factory=PhysicsSpec)
    interaction: InteractionSpec = field(default_factory=InteractionSpec)
    tile_effect: Optional[TileEffectSpec] = None
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    render: RenderSpec = field(default_factory=RenderSpec)
    spawn: SpawnSpec = field(default_factory=SpawnSpec)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, type_id: str, data: dict) -> "ObjectDefinition":
        """
        Construct an ObjectDefinition from a plain dictionary (e.g. parsed
        from YAML ``objects:`` section).

        Args:
            type_id: Unique identifier for the object type.
            data: Dictionary of definition fields.

        Returns:
            A new ObjectDefinition instance.
        """
        edible = EdibleSpec(**data["edible"]) if "edible" in data else None
        seed = SeedSpec(**data["seed"]) if "seed" in data else None
        plant = PlantSpec(**data["plant"]) if "plant" in data else None
        fertilizer = (
            FertilizerSpec(**data["fertilizer"]) if "fertilizer" in data else None
        )
        tool = ToolSpec(**data["tool"]) if "tool" in data else None

        physics = PhysicsSpec(**data["physics"]) if "physics" in data else PhysicsSpec()
        interaction = (
            InteractionSpec(**data["interaction"])
            if "interaction" in data
            else InteractionSpec()
        )

        tile_effect = None
        if "tile_effect" in data:
            te_data = dict(data["tile_effect"])
            # spread_blocked_by may come from YAML as a list already
            tile_effect = TileEffectSpec(**te_data)

        observation = (
            ObservationSpec(**data["observation"])
            if "observation" in data
            else ObservationSpec()
        )

        render = RenderSpec(**data["render"]) if "render" in data else RenderSpec()

        spawn = SpawnSpec(**data["spawn"]) if "spawn" in data else SpawnSpec()

        return cls(
            type_id=type_id,
            display_name=data.get("display_name", type_id),
            category=data.get("category", "object"),
            edible=edible,
            seed=seed,
            plant=plant,
            fertilizer=fertilizer,
            tool=tool,
            physics=physics,
            interaction=interaction,
            tile_effect=tile_effect,
            observation=observation,
            render=render,
            spawn=spawn,
        )

    def to_dict(self) -> dict:
        """
        Serialise the definition to a plain dictionary (for YAML export).

        Returns:
            Dictionary representation of this definition.
        """
        d: Dict[str, Any] = {
            "display_name": self.display_name,
            "category": self.category,
        }

        if self.edible is not None:
            d["edible"] = {
                "calories": self.edible.calories,
                "toxicity": self.edible.toxicity,
                "freshness": self.edible.freshness,
            }
        if self.seed is not None:
            d["seed"] = {
                "grows_into": self.seed.grows_into,
                "grow_time": self.seed.grow_time,
                "required_fertility": self.seed.required_fertility,
                "required_moisture": self.seed.required_moisture,
                "max_age": self.seed.max_age,
            }
        if self.plant is not None:
            d["plant"] = {
                "mature_age": self.plant.mature_age,
                "max_age": self.plant.max_age,
                "produces": self.plant.produces,
                "spawn_rate": self.plant.spawn_rate,
            }
        if self.fertilizer is not None:
            d["fertilizer"] = {
                "fertility_boost": self.fertilizer.fertility_boost,
                "duration": self.fertilizer.duration,
                "radius": self.fertilizer.radius,
            }
        if self.tool is not None:
            d["tool"] = {
                "effect_type": self.tool.effect_type,
                "efficiency": self.tool.efficiency,
            }

        d["physics"] = {
            "decay_rate": self.physics.decay_rate,
            "decompose_into": self.physics.decompose_into,
            "decompose_chance": self.physics.decompose_chance,
            "nutrient_return": self.physics.nutrient_return,
        }
        d["interaction"] = {
            "pickable": self.interaction.pickable,
            "usable": self.interaction.usable,
            "passable": self.interaction.passable,
            "blocks_growth": self.interaction.blocks_growth,
        }
        if self.tile_effect is not None:
            d["tile_effect"] = {
                "germination_multiplier": self.tile_effect.germination_multiplier,
                "growth_multiplier": self.tile_effect.growth_multiplier,
                "spawn_rate_multiplier": self.tile_effect.spawn_rate_multiplier,
                "spread_type_id": self.tile_effect.spread_type_id,
                "spread_radius": self.tile_effect.spread_radius,
                "spread_interval": self.tile_effect.spread_interval,
                "spread_blocked_by": self.tile_effect.spread_blocked_by,
                "spread_chance": self.tile_effect.spread_chance,
                "converts_terrain": self.tile_effect.converts_terrain,
                "fertility_override": self.tile_effect.fertility_override,
                "moisture_override": self.tile_effect.moisture_override,
                "reclaim_terrain": self.tile_effect.reclaim_terrain,
                "reclaim_interval": self.tile_effect.reclaim_interval,
            }
        d["observation"] = {
            "vision_encoding": self.observation.vision_encoding,
            "value_source": self.observation.value_source,
        }
        d["render"] = {
            "char": self.render.char,
            "color": list(self.render.color),
        }

        if (
            self.spawn.initial_count > 0
            or self.spawn.respawn_rate > 0
            or self.spawn.max_count > 0
        ):
            d["spawn"] = {
                "initial_count": self.spawn.initial_count,
                "terrain": self.spawn.terrain,
                "respawn_rate": self.spawn.respawn_rate,
                "max_count": self.spawn.max_count,
            }

        return d


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ObjectRegistry:
    """
    Singleton registry mapping *type_id* → :class:`ObjectDefinition`.

    Provides:
    * ``register()`` / ``get()`` – definition CRUD.
    * ``create()`` – factory to build a fully-configured :class:`WorldObject`.
    * ``get_category()`` – fast category lookup (used by reward / observation).
    * ``load_from_config()`` – bulk-load definitions from YAML dict.
    """

    _definitions: Dict[str, ObjectDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, definition: ObjectDefinition) -> None:
        """
        Register (or replace) an object definition.

        Args:
            definition: The ObjectDefinition to register.
        """
        cls._definitions[definition.type_id] = definition

    @classmethod
    def get(cls, type_id: str) -> Optional[ObjectDefinition]:
        """
        Look up a definition by type_id.

        Args:
            type_id: Unique object type identifier.

        Returns:
            ObjectDefinition if found, None otherwise.
        """
        return cls._definitions.get(type_id)

    @classmethod
    def all_definitions(cls) -> Dict[str, ObjectDefinition]:
        """Return a shallow copy of all registered definitions."""
        return dict(cls._definitions)

    @classmethod
    def clear(cls) -> None:
        """Remove all registered definitions (useful for testing)."""
        cls._definitions.clear()

    @classmethod
    def type_ids(cls) -> List[str]:
        """Return a list of all registered type_ids."""
        return list(cls._definitions.keys())

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, type_id: str, x: int, y: int, **overrides) -> WorldObject:
        """
        Create a :class:`WorldObject` from a registered definition.

        The resulting object has:
        * ``type_id`` set on the WorldObject for fast reverse lookup.
        * All components specified in the definition, with default values
          that can be individually overridden via ``**overrides``.

        Override keys match component field names.  For example::

            ObjectRegistry.create("berry", 5, 10, calories=50.0, freshness=0.8)

        Args:
            type_id: Registered object type identifier.
            x: World X-coordinate.
            y: World Y-coordinate.
            **overrides: Optional per-field overrides for component parameters.

        Returns:
            A new WorldObject fully configured with components.

        Raises:
            KeyError: If *type_id* is not registered.
        """
        defn = cls._definitions.get(type_id)
        if defn is None:
            raise KeyError(
                f"Unknown object type: {type_id!r}. "
                f"Registered types: {list(cls._definitions.keys())}"
            )

        obj = WorldObject(x, y)
        obj.type_id = type_id
        obj.is_terrain = defn.tile_effect is not None

        # --- Edible ---
        if defn.edible is not None:
            obj.add_component(
                EdibleComponent(
                    calories=overrides.get("calories", defn.edible.calories),
                    toxicity=overrides.get("toxicity", defn.edible.toxicity),
                    freshness=overrides.get("freshness", defn.edible.freshness),
                )
            )

        # --- Seed ---
        if defn.seed is not None:
            obj.add_component(
                SeedComponent(
                    plant_type=overrides.get("grows_into", defn.seed.grows_into),
                    grow_time=overrides.get("grow_time", defn.seed.grow_time),
                    required_fertility=overrides.get(
                        "required_fertility", defn.seed.required_fertility
                    ),
                    required_moisture=overrides.get(
                        "required_moisture", defn.seed.required_moisture
                    ),
                    max_age=overrides.get("seed_max_age", defn.seed.max_age),
                )
            )

        # --- Plant ---
        if defn.plant is not None:
            obj.add_component(
                PlantComponent(
                    mature_age=overrides.get("mature_age", defn.plant.mature_age),
                    max_age=overrides.get("plant_max_age", defn.plant.max_age),
                    spawn_resource_type=overrides.get("produces", defn.plant.produces),
                    spawn_rate=overrides.get("spawn_rate", defn.plant.spawn_rate),
                )
            )

        # --- Fertilizer ---
        if defn.fertilizer is not None:
            obj.add_component(
                FertilizerComponent(
                    fertility_boost=overrides.get(
                        "fertility_boost", defn.fertilizer.fertility_boost
                    ),
                    duration=overrides.get("duration", defn.fertilizer.duration),
                    radius=overrides.get("radius", defn.fertilizer.radius),
                )
            )

        # --- Tool ---
        if defn.tool is not None:
            obj.add_component(
                ToolComponent(
                    effect_type=overrides.get("effect_type", defn.tool.effect_type),
                    efficiency=overrides.get("efficiency", defn.tool.efficiency),
                )
            )

        # --- Cached flags for hot-path checks ---
        obj.is_terrain = defn.tile_effect is not None

        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_category(cls, obj: WorldObject) -> str:
        """
        Return the semantic category for a world object.

        Uses ``obj.type_id`` if set, otherwise falls back to component-based
        inference for backward compatibility.

        Args:
            obj: WorldObject to categorise.

        Returns:
            Category string: "food", "seed", "plant", "fertilizer", "tool",
            or "object" as fallback.
        """
        tid = getattr(obj, "type_id", "")
        if tid:
            defn = cls._definitions.get(tid)
            if defn is not None:
                return defn.category

        # Fallback: infer from components
        if obj.has_component(EdibleComponent):
            return "food"
        if obj.has_component(SeedComponent):
            return "seed"
        if obj.has_component(FertilizerComponent):
            return "fertilizer"
        if obj.has_component(PlantComponent):
            return "plant"
        if obj.has_component(ToolComponent):
            return "tool"
        return "object"

    @classmethod
    def get_observation_encoding(cls, obj: WorldObject) -> Optional[float]:
        """
        Return the observation vision_encoding for a world object.

        Args:
            obj: WorldObject to look up.

        Returns:
            Float encoding value if definition found, None otherwise.
        """
        tid = getattr(obj, "type_id", "")
        if tid:
            defn = cls._definitions.get(tid)
            if defn is not None:
                return defn.observation.vision_encoding
        return None

    @classmethod
    def get_physics(cls, obj: WorldObject) -> Optional[PhysicsSpec]:
        """
        Return the PhysicsSpec for a world object.

        Args:
            obj: WorldObject to look up.

        Returns:
            PhysicsSpec if definition found, None otherwise.
        """
        tid = getattr(obj, "type_id", "")
        if tid:
            defn = cls._definitions.get(tid)
            if defn is not None:
                return defn.physics
        return None

    @classmethod
    def is_pickable(cls, obj: WorldObject) -> bool:
        """
        Check whether an object can be picked up by an agent.

        Uses the registry ``InteractionSpec.pickable`` when the object
        has a ``type_id``; falls back to ``True`` for legacy objects
        *unless* they carry a PlantComponent (plants are not pickable
        by default).

        Args:
            obj: WorldObject to check.

        Returns:
            True if object can be picked up, False otherwise.
        """
        tid = getattr(obj, "type_id", "")
        if tid:
            defn = cls._definitions.get(tid)
            if defn is not None:
                return defn.interaction.pickable

        # Legacy fallback: plants are NOT pickable
        if obj.has_component(PlantComponent):
            return False
        return True

    @classmethod
    def is_terrain_layer(cls, obj) -> bool:
        """
        Return True if *obj* is a terrain-layer object (has a TileEffectSpec).

        Terrain-layer objects (sand, oasis, lava …) are transparent to the
        stacking check – other objects can be placed on the same tile.

        Args:
            obj: WorldObject to check (may be None).

        Returns:
            True if the object is a terrain-layer, False otherwise.
        """
        if obj is None:
            return False
        return getattr(obj, "is_terrain", False)

    @classmethod
    def get_tile_effect(cls, obj: WorldObject) -> Optional[TileEffectSpec]:
        """
        Return the TileEffectSpec for a world object, or None.

        Args:
            obj: WorldObject to look up.

        Returns:
            TileEffectSpec if the definition has one, else None.
        """
        tid = getattr(obj, "type_id", "")
        if tid:
            defn = cls._definitions.get(tid)
            if defn is not None:
                return defn.tile_effect
        return None

    @classmethod
    def get_interaction(cls, obj: WorldObject) -> InteractionSpec:
        """
        Return the InteractionSpec for a world object.

        Falls back to a default InteractionSpec (everything True,
        except plants are not pickable) for legacy objects.

        Args:
            obj: WorldObject to look up.

        Returns:
            InteractionSpec.
        """
        tid = getattr(obj, "type_id", "")
        if tid:
            defn = cls._definitions.get(tid)
            if defn is not None:
                return defn.interaction

        # Legacy fallback
        if obj.has_component(PlantComponent):
            return InteractionSpec(
                pickable=False, usable=False, passable=True, blocks_growth=False
            )
        return InteractionSpec()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @classmethod
    def load_from_config(cls, objects_config: dict, strict: bool = True) -> int:
        """
        Bulk-register object definitions from a YAML ``objects:`` section.

        Each key is a *type_id* and the value is a dict matching
        :meth:`ObjectDefinition.from_dict` format. Supports
        ``extends: <type_id>`` inheritance (from builtins or earlier
        entries) and ``vision_encoding: auto`` band allocation.

        The whole batch is validated BEFORE anything registers: unknown
        sections/fields (with "did you mean" suggestions), dangling
        cross-references (grows_into/produces/decompose_into/
        spread_type_id), and bad encodings are collected and raised
        together as :class:`ObjectValidationError`. Encoding collisions
        across the full registry are emitted as warnings.

        Args:
            objects_config: Dictionary of type_id → definition dict.
            strict: Raise on validation errors (False = legacy behaviour:
                print the errors and register nothing invalid-free).

        Returns:
            Number of definitions loaded.

        Raises:
            ObjectValidationError: If strict and any definition is invalid.
        """
        # Imported lazily — object_validation imports this module's specs
        from world.object_validation import (
            ObjectValidationError,
            allocate_auto_encodings,
            resolve_definitions,
            validate_cross_references,
            validate_definition_dict,
            warn_encoding_collisions,
        )

        def _registered_as_dict(type_id: str):
            defn = cls.get(type_id)
            return defn.to_dict() if defn is not None else None

        resolved, errors = resolve_definitions(objects_config, _registered_as_dict)

        for type_id, data in resolved.items():
            errors.extend(validate_definition_dict(type_id, data))

        if not errors:
            taken = {
                tid: d.observation.vision_encoding
                for tid, d in cls._definitions.items()
            }
            errors.extend(allocate_auto_encodings(resolved, taken))

        known_ids = set(cls._definitions) | set(resolved)
        errors.extend(validate_cross_references(resolved, known_ids))

        if errors:
            exc = ObjectValidationError(errors)
            if strict:
                raise exc
            print(str(exc))
            return 0

        count = 0
        for type_id, data in resolved.items():
            cls.register(ObjectDefinition.from_dict(type_id, data))
            count += 1

        warn_encoding_collisions(
            {tid: d.observation.vision_encoding for tid, d in cls._definitions.items()}
        )
        return count


# ---------------------------------------------------------------------------
# Built-in object definitions
# ---------------------------------------------------------------------------


def register_builtin_objects() -> None:
    """
    Register all built-in object types.

    This should be called once during simulation startup (before any
    objects are created).  Safe to call multiple times – existing
    definitions are simply replaced.

    Built-in types:
    * ``berry`` – edible food item with freshness decay.
    * ``berry_seed`` – plantable seed that grows into a berry plant.
    * ``berry_plant`` – plant that produces berries when mature (NOT pickable).
    * ``fertilizer`` – boosts soil fertility in a radius.
    * ``sand`` – terrain hazard that slows growth, reduces production,
      and spreads to neighbouring soil tiles when no plants block it.
    """

    # ---- Berry (food) ----
    ObjectRegistry.register(
        ObjectDefinition(
            type_id="berry",
            display_name="Berry",
            category="food",
            edible=EdibleSpec(
                calories=20.0,
                toxicity=0.0,
                freshness=1.0,
            ),
            physics=PhysicsSpec(
                decay_rate=0.01,
                decompose_into="berry_seed",
                decompose_chance=0.7,
                nutrient_return=0.15,
            ),
            interaction=InteractionSpec(
                pickable=True,
                usable=False,
                passable=True,
            ),
            observation=ObservationSpec(
                vision_encoding=1.0,
                value_source="freshness",
            ),
            render=RenderSpec(
                char="o",
                color=[220, 20, 60],
            ),
        )
    )

    # ---- Berry Seed ----
    ObjectRegistry.register(
        ObjectDefinition(
            type_id="berry_seed",
            display_name="Berry Seed",
            category="seed",
            seed=SeedSpec(
                grows_into="berry_plant",
                grow_time=50,
                required_fertility=0.3,
                required_moisture=0.2,
                max_age=200,
            ),
            physics=PhysicsSpec(
                nutrient_return=0.0,
            ),
            interaction=InteractionSpec(
                pickable=True,
                usable=True,  # agents can plant seeds via USE action
                passable=True,
            ),
            observation=ObservationSpec(
                vision_encoding=0.6,
                value_source="viability",
            ),
            render=RenderSpec(
                char="s",
                color=[205, 170, 125],
            ),
        )
    )

    # ---- Berry Plant (NOT pickable) ----
    ObjectRegistry.register(
        ObjectDefinition(
            type_id="berry_plant",
            display_name="Berry Plant",
            category="plant",
            plant=PlantSpec(
                mature_age=100,
                max_age=500,
                produces="berry",
                spawn_rate=0.1,
            ),
            physics=PhysicsSpec(
                nutrient_return=0.15,
            ),
            interaction=InteractionSpec(
                pickable=False,  # plants cannot be picked up
                usable=False,
                passable=True,
                blocks_growth=False,
            ),
            observation=ObservationSpec(
                vision_encoding=0.75,
                value_source="maturity",
            ),
            render=RenderSpec(
                char="P",
                color=[34, 139, 34],
            ),
        )
    )

    # ---- Fertilizer ----
    ObjectRegistry.register(
        ObjectDefinition(
            type_id="fertilizer",
            display_name="Fertilizer",
            category="fertilizer",
            fertilizer=FertilizerSpec(
                fertility_boost=0.2,
                duration=100,
                radius=2,
            ),
            physics=PhysicsSpec(),
            interaction=InteractionSpec(
                pickable=True,
                usable=True,  # agents can apply fertilizer via USE action
                passable=True,
            ),
            observation=ObservationSpec(
                vision_encoding=0.4,
                value_source="duration",
            ),
            render=RenderSpec(
                char="f",
                color=[139, 90, 43],
            ),
        )
    )

    # ---- Sand (terrain hazard, not pickable) ----
    # Sand dramatically hinders plant life:
    # - Seed germination chance ×0.1 (10× harder)
    # - Plant growth rate ×0.1 (10× slower to mature)
    # - Resource production rate ×0.3 (much less food)
    # - Spreads to adjacent soil tiles if no plant is within radius
    #   for spread_interval ticks
    # - Kills fertility and moisture on its tile
    ObjectRegistry.register(
        ObjectDefinition(
            type_id="sand",
            display_name="Sand",
            category="terrain",
            interaction=InteractionSpec(
                pickable=False,
                usable=False,
                passable=True,  # agents can walk on sand
                blocks_growth=False,  # seeds CAN germinate (but multiplier is 0.1)
            ),
            tile_effect=TileEffectSpec(
                germination_multiplier=0.1,  # 10× harder to germinate
                growth_multiplier=0.1,  # 10× slower growth
                spawn_rate_multiplier=0.3,  # 70% less food production
                spread_type_id="sand",  # spreads by placing more sand
                spread_radius=1,  # adjacent tiles only
                spread_interval=200,  # 200 ticks without a plant → spread
                spread_blocked_by=["plant"],  # any plant category blocks spreading
                spread_chance=0.05,  # 5% per tick once interval met
                converts_terrain="sand",  # mark tile terrain as SAND
                # B2 fix: clamps sit AT the germination thresholds (0.3
                # fertility / 0.2 moisture) so the ×0.1 multiplier is what
                # makes sand "harder" — 0.05 made germination impossible
                fertility_override=0.30,
                moisture_override=0.20,
                reclaim_terrain="soil",  # plants can reclaim sand → soil
                reclaim_interval=150,  # 150 ticks with a plant → reclaim
            ),
            physics=PhysicsSpec(),
            observation=ObservationSpec(
                vision_encoding=0.15,  # distinct from rock(0.0), water(0.25)
                value_source="none",
            ),
            render=RenderSpec(
                char=":",
                color=[210, 180, 120],
            ),
        )
    )
