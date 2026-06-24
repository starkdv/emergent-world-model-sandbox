"""
Tests for W5 — social dynamics & open-endedness instruments.

Covers:
- USE-as-trade: facing an agent + transfer_enabled → first inventory item
  hands over; recipient-full case rejects gracefully; gate off → legacy
  seed/fertilizer behaviour.
- Action mask gives USE when only a trade target is available.
- Analyzer society metrics: role-entropy, behavioural novelty (JS),
  territory (bbox / overlap), and the "give" counts.

The analyzer function is loaded the same way test_signal_analyzer.py loads
its sibling — exec the script's function-definition prefix in isolation so
the (long) analysis script doesn't run at import.

Author: Karan Vasa
"""

import pathlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import utils.agents.agent_utils as au
from agents.actions import Action
from world.objects import WorldObject, EdibleComponent, SeedComponent
from world.tiles import TerrainType
from world.world import World
from world.object_registry import ObjectRegistry, register_builtin_objects

# --- analyzer prefix loader ------------------------------------------------

_PATH = pathlib.Path("scripts/analyze_logs.py").resolve()
_PREFIX = _PATH.read_text(encoding="utf-8").split("\nimport argparse")[0]
_NS: dict = {"__name__": "analyze_logs_partial", "__file__": str(_PATH)}
exec(compile(_PREFIX, str(_PATH), "exec"), _NS)
_compute_society_metrics = _NS["_compute_society_metrics"]


@pytest.fixture(autouse=True)
def _reset_registry():
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    ObjectRegistry._definitions.clear()


# --- helpers ---------------------------------------------------------------


def _soil_world(n=9, **kw):
    w = World(
        n,
        n,
        seed=1,
        soil_ratio=1.0,
        rock_ratio=0.0,
        water_ratio=0.0,
        sand_ratio=0.0,
        parallel=False,
        **kw,
    )
    for y in range(n):
        for x in range(n):
            t = w.get_tile(x, y)
            t.terrain_type = TerrainType.SOIL
            t.fertility = 0.6
            t.moisture = 0.6
    return w


_AGENT_COUNTER = 0


def _agent(x, y, energy=100.0, direction=(0, -1), inv=None):
    global _AGENT_COUNTER
    _AGENT_COUNTER += 1
    return SimpleNamespace(
        id=_AGENT_COUNTER,
        x=x,
        y=y,
        energy=energy,
        max_energy=200.0,
        age=0,
        max_age=1000,
        direction=direction,
        inventory=list(inv or []),
        inventory_size=5,
        metabolism_rate=0.5,
        alive=True,
        fitness=0.0,
    )


def _put_berry(world, x, y):
    """Add a berry-like edible object to (x, y) and return its id."""
    obj = WorldObject(x=x, y=y)
    obj.type_id = "berry"
    obj.add_component(EdibleComponent(calories=20.0, freshness=1.0))
    world.add_object(obj)
    return obj.id


def _put_seed(world, x, y):
    obj = WorldObject(x=x, y=y)
    obj.type_id = "plant_seed"
    obj.add_component(SeedComponent(plant_type="plant", max_age=200))
    world.add_object(obj)
    return obj.id


# ===========================================================================
# USE-as-trade (execute_use + get_action_mask)
# ===========================================================================


class TestTradeViaUse:
    def test_transfer_off_falls_through_to_legacy_use(self):
        # Default world: transfer disabled. Facing agent shouldn't matter.
        w = _soil_world()
        giver = _agent(4, 4, direction=(0, -1))
        recv = _agent(4, 3)  # tile in front
        w.agents = {giver.id: giver, recv.id: recv}
        bid = _put_berry(w, 0, 0)
        giver.inventory.append(bid)

        res = au.execute_use(giver, w)
        # Berry is not a seed/fertilizer → legacy USE refuses it.
        assert not res.success
        # Recipient still has nothing; giver still holds the berry.
        assert recv.inventory == []
        assert giver.inventory == [bid]

    def test_transfer_on_facing_agent_hands_over_first_item(self):
        w = _soil_world(social_config={"transfer_enabled": True})
        giver = _agent(4, 4, direction=(0, -1))
        recv = _agent(4, 3)
        w.agents = {giver.id: giver, recv.id: recv}
        bid = _put_berry(w, 0, 0)
        giver.inventory.append(bid)

        res = au.execute_use(giver, w)
        assert res.success
        assert res.interaction_kind == "give"
        assert res.target_x == 4 and res.target_y == 3
        assert giver.inventory == []
        assert recv.inventory == [bid]
        # Object's recorded position now matches the recipient
        obj = w.objects[bid]
        assert (obj.x, obj.y) == (recv.x, recv.y)

    def test_transfer_blocked_when_recipient_full(self):
        w = _soil_world(social_config={"transfer_enabled": True})
        giver = _agent(4, 4, direction=(0, -1))
        recv = _agent(4, 3)
        # Fill recipient inventory to capacity with dummy ints (the path only
        # checks len(), not the contents).
        recv.inventory = [-1] * recv.inventory_size
        w.agents = {giver.id: giver, recv.id: recv}
        bid = _put_berry(w, 0, 0)
        giver.inventory.append(bid)

        res = au.execute_use(giver, w)
        assert not res.success
        assert res.interaction_kind == "give_full"
        # No transfer happened
        assert giver.inventory == [bid]
        assert len(recv.inventory) == recv.inventory_size

    def test_transfer_with_no_facing_agent_falls_through(self):
        # Trade enabled but nobody in front → behave like legacy USE
        w = _soil_world(social_config={"transfer_enabled": True})
        giver = _agent(4, 4, direction=(0, -1))
        sid = _put_seed(w, 0, 0)
        giver.inventory.append(sid)
        w.agents = {giver.id: giver}

        res = au.execute_use(giver, w)
        # No recipient → seed gets planted (the tile is soil/fertile)
        assert res.success
        assert res.interaction_kind in {"plant_seed", "plant_seed_nearby"}

    def test_action_mask_enables_use_when_only_a_trade_target_exists(self):
        # Inventory has a berry (not plantable). Without a recipient,
        # legacy USE has nothing to do → mask should be 0. Adding a
        # facing recipient with the trade flag on flips it to 1.
        w_no_trade = _soil_world()
        giver = _agent(4, 4, direction=(0, -1))
        bid = _put_berry(w_no_trade, 0, 0)
        giver.inventory.append(bid)
        # Make sure giver's tile already has an "occupant" so seed-planting
        # logic also wouldn't kick in; here berries aren't seeds anyway.
        w_no_trade.agents = {giver.id: giver}
        # Mock brain.output_size so the mask returns 8 entries
        giver.brain = SimpleNamespace(output_size=8)
        mask_off = au.get_action_mask(giver, w_no_trade)
        assert mask_off[Action.USE.value] == 0.0

        # With trade on + recipient ahead → USE is allowed.
        w_trade = _soil_world(social_config={"transfer_enabled": True})
        giver2 = _agent(4, 4, direction=(0, -1))
        recv2 = _agent(4, 3)
        bid2 = _put_berry(w_trade, 0, 0)
        giver2.inventory.append(bid2)
        giver2.brain = SimpleNamespace(output_size=8)
        w_trade.agents = {giver2.id: giver2, recv2.id: recv2}
        mask_on = au.get_action_mask(giver2, w_trade)
        assert mask_on[Action.USE.value] == 1.0


# ===========================================================================
# Society metrics (analyzer)
# ===========================================================================


def _df(rows):
    return pd.DataFrame(rows)


class TestSocietyMetrics:
    def test_empty_when_fewer_than_two_agents(self):
        df = _df([{"action": "EAT", "agent_id": 1}] * 3)
        assert _compute_society_metrics(df) == {}

    def test_role_entropy_high_when_dominant_actions_diverge(self):
        rows = (
            [{"action": "EAT", "agent_id": 1}] * 10
            + [{"action": "MOVE_FORWARD", "agent_id": 2}] * 10
            + [{"action": "TURN_LEFT", "agent_id": 3}] * 10
        )
        out = _compute_society_metrics(_df(rows))
        assert out["distinct_roles"] == 3
        # Three roles evenly → max normalised entropy
        assert out["role_entropy_norm"] == pytest.approx(1.0, abs=1e-6)

    def test_role_entropy_low_when_all_agents_share_dominant_action(self):
        rows = []
        for aid in range(1, 5):
            rows += [{"action": "EAT", "agent_id": aid}] * 9
            rows += [{"action": "WAIT", "agent_id": aid}]
        out = _compute_society_metrics(_df(rows))
        # All four agents have "EAT" as dominant → single role
        assert out["distinct_roles"] == 1
        assert out["role_entropy_norm"] == 0.0

    def test_novelty_zero_when_distributions_identical(self):
        # Two agents with the SAME action distribution → JS = 0
        rows = []
        for aid in (1, 2):
            rows += [{"action": "EAT", "agent_id": aid}] * 5
            rows += [{"action": "WAIT", "agent_id": aid}] * 5
        out = _compute_society_metrics(_df(rows))
        assert out["novelty_mean_js"] == 0.0

    def test_novelty_positive_when_distributions_disjoint(self):
        rows = [{"action": "EAT", "agent_id": 1}] * 10 + [
            {"action": "WAIT", "agent_id": 2}
        ] * 10
        out = _compute_society_metrics(_df(rows))
        assert out["novelty_mean_js"] > 0.5  # ~1.0 for disjoint binary dists

    def test_territory_bbox_and_overlap(self):
        # Two agents in disjoint regions → overlap 0, bbox sensible
        rows = []
        # Agent 1: 5x5 square at (0,0)-(4,4)
        for x in range(5):
            for y in range(5):
                rows.append({"action": "MOVE_FORWARD", "agent_id": 1, "x": x, "y": y})
        # Agent 2: 5x5 square at (10,10)-(14,14)
        for x in range(10, 15):
            for y in range(10, 15):
                rows.append({"action": "MOVE_FORWARD", "agent_id": 2, "x": x, "y": y})
        out = _compute_society_metrics(_df(rows))
        assert out["mean_bbox_area"] == 25.0
        assert out["mean_visited_cells"] == 25.0
        assert out["mean_territory_overlap"] == 0.0

    def test_trade_metrics_count_give_actions(self):
        rows = [
            {
                "action": "USE",
                "agent_id": 1,
                "interaction_kind": "give",
                "target_x": 4,
                "target_y": 3,
            },
            {
                "action": "USE",
                "agent_id": 1,
                "interaction_kind": "give",
                "target_x": 4,
                "target_y": 3,
            },
            {
                "action": "USE",
                "agent_id": 2,
                "interaction_kind": "give",
                "target_x": 5,
                "target_y": 5,
            },
            {"action": "SIGNAL", "agent_id": 1, "interaction_kind": "signal"},
            {"action": "WAIT", "agent_id": 1, "interaction_kind": ""},
            {"action": "WAIT", "agent_id": 2, "interaction_kind": ""},
        ]
        out = _compute_society_metrics(_df(rows))
        assert out["give_actions"] == 3
        assert out["givers"] == 2
        assert out["distinct_recipients"] == 2
        assert out["give_per_signal"] == pytest.approx(3.0)

    def test_handles_missing_action_or_agent_column(self):
        # No action column
        assert _compute_society_metrics(pd.DataFrame({"x": [1, 2]})) == {}
        # No agent_id column
        assert _compute_society_metrics(pd.DataFrame({"action": ["EAT", "WAIT"]})) == {}
