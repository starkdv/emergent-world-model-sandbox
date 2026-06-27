"""
Tests for F5 recording & replay (render/recorder.py).

record() writes a JSONL stream (snapshot + one delta per tick); Recording.load
reads it back; ReplaySession replays it through the SimSession interface and
loops with a snapshot resync frame.

Author: Karan Vasa
"""

import json
import os

import pytest

from render.recorder import Recording, ReplaySession, record, replay_session_from_file
from render.sim_session import build_demo_world
from world.object_registry import ObjectRegistry, register_builtin_objects


@pytest.fixture(autouse=True)
def _registry():
    ObjectRegistry._definitions.clear()
    register_builtin_objects()
    yield
    from agents.agent import Agent

    Agent.brain_config = None
    ObjectRegistry._definitions.clear()


def _record(tmp_path, ticks=8):
    s = build_demo_world(width=20, height=20, n_agents=4, seed=4)
    path = os.path.join(str(tmp_path), "rec.jsonl")
    record(s, ticks, path)
    return path


def test_record_writes_snapshot_then_deltas(tmp_path):
    path = _record(tmp_path, ticks=8)
    lines = [json.loads(x) for x in open(path) if x.strip()]
    assert len(lines) == 9  # snapshot + 8 deltas
    assert lines[0]["type"] == "snapshot"
    assert all(f["type"] == "delta" for f in lines[1:])


def test_recording_load_roundtrip(tmp_path):
    path = _record(tmp_path, ticks=5)
    rec = Recording.load(path)
    assert len(rec) == 6
    assert rec.snapshot_frame["type"] == "snapshot"
    assert len(rec.deltas) == 5


def test_replay_session_snapshot_and_steps(tmp_path):
    path = _record(tmp_path, ticks=4)
    rs = replay_session_from_file(path)
    snap = rs.snapshot()
    assert snap["type"] == "snapshot"
    # 4 deltas come back in order
    kinds = [rs.step()["type"] for _ in range(4)]
    assert kinds == ["delta", "delta", "delta", "delta"]


def test_replay_loops_with_resync_snapshot(tmp_path):
    path = _record(tmp_path, ticks=3)
    rs = replay_session_from_file(path)
    rs.snapshot()
    # 3 deltas, then the loop should re-emit the snapshot as a resync frame
    seq = [rs.step()["type"] for _ in range(4)]
    assert seq == ["delta", "delta", "delta", "snapshot"]
    # and continue with deltas after the resync
    assert rs.step()["type"] == "delta"


def test_bad_recording_rejected(tmp_path):
    path = os.path.join(str(tmp_path), "bad.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({"type": "delta"}) + "\n")  # no leading snapshot
    with pytest.raises(ValueError):
        Recording.load(path)


def test_replay_via_server_session_interface(tmp_path):
    # The server only needs snapshot()/step()/tick — confirm ReplaySession
    # satisfies that contract so render.server can stream it unchanged.
    path = _record(tmp_path, ticks=3)
    rs = replay_session_from_file(path)
    assert hasattr(rs, "snapshot") and hasattr(rs, "step")
    assert isinstance(rs.tick, int)
    rs.snapshot()
    assert isinstance(rs.step(), dict)
