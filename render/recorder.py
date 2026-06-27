"""
Recording & replay for the 3D frontend (phase F5).

Capture a run's render stream to disk and play it back later — "scrub a saved
run" without re-simulating. The format is **JSONL**: line 0 is the F0 snapshot,
each subsequent line is one per-tick delta (exactly the frames the live server
streams). It is a faithful recording of what the viewer showed, so playback
needs no sim and no extra dependencies.

  * ``record(session, ticks, path)`` — step a SimSession and append frames.
  * ``Recording.load(path)`` — read frames back.
  * ``ReplaySession`` — exposes the same ``snapshot()`` / ``step()`` interface
    as ``SimSession``, so ``render.server.serve`` streams a recording with zero
    changes. It loops: after the last delta it re-emits the snapshot (a resync
    frame the client already handles) and continues.

Author: Karan Vasa
"""

from __future__ import annotations

import json
from typing import List

from render.sim_session import SimSession


def record(session: SimSession, ticks: int, path: str) -> str:
    """
    Record ``ticks`` deltas (plus the opening snapshot) from ``session`` to a
    JSONL file at ``path``. Returns the path.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(session.snapshot()) + "\n")
        for _ in range(max(0, ticks)):
            f.write(json.dumps(session.step()) + "\n")
    return path


class Recording:
    """A loaded recording: frame 0 is the snapshot, the rest are deltas."""

    def __init__(self, frames: List[dict]):
        if not frames:
            raise ValueError("empty recording")
        if frames[0].get("type") != "snapshot":
            raise ValueError("first frame must be a snapshot")
        self.frames = frames

    @classmethod
    def load(cls, path: str) -> "Recording":
        frames: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
        return cls(frames)

    @property
    def snapshot_frame(self) -> dict:
        return self.frames[0]

    @property
    def deltas(self) -> List[dict]:
        return self.frames[1:]

    def __len__(self) -> int:
        return len(self.frames)


class ReplaySession:
    """
    Streams a :class:`Recording` through the SimSession interface.

    ``snapshot()`` returns the recorded opening frame; ``step()`` walks the
    recorded deltas and, on wrap, re-emits the snapshot as a resync frame
    (the client rebuilds on a ``snapshot`` event) before continuing.
    """

    def __init__(self, recording: Recording):
        self.recording = recording
        self._pos = 1  # next step() returns the first delta
        self._tick = int(recording.snapshot_frame.get("tick", 0))

    def snapshot(self) -> dict:
        self._pos = 1
        frame = self.recording.frames[0]
        self._tick = int(frame.get("tick", 0))
        return frame

    def step(self, n: int = 1) -> dict:
        frames = self.recording.frames
        frame = frames[self._pos]
        self._pos += 1
        if self._pos >= len(frames):
            self._pos = 0  # next step returns frame 0 (snapshot) → client resync
        self._tick = int(frame.get("tick", self._tick))
        return frame

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def alive_agents(self) -> int:
        return 0  # not tracked in replay


def replay_session_from_file(path: str) -> ReplaySession:
    return ReplaySession(Recording.load(path))


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Record a run's render stream (F5)")
    p.add_argument("--out", default="recording.jsonl")
    p.add_argument("--ticks", type=int, default=300)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--agents", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    if args.checkpoint:
        from render.sim_session import session_from_checkpoint

        session = session_from_checkpoint(args.checkpoint)
    else:
        from render.sim_session import build_demo_world

        session = build_demo_world(
            width=args.width, height=args.height, n_agents=args.agents, seed=args.seed
        )
    record(session, args.ticks, args.out)
    print(f"Recorded {args.ticks} ticks → {args.out}")


if __name__ == "__main__":
    main()
