"""
Live state server for the 3D frontend (phase F3a) — stdlib only.

Streams a :class:`SimSession` to browser clients over **Server-Sent Events**
(SSE) and serves the static ``web/`` client. SSE (not WebSocket) is deliberate:
a spectator view is one-directional (server → client), SSE needs no extra
dependency (it rides plain HTTP, consumed by the browser's ``EventSource``), and
it reconnects automatically. Camera/controls live entirely client-side, so no
client→server channel is needed.

Endpoints:
  * ``GET /``            → the web client (``web/index.html``)
  * ``GET /<file>``      → static assets under ``web/``
  * ``GET /api/snapshot``→ one full snapshot as JSON (handy for debugging / F1)
  * ``GET /api/stream``  → SSE: a ``snapshot`` event, then a ``delta`` event per
                           tick at ``--tps`` ticks/second

Run:  ``python -m render.server``  (then open http://localhost:8000)

The HTTP layer is stdlib ``ThreadingHTTPServer``; the sim advances in one
background thread and each client streams from the shared session. This module
imports nothing outside the standard library + the project, so it never affects
the test suite or CI.

Author: Karan Vasa
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

WEB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web"
)

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".ico": "image/x-icon",
}


class _Broadcaster:
    """
    Advances a session on a timer and streams it to SSE clients.

    Each client gets a **fresh snapshot of the current world at connect** plus
    its **own delta tracker**, so a client that joins late sees the live state
    (not a stale startup snapshot) — this is what keeps the initial population
    from appearing frozen at tick 0. Stepping and per-client delta computation
    happen under one lock so the world isn't mutated mid-read.

    Two modes:
      * live (the session exposes ``world``): the world is advanced once per
        tick and each client's tracker diffs it.
      * replay (no ``world``): recorded frames are fanned out to every client
        (snapshots in the recording resync late joiners on loop).
    """

    def __init__(self, session, tps: float = 10.0):
        self.session = session
        self.interval = 1.0 / max(0.1, tps)
        self._live = hasattr(session, "world")
        self._clients: list[dict] = []  # {queue, tracker}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # replay (frame) mode: one shared opening frame
        self._frame_snapshot = None if self._live else session.snapshot()

    def snapshot(self) -> dict:
        """A fresh snapshot of the current world (for /api/snapshot)."""
        if self._live:
            from render.state_bridge import world_snapshot

            with self._lock:
                return world_snapshot(self.session.world)
        return self._frame_snapshot

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        from render.state_bridge import world_snapshot  # noqa: F401 (live only)

        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                with self._lock:
                    if self._live:
                        self.session.world.update()
                        for c in self._clients:
                            c["queue"].append(c["tracker"].delta(self.session.world))
                    else:
                        frame = self.session.step()
                        for c in self._clients:
                            c["queue"].append(frame)
            except Exception as exc:  # keep the server alive on a sim error
                with self._lock:
                    for c in self._clients:
                        c["queue"].append({"type": "error", "message": str(exc)})
            dt = time.perf_counter() - t0
            self._stop.wait(max(0.0, self.interval - dt))

    def add_client(self) -> dict:
        """Register a client; return its record with a fresh snapshot + queue."""
        with self._lock:
            if self._live:
                from render.state_bridge import StateTracker, world_snapshot

                tracker = StateTracker()
                snap = world_snapshot(self.session.world)
                tracker.delta(self.session.world)  # prime to current state
                client = {"queue": [], "tracker": tracker, "snapshot": snap}
            else:
                client = {
                    "queue": [],
                    "tracker": None,
                    "snapshot": self._frame_snapshot,
                }
            self._clients.append(client)
            return client

    def remove_client(self, client: dict) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)


def _make_handler(broadcaster: "_Broadcaster"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet by default
            pass

        # -- helpers --
        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, rel: str):
            rel = rel.lstrip("/") or "index.html"
            path = os.path.normpath(os.path.join(WEB_DIR, rel))
            if not path.startswith(WEB_DIR) or not os.path.isfile(path):
                self._send_json({"error": "not found", "path": rel}, code=404)
                return
            ext = os.path.splitext(path)[1]
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header(
                "Content-Type", _CONTENT_TYPES.get(ext, "application/octet-stream")
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            client = broadcaster.add_client()
            q = client["queue"]
            try:
                # fresh snapshot of the CURRENT world for THIS client
                self._sse_send("snapshot", client["snapshot"])
                while True:
                    if q:
                        msg = q.pop(0)
                        self._sse_send(msg.get("type", "delta"), msg)
                    else:
                        # keep-alive comment so proxies don't time out
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        time.sleep(broadcaster.interval)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                broadcaster.remove_client(client)

        def _sse_send(self, event: str, data: dict):
            payload = "event: %s\ndata: %s\n\n" % (event, json.dumps(data))
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()

        # -- routing --
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/snapshot":
                self._send_json(broadcaster.snapshot())
            elif path == "/api/stream":
                self._serve_stream()
            else:
                self._serve_static(path)

    return Handler


def serve(session, host: str = "127.0.0.1", port: int = 8000, tps: float = 10.0):
    """Start the broadcaster + HTTP server (blocking)."""
    broadcaster = _Broadcaster(session, tps=tps)
    broadcaster.start()
    httpd = ThreadingHTTPServer((host, port), _make_handler(broadcaster))
    print(f"Serving 3D frontend on http://{host}:{port}  (tps={tps})")
    print(f"  static: {WEB_DIR}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        broadcaster.stop()
        httpd.server_close()


_EPILOG = """
What this serves
----------------
A read-only 3D view of the REAL simulation, streamed over SSE. Open the printed
URL in a browser. In the viewer: drag=orbit, scroll=zoom, WASD/QE=fly,
F=follow an agent, V=toggle the agent's 5x5 vision grid, click anything to
inspect it. The HUD `brain` field shows the live architecture mix (e.g.
"v3.5", or "v3.84 + v2.15" in a cohort world).

Worlds (pick one source; default is --config config/default.yaml)
-----------------------------------------------------------------
  --config FILE   Build the real world from a YAML config (size, biomes,
                  brain version, learning). THIS is the normal mode.
  --demo          A fixed self-contained scene (no config) for a quick look.
  --checkpoint F  Fly around a saved run (W6b checkpoint .pkl).
  --replay F      Replay a recording (render.recorder JSONL) on a loop.

The interesting configs (committed in config/)
----------------------------------------------
  config/default.yaml             v3 world, the general-purpose scene.
  config/worldmodel_v35.yaml      Brain v3.5 (78-dim obs + SIGNAL) + PPO.
  config/planning_curiosity_v35.yaml
                                  v3.5 + PPO + each agent runs a per-agent
                                  WORLD MODEL to PLAN (imagined latent rollouts
                                  choose the action) and be CURIOUS (intrinsic
                                  reward for surprise). Watch them explore ~2x
                                  more and forage far more deliberately.

Learning (on by default for --config)
-------------------------------------
  Agents learn live with the config's learning.algorithm (PPO). This also
  trains each agent's planner world model. Use --no-learn to freeze them
  (pure evolution / ablation; faster, but the planner won't improve).

Logging — capture data while you watch
--------------------------------------
  --log               per-action + per-state CSVs (for scripts/analyze_logs.py)
  --world-model-log   transition CSVs f(obs,a)->(next_obs,r,done) used to TRAIN
                      an offline PopulationWorldModel
  --log-dir DIR       where logs go (default data/logs)

Seed with trained weights
-------------------------
  --load-weights F    start every agent from a pre-trained genome (.npz from
                      `main.py --save-weights` or scripts/dream_evolve.py),
                      migrated onto the configured brain if layouts differ.

End-to-end workflow
-------------------
  # 1. WATCH agents plan + be curious (Codespaces: forward the port, open it)
  python -m render.server --config config/planning_curiosity_v35.yaml

  # 2. Same, but also CAPTURE data for training a world model
  python -m render.server --config config/worldmodel_v35.yaml \\
      --world-model-log --log --log-dir data/logs

  # 3. TRAIN an offline world model from the captured transitions
  python scripts/train_world_model.py \\
      --transitions "data/logs/transitions_*.csv" \\
      --config config/worldmodel_v35.yaml \\
      --out data/world_models/wm.pt --report data/world_models/wm.txt

  # 4. ANALYZE behaviour (e.g. compare planning vs not)
  python scripts/analyze_logs.py --file data/logs/agent_actions_*.csv

Note: the planner imagines rollouts every decision, so a planning world runs
slower than --tps (it advances as fast as the CPU allows); the view stays live.
"""


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Live 3D-frontend state server (SSE) for the real simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tps", type=float, default=10.0, help="target sim ticks/second")
    p.add_argument("--width", type=int, default=96, help="(--demo only) world width")
    p.add_argument("--height", type=int, default=96, help="(--demo only) world height")
    p.add_argument("--agents", type=int, default=24, help="(--demo only) agent count")
    p.add_argument("--seed", type=int, default=7, help="(--demo only) RNG seed")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Resume a W6b checkpoint instead of a fresh demo world",
    )
    p.add_argument(
        "--replay",
        default=None,
        help="Play back a recording (JSONL from render.recorder) on a loop",
    )
    p.add_argument(
        "--config",
        default="config/default.yaml",
        help="Build the world from this config file (size, terrain/biomes, "
        "brain version, learning). Use --demo for the fixed self-contained scene.",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use the fixed demo world instead of --config",
    )
    p.add_argument(
        "--no-learn",
        action="store_true",
        help="Disable live RL learning (default: on for --config). With it off "
        "the per-agent planner world model never improves.",
    )
    p.add_argument(
        "--load-weights",
        default=None,
        metavar="NPZ",
        help="Seed every agent with pre-trained genome weights (.npz)",
    )
    p.add_argument(
        "--log",
        action="store_true",
        help="Write per-action + per-state CSVs during the live run",
    )
    p.add_argument(
        "--world-model-log",
        action="store_true",
        help="Write transition CSVs (for scripts/train_world_model.py)",
    )
    p.add_argument(
        "--log-dir",
        default="data/logs",
        help="Directory for --log / --world-model-log output (default data/logs)",
    )
    args = p.parse_args(argv)

    if args.replay:
        from render.recorder import replay_session_from_file

        session = replay_session_from_file(args.replay)
        print(f"Replaying recording: {args.replay}")
    elif args.checkpoint:
        from render.sim_session import session_from_checkpoint

        session = session_from_checkpoint(args.checkpoint)
    elif args.demo:
        from render.sim_session import build_demo_world

        session = build_demo_world(
            width=args.width, height=args.height, n_agents=args.agents, seed=args.seed
        )
    else:
        from render.sim_session import session_from_config

        session = session_from_config(
            args.config,
            learning=not args.no_learn,
            load_weights=args.load_weights,
        )
        print(f"World from config: {args.config}")
        print(f"  learning: {'OFF' if args.no_learn else 'ON (per config)'}")

    # Optional live logging. Built AFTER the session so the observation layout
    # (78-dim under v3.5) is active and the transition columns size correctly.
    # Agents auto-log via these class-level loggers as the world advances.
    loggers = []
    if (args.log or args.world_model_log) and hasattr(session, "world"):
        from agents.agent import Agent

        if args.log:
            from utils.data import AgentLogger

            Agent.logger = AgentLogger(args.log_dir)
            loggers.append(Agent.logger)
            print(f"  logging actions/states -> {args.log_dir}")
        if args.world_model_log:
            from utils.data.async_logger import AsyncWorldModelLogger

            Agent.world_model_logger = AsyncWorldModelLogger(output_dir=args.log_dir)
            loggers.append(Agent.world_model_logger)
            print(f"  logging transitions -> {args.log_dir}")
    elif args.log or args.world_model_log:
        print("  (logging needs a live world — ignored for --replay)")

    try:
        serve(session, host=args.host, port=args.port, tps=args.tps)
    finally:
        for lg in loggers:
            try:
                lg.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
