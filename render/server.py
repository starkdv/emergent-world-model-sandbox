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
    Steps a SimSession on a timer and hands the latest snapshot + a running
    delta log to clients. Each SSE client replays the current snapshot, then
    receives every delta produced after it connected.
    """

    def __init__(self, session, tps: float = 10.0):
        self.session = session
        self.interval = 1.0 / max(0.1, tps)
        self._snapshot = session.snapshot()
        self._listeners: list[list] = []  # each: a per-client queue (list) + lock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def snapshot(self) -> dict:
        return self._snapshot

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                delta = self.session.step()
            except Exception as exc:  # keep the server alive on a sim error
                delta = {"type": "error", "message": str(exc)}
            with self._lock:
                for q in self._listeners:
                    q.append(delta)
            dt = time.perf_counter() - t0
            self._stop.wait(max(0.0, self.interval - dt))

    def add_listener(self) -> list:
        q: list = []
        with self._lock:
            self._listeners.append(q)
        return q

    def remove_listener(self, q: list) -> None:
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)


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
            q = broadcaster.add_listener()
            try:
                self._sse_send("snapshot", broadcaster.snapshot())
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
                broadcaster.remove_listener(q)

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


def main(argv=None):
    p = argparse.ArgumentParser(description="Live 3D-frontend state server (SSE)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tps", type=float, default=10.0, help="sim ticks per second")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--agents", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
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
    args = p.parse_args(argv)

    if args.replay:
        from render.recorder import replay_session_from_file

        session = replay_session_from_file(args.replay)
        print(f"Replaying recording: {args.replay}")
    elif args.checkpoint:
        from render.sim_session import session_from_checkpoint

        session = session_from_checkpoint(args.checkpoint)
    else:
        from render.sim_session import build_demo_world

        session = build_demo_world(
            width=args.width, height=args.height, n_agents=args.agents, seed=args.seed
        )
    serve(session, host=args.host, port=args.port, tps=args.tps)


if __name__ == "__main__":
    main()
