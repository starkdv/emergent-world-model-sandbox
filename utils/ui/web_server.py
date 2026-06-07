"""
Browser-based Three.js renderer backend for the Emergent World-Model Sandbox.

This module hosts the live simulation behind a small, dependency-free HTTP
server (Python standard library only).  A background thread advances the
``World`` while a :class:`http.server.ThreadingHTTPServer` serves:

* the static Three.js front-end (``web/`` directory), and
* a JSON API the browser polls for live state and uses to control the run.

It exposes the same ``run()`` entry point as the Pygame / GPU renderers so it
can be selected with a single CLI flag (``--web``).

API surface
-----------
``GET  /``                     → ``web/index.html``
``GET  /static/<path>``        → static asset
``GET  /api/meta``             → static metadata (world size, object registry…)
``GET  /api/state``            → per-frame dynamic snapshot
``GET  /api/terrain``          → flat terrain-type grid
``GET  /api/inspect/agent/<id>``
``GET  /api/inspect/object/<id>``
``GET  /api/inspect/tile?x=&y=``
``POST /api/control``          → ``{"cmd": "pause|resume|toggle|step|set_speed|
                                   spawn|reset", ...}``

Author: Karan Vasa
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse, parse_qs

from world.object_registry import ObjectRegistry
from utils.ui import web_serialize as ws

# Project root → web/ directory holding the front-end assets.
_WEB_DIR = Path(__file__).resolve().parents[2] / "web"

# Allowed static file extensions → MIME types.
_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".map": "application/json; charset=utf-8",
}


class WebSimulationServer:
    """
    Hosts a live ``World`` and serves a Three.js browser client.

    The simulation runs on a dedicated background thread; HTTP requests are
    handled on separate threads.  A single lock serialises all world access
    (advancing the simulation and reading it for serialisation) so the JSON
    snapshots are always internally consistent.

    Attributes:
        world: The live World instance (replaced on reset).
        host: Bind address.
        port: Bind port.
        paused: Whether the simulation is currently paused.
        speed: Speed multiplier applied to the base tick rate.
    """

    def __init__(
        self,
        world,
        config: Optional[dict] = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        ticks_per_second: float = 10.0,
        world_factory: Optional[Callable[[], Any]] = None,
        open_browser: bool = False,
        start_paused: bool = True,
    ):
        """
        Initialise the web simulation server.

        Args:
            world: The World instance to host.
            config: Optional simulation config dict (for the meta payload).
            host: Bind address (default localhost).
            port: Bind port (default 8000).
            ticks_per_second: Base simulation rate at speed 1.0x.
            world_factory: Optional zero-arg callable returning a fresh World;
                enables the "Reset" control.  If None, reset is unavailable.
            open_browser: Whether to open the default browser on start.
            start_paused: Whether to begin paused (recommended).
        """
        self.world = world
        self.config = config or {}
        self.host = host
        self.port = port
        self.base_tps = max(0.1, float(ticks_per_second))
        self.world_factory = world_factory
        self.open_browser = open_browser

        self.paused = bool(start_paused)
        self.speed = 1.0
        self.running = True

        # Measured simulation throughput (exponential moving average).
        self._sim_tps = 0.0

        # Terrain change tracking → client only refetches the grid on change.
        self._terrain_version = 1
        self._terrain_sig = ws.terrain_signature(world)

        # Single lock guarding all world reads/writes.
        self._lock = threading.RLock()
        self._sim_thread: Optional[threading.Thread] = None
        self._httpd: Optional[ThreadingHTTPServer] = None

        # Cached meta payload (rebuilt on reset).
        self._meta_cache = ws.build_meta(world, self.config)

    # ------------------------------------------------------------------
    # Simulation thread
    # ------------------------------------------------------------------

    def _advance_once(self) -> None:
        """Advance the world by a single tick and run any world-model logging."""
        self.world.update()

        from agents.agent import Agent

        if Agent.world_model_logger is not None:
            Agent.world_model_logger.log_world_state(self.world.tick, self.world)

        # Detect terrain changes (sand spread / reclaim) cheaply.
        sig = ws.terrain_signature(self.world)
        if sig != self._terrain_sig:
            self._terrain_sig = sig
            self._terrain_version += 1

    def _sim_loop(self) -> None:
        """Background loop advancing the simulation at the configured rate."""
        last = time.perf_counter()
        while self.running:
            target_tps = self.base_tps * max(0.0, self.speed)
            if self.paused or target_tps <= 0.0:
                time.sleep(0.02)
                last = time.perf_counter()
                continue

            period = 1.0 / target_tps
            with self._lock:
                self._advance_once()

            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                inst_tps = 1.0 / dt
                # Exponential moving average for a stable HUD readout.
                self._sim_tps = (
                    inst_tps
                    if self._sim_tps == 0.0
                    else 0.9 * self._sim_tps + 0.1 * inst_tps
                )

            # Sleep the remainder of the period (skip when running flat out).
            sleep_for = period - (time.perf_counter() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Control commands (invoked from HTTP handler threads)
    # ------------------------------------------------------------------

    def handle_control(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a control command and return a small status dictionary.

        Args:
            payload: Parsed JSON body, must contain a ``cmd`` key.

        Returns:
            A status dictionary echoing the resulting control state.
        """
        cmd = payload.get("cmd", "")

        if cmd == "pause":
            self.paused = True
        elif cmd == "resume":
            self.paused = False
        elif cmd == "toggle":
            self.paused = not self.paused
        elif cmd == "set_speed":
            try:
                self.speed = max(0.0, min(50.0, float(payload.get("speed", 1.0))))
            except (TypeError, ValueError):
                pass
        elif cmd == "step":
            try:
                n = max(1, min(1000, int(payload.get("ticks", 1))))
            except (TypeError, ValueError):
                n = 1
            with self._lock:
                for _ in range(n):
                    self._advance_once()
        elif cmd == "spawn":
            return self._handle_spawn(payload)
        elif cmd == "reset":
            return self._handle_reset()
        else:
            return {"ok": False, "error": f"unknown cmd: {cmd!r}"}

        return {"ok": True, "paused": self.paused, "speed": self.speed}

    def _handle_spawn(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Spawn a registered object at a tile (used by the browser spawn tool)."""
        type_id = payload.get("type_id", "")
        if ObjectRegistry.get(type_id) is None:
            return {"ok": False, "error": f"unknown type: {type_id!r}"}
        try:
            x = int(payload.get("x"))
            y = int(payload.get("y"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid coordinates"}
        with self._lock:
            if not self.world.is_valid_position(x, y):
                return {"ok": False, "error": "out of bounds"}
            obj = ObjectRegistry.create(type_id, x, y)
            ok = self.world.add_object(obj)
        return {"ok": bool(ok), "type_id": type_id, "x": x, "y": y}

    def _handle_reset(self) -> Dict[str, Any]:
        """Rebuild the world from the factory, if one was supplied."""
        if self.world_factory is None:
            return {"ok": False, "error": "reset unavailable (no world factory)"}
        with self._lock:
            self.paused = True
            self.world = self.world_factory()
            self._terrain_sig = ws.terrain_signature(self.world)
            self._terrain_version += 1
            self._sim_tps = 0.0
            self._meta_cache = ws.build_meta(self.world, self.config)
        return {"ok": True, "reset": True}

    # ------------------------------------------------------------------
    # Snapshot builders (thread-safe)
    # ------------------------------------------------------------------

    def get_meta(self) -> Dict[str, Any]:
        """Return the cached static metadata payload."""
        with self._lock:
            meta = dict(self._meta_cache)
        meta["terrain_version"] = self._terrain_version
        meta["base_tps"] = self.base_tps
        meta["reset_available"] = self.world_factory is not None
        return meta

    def get_state(self) -> Dict[str, Any]:
        """Return the current per-frame dynamic state snapshot."""
        with self._lock:
            state = ws.build_state(self.world, self.paused, self.speed, self._sim_tps)
            state["terrain_version"] = self._terrain_version
        return state

    def get_terrain(self) -> Dict[str, Any]:
        """Return the current terrain grid with its version stamp."""
        with self._lock:
            terrain = ws.build_terrain(self.world)
            terrain["version"] = self._terrain_version
        return terrain

    def get_inspect_agent(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """Return full detail for one agent (thread-safe)."""
        with self._lock:
            return ws.inspect_agent(self.world, agent_id)

    def get_inspect_object(self, object_id: int) -> Optional[Dict[str, Any]]:
        """Return full detail for one world object (thread-safe)."""
        with self._lock:
            return ws.inspect_object(self.world, object_id)

    def get_inspect_tile(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        """Return full detail for one tile (thread-safe)."""
        with self._lock:
            return ws.inspect_tile(self.world, x, y)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the simulation thread and serve the browser client (blocking).

        Mirrors the ``run()`` API of the Pygame / GPU renderers.  Press
        Ctrl+C in the terminal to stop.
        """
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

        handler = partial(_RequestHandler, self)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)

        url = f"http://{self.host}:{self.port}/"
        print("\n" + "=" * 60)
        print("  Emergent World-Model Sandbox — Three.js Web UI")
        print("=" * 60)
        print(f"  Serving at: {url}")
        print("  Open the URL in a browser to view the live simulation.")
        print("  Controls (in-browser): Play/Pause, Step, Speed, Reset, Spawn.")
        print("  Press Ctrl+C in this terminal to stop the server.")
        print("=" * 60 + "\n")

        if self.open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass

        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down web server...")
        finally:
            self.running = False
            self._httpd.shutdown()
            self._httpd.server_close()
            print(f"Simulation ended at tick {self.world.tick}")


class _RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler bound to a :class:`WebSimulationServer` instance."""

    # ``server_ref`` is injected via functools.partial as the first arg.
    def __init__(self, server_ref: WebSimulationServer, *args, **kwargs):
        self.server_ref = server_ref
        super().__init__(*args, **kwargs)

    # Silence the default per-request stderr logging (too noisy at 30 Hz).
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    # -- helpers --------------------------------------------------------

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        # Prevent path traversal outside the web directory.
        try:
            path.resolve().relative_to(_WEB_DIR.resolve())
        except ValueError:
            self._send_json({"error": "forbidden"}, status=403)
            return

        mime = _MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # -- routing --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        route = parsed.path

        try:
            if route == "/" or route == "/index.html":
                self._send_file(_WEB_DIR / "index.html")
            elif route.startswith("/static/"):
                rel = route[len("/static/") :].lstrip("/")
                self._send_file(_WEB_DIR / "static" / rel)
            elif route == "/api/meta":
                self._send_json(self.server_ref.get_meta())
            elif route == "/api/state":
                self._send_json(self.server_ref.get_state())
            elif route == "/api/terrain":
                self._send_json(self.server_ref.get_terrain())
            elif route.startswith("/api/inspect/agent/"):
                aid = int(route.rsplit("/", 1)[-1])
                data = self.server_ref.get_inspect_agent(aid)
                self._send_json(data or {"error": "not found"}, 200 if data else 404)
            elif route.startswith("/api/inspect/object/"):
                oid = int(route.rsplit("/", 1)[-1])
                data = self.server_ref.get_inspect_object(oid)
                self._send_json(data or {"error": "not found"}, 200 if data else 404)
            elif route == "/api/inspect/tile":
                qs = parse_qs(parsed.query)
                x = int(qs.get("x", ["0"])[0])
                y = int(qs.get("y", ["0"])[0])
                data = self.server_ref.get_inspect_tile(x, y)
                self._send_json(data or {"error": "not found"}, 200 if data else 404)
            else:
                self._send_json({"error": "not found"}, status=404)
        except (ValueError, KeyError) as exc:
            self._send_json({"error": str(exc)}, status=400)
        except BrokenPipeError:
            # Client navigated away mid-response — safe to ignore.
            pass

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        if parsed.path != "/api/control":
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
            result = self.server_ref.handle_control(payload)
            self._send_json(result)
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid JSON"}, status=400)
        except BrokenPipeError:
            pass
