"""
WebSocket broadcast server for live simulation streaming.

Runs an asyncio event loop on a dedicated daemon thread and exposes a single
thread-safe entry point, ``publish(frame)``, callable from the (synchronous)
simulation thread. Internally it uses newest-wins coalescing: each ``publish``
stores the latest serialized frame and wakes a sender task, which broadcasts the
most recent frame to all connected clients. If the simulation outruns the
clients, intermediate frames are dropped rather than queued -- the right
behaviour for live viewing, and it guarantees ``publish`` never blocks the tick
loop.

On connect, each client first receives the ``init`` message (full scene /
terrain) produced by the ``get_init`` callback, then the latest frame, then the
ongoing stream. Inbound messages from clients are currently ignored; the
``async for`` consume loop is the natural place to add control commands
(pause / speed / seek) later.

``websockets`` is an optional dependency -- importing this module without it
installed is fine; the error is raised only when ``start()`` is called.

Author: streaming integration
"""

import asyncio
import json
import threading
from typing import Callable, Optional, Set

try:
    import websockets
except ImportError:  # pragma: no cover - optional dependency
    websockets = None


class StreamServer:
    """
    Thread-backed WebSocket server that broadcasts world snapshots.

    Typical usage from the simulation thread::

        server = StreamServer(get_init=lambda: build_init(world))
        server.start()
        ...
        for _ in range(total_ticks):
            world.update()
            server.publish(build_frame(world))
        server.stop()

    Author: streaming integration
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        get_init: Optional[Callable[[], Optional[dict]]] = None,
    ):
        """
        Initialise the server (does not start the thread).

        Args:
            host: Interface to bind. Defaults to loopback for safety.
            port: TCP port for the WebSocket endpoint.
            get_init: Optional callable returning the per-client ``init`` dict
                (full scene description). Invoked once per client connection,
                inside the server's event loop -- keep it cheap and read-only.
        """
        self.host = host
        self.port = port
        self._get_init = get_init

        self._clients: Set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._wake: Optional[asyncio.Event] = None
        self._sender_task = None
        self._latest: Optional[str] = None
        self._started = threading.Event()
        self._start_error: Optional[BaseException] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """
        Start the event-loop thread and begin listening.

        Blocks (briefly) until the listening socket is bound, so callers can
        rely on the endpoint being reachable on return.

        Raises:
            RuntimeError: If the ``websockets`` package is not installed.
        """
        if websockets is None:
            raise RuntimeError(
                "The 'websockets' package is required for --stream. "
                "Install it with: pip install websockets"
            )
        self._thread = threading.Thread(
            target=self._run, name="stream-server", daemon=True
        )
        self._thread.start()
        # Wait for the listener to bind (or give up after 5s).
        bound = self._started.wait(timeout=5)
        if self._start_error is not None:
            raise RuntimeError(
                f"Stream server failed to start: {self._start_error!r}"
            )
        if not bound:
            raise RuntimeError("Stream server failed to start within 5s")

    def _run(self) -> None:
        """Thread target: own event loop, serve, then run forever."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve())
        except BaseException as e:  # surface bind/serve failures to start()
            self._start_error = e
            self._started.set()
            return
        self._loop.run_forever()

    async def _serve(self) -> None:
        """Bind the WebSocket listener and launch the sender task."""
        self._wake = asyncio.Event()
        self._server = await websockets.serve(self._handler, self.host, self.port)
        self._sender_task = asyncio.ensure_future(self._sender())
        self._started.set()

    # -- connection handling ----------------------------------------------

    async def _handler(self, ws) -> None:
        """
        Per-client coroutine: send init + latest frame, then idle-consume.

        Args:
            ws: The connected WebSocket.
        """
        self._clients.add(ws)
        try:
            if self._get_init is not None:
                try:
                    init = self._get_init()
                except Exception:
                    init = None
                if init is not None:
                    await ws.send(json.dumps(init, separators=(",", ":")))
            if self._latest is not None:
                await ws.send(self._latest)
            # Consume (and ignore) inbound messages. Hook control commands here.
            async for _message in ws:
                pass
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    async def _sender(self) -> None:
        """Coalescing broadcast loop: wake -> send most-recent frame to all."""
        assert self._wake is not None
        while True:
            await self._wake.wait()
            self._wake.clear()
            data = self._latest
            if data:
                await self._broadcast(data)

    async def _broadcast(self, data: str) -> None:
        """Send one serialized frame to every client; drop failed ones."""
        if not self._clients:
            return
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    # -- producer API (called from the simulation thread) -----------------

    def publish(self, frame: dict) -> None:
        """
        Hand a snapshot to the server for broadcast (thread-safe, non-blocking).

        Serializes immediately (in the caller's thread, where the dict is
        owned), stores it as the latest frame, and signals the sender task.
        Returns instantly; intermediate frames may be dropped under load.

        Args:
            frame: A JSON-serializable snapshot dict (see snapshot.build_frame).
        """
        if self._loop is None or self._wake is None:
            return
        self._latest = json.dumps(frame, separators=(",", ":"))
        self._loop.call_soon_threadsafe(self._wake.set)

    @property
    def client_count(self) -> int:
        """Number of currently connected clients."""
        return len(self._clients)

    def stop(self) -> None:
        """
        Close all connections and stop the event loop thread.

        Safe to call even if the server never fully started.
        """
        if self._loop is None:
            return

        async def _close():
            if self._sender_task is not None:
                self._sender_task.cancel()
                try:
                    await self._sender_task
                except (asyncio.CancelledError, Exception):
                    pass
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
            for ws in list(self._clients):
                try:
                    await ws.close()
                except Exception:
                    pass

        try:
            fut = asyncio.run_coroutine_threadsafe(_close(), self._loop)
            fut.result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
