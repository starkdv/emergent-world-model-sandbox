"""
Live simulation streaming over WebSocket.

Exposes a thread-backed broadcast server (:class:`StreamServer`) and snapshot
builders (:func:`build_init`, :func:`build_frame`) that serialize world state
into JSON-friendly messages for a browser / Three.js client. The simulation
engine is untouched: streaming is an opt-in side-channel wired into the headless
tick loop, consuming the same snapshot-then-handoff discipline the loggers use.

Author: streaming integration
"""

from utils.stream.server import StreamServer
from utils.stream.snapshot import build_init, build_frame

__all__ = ["StreamServer", "build_init", "build_frame"]
