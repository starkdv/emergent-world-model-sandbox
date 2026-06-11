"""
Data sub-module for logging, export, and analysis utilities.

Author: Karan Vasa
"""

from utils.data.async_logger import AsyncWorldModelLogger

__all__ = ["AsyncWorldModelLogger"]

# agent_logger.py (AgentLogger / WorldModelLogger, used by --log) has never
# been committed to the repository — importing it unconditionally made the
# whole utils.data package unimportable, breaking --world-model-log and
# several test modules. Import it defensively until the file is added.
try:
    from utils.data.agent_logger import AgentLogger, WorldModelLogger  # noqa: F401

    __all__ += ["AgentLogger", "WorldModelLogger"]
except ImportError:  # pragma: no cover - module not in the repository
    pass
