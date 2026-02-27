"""
Observation vector construction for agent perception.

**This module re-exports from utils.agents.perception** which is
the canonical implementation.  Kept for backward-compatibility only —
all new work should go into utils/agents/perception.py.

Author: Karan Vasa
Date: November 14, 2025
Updated: February 2026 — redirected to canonical perception module
"""

# Re-export from canonical location
from utils.agents.perception import (  # noqa: F401
    build_observation,
    get_observation_size,
)
