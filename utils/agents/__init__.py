"""
Utility modules for agent-related functions.

This package contains helper functions and classes that support
the core agent functionality but are not essential to the agent's
primary operations.
"""

from utils.agents.perception import (
    build_observation,
    get_observation_size
)

from utils.agents.learning_utils import (
    Experience,
    ReplayBuffer,
    RewardShaper,
    BestAgentTracker
)

from utils.agents.evolution_utils import (
    EvolutionStats,
    calculate_fitness,
    adaptive_mutation_std
)

__all__ = [
    # Perception
    'build_observation',
    'get_observation_size',
    
    # Learning utilities
    'Experience',
    'ReplayBuffer',
    'RewardShaper',
    'BestAgentTracker',
    
    # Evolution utilities
    'EvolutionStats',
    'calculate_fitness',
    'adaptive_mutation_std',
]
