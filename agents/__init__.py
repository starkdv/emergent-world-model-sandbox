"""
Agents module for the Emergent World-Model Sandbox.

This module contains agent representation, neural network brains,
genome management, and evolutionary mechanics.

Author: Karan Vasa
"""

from agents.agent import Agent
from agents.brain import Brain
from agents.genome import Genome, create_default_trait_config
from agents.actions import Action, ActionResult, DIRECTIONS
from agents.observation import build_observation, get_observation_size
from agents.learning import AgentLearner, Experience, ReplayBuffer, RewardShaper

__version__ = "0.1.0"

__all__ = [
    'Agent',
    'Brain',
    'Genome',
    'create_default_trait_config',
    'Action',
    'ActionResult',
    'DIRECTIONS',
    'build_observation',
    'get_observation_size',
    'AgentLearner',
    'Experience',
    'ReplayBuffer',
    'RewardShaper',
]
