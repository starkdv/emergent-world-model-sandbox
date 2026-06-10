"""
Backward-compatibility shim for brain neural network utilities.

The canonical implementations now live in the agents.brain package:
- activations / GRU step:      agents/brain/modules.py
- genome layout (count/pack/unpack): agents/brain/spec.py (ParamSpec)

This module re-exports them so existing imports keep working. New code
should import from agents.brain directly.

Author: Karan Vasa
Date: February 14, 2026
Updated: June 2026 — delegated to agents.brain.spec / agents.brain.modules
"""

from typing import Optional

import numpy as np

from agents.brain.modules import sigmoid, softmax, gru_step  # noqa: F401
from agents.brain.spec import build_brain_param_spec, build_nested_params

__all__ = [
    "sigmoid",
    "softmax",
    "gru_step",
    "calculate_weight_count",
    "unpack_weights",
]


def calculate_weight_count(
    input_size: int = 72,
    encoder_layers: Optional[list[int]] = None,
    gru_hidden_size: int = 32,
    output_size: int = 8,
) -> int:
    """
    Calculate total number of weights needed for the Brain network.

    Derived from the declarative ParamSpec (single source of truth).

    Args:
        input_size: Size of observation vector
        encoder_layers: Sizes of encoder hidden layers (default: [32])
        gru_hidden_size: Size of GRU hidden state
        output_size: Number of actions

    Returns:
        Total number of weights (including biases)
    """
    return build_brain_param_spec(
        input_size, encoder_layers, gru_hidden_size, output_size
    ).count()


def unpack_weights(
    flat_weights: np.ndarray,
    input_size: int,
    encoder_layers: list[int],
    gru_hidden_size: int,
    output_size: int,
) -> dict:
    """
    Unpack flattened weight vector into structured parameters.

    Derived from the declarative ParamSpec (single source of truth).
    The returned arrays are views into ``flat_weights``.

    Args:
        flat_weights: Flattened weight vector from genome
        input_size: Size of observation vector
        encoder_layers: Sizes of encoder hidden layers
        gru_hidden_size: Size of GRU hidden state
        output_size: Number of actions

    Returns:
        Dictionary of parameter matrices and biases
    """
    spec = build_brain_param_spec(
        input_size, encoder_layers, gru_hidden_size, output_size
    )
    named = spec.unpack(np.asarray(flat_weights))
    return build_nested_params(named, len(encoder_layers))
