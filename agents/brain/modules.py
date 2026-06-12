"""
Pure neural network building blocks for the brain.

These are stateless functions of (input, params) with no knowledge of
agents, the world, or genome layout. utils/agents/brain_utils.py
re-exports them for backward compatibility.

Author: Karan Vasa
Date: June 2026
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    Args:
        x: Input array

    Returns:
        Sigmoid-activated output (0 to 1)
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.

    Args:
        x: Input logits

    Returns:
        Probability distribution (sums to 1)
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def gru_step(x: np.ndarray, h: np.ndarray, gru_params: dict) -> np.ndarray:
    """
    Single GRU recurrent step.

    GRU equations:
        r = sigmoid(x @ Wr_input + h @ Wr_hidden + br)   # reset gate
        z = sigmoid(x @ Wz_input + h @ Wz_hidden + bz)   # update gate
        h_tilde = tanh(x @ Wh_input + (r * h) @ Wh_hidden + bh)  # candidate
        h_next = (1 - z) * h + z * h_tilde               # new hidden state

    Args:
        x: Current input (encoder output)
        h: Previous hidden state
        gru_params: Dictionary containing GRU weights

    Returns:
        New hidden state
    """
    # Reset gate
    r = sigmoid(
        x @ gru_params["Wr_input"] + h @ gru_params["Wr_hidden"] + gru_params["br"]
    )

    # Update gate
    z = sigmoid(
        x @ gru_params["Wz_input"] + h @ gru_params["Wz_hidden"] + gru_params["bz"]
    )

    # Candidate hidden state
    h_tilde = np.tanh(
        x @ gru_params["Wh_input"]
        + (r * h) @ gru_params["Wh_hidden"]
        + gru_params["bh"]
    )

    # New hidden state
    h_next = (1 - z) * h + z * h_tilde

    return h_next
