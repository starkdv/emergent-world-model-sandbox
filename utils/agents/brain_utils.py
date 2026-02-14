"""
Utility functions for neural network brain operations.

Contains activation functions, weight calculations, and other
neural network utilities used by the Brain class.

Author: Karan Vasa
Date: February 14, 2026
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


def calculate_weight_count(
    input_size: int = 64,
    encoder_layers: list[int] = None,
    gru_hidden_size: int = 32,
    output_size: int = 8
) -> int:
    """
    Calculate total number of weights needed for the Brain network.
    
    This computes the total parameter count for:
    - Encoder MLP layers
    - GRU recurrent layer (3 gates)
    - Policy head (actor)
    - Value head (critic)
    
    Args:
        input_size: Size of observation vector
        encoder_layers: Sizes of encoder hidden layers (default: [32])
        gru_hidden_size: Size of GRU hidden state
        output_size: Number of actions
        
    Returns:
        Total number of weights (including biases)
    """
    if encoder_layers is None:
        encoder_layers = [32]
    
    total = 0
    
    # 1. Encoder weights
    encoder_sizes = [input_size] + encoder_layers
    for i in range(len(encoder_sizes) - 1):
        in_size = encoder_sizes[i]
        out_size = encoder_sizes[i + 1]
        total += in_size * out_size + out_size  # weights + biases
    
    encoder_out = encoder_layers[-1]
    
    # 2. GRU weights (3 gates: reset, update, candidate)
    # Each gate: input_weights + hidden_weights + bias
    for _ in range(3):
        total += encoder_out * gru_hidden_size  # input weights
        total += gru_hidden_size * gru_hidden_size  # hidden weights
        total += gru_hidden_size  # bias
    
    # 3. Policy head
    total += gru_hidden_size * output_size + output_size  # weights + bias
    
    # 4. Value head
    total += gru_hidden_size + 1  # weight + bias (scalar output)
    
    return total


def unpack_weights(
    flat_weights: np.ndarray,
    input_size: int,
    encoder_layers: list[int],
    gru_hidden_size: int,
    output_size: int
) -> dict:
    """
    Unpack flattened weight vector into structured parameters.
    
    Creates separate dictionaries for:
    - Encoder (feedforward layers)
    - GRU (recurrent update gates)
    - Policy head (action logits)
    - Value head (state value)
    
    Args:
        flat_weights: Flattened weight vector from genome
        input_size: Size of observation vector
        encoder_layers: Sizes of encoder hidden layers
        gru_hidden_size: Size of GRU hidden state
        output_size: Number of actions
        
    Returns:
        Dictionary of parameter matrices and biases
    """
    params = {
        'encoder_weights': [],
        'encoder_biases': [],
        'gru': {},
        'policy_head': {},
        'value_head': {}
    }
    
    idx = 0
    
    # 1. Encoder MLP
    encoder_sizes = [input_size] + encoder_layers
    for i in range(len(encoder_sizes) - 1):
        in_size = encoder_sizes[i]
        out_size = encoder_sizes[i + 1]
        
        # Weight matrix
        w_size = in_size * out_size
        w = flat_weights[idx:idx + w_size].reshape(in_size, out_size)
        params['encoder_weights'].append(w)
        idx += w_size
        
        # Bias vector
        b = flat_weights[idx:idx + out_size]
        params['encoder_biases'].append(b)
        idx += out_size
    
    # Encoder output size
    encoder_out = encoder_layers[-1]
    
    # 2. GRU parameters
    # GRU has 3 gates: reset (r), update (z), candidate (h_tilde)
    # Each gate: W_input @ x + W_hidden @ h + bias
    
    # Reset gate
    params['gru']['Wr_input'] = flat_weights[idx:idx + encoder_out * gru_hidden_size].reshape(encoder_out, gru_hidden_size)
    idx += encoder_out * gru_hidden_size
    params['gru']['Wr_hidden'] = flat_weights[idx:idx + gru_hidden_size * gru_hidden_size].reshape(gru_hidden_size, gru_hidden_size)
    idx += gru_hidden_size * gru_hidden_size
    params['gru']['br'] = flat_weights[idx:idx + gru_hidden_size]
    idx += gru_hidden_size
    
    # Update gate
    params['gru']['Wz_input'] = flat_weights[idx:idx + encoder_out * gru_hidden_size].reshape(encoder_out, gru_hidden_size)
    idx += encoder_out * gru_hidden_size
    params['gru']['Wz_hidden'] = flat_weights[idx:idx + gru_hidden_size * gru_hidden_size].reshape(gru_hidden_size, gru_hidden_size)
    idx += gru_hidden_size * gru_hidden_size
    params['gru']['bz'] = flat_weights[idx:idx + gru_hidden_size]
    idx += gru_hidden_size
    
    # Candidate hidden state
    params['gru']['Wh_input'] = flat_weights[idx:idx + encoder_out * gru_hidden_size].reshape(encoder_out, gru_hidden_size)
    idx += encoder_out * gru_hidden_size
    params['gru']['Wh_hidden'] = flat_weights[idx:idx + gru_hidden_size * gru_hidden_size].reshape(gru_hidden_size, gru_hidden_size)
    idx += gru_hidden_size * gru_hidden_size
    params['gru']['bh'] = flat_weights[idx:idx + gru_hidden_size]
    idx += gru_hidden_size
    
    # 3. Policy head (GRU hidden → action logits)
    params['policy_head']['W'] = flat_weights[idx:idx + gru_hidden_size * output_size].reshape(gru_hidden_size, output_size)
    idx += gru_hidden_size * output_size
    params['policy_head']['b'] = flat_weights[idx:idx + output_size]
    idx += output_size
    
    # 4. Value head (GRU hidden → scalar value)
    params['value_head']['W'] = flat_weights[idx:idx + gru_hidden_size].reshape(gru_hidden_size, 1)
    idx += gru_hidden_size
    params['value_head']['b'] = flat_weights[idx:idx + 1]
    idx += 1
    
    return params


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
    r = sigmoid(x @ gru_params['Wr_input'] + h @ gru_params['Wr_hidden'] + gru_params['br'])
    
    # Update gate
    z = sigmoid(x @ gru_params['Wz_input'] + h @ gru_params['Wz_hidden'] + gru_params['bz'])
    
    # Candidate hidden state
    h_tilde = np.tanh(x @ gru_params['Wh_input'] + (r * h) @ gru_params['Wh_hidden'] + gru_params['bh'])
    
    # New hidden state
    h_next = (1 - z) * h + z * h_tilde
    
    return h_next
