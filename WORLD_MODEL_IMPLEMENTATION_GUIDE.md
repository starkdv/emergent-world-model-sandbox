# World Model Implementation Guide

**Author:** Karan Vasa  
**Date:** November 28, 2025  
**Status:** Implementation Roadmap

---

## Executive Summary

This document provides a comprehensive analysis of the current codebase and a detailed implementation guide for adding World Model capabilities. The goal is to enable agents to learn predictive models of their environment, leading to more intelligent behaviors through planning and curiosity-driven exploration.

---

## Table of Contents

1. [Current System Analysis](#1-current-system-analysis)
2. [What is a World Model?](#2-what-is-a-world-model)
3. [Integration Points](#3-integration-points)
4. [Implementation Roadmap](#4-implementation-roadmap)
5. [Detailed Implementation](#5-detailed-implementation)
6. [Configuration & Testing](#6-configuration--testing)
7. [Expected Outcomes](#7-expected-outcomes)

---

## 1. Current System Analysis

### 1.1 Existing Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     CURRENT SYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   World     │    │    Agent    │    │   Brain     │         │
│  │  (world.py) │    │  (agent.py) │    │  (brain.py) │         │
│  │             │    │             │    │             │         │
│  │ • Tiles     │───▶│ • Position  │───▶│ • Weights   │         │
│  │ • Objects   │    │ • Energy    │    │ • Forward() │         │
│  │ • Systems   │    │ • Inventory │    │ • Actions   │         │
│  │ • Agents    │    │ • Actions   │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                 │
│         │                  ▼                  │                 │
│         │          ┌─────────────┐            │                 │
│         │          │ Observation │            │                 │
│         └─────────▶│  (64 dims)  │────────────┘                 │
│                    └─────────────┘                              │
│                           │                                     │
│                           ▼                                     │
│                    ┌─────────────┐                              │
│                    │  Learning   │                              │
│                    │ (learning.py)│                              │
│                    │             │                              │
│                    │ • Replay    │                              │
│                    │ • Rewards   │                              │
│                    │ • Gradients │                              │
│                    └─────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Components Analysis

#### 1.2.1 Observation Vector (64 dimensions)

**File:** `agents/observation.py`

```python
# Current observation structure:
# - Agent internal state: 8 features
#   - energy_ratio (0-1)
#   - age_ratio (0-1)
#   - direction (4 one-hot: N, E, S, W)
#   - has_inventory_space (0 or 1)
#   - metabolism_rate (normalized)

# - Vision grid: 50 features (5×5 grid × 2 features)
#   - tile_type (0=rock, 0.25=water, 0.5=soil, 0.75=plant, 1.0=food)
#   - tile_value (fertility, calories, maturity)

# - Inventory: 6 features
#   - fullness, has_food, has_seed, has_fertilizer, total_calories, count
```

**State Vector for World Model:**
- Input: 64 observation features + 8 action one-hot = **72 features**
- Output: 64 predicted next observation features

#### 1.2.2 Action Space

**File:** `agents/actions.py`

```python
class Action(IntEnum):
    MOVE_FORWARD = 0    # Move one tile in current direction
    TURN_LEFT = 1       # Rotate 90° counter-clockwise
    TURN_RIGHT = 2      # Rotate 90° clockwise
    PICK_UP = 3         # Pick up object
    DROP = 4            # Drop object
    EAT = 5             # Consume food
    USE = 6             # Plant seed / use item
    WAIT = 7            # Do nothing
```

**For World Model:** One-hot encode action → 8 dimensions

#### 1.2.3 Experience Replay Buffer

**File:** `agents/learning.py`

```python
class Experience:
    observation: np.ndarray      # State s_t (64 dims)
    action: int                  # Action a_t (0-7)
    reward: float                # Reward r_t
    next_observation: np.ndarray # State s_{t+1} (64 dims)
    done: bool                   # Episode terminated?
```

**✅ PERFECT FOR WORLD MODEL TRAINING!**
The experience buffer already stores (s_t, a_t, s_{t+1}) tuples needed for training.

#### 1.2.4 Brain Architecture

**File:** `agents/brain.py`

```python
# Current policy network:
# Input: 64 → Hidden: [32, 16] → Output: 8
# Total weights: 64×32 + 32 + 32×16 + 16 + 16×8 + 8 = 2,744
```

**World Model Network (NEW):**
```python
# Forward dynamics:
# Input: 72 → Hidden: [128, 128, 64] → Output: 64
# Total weights: 72×128 + 128 + 128×128 + 128 + 128×64 + 64 + 64×64 + 64 ≈ 34,112
```

#### 1.2.5 Current Learning System

**File:** `agents/learning.py`

```python
class AgentLearner:
    # Current: Policy gradient (REINFORCE)
    # - Updates policy network weights
    # - Uses shaped rewards for survival
    # - Full backpropagation through brain
    
    def learn(self, brain):
        # Sample batch from replay buffer
        # Compute advantages
        # Backpropagate policy gradient
        # Sync weights to genome
```

**Enhancement needed:** Add world model training alongside policy learning.

### 1.3 Data Flow Analysis

```
Current Flow:
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐
│  World   │───▶│Observation│───▶│  Brain   │───▶│   Action   │
│  State   │    │  Vector   │    │ Forward  │    │ Execution  │
└──────────┘    └───────────┘    └──────────┘    └────────────┘
     │                                                  │
     │              ┌───────────────┐                   │
     └──────────────│ Learning.py   │◀──────────────────┘
                    │ (experience)  │
                    └───────────────┘

World Model Flow (NEW):
┌──────────┐    ┌───────────┐    ┌──────────────┐
│  State   │───▶│World Model│───▶│ Predicted    │
│  s_t     │    │ Network   │    │ State s_{t+1}│
└──────────┘    └───────────┘    └──────────────┘
     │                │                  │
     │                │                  ▼
     │                │          ┌──────────────┐
     │                │          │ Curiosity    │
     │                └─────────▶│ Reward       │
     │                           │ (pred error) │
     │                           └──────────────┘
     │                                  │
     ▼                                  ▼
┌──────────────────────────────────────────────┐
│           Enhanced Learning System           │
│  • Policy Gradient (existing)                │
│  • World Model Training (NEW)                │
│  • Curiosity-Driven Exploration (NEW)        │
│  • Model-Based Planning (NEW)                │
└──────────────────────────────────────────────┘
```

---

## 2. What is a World Model?

### 2.1 Definition

A **World Model** is a learned internal representation that agents use to:

1. **Predict future states** given current state and action
2. **Imagine consequences** before taking actions
3. **Generate curiosity** about unpredictable states
4. **Plan ahead** by simulating trajectories

### 2.2 Key Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      WORLD MODEL SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FORWARD DYNAMICS MODEL                                      │
│     f(s_t, a_t) → ŝ_{t+1}                                       │
│     Predicts: "What happens if I do action A in state S?"       │
│                                                                 │
│  2. INVERSE DYNAMICS MODEL                                      │
│     g(s_t, s_{t+1}) → â_t                                       │
│     Predicts: "What action caused this state change?"           │
│                                                                 │
│  3. INTRINSIC CURIOSITY MODULE (ICM)                           │
│     curiosity_reward = η × ||ŝ_{t+1} - s_{t+1}||²              │
│     Reward: "How surprised am I by the outcome?"                │
│                                                                 │
│  4. MODEL-BASED PLANNER                                         │
│     Simulates future trajectories using forward model           │
│     Selects: "Which action sequence leads to best outcome?"     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Benefits for This Project

| Current System | With World Model |
|----------------|------------------|
| Learns from trial-and-error only | Can imagine consequences |
| Random exploration | Curiosity-driven exploration |
| Reactive decisions | Planning-based decisions |
| Slow learning (many experiences needed) | Sample-efficient learning |
| No generalization | Can transfer knowledge |

---

## 3. Integration Points

### 3.1 Files to Modify

| File | Changes Needed |
|------|----------------|
| `agents/learning.py` | Add world model training |
| `agents/agent.py` | Integrate world model in update loop |
| `agents/brain.py` | Add world model network (or new file) |
| `config/*.yaml` | Add world model configuration |
| `main.py` | Add --world-model flag |

### 3.2 Files to Create

| New File | Purpose |
|----------|---------|
| `agents/world_model.py` | World model networks & training |
| `agents/curiosity.py` | Intrinsic curiosity module |
| `agents/planner.py` | Model-based planning |
| `tests/test_world_model.py` | Unit tests |
| `analysis/world_model_analysis.py` | Evaluation metrics |

### 3.3 Experience Buffer Enhancement

**Current buffer stores:**
```python
Experience(observation, action, reward, next_observation, done)
```

**Enhanced buffer (no changes needed!):**
The current buffer already has all data needed for world model training:
- `observation` = s_t (input to forward model)
- `action` = a_t (input to forward model)
- `next_observation` = s_{t+1} (target for forward model)

---

## 4. Implementation Roadmap

### Phase 3.1: Forward Dynamics Model (Week 1)

**Goal:** Predict next state given current state and action

```python
# agents/world_model.py

class ForwardDynamicsModel:
    """
    Predicts next observation given current observation and action.
    
    Architecture:
        Input: [observation (64) + action_onehot (8)] = 72
        Hidden: [128, 128, 64]
        Output: 64 (predicted next observation)
    """
    
    def __init__(self, obs_size=64, action_size=8, hidden_sizes=[128, 128, 64]):
        self.obs_size = obs_size
        self.action_size = action_size
        self.input_size = obs_size + action_size  # 72
        self.output_size = obs_size  # 64
        
        # Initialize weights
        self.weights, self.biases = self._init_weights(hidden_sizes)
    
    def predict(self, observation, action):
        """
        Predict next state.
        
        Args:
            observation: Current state (64,)
            action: Action index (int)
            
        Returns:
            Predicted next observation (64,)
        """
        # One-hot encode action
        action_onehot = np.zeros(self.action_size)
        action_onehot[action] = 1.0
        
        # Concatenate inputs
        x = np.concatenate([observation, action_onehot])
        
        # Forward pass
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(x @ w + b)
        
        # Output layer (no activation - predicting raw features)
        x = x @ self.weights[-1] + self.biases[-1]
        
        return x
    
    def train(self, experiences, learning_rate=0.001):
        """
        Train on batch of experiences.
        
        Loss = MSE(predicted_next_obs, actual_next_obs)
        """
        total_loss = 0.0
        
        for exp in experiences:
            # Forward pass
            predicted = self.predict(exp.observation, exp.action)
            target = exp.next_observation
            
            # Compute loss
            error = predicted - target
            loss = np.mean(error ** 2)
            total_loss += loss
            
            # Backpropagation (similar to brain.py)
            # ... gradient computation ...
            
        return total_loss / len(experiences)
```

**Integration with AgentLearner:**

```python
# agents/learning.py - MODIFIED

class AgentLearner:
    def __init__(self, ...):
        # Existing
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.reward_shaper = RewardShaper()
        
        # NEW: World model
        self.world_model = ForwardDynamicsModel()
        self.world_model_enabled = False
    
    def learn(self, brain):
        """Update both policy and world model."""
        
        # Existing policy learning
        policy_loss = self._policy_gradient_update(brain)
        
        # NEW: World model learning
        if self.world_model_enabled:
            wm_loss = self.world_model.train(
                self.replay_buffer.sample(self.batch_size)
            )
            return policy_loss, wm_loss
        
        return policy_loss, 0.0
```

### Phase 3.2: Inverse Dynamics Model (Week 2)

**Goal:** Learn which features are controllable

```python
class InverseDynamicsModel:
    """
    Predicts action from state transition.
    
    Helps identify which state features are:
    - Controllable (change based on agent action)
    - Uncontrollable (change due to environment)
    """
    
    def __init__(self, obs_size=64, action_size=8, hidden_sizes=[128, 64]):
        self.input_size = obs_size * 2  # s_t and s_{t+1}
        self.output_size = action_size
        
        self.weights, self.biases = self._init_weights(hidden_sizes)
    
    def predict(self, obs, next_obs):
        """Predict which action was taken."""
        x = np.concatenate([obs, next_obs])
        
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(x @ w + b)
        
        # Softmax output
        logits = x @ self.weights[-1] + self.biases[-1]
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        
        return probs
    
    def train(self, experiences, learning_rate=0.001):
        """
        Train with cross-entropy loss.
        
        Loss = -log(predicted_prob[actual_action])
        """
        total_loss = 0.0
        
        for exp in experiences:
            probs = self.predict(exp.observation, exp.next_observation)
            loss = -np.log(probs[exp.action] + 1e-8)
            total_loss += loss
            
            # Backpropagation...
            
        return total_loss / len(experiences)
```

### Phase 3.3: Intrinsic Curiosity Module (Week 3)

**Goal:** Reward exploration of novel/unpredictable states

```python
# agents/curiosity.py

class IntrinsicCuriosityModule:
    """
    Generates curiosity-based intrinsic rewards.
    
    Curiosity = prediction error of forward model
    High error = novel state = high reward
    """
    
    def __init__(
        self,
        forward_model: ForwardDynamicsModel,
        inverse_model: InverseDynamicsModel,
        curiosity_weight: float = 0.05,
        decay_rate: float = 0.999
    ):
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.curiosity_weight = curiosity_weight
        self.decay_rate = decay_rate
        
        # Track prediction errors for normalization
        self.error_history = []
        self.max_history = 1000
    
    def compute_curiosity_reward(self, obs, action, next_obs):
        """
        Compute intrinsic reward based on prediction error.
        
        Args:
            obs: Current observation
            action: Action taken
            next_obs: Actual next observation
            
        Returns:
            Intrinsic curiosity reward
        """
        # Predict next state
        predicted_next = self.forward_model.predict(obs, action)
        
        # Compute prediction error
        error = np.mean((predicted_next - next_obs) ** 2)
        
        # Track error for normalization
        self.error_history.append(error)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Normalize error (prevent reward explosion)
        mean_error = np.mean(self.error_history)
        std_error = np.std(self.error_history) + 1e-8
        normalized_error = (error - mean_error) / std_error
        
        # Scale to reasonable reward range
        curiosity_reward = self.curiosity_weight * np.clip(normalized_error, -3, 3)
        
        return curiosity_reward
    
    def decay_curiosity(self):
        """Reduce curiosity weight over time (optional)."""
        self.curiosity_weight *= self.decay_rate
```

**Integration with RewardShaper:**

```python
# agents/learning.py - MODIFIED

class RewardShaper:
    def __init__(self, curiosity_module=None):
        self.curiosity_module = curiosity_module
        # ... existing code ...
    
    def calculate_reward(self, action, action_result, ...):
        """Calculate total reward = extrinsic + intrinsic."""
        
        # Existing extrinsic rewards
        extrinsic_reward = self._calculate_extrinsic(...)
        
        # NEW: Intrinsic curiosity reward
        intrinsic_reward = 0.0
        if self.curiosity_module is not None:
            intrinsic_reward = self.curiosity_module.compute_curiosity_reward(
                observation, action, next_observation
            )
        
        return extrinsic_reward + intrinsic_reward
```

### Phase 3.4: Model-Based Planning (Week 4)

**Goal:** Use world model to plan action sequences

```python
# agents/planner.py

class ModelBasedPlanner:
    """
    Uses world model to simulate future and plan actions.
    
    Implements multiple planning algorithms:
    - Random Shooting
    - Cross-Entropy Method (CEM)
    - Simple lookahead
    """
    
    def __init__(
        self,
        forward_model: ForwardDynamicsModel,
        horizon: int = 10,
        num_trajectories: int = 50,
        method: str = "random_shooting"
    ):
        self.forward_model = forward_model
        self.horizon = horizon
        self.num_trajectories = num_trajectories
        self.method = method
    
    def plan(self, current_obs, reward_fn):
        """
        Plan best action sequence from current state.
        
        Args:
            current_obs: Current observation
            reward_fn: Function(obs) -> estimated reward
            
        Returns:
            Best action to take now
        """
        if self.method == "random_shooting":
            return self._random_shooting(current_obs, reward_fn)
        elif self.method == "cem":
            return self._cross_entropy_method(current_obs, reward_fn)
        else:
            return self._simple_lookahead(current_obs, reward_fn)
    
    def _random_shooting(self, obs, reward_fn):
        """
        Random Shooting: Sample random trajectories, pick best.
        """
        best_action = 0
        best_value = float('-inf')
        
        for _ in range(self.num_trajectories):
            # Sample random action sequence
            actions = np.random.randint(0, 8, size=self.horizon)
            
            # Simulate trajectory
            trajectory_value = self._simulate_trajectory(obs, actions, reward_fn)
            
            # Track best
            if trajectory_value > best_value:
                best_value = trajectory_value
                best_action = actions[0]  # Return first action
        
        return best_action
    
    def _simulate_trajectory(self, obs, actions, reward_fn):
        """Simulate trajectory using world model."""
        current_obs = obs.copy()
        total_reward = 0.0
        discount = 0.95
        
        for t, action in enumerate(actions):
            # Predict next state
            next_obs = self.forward_model.predict(current_obs, action)
            
            # Estimate reward
            reward = reward_fn(next_obs)
            total_reward += (discount ** t) * reward
            
            current_obs = next_obs
        
        return total_reward
    
    def _simple_lookahead(self, obs, reward_fn):
        """
        Simple 1-step lookahead for each action.
        """
        best_action = 0
        best_value = float('-inf')
        
        for action in range(8):
            predicted_next = self.forward_model.predict(obs, action)
            value = reward_fn(predicted_next)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
```

**Integration with Agent Decision-Making:**

```python
# agents/agent.py - MODIFIED

def update(self, world):
    """Update agent for one tick."""
    
    # ... existing code ...
    
    # Decision making
    observation = build_observation(self, world)
    
    if self.use_planning and self.learner.planner is not None:
        # Model-based planning
        action_idx = self.learner.planner.plan(
            observation,
            reward_fn=self._estimate_observation_value
        )
    else:
        # Model-free (existing)
        action_idx = self.brain.decide_action(observation)
    
    # Execute action
    # ... existing code ...

def _estimate_observation_value(self, obs):
    """Estimate value of an observation for planning."""
    # Simple heuristic: higher energy, near food = good
    energy = obs[0]  # First feature is energy ratio
    food_nearby = max(obs[8::2])  # Check vision for food (type=1.0)
    
    return energy * 0.5 + food_nearby * 0.5
```

### Phase 3.5: Latent World Model (Optional - Advanced)

**Goal:** Learn compressed representation of state

```python
class LatentWorldModel:
    """
    Learns dynamics in compressed latent space.
    
    Components:
    - Encoder: observation → latent
    - Dynamics: latent + action → next_latent
    - Decoder: latent → observation
    """
    
    def __init__(self, obs_size=64, latent_size=16, action_size=8):
        self.obs_size = obs_size
        self.latent_size = latent_size
        self.action_size = action_size
        
        # Encoder: obs → latent
        self.encoder = self._build_encoder()
        
        # Dynamics: latent + action → next_latent
        self.dynamics = self._build_dynamics()
        
        # Decoder: latent → obs
        self.decoder = self._build_decoder()
    
    def encode(self, obs):
        """Compress observation to latent space."""
        # ... neural network forward pass ...
    
    def decode(self, latent):
        """Reconstruct observation from latent."""
        # ... neural network forward pass ...
    
    def predict_latent(self, latent, action):
        """Predict next latent state."""
        # ... neural network forward pass ...
    
    def predict(self, obs, action):
        """Full forward model: obs + action → next_obs."""
        latent = self.encode(obs)
        next_latent = self.predict_latent(latent, action)
        next_obs = self.decode(next_latent)
        return next_obs
    
    def train(self, experiences):
        """
        Train with combined loss:
        L = L_reconstruction + L_prediction + L_regularization
        """
        # ... training logic ...
```

---

## 5. Detailed Implementation

### 5.1 New File: `agents/world_model.py`

```python
"""
World Model for agent learning.

Implements forward dynamics prediction, inverse dynamics,
and integrates with the learning system.

Author: Karan Vasa
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Optional


class ForwardDynamicsModel:
    """
    Predicts next observation given current observation and action.
    
    f(s_t, a_t) → ŝ_{t+1}
    
    Architecture:
        Input: observation (64) + action_onehot (8) = 72
        Hidden: [128, 128, 64] with tanh activations
        Output: 64 (predicted next observation)
        
    Training:
        Loss = MSE(predicted, actual)
        Optimizer: SGD with momentum
    """
    
    def __init__(
        self,
        obs_size: int = 64,
        action_size: int = 8,
        hidden_sizes: List[int] = None,
        learning_rate: float = 0.001
    ):
        if hidden_sizes is None:
            hidden_sizes = [128, 128, 64]
        
        self.obs_size = obs_size
        self.action_size = action_size
        self.input_size = obs_size + action_size
        self.output_size = obs_size
        self.learning_rate = learning_rate
        
        # Build network
        layer_sizes = [self.input_size] + hidden_sizes + [self.output_size]
        self.weights, self.biases = self._init_weights(layer_sizes)
        
        # Momentum for SGD
        self.weight_velocity = [np.zeros_like(w) for w in self.weights]
        self.bias_velocity = [np.zeros_like(b) for b in self.biases]
        self.momentum = 0.9
        
        # Metrics
        self.training_losses = []
        self.prediction_errors = []
    
    def _init_weights(self, layer_sizes: List[int]) -> Tuple[List, List]:
        """Xavier initialization for weights."""
        weights = []
        biases = []
        
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            # Xavier init
            std = np.sqrt(2.0 / (in_size + out_size))
            w = np.random.randn(in_size, out_size) * std
            b = np.zeros(out_size)
            
            weights.append(w)
            biases.append(b)
        
        return weights, biases
    
    def forward(self, obs: np.ndarray, action: int) -> Tuple[np.ndarray, List]:
        """
        Forward pass with activation caching for backprop.
        
        Returns:
            predicted_next_obs, list of activations
        """
        # One-hot encode action
        action_onehot = np.zeros(self.action_size)
        action_onehot[action] = 1.0
        
        # Concatenate inputs
        x = np.concatenate([obs.flatten(), action_onehot])
        
        # Store activations for backprop
        activations = [x]
        pre_activations = []
        
        # Hidden layers with tanh
        for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            z = activations[-1] @ w + b
            pre_activations.append(z)
            a = np.tanh(z)
            activations.append(a)
        
        # Output layer (linear - no activation)
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        output = z  # Linear output
        activations.append(output)
        
        return output, activations, pre_activations
    
    def predict(self, obs: np.ndarray, action: int) -> np.ndarray:
        """Predict next observation (inference only)."""
        output, _, _ = self.forward(obs, action)
        return output
    
    def train_step(self, obs: np.ndarray, action: int, target: np.ndarray) -> float:
        """
        Single training step with backpropagation.
        
        Returns:
            MSE loss
        """
        # Forward pass
        predicted, activations, pre_activations = self.forward(obs, action)
        
        # Compute MSE loss
        error = predicted - target
        loss = np.mean(error ** 2)
        
        # Backward pass
        # Output layer gradient (linear activation → gradient = error)
        delta = error * (2.0 / len(error))  # MSE gradient
        
        dw = np.outer(activations[-2], delta)
        db = delta
        
        # Apply momentum SGD
        self.weight_velocity[-1] = (
            self.momentum * self.weight_velocity[-1] - 
            self.learning_rate * dw
        )
        self.bias_velocity[-1] = (
            self.momentum * self.bias_velocity[-1] - 
            self.learning_rate * db
        )
        
        self.weights[-1] += self.weight_velocity[-1]
        self.biases[-1] += self.bias_velocity[-1]
        
        # Backprop through hidden layers
        for l in reversed(range(len(self.weights) - 1)):
            # Gradient through tanh: (1 - tanh²(z))
            delta = (delta @ self.weights[l + 1].T) * (1 - np.tanh(pre_activations[l]) ** 2)
            
            dw = np.outer(activations[l], delta)
            db = delta
            
            self.weight_velocity[l] = (
                self.momentum * self.weight_velocity[l] - 
                self.learning_rate * dw
            )
            self.bias_velocity[l] = (
                self.momentum * self.bias_velocity[l] - 
                self.learning_rate * db
            )
            
            self.weights[l] += self.weight_velocity[l]
            self.biases[l] += self.bias_velocity[l]
        
        return loss
    
    def train_batch(self, experiences: List) -> float:
        """Train on batch of experiences."""
        if len(experiences) == 0:
            return 0.0
        
        total_loss = 0.0
        for exp in experiences:
            loss = self.train_step(
                exp.observation,
                exp.action,
                exp.next_observation
            )
            total_loss += loss
        
        avg_loss = total_loss / len(experiences)
        self.training_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, experiences: List) -> dict:
        """Evaluate prediction accuracy on experiences."""
        if len(experiences) == 0:
            return {'mse': 0.0, 'mae': 0.0, 'per_feature_mse': []}
        
        predictions = []
        targets = []
        
        for exp in experiences:
            pred = self.predict(exp.observation, exp.action)
            predictions.append(pred)
            targets.append(exp.next_observation)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        per_feature_mse = np.mean((predictions - targets) ** 2, axis=0)
        
        return {
            'mse': mse,
            'mae': mae,
            'per_feature_mse': per_feature_mse.tolist()
        }


class InverseDynamicsModel:
    """
    Predicts action from state transition.
    
    g(s_t, s_{t+1}) → â_t
    
    Helps identify controllable features and learn state representations.
    """
    
    def __init__(
        self,
        obs_size: int = 64,
        action_size: int = 8,
        hidden_sizes: List[int] = None,
        learning_rate: float = 0.001
    ):
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        
        self.obs_size = obs_size
        self.action_size = action_size
        self.input_size = obs_size * 2  # s_t and s_{t+1}
        self.output_size = action_size
        self.learning_rate = learning_rate
        
        # Build network
        layer_sizes = [self.input_size] + hidden_sizes + [self.output_size]
        self.weights, self.biases = self._init_weights(layer_sizes)
    
    def _init_weights(self, layer_sizes):
        weights = []
        biases = []
        for i in range(len(layer_sizes) - 1):
            std = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * std)
            biases.append(np.zeros(layer_sizes[i + 1]))
        return weights, biases
    
    def predict(self, obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        """Predict action probabilities."""
        x = np.concatenate([obs.flatten(), next_obs.flatten()])
        
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(x @ w + b)
        
        # Softmax output
        logits = x @ self.weights[-1] + self.biases[-1]
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        
        return probs
    
    def train_batch(self, experiences: List) -> float:
        """Train with cross-entropy loss."""
        total_loss = 0.0
        
        for exp in experiences:
            probs = self.predict(exp.observation, exp.next_observation)
            loss = -np.log(probs[exp.action] + 1e-8)
            total_loss += loss
            
            # Simplified gradient update (output layer only for speed)
            # Full backprop can be implemented similarly to ForwardModel
            grad = probs.copy()
            grad[exp.action] -= 1.0
            
            # Update output layer
            x = np.concatenate([exp.observation.flatten(), exp.next_observation.flatten()])
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                x = np.tanh(x @ w + b)
            
            dw = np.outer(x, grad) * self.learning_rate
            db = grad * self.learning_rate
            
            self.weights[-1] -= dw
            self.biases[-1] -= db
        
        return total_loss / len(experiences) if experiences else 0.0


class WorldModel:
    """
    Combined world model with forward and inverse dynamics.
    
    Provides unified interface for:
    - State prediction
    - Action inference
    - Curiosity computation
    - Model-based planning
    """
    
    def __init__(
        self,
        obs_size: int = 64,
        action_size: int = 8,
        forward_hidden: List[int] = None,
        inverse_hidden: List[int] = None,
        learning_rate: float = 0.001,
        curiosity_weight: float = 0.05
    ):
        if forward_hidden is None:
            forward_hidden = [128, 128, 64]
        if inverse_hidden is None:
            inverse_hidden = [128, 64]
        
        self.forward_model = ForwardDynamicsModel(
            obs_size, action_size, forward_hidden, learning_rate
        )
        self.inverse_model = InverseDynamicsModel(
            obs_size, action_size, inverse_hidden, learning_rate
        )
        
        self.curiosity_weight = curiosity_weight
        self.prediction_errors = []
        self.enabled = False
    
    def predict_next_state(self, obs: np.ndarray, action: int) -> np.ndarray:
        """Predict next observation."""
        return self.forward_model.predict(obs, action)
    
    def predict_action(self, obs: np.ndarray, next_obs: np.ndarray) -> int:
        """Predict which action caused transition."""
        probs = self.inverse_model.predict(obs, next_obs)
        return np.argmax(probs)
    
    def compute_curiosity(
        self, 
        obs: np.ndarray, 
        action: int, 
        next_obs: np.ndarray
    ) -> float:
        """
        Compute curiosity reward based on prediction error.
        
        High error = surprising = high curiosity reward
        """
        predicted = self.forward_model.predict(obs, action)
        error = np.mean((predicted - next_obs) ** 2)
        
        # Track for normalization
        self.prediction_errors.append(error)
        if len(self.prediction_errors) > 1000:
            self.prediction_errors.pop(0)
        
        # Normalize
        mean_error = np.mean(self.prediction_errors)
        std_error = np.std(self.prediction_errors) + 1e-8
        normalized = (error - mean_error) / std_error
        
        # Clip and scale
        curiosity = self.curiosity_weight * np.clip(normalized, -2, 2)
        
        return curiosity
    
    def train(self, experiences: List) -> Tuple[float, float]:
        """
        Train both models.
        
        Returns:
            (forward_loss, inverse_loss)
        """
        forward_loss = self.forward_model.train_batch(experiences)
        inverse_loss = self.inverse_model.train_batch(experiences)
        return forward_loss, inverse_loss
    
    def get_metrics(self) -> dict:
        """Get training metrics."""
        return {
            'forward_losses': self.forward_model.training_losses[-100:],
            'mean_prediction_error': np.mean(self.prediction_errors[-100:]) if self.prediction_errors else 0,
            'curiosity_weight': self.curiosity_weight
        }
```

### 5.2 Modified: `agents/learning.py`

Add world model integration:

```python
# At top of file, add import:
from agents.world_model import WorldModel

# Modify AgentLearner class:

class AgentLearner:
    def __init__(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        batch_size: int = 32,
        buffer_capacity: int = 1000,
        # NEW: World model parameters
        world_model_enabled: bool = False,
        curiosity_weight: float = 0.05
    ):
        # Existing initialization...
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.reward_shaper = RewardShaper()
        
        # NEW: World model
        self.world_model = None
        if world_model_enabled:
            self.world_model = WorldModel(
                obs_size=64,
                action_size=8,
                curiosity_weight=curiosity_weight
            )
            self.world_model.enabled = True
    
    def get_curiosity_reward(
        self, 
        obs: np.ndarray, 
        action: int, 
        next_obs: np.ndarray
    ) -> float:
        """Get intrinsic curiosity reward."""
        if self.world_model is None or not self.world_model.enabled:
            return 0.0
        return self.world_model.compute_curiosity(obs, action, next_obs)
    
    def learn(self, brain) -> Tuple[float, float, float]:
        """
        Update policy and world model.
        
        Returns:
            (policy_loss, world_model_loss, inverse_loss)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
        
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Existing policy learning
        policy_loss = self._policy_gradient_update(brain, experiences)
        
        # NEW: World model learning
        wm_loss, inv_loss = 0.0, 0.0
        if self.world_model is not None and self.world_model.enabled:
            wm_loss, inv_loss = self.world_model.train(experiences)
        
        return policy_loss, wm_loss, inv_loss
    
    def _policy_gradient_update(self, brain, experiences) -> float:
        """Existing policy gradient code..."""
        # ... keep existing implementation ...
```

### 5.3 Modified: `config/training_easy.yaml`

Add world model configuration:

```yaml
# World Model Settings - NEW!
world_model:
  enabled: false  # Set via --world-model flag
  
  # Forward dynamics model
  forward_model:
    hidden_layers: [128, 128, 64]
    learning_rate: 0.001
    update_frequency: 5  # Train every N ticks
  
  # Inverse dynamics model
  inverse_model:
    hidden_layers: [128, 64]
    learning_rate: 0.001
  
  # Curiosity settings
  curiosity:
    enabled: true
    weight: 0.05  # Scales curiosity reward
    decay_rate: 0.999  # Reduces curiosity over time
    normalize: true  # Normalize prediction errors
  
  # Model-based planning
  planning:
    enabled: false  # Use for decision making
    horizon: 10  # Steps to plan ahead
    num_trajectories: 50  # Rollouts for random shooting
    method: "simple_lookahead"  # "random_shooting", "cem", "simple_lookahead"
```

### 5.4 Modified: `main.py`

Add world model command-line flag and setup:

```python
# Add argument
parser.add_argument(
    '--world-model',
    action='store_true',
    help='Enable world model for curiosity-driven learning'
)

# In main():
if args.world_model:
    print("\n🧠 World Model ENABLED")
    print(f"   Curiosity weight: {config['world_model']['curiosity']['weight']}")
    print(f"   Planning: {config['world_model']['planning']['enabled']}")
    
    # Enable on all agents
    for agent in world.agents.values():
        if hasattr(agent, 'learner') and agent.learner is not None:
            agent.learner.world_model.enabled = True
```

---

## 6. Configuration & Testing

### 6.1 Test File: `tests/test_world_model.py`

```python
"""
Unit tests for World Model components.
"""

import pytest
import numpy as np
from agents.world_model import ForwardDynamicsModel, InverseDynamicsModel, WorldModel


class TestForwardDynamicsModel:
    """Tests for forward dynamics prediction."""
    
    def test_initialization(self):
        """Test model initializes correctly."""
        model = ForwardDynamicsModel()
        
        assert model.input_size == 72  # 64 obs + 8 action
        assert model.output_size == 64
        assert len(model.weights) == 4  # 3 hidden + 1 output
    
    def test_predict_shape(self):
        """Test prediction returns correct shape."""
        model = ForwardDynamicsModel()
        
        obs = np.random.randn(64)
        action = 3
        
        predicted = model.predict(obs, action)
        
        assert predicted.shape == (64,)
    
    def test_training_reduces_loss(self):
        """Test that training reduces prediction error."""
        model = ForwardDynamicsModel(learning_rate=0.01)
        
        # Create synthetic transition
        obs = np.random.randn(64)
        action = 5
        next_obs = obs + 0.1  # Simple deterministic transition
        
        # Measure initial error
        initial_pred = model.predict(obs, action)
        initial_error = np.mean((initial_pred - next_obs) ** 2)
        
        # Train on this transition multiple times
        class FakeExperience:
            def __init__(self, o, a, no):
                self.observation = o
                self.action = a
                self.next_observation = no
        
        exp = FakeExperience(obs, action, next_obs)
        for _ in range(100):
            model.train_batch([exp])
        
        # Measure final error
        final_pred = model.predict(obs, action)
        final_error = np.mean((final_pred - next_obs) ** 2)
        
        assert final_error < initial_error


class TestInverseDynamicsModel:
    """Tests for inverse dynamics prediction."""
    
    def test_predict_probabilities(self):
        """Test inverse model outputs valid probabilities."""
        model = InverseDynamicsModel()
        
        obs = np.random.randn(64)
        next_obs = np.random.randn(64)
        
        probs = model.predict(obs, next_obs)
        
        assert probs.shape == (8,)
        assert np.isclose(np.sum(probs), 1.0)
        assert np.all(probs >= 0)


class TestWorldModel:
    """Tests for combined world model."""
    
    def test_curiosity_computation(self):
        """Test curiosity reward is computed."""
        wm = WorldModel(curiosity_weight=0.1)
        
        obs = np.random.randn(64)
        action = 2
        next_obs = np.random.randn(64)  # Random = high error
        
        curiosity = wm.compute_curiosity(obs, action, next_obs)
        
        assert isinstance(curiosity, float)
    
    def test_combined_training(self):
        """Test both models train together."""
        wm = WorldModel()
        
        class FakeExperience:
            def __init__(self):
                self.observation = np.random.randn(64)
                self.action = np.random.randint(0, 8)
                self.next_observation = np.random.randn(64)
        
        experiences = [FakeExperience() for _ in range(32)]
        
        fwd_loss, inv_loss = wm.train(experiences)
        
        assert fwd_loss >= 0
        assert inv_loss >= 0
```

### 6.2 Run Tests

```bash
pytest tests/test_world_model.py -v
```

---

## 7. Expected Outcomes

### 7.1 Success Criteria

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| State prediction error | N/A | < 10% MSE | New capability |
| Exploration coverage | ~30% world | 70%+ world | +40% |
| Sample efficiency | 2500 ticks to survive | 1500 ticks | 40% faster |
| Survival rate | 30-40% | 50%+ | +15-20% |
| Novel behaviors | Reactive only | Predictive | Qualitative |

### 7.2 Emergent Behaviors Expected

1. **Anticipatory Food Finding**
   - Agents move toward predicted food locations
   - Plan routes through fertile areas

2. **Danger Avoidance**
   - Predict low-energy states → avoid actions that lead there
   - Learn to avoid areas prone to calamities

3. **Efficient Exploration**
   - Curiosity drives exploration of unvisited areas
   - Balance exploitation vs exploration naturally

4. **Resource Planning**
   - Predict food decay → eat before spoilage
   - Plan seed planting for future food

### 7.3 Implementation Priority

```
Week 1: Forward Dynamics Model
        ├── Create agents/world_model.py
        ├── Implement ForwardDynamicsModel class
        ├── Integrate with AgentLearner
        └── Add basic tests

Week 2: Inverse Model + Curiosity
        ├── Implement InverseDynamicsModel
        ├── Create curiosity reward system
        ├── Integrate with RewardShaper
        └── Test curiosity-driven exploration

Week 3: Model-Based Planning
        ├── Implement simple lookahead
        ├── Add random shooting planner
        ├── Integrate with agent decision-making
        └── Compare model-free vs model-based

Week 4: Evaluation & Tuning
        ├── Run comparison experiments
        ├── Tune hyperparameters
        ├── Document results
        └── Optimize performance
```

---

## Appendix A: Current Codebase Statistics

```
Files analyzed:
├── agents/
│   ├── agent.py        - 705 lines (core agent logic)
│   ├── brain.py        - 217 lines (neural network)
│   ├── genome.py       - 299 lines (genetic encoding)
│   ├── learning.py     - 454 lines (RL system)
│   ├── observation.py  - 263 lines (state encoding)
│   ├── actions.py      - 50 lines (action definitions)
│   └── evolution.py    - 373 lines (reproduction)
├── world/
│   ├── world.py        - 525 lines (simulation)
│   ├── systems.py      - 675 lines (update systems)
│   └── ...
└── Total: ~6,800+ lines
```

## Appendix B: Key Integration Points Summary

| Component | File | Line | Integration |
|-----------|------|------|-------------|
| Experience Storage | learning.py | 42-52 | Already stores s, a, s' |
| Training Loop | learning.py | 275-360 | Add WM training |
| Reward Shaping | learning.py | 96-195 | Add curiosity reward |
| Decision Making | agent.py | 200-230 | Add planning option |
| Configuration | config/*.yaml | New section | Add world_model block |

---

**Document Version:** 1.0  
**Last Updated:** November 28, 2025  
**Next Steps:** Begin Phase 3.1 - Forward Dynamics Model
