"""
Reinforcement learning system for agents.

This module implements Actor-Critic reinforcement learning that allows
agents to improve their neural network weights during their lifetime based
on rewards from the environment.

Key features:
- Actor-Critic learning with GRU hidden states
- TD advantage estimation
- Policy gradient + value function learning
- Entropy regularization for exploration
- Online learning (agents improve during lifetime)
- Knowledge transfer to offspring through evolved weights

Author: Karan Vasa
Date: November 15, 2025
"""

import random
import numpy as np
from typing import Tuple, TYPE_CHECKING

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

# Import utility classes from utils.agents
from utils.agents import Experience, ReplayBuffer, RewardShaper

if TYPE_CHECKING:
    from agents.brain import Brain
    from agents.agent import Agent


class AgentLearner:
    """
    Actor-Critic learning system for agent neural networks.
    
    Implements Actor-Critic algorithm with GRU hidden states:
    - Policy gradient for action selection (Actor)
    - Value function for advantage estimation (Critic)
    - TD advantage for variance reduction
    - Entropy regularization for exploration
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        batch_size: int = 32,
        buffer_capacity: int = 1000,
        entropy_coef: float = 0.01,  # Entropy bonus coefficient
        compute_backend: str = "auto",
        compute_device: str = "auto",
    ):
        """
        Initialize the learner.
        
        Args:
            learning_rate: Learning rate for gradient updates
            discount_factor: Discount factor for future rewards
            batch_size: Batch size for learning updates
            buffer_capacity: Size of experience replay buffer
            entropy_coef: Coefficient for entropy bonus (encourages exploration)
            compute_backend: 'auto', 'numpy', or 'torch'
            compute_device: 'auto', 'cpu', 'cuda', or 'mps' (when using torch)
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.reward_shaper = RewardShaper()

        self.compute_backend, self.compute_device = self._resolve_compute_backend(
            compute_backend,
            compute_device,
        )

    def _resolve_compute_backend(self, compute_backend: str, compute_device: str) -> tuple[str, str]:
        """
        Resolve compute backend/device with safe fallbacks.
        """
        backend = (compute_backend or "auto").lower()
        device = (compute_device or "auto").lower()

        if backend not in {"auto", "numpy", "torch"}:
            backend = "auto"

        if backend == "numpy":
            return "numpy", "cpu"

        if backend == "torch" and not TORCH_AVAILABLE:
            return "numpy", "cpu"

        if backend == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available() and self._torch_device_usable("cuda"):
                return "torch", "cuda"
            if (
                TORCH_AVAILABLE
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and self._torch_device_usable("mps")
            ):
                return "torch", "mps"
            return "numpy", "cpu"

        # backend == "torch"
        if device == "auto":
            if torch.cuda.is_available() and self._torch_device_usable("cuda"):
                return "torch", "cuda"
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and self._torch_device_usable("mps")
            ):
                return "torch", "mps"
            return "torch", "cpu"

        if device == "cuda" and (not torch.cuda.is_available() or not self._torch_device_usable("cuda")):
            return "torch", "cpu"

        if device == "mps":
            if not (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and self._torch_device_usable("mps")
            ):
                return "torch", "cpu"

        return "torch", device

    def _torch_device_usable(self, device_name: str) -> bool:
        """
        Check whether torch operations can actually execute on the device.

        Some environments report CUDA available but fail at kernel execution
        due to driver / architecture mismatch.
        """
        if not TORCH_AVAILABLE:
            return False
        try:
            device = torch.device(device_name)
            a = torch.tensor([1.0], device=device)
            b = torch.tensor([2.0], device=device)
            _ = (a + b).item()
            return True
        except Exception:
            return False
    
    def store_experience(
        self,
        observation: np.ndarray,
        hidden_state: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_hidden_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store an experience in the replay buffer.
        
        Args:
            observation: State before action
            hidden_state: Hidden state before action
            action: Action taken
            reward: Reward received
            next_observation: State after action
            next_hidden_state: Hidden state after action
            done: Whether episode ended
        """
        experience = Experience(
            observation, hidden_state, action, reward,
            next_observation, next_hidden_state, done
        )
        self.replay_buffer.add(experience)
    
    def learn(self, brain: 'Brain') -> float:
        """
        Update brain weights using Actor-Critic learning.
        
        Computes:
        - TD advantage: A = r + γV(s') * (1-done) - V(s)
        - Policy loss: -log π(a|s) * A
        - Value loss: 0.5 * (V(s) - target)^2
        - Entropy bonus: -β * H(π)
        
        Args:
            brain: Agent's brain to update
            
        Returns:
            Average loss for this update
        """        
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        experiences = self.replay_buffer.sample(self.batch_size)

        if self.compute_backend == "torch" and TORCH_AVAILABLE:
            total_loss, total_policy_loss, total_value_loss, total_entropy = self._learn_vectorized_torch(brain, experiences)
        else:
            total_loss, total_policy_loss, total_value_loss, total_entropy = self._learn_vectorized_numpy(brain, experiences)
        
        # Sync updated parameters back to genome
        self._sync_genome_weights(brain)
        
        # Debug logging occasionally
        if random.random() < 0.01:
            avg_policy = total_policy_loss / len(experiences)
            avg_value = total_value_loss / len(experiences)
            avg_entropy = total_entropy / len(experiences)
            print(f"  [LEARN] Policy: {avg_policy:.3f}, Value: {avg_value:.3f}, Entropy: {avg_entropy:.3f}")
        
        return total_loss / len(experiences)
    
    def _forward_batch_numpy(
        self,
        brain: 'Brain',
        obs_batch: np.ndarray,
        h_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batched forward pass using numpy arrays.
        """
        x = obs_batch

        for i in range(len(brain.params['encoder_weights'])):
            x = np.tanh(x @ brain.params['encoder_weights'][i] + brain.params['encoder_biases'][i])

        gru = brain.params['gru']
        r = 1.0 / (1.0 + np.exp(-np.clip(x @ gru['Wr_input'] + h_batch @ gru['Wr_hidden'] + gru['br'], -20, 20)))
        z = 1.0 / (1.0 + np.exp(-np.clip(x @ gru['Wz_input'] + h_batch @ gru['Wz_hidden'] + gru['bz'], -20, 20)))
        h_tilde = np.tanh(x @ gru['Wh_input'] + (r * h_batch) @ gru['Wh_hidden'] + gru['bh'])
        h_next = (1 - z) * h_batch + z * h_tilde

        logits = h_next @ brain.params['policy_head']['W'] + brain.params['policy_head']['b']
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        values = (h_next @ brain.params['value_head']['W'] + brain.params['value_head']['b']).squeeze(-1)
        return probs, values, h_next

    def _learn_vectorized_numpy(self, brain: 'Brain', experiences: list[Experience]) -> tuple[float, float, float, float]:
        """
        Vectorized numpy learning step over a full batch.
        """
        obs = np.stack([exp.observation for exp in experiences], axis=0)
        hidden = np.stack([exp.hidden_state for exp in experiences], axis=0)
        actions = np.array([exp.action for exp in experiences], dtype=np.int64)
        rewards = np.array([exp.reward for exp in experiences], dtype=np.float32)
        next_obs = np.stack([exp.next_observation for exp in experiences], axis=0)
        next_hidden = np.stack([exp.next_hidden_state for exp in experiences], axis=0)
        dones = np.array([1 if exp.done else 0 for exp in experiences], dtype=np.float32)

        probs, values, h_out = self._forward_batch_numpy(brain, obs, hidden)
        _, next_values, _ = self._forward_batch_numpy(brain, next_obs, next_hidden)
        next_values = next_values * (1.0 - dones)

        td_target = rewards + self.discount_factor * next_values
        advantage = td_target - values

        batch_idx = np.arange(len(experiences))
        selected = np.clip(probs[batch_idx, actions], 1e-8, 1.0)
        policy_loss = -np.log(selected) * advantage
        value_loss = 0.5 * (values - td_target) ** 2
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1)
        loss = policy_loss + value_loss - self.entropy_coef * entropy

        action_gradient = probs.copy()
        action_gradient[batch_idx, actions] -= 1.0
        action_gradient *= (advantage * self.learning_rate)[:, None]

        brain.params['policy_head']['W'] -= h_out.T @ action_gradient
        brain.params['policy_head']['b'] -= np.sum(action_gradient, axis=0)

        value_gradient = 2.0 * (values - td_target) * self.learning_rate
        dW_value = np.sum(h_out * value_gradient[:, None], axis=0).reshape(-1, 1)
        dB_value = np.array([np.sum(value_gradient)], dtype=brain.params['value_head']['b'].dtype)

        brain.params['value_head']['W'] -= dW_value
        brain.params['value_head']['b'] -= dB_value

        return float(np.sum(np.abs(loss))), float(np.sum(np.abs(policy_loss))), float(np.sum(value_loss)), float(np.sum(entropy))

    def _forward_batch_torch(self, brain: 'Brain', obs_batch, h_batch, device):
        """
        Batched forward pass using torch tensors.
        """
        x = obs_batch

        for i in range(len(brain.params['encoder_weights'])):
            w = torch.as_tensor(brain.params['encoder_weights'][i], dtype=torch.float32, device=device)
            b = torch.as_tensor(brain.params['encoder_biases'][i], dtype=torch.float32, device=device)
            x = torch.tanh(x @ w + b)

        gru = brain.params['gru']
        wr_i = torch.as_tensor(gru['Wr_input'], dtype=torch.float32, device=device)
        wr_h = torch.as_tensor(gru['Wr_hidden'], dtype=torch.float32, device=device)
        br = torch.as_tensor(gru['br'], dtype=torch.float32, device=device)
        wz_i = torch.as_tensor(gru['Wz_input'], dtype=torch.float32, device=device)
        wz_h = torch.as_tensor(gru['Wz_hidden'], dtype=torch.float32, device=device)
        bz = torch.as_tensor(gru['bz'], dtype=torch.float32, device=device)
        wh_i = torch.as_tensor(gru['Wh_input'], dtype=torch.float32, device=device)
        wh_h = torch.as_tensor(gru['Wh_hidden'], dtype=torch.float32, device=device)
        bh = torch.as_tensor(gru['bh'], dtype=torch.float32, device=device)

        r = torch.sigmoid(torch.clamp(x @ wr_i + h_batch @ wr_h + br, -20, 20))
        z = torch.sigmoid(torch.clamp(x @ wz_i + h_batch @ wz_h + bz, -20, 20))
        h_tilde = torch.tanh(x @ wh_i + (r * h_batch) @ wh_h + bh)
        h_next = (1 - z) * h_batch + z * h_tilde

        pw = torch.as_tensor(brain.params['policy_head']['W'], dtype=torch.float32, device=device)
        pb = torch.as_tensor(brain.params['policy_head']['b'], dtype=torch.float32, device=device)
        logits = h_next @ pw + pb
        probs = torch.softmax(logits, dim=1)

        vw = torch.as_tensor(brain.params['value_head']['W'], dtype=torch.float32, device=device)
        vb = torch.as_tensor(brain.params['value_head']['b'], dtype=torch.float32, device=device)
        values = (h_next @ vw + vb).squeeze(-1)
        return probs, values, h_next

    def _learn_vectorized_torch(self, brain: 'Brain', experiences: list[Experience]) -> tuple[float, float, float, float]:
        """
        Vectorized torch learning step (uses GPU when available).
        """
        device = torch.device(self.compute_device)

        obs = torch.as_tensor(np.stack([exp.observation for exp in experiences], axis=0), dtype=torch.float32, device=device)
        hidden = torch.as_tensor(np.stack([exp.hidden_state for exp in experiences], axis=0), dtype=torch.float32, device=device)
        actions = torch.as_tensor([exp.action for exp in experiences], dtype=torch.long, device=device)
        rewards = torch.as_tensor([exp.reward for exp in experiences], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(np.stack([exp.next_observation for exp in experiences], axis=0), dtype=torch.float32, device=device)
        next_hidden = torch.as_tensor(np.stack([exp.next_hidden_state for exp in experiences], axis=0), dtype=torch.float32, device=device)
        dones = torch.as_tensor([1 if exp.done else 0 for exp in experiences], dtype=torch.float32, device=device)

        probs, values, h_out = self._forward_batch_torch(brain, obs, hidden, device)
        _, next_values, _ = self._forward_batch_torch(brain, next_obs, next_hidden, device)
        next_values = next_values * (1.0 - dones)

        td_target = rewards + self.discount_factor * next_values
        advantage = td_target - values

        batch_idx = torch.arange(len(experiences), device=device)
        selected = torch.clamp(probs[batch_idx, actions], 1e-8, 1.0)
        policy_loss = -torch.log(selected) * advantage
        value_loss = 0.5 * (values - td_target) ** 2
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, 1e-8, 1.0)), dim=1)
        loss = policy_loss + value_loss - self.entropy_coef * entropy

        action_gradient = probs.clone()
        action_gradient[batch_idx, actions] -= 1.0
        action_gradient = action_gradient * (advantage * self.learning_rate).unsqueeze(1)

        dW_policy = h_out.transpose(0, 1) @ action_gradient
        dB_policy = torch.sum(action_gradient, dim=0)

        value_gradient = 2.0 * (values - td_target) * self.learning_rate
        dW_value = torch.sum(h_out * value_gradient.unsqueeze(1), dim=0).unsqueeze(1)
        dB_value = torch.sum(value_gradient).unsqueeze(0)

        brain.params['policy_head']['W'] -= dW_policy.detach().cpu().numpy()
        brain.params['policy_head']['b'] -= dB_policy.detach().cpu().numpy()
        brain.params['value_head']['W'] -= dW_value.detach().cpu().numpy()
        brain.params['value_head']['b'] -= dB_value.detach().cpu().numpy()

        return (
            float(torch.sum(torch.abs(loss)).detach().cpu().item()),
            float(torch.sum(torch.abs(policy_loss)).detach().cpu().item()),
            float(torch.sum(value_loss).detach().cpu().item()),
            float(torch.sum(entropy).detach().cpu().item()),
        )
    
    
    def _sync_genome_weights(self, brain: 'Brain') -> None:
        """
        Sync brain parameters back to genome.
        
        This ensures learned weights are stored in the genome
        and can be passed to offspring (Lamarckian inheritance).
        
        Args:
            brain: Brain with updated parameters
        """
        # Flatten all parameters back into a single vector
        flat_weights = []
        
        # 1. Encoder
        for w, b in zip(brain.params['encoder_weights'], brain.params['encoder_biases']):
            flat_weights.extend(w.flatten())
            flat_weights.extend(b.flatten())
        
        # 2. GRU (3 gates: reset, update, candidate)
        gru = brain.params['gru']

        # Reset gate
        flat_weights.extend(gru['Wr_input'].flatten())
        flat_weights.extend(gru['Wr_hidden'].flatten())
        flat_weights.extend(gru['br'].flatten())

        # Update gate
        flat_weights.extend(gru['Wz_input'].flatten())
        flat_weights.extend(gru['Wz_hidden'].flatten())
        flat_weights.extend(gru['bz'].flatten())

        # Candidate
        flat_weights.extend(gru['Wh_input'].flatten())
        flat_weights.extend(gru['Wh_hidden'].flatten())
        flat_weights.extend(gru['bh'].flatten())
        
        # 3. Policy head
        flat_weights.extend(brain.params['policy_head']['W'].flatten())
        flat_weights.extend(brain.params['policy_head']['b'].flatten())
        
        # 4. Value head
        flat_weights.extend(brain.params['value_head']['W'].flatten())
        flat_weights.extend(brain.params['value_head']['b'].flatten())
        
        # Update genome
        brain.genome.weights = np.array(flat_weights, dtype=np.float32)
