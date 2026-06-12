"""
Dream-based evolution — evolve genomes inside a learned world model.

The final piece of the Phase 4 world-model stack. Where the per-agent
dynamics head (agents/brain) lets ONE agent imagine, this module builds a
POPULATION-LEVEL model of the environment itself and uses it as a cheap
virtual world for evolution:

    real world (slow, ground truth)
        │  --world-model-log → transitions CSV
        ▼
    PopulationWorldModel              f(obs, a) → (Δobs, r̂, done?)
        │  thousands of imagined episodes per second
        ▼
    dream_evolution()                 evaluate → select → mutate, repeat
        │  best genomes (.npz)
        ▼
    real world again (grounding)      main.py --load-weights ...

Design notes:
- The population model is **observation-space** and policy-agnostic
  (unlike the per-agent latent head, which is tied to one genome's
  encoder) — any genome can be evaluated inside it.
- It predicts the observation *delta* rather than the next observation:
  in a mostly-static world Δobs ≈ 0, so the residual parameterisation
  concentrates learning on what actually changes.
- Dreams must be **grounded**: a model is only as good as its data, and
  evolution will exploit its errors. Re-evaluate dream champions in the
  real environment (the returned weights are deliberately in the same
  .npz format `main.py --load-weights` consumes).

CLI entry point: `python scripts/dream_evolve.py --transitions <csv>`.

Author: Karan Vasa
Date: June 2026
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment without torch
    torch = None
    TORCH_AVAILABLE = False

from agents.brain import create_brain
from agents.genome import Genome, create_default_trait_config

# ---------------------------------------------------------------------------
# Transition data
# ---------------------------------------------------------------------------


@dataclass
class TransitionDataset:
    """
    Flat arrays of logged transitions for world-model training.

    Attributes:
        obs: (N, obs_dim) observations
        actions: (N,) action indices
        rewards: (N,) rewards
        next_obs: (N, obs_dim) next observations
        dones: (N,) terminal flags
    """

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)


def load_transitions_csv(path: str, obs_dim: int = 72) -> TransitionDataset:
    """
    Load a transitions CSV produced by AsyncWorldModelLogger
    (``--world-model-log``) into training arrays.

    Args:
        path: Path to a ``transitions_*.csv`` file
        obs_dim: Observation dimensionality (number of obs_i columns)

    Returns:
        TransitionDataset with float32 arrays
    """
    import csv

    obs_list, act_list, rew_list, next_list, done_list = [], [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                obs_list.append([float(row[f"obs_{i}"]) for i in range(obs_dim)])
                next_list.append([float(row[f"obs_next_{i}"]) for i in range(obs_dim)])
                act_list.append(int(float(row["action_value"])))
                rew_list.append(float(row["reward"]))
                done_list.append(1.0 if row["done"] in ("True", "1", "true") else 0.0)
            except (KeyError, ValueError):
                continue  # skip malformed rows (e.g. from older log formats)

    return TransitionDataset(
        obs=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(act_list, dtype=np.int64),
        rewards=np.asarray(rew_list, dtype=np.float32),
        next_obs=np.asarray(next_list, dtype=np.float32),
        dones=np.asarray(done_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Population-level world model
# ---------------------------------------------------------------------------


class PopulationWorldModel:
    """
    Observation-space dynamics model shared by the whole population.

        x = [obs ‖ onehot(action)]
        hid = tanh(x·W1+b1);  hid = tanh(hid·W2+b2)
        Δobs = hid·W_o + b_o          →  next_obs = clip(obs + Δobs, 0, 1)
        r̂    = hid·W_r + b_r
        p̂_done = σ(hid·W_d + b_d)

    Trained with MSE on (Δobs, reward) and binary cross-entropy on done.
    Policy-agnostic: any genome can be rolled out inside it.
    """

    def __init__(
        self,
        obs_dim: int = 72,
        n_actions: int = 8,
        hidden: int = 128,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        """
        Initialize the model.

        Args:
            obs_dim: Observation dimensionality
            n_actions: Number of discrete actions
            hidden: Hidden layer width (two layers)
            lr: Adam learning rate
            device: torch device
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PopulationWorldModel requires torch")
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.device = torch.device(device)

        in_dim = obs_dim + n_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
        ).to(self.device)
        self.delta_head = torch.nn.Linear(hidden, obs_dim).to(self.device)
        self.reward_head = torch.nn.Linear(hidden, 1).to(self.device)
        self.done_head = torch.nn.Linear(hidden, 1).to(self.device)

        params = (
            list(self.net.parameters())
            + list(self.delta_head.parameters())
            + list(self.reward_head.parameters())
            + list(self.done_head.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _forward(self, obs: "torch.Tensor", onehot: "torch.Tensor"):
        hid = self.net(torch.cat([obs, onehot], dim=1))
        return self.delta_head(hid), self.reward_head(hid), self.done_head(hid)

    def fit(
        self,
        data: TransitionDataset,
        epochs: int = 10,
        batch_size: int = 256,
        verbose: bool = False,
    ) -> list[float]:
        """
        Train on logged transitions.

        Args:
            data: Transition dataset
            epochs: Passes over the data
            batch_size: Minibatch size
            verbose: Print per-epoch losses

        Returns:
            Mean total loss per epoch
        """
        n = len(data)
        obs = torch.as_tensor(data.obs, device=self.device)
        onehot = torch.nn.functional.one_hot(
            torch.as_tensor(data.actions, device=self.device),
            num_classes=self.n_actions,
        ).float()
        delta_target = torch.as_tensor(data.next_obs - data.obs, device=self.device)
        reward_target = torch.as_tensor(data.rewards, device=self.device)
        done_target = torch.as_tensor(data.dones, device=self.device)

        epoch_losses: list[float] = []
        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total, batches = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                d_pred, r_pred, done_logit = self._forward(obs[idx], onehot[idx])

                loss = (
                    torch.nn.functional.mse_loss(d_pred, delta_target[idx])
                    + torch.nn.functional.mse_loss(
                        r_pred.squeeze(-1), reward_target[idx]
                    )
                    + torch.nn.functional.binary_cross_entropy_with_logits(
                        done_logit.squeeze(-1), done_target[idx]
                    )
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total += float(loss.detach().cpu().item())
                batches += 1

            epoch_losses.append(total / max(1, batches))
            if verbose:
                print(f"  [WM] epoch {epoch + 1}/{epochs}  loss={epoch_losses[-1]:.5f}")
        return epoch_losses

    def predict(
        self, obs: np.ndarray, action_idx: int
    ) -> tuple[np.ndarray, float, float]:
        """
        One imagined step.

        Args:
            obs: Current observation
            action_idx: Action taken

        Returns:
            (next_obs clipped to [0,1], predicted reward, done probability)
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            onehot = torch.zeros(1, self.n_actions, device=self.device)
            onehot[0, action_idx] = 1.0
            d_pred, r_pred, done_logit = self._forward(obs_t, onehot)
            next_obs = np.clip(obs + d_pred.squeeze(0).cpu().numpy(), 0.0, 1.0).astype(
                np.float32
            )
            return (
                next_obs,
                float(r_pred.item()),
                float(torch.sigmoid(done_logit).item()),
            )

    def save(self, path: str) -> None:
        """Save model weights and sizes."""
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "hidden": self.hidden,
                "net": self.net.state_dict(),
                "delta_head": self.delta_head.state_dict(),
                "reward_head": self.reward_head.state_dict(),
                "done_head": self.done_head.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PopulationWorldModel":
        """Load a saved model."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            obs_dim=ckpt["obs_dim"],
            n_actions=ckpt["n_actions"],
            hidden=ckpt["hidden"],
            device=device,
        )
        model.net.load_state_dict(ckpt["net"])
        model.delta_head.load_state_dict(ckpt["delta_head"])
        model.reward_head.load_state_dict(ckpt["reward_head"])
        model.done_head.load_state_dict(ckpt["done_head"])
        return model


# ---------------------------------------------------------------------------
# Dream rollouts & evolution
# ---------------------------------------------------------------------------


def evaluate_in_dream(
    genome: Genome,
    model,
    seed_observations: np.ndarray,
    brain_config: Optional[dict] = None,
    episodes: int = 4,
    steps: int = 64,
    done_threshold: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Fitness of one genome measured entirely inside the world model.

    Rolls the genome's policy in imagination: the brain decides, the
    model dreams the consequence, accumulated predicted reward is the
    fitness. No action masks exist inside dreams (masks encode world
    state the model does not expose), so all actions are available —
    one of the reasons grounding in the real world stays mandatory.

    Args:
        genome: Genome to evaluate
        model: Anything with ``predict(obs, action) → (obs', r, p_done)``
        seed_observations: (M, obs_dim) pool of real start states
        brain_config: ``brain`` config for building the policy
        episodes: Dream episodes to average over
        steps: Max steps per dream episode
        done_threshold: End the episode when p_done exceeds this
        rng: Random generator for seed-state sampling

    Returns:
        Mean dream return across episodes
    """
    rng = rng or np.random.default_rng()
    brain = create_brain(genome, brain_config)
    mask = np.ones(brain.output_size, dtype=np.float32)

    total = 0.0
    for _ in range(episodes):
        obs = seed_observations[rng.integers(len(seed_observations))].copy()
        h = brain.initial_state()
        for _ in range(steps):
            action, h, _ = brain.decide(obs, h, action_mask=mask)
            obs, reward, p_done = model.predict(obs, action.value)
            total += reward
            if p_done > done_threshold:
                break
    return total / episodes


def dream_evolution(
    model,
    seed_observations: np.ndarray,
    brain_config: Optional[dict] = None,
    initial_genomes: Optional[list[Genome]] = None,
    population_size: int = 32,
    generations: int = 10,
    elite_count: int = 4,
    mutation_std: float = 0.05,
    episodes: int = 4,
    steps: int = 64,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> list[tuple[float, Genome]]:
    """
    Evolve genomes inside the learned world model.

    Classic (μ+λ) loop, but every fitness evaluation is imagined:
    evaluate → keep elites → refill with mutated elite clones → repeat.
    Orders of magnitude cheaper per evaluation than the real simulation.

    IMPORTANT: dream fitness is only a proxy. Evolution will exploit
    model errors, so champions must be re-validated in the real
    environment (``main.py --load-weights``) — that grounding step is
    deliberately left to the caller.

    Args:
        model: World model with ``predict(obs, action)``
        seed_observations: (M, obs_dim) pool of real start states
        brain_config: ``brain`` config sizing the genomes
        initial_genomes: Starting population (None → random genomes)
        population_size: Genomes per generation
        generations: Number of dream generations
        elite_count: Elites preserved unmutated each generation
        mutation_std: Gaussian mutation std for offspring
        episodes: Dream episodes per fitness evaluation
        steps: Max steps per dream episode
        seed: RNG seed for reproducibility
        verbose: Print per-generation stats

    Returns:
        Final population as (dream_fitness, genome), best first
    """
    from agents.brain import calculate_weight_count_for_config

    rng = np.random.default_rng(seed)
    trait_config = create_default_trait_config()
    weight_count = calculate_weight_count_for_config(brain_config)

    population: list[Genome] = list(initial_genomes or [])
    while len(population) < population_size:
        population.append(Genome.random(weight_count, trait_config))
    population = population[:population_size]

    scored: list[tuple[float, Genome]] = []
    for gen in range(generations):
        scored = [
            (
                evaluate_in_dream(
                    g,
                    model,
                    seed_observations,
                    brain_config,
                    episodes=episodes,
                    steps=steps,
                    rng=rng,
                ),
                g,
            )
            for g in population
        ]
        scored.sort(key=lambda fg: fg[0], reverse=True)

        if verbose:
            fits = [f for f, _ in scored]
            print(
                f"  [DREAM] gen {gen + 1}/{generations}  "
                f"best={fits[0]:.3f}  mean={np.mean(fits):.3f}"
            )

        # Elites survive unchanged; the rest are mutated elite clones
        elites = [g for _, g in scored[:elite_count]]
        next_population = list(elites)
        while len(next_population) < population_size:
            parent = elites[rng.integers(len(elites))]
            child = parent.copy()
            child.weights = child.weights + rng.normal(
                0.0, mutation_std, size=child.weights.shape
            ).astype(child.weights.dtype)
            child.generation = parent.generation + 1
            next_population.append(child)
        population = next_population

    return scored
