"""
Dream-based evolution CLI — evolve genomes inside a learned world model.

Closes the world-model loop end-to-end:

  1. Collect real experience:
       python main.py --no-viz --world-model-log --mode rl --learning
  2. Train a population world model on the logs and evolve in the dream:
       python scripts/dream_evolve.py --transitions data/logs/transitions_*.csv
  3. GROUND the dream champions in the real environment:
       python main.py --load-weights data/weights/dream_best.npz ...

Step 3 is not optional: dream fitness is a proxy, and evolution will
exploit world-model errors. See agents/dream.py for the methodology.

Author: Karan Vasa
Date: June 2026
"""

import argparse
import glob
import os
import sys

import numpy as np
import yaml

# Allow running from anywhere: put the repo root on sys.path so the
# agents/world/utils packages resolve (this file lives in scripts/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    """Parse arguments, train the world model, run dream evolution."""
    parser = argparse.ArgumentParser(
        description="Dream-based evolution: train a population world model "
        "from transition logs and evolve genomes inside it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/dream_evolve.py --transitions data/logs/transitions_*.csv
  python scripts/dream_evolve.py --transitions logs.csv --generations 30 --population 64
  python scripts/dream_evolve.py --transitions logs.csv --config config/default.yaml \\
      --out data/weights/dream_best.npz
        """,
    )
    parser.add_argument(
        "--transitions",
        type=str,
        required=True,
        help="Transitions CSV from --world-model-log (glob patterns allowed)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Config file whose brain section sizes the genomes",
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="World-model training epochs"
    )
    parser.add_argument("--generations", type=int, default=20, help="Dream generations")
    parser.add_argument(
        "--population", type=int, default=32, help="Genomes per generation"
    )
    parser.add_argument(
        "--episodes", type=int, default=4, help="Dream episodes per evaluation"
    )
    parser.add_argument(
        "--steps", type=int, default=64, help="Max steps per dream episode"
    )
    parser.add_argument("--mutation-std", type=float, default=0.05, help="Mutation std")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--seed-weights",
        type=str,
        default=None,
        help="Optional .npz of real-world weights to seed the population",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/weights/dream_best.npz",
        help="Output .npz (loadable via main.py --load-weights)",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Optionally save the trained world model (.pt)",
    )
    args = parser.parse_args()

    try:
        from agents.dream import (
            PopulationWorldModel,
            dream_evolution,
            load_transitions_csv,
        )
        from agents.genome import Genome, create_default_trait_config
        from agents.brain import calculate_weight_count_for_config
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.seed is not None:
        np.random.seed(args.seed)

    # ── 1. Load transitions ─────────────────────────────────────────
    paths = sorted(glob.glob(args.transitions))
    if not paths:
        print(f"Error: no files match {args.transitions}", file=sys.stderr)
        return 1

    brain_cfg = yaml.safe_load(open(args.config))["brain"]
    obs_dim = brain_cfg.get("input_size", 72)

    datasets = [load_transitions_csv(p, obs_dim=obs_dim) for p in paths]
    obs = np.concatenate([d.obs for d in datasets])
    actions = np.concatenate([d.actions for d in datasets])
    rewards = np.concatenate([d.rewards for d in datasets])
    next_obs = np.concatenate([d.next_obs for d in datasets])
    dones = np.concatenate([d.dones for d in datasets])

    from agents.dream import TransitionDataset

    data = TransitionDataset(obs, actions, rewards, next_obs, dones)
    if len(data) < 500:
        print(
            f"Warning: only {len(data)} transitions — the world model will be "
            "poor. Collect more with: python main.py --no-viz --world-model-log"
        )
    print(f"Loaded {len(data)} transitions from {len(paths)} file(s)")

    # ── 2. Train the population world model ────────────────────────
    print(f"\nTraining world model ({args.epochs} epochs)...")
    model = PopulationWorldModel(obs_dim=obs_dim)
    losses = model.fit(data, epochs=args.epochs, verbose=True)
    print(f"World model trained (loss {losses[0]:.5f} → {losses[-1]:.5f})")
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        model.save(args.save_model)
        print(f"World model saved to {args.save_model}")

    # ── 3. Seed the population ──────────────────────────────────────
    weight_count = calculate_weight_count_for_config(brain_cfg)
    initial = None
    if args.seed_weights:
        loaded = np.load(args.seed_weights)
        if "weights_0" in loaded and len(loaded["weights_0"]) == weight_count:
            initial = [
                Genome(
                    loaded["weights_0"].astype(np.float32),
                    {k: lo for k, (lo, hi) in create_default_trait_config().items()},
                )
            ]
            print(f"Seeded population with weights from {args.seed_weights}")
        else:
            print(
                "Warning: --seed-weights incompatible with the configured "
                "brain — starting from random genomes"
            )

    # ── 4. Evolve in the dream ──────────────────────────────────────
    print(
        f"\nDream evolution: {args.generations} generations × "
        f"{args.population} genomes ({args.episodes}×{args.steps}-step dreams)"
    )
    scored = dream_evolution(
        model,
        seed_observations=data.obs,
        brain_config=brain_cfg,
        initial_genomes=initial,
        population_size=args.population,
        generations=args.generations,
        mutation_std=args.mutation_std,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        verbose=True,
    )

    # ── 5. Save champions (main.py --load-weights format) ──────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_dict = {}
    for i, (fitness, genome) in enumerate(scored[:5]):
        out_dict[f"weights_{i}"] = genome.weights
        out_dict[f"fitness_{i}"] = np.array([fitness])
    np.savez(args.out, **out_dict)

    print(f"\nTop-5 dream champions saved to {args.out}")
    print(f"Best dream fitness: {scored[0][0]:.3f}")
    print(
        "\nIMPORTANT — ground the result in the real environment:\n"
        f"  python main.py --load-weights {args.out} --no-viz --seed 42\n"
        "Dream fitness is a proxy; evolution exploits model errors."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
