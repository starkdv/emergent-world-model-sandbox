"""
Train a population world model from logged transitions and save it.

This is the focused "train the world model" step of the dream-model loop
(`agents/dream.py`): it loads transitions produced by a headless run with
``--world-model-log``, fits a :class:`PopulationWorldModel`
(``f(obs, a) → (Δobs, r̂, done?)``), reports held-out accuracy, and saves the
model to a ``.pt`` file.

Unlike ``scripts/dream_evolve.py`` (which also runs dream *evolution* and
assumes 8 actions), this script sizes the model from the config's **brain
version** so it works for **Brain v3.5** too: 78-dim observations and **9**
actions (the SIGNAL action). The action count is also widened to cover any
action index actually seen in the data, so it never index-errors on SIGNAL.

Usage:
    python scripts/train_world_model.py \
        --transitions data/logs/transitions_*.csv \
        --config config/worldmodel_v35.yaml \
        --out data/world_models/world_model_v35.pt \
        --report data/world_models/world_model_v35.txt

Author: Karan Vasa
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _obs_dim_and_actions(config: dict) -> tuple[int, int]:
    """Resolve (obs_dim, n_actions) from the config's brain version."""
    from agents.brain import _is_v35
    from agents.brain.spec import set_observation_version, get_active_observation_spec

    brain_cfg = config.get("brain", {"version": 3})
    version = brain_cfg.get("version", 3)
    # Activate the matching observation layout so the size is authoritative.
    set_observation_version(2 if _is_v35(version) else 1)
    obs_dim = get_active_observation_spec().size
    if _is_v35(version):
        n_actions = 9  # v3.5 adds the SIGNAL action
    else:
        n_actions = int(brain_cfg.get("output_size", 8))
    return obs_dim, n_actions


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train + save a population world model")
    p.add_argument(
        "--transitions", required=True, help="CSV glob from --world-model-log"
    )
    p.add_argument("--config", default="config/worldmodel_v35.yaml")
    p.add_argument("--out", default="data/world_models/world_model.pt")
    p.add_argument("--report", default=None, help="Write a text training report here")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1, help="held-out fraction")
    p.add_argument("--max-rows", type=int, default=0, help="cap rows (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    try:
        import torch
    except Exception:
        print("Error: torch is required to train the world model", file=sys.stderr)
        return 1

    from agents.dream import (
        PopulationWorldModel,
        TransitionDataset,
        load_transitions_csv,
    )

    config = yaml.safe_load(open(args.config))
    obs_dim, n_actions = _obs_dim_and_actions(config)
    print(f"Config {args.config}: obs_dim={obs_dim}, n_actions={n_actions}")

    paths = sorted(glob.glob(args.transitions))
    if not paths:
        print(f"Error: no files match {args.transitions}", file=sys.stderr)
        return 1
    print(f"Loading {len(paths)} transition file(s)...")
    datasets = [load_transitions_csv(pth, obs_dim=obs_dim) for pth in paths]
    obs = np.concatenate([d.obs for d in datasets])
    actions = np.concatenate([d.actions for d in datasets])
    rewards = np.concatenate([d.rewards for d in datasets])
    next_obs = np.concatenate([d.next_obs for d in datasets])
    dones = np.concatenate([d.dones for d in datasets])

    n_total = len(actions)
    if n_total == 0:
        print("Error: no usable transitions parsed", file=sys.stderr)
        return 1

    # Widen the action space if the data contains a higher index (safety).
    seen_actions = int(actions.max()) + 1 if n_total else 0
    if seen_actions > n_actions:
        print(f"  widening n_actions {n_actions} -> {seen_actions} (seen in data)")
        n_actions = seen_actions

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    if args.max_rows and args.max_rows < n_total:
        perm = perm[: args.max_rows]
    obs, actions, rewards, next_obs, dones = (
        obs[perm],
        actions[perm],
        rewards[perm],
        next_obs[perm],
        dones[perm],
    )
    n = len(actions)
    n_val = int(n * args.val_frac)
    n_train = n - n_val
    print(f"Transitions: {n_total} total, using {n} (train {n_train} / val {n_val})")
    print(f"  action histogram: {np.bincount(actions, minlength=n_actions).tolist()}")
    print(f"  reward mean/std: {rewards.mean():.4f} / {rewards.std():.4f}")
    print(f"  done rate: {dones.mean():.4f}")

    train = TransitionDataset(
        obs[:n_train],
        actions[:n_train],
        rewards[:n_train],
        next_obs[:n_train],
        dones[:n_train],
    )

    # --- train ---
    print(f"\nTraining world model: {args.epochs} epochs, hidden={args.hidden}")
    model = PopulationWorldModel(
        obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden, lr=args.lr
    )
    losses = model.fit(
        train, epochs=args.epochs, batch_size=args.batch_size, verbose=True
    )

    # --- held-out evaluation (batched forward, no grad) ---
    val_report = {}
    if n_val > 0:
        with torch.no_grad():
            vobs = torch.as_tensor(obs[n_train:], dtype=torch.float32)
            vact = torch.nn.functional.one_hot(
                torch.as_tensor(actions[n_train:]), num_classes=n_actions
            ).float()
            d_pred, r_pred, done_logit = model._forward(vobs, vact)
            d_true = torch.as_tensor(next_obs[n_train:] - obs[n_train:])
            delta_mse = float(torch.mean((d_pred - d_true) ** 2))
            # baseline: predicting Δobs = 0 (a mostly-static world is the bar)
            baseline_mse = float(torch.mean(d_true**2))
            r_mse = float(
                torch.mean(
                    (r_pred.squeeze(-1) - torch.as_tensor(rewards[n_train:])) ** 2
                )
            )
            done_pred = (torch.sigmoid(done_logit.squeeze(-1)) > 0.5).float()
            done_acc = float(
                torch.mean((done_pred == torch.as_tensor(dones[n_train:])).float())
            )
        val_report = {
            "delta_mse": delta_mse,
            "baseline_delta_mse": baseline_mse,
            "reward_mse": r_mse,
            "done_acc": done_acc,
        }
        print(
            f"\nHeld-out: Δobs MSE {delta_mse:.6f} (baseline Δ=0: {baseline_mse:.6f}), "
            f"reward MSE {r_mse:.4f}, done acc {done_acc:.3f}"
        )

    # --- save ---
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWorld model saved: {args.out} ({size_kb:.1f} KB)")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("Population World Model — training report\n")
            f.write("=" * 50 + "\n")
            f.write(f"config:        {args.config}\n")
            f.write(f"obs_dim:       {obs_dim}\n")
            f.write(f"n_actions:     {n_actions}\n")
            f.write(f"hidden:        {args.hidden}\n")
            f.write(f"epochs:        {args.epochs}\n")
            f.write(f"transitions:   {n_total} total, trained on {n_train}\n")
            f.write(f"reward mean/std: {rewards.mean():.4f} / {rewards.std():.4f}\n")
            f.write(f"done rate:     {dones.mean():.4f}\n")
            f.write(
                "action histogram: "
                f"{np.bincount(actions, minlength=n_actions).tolist()}\n\n"
            )
            f.write("training loss per epoch (MSE Δobs + MSE reward + BCE done):\n")
            for i, lo in enumerate(losses):
                f.write(f"  epoch {i + 1:>3}: {lo:.6f}\n")
            if val_report:
                f.write("\nheld-out evaluation:\n")
                f.write(f"  Δobs MSE:          {val_report['delta_mse']:.6f}\n")
                f.write(
                    f"  baseline (Δ=0):    {val_report['baseline_delta_mse']:.6f}\n"
                )
                f.write(f"  reward MSE:        {val_report['reward_mse']:.4f}\n")
                f.write(f"  done accuracy:     {val_report['done_acc']:.3f}\n")
        print(f"Report written: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
