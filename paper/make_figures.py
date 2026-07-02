"""
Generate the paper's data-driven figures from real simulation state.

Produces (into paper/figures/):
  - ui_2d.png          top-down biome/elevation map with live entities
  - ui_iso.png         isometric voxel-style render of a world sub-region
  - worldmodel_loss.png  world-model training loss + held-out accuracy
  - population_dynamics.png  population & fitness vs tick across runs (if metrics present)

Run: python paper/make_figures.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)

# biome palette (matches the viewer): soil, rock, water, sand
BIOME = {
    0: (0.31, 0.62, 0.23),
    1: (0.54, 0.55, 0.57),
    2: (0.18, 0.44, 0.69),
    3: (0.84, 0.77, 0.54),
}


def _world_snapshot():
    from render.sim_session import session_from_config
    from render.state_bridge import world_snapshot, decode_grid

    s = session_from_config("config/default.yaml", learning=False)
    for _ in range(150):  # let plants fruit and agents spread
        s.world.update()
    snap = world_snapshot(s.world)
    t = snap["terrain"]
    W, H = t["width"], t["height"]
    elev = decode_grid(t["elevation"]).reshape(H, W).astype(float) / 255.0
    terr = decode_grid(t["terrain"]).reshape(H, W)
    return snap, W, H, elev, terr


def _rgb(terr, elev):
    """Biome colour with simple elevation shading."""
    H, W = terr.shape
    img = np.zeros((H, W, 3))
    for code, c in BIOME.items():
        img[terr == code] = c
    shade = 0.65 + 0.45 * elev[..., None]  # brighter on highlands
    return np.clip(img * shade, 0, 1)


def fig_2d(snap, W, H, elev, terr):
    img = _rgb(terr, elev)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=160)
    ax.imshow(img, origin="upper", interpolation="nearest")
    # entities
    cats = {
        "plant": ([], [], "#1f7a2e", 5, "Plant"),
        "food": ([], [], "#ff2d2d", 5, "Berry"),
        "seed": ([], [], "#f0d27a", 4, "Seed"),
    }
    for o in snap["objects"]:
        cat = o.get("category", "")
        key = (
            "plant"
            if "plant" in cat
            else "food" if "food" in cat else "seed" if "seed" in cat else None
        )
        if key:
            cats[key][0].append(o["x"])
            cats[key][1].append(o["y"])
    for xs, ys, c, sz, lab in cats.values():
        if xs:
            ax.scatter(xs, ys, s=sz, c=c, edgecolors="none", label=lab)
    ax.scatter(
        [a["x"] for a in snap["agents"]],
        [a["y"] for a in snap["agents"]],
        s=22,
        marker="o",
        facecolors="none",
        edgecolors="white",
        linewidths=0.9,
        label="Agent",
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Top-down 2-D view ({W}×{H}, tick {snap['tick']})", fontsize=9)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85, markerscale=1.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "ui_2d.png"), bbox_inches="tight")
    plt.close(fig)
    print("ui_2d.png")


def fig_iso(snap, W, H, elev, terr, crop=64):
    """Isometric voxel-style render of a crop of the world."""
    x0, y0 = (W - crop) // 2, (H - crop) // 2
    sub_e = elev[y0 : y0 + crop, x0 : x0 + crop]
    sub_t = terr[y0 : y0 + crop, x0 : x0 + crop]
    tw, th, hscale = 1.0, 0.5, 6.0
    polys, colors = [], []
    # painter's order: far (small x+y) first
    order = sorted(
        ((x, y) for y in range(crop) for x in range(crop)), key=lambda p: p[0] + p[1]
    )
    for x, y in order:
        h = max(0.06, sub_e[y, x]) * hscale
        sx = (x - y) * tw
        sy = (x + y) * th
        base = np.array(BIOME[int(sub_t[y, x])])
        top = np.clip(base * (0.7 + 0.5 * sub_e[y, x]), 0, 1)
        # top diamond
        polys.append(
            [
                (sx, sy - h),
                (sx + tw, sy + th - h),
                (sx, sy + 2 * th - h),
                (sx - tw, sy + th - h),
            ]
        )
        colors.append(top)
        # left & right faces for depth
        polys.append(
            [
                (sx - tw, sy + th - h),
                (sx, sy + 2 * th - h),
                (sx, sy + 2 * th),
                (sx - tw, sy + th),
            ]
        )
        colors.append(np.clip(top * 0.6, 0, 1))
        polys.append(
            [
                (sx, sy + 2 * th - h),
                (sx + tw, sy + th - h),
                (sx + tw, sy + th),
                (sx, sy + 2 * th),
            ]
        )
        colors.append(np.clip(top * 0.8, 0, 1))
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=160)
    ax.add_collection(
        PolyCollection(polys, facecolors=colors, edgecolors="none", linewidths=0)
    )
    # agents in the crop, as markers floating on their columns
    ax_x, ax_y = [], []
    for a in snap["agents"]:
        x, y = a["x"] - x0, a["y"] - y0
        if 0 <= x < crop and 0 <= y < crop:
            h = max(0.06, sub_e[y, x]) * hscale
            ax_x.append((x - y) * tw)
            ax_y.append((x + y) * th - h + th)
    ax.scatter(
        ax_x,
        ax_y,
        s=10,
        c="white",
        edgecolors="black",
        linewidths=0.4,
        zorder=5,
        label="Agent",
    )
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()
    ax.set_title(f"Isometric view ({crop}×{crop} crop)", fontsize=9)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "ui_iso.png"), bbox_inches="tight")
    plt.close(fig)
    print("ui_iso.png")


def fig_worldmodel_loss():
    # real per-epoch losses from docs/sample_world_model/training_report.txt
    rep = "docs/sample_world_model/training_report.txt"
    import re

    losses = []
    for line in open(rep):
        m = re.match(r"\s*epoch\s+\d+:\s*([0-9.]+)", line)  # only per-epoch lines
        if m:
            losses.append(float(m.group(1)))
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(6.6, 2.7), dpi=160, gridspec_kw={"width_ratios": [1.5, 1]}
    )
    a1.plot(range(1, len(losses) + 1), losses, "-o", ms=3, color="#1f6fb0")
    a1.set_xlabel("epoch")
    a1.set_ylabel("training loss")
    a1.set_title("World-model training", fontsize=9)
    a1.grid(alpha=0.3)
    # held-out Δobs vs baseline
    a2.bar(
        ["model", "baseline\n(Δ=0)"], [0.026816, 0.033710], color=["#2e8b57", "#bbbbbb"]
    )
    a2.set_ylabel("held-out Δobs MSE")
    a2.set_title("Next-state error\n(done acc 0.999)", fontsize=9)
    for i, v in enumerate([0.026816, 0.033710]):
        a2.text(i, v + 0.0006, f"{v:.4f}", ha="center", fontsize=7)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG, "worldmodel_loss.png"), bbox_inches="tight")
    plt.close(fig)
    print("worldmodel_loss.png")


def fig_population(_unused=None):
    """Plot the committed fine-grained population-sweep CSVs (paper/data/)."""
    import csv

    data_dir = os.path.join(os.path.dirname(FIG), "data")
    files = sorted(glob.glob(os.path.join(data_dir, "pop_founders*.csv")))
    if not files:
        print("(no paper/data/pop_founders*.csv — skipping population figure)")
        return
    cap = 60
    colors = {"08": "#d1495b", "24": "#2e8b57", "48": "#1f6fb0"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7), dpi=160)
    for f in files:
        n = os.path.basename(f).split("founders")[1].split(".")[0]
        tick, pop, fit = [], [], []
        for row in csv.DictReader(open(f)):
            tick.append(int(row["tick"]))
            pop.append(int(row["pop"]))
            fit.append(float(row["mean_fitness"]))
        c = colors.get(n, "#888")
        a1.plot(tick, pop, "-", lw=1.6, color=c, label=str(int(n)))
        a2.plot(tick, fit, "-", lw=1.6, color=c, label=str(int(n)))
    a1.axhline(cap, ls="--", lw=0.8, color="#888")
    a1.set_xlabel("tick")
    a1.set_ylabel("alive agents")
    a1.set_title("Population dynamics", fontsize=9)
    a1.grid(alpha=0.3)
    a1.legend(fontsize=6, title="founders")
    a2.set_xlabel("tick")
    a2.set_ylabel("mean fitness")
    a2.set_title("Mean fitness", fontsize=9)
    a2.grid(alpha=0.3)
    a2.legend(fontsize=6, title="founders")
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG, "population_dynamics.png"), bbox_inches="tight")
    plt.close(fig)
    print("population_dynamics.png")


def _arm_stats(path, arms, metric="peak_fitness"):
    """(mean, pstdev) per arm from a results.csv of per-seed rows."""
    import csv
    import statistics as st

    rows = list(csv.DictReader(open(path)))
    out = {}
    for a in arms:
        v = [float(r[metric]) for r in rows if r["variant"] == a]
        if v:
            out[a] = (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0)
    return out


def _bars(ax, names, stats, colors, labels):
    xs = np.arange(len(names))
    mus = [stats[n][0] for n in names]
    sds = [stats[n][1] for n in names]
    ax.bar(xs, mus, yerr=sds, capsize=4, color=[colors[n] for n in names], width=0.62)
    for x, m, s in zip(xs, mus, sds):
        ax.text(x, m + s + 1.2, f"{m:.1f}", ha="center", fontsize=7, color="#222")
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[n] for n in names], fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)


def fig_multiseed():
    """4-seed replication of the planner ladder (Sec. 6.5)."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(FIG)),
        "docs",
        "sample_planning_multiseed",
        "results.csv",
    )
    if not os.path.exists(path):
        print("(no multiseed results.csv — skipping)")
        return
    colors = {
        "shooting": "#888888",
        "policy_shooting": "#2e8b57",
        "cem": "#1f6fb0",
        "imag_off": "#888888",
        "imag_on": "#d1495b",
    }
    labels = {
        "shooting": "shooting\n(baseline)",
        "policy_shooting": "policy\nshooting (P1)",
        "cem": "CEM (P2)",
        "imag_off": "imagination\noff",
        "imag_on": "imagination\non (P3)",
    }
    stats = _arm_stats(path, list(colors))
    fig, (a1, a2) = plt.subplots(
        1,
        2,
        figsize=(6.8, 2.6),
        dpi=160,
        sharey=True,
        gridspec_kw={"width_ratios": [3, 2]},
    )
    _bars(a1, ["shooting", "policy_shooting", "cem"], stats, colors, labels)
    _bars(a2, ["imag_off", "imag_on"], stats, colors, labels)
    a1.set_ylabel("peak fitness")
    a1.set_title("Planner strategies (planner on)", fontsize=9)
    a2.set_title("Imagination A/B (planner off)", fontsize=9)
    fig.suptitle("4-seed replication, 2000 ticks (mean ± std)", fontsize=9, y=1.02)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG, "multiseed_replication.png"), bbox_inches="tight")
    plt.close(fig)
    print("multiseed_replication.png")


def fig_warmup_sweep():
    """4-seed warmup confirmation + switch-point sweep (Sec. 6.6)."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(FIG)),
        "docs",
        "sample_planning_warmup_sweep",
        "results.csv",
    )
    if not os.path.exists(path):
        print("(no warmup sweep results.csv — skipping)")
        return
    arms = ["baseline", "sched4k", "sched5k", "sched6k"]
    colors = dict(zip(arms, ["#888888", "#2e8b57", "#1f6fb0", "#d1495b"]))
    labels = {
        "baseline": "baseline\n(shooting)",
        "sched4k": "sched@4k",
        "sched5k": "sched@5k",
        "sched6k": "sched@6k",
    }
    stats = _arm_stats(path, arms)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6), dpi=160)
    _bars(a1, arms, stats, colors, labels)
    a1.set_ylabel("peak fitness")
    a1.set_title("Peak fitness by arm", fontsize=9)
    ws = [4000, 5000, 6000]
    mus = [stats[a][0] for a in arms[1:]]
    sds = [stats[a][1] for a in arms[1:]]
    a2.errorbar(
        ws,
        mus,
        yerr=sds,
        marker="o",
        ms=4,
        capsize=4,
        color="#1f6fb0",
        lw=1.6,
        label="scheduled",
    )
    a2.axhline(
        stats["baseline"][0],
        ls="--",
        lw=1.2,
        color="#888888",
        label="baseline (shooting)",
    )
    a2.set_xticks(ws)
    a2.set_xlabel("warmup_ticks (switch point)")
    a2.set_ylabel("peak fitness")
    a2.set_title("Switch-point tuning", fontsize=9)
    a2.grid(alpha=0.3)
    a2.legend(fontsize=6.5)
    fig.suptitle(
        "Warmup scheduling: 4 seeds, 7000 ticks (mean ± std)", fontsize=9, y=1.02
    )
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG, "warmup_sweep.png"), bbox_inches="tight")
    plt.close(fig)
    print("warmup_sweep.png")


def fig_rollout_error():
    """k-step open-loop rollout error over training (Sec. 6.7).

    Reads paper/data/wmq_{arm}_s{seed}.csv (columns tick, wm_rollout_error).
    """
    import csv

    data_dir = os.path.join(os.path.dirname(FIG), "data")
    arms = {
        "1step": ("#1f6fb0", "1-step loss"),
        "multistep": ("#2e8b57", "+ multi-step loss"),
    }
    fig, ax = plt.subplots(figsize=(4.6, 2.7), dpi=160)
    plotted = False
    for arm, (color, label) in arms.items():
        files = sorted(glob.glob(os.path.join(data_dir, f"wmq_{arm}_s*.csv")))
        series = []
        for f in files:
            rows = list(csv.DictReader(open(f)))
            t = [int(r["tick"]) for r in rows if r["wm_rollout_error"]]
            e = [float(r["wm_rollout_error"]) for r in rows if r["wm_rollout_error"]]
            if t:
                series.append((t, e))
                ax.plot(t, e, "-", lw=0.7, color=color, alpha=0.35)
        if series:
            ticks = series[0][0]
            mean = np.mean([e for _, e in series], axis=0)
            ax.plot(ticks, mean, "-o", ms=3, lw=1.8, color=color, label=label)
            plotted = True
    if not plotted:
        plt.close(fig)
        print("(no paper/data/wmq_*.csv — skipping rollout-error figure)")
        return
    ax.set_xlabel("world tick")
    ax.set_ylabel("open-loop latent MSE (k=3)")
    ax.set_title(
        "World-model rollout error during training\n(3 seeds; thin = per seed, thick = mean)",
        fontsize=8.5,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG, "rollout_error.png"), bbox_inches="tight")
    plt.close(fig)
    print("rollout_error.png")


if __name__ == "__main__":
    fig_worldmodel_loss()
    snap, W, H, elev, terr = _world_snapshot()
    fig_2d(snap, W, H, elev, terr)
    fig_iso(snap, W, H, elev, terr)
    pop_glob = sys.argv[1] if len(sys.argv) > 1 else "data/logs/*_metrics.csv"
    fig_population(pop_glob)
    fig_multiseed()
    fig_warmup_sweep()
    fig_rollout_error()
    print("done")
