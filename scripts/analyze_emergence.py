"""
Emergence instrument — behavioural metrics with explicit null models.

`scripts/analyze_logs.py` answers "is the simulation healthy?". This script
answers a narrower and harder question: **is anything emerging?**

Every metric here is a claim about behaviour that the code did not ask for,
and every one of them is paired with a null model or an ablation in
docs/BRAIN_V4_PROPOSAL.md §5. The metric families are:

  E1  return-to-patch        revisit rate to tiles where the agent ate,
                             against a spatially matched random-tile null
  E2  cultivation with intent  P(the planter harvests its own planted tile at
                             a latency >= the plant's maturation time), against
                             a spatially matched null
  E3  division of labour     role_MI = H(a) - E_agent[H(a|agent)] in bits:
                             the mutual information between *who* an agent is
                             and *what* it does. 0 bits = one shared policy.
  E4  behavioural complexity trigram entropy, H(a_t | a_{t-1}), and the
                             fraction of agents whose modal action exceeds 50%
                             of their life (policy collapse)
  E5  spatial structure      unique tiles per agent and mean pairwise Jaccard
                             overlap of visited-tile sets (territoriality)

Input is a run directory containing the CSVs written by `main.py --log`
(`agent_actions_*.csv`) and, optionally, `--metrics-csv` output named
`metrics.csv`. Output is one JSON object per run on stdout, so a campaign is

    python scripts/analyze_emergence.py runs/*/ > emergence.json

Author: Karan Vasa
Date: September 2026
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Ticks from USE(plant seed) to the first berry, for the default object pack:
# germination (plants.growth_time 50) + maturity (plants.mature_age 100) +
# a few ticks of production at seed_spawn_rate 0.1. A "harvest" earlier than
# this cannot be the fruit of that seed, so E2 reports both.
DEFAULT_MATURATION_TICKS = 160

# Actions that are valid in essentially every world state. Their share is the
# clearest single read-out of a degenerate always-available-action attractor.
ALWAYS_VALID = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "WAIT", "SIGNAL"]

ACTIONS = ALWAYS_VALID + ["PICK_UP", "DROP", "EAT", "USE"]


def entropy_bits(p) -> float:
    """
    Shannon entropy of a discrete distribution, in bits.

    Args:
        p: Non-negative weights (need not be normalised; zeros are dropped)

    Returns:
        Entropy in bits (0.0 for an empty or point distribution)
    """
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _load_actions(run_dir: str):
    """Load the newest agent_actions_*.csv in a run directory, or None."""
    files = sorted(glob.glob(os.path.join(run_dir, "agent_actions_*.csv")))
    if not files:
        return None
    df = pd.read_csv(files[-1])
    if df.empty:
        return None
    df["success"] = df["success"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df.sort_values(["agent_id", "tick"])


def _repertoire(df: pd.DataFrame, out: dict) -> None:
    """E3/E4 — action mix, role mutual information, sequence structure."""
    counts = df["action"].value_counts()
    total = int(counts.sum())
    for action in ACTIONS:
        out[f"pct_{action}"] = round(100.0 * counts.get(action, 0) / total, 3)
    out["pct_always_valid"] = round(sum(out[f"pct_{a}"] for a in ALWAYS_VALID), 3)
    out["H_pop"] = round(entropy_bits(counts.values), 4)

    # role_MI = I(agent ; action). Agents with < 100 logged actions are
    # excluded: a short life cannot express a role.
    tab = pd.crosstab(df["agent_id"], df["action"])
    tab = tab[tab.sum(axis=1) >= 100]
    out["n_agents_scored"] = int(len(tab))
    if len(tab) >= 2:
        per_agent = tab.div(tab.sum(axis=1), axis=0).values
        weights = (tab.sum(axis=1) / tab.values.sum()).values
        h_individual = np.array([entropy_bits(row) for row in per_agent])
        h_mixture = entropy_bits((per_agent * weights[:, None]).sum(axis=0))
        out["H_ind_mean"] = round(float((h_individual * weights).sum()), 4)
        out["role_MI"] = round(float(h_mixture - out["H_ind_mean"]), 4)
        out["modal_frac_mean"] = round(float(per_agent.max(axis=1).mean()), 4)
        out["frac_agents_collapsed"] = round(
            float((per_agent.max(axis=1) > 0.5).mean()), 4
        )
        # Null for role_MI: keep every agent's action COUNT but reassign the
        # actions themselves by shuffling the agent label within each tick
        # block, so co-occurrence in time is preserved and identity is
        # destroyed. Finite-sample MI is positive even under the null, which
        # is exactly why it has to be reported.
        rng_mi = np.random.default_rng(7)
        scored = set(tab.index)
        sub = df[df["agent_id"].isin(scored)].copy()
        sub["block"] = sub["tick"] // 100
        shuffled = sub.groupby("block")["agent_id"].transform(
            lambda col: col.values[rng_mi.permutation(len(col))]
        )
        tab_null = pd.crosstab(shuffled, sub["action"])
        tab_null = tab_null[tab_null.sum(axis=1) >= 100]
        if len(tab_null) >= 2:
            pn = tab_null.div(tab_null.sum(axis=1), axis=0).values
            wn = (tab_null.sum(axis=1) / tab_null.values.sum()).values
            hn = np.array([entropy_bits(r) for r in pn])
            out["role_MI_null"] = round(
                float(entropy_bits((pn * wn[:, None]).sum(axis=0)) - (hn * wn).sum()), 4
            )

    # H(a_t | a_{t-1}) — how much of the next action the previous one explains
    prev = df.groupby("agent_id")["action"].shift(1)
    joint = pd.crosstab(prev, df["action"])
    if joint.values.sum() > 0:
        j = joint.values / joint.values.sum()
        h_cond = entropy_bits(j.ravel()) - entropy_bits(j.sum(axis=1))
        out["H_cond"] = round(float(h_cond), 4)
        out["seq_structure_bits"] = round(float(out["H_pop"] - h_cond), 4)

    grouped = df.groupby("agent_id")["action"]
    trigrams = (grouped.shift(2) + ">" + grouped.shift(1) + ">" + df["action"]).dropna()
    if len(trigrams):
        vc = trigrams.value_counts()
        out["trigram_types"] = int(len(vc))
        out["trigram_H"] = round(entropy_bits(vc.values), 4)
        out["trigram_top1"] = str(vc.index[0])
        out["trigram_top1_pct"] = round(100.0 * vc.iloc[0] / vc.sum(), 3)


def _rates(df: pd.DataFrame, out: dict, n_ticks: int) -> None:
    """Interaction rates per agent per 1000 ticks, plus success rates."""
    per = 1000.0 / max(n_ticks, 1) / max(out["n_agents"], 1)
    for action in ["EAT", "PICK_UP", "USE", "DROP", "SIGNAL"]:
        ok = df[(df["action"] == action) & df["success"]]
        tried = int((df["action"] == action).sum())
        out[f"rate_{action}_ok"] = round(len(ok) * per, 4)
        out[f"succ_{action}"] = round(100.0 * len(ok) / tried, 2) if tried else None
    if "interaction_kind" in df.columns:
        kinds = df["interaction_kind"].dropna().astype(str).value_counts()
        out["interaction_kinds"] = {
            k: int(v) for k, v in kinds.items() if k not in ("nan", "")
        }


def _return_to_patch(df: pd.DataFrame, out: dict, rng) -> None:
    """
    E1 — does an agent go back to a tile where it previously ate?

    For each successful EAT at (x, y) by agent A at tick t, ask whether A
    stands on that tile again after t. The null draws, for each such event, a
    random tile from the same agent's own visited set and asks the same
    question — so the null absorbs "the agent walks in a small area anyway",
    which is the whole difficulty of this measurement.
    """
    if not {"x_after", "y_after"}.issubset(df.columns):
        return
    hits = nulls = events = 0
    for agent_id, traj in df.groupby("agent_id"):
        eats = traj[(traj["action"] == "EAT") & traj["success"]]
        if eats.empty:
            continue
        visited = list(zip(traj["x_after"], traj["y_after"]))
        if len(visited) < 2:
            continue
        ticks = traj["tick"].values
        for t0, ex, ey in zip(eats["tick"], eats["x_after"], eats["y_after"]):
            later = [p for p, tk in zip(visited, ticks) if tk > t0]
            if not later:
                continue
            events += 1
            hits += int((ex, ey) in later)
            rx, ry = visited[int(rng.integers(len(visited)))]
            nulls += int((rx, ry) in later)
    if events:
        out["patch_return_events"] = events
        out["patch_return_rate"] = round(hits / events, 4)
        out["patch_return_null"] = round(nulls / events, 4)
        out["patch_return_index"] = round(hits / max(nulls, 1e-9), 3) if nulls else None


def _cultivation(df: pd.DataFrame, out: dict, rng, maturation: int) -> None:
    """
    E2 — does a planter eat what it planted?

    Reports the raw "someone ate within 1 tile of the planted tile later" rate
    (which is mostly a spatial-correlation artefact: agents plant where food
    already is), the same rate restricted to latencies >= the maturation time
    (which is the only window in which the fruit of that seed can exist), and
    the sub-rate where the eater is the planter.
    """
    if "interaction_kind" not in df.columns:
        return
    if not {"target_x", "target_y"}.issubset(df.columns):
        return
    plants = df[
        (df["action"] == "USE")
        & df["success"]
        & df["interaction_kind"].astype(str).str.startswith("plant_seed")
    ].dropna(subset=["target_x", "target_y"])
    eats = df[(df["action"] == "EAT") & df["success"]]
    out["n_plant"] = int(len(plants))
    out["n_eat"] = int(len(eats))
    if plants.empty or eats.empty:
        return
    eat_events = list(
        zip(eats["tick"], eats["x_after"], eats["y_after"], eats["agent_id"])
    )

    def first_harvest(t0, px, py, min_latency):
        for t1, ex, ey, eater in eat_events:
            if t1 > t0 + min_latency and abs(ex - px) <= 1 and abs(ey - py) <= 1:
                return t1 - t0, eater
        return None

    any_hit = self_hit = mature_hit = mature_self = 0
    latencies = []
    for t0, px, py, planter in zip(
        plants["tick"], plants["target_x"], plants["target_y"], plants["agent_id"]
    ):
        first = first_harvest(t0, px, py, 0)
        if first:
            any_hit += 1
            latencies.append(first[0])
            self_hit += int(first[1] == planter)
        ripe = first_harvest(t0, px, py, maturation)
        if ripe:
            mature_hit += 1
            mature_self += int(ripe[1] == planter)
    n = len(plants)
    out["plant_harvest_rate"] = round(any_hit / n, 4)
    out["plant_harvest_self_rate"] = round(self_hit / n, 4)
    out["plant_harvest_mature_rate"] = round(mature_hit / n, 4)
    out["plant_harvest_mature_self_rate"] = round(mature_self / n, 4)
    out["plant_harvest_latency_med"] = (
        float(np.median(latencies)) if latencies else None
    )

    # Spatially matched null: the same number of tiles, drawn from tiles the
    # population actually stood on, each paired with a real planting's tick.
    visited = df[["x_after", "y_after"]].dropna().values
    if len(visited):
        idx = rng.choice(len(visited), min(n, len(visited)), replace=False)
        ticks = plants["tick"].values[: len(idx)]
        out["plant_harvest_null"] = round(
            sum(
                1
                for (px, py), t0 in zip(visited[idx], ticks)
                if first_harvest(t0, px, py, 0)
            )
            / max(len(idx), 1),
            4,
        )
        out["plant_harvest_mature_null"] = round(
            sum(
                1
                for (px, py), t0 in zip(visited[idx], ticks)
                if first_harvest(t0, px, py, maturation)
            )
            / max(len(idx), 1),
            4,
        )


def _spatial(df: pd.DataFrame, out: dict, rng) -> None:
    """E5 — territory size and pairwise overlap of visited-tile sets."""
    if not {"x_after", "y_after"}.issubset(df.columns):
        return
    visited = {
        agent_id: set(zip(traj["x_after"], traj["y_after"]))
        for agent_id, traj in df.groupby("agent_id")
        if len(traj) >= 100
    }
    if len(visited) < 2:
        return
    out["tiles_per_agent"] = round(
        float(np.mean([len(s) for s in visited.values()])), 2
    )
    keys = list(visited)
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1 :]]
    if len(pairs) > 400:
        pairs = [pairs[i] for i in rng.choice(len(pairs), 400, replace=False)]
    jaccard = [
        len(visited[a] & visited[b]) / max(len(visited[a] | visited[b]), 1)
        for a, b in pairs
    ]
    out["territory_overlap"] = round(float(np.mean(jaccard)), 4)


def _quarters(df: pd.DataFrame, out: dict) -> None:
    """Time course: is the population still changing, and toward what?"""
    q = pd.qcut(df["tick"], 4, labels=False, duplicates="drop")
    labels = sorted(pd.Series(q).dropna().unique())
    out["mix_quarters"] = {
        a: [
            round(
                100.0 * ((df[q == k]["action"] == a).sum()) / max((q == k).sum(), 1), 2
            )
            for k in labels
        ]
        for a in ALWAYS_VALID
    }
    out["eatpct_quarters"] = [
        round(
            100.0
            * ((df[q == k]["action"] == "EAT") & df[q == k]["success"]).sum()
            / max((q == k).sum(), 1),
            3,
        )
        for k in labels
    ]
    role_mi = []
    for k in labels:
        sub = df[q == k]
        tab = pd.crosstab(sub["agent_id"], sub["action"])
        tab = tab[tab.sum(axis=1) >= 40]
        if len(tab) >= 2:
            per_agent = tab.div(tab.sum(axis=1), axis=0).values
            weights = (tab.sum(axis=1) / tab.values.sum()).values
            h_ind = np.array([entropy_bits(r) for r in per_agent])
            role_mi.append(
                round(
                    float(
                        entropy_bits((per_agent * weights[:, None]).sum(axis=0))
                        - (h_ind * weights).sum()
                    ),
                    4,
                )
            )
    out["role_MI_quarters"] = role_mi


def _metrics_tail(run_dir: str, out: dict) -> None:
    """Last row of the --metrics-csv file, if present."""
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return
    m = pd.read_csv(path)
    if m.empty:
        return
    last = m.iloc[-1]
    for col in [
        "alive_agents",
        "total_food",
        "total_plants",
        "total_seeds",
        "mean_energy",
        "mean_age",
        "mean_fitness",
        "wm_rollout_error",
    ]:
        if col in m.columns:
            value = last[col]
            out[f"end_{col}"] = None if pd.isna(value) else float(value)


def analyze_run(run_dir: str, maturation: int = DEFAULT_MATURATION_TICKS) -> dict:
    """
    Compute the full emergence metric set for one run directory.

    Args:
        run_dir: Directory holding agent_actions_*.csv (and optionally
            metrics.csv written by --metrics-csv)
        maturation: Ticks from planting to the first harvestable fruit,
            used by the E2 latency filter

    Returns:
        Dict of metrics (JSON-serialisable), or None if there is no log
    """
    df = _load_actions(run_dir)
    if df is None:
        return None
    rng = np.random.default_rng(0)
    out = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "ticks": int(df["tick"].max()),
        "n_agents": int(df["agent_id"].nunique()),
        "n_logged_actions": int(len(df)),
    }
    _repertoire(df, out)
    _rates(df, out, out["ticks"])
    _return_to_patch(df, out, rng)
    _cultivation(df, out, rng, maturation)
    _spatial(df, out, rng)
    _quarters(df, out)
    if "age" in df.columns:
        out["max_age"] = int(df["age"].max())
        out["mean_final_age"] = round(
            float(df.groupby("agent_id")["age"].max().mean()), 2
        )
    _metrics_tail(run_dir, out)
    return out


def main(argv) -> int:
    """Analyze every run directory given on the command line."""
    if len(argv) < 2:
        print(__doc__)
        print("usage: python scripts/analyze_emergence.py RUN_DIR [RUN_DIR ...]")
        return 1
    rows = []
    for run_dir in sorted(argv[1:]):
        try:
            row = analyze_run(run_dir)
        except Exception as exc:  # noqa: BLE001 — one bad run must not kill a campaign
            print(f"ERROR {run_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if row is None:
            print(f"SKIP {run_dir}: no agent_actions_*.csv", file=sys.stderr)
            continue
        rows.append(row)
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
