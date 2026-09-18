"""
Evolution instrument — is the population diversifying, or converging?

`analyze_emergence.py` asks what agents *do*. This asks what they *are*, and
whether selection is producing structure in trait space. The three questions,
each with a null:

E8  Evolutionary branching
    Does a heritable trait's distribution split from one mode into two? This
    is the signature of disruptive selection under negative frequency
    dependence (Dieckmann & Doebeli 1999) and is the closest thing to emergent
    speciation this sandbox can show. Measured two ways:

      * **Bimodality coefficient**  BC = (g^2 + 1) / (k + 3(n-1)^2/((n-2)(n-3)))
        where g is sample skewness and k sample excess kurtosis. BC > 5/9 =
        0.555 is the standard threshold (a uniform distribution sits exactly
        at 5/9; a normal at 3/9).
      * **Two-component Gaussian mixture gain** — fit 1- and 2-component
        mixtures by EM and report the separation of the two means in pooled
        standard deviations. A split with |mu1 - mu2| > 2*sigma and both
        weights > 0.15 is a real branch rather than a fat tail.

E9  Selection strength and its coupling to behaviour
    Crow's opportunity for selection  I = Var(w) / mean(w)^2  over completed
    lifetimes, where w is offspring that reached reproductive age. I is the
    upper bound on the rate of adaptation: if it is ~0, nothing downstream of
    evolution can matter. Paired with the parent-offspring regression, whose
    slope b gives a heritability proxy h^2 = 2b.

E10 Local adaptation
    Does trait value correlate with *where* an agent lives? Correlation
    between a trait and the local terrain, against a label-shuffled null.

Input is a run directory containing `agent_states_*.csv` (from `main.py --log`).

    python scripts/analyze_evolution.py runs/*/ > evolution.json

Author: Karan Vasa
Date: September 2026
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Standard bimodality-coefficient threshold: a uniform distribution sits
# exactly here, so anything above is "flatter than uniform" = plausibly split.
BIMODALITY_THRESHOLD = 5.0 / 9.0

# A two-component fit only counts as a branch if the components are this far
# apart (in pooled SDs) and neither is a negligible sliver.
BRANCH_SEPARATION_SD = 2.0
BRANCH_MIN_WEIGHT = 0.15

NICHE_TRAITS = ("body_size", "visual_acuity", "diet")


def bimodality_coefficient(x: np.ndarray) -> float:
    """
    Sarle's bimodality coefficient.

        BC = (g^2 + 1) / (k + 3(n-1)^2 / ((n-2)(n-3)))

    with g the sample skewness and k the sample excess kurtosis. Values above
    5/9 indicate a distribution flatter-or-more-split than uniform.

    Args:
        x: Samples

    Returns:
        BC, or nan for fewer than 4 samples / zero variance
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return float("nan")
    sd = x.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    z = (x - x.mean()) / sd
    g = float((z**3).mean())
    k = float((z**4).mean() - 3.0)
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if abs(denom) < 1e-12:
        return float("nan")
    return float((g**2 + 1.0) / denom)


def two_component_em(x: np.ndarray, iters: int = 200, seed: int = 0) -> dict:
    """
    Fit a two-component 1-D Gaussian mixture by EM.

    Initialised at the 25th and 75th percentiles, which is enough for a
    one-dimensional trait and avoids the usual random-restart machinery.

    Args:
        x: Samples
        iters: EM iterations
        seed: Unused placeholder for reproducibility of future variants

    Returns:
        Dict with the two means, SDs, weights, and their separation in pooled
        SDs — or an empty dict when the sample is too small.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 30 or x.std() < 1e-9:
        return {}
    mu = np.array([np.percentile(x, 25), np.percentile(x, 75)], dtype=float)
    if abs(mu[1] - mu[0]) < 1e-9:
        return {}
    sd = np.array([x.std(), x.std()], dtype=float) / 2.0 + 1e-6
    w = np.array([0.5, 0.5])

    for _ in range(iters):
        # E step: responsibilities
        d = np.stack(
            [
                w[j]
                * np.exp(-0.5 * ((x - mu[j]) / sd[j]) ** 2)
                / (sd[j] * np.sqrt(2 * np.pi))
                for j in range(2)
            ]
        )
        total = d.sum(axis=0)
        total[total < 1e-300] = 1e-300
        r = d / total
        # M step
        nk = r.sum(axis=1)
        if np.any(nk < 1e-9):
            return {}
        w = nk / n
        mu = (r * x).sum(axis=1) / nk
        sd = np.sqrt((r * (x - mu[:, None]) ** 2).sum(axis=1) / nk) + 1e-6

    order = np.argsort(mu)
    mu, sd, w = mu[order], sd[order], w[order]
    pooled = float(np.sqrt((w * sd**2).sum()))
    separation = float(abs(mu[1] - mu[0]) / max(pooled, 1e-9))
    return {
        "mu": [round(float(m), 4) for m in mu],
        "sd": [round(float(s), 4) for s in sd],
        "weight": [round(float(v), 4) for v in w],
        "separation_sd": round(separation, 4),
        "branched": bool(
            separation >= BRANCH_SEPARATION_SD and w.min() >= BRANCH_MIN_WEIGHT
        ),
    }


def _load_states(run_dir: str):
    """Newest agent_states_*.csv in a run directory, or None."""
    files = sorted(glob.glob(os.path.join(run_dir, "agent_states_*.csv")))
    if not files:
        return None
    df = pd.read_csv(files[-1])
    return None if df.empty else df


def _trait_report(df: pd.DataFrame, trait: str, out: dict) -> None:
    """Distribution, branching and drift for one trait."""
    if trait not in df.columns:
        return
    last_tick = df["tick"].max()
    window = df[df["tick"] >= last_tick * 0.9]
    # One row per agent (its last observation) so long-lived agents do not
    # dominate the distribution.
    final = window.groupby("agent_id")[trait].last().to_numpy(dtype=float)
    if final.size < 4:
        return
    out[f"{trait}_mean"] = round(float(final.mean()), 4)
    out[f"{trait}_sd"] = round(float(final.std()), 4)
    out[f"{trait}_bc"] = round(bimodality_coefficient(final), 4)
    out[f"{trait}_bimodal"] = bool(out[f"{trait}_bc"] > BIMODALITY_THRESHOLD)
    mix = two_component_em(final)
    if mix:
        out[f"{trait}_mixture"] = mix
        out[f"{trait}_branched"] = mix["branched"]

    # Drift: mean in the first 10% of ticks vs the last 10%
    early = df[df["tick"] <= max(last_tick * 0.1, 1)]
    if not early.empty:
        first = early.groupby("agent_id")[trait].first().to_numpy(dtype=float)
        if first.size:
            out[f"{trait}_drift"] = round(float(final.mean() - first.mean()), 4)
            out[f"{trait}_sd_change"] = round(float(final.std() - first.std()), 4)


def analyze_run(run_dir: str) -> dict:
    """
    Trait-space report for one run directory.

    Args:
        run_dir: Directory holding agent_states_*.csv

    Returns:
        Dict of metrics, or None when there is no state log
    """
    df = _load_states(run_dir)
    if df is None:
        return None
    out = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "ticks": int(df["tick"].max()),
        "n_agents": int(df["agent_id"].nunique()),
    }

    if "generation" in df.columns:
        out["max_generation"] = int(df["generation"].max())
        out["mean_generation_end"] = round(
            float(df[df["tick"] >= df["tick"].max() * 0.9]["generation"].mean()), 2
        )
        # Generations per 1000 ticks — the currency evolution actually spends
        out["generations_per_1k"] = round(
            1000.0 * out["max_generation"] / max(out["ticks"], 1), 3
        )
    if "lineage_id" in df.columns:
        last = df[df["tick"] == df["tick"].max()]
        out["surviving_lineages"] = int(last["lineage_id"].nunique())
        out["lineages_seen"] = int(df["lineage_id"].nunique())

    for trait in NICHE_TRAITS:
        _trait_report(df, trait, out)

    # E9 — Crow's opportunity for selection over completed lifetimes.
    if "offspring_matured" in df.columns:
        final_w = df.groupby("agent_id")["offspring_matured"].max().to_numpy(float)
        # Exclude agents still alive at the end: their tally is censored.
        alive_ids = set(df[df["tick"] == df["tick"].max()]["agent_id"])
        completed = df[~df["agent_id"].isin(alive_ids)]
        if len(completed):
            w = completed.groupby("agent_id")["offspring_matured"].max().to_numpy(float)
            if w.size >= 10 and w.mean() > 1e-9:
                out["crow_I"] = round(float(w.var() / w.mean() ** 2), 4)
                out["mean_offspring"] = round(float(w.mean()), 4)
                out["frac_zero_offspring"] = round(float((w == 0).mean()), 4)
                out["n_completed_lifetimes"] = int(w.size)
        if final_w.size:
            out["max_offspring"] = int(final_w.max())

    # E10 — local adaptation: does a trait predict where an agent sits?
    if {"x", "y"}.issubset(df.columns):
        last = df[df["tick"] >= df["tick"].max() * 0.9]
        for trait in NICHE_TRAITS:
            if trait not in last.columns:
                continue
            per_agent = last.groupby("agent_id").agg(
                {trait: "last", "x": "mean", "y": "mean"}
            )
            if len(per_agent) < 10 or per_agent[trait].std() < 1e-9:
                continue
            rng = np.random.default_rng(0)
            values = per_agent[trait].to_numpy(float)
            coords = per_agent[["x", "y"]].to_numpy(float)
            # Moran-style: correlation between trait similarity and spatial
            # proximity, against a label-shuffled null.
            real = _spatial_autocorr(values, coords)
            null = np.mean(
                [_spatial_autocorr(rng.permutation(values), coords) for _ in range(20)]
            )
            out[f"{trait}_spatial_autocorr"] = round(float(real), 4)
            out[f"{trait}_spatial_null"] = round(float(null), 4)

    return out


def _spatial_autocorr(values: np.ndarray, coords: np.ndarray) -> float:
    """
    Distance-weighted trait autocorrelation (a Moran's I in spirit).

    w_ij = 1 / (1 + d_ij); I = sum_ij w_ij z_i z_j / (sum_ij w_ij) with z the
    standardised trait. Positive means nearby agents resemble each other.
    """
    n = values.size
    z = (values - values.mean()) / (values.std() + 1e-12)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    w = 1.0 / (1.0 + dist)
    np.fill_diagonal(w, 0.0)
    denom = w.sum()
    if denom < 1e-12:
        return 0.0
    return float((w * np.outer(z, z)).sum() / denom) * n / max(n - 1, 1)


def main(argv) -> int:
    """Analyze every run directory given on the command line."""
    if len(argv) < 2:
        print(__doc__)
        print("usage: python scripts/analyze_evolution.py RUN_DIR [RUN_DIR ...]")
        return 1
    rows = []
    for run_dir in sorted(argv[1:]):
        try:
            row = analyze_run(run_dir)
        except Exception as exc:  # noqa: BLE001 — one bad run must not kill a campaign
            print(f"ERROR {run_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if row is None:
            print(f"SKIP {run_dir}: no agent_states_*.csv", file=sys.stderr)
            continue
        rows.append(row)
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
