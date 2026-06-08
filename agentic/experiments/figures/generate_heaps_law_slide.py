#!/usr/bin/env python3
"""
Slide-optimized version of the Heaps' law figure.

Differences vs the paper figure (heaps_law.pdf):
  - Single panel only (no histogram subplot)
  - Cleaner legend: drops the "Heaps fit V(n)=..." entries (the dashed
    lines themselves communicate the fit; the formula belongs in the paper)
  - Shows BOTH null bands (cluster-permutation and bipartite-configuration)
    so the figure matches the slide right-column "Two nulls" block
  - No trim hack needed in the slide

Loads cached null model statistics from heaps_null_model.json; only the
vocabulary growth curves are recomputed (fast).

Output: experiments/figures/heaps_law_slide.pdf
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

RESULTS_DIR = Path(__file__).parent.parent / "results" / "hierarchy_v2"
FIGURES_DIR = Path(__file__).parent


def heaps_law(n, K, beta):
    return K * np.power(n, beta)


def build_concept_to_mc(hierarchy):
    mp = {}
    for mc_id, raw_concepts in hierarchy["levels"][0]["cluster_membership"].items():
        for rc in raw_concepts:
            mp[rc.lower()] = mc_id
    return mp


def vocabulary_growth(episodes, mapping=None):
    """Return (x, y) where y[i] = distinct items seen after episode i.

    If mapping is None, count raw concepts; else map each concept through
    `mapping` and count distinct mapped values.
    """
    seen = set()
    y = []
    for ep in episodes:
        for c in ep["concepts"]:
            c_lower = c.lower()
            if mapping is None:
                seen.add(c_lower)
            else:
                m = mapping.get(c_lower)
                if m is not None:
                    seen.add(m)
        y.append(len(seen))
    x = np.arange(1, len(episodes) + 1, dtype=float)
    return x, np.array(y, dtype=float)


def main():
    plt.rcParams.update({
        "font.size": 13, "font.family": "serif", "savefig.dpi": 300,
        "figure.dpi": 100, "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    })
    sns.set_style("whitegrid")

    # --- Load data ----------------------------------------------------------
    with open(RESULTS_DIR / "extraction_state.json") as f:
        episodes = json.load(f)["episodes"]
    with open(RESULTS_DIR / "hierarchy_semantic.json") as f:
        hierarchy = json.load(f)
    with open(RESULTS_DIR / "heaps_null_model.json") as f:
        nulls = json.load(f)

    concept_to_mc = build_concept_to_mc(hierarchy)
    real_K = nulls["real_K"]
    real_beta = nulls["real_beta"]

    # --- Compute vocabulary growth curves -----------------------------------
    x_raw, y_raw = vocabulary_growth(episodes)
    x_mc, y_mc = vocabulary_growth(episodes, concept_to_mc)

    popt_raw, _ = curve_fit(heaps_law, x_raw, y_raw, p0=[1.0, 0.9], maxfev=10000)
    K_raw, beta_raw = popt_raw

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x_fit = np.linspace(1, len(episodes), 500)

    # Raw concepts: solid line + dashed fit (no legend entry for the fit).
    ax.plot(x_raw, y_raw, "-", color="#2c3e50", linewidth=1.8, alpha=0.95,
            label=fr"Raw concepts ($\beta = {beta_raw:.2f}$)")
    ax.plot(x_fit, heaps_law(x_fit, *popt_raw), "--", color="#2c3e50",
            linewidth=1.0, alpha=0.5)

    # Meta-concepts: solid line + dashed fit.
    ax.plot(x_mc, y_mc, "-", color="#2980b9", linewidth=1.8, alpha=0.95,
            label=fr"Meta-concepts ($\beta = {real_beta:.2f}$)")
    ax.plot(x_fit, heaps_law(x_fit, real_K, real_beta), "--", color="#2980b9",
            linewidth=1.0, alpha=0.5)

    # Cluster-permutation null band (using its mean beta).
    cn = nulls["null_model"]
    cn_lo = heaps_law(x_fit, real_K, cn["percentiles"]["2.5"])
    cn_hi = heaps_law(x_fit, real_K, cn["percentiles"]["97.5"])
    ax.fill_between(x_fit, cn_lo, cn_hi, color="#95a5a6", alpha=0.45,
                    label=fr"Null: shuffled clusters ($\beta \approx {cn['mean_beta']:.2f}$)")

    # Bipartite-configuration null band.
    bn = nulls["bipartite_null_model"]
    bn_lo = heaps_law(x_fit, real_K, bn["percentiles"]["2.5"])
    bn_hi = heaps_law(x_fit, real_K, bn["percentiles"]["97.5"])
    ax.fill_between(x_fit, bn_lo, bn_hi, color="#16a085", alpha=0.35,
                    label=fr"Null: shuffled links ($\beta \approx {bn['mean_beta']:.2f}$)")

    # Linear reference.
    linear_slope = y_raw[-1] / len(episodes)
    ax.plot(x_fit, linear_slope * x_fit, ":", color="#bdc3c7", linewidth=1.2,
            alpha=0.7, label=r"Linear ($\beta = 1$)")

    ax.set_xlabel("Number of episodes processed")
    ax.set_ylabel("Cumulative vocabulary size")
    ax.set_title("Vocabulary Growth: Heaps' Law", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc",
              fontsize=10)
    ax.set_xlim(0, len(episodes) + 20)
    ax.set_ylim(0, None)

    plt.tight_layout()
    out_pdf = FIGURES_DIR / "heaps_law_slide.pdf"
    out_png = FIGURES_DIR / "heaps_law_slide.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, format="png", bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    print(f"  Raw concepts:   beta = {beta_raw:.3f}")
    print(f"  Meta-concepts:  beta = {real_beta:.3f}")
    print(f"  Cluster null:   beta in [{cn['percentiles']['2.5']:.3f}, {cn['percentiles']['97.5']:.3f}]")
    print(f"  Bipartite null: beta in [{bn['percentiles']['2.5']:.3f}, {bn['percentiles']['97.5']:.3f}]")


if __name__ == "__main__":
    main()
