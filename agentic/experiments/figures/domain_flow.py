#!/usr/bin/env python3
"""
Domain Concept Flow: Asymmetric directed heatmap + broadcast/porosity summary.

Shows the fraction of target-domain episodes that contain source-domain concepts,
revealing the directed knowledge dependency structure hidden in the bipartite graph.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results" / "hierarchy_v2"
FIGURES_DIR = Path(__file__).parent

DOMAIN_COLORS = {
    "MC3_C0": "#e69f00", "MC3_C1": "#0072b2", "MC3_C2": "#009e73",
    "MC3_C3": "#cc79a7", "MC3_C4": "#d55e00", "MC3_C5": "#f0e442",
    "MC3_C6": "#56b4e9", "MC3_C7": "#999999",
}

SHORT_LABELS = {
    "MC3_C0": "DevOps", "MC3_C1": "Soft. Eng.", "MC3_C2": "Statistics",
    "MC3_C3": "LLM Eng.", "MC3_C4": "Philosophy", "MC3_C5": "AI Safety",
    "MC3_C6": "Math", "MC3_C7": "ML/Networks",
}


def darken_color(hex_color, factor=0.65):
    """Darken a hex color by the given factor for better tick-label readability."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


def main():
    with open(RESULTS_DIR / "domain_flow_analysis.json") as f:
        data = json.load(f)

    flow = data["flow_matrix"]
    metrics = data["domain_metrics"]
    order = data["domain_order"]
    n = len(order)

    # Build matrix
    matrix = np.zeros((n, n))
    for i, src in enumerate(order):
        for j, tgt in enumerate(order):
            if src == tgt:
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = flow[src][tgt]["fraction"]

    labels = [SHORT_LABELS[d] for d in order]

    # --- Figure: 2-panel layout ---
    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(16, 7),
                                           gridspec_kw={"width_ratios": [3, 2]})

    # == Panel 1: Asymmetric flow heatmap ==
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color="#f0f0f0")

    im = ax_heat.imshow(matrix, cmap=cmap, vmin=0, vmax=0.55, aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i == j:
                ax_heat.text(j, i, "—", ha="center", va="center",
                            fontsize=10, color="#aaaaaa")
            else:
                val = matrix[i, j]
                color = "white" if val > 0.35 else "black"
                ax_heat.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=9, fontweight="bold" if val > 0.30 else "normal",
                            color=color)

    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax_heat.set_yticklabels(labels, fontsize=10)

    # Color the tick labels by domain (darkened for readability)
    for i, dom in enumerate(order):
        dark = darken_color(DOMAIN_COLORS[dom])
        ax_heat.get_yticklabels()[i].set_color(dark)
        ax_heat.get_xticklabels()[i].set_color(dark)

    ax_heat.set_xlabel("Target domain (whose episodes?)", fontsize=11, labelpad=10)
    ax_heat.set_ylabel("Source domain (whose concepts?)", fontsize=11, labelpad=10)
    ax_heat.set_title("Concept Flow: % of Target Episodes\nContaining Source Concepts",
                      fontsize=13, fontweight="bold", pad=10)

    # Colorbar
    cb = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cb.set_label("Fraction of target episodes", fontsize=9)

    # == Panel 2: Broadcast reach vs Porosity ==
    domains = order
    broadcast = [metrics[d]["broadcast_reach"] for d in domains]
    porosity = [metrics[d]["porosity"] for d in domains]
    breadth = [metrics[d]["reach_breadth"] for d in domains]
    colors = [DOMAIN_COLORS[d] for d in domains]

    x = np.arange(n)
    width = 0.35

    bars1 = ax_bar.barh(x + width/2, broadcast, width, color=colors, alpha=0.85,
                         edgecolor="white", linewidth=0.5, label="Broadcast reach")
    bars2 = ax_bar.barh(x - width/2, porosity, width, color=colors, alpha=0.55,
                         edgecolor=colors, linewidth=1.5, label="Porosity (incoming)")

    ax_bar.set_yticks(x)
    ax_bar.set_yticklabels([SHORT_LABELS[d] for d in domains], fontsize=10)
    for i, dom in enumerate(domains):
        ax_bar.get_yticklabels()[i].set_color(darken_color(DOMAIN_COLORS[dom]))

    ax_bar.set_xlabel("Avg. fraction of other-domain episodes", fontsize=10)
    ax_bar.set_title("Broadcast Reach vs. Porosity",
                     fontsize=13, fontweight="bold", pad=10)
    ax_bar.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Annotate breadth
    for i, (b, br) in enumerate(zip(broadcast, breadth)):
        ax_bar.text(b + 0.005, i + width/2, f"{br}/7",
                   va="center", fontsize=8, color="#666666")

    ax_bar.set_xlim(0, 0.50)
    ax_bar.invert_yaxis()

    plt.tight_layout(w_pad=3)

    for fmt in ["pdf", "png"]:
        out = FIGURES_DIR / f"domain_flow.{fmt}"
        fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
