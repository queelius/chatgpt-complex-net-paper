#!/usr/bin/env python3
"""
Zoom-In Bridge: Two episodes connected through the concept hierarchy.

Shows how two episodes from different domains (Statistics and Philosophy)
are related through shared concepts spanning 4 domains — a connection
invisible to partition-based methods but captured by the bipartite graph.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results" / "hierarchy_v2"
FIGURES_DIR = Path(__file__).parent

DOMAIN_COLORS = {
    "MC3_C0": "#e69f00", "MC3_C1": "#0072b2", "MC3_C2": "#009e73",
    "MC3_C3": "#cc79a7", "MC3_C4": "#d55e00", "MC3_C5": "#f0e442",
    "MC3_C6": "#56b4e9", "MC3_C7": "#999999",
}

DOMAIN_LABELS = {
    "MC3_C0": "DevOps & Integration", "MC3_C1": "Software Engineering",
    "MC3_C2": "Statistical Methods", "MC3_C3": "LLM Engineering",
    "MC3_C4": "Philosophy & AI Theory", "MC3_C5": "AI Safety & Research",
    "MC3_C6": "Math & Optimization", "MC3_C7": "ML & Network Science",
}

# The chosen pair
EP1_ID = "ai-language-models-compared"
EP2_ID = "language-models-and-computation"


def load_data():
    with open(RESULTS_DIR / "hierarchy_semantic.json") as f:
        hierarchy = json.load(f)
    with open(RESULTS_DIR / "extraction_state.json") as f:
        extraction = json.load(f)
    return hierarchy, extraction


def get_episode_concepts(extraction, hierarchy, ep_id):
    """Get an episode's concepts with their hierarchy info."""
    mc_cm = hierarchy["levels"][0]["cluster_membership"]
    th_cm = hierarchy["levels"][1]["cluster_membership"]
    dm_cm = hierarchy["levels"][2]["cluster_membership"]

    concept_to_mc = {}
    for mc_id, concepts in mc_cm.items():
        for c in concepts:
            concept_to_mc[c] = mc_id

    concept_to_domain = {}
    for dom_id, concepts in dm_cm.items():
        for c in concepts:
            concept_to_domain[c] = dom_id

    mc_to_domain = {}
    for mc_id, concepts in mc_cm.items():
        dc = Counter(concept_to_domain.get(c, "?") for c in concepts)
        mc_to_domain[mc_id] = dc.most_common(1)[0][0]

    mc_labels = {mc_id: concepts[0] for mc_id, concepts in mc_cm.items()}

    # Find the episode
    for ep in extraction["episodes"]:
        if ep["episode_id"] == ep_id:
            result = []
            seen_mc = set()
            for c in ep["concepts"]:
                mc = concept_to_mc.get(c.lower())
                if mc and mc not in seen_mc:
                    seen_mc.add(mc)
                    dom = mc_to_domain.get(mc, "?")
                    result.append({
                        "raw": c.lower(),
                        "mc_id": mc,
                        "mc_label": mc_labels.get(mc, mc),
                        "domain": dom,
                        "domain_label": DOMAIN_LABELS.get(dom, dom),
                    })
            return result
    return []


def draw_bridge(ep1_concepts, ep2_concepts):
    """Draw the zoom-in bridge figure."""
    # Classify concepts: shared vs unique
    mc1 = {c["mc_id"] for c in ep1_concepts}
    mc2 = {c["mc_id"] for c in ep2_concepts}
    shared_mcs = mc1 & mc2
    unique1_mcs = mc1 - mc2
    unique2_mcs = mc2 - mc1

    # Build lookup
    mc_info = {}
    for c in ep1_concepts + ep2_concepts:
        mc_info[c["mc_id"]] = c

    # Layout parameters
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor="white")

    # Positions — generous spacing
    EP_Y = 0.0       # episodes at bottom
    SHARED_Y = 5.0   # shared concepts in middle
    UNIQUE_Y = 2.2   # unique concepts near their episode
    DOMAIN_Y = 8.5   # domain labels at top

    EP1_X = -5.5
    EP2_X = 5.5
    CENTER_X = 0.0

    # --- Draw episodes ---
    ep1_title = EP1_ID.replace("-", " ").title()
    ep2_title = EP2_ID.replace("-", " ").title()

    # Episode 1 box
    ep1_dom = Counter(c["domain"] for c in ep1_concepts).most_common(1)[0][0]
    ep2_dom = Counter(c["domain"] for c in ep2_concepts).most_common(1)[0][0]

    for ex, title, dom, concepts_list in [
        (EP1_X, ep1_title, ep1_dom, ep1_concepts),
        (EP2_X, ep2_title, ep2_dom, ep2_concepts),
    ]:
        color = DOMAIN_COLORS.get(dom, "#999")
        box = mpatches.FancyBboxPatch(
            (ex - 2.8, EP_Y - 0.5), 5.6, 1.0,
            boxstyle="round,pad=0.15",
            facecolor=mcolors.to_rgba(color, 0.15),
            edgecolor=color, linewidth=2.0, zorder=5)
        ax.add_patch(box)
        ax.text(ex, EP_Y, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#333333", zorder=6,
                wrap=True)
        # Domain label under episode
        ax.text(ex, EP_Y - 0.85, DOMAIN_LABELS.get(dom, ""),
                ha="center", va="top", fontsize=9, fontstyle="italic",
                color=color, zorder=6)

    # --- Draw shared concepts (center column) ---
    shared_list = sorted(shared_mcs, key=lambda m: mc_info[m]["domain"])
    n_shared = len(shared_list)

    shared_positions = {}
    spacing = min(2.5, 11.0 / max(n_shared, 1))
    start_x = CENTER_X - (n_shared - 1) * spacing / 2

    for i, mc_id in enumerate(shared_list):
        info = mc_info[mc_id]
        x = start_x + i * spacing
        y = SHARED_Y
        shared_positions[mc_id] = (x, y)
        color = DOMAIN_COLORS.get(info["domain"], "#999")

        # Concept node — larger
        circle = plt.Circle((x, y), 0.45, facecolor=mcolors.to_rgba(color, 0.3),
                           edgecolor=color, linewidth=2.0, zorder=5)
        ax.add_patch(circle)

        # Concept label
        label = info["mc_label"]
        ax.text(x, y + 0.7, label, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#333333", zorder=6,
                rotation=25, rotation_mode="anchor",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.9))

        # Domain tag below concept
        ax.text(x, y - 0.7, info["domain_label"], ha="center", va="top",
                fontsize=8, color=color, fontstyle="italic", zorder=6)

        # Lines from both episodes to this shared concept
        for ex in [EP1_X, EP2_X]:
            ax.plot([ex, x], [EP_Y + 0.5, y - 0.45],
                   color=color, alpha=0.5, linewidth=2.0, zorder=2,
                   linestyle="-")

    # --- Draw unique concepts ---
    for unique_mcs, ep_x, side in [
        (unique1_mcs, EP1_X, -1),
        (unique2_mcs, EP2_X, 1),
    ]:
        unique_list = sorted(unique_mcs)
        for i, mc_id in enumerate(unique_list):
            info = mc_info[mc_id]
            x = ep_x + side * (1.0 + i * 2.2)
            y = UNIQUE_Y
            color = DOMAIN_COLORS.get(info["domain"], "#999")

            # Concept node (dashed = unique)
            circle = plt.Circle((x, y), 0.35,
                               facecolor=mcolors.to_rgba(color, 0.12),
                               edgecolor=color, linewidth=1.2,
                               linestyle="--", zorder=5)
            ax.add_patch(circle)

            label = info["mc_label"]
            if len(label) > 40:
                label = label[:38] + "…"
            ax.text(x, y + 0.55, label, ha="center", va="bottom",
                    fontsize=8, color="#555555", zorder=6)
            ax.text(x, y - 0.55, info["domain_label"], ha="center", va="top",
                    fontsize=7.5, color=color, fontstyle="italic", zorder=6)

            # Line from episode only
            ax.plot([ep_x, x], [EP_Y + 0.5, y - 0.35],
                   color=color, alpha=0.35, linewidth=1.5, linestyle="--",
                   zorder=2)

    # --- Draw domain grouping arcs at top ---
    shared_domains = set(mc_info[mc]["domain"] for mc in shared_mcs)
    domain_positions = {}
    dom_list = sorted(shared_domains)
    dom_spacing = 12.0 / max(len(dom_list), 1)
    dom_start = CENTER_X - (len(dom_list) - 1) * dom_spacing / 2

    for i, dom in enumerate(dom_list):
        dx = dom_start + i * dom_spacing
        dy = DOMAIN_Y
        domain_positions[dom] = (dx, dy)
        color = DOMAIN_COLORS.get(dom, "#999")

        ax.text(dx, dy, DOMAIN_LABELS.get(dom, dom), ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=6,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=mcolors.to_rgba(color, 0.75),
                          edgecolor=color, linewidth=2.0))

        # Lines from domain to its shared concepts
        for mc_id in shared_list:
            if mc_info[mc_id]["domain"] == dom:
                cx, cy = shared_positions[mc_id]
                ax.plot([dx, cx], [dy - 0.5, cy + 0.35],
                       color=color, alpha=0.25, linewidth=1.0,
                       linestyle=":", zorder=1)

    # --- Annotations ---
    # "Shared concepts" label
    ax.text(CENTER_X, SHARED_Y + 2.2, "shared meta-concepts",
            ha="center", va="center", fontsize=9, color="#999999",
            fontstyle="italic")

    # Arrow annotation showing the bridge
    ax.annotate("", xy=(2.5, SHARED_Y), xytext=(-2.5, SHARED_Y),
                arrowprops=dict(arrowstyle="<->", color="#cccccc",
                               lw=1.5, connectionstyle="arc3,rad=0.2"))

    # Stats
    n_unique1 = len(unique1_mcs)
    n_unique2 = len(unique2_mcs)
    stats = (f"{n_shared} shared concepts across {len(shared_domains)} domains  ·  "
             f"{n_unique1} + {n_unique2} unique")
    ax.text(0.5, 0.02, stats, transform=ax.transAxes,
            ha="center", fontsize=9, color="#888888")

    # Styling
    ax.set_xlim(-11, 11)
    ax.set_ylim(-2.2, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0.5)
    return fig


def main():
    print("Loading data...")
    hierarchy, extraction = load_data()

    print(f"Getting concepts for {EP1_ID}...")
    ep1_concepts = get_episode_concepts(extraction, hierarchy, EP1_ID)
    print(f"  {len(ep1_concepts)} meta-concepts")
    for c in ep1_concepts:
        print(f"    [{c['domain_label']:>25}] {c['mc_label']}")

    print(f"\nGetting concepts for {EP2_ID}...")
    ep2_concepts = get_episode_concepts(extraction, hierarchy, EP2_ID)
    print(f"  {len(ep2_concepts)} meta-concepts")
    for c in ep2_concepts:
        print(f"    [{c['domain_label']:>25}] {c['mc_label']}")

    print("\nDrawing bridge figure...")
    fig = draw_bridge(ep1_concepts, ep2_concepts)

    for fmt in ["pdf", "png"]:
        out = FIGURES_DIR / f"zoom_in_bridge.{fmt}"
        fig.savefig(out, dpi=250, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {out}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
