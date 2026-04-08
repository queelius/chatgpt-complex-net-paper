#!/usr/bin/env python3
"""
Bipartite Knowledge Map: Episodes + Meta-Concepts + Multiple Link Types.

Shows the full multiplex network:
  - Inner ring: 500 meta-concept nodes (colored by domain, sized by degree)
  - Outer ring: 1,908 episode nodes (small dots, colored by primary domain)
  - Bipartite edges: episode ↔ meta-concept (thin, colored)
  - Co-occurrence edges: meta-concept ↔ meta-concept (gray, thicker)

Layout: domain-sector radial — each domain gets a wedge of the circle,
with concepts near the center and episodes radiating outward.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent / "results" / "hierarchy_v2"
FIGURES_DIR = Path(__file__).parent
EXTRACTION_STATE = RESULTS_DIR / "extraction_state.json"
HIERARCHY_JSON = RESULTS_DIR / "hierarchy_semantic.json"

# ---------------------------------------------------------------------------
# Domain config — same as knowledge_map.py
# ---------------------------------------------------------------------------
DOMAIN_ORDER = [
    "MC3_C1",  # Software Engineering (largest, top)
    "MC3_C0",  # DevOps & Integration
    "MC3_C3",  # LLM Engineering
    "MC3_C7",  # ML & Network Science
    "MC3_C6",  # Math & Optimization
    "MC3_C2",  # Statistical Methods
    "MC3_C5",  # AI Safety & Research
    "MC3_C4",  # Philosophy & AI Theory
]

DOMAIN_LABELS = {
    "MC3_C0": "DevOps & Integration",
    "MC3_C1": "Software Engineering",
    "MC3_C2": "Statistical Methods",
    "MC3_C3": "LLM Engineering",
    "MC3_C4": "Philosophy & AI Theory",
    "MC3_C5": "AI Safety & Research",
    "MC3_C6": "Math & Optimization",
    "MC3_C7": "ML & Network Science",
}

DOMAIN_COLORS = {
    "MC3_C0": "#e69f00",  # orange
    "MC3_C1": "#0072b2",  # blue
    "MC3_C2": "#009e73",  # green
    "MC3_C3": "#cc79a7",  # pink
    "MC3_C4": "#d55e00",  # vermillion
    "MC3_C5": "#f0e442",  # yellow
    "MC3_C6": "#56b4e9",  # sky blue
    "MC3_C7": "#999999",  # grey
}


def load_data():
    with open(HIERARCHY_JSON) as f:
        hierarchy = json.load(f)
    with open(EXTRACTION_STATE) as f:
        extraction = json.load(f)
    return hierarchy, extraction


def build_mappings(hierarchy, extraction):
    """Build all mappings needed for the bipartite layout."""
    mc_cm = hierarchy["levels"][0]["cluster_membership"]
    dm_cm = hierarchy["levels"][2]["cluster_membership"]

    # concept -> mc
    concept_to_mc = {}
    for mc_id, concepts in mc_cm.items():
        for c in concepts:
            concept_to_mc[c] = mc_id

    # concept -> domain
    concept_to_domain = {}
    for dom_id, concepts in dm_cm.items():
        for c in concepts:
            concept_to_domain[c] = dom_id

    # mc -> domain (majority vote)
    mc_to_domain = {}
    for mc_id, concepts in mc_cm.items():
        dc = Counter(concept_to_domain.get(c, "?") for c in concepts)
        mc_to_domain[mc_id] = dc.most_common(1)[0][0]

    # mc -> representative label
    mc_labels = {mc_id: concepts[0] for mc_id, concepts in mc_cm.items()}

    # Episode -> meta-concepts, domain distribution
    ep_mcs = {}          # episode_id -> set of mc_ids
    ep_domain_dist = {}  # episode_id -> Counter of domains
    mc_episode_count = Counter()

    for ep in extraction["episodes"]:
        eid = ep["episode_id"]
        mcs = set()
        dom_counts = Counter()
        for c in ep["concepts"]:
            mc = concept_to_mc.get(c.lower())
            if mc:
                mcs.add(mc)
                dom = mc_to_domain.get(mc)
                if dom:
                    dom_counts[dom] += 1
        ep_mcs[eid] = mcs
        ep_domain_dist[eid] = dom_counts
        for mc in mcs:
            mc_episode_count[mc] += 1

    # Build co-occurrence edges
    cooc_weights = Counter()
    for mcs in ep_mcs.values():
        mcs_list = sorted(mcs)
        for i in range(len(mcs_list)):
            for j in range(i + 1, len(mcs_list)):
                cooc_weights[(mcs_list[i], mcs_list[j])] += 1

    return (concept_to_mc, mc_to_domain, mc_labels,
            ep_mcs, ep_domain_dist, mc_episode_count, cooc_weights)


def compute_positions(mc_to_domain, mc_episode_count, ep_mcs, ep_domain_dist):
    """Compute radial positions for all nodes.

    Layout:
      - Domain sectors arranged around a circle
      - Meta-concepts at inner radius, jittered within their sector
      - Episodes at outer radius, positioned by domain weight centroid
    """
    n_domains = len(DOMAIN_ORDER)

    # Sector angles — each domain gets a wedge proportional to its size
    # (number of episodes with that primary domain)
    ep_primary = {}
    for eid, dc in ep_domain_dist.items():
        if dc:
            ep_primary[eid] = dc.most_common(1)[0][0]

    domain_ep_counts = Counter(ep_primary.values())
    total_eps = sum(domain_ep_counts.values())

    # Compute sector boundaries (proportional to domain size)
    sector_starts = {}
    sector_widths = {}
    current_angle = -np.pi / 2  # start at top
    for dom_id in DOMAIN_ORDER:
        width = 2 * np.pi * domain_ep_counts.get(dom_id, 1) / total_eps
        sector_starts[dom_id] = current_angle
        sector_widths[dom_id] = width
        current_angle += width

    # Sector center angles
    sector_centers = {d: sector_starts[d] + sector_widths[d] / 2
                      for d in DOMAIN_ORDER}

    # --- Meta-concept positions (inner ring) ---
    CONCEPT_RADIUS = 6.0
    mc_pos = {}

    # Group meta-concepts by domain
    mc_by_domain = defaultdict(list)
    for mc_id, dom in mc_to_domain.items():
        mc_by_domain[dom].append(mc_id)

    rng = np.random.RandomState(42)
    for dom_id in DOMAIN_ORDER:
        mcs = mc_by_domain.get(dom_id, [])
        if not mcs:
            continue
        # Sort by frequency for deterministic layout
        mcs = sorted(mcs, key=lambda m: -mc_episode_count.get(m, 0))
        center_angle = sector_centers[dom_id]
        width = sector_widths[dom_id] * 0.85  # leave gap between sectors

        for i, mc_id in enumerate(mcs):
            # Spread within sector: angle varies, radius jittered
            frac = (i + 0.5) / len(mcs)  # 0 to 1
            angle = center_angle + width * (frac - 0.5)
            # Jitter radius for visual spread
            r = CONCEPT_RADIUS + rng.uniform(-1.2, 1.2)
            mc_pos[mc_id] = np.array([r * np.cos(angle), r * np.sin(angle)])

    # --- Episode positions (outer ring) ---
    EPISODE_RADIUS_MIN = 9.5
    EPISODE_RADIUS_MAX = 14.0
    ep_pos = {}

    # For each episode, compute its angular position as domain-weighted centroid
    for eid, dc in ep_domain_dist.items():
        if not dc:
            continue

        # Weighted angle: centroid of contributing domain sector centers
        total = sum(dc.values())
        # Use complex number trick for circular mean
        z = sum(dc[d] * np.exp(1j * sector_centers.get(d, 0))
                for d in dc if d in sector_centers)
        angle = np.angle(z)

        # Radius: more domains → further out (cross-domain episodes at periphery)
        n_doms = len(dc)
        r_frac = min((n_doms - 1) / 4.0, 1.0)  # 1 domain → inner, 4+ → outer
        r = EPISODE_RADIUS_MIN + r_frac * (EPISODE_RADIUS_MAX - EPISODE_RADIUS_MIN)
        # Jitter
        r += rng.uniform(-1.0, 1.0)
        angle += rng.uniform(-0.04, 0.04)

        ep_pos[eid] = np.array([r * np.cos(angle), r * np.sin(angle)])

    return mc_pos, ep_pos, sector_centers, sector_starts, sector_widths


def draw_bipartite_map(mc_pos, ep_pos, mc_to_domain, mc_labels, mc_episode_count,
                        ep_mcs, ep_domain_dist, cooc_weights,
                        sector_centers, sector_starts, sector_widths):
    """Draw the full bipartite knowledge map."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor="white")

    # ── 1. Draw sector wedge backgrounds ──
    for dom_id in DOMAIN_ORDER:
        start = np.degrees(sector_starts[dom_id])
        end = np.degrees(sector_starts[dom_id] + sector_widths[dom_id])
        wedge = mpatches.Wedge((0, 0), 16, start, end,
                               facecolor=DOMAIN_COLORS[dom_id],
                               alpha=0.04, zorder=0)
        ax.add_patch(wedge)

    # ── 2. Draw bipartite edges (episode → concept) ──
    # Threshold: only show links to meta-concepts appearing in MC_FREQ_MIN+ episodes
    # This is analogous to the cosine similarity cutoff θ in the conference paper
    MC_FREQ_MIN = 15  # show links to frequent concepts only
    frequent_mcs = {mc for mc, cnt in mc_episode_count.items() if cnt >= MC_FREQ_MIN}

    bip_segments = []
    bip_colors = []
    for eid, mcs in ep_mcs.items():
        if eid not in ep_pos:
            continue
        ex, ey = ep_pos[eid]
        for mc_id in mcs:
            if mc_id not in mc_pos or mc_id not in frequent_mcs:
                continue
            mx, my = mc_pos[mc_id]
            bip_segments.append([(ex, ey), (mx, my)])
            dom = mc_to_domain.get(mc_id, "MC3_C1")
            bip_colors.append(mcolors.to_rgba(DOMAIN_COLORS.get(dom, "#ccc"),
                                               alpha=0.10))

    if bip_segments:
        lc = LineCollection(bip_segments, colors=bip_colors,
                           linewidths=0.2, zorder=1)
        ax.add_collection(lc)
    print(f"  Bipartite edges drawn: {len(bip_segments)} "
          f"(meta-concepts with freq >= {MC_FREQ_MIN}: {len(frequent_mcs)})")

    # ── 3. Draw co-occurrence edges (concept ↔ concept) ──
    COOC_THRESHOLD = 6
    cooc_segments = []
    cooc_colors = []
    max_cooc = max(cooc_weights.values()) if cooc_weights else 1

    for (u, v), w in cooc_weights.items():
        if w < COOC_THRESHOLD:
            continue
        if u not in mc_pos or v not in mc_pos:
            continue
        cooc_segments.append([(mc_pos[u][0], mc_pos[u][1]),
                              (mc_pos[v][0], mc_pos[v][1])])
        # Cross-domain co-occurrence: darker
        if mc_to_domain.get(u) != mc_to_domain.get(v):
            a = 0.05 + 0.20 * (w / max_cooc)
            cooc_colors.append(mcolors.to_rgba("#555555", alpha=a))
        else:
            dom = mc_to_domain.get(u, "MC3_C1")
            a = 0.08 + 0.25 * (w / max_cooc)
            cooc_colors.append(mcolors.to_rgba(DOMAIN_COLORS.get(dom, "#ccc"),
                                                alpha=a))

    if cooc_segments:
        lc = LineCollection(cooc_segments, colors=cooc_colors,
                           linewidths=0.4, zorder=2)
        ax.add_collection(lc)
    print(f"  Co-occurrence edges drawn: {len(cooc_segments)}")

    # ── 4. Draw episode nodes (outer ring — small dots) ──
    ep_x, ep_y, ep_colors, ep_sizes = [], [], [], []
    for eid, pos in ep_pos.items():
        ep_x.append(pos[0])
        ep_y.append(pos[1])
        dc = ep_domain_dist.get(eid, {})
        primary = dc.most_common(1)[0][0] if dc else "MC3_C1"
        ep_colors.append(DOMAIN_COLORS.get(primary, "#cccccc"))
        # Size by number of concepts (more concepts → slightly larger)
        n_concepts = len(ep_mcs.get(eid, set()))
        ep_sizes.append(4 + 1.5 * n_concepts)

    ax.scatter(ep_x, ep_y, s=ep_sizes, c=ep_colors,
               edgecolors="none", alpha=0.65, zorder=3)
    print(f"  Episode nodes drawn: {len(ep_x)}")

    # ── 5. Draw meta-concept nodes (inner ring — larger) ──
    mc_nodes = list(mc_pos.keys())
    mc_x = [mc_pos[n][0] for n in mc_nodes]
    mc_y = [mc_pos[n][1] for n in mc_nodes]
    mc_colors = [DOMAIN_COLORS.get(mc_to_domain.get(n, "MC3_C1"), "#ccc")
                 for n in mc_nodes]

    # Size by episode count (log-scaled)
    mc_freqs = np.array([mc_episode_count.get(n, 1) for n in mc_nodes], dtype=float)
    log_freq = np.log1p(mc_freqs)
    mc_sizes = 20 + 250 * (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min() + 1e-10)

    ax.scatter(mc_x, mc_y, s=mc_sizes, c=mc_colors,
               edgecolors="white", linewidths=0.4, alpha=0.9, zorder=5,
               marker="o")

    # ── 6. Domain labels ── horizontal, positioned outside each sector
    LABEL_RADIUS = 14.8
    for dom_id in DOMAIN_ORDER:
        angle = sector_centers[dom_id]
        lx = LABEL_RADIUS * np.cos(angle)
        ly = LABEL_RADIUS * np.sin(angle)
        label = DOMAIN_LABELS[dom_id]
        # Horizontal alignment based on position
        if np.cos(angle) > 0.3:
            ha = "left"
        elif np.cos(angle) < -0.3:
            ha = "right"
        else:
            ha = "center"
        ax.text(lx, ly, label, fontsize=12, fontweight="bold",
                ha=ha, va="center",
                color=DOMAIN_COLORS[dom_id], alpha=0.9, zorder=7,
                fontfamily="sans-serif",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=DOMAIN_COLORS[dom_id], alpha=0.85,
                          linewidth=1.5))

    # ── 7. Label top concept per domain ──
    labels_per_domain = defaultdict(list)
    for mc_id in sorted(mc_nodes, key=lambda n: -mc_episode_count.get(n, 0)):
        dom = mc_to_domain.get(mc_id)
        if len(labels_per_domain[dom]) < 1:
            labels_per_domain[dom].append(mc_id)

    for mc_id in [n for ns in labels_per_domain.values() for n in ns]:
        label = mc_labels.get(mc_id, mc_id)
        if len(label) > 25:
            label = label[:23] + "…"
        x, y = mc_pos[mc_id]
        ax.annotate(label, (x, y), fontsize=8, fontstyle="italic",
                    ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points",
                    color="#222222", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              alpha=0.7, edgecolor="none"))

    # ── 8. Ring guides ──
    for r in [6.0, 10.0, 14.5]:
        circle = plt.Circle((0, 0), r, fill=False, color="#dddddd",
                           linewidth=0.3, linestyle="--", zorder=0)
        ax.add_patch(circle)

    # Styling
    ax.set_xlim(-17, 17)
    ax.set_ylim(-17, 17)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0.5)
    return fig


def main():
    print("Loading data...")
    hierarchy, extraction = load_data()

    print("Building mappings...")
    (concept_to_mc, mc_to_domain, mc_labels,
     ep_mcs, ep_domain_dist, mc_episode_count, cooc_weights) = build_mappings(
         hierarchy, extraction)

    print("Computing positions...")
    mc_pos, ep_pos, sector_centers, sector_starts, sector_widths = compute_positions(
        mc_to_domain, mc_episode_count, ep_mcs, ep_domain_dist)

    print(f"  Meta-concepts positioned: {len(mc_pos)}")
    print(f"  Episodes positioned: {len(ep_pos)}")

    print("Drawing bipartite knowledge map...")
    fig = draw_bipartite_map(
        mc_pos, ep_pos, mc_to_domain, mc_labels, mc_episode_count,
        ep_mcs, ep_domain_dist, cooc_weights,
        sector_centers, sector_starts, sector_widths)

    for fmt in ["pdf", "png"]:
        out = FIGURES_DIR / f"knowledge_map_bipartite.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {out}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
