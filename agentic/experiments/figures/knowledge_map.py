#!/usr/bin/env python3
"""
Knowledge Map: Centerpiece visualization of the emergent hierarchical memory.

Builds the meta-concept co-occurrence network (500 nodes, ~6K edges),
colors by domain, sizes by frequency, and produces a publication-quality
force-directed layout that reveals the cognitive architecture.

Also exports GEXF for optional Gephi refinement.
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent / "results" / "hierarchy_v2"
FIGURES_DIR = Path(__file__).parent
EXTRACTION_STATE = RESULTS_DIR / "extraction_state.json"
HIERARCHY_JSON = RESULTS_DIR / "hierarchy_semantic.json"
EMBEDDINGS_NPY = RESULTS_DIR / "concept_embeddings.npy"

# ---------------------------------------------------------------------------
# Domain labels and colors — consistent with paper Table 2
# ---------------------------------------------------------------------------
DOMAIN_LABELS = {
    "MC3_C0": "DevOps &\nIntegration",
    "MC3_C1": "Software\nEngineering",
    "MC3_C2": "Statistical\nMethods",
    "MC3_C3": "LLM\nEngineering",
    "MC3_C4": "Philosophy &\nAI Theory",
    "MC3_C5": "AI Safety &\nResearch",
    "MC3_C6": "Math &\nOptimization",
    "MC3_C7": "ML & Network\nScience",
}

# Hand-picked qualitative palette — distinct, colorblind-friendly-ish
DOMAIN_COLORS = {
    "MC3_C0": "#e69f00",  # orange — DevOps
    "MC3_C1": "#0072b2",  # blue — Software Engineering
    "MC3_C2": "#009e73",  # green — Statistics
    "MC3_C3": "#cc79a7",  # pink — LLM Engineering
    "MC3_C4": "#d55e00",  # vermillion — Philosophy
    "MC3_C5": "#f0e442",  # yellow — AI Safety
    "MC3_C6": "#56b4e9",  # sky blue — Math
    "MC3_C7": "#999999",  # grey — ML/Networks
}


def load_data():
    """Load all data needed for the knowledge map."""
    with open(HIERARCHY_JSON) as f:
        hierarchy = json.load(f)
    with open(EXTRACTION_STATE) as f:
        extraction = json.load(f)
    embeddings = np.load(EMBEDDINGS_NPY)
    return hierarchy, extraction, embeddings


def build_concept_mappings(hierarchy):
    """Build concept -> meta-concept, theme, domain mappings."""
    mc_cm = hierarchy["levels"][0]["cluster_membership"]  # level 1: meta-concepts
    th_cm = hierarchy["levels"][1]["cluster_membership"]  # level 2: themes
    dm_cm = hierarchy["levels"][2]["cluster_membership"]  # level 3: domains

    concept_to_mc = {}
    for mc_id, concepts in mc_cm.items():
        for c in concepts:
            concept_to_mc[c] = mc_id

    concept_to_theme = {}
    for th_id, concepts in th_cm.items():
        for c in concepts:
            concept_to_theme[c] = th_id

    concept_to_domain = {}
    for dom_id, concepts in dm_cm.items():
        for c in concepts:
            concept_to_domain[c] = dom_id

    # Meta-concept -> domain (majority vote of member concepts)
    mc_to_domain = {}
    for mc_id, concepts in mc_cm.items():
        domain_counts = Counter(concept_to_domain.get(c, "?") for c in concepts)
        mc_to_domain[mc_id] = domain_counts.most_common(1)[0][0]

    # Meta-concept -> representative label (most frequent member concept)
    mc_labels = {}
    for mc_id, concepts in mc_cm.items():
        mc_labels[mc_id] = concepts[0]  # first one is typically the seed/most representative

    return concept_to_mc, mc_to_domain, mc_labels, mc_cm


def build_cooccurrence_network(extraction, concept_to_mc, mc_to_domain):
    """Build the meta-concept co-occurrence network from episode data."""
    # Episode -> set of meta-concepts
    episode_mcs = []
    mc_episode_count = Counter()

    for ep in extraction["episodes"]:
        mcs = set()
        for c in ep["concepts"]:
            c_lower = c.lower()
            mc = concept_to_mc.get(c_lower)
            if mc:
                mcs.add(mc)
        episode_mcs.append(mcs)
        for mc in mcs:
            mc_episode_count[mc] += 1

    # Build co-occurrence edges
    edge_weights = Counter()
    for mcs in episode_mcs:
        mcs_list = sorted(mcs)
        for i in range(len(mcs_list)):
            for j in range(i + 1, len(mcs_list)):
                edge_weights[(mcs_list[i], mcs_list[j])] += 1

    # Build NetworkX graph
    G = nx.Graph()
    for mc_id in mc_to_domain:
        G.add_node(mc_id, domain=mc_to_domain[mc_id],
                    frequency=mc_episode_count.get(mc_id, 0))

    for (u, v), w in edge_weights.items():
        if u in G and v in G:
            G.add_edge(u, v, weight=w)

    return G, mc_episode_count


def compute_layout(G, embeddings, hierarchy, concept_to_mc, method="domain_aware"):
    """Compute node positions.

    Methods:
      - "domain_aware": domains on a circle, spring layout within each domain
      - "sfdp": graphviz scalable force-directed (graph structure)
      - "tsne": t-SNE on mean meta-concept embeddings (semantic space)
      - "hybrid": t-SNE init → spring refinement with graph edges
    """
    mc_cm = hierarchy["levels"][0]["cluster_membership"]
    dm_cm = hierarchy["levels"][2]["cluster_membership"]  # domains

    # Build mc -> domain mapping
    concept_to_domain_local = {}
    for dom_id, concepts in dm_cm.items():
        for c in concepts:
            concept_to_domain_local[c] = dom_id
    mc_to_domain_local = {}
    for mc_id, concepts in mc_cm.items():
        domain_counts = Counter(concept_to_domain_local.get(c, "?") for c in concepts)
        mc_to_domain_local[mc_id] = domain_counts.most_common(1)[0][0]

    if method == "domain_aware":
        # Place domain centers on a circle, then spring-layout within each domain
        # Arrange domains around circle in conceptual order:
        # practical → theoretical gradient
        domain_order = [
            "MC3_C1",  # Software Engineering (largest, top)
            "MC3_C0",  # DevOps & Integration
            "MC3_C3",  # LLM Engineering
            "MC3_C7",  # ML & Network Science
            "MC3_C6",  # Math & Optimization
            "MC3_C2",  # Statistical Methods
            "MC3_C5",  # AI Safety & Research
            "MC3_C4",  # Philosophy & AI Theory
        ]
        n_domains = len(domain_order)
        domain_radius = 9.0  # radius for domain centers

        # Domain center positions on a circle
        domain_centers = {}
        for i, dom_id in enumerate(domain_order):
            angle = 2 * np.pi * i / n_domains - np.pi / 2
            domain_centers[dom_id] = np.array([
                domain_radius * np.cos(angle),
                domain_radius * np.sin(angle)
            ])

        # Group nodes by domain
        domain_nodes = defaultdict(list)
        for n in G.nodes():
            dom = mc_to_domain_local.get(n, domain_order[0])
            domain_nodes[dom].append(n)

        # Spring layout per domain subgraph
        pos = {}
        for dom_id, nodes in domain_nodes.items():
            if not nodes:
                continue
            subG = G.subgraph(nodes)
            center = domain_centers[dom_id]

            # Local radius scales with number of nodes (sqrt for area)
            local_radius = 1.5 + 0.12 * np.sqrt(len(nodes))

            if len(nodes) == 1:
                pos[nodes[0]] = center
            else:
                # Spring layout in local coordinates
                local_pos = nx.spring_layout(subG, k=1.5, iterations=80, seed=42)
                # Scale and translate to domain center
                coords = np.array([local_pos[n] for n in nodes])
                # Normalize to [-1, 1]
                if coords.max() != coords.min():
                    coords = 2 * (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-10) - 1
                coords *= local_radius
                for j, n in enumerate(nodes):
                    pos[n] = center + coords[j]

        print(f"Using domain-aware layout ({n_domains} domains, radius={domain_radius})")
        return pos

    if method == "sfdp":
        # Use graphviz sfdp on a SPARSIFIED graph for better community separation
        # The full graph (avg degree ~24) is too dense for visual clustering
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            # Build sparse layout graph: only keep strong edges
            G_sparse = nx.Graph()
            G_sparse.add_nodes_from(G.nodes())
            threshold = sorted([d["weight"] for _, _, d in G.edges(data=True)],
                              reverse=True)[min(len(G.edges()) // 4, 1500)]
            for u, v, d in G.edges(data=True):
                if d["weight"] >= threshold:
                    G_sparse.add_edge(u, v, weight=d["weight"])
            print(f"  Layout graph: {G_sparse.number_of_edges()} edges "
                  f"(threshold weight >= {threshold})")
            pos = graphviz_layout(G_sparse, prog="sfdp")
            print("Using graphviz sfdp layout (sparsified)")
            return pos
        except Exception as e:
            print(f"sfdp failed ({e}), falling back to spring")
            return nx.spring_layout(G, k=2.0, iterations=100, seed=42)

    # Build meta-concept embeddings for embedding-based layouts
    unique_concepts = []
    seen = set()
    for mc_id, concepts in mc_cm.items():
        for c in concepts:
            if c not in seen:
                unique_concepts.append(c)
                seen.add(c)

    concept_to_idx = {c: i for i, c in enumerate(unique_concepts)}
    mc_embeddings = {}
    for mc_id, concepts in mc_cm.items():
        if mc_id not in G:
            continue
        idxs = [concept_to_idx[c] for c in concepts if c in concept_to_idx]
        if idxs:
            mc_embeddings[mc_id] = embeddings[idxs].mean(axis=0)

    mc_ids = sorted(mc_embeddings.keys())
    mc_emb_matrix = np.array([mc_embeddings[mc] for mc in mc_ids])

    if method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=40,
                    learning_rate=200, max_iter=2000, metric="cosine")
        coords_2d = tsne.fit_transform(mc_emb_matrix)
        coords_2d -= coords_2d.min(axis=0)
        coords_2d /= coords_2d.max(axis=0)
        pos = {mc: coords_2d[i] for i, mc in enumerate(mc_ids)}
        print(f"Using t-SNE layout (KL divergence: {tsne.kl_divergence_:.3f})")
        return pos

    # hybrid: t-SNE then spring
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=40,
                learning_rate=200, max_iter=2000, metric="cosine")
    coords_2d = tsne.fit_transform(mc_emb_matrix)
    coords_2d -= coords_2d.min(axis=0)
    coords_2d /= coords_2d.max(axis=0)
    init_pos = {mc: coords_2d[i] for i, mc in enumerate(mc_ids)}
    pos = nx.spring_layout(G, pos=init_pos, k=0.1, iterations=20, seed=42)
    print(f"Using hybrid t-SNE + spring layout")
    return pos


def draw_knowledge_map(G, pos, mc_to_domain, mc_labels, mc_episode_count,
                        edge_weight_threshold=5, label_top_n=20):
    """Draw the publication-quality knowledge map."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14), facecolor="white")

    nodes = list(G.nodes())
    domains = [mc_to_domain.get(n, "MC3_C1") for n in nodes]
    colors = [DOMAIN_COLORS.get(d, "#cccccc") for d in domains]
    freqs = [mc_episode_count.get(n, 1) for n in nodes]

    # Node sizes: scale by log-frequency for less extreme range
    freq_arr = np.array(freqs, dtype=float)
    log_freq = np.log1p(freq_arr)
    min_size, max_size = 20, 500
    if log_freq.max() > log_freq.min():
        sizes = min_size + (max_size - min_size) * (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min())
    else:
        sizes = np.full_like(log_freq, 80)

    # Collect edges — different thresholds for intra vs cross-domain
    # Cross-domain edges ARE the story: show more of them
    # Intra-domain structure is obvious from spatial clustering
    from matplotlib.collections import LineCollection

    intra_segments = []
    intra_colors_list = []
    cross_segments = []
    cross_alphas = []

    all_weights = [d["weight"] for _, _, d in G.edges(data=True)]
    max_w = max(all_weights)

    CROSS_THRESHOLD = 3
    INTRA_THRESHOLD = 6

    for u, v, d in G.edges(data=True):
        w = d["weight"]
        seg = [(pos[u][0], pos[u][1]), (pos[v][0], pos[v][1])]
        if mc_to_domain.get(u) == mc_to_domain.get(v):
            if w >= INTRA_THRESHOLD:
                dom = mc_to_domain.get(u, "MC3_C1")
                c = mcolors.to_rgba(DOMAIN_COLORS.get(dom, "#cccccc"),
                                    alpha=0.08 + 0.25 * (w / max_w))
                intra_segments.append(seg)
                intra_colors_list.append(c)
        else:
            if w >= CROSS_THRESHOLD:
                a = 0.04 + 0.18 * (w / max_w)
                cross_segments.append(seg)
                cross_alphas.append(a)

    # Draw cross-domain edges first (behind) — the bridges are key
    if cross_segments:
        cross_colors = [mcolors.to_rgba("#777777", alpha=a) for a in cross_alphas]
        lc = LineCollection(cross_segments, colors=cross_colors, linewidths=0.4, zorder=1)
        ax.add_collection(lc)

    # Draw intra-domain edges
    if intra_segments:
        lc = LineCollection(intra_segments, colors=intra_colors_list, linewidths=0.5, zorder=2)
        ax.add_collection(lc)

    # Draw nodes
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    ax.scatter(node_x, node_y, s=sizes, c=colors,
               edgecolors="white", linewidths=0.5, zorder=4, alpha=0.9)

    # Domain labels — positioned outside each cluster along the radial direction
    for dom_id, dom_label in DOMAIN_LABELS.items():
        dom_nodes = [n for n in nodes if mc_to_domain.get(n) == dom_id]
        if not dom_nodes:
            continue
        cx = np.mean([pos[n][0] for n in dom_nodes])
        cy = np.mean([pos[n][1] for n in dom_nodes])
        # Push label outward from center
        dist = np.sqrt(cx**2 + cy**2) + 0.01
        label_x = cx + 2.5 * cx / dist
        label_y = cy + 2.5 * cy / dist
        ax.annotate(dom_label.replace("\n", " "), (label_x, label_y),
                    fontsize=12, fontweight="bold",
                    ha="center", va="center",
                    color=DOMAIN_COLORS[dom_id], alpha=0.85, zorder=6,
                    fontfamily="sans-serif",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=DOMAIN_COLORS[dom_id], alpha=0.9,
                              linewidth=1.5))

    # Label top-N most connected meta-concepts
    # Pick diverse labels: top 2 per domain for clarity
    labels_per_domain = defaultdict(list)
    degree_sorted = sorted(nodes, key=lambda n: G.degree(n, weight="weight"), reverse=True)
    for n in degree_sorted:
        dom = mc_to_domain.get(n)
        if len(labels_per_domain[dom]) < 2:
            labels_per_domain[dom].append(n)
    labeled_nodes = [n for ns in labels_per_domain.values() for n in ns]

    for n in labeled_nodes:
        label = mc_labels.get(n, n)
        if len(label) > 35:
            label = label[:33] + "…"
        x, y = pos[n]
        ax.annotate(label, (x, y), fontsize=5.5, fontstyle="italic",
                    ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points",
                    color="#444444", zorder=6)

    # No separate legend needed — domain labels on the graph serve as legend

    # Styling
    pad = 0.05
    x_range = max(node_x) - min(node_x)
    y_range = max(node_y) - min(node_y)
    ax.set_xlim(min(node_x) - pad * x_range, max(node_x) + pad * x_range)
    ax.set_ylim(min(node_y) - pad * y_range, max(node_y) + pad * y_range)
    ax.set_aspect("equal")
    ax.axis("off")

    # Stats annotation
    n_cross = len(cross_segments)
    stats_text = (f"{G.number_of_nodes()} meta-concepts  ·  "
                  f"{G.number_of_edges():,} co-occurrence edges  ·  "
                  f"{n_cross:,} cross-domain bridges")
    ax.text(0.5, 0.01, stats_text, transform=ax.transAxes,
            ha="center", fontsize=9, color="#888888")

    plt.tight_layout(pad=0.5)
    return fig


def export_gexf(G, mc_to_domain, mc_labels, mc_episode_count, mc_cm):
    """Export network as GEXF for Gephi refinement."""
    for n in G.nodes():
        G.nodes[n]["label"] = mc_labels.get(n, n)
        G.nodes[n]["domain"] = DOMAIN_LABELS.get(mc_to_domain.get(n, ""), "Unknown").replace("\n", " ")
        G.nodes[n]["domain_id"] = mc_to_domain.get(n, "")
        G.nodes[n]["frequency"] = mc_episode_count.get(n, 0)
        G.nodes[n]["n_concepts"] = len(mc_cm.get(n, []))

    out_path = FIGURES_DIR / "knowledge_map.gexf"
    nx.write_gexf(G, out_path)
    print(f"GEXF exported: {out_path}")


def main():
    print("Loading data...")
    hierarchy, extraction, embeddings = load_data()

    print("Building mappings...")
    concept_to_mc, mc_to_domain, mc_labels, mc_cm = build_concept_mappings(hierarchy)

    print("Building co-occurrence network...")
    G, mc_episode_count = build_cooccurrence_network(extraction, concept_to_mc, mc_to_domain)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Remove isolated nodes
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    print(f"  After removing {len(isolates)} isolates: {G.number_of_nodes()} nodes")

    print("Computing layout...")
    pos = compute_layout(G, embeddings, hierarchy, concept_to_mc)

    print("Drawing knowledge map...")
    fig = draw_knowledge_map(G, pos, mc_to_domain, mc_labels, mc_episode_count,
                              edge_weight_threshold=2, label_top_n=25)

    # Save
    for fmt in ["pdf", "png"]:
        out_path = FIGURES_DIR / f"knowledge_map.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {out_path}")
    plt.close(fig)

    print("Exporting GEXF...")
    export_gexf(G, mc_to_domain, mc_labels, mc_episode_count, mc_cm)

    print("Done.")


if __name__ == "__main__":
    main()
