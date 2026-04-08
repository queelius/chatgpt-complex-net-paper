#!/usr/bin/env python3
"""
Heaps' law null model and clustering sensitivity analysis.

Addresses review issues:
  C1 — Is β=0.320 an artifact of the k=500 ceiling?
  M7 — How sensitive are key metrics to the choice of k?
  M3 — Bootstrap CIs for β, Erdos-Renyi comparison for σ
  S4 — Chronological vs alphabetical episode ordering

Outputs:
  experiments/results/hierarchy_v2/heaps_null_model.json
  experiments/results/hierarchy_v2/clustering_sensitivity.json
  experiments/figures/heaps_law.pdf (updated with null band)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

RESULTS_DIR = Path(__file__).parent.parent / "results" / "hierarchy_v2"
FIGURES_DIR = Path(__file__).parent


def setup_style():
    plt.rcParams.update({
        'font.size': 13, 'font.family': 'serif', 'savefig.dpi': 300,
        'figure.dpi': 100, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    })
    sns.set_style("whitegrid")


def heaps_law(n, K, beta):
    return K * np.power(n, beta)


def compute_heaps_beta(episode_order, concept_to_cluster):
    """Compute Heaps' β for a given episode ordering and concept→cluster map."""
    vocab = set()
    counts = []
    for ep in episode_order:
        for concept in ep["concepts"]:
            c_lower = concept.lower()
            cluster = concept_to_cluster.get(c_lower)
            if cluster is not None:
                vocab.add(cluster)
        counts.append(len(vocab))

    x = np.arange(1, len(counts) + 1, dtype=float)
    y = np.array(counts, dtype=float)

    try:
        popt, _ = curve_fit(heaps_law, x, y, p0=[1.0, 0.5], maxfev=10000)
        return popt[1], popt[0], y  # beta, K, vocab_growth_curve
    except RuntimeError:
        return np.nan, np.nan, y


def build_concept_to_mc(hierarchy):
    """Build raw concept → meta-concept cluster ID mapping."""
    concept_to_mc = {}
    mc_level = hierarchy["levels"][0]
    for mc_id, raw_concepts in mc_level["cluster_membership"].items():
        for rc in raw_concepts:
            concept_to_mc[rc.lower()] = mc_id
    return concept_to_mc


def build_metaconcept_network(episodes, concept_to_mc):
    """Build the meta-concept co-occurrence network from the bipartite graph."""
    import networkx as nx

    # Episode → set of meta-concepts
    episode_mcs = []
    for ep in episodes:
        mcs = set()
        for c in ep["concepts"]:
            mc = concept_to_mc.get(c.lower())
            if mc is not None:
                mcs.add(mc)
        episode_mcs.append(mcs)

    # Co-occurrence: two meta-concepts linked if they share an episode
    edge_weights = defaultdict(int)
    for mcs in episode_mcs:
        mcs_list = sorted(mcs)
        for i, a in enumerate(mcs_list):
            for b in mcs_list[i + 1:]:
                edge_weights[(a, b)] += 1

    G = nx.Graph()
    for (a, b), w in edge_weights.items():
        G.add_edge(a, b, weight=w)

    return G


def compute_small_world(G, n_random=100, seed=42):
    """Compute small-world σ and comparison to Erdos-Renyi random graphs."""
    import networkx as nx

    if not nx.is_connected(G):
        gcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc_nodes).copy()

    n = G.number_of_nodes()
    m = G.number_of_edges()
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)

    # Random graph comparison
    rng = np.random.default_rng(seed)
    C_rands = []
    L_rands = []
    for i in range(n_random):
        R = nx.gnm_random_graph(n, m, seed=int(rng.integers(0, 2**31)))
        if nx.is_connected(R):
            C_rands.append(nx.average_clustering(R))
            L_rands.append(nx.average_shortest_path_length(R))

    C_rand = np.mean(C_rands)
    L_rand = np.mean(L_rands)
    sigma = (C / C_rand) / (L / L_rand) if C_rand > 0 and L_rand > 0 else np.nan

    return {
        "nodes": n, "edges": m,
        "C": round(C, 4), "C_rand": round(C_rand, 4), "C_ratio": round(C / C_rand, 2),
        "L": round(L, 4), "L_rand": round(L_rand, 4), "L_ratio": round(L / L_rand, 2),
        "sigma": round(sigma, 2),
        "C_rand_std": round(float(np.std(C_rands)), 4),
        "L_rand_std": round(float(np.std(L_rands)), 4),
        "n_random_graphs": len(C_rands),
    }


def null_model_permutation(episodes, concept_to_mc, n_permutations=1000, seed=42):
    """
    Null model: randomly reassign concepts to clusters, preserving cluster sizes.
    Compute Heaps' β for each permutation to get a null distribution.
    """
    rng = np.random.default_rng(seed)

    # Get all unique concepts that map to meta-concepts
    all_concepts = sorted(concept_to_mc.keys())
    original_clusters = [concept_to_mc[c] for c in all_concepts]

    null_betas = []
    for i in range(n_permutations):
        # Shuffle cluster assignments (preserving which cluster IDs exist and their sizes)
        shuffled = rng.permutation(original_clusters)
        shuffled_map = dict(zip(all_concepts, shuffled))
        beta, _, _ = compute_heaps_beta(episodes, shuffled_map)
        if not np.isnan(beta):
            null_betas.append(beta)
        if (i + 1) % 100 == 0:
            print(f"  Null model permutation {i + 1}/{n_permutations}")

    return np.array(null_betas)


def null_model_bipartite(episodes, concept_to_mc, n_permutations=1000, seed=42):
    """
    Stricter null model (R3): randomize meta-concept co-occurrence across episodes
    while preserving both marginals (meta-concepts per episode AND episodes per
    meta-concept). This is a bipartite configuration model.

    Uses the Curveball algorithm for efficient uniform sampling from the space
    of binary matrices with fixed row and column sums.
    """
    rng = np.random.default_rng(seed)

    # Build the episode x meta-concept binary matrix
    all_mcs = sorted(set(concept_to_mc.values()))
    mc_to_idx = {mc: i for i, mc in enumerate(all_mcs)}
    n_episodes = len(episodes)
    n_mcs = len(all_mcs)

    matrix = np.zeros((n_episodes, n_mcs), dtype=np.int8)
    for i, ep in enumerate(episodes):
        seen = set()
        for c in ep["concepts"]:
            mc = concept_to_mc.get(c.lower())
            if mc is not None and mc not in seen:
                seen.add(mc)
                matrix[i, mc_to_idx[mc]] = 1

    # Curveball: repeatedly pick two random rows, trade their column sets
    # while preserving row degrees. After enough trades, column degrees
    # are preserved in expectation and the matrix converges to a uniform
    # sample. Use n_episodes * 5 trades per sample as burn-in.
    n_trades = n_episodes * 5

    def compute_beta_from_matrix(mat):
        """Compute Heaps' beta from the episode x meta-concept matrix."""
        vocab = set()
        counts = []
        for i in range(mat.shape[0]):
            row_mcs = np.where(mat[i] > 0)[0]
            for j in row_mcs:
                vocab.add(j)
            counts.append(len(vocab))
        x = np.arange(1, len(counts) + 1, dtype=float)
        y = np.array(counts, dtype=float)
        try:
            popt, _ = curve_fit(heaps_law, x, y, p0=[1.0, 0.5], maxfev=10000)
            return popt[1]
        except RuntimeError:
            return np.nan

    def curveball_trade(mat, rng):
        """One Curveball trade: pick two random rows, swap exclusive columns."""
        r1, r2 = rng.choice(mat.shape[0], size=2, replace=False)
        cols1 = set(np.where(mat[r1] > 0)[0])
        cols2 = set(np.where(mat[r2] > 0)[0])
        exclusive1 = cols1 - cols2  # in r1 but not r2
        exclusive2 = cols2 - cols1  # in r2 but not r1
        if not exclusive1 or not exclusive2:
            return  # nothing to trade
        # Choose a random subset to swap
        tradeable = min(len(exclusive1), len(exclusive2))
        n_swap = rng.integers(1, tradeable + 1)
        swap_from_1 = rng.choice(list(exclusive1), size=n_swap, replace=False)
        swap_from_2 = rng.choice(list(exclusive2), size=n_swap, replace=False)
        for c in swap_from_1:
            mat[r1, c] = 0
            mat[r2, c] = 1
        for c in swap_from_2:
            mat[r2, c] = 0
            mat[r1, c] = 1

    null_betas = []
    for i in range(n_permutations):
        # Start from the original matrix each time and apply trades
        shuffled = matrix.copy()
        for _ in range(n_trades):
            curveball_trade(shuffled, rng)
        beta = compute_beta_from_matrix(shuffled)
        if not np.isnan(beta):
            null_betas.append(beta)
        if (i + 1) % 100 == 0:
            print(f"  Bipartite null permutation {i + 1}/{n_permutations}")

    return np.array(null_betas)


def beta_vs_k(episodes, hierarchy, ks=(50, 100, 200, 300, 400, 500, 600, 700, 1000)):
    """
    Compute Heaps' β as a function of the number of clusters k.
    Re-clusters the raw concepts at each k using Ward linkage.
    """
    # Load concept embeddings from the hierarchy data
    # We need to re-cluster, so we need the concept embedding matrix
    # The hierarchy_semantic.json has concept_co_occurrence but not embeddings
    # We'll use the hierarchy's dendrogram approach
    mc_level = hierarchy["levels"][0]  # level 1 = meta-concepts at k=500
    all_raw_concepts = []
    for mc_id, raw_concepts in mc_level["cluster_membership"].items():
        all_raw_concepts.extend(raw_concepts)
    all_raw_concepts = sorted(set(c.lower() for c in all_raw_concepts))

    # We need concept embeddings to re-cluster. Check if they exist.
    embeddings_file = RESULTS_DIR / "concept_embeddings.npy"

    if not embeddings_file.exists():
        print(f"  WARNING: {embeddings_file} not found. Cannot re-cluster.")
        print("  Using the existing hierarchy's cluster membership for k=500 only.")
        return None

    embeddings = np.load(embeddings_file)

    # Reconstruct concept order: the embeddings were created in
    # frequency-sorted order (most common concept first)
    from collections import Counter

    with open(RESULTS_DIR / "extraction_state.json") as f:
        es = json.load(f)
    concept_counter = Counter()
    for ep in es["episodes"]:
        for c in ep["concepts"]:
            concept_counter[c.lower()] += 1
    concept_order = [c for c, _ in concept_counter.most_common()]
    assert len(concept_order) == len(embeddings), (
        f"Concept count mismatch: {len(concept_order)} vs {len(embeddings)}"
    )

    # Build linkage once
    print("  Computing Ward linkage on concept embeddings...")
    Z = linkage(embeddings, method='ward', metric='euclidean')

    results = []
    for k in ks:
        labels = fcluster(Z, t=k, criterion='maxclust')
        concept_to_cluster = {}
        for idx, concept in enumerate(concept_order):
            concept_to_cluster[concept.lower()] = f"C{labels[idx]}"

        beta, K, _ = compute_heaps_beta(episodes, concept_to_cluster)

        # Build co-occurrence network for this k
        G = build_metaconcept_network(episodes, concept_to_cluster)
        sw = compute_small_world(G, n_random=20)

        # Silhouette
        if k < len(embeddings):
            sil = silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings)))
        else:
            sil = None

        results.append({
            "k": k, "beta": round(beta, 4), "K": round(K, 3),
            "sigma": sw["sigma"], "nodes": sw["nodes"], "edges": sw["edges"],
            "C": sw["C"], "L": sw["L"], "modularity": None,  # Would need Louvain
            "silhouette": round(sil, 4) if sil is not None else None,
        })
        print(f"  k={k}: β={beta:.4f}, σ={sw['sigma']:.2f}, nodes={sw['nodes']}, edges={sw['edges']}")

    return results


def bootstrap_beta(episodes, concept_to_mc, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI for Heaps' β by resampling episodes."""
    rng = np.random.default_rng(seed)
    n = len(episodes)
    betas = []

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        resampled = [episodes[idx] for idx in indices]
        beta, _, _ = compute_heaps_beta(resampled, concept_to_mc)
        if not np.isnan(beta):
            betas.append(beta)
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap {i + 1}/{n_bootstrap}")

    betas = np.array(betas)
    return {
        "mean": round(float(np.mean(betas)), 4),
        "std": round(float(np.std(betas)), 4),
        "ci_lower": round(float(np.percentile(betas, 2.5)), 4),
        "ci_upper": round(float(np.percentile(betas, 97.5)), 4),
        "n_bootstrap": n_bootstrap,
    }


def main():
    setup_style()

    # ── Load data ────────────────────────────────────────────────────
    print("Loading data...")
    with open(RESULTS_DIR / "extraction_state.json") as f:
        extraction_state = json.load(f)
    with open(RESULTS_DIR / "hierarchy_semantic.json") as f:
        hierarchy = json.load(f)

    episodes = extraction_state["episodes"]
    concept_to_mc = build_concept_to_mc(hierarchy)

    # ── C1: Real β ───────────────────────────────────────────────────
    print("\n=== Real Heaps' β (alphabetical order) ===")
    beta_real, K_real, growth_real = compute_heaps_beta(episodes, concept_to_mc)
    print(f"  β = {beta_real:.4f}, K = {K_real:.3f}")

    # ── S4: Chronological ordering ───────────────────────────────────
    print("\n=== Chronological ordering ===")
    # Episodes are in alphabetical order by ID. Check if we have timestamps.
    # The extraction_state only has episode_id and concepts, no dates.
    # Try to sort by episode ID which is often date-derived, or just note
    # that we use the existing order (which is alphabetical).
    # For a proper chronological test, we'd need conversation dates.
    # Let's also compute β for random orderings to show ordering effect.
    rng = np.random.default_rng(42)
    random_order_betas = []
    for i in range(100):
        shuffled_eps = list(episodes)
        rng.shuffle(shuffled_eps)
        b, _, _ = compute_heaps_beta(shuffled_eps, concept_to_mc)
        if not np.isnan(b):
            random_order_betas.append(b)
    random_order_betas = np.array(random_order_betas)
    print(f"  Alphabetical β = {beta_real:.4f}")
    print(f"  Random order β: mean={np.mean(random_order_betas):.4f}, "
          f"std={np.std(random_order_betas):.4f}, "
          f"range=[{np.min(random_order_betas):.4f}, {np.max(random_order_betas):.4f}]")

    # ── C1: Null model ───────────────────────────────────────────────
    print("\n=== Null model (1000 permutations) ===")
    null_betas = null_model_permutation(episodes, concept_to_mc, n_permutations=1000)
    null_mean = float(np.mean(null_betas))
    null_std = float(np.std(null_betas))
    # Two-sided: fraction of null betas as extreme as observed
    # Since real β > null β, test: P(null >= real)
    p_value = float(np.mean(null_betas >= beta_real))
    print(f"  Null β: mean={null_mean:.4f}, std={null_std:.4f}")
    print(f"  Real β={beta_real:.4f}, p-value (null ≥ real) = {p_value:.4f}")
    if p_value < 0.05:
        print("  ✓ Real β is significantly HIGHER than null → semantic structure creates meaningful distinctions")
    else:
        print("  ~ Real β is within null range → no significant difference from random clustering")

    # ── R3: Bipartite configuration null (stricter) ────────────────
    print("\n=== Bipartite null model (1000 permutations) ===")
    print("  (Randomizes episode-metaconcept assignments preserving both marginals)")
    bip_null_betas = null_model_bipartite(episodes, concept_to_mc, n_permutations=1000)
    bip_null_mean = float(np.mean(bip_null_betas))
    bip_null_std = float(np.std(bip_null_betas))
    bip_p_value = float(np.mean(bip_null_betas >= beta_real))
    print(f"  Bipartite null β: mean={bip_null_mean:.4f}, std={bip_null_std:.4f}")
    print(f"  Real β={beta_real:.4f}, p-value (null ≥ real) = {bip_p_value:.4f}")
    if bip_p_value < 0.05:
        print("  ✓ Real β significantly HIGHER than bipartite null → temporal exploration dynamics matter")
    else:
        print("  ~ Real β within bipartite null range → structure explainable by degree sequence alone")

    # ── M3: Bootstrap CI for β ───────────────────────────────────────
    print("\n=== Bootstrap CI for β ===")
    bootstrap = bootstrap_beta(episodes, concept_to_mc, n_bootstrap=1000)
    print(f"  β = {bootstrap['mean']:.4f} ± {bootstrap['std']:.4f}")
    print(f"  95% CI: [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]")

    # ── M1: Meta-concept network metrics ─────────────────────────────
    print("\n=== Meta-concept co-occurrence network ===")
    G = build_metaconcept_network(episodes, concept_to_mc)
    sw = compute_small_world(G, n_random=100)
    print(f"  Nodes: {sw['nodes']}, Edges: {sw['edges']}")
    print(f"  C={sw['C']}, C_rand={sw['C_rand']}, C/C_rand={sw['C_ratio']}")
    print(f"  L={sw['L']}, L_rand={sw['L_rand']}, L/L_rand={sw['L_ratio']}")
    print(f"  σ = {sw['sigma']}")

    # Degree distribution
    import networkx as nx
    if not nx.is_connected(G):
        gcc_nodes = max(nx.connected_components(G), key=len)
        G_gcc = G.subgraph(gcc_nodes).copy()
    else:
        G_gcc = G
    degrees = [d for _, d in G_gcc.degree()]
    degree_stats = {
        "mean": round(float(np.mean(degrees)), 2),
        "std": round(float(np.std(degrees)), 2),
        "max": int(np.max(degrees)),
        "min": int(np.min(degrees)),
    }

    # ── M7: Clustering sensitivity (if embeddings available) ─────────
    print("\n=== Clustering sensitivity analysis ===")
    sensitivity = beta_vs_k(episodes, hierarchy)

    # ── Save results ─────────────────────────────────────────────────
    print("\n=== Saving results ===")

    null_model_results = {
        "real_beta": round(beta_real, 4),
        "real_K": round(K_real, 3),
        "null_model": {
            "n_permutations": 1000,
            "mean_beta": round(null_mean, 4),
            "std_beta": round(null_std, 4),
            "p_value": round(p_value, 4),
            "min_beta": round(float(np.min(null_betas)), 4),
            "max_beta": round(float(np.max(null_betas)), 4),
            "percentiles": {
                "2.5": round(float(np.percentile(null_betas, 2.5)), 4),
                "5": round(float(np.percentile(null_betas, 5)), 4),
                "50": round(float(np.percentile(null_betas, 50)), 4),
                "95": round(float(np.percentile(null_betas, 95)), 4),
                "97.5": round(float(np.percentile(null_betas, 97.5)), 4),
            },
        },
        "bootstrap_ci": bootstrap,
        "ordering_analysis": {
            "alphabetical_beta": round(beta_real, 4),
            "random_order_mean": round(float(np.mean(random_order_betas)), 4),
            "random_order_std": round(float(np.std(random_order_betas)), 4),
            "random_order_range": [
                round(float(np.min(random_order_betas)), 4),
                round(float(np.max(random_order_betas)), 4),
            ],
        },
        "bipartite_null_model": {
            "description": "Bipartite configuration model: randomizes which meta-concepts "
                           "appear in which episodes, preserving both row and column degree "
                           "sequences. Stricter than cluster-permutation null.",
            "n_permutations": 1000,
            "mean_beta": round(bip_null_mean, 4),
            "std_beta": round(bip_null_std, 4),
            "p_value": round(bip_p_value, 4),
            "min_beta": round(float(np.min(bip_null_betas)), 4),
            "max_beta": round(float(np.max(bip_null_betas)), 4),
            "percentiles": {
                "2.5": round(float(np.percentile(bip_null_betas, 2.5)), 4),
                "5": round(float(np.percentile(bip_null_betas, 5)), 4),
                "50": round(float(np.percentile(bip_null_betas, 50)), 4),
                "95": round(float(np.percentile(bip_null_betas, 95)), 4),
                "97.5": round(float(np.percentile(bip_null_betas, 97.5)), 4),
            },
        },
        "metaconcept_network": {
            **sw,
            "degree_stats": degree_stats,
        },
    }

    with open(RESULTS_DIR / "heaps_null_model.json", "w") as f:
        json.dump(null_model_results, f, indent=2)
    print(f"  Saved heaps_null_model.json")

    if sensitivity is not None:
        with open(RESULTS_DIR / "clustering_sensitivity.json", "w") as f:
            json.dump({"sensitivity": sensitivity}, f, indent=2)
        print(f"  Saved clustering_sensitivity.json")

    # ── Updated figure with null model band ──────────────────────────
    print("\n=== Generating updated Heaps' law figure ===")

    # Also compute raw concept β
    raw_vocab = set()
    raw_counts = []
    for ep in episodes:
        for c in ep["concepts"]:
            raw_vocab.add(c.lower())
        raw_counts.append(len(raw_vocab))
    raw_counts = np.array(raw_counts, dtype=float)
    x = np.arange(1, len(episodes) + 1, dtype=float)
    popt_raw, _ = curve_fit(heaps_law, x, raw_counts, p0=[1.0, 0.9], maxfev=10000)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Vocabulary growth with null band
    ax = axes[0]
    ax.plot(x, raw_counts, '-', color='#2c3e50', linewidth=1.5, alpha=0.9,
            label=f'Raw concepts ($\\beta = {popt_raw[1]:.3f}$)')
    x_fit = np.linspace(1, len(episodes), 500)
    ax.plot(x_fit, heaps_law(x_fit, *popt_raw), '--', color='#e74c3c',
            linewidth=1.5, alpha=0.8,
            label=f'Heaps fit: $V(n) = {popt_raw[0]:.1f} \\cdot n^{{{popt_raw[1]:.3f}}}$')

    ax.plot(x, growth_real, '-', color='#2980b9', linewidth=1.5, alpha=0.9,
            label=f'Meta-concepts ($\\beta = {beta_real:.3f}$)')
    ax.plot(x_fit, heaps_law(x_fit, K_real, beta_real), '--', color='#e67e22',
            linewidth=1.5, alpha=0.8,
            label=f'Heaps fit: $V(n) = {K_real:.1f} \\cdot n^{{{beta_real:.3f}}}$')

    # Null model band
    null_lo = heaps_law(x_fit, K_real, np.percentile(null_betas, 2.5))
    null_hi = heaps_law(x_fit, K_real, np.percentile(null_betas, 97.5))
    ax.fill_between(x_fit, null_lo, null_hi, color='#2980b9', alpha=0.15,
                     label=f'Null 95% CI ($\\beta$ = {null_mean:.3f} ± {null_std:.3f})')

    # Linear reference
    linear_slope = raw_counts[-1] / len(episodes)
    ax.plot(x_fit, linear_slope * x_fit, ':', color='#bdc3c7', linewidth=1.2,
            alpha=0.7, label='Linear ($\\beta = 1$)')

    ax.set_xlabel('Number of episodes processed')
    ax.set_ylabel('Cumulative vocabulary size')
    ax.set_title("Vocabulary Growth: Heaps' Law", fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='#cccccc', fontsize=8)
    ax.set_xlim(0, len(episodes) + 20)
    ax.set_ylim(0, None)

    # Panel 2: Both null model distributions
    ax2 = axes[1]
    ax2.hist(null_betas, bins=40, density=True, color='#95a5a6', alpha=0.6,
             edgecolor='white', label=f'Cluster-permutation null ($\\mu$={null_mean:.3f})')
    ax2.hist(bip_null_betas, bins=40, density=True, color='#3498db', alpha=0.5,
             edgecolor='white', label=f'Bipartite config. null ($\\mu$={bip_null_mean:.3f})')
    ax2.axvline(beta_real, color='#e74c3c', linewidth=2, linestyle='-',
                label=f'Observed $\\beta$ = {beta_real:.3f}')

    # p-value annotations
    p_strs = []
    for name, pv in [("cluster", p_value), ("bipartite", bip_p_value)]:
        p_strs.append(f"{name}: {'p < 0.001' if pv < 0.001 else f'p = {pv:.3f}'}")
    ax2.text(0.95, 0.95, "\n".join(p_strs), transform=ax2.transAxes,
             fontsize=9, fontweight='bold', va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.set_xlabel("Heaps' exponent $\\beta$")
    ax2.set_ylabel('Density')
    ax2.set_title('Null Models: Cluster Permutation vs Bipartite', fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=8)

    plt.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'heaps_law.{fmt}',
                    format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved heaps_law.pdf/png")

    # ── Save canonical meta-concept network metrics ──────────────────
    mc_network_file = RESULTS_DIR / "metaconcept_network_metrics.json"
    with open(mc_network_file, "w") as f:
        json.dump(null_model_results["metaconcept_network"], f, indent=2)
    print(f"  Saved metaconcept_network_metrics.json")


if __name__ == '__main__':
    main()
