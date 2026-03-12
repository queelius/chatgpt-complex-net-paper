"""Concept-level network experiments (F78-F82).

Runs five analyses:
  F78: Weighted vs unweighted network comparison
  F79: Edge weight structure and community weight profiles
  F80: Concept network percolation (phase transition)
  F81: Concept entropy and information content
  F82: Concept network gravity model (mass × distance)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score

RESULTS_DIR = Path(__file__).parent / "results"


def load_concept_data():
    """Load concept embeddings and similarity matrix."""
    with open(RESULTS_DIR / "concept_embeddings.json") as f:
        meta = json.load(f)
    sim = np.load(RESULTS_DIR / "concept_similarity_matrix.npy")
    return meta["concepts"], sim


def build_network(concepts, sim, threshold=0.70, weighted=True):
    """Build concept network at given threshold."""
    G = nx.Graph()
    for c in concepts:
        G.add_node(c["index"], name=c["name"], platform=c["platform"],
                    source_community=c["source_community"])
    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                if weighted:
                    G.add_edge(i, j, weight=float(sim[i, j]))
                else:
                    G.add_edge(i, j)
    return G


def louvain_partition(G, **kwargs):
    import community.community_louvain as community_louvain
    return community_louvain.best_partition(G, random_state=42, **kwargs)


# ─── F78: Weighted vs Unweighted Comparison ───────────────────────────────

def run_weighted_comparison(threshold=0.70):
    """F78: Do edge weights change community structure and network metrics?"""
    print("\n" + "=" * 60)
    print("F78: Weighted vs Unweighted Network Comparison")
    print("=" * 60)

    concepts, sim = load_concept_data()
    G_w = build_network(concepts, sim, threshold, weighted=True)
    G_u = build_network(concepts, sim, threshold, weighted=False)

    import community.community_louvain as community_louvain

    part_w = community_louvain.best_partition(G_w, random_state=42)
    part_u = community_louvain.best_partition(G_u, random_state=42)

    Q_w = community_louvain.modularity(part_w, G_w)
    Q_u = community_louvain.modularity(part_u, G_u)

    nodes = sorted(G_w.nodes())
    nmi = normalized_mutual_info_score(
        [part_w[n] for n in nodes], [part_u[n] for n in nodes])

    # Centrality comparison
    bc_w = nx.betweenness_centrality(G_w, weight="weight")
    bc_u = nx.betweenness_centrality(G_u)
    pr_w = nx.pagerank(G_w, weight="weight")
    pr_u = nx.pagerank(G_u)

    rho_bc, p_bc = stats.spearmanr([bc_w[n] for n in nodes], [bc_u[n] for n in nodes])
    rho_pr, p_pr = stats.spearmanr([pr_w[n] for n in nodes], [pr_u[n] for n in nodes])

    # Clustering comparison
    cc_w = nx.average_clustering(G_w, weight="weight")
    cc_u = nx.average_clustering(G_u)

    # Cross-platform edges by weight
    cross_weights = []
    same_weights = []
    for u, v, d in G_w.edges(data=True):
        w = d["weight"]
        if concepts[u]["platform"] != concepts[v]["platform"]:
            cross_weights.append(w)
        else:
            same_weights.append(w)

    mw_U, mw_p = stats.mannwhitneyu(cross_weights, same_weights, alternative="two-sided")

    print(f"Weighted:   {len(set(part_w.values()))} communities, Q={Q_w:.4f}")
    print(f"Unweighted: {len(set(part_u.values()))} communities, Q={Q_u:.4f}")
    print(f"NMI(weighted, unweighted): {nmi:.4f}")
    print(f"\nCentrality correlation (weighted vs unweighted):")
    print(f"  Betweenness: ρ={rho_bc:.4f}")
    print(f"  PageRank: ρ={rho_pr:.4f}")
    print(f"\nClustering: weighted={cc_w:.4f}, unweighted={cc_u:.4f}")
    print(f"\nEdge weight by type:")
    print(f"  Cross-platform: mean={np.mean(cross_weights):.4f}, n={len(cross_weights)}")
    print(f"  Same-platform:  mean={np.mean(same_weights):.4f}, n={len(same_weights)}")
    print(f"  Mann-Whitney U={mw_U:.0f}, p={mw_p:.4g}")

    output = {
        "experiment": "F78: Weighted vs Unweighted Comparison",
        "threshold": threshold,
        "weighted": {
            "communities": len(set(part_w.values())),
            "modularity": round(Q_w, 4),
            "clustering": round(cc_w, 4),
        },
        "unweighted": {
            "communities": len(set(part_u.values())),
            "modularity": round(Q_u, 4),
            "clustering": round(cc_u, 4),
        },
        "nmi_weighted_unweighted": round(nmi, 4),
        "centrality_correlation": {
            "betweenness": {"rho": round(rho_bc, 4), "p": round(p_bc, 6)},
            "pagerank": {"rho": round(rho_pr, 4), "p": round(p_pr, 6)},
        },
        "edge_weight_by_type": {
            "cross_platform": {
                "mean": round(float(np.mean(cross_weights)), 4),
                "std": round(float(np.std(cross_weights)), 4),
                "n": len(cross_weights),
            },
            "same_platform": {
                "mean": round(float(np.mean(same_weights)), 4),
                "std": round(float(np.std(same_weights)), 4),
                "n": len(same_weights),
            },
            "mann_whitney_p": round(float(mw_p), 6),
        },
    }

    with open(RESULTS_DIR / "concept_weighted_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_weighted_comparison.json")
    return output


# ─── F79: Edge Weight Structure ───────────────────────────────────────────

def run_weight_structure(threshold=0.70):
    """F79: Analyze edge weight distribution and community weight profiles."""
    print("\n" + "=" * 60)
    print("F79: Edge Weight Structure and Community Profiles")
    print("=" * 60)

    concepts, sim = load_concept_data()
    G = build_network(concepts, sim, threshold)
    partition = louvain_partition(G)

    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"\nWeight distribution:")
    print(f"  Mean: {np.mean(weights):.4f}")
    print(f"  Std: {np.std(weights):.4f}")
    print(f"  Median: {np.median(weights):.4f}")
    print(f"  Skewness: {stats.skew(weights):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(weights):.4f}")
    print(f"  Range: [{min(weights):.4f}, {max(weights):.4f}]")

    # Strength distribution (weighted degree)
    strengths = dict(G.degree(weight="weight"))
    degrees = dict(G.degree())

    # Strength vs degree correlation
    s_vals = [strengths[n] for n in G.nodes()]
    d_vals = [degrees[n] for n in G.nodes()]
    rho_sd, p_sd = stats.spearmanr(s_vals, d_vals)
    print(f"\nStrength vs degree: ρ={rho_sd:.4f}, p={p_sd:.4g}")

    # Mean weight per node (strength/degree)
    mean_w_per_node = {n: strengths[n] / degrees[n] if degrees[n] > 0 else 0
                       for n in G.nodes()}

    # Platform comparison of mean weight
    cg_mean_w = [mean_w_per_node[n] for n in G.nodes()
                 if concepts[n]["platform"] == "chatgpt" and degrees[n] > 0]
    ag_mean_w = [mean_w_per_node[n] for n in G.nodes()
                 if concepts[n]["platform"] == "agentic" and degrees[n] > 0]

    mw_U, mw_p = stats.mannwhitneyu(cg_mean_w, ag_mean_w, alternative="two-sided")
    print(f"\nMean edge weight per node:")
    print(f"  ChatGPT: {np.mean(cg_mean_w):.4f}")
    print(f"  Agentic: {np.mean(ag_mean_w):.4f}")
    print(f"  Mann-Whitney p={mw_p:.4g}")

    # Community weight profiles
    community_weights = defaultdict(lambda: {"internal": [], "external": []})
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        if partition[u] == partition[v]:
            community_weights[partition[u]]["internal"].append(w)
        else:
            community_weights[partition[u]]["external"].append(w)
            community_weights[partition[v]]["external"].append(w)

    comm_profiles = []
    for comm_id in sorted(set(partition.values())):
        internal = community_weights[comm_id]["internal"]
        external = community_weights[comm_id]["external"]
        members = [n for n, c in partition.items() if c == comm_id]
        n_cg = sum(1 for n in members if concepts[n]["platform"] == "chatgpt")
        n_ag = sum(1 for n in members if concepts[n]["platform"] == "agentic")

        profile = {
            "community": comm_id,
            "size": len(members),
            "chatgpt": n_cg,
            "agentic": n_ag,
            "internal_edges": len(internal),
            "external_edges": len(external),
            "mean_internal_weight": round(float(np.mean(internal)), 4) if internal else 0,
            "mean_external_weight": round(float(np.mean(external)), 4) if external else 0,
            "weight_ratio": round(
                float(np.mean(internal)) / float(np.mean(external)), 3
            ) if internal and external and np.mean(external) > 0 else None,
        }
        comm_profiles.append(profile)
        print(f"\n  Community {comm_id} ({len(members)} nodes, {n_cg}CG+{n_ag}AG):")
        print(f"    Internal: {len(internal)} edges, mean w={profile['mean_internal_weight']:.4f}")
        print(f"    External: {len(external)} edges, mean w={profile['mean_external_weight']:.4f}")
        if profile["weight_ratio"]:
            print(f"    Weight ratio (int/ext): {profile['weight_ratio']:.3f}")

    # Disparity: is weight concentrated on few edges per node?
    # Compute normalized HHI of edge weights per node
    hhi_per_node = []
    for node in G.nodes():
        if degrees[node] < 2:
            continue
        nbr_weights = [G[node][nbr]["weight"] for nbr in G.neighbors(node)]
        total = sum(nbr_weights)
        shares = [w / total for w in nbr_weights]
        hhi = sum(s ** 2 for s in shares)
        # Normalized: (HHI - 1/k) / (1 - 1/k)
        k = len(nbr_weights)
        hhi_norm = (hhi - 1/k) / (1 - 1/k) if k > 1 else 0
        hhi_per_node.append(hhi_norm)

    mean_hhi = float(np.mean(hhi_per_node))
    print(f"\nWeight disparity (normalized HHI per node):")
    print(f"  Mean: {mean_hhi:.4f} (0=uniform, 1=concentrated)")
    print(f"  Std: {np.std(hhi_per_node):.4f}")

    output = {
        "experiment": "F79: Edge Weight Structure",
        "threshold": threshold,
        "weight_distribution": {
            "mean": round(float(np.mean(weights)), 4),
            "std": round(float(np.std(weights)), 4),
            "median": round(float(np.median(weights)), 4),
            "skewness": round(float(stats.skew(weights)), 4),
            "kurtosis": round(float(stats.kurtosis(weights)), 4),
            "min": round(float(min(weights)), 4),
            "max": round(float(max(weights)), 4),
        },
        "strength_degree_correlation": {"rho": round(rho_sd, 4), "p": round(p_sd, 6)},
        "mean_weight_by_platform": {
            "chatgpt": round(float(np.mean(cg_mean_w)), 4),
            "agentic": round(float(np.mean(ag_mean_w)), 4),
            "mann_whitney_p": round(float(mw_p), 6),
        },
        "community_profiles": comm_profiles,
        "weight_disparity_hhi": {
            "mean": round(mean_hhi, 4),
            "std": round(float(np.std(hhi_per_node)), 4),
        },
    }

    with open(RESULTS_DIR / "concept_weight_structure.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_weight_structure.json")
    return output


# ─── F80: Percolation Analysis ────────────────────────────────────────────

def run_percolation():
    """F80: Phase transition analysis — concept network percolation threshold."""
    print("\n" + "=" * 60)
    print("F80: Concept Network Percolation Analysis")
    print("=" * 60)

    concepts, sim = load_concept_data()
    n = len(concepts)

    # Sweep thresholds from low to high
    thresholds = np.arange(0.45, 0.96, 0.01)
    results = []

    for theta in thresholds:
        G = build_network(concepts, sim, float(theta))
        E = G.number_of_edges()
        if E == 0:
            results.append({
                "theta": round(float(theta), 3),
                "edges": 0, "gc_size": 0, "gc_fraction": 0,
                "n_components": n, "second_largest": 0,
                "density": 0, "avg_degree": 0,
            })
            continue

        components = sorted(nx.connected_components(G), key=len, reverse=True)
        gc_size = len(components[0])
        second = len(components[1]) if len(components) > 1 else 0

        results.append({
            "theta": round(float(theta), 3),
            "edges": E,
            "gc_size": gc_size,
            "gc_fraction": round(gc_size / n, 4),
            "n_components": len(components),
            "second_largest": second,
            "density": round(nx.density(G), 4),
            "avg_degree": round(2 * E / n, 2),
        })

    # Find percolation threshold (where GC drops below 50%)
    gc_fracs = [r["gc_fraction"] for r in results]
    theta_c = None
    for i in range(len(gc_fracs) - 1):
        if gc_fracs[i] >= 0.5 and gc_fracs[i + 1] < 0.5:
            theta_c = results[i + 1]["theta"]
            break

    # Find susceptibility peak (max second-largest component)
    susceptibility = [r["second_largest"] for r in results]
    peak_idx = np.argmax(susceptibility)
    theta_peak = results[peak_idx]["theta"]

    print(f"Percolation threshold (GC < 50%): θ_c ≈ {theta_c}")
    print(f"Susceptibility peak (max 2nd component): θ ≈ {theta_peak}, "
          f"size={susceptibility[peak_idx]}")

    # Print key thresholds
    for r in results:
        if r["theta"] in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            print(f"  θ={r['theta']:.2f}: {r['edges']} edges, GC={r['gc_fraction']:.1%}, "
                  f"{r['n_components']} components, 2nd={r['second_largest']}")

    # Phase transition sharpness: derivative of GC fraction
    gc_derivative = []
    for i in range(1, len(results)):
        dGC = results[i]["gc_fraction"] - results[i-1]["gc_fraction"]
        dTheta = results[i]["theta"] - results[i-1]["theta"]
        gc_derivative.append({
            "theta": round((results[i]["theta"] + results[i-1]["theta"]) / 2, 3),
            "dGC_dTheta": round(dGC / dTheta if dTheta != 0 else 0, 4),
        })

    # Find steepest drop
    steepest = min(gc_derivative, key=lambda x: x["dGC_dTheta"])
    print(f"\nSteepest GC drop: θ≈{steepest['theta']}, dGC/dθ={steepest['dGC_dTheta']:.2f}")

    # Cross-platform GC composition at key thresholds
    cross_platform_gc = []
    for theta_val in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        G = build_network(concepts, sim, theta_val)
        if G.number_of_edges() == 0:
            continue
        gc = max(nx.connected_components(G), key=len)
        cg_in_gc = sum(1 for n in gc if concepts[n]["platform"] == "chatgpt")
        ag_in_gc = sum(1 for n in gc if concepts[n]["platform"] == "agentic")
        cross_platform_gc.append({
            "theta": theta_val,
            "gc_size": len(gc),
            "chatgpt": cg_in_gc,
            "agentic": ag_in_gc,
            "agentic_fraction": round(ag_in_gc / len(gc), 3) if gc else 0,
        })

    print(f"\nPlatform composition of GC:")
    for r in cross_platform_gc:
        print(f"  θ={r['theta']:.2f}: GC={r['gc_size']}, "
              f"{r['chatgpt']}CG+{r['agentic']}AG ({r['agentic_fraction']:.1%} agentic)")

    output = {
        "experiment": "F80: Concept Network Percolation",
        "n_concepts": n,
        "percolation_threshold": theta_c,
        "susceptibility_peak": theta_peak,
        "steepest_drop": steepest,
        "sweep_results": results,
        "gc_derivative": gc_derivative,
        "cross_platform_gc": cross_platform_gc,
    }

    with open(RESULTS_DIR / "concept_percolation.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_percolation.json")
    return output


# ─── F81: Network Entropy ────────────────────────────────────────────────

def run_entropy(threshold=0.70):
    """F81: Structural entropy and information content of concept network."""
    print("\n" + "=" * 60)
    print("F81: Concept Network Entropy Analysis")
    print("=" * 60)

    concepts, sim = load_concept_data()
    G = build_network(concepts, sim, threshold)
    n = G.number_of_nodes()
    partition = louvain_partition(G)

    # 1. Degree distribution entropy
    degrees = [G.degree(node) for node in G.nodes()]
    degree_counts = Counter(degrees)
    total = sum(degree_counts.values())
    degree_probs = [c / total for c in degree_counts.values()]
    H_degree = -sum(p * np.log2(p) for p in degree_probs if p > 0)

    # Maximum possible (uniform)
    H_degree_max = np.log2(len(degree_counts))

    print(f"Degree distribution entropy: H={H_degree:.3f} bits "
          f"(max={H_degree_max:.3f}, ratio={H_degree/H_degree_max:.3f})")

    # 2. Community size entropy
    comm_sizes = Counter(partition.values())
    total_nodes = sum(comm_sizes.values())
    comm_probs = [c / total_nodes for c in comm_sizes.values()]
    H_community = -sum(p * np.log2(p) for p in comm_probs if p > 0)
    H_community_max = np.log2(len(comm_sizes))

    print(f"Community size entropy: H={H_community:.3f} bits "
          f"(max={H_community_max:.3f}, ratio={H_community/H_community_max:.3f})")

    # 3. Weight distribution entropy
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    # Bin weights for entropy calculation
    hist, _ = np.histogram(weights, bins=20, density=True)
    hist = hist / hist.sum()  # Normalize
    H_weight = -sum(p * np.log2(p) for p in hist if p > 0)
    H_weight_max = np.log2(len(hist))

    print(f"Weight entropy: H={H_weight:.3f} bits "
          f"(max={H_weight_max:.3f}, ratio={H_weight/H_weight_max:.3f})")

    # 4. Platform mixing entropy per community
    comm_mixing_entropy = []
    for comm_id in sorted(set(partition.values())):
        members = [n for n, c in partition.items() if c == comm_id]
        cg = sum(1 for n in members if concepts[n]["platform"] == "chatgpt")
        ag = sum(1 for n in members if concepts[n]["platform"] == "agentic")
        total_c = cg + ag
        if total_c > 1 and cg > 0 and ag > 0:
            p_cg = cg / total_c
            p_ag = ag / total_c
            h = -(p_cg * np.log2(p_cg) + p_ag * np.log2(p_ag))
        else:
            h = 0.0
        comm_mixing_entropy.append({
            "community": comm_id,
            "size": len(members),
            "chatgpt": cg,
            "agentic": ag,
            "mixing_entropy": round(h, 4),
        })

    mean_mixing = np.mean([c["mixing_entropy"] for c in comm_mixing_entropy])
    print(f"\nPlatform mixing entropy per community:")
    for c in comm_mixing_entropy:
        print(f"  C{c['community']}: H={c['mixing_entropy']:.3f} "
              f"({c['chatgpt']}CG+{c['agentic']}AG)")
    print(f"Mean mixing entropy: {mean_mixing:.3f} (max=1.0 for perfect 50/50)")

    # 5. Neighborhood diversity entropy
    nbr_entropy = []
    for node in G.nodes():
        nbrs = list(G.neighbors(node))
        if len(nbrs) < 2:
            nbr_entropy.append(0)
            continue
        # Platform diversity of neighbors
        plats = Counter(concepts[n]["platform"] for n in nbrs)
        total_n = sum(plats.values())
        probs = [c / total_n for c in plats.values()]
        h = -sum(p * np.log2(p) for p in probs if p > 0)
        nbr_entropy.append(h)

    cg_nbr_h = [nbr_entropy[n] for n in G.nodes()
                if concepts[n]["platform"] == "chatgpt" and G.degree(n) >= 2]
    ag_nbr_h = [nbr_entropy[n] for n in G.nodes()
                if concepts[n]["platform"] == "agentic" and G.degree(n) >= 2]

    print(f"\nNeighborhood platform diversity entropy:")
    print(f"  ChatGPT nodes: mean H={np.mean(cg_nbr_h):.3f}")
    print(f"  Agentic nodes: mean H={np.mean(ag_nbr_h):.3f}")

    # 6. Von Neumann entropy of the graph
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.linalg.eigvalsh(L)
    # Normalize eigenvalues to form a density matrix
    trace_L = np.sum(eigenvalues)
    if trace_L > 0:
        rho = eigenvalues / trace_L
        H_vn = -sum(r * np.log2(r) for r in rho if r > 1e-15)
    else:
        H_vn = 0
    H_vn_max = np.log2(n)

    print(f"\nVon Neumann entropy: H={H_vn:.3f} bits "
          f"(max={H_vn_max:.3f}, ratio={H_vn/H_vn_max:.3f})")

    output = {
        "experiment": "F81: Concept Network Entropy",
        "threshold": threshold,
        "degree_entropy": {"H": round(H_degree, 4), "H_max": round(H_degree_max, 4),
                           "ratio": round(H_degree / H_degree_max, 4)},
        "community_entropy": {"H": round(H_community, 4), "H_max": round(H_community_max, 4),
                              "ratio": round(H_community / H_community_max, 4)},
        "weight_entropy": {"H": round(H_weight, 4), "H_max": round(H_weight_max, 4),
                           "ratio": round(H_weight / H_weight_max, 4)},
        "von_neumann_entropy": {"H": round(H_vn, 4), "H_max": round(H_vn_max, 4),
                                "ratio": round(H_vn / H_vn_max, 4)},
        "community_mixing_entropy": comm_mixing_entropy,
        "mean_mixing_entropy": round(float(mean_mixing), 4),
        "neighborhood_diversity": {
            "chatgpt_mean": round(float(np.mean(cg_nbr_h)), 4),
            "agentic_mean": round(float(np.mean(ag_nbr_h)), 4),
        },
    }

    with open(RESULTS_DIR / "concept_entropy.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_entropy.json")
    return output


# ─── F82: Gravity Model ──────────────────────────────────────────────────

def run_gravity_model(threshold=0.70):
    """F82: Gravity model — does community size predict cross-community flow?"""
    print("\n" + "=" * 60)
    print("F82: Concept Network Gravity Model")
    print("=" * 60)

    concepts, sim = load_concept_data()
    G = build_network(concepts, sim, threshold)
    partition = louvain_partition(G)

    # Community properties
    comms = defaultdict(list)
    for node, comm in partition.items():
        comms[comm].append(node)

    # For each pair of communities, compute:
    # - "mass" = community size
    # - "distance" = mean shortest path between communities
    # - "flow" = number of cross-community edges
    comm_ids = sorted(comms.keys())

    # Precompute shortest paths (subsample for large communities)
    rng = np.random.default_rng(42)
    gravity_data = []

    for i, c1 in enumerate(comm_ids):
        for j, c2 in enumerate(comm_ids):
            if c1 >= c2:
                continue

            nodes1 = comms[c1]
            nodes2 = comms[c2]

            # Count edges between communities
            flow = sum(1 for u, v in G.edges()
                       if (partition[u] == c1 and partition[v] == c2)
                       or (partition[u] == c2 and partition[v] == c1))

            # Mean edge weight between communities
            cross_weights = [G[u][v]["weight"] for u, v in G.edges()
                            if (partition[u] == c1 and partition[v] == c2)
                            or (partition[u] == c2 and partition[v] == c1)]
            mean_weight = float(np.mean(cross_weights)) if cross_weights else 0

            # Sample shortest paths
            n_sample = min(50, len(nodes1) * len(nodes2))
            paths = []
            for _ in range(n_sample):
                n1 = rng.choice(nodes1)
                n2 = rng.choice(nodes2)
                try:
                    p = nx.shortest_path_length(G, n1, n2)
                    paths.append(p)
                except nx.NetworkXNoPath:
                    pass

            mean_dist = float(np.mean(paths)) if paths else float('inf')

            # Gravity: mass1 * mass2 / distance^2
            mass1 = len(nodes1)
            mass2 = len(nodes2)
            gravity = (mass1 * mass2) / (mean_dist ** 2) if mean_dist > 0 else 0

            # Platform mixing
            cg1 = sum(1 for n in nodes1 if concepts[n]["platform"] == "chatgpt")
            ag1 = len(nodes1) - cg1
            cg2 = sum(1 for n in nodes2 if concepts[n]["platform"] == "chatgpt")
            ag2 = len(nodes2) - cg2

            gravity_data.append({
                "comm_pair": f"{c1}-{c2}",
                "size1": mass1,
                "size2": mass2,
                "platform1": f"{cg1}CG+{ag1}AG",
                "platform2": f"{cg2}CG+{ag2}AG",
                "flow": flow,
                "mean_weight": round(mean_weight, 4),
                "mean_distance": round(mean_dist, 3),
                "gravity": round(gravity, 2),
            })

    # Correlation: gravity vs flow
    gravities = [d["gravity"] for d in gravity_data if d["flow"] > 0]
    flows = [d["flow"] for d in gravity_data if d["flow"] > 0]

    if len(gravities) >= 3:
        rho_gf, p_gf = stats.spearmanr(gravities, flows)
        print(f"\nGravity vs flow correlation: ρ={rho_gf:.4f}, p={p_gf:.4g}")
    else:
        rho_gf, p_gf = 0, 1

    # Also correlate mass product vs flow
    mass_products = [d["size1"] * d["size2"] for d in gravity_data if d["flow"] > 0]
    if len(mass_products) >= 3:
        rho_mf, p_mf = stats.spearmanr(mass_products, flows)
        print(f"Mass product vs flow: ρ={rho_mf:.4f}, p={p_mf:.4g}")
    else:
        rho_mf, p_mf = 0, 1

    # Print top flows
    gravity_data.sort(key=lambda x: -x["flow"])
    print(f"\nTop community pairs by flow:")
    for d in gravity_data[:7]:
        print(f"  {d['comm_pair']}: flow={d['flow']}, "
              f"sizes={d['size1']}×{d['size2']}, "
              f"dist={d['mean_distance']:.2f}, "
              f"gravity={d['gravity']:.1f}")

    output = {
        "experiment": "F82: Gravity Model",
        "threshold": threshold,
        "gravity_flow_correlation": {"rho": round(rho_gf, 4), "p": round(p_gf, 6)},
        "mass_flow_correlation": {"rho": round(rho_mf, 4), "p": round(p_mf, 6)},
        "community_pairs": gravity_data,
    }

    with open(RESULTS_DIR / "concept_gravity.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_gravity.json")
    return output


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Concept-level experiments F78-F82")
    parser.add_argument("--f78", action="store_true", help="Weighted vs unweighted")
    parser.add_argument("--f79", action="store_true", help="Weight structure")
    parser.add_argument("--f80", action="store_true", help="Percolation")
    parser.add_argument("--f81", action="store_true", help="Entropy")
    parser.add_argument("--f82", action="store_true", help="Gravity model")
    parser.add_argument("--all", action="store_true", help="Run all")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    if not any([args.f78, args.f79, args.f80, args.f81, args.f82, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.f78 or args.all:
        run_weighted_comparison(args.threshold)
    if args.f79 or args.all:
        run_weight_structure(args.threshold)
    if args.f80 or args.all:
        run_percolation()
    if args.f81 or args.all:
        run_entropy(args.threshold)
    if args.f82 or args.all:
        run_gravity_model(args.threshold)
