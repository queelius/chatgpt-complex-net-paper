"""Concept-level network experiments (F83-F87).

Runs five analyses:
  F83: Dimensionality reduction (t-SNE + PCA of similarity space)
  F84: Hierarchical community structure (dendrogram from similarity)
  F85: Triad significance profile (motif census vs null models)
  F86: Community semantic coherence (intra- vs inter-community similarity)
  F87: Backbone extraction via threshold sweep (minimum spanning tree + graduated)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

RESULTS_DIR = Path(__file__).parent / "results"


def load_concept_data():
    """Load concept embeddings and similarity matrix."""
    with open(RESULTS_DIR / "concept_embeddings.json") as f:
        meta = json.load(f)
    sim = np.load(RESULTS_DIR / "concept_similarity_matrix.npy")
    return meta["concepts"], sim


def build_network(concepts, sim, threshold=0.70):
    """Build unweighted concept network at given threshold."""
    G = nx.Graph()
    for c in concepts:
        G.add_node(c["index"], name=c["name"], platform=c["platform"],
                    source_community=c["source_community"])
    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                G.add_edge(i, j, weight=float(sim[i, j]))
    return G


def get_louvain_communities(G, seed=42):
    """Get Louvain communities."""
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, random_state=seed)
        return partition
    except ImportError:
        print("WARNING: python-louvain not installed")
        return {n: 0 for n in G.nodes()}


def run_f83(concepts, sim, threshold=0.70):
    """F83: Dimensionality reduction of concept similarity space.

    Apply t-SNE and PCA to the similarity matrix (converted to distance).
    Analyze cluster separation by platform and community.
    """
    print("=" * 60)
    print("F83: Dimensionality Reduction of Concept Space")
    print("=" * 60)

    n = len(concepts)
    # Convert similarity to distance
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    # Ensure symmetric and non-negative
    dist = np.maximum(dist, 0.0)
    dist = (dist + dist.T) / 2

    # --- PCA on similarity matrix (treating rows as feature vectors) ---
    pca = PCA(n_components=min(10, n))
    pca_coords = pca.fit_transform(sim)
    var_explained = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_explained)

    print(f"\nPCA explained variance:")
    for i in range(min(5, len(var_explained))):
        print(f"  PC{i+1}: {var_explained[i]:.4f} (cum: {cumvar[i]:.4f})")
    print(f"  PCs for 90% variance: {int(np.searchsorted(cumvar, 0.90)) + 1}")
    print(f"  PCs for 95% variance: {int(np.searchsorted(cumvar, 0.95)) + 1}")

    # --- t-SNE (2D) ---
    tsne = TSNE(n_components=2, metric="precomputed", random_state=42,
                perplexity=min(30, n - 1), init="random")
    tsne_coords = tsne.fit_transform(dist)

    # --- Cluster quality metrics ---
    # Get Louvain communities for coloring
    G = build_network(concepts, sim, threshold)
    partition = get_louvain_communities(G)

    platforms = [c["platform"] for c in concepts]
    communities = [partition.get(i, -1) for i in range(n)]

    # Silhouette-like metric: mean intra-class sim minus mean inter-class sim
    def class_separation(labels):
        unique = list(set(labels))
        if len(unique) < 2:
            return 0.0
        intra_sims, inter_sims = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    intra_sims.append(sim[i, j])
                else:
                    inter_sims.append(sim[i, j])
        return np.mean(intra_sims) - np.mean(inter_sims) if intra_sims and inter_sims else 0.0

    platform_sep = class_separation(platforms)
    community_sep = class_separation(communities)

    print(f"\nClass separation (mean intra - mean inter similarity):")
    print(f"  By platform:  {platform_sep:.4f}")
    print(f"  By community: {community_sep:.4f}")
    print(f"  Ratio: community/platform = {community_sep / platform_sep:.2f}x")

    # t-SNE centroid distances
    plat_labels = np.array([0 if p == "chatgpt" else 1 for p in platforms])
    cg_centroid = tsne_coords[plat_labels == 0].mean(axis=0)
    ag_centroid = tsne_coords[plat_labels == 1].mean(axis=0)
    centroid_dist = np.linalg.norm(cg_centroid - ag_centroid)
    overall_spread = np.std(tsne_coords)
    print(f"\nt-SNE platform separation:")
    print(f"  Centroid distance: {centroid_dist:.2f}")
    print(f"  Overall spread (std): {overall_spread:.2f}")
    print(f"  Normalized separation: {centroid_dist / overall_spread:.3f}")

    # Platform overlap in t-SNE: fraction of agentic points within ChatGPT convex hull
    from scipy.spatial import ConvexHull
    cg_pts = tsne_coords[plat_labels == 0]
    ag_pts = tsne_coords[plat_labels == 1]

    try:
        hull = ConvexHull(cg_pts)
        # Check which agentic points fall inside
        from scipy.spatial import Delaunay
        hull_delaunay = Delaunay(cg_pts[hull.vertices])
        ag_inside = sum(hull_delaunay.find_simplex(ag_pts) >= 0)
        ag_overlap = ag_inside / len(ag_pts)
        print(f"  Agentic inside ChatGPT hull: {ag_inside}/{len(ag_pts)} ({ag_overlap:.1%})")
    except Exception as e:
        ag_overlap = None
        print(f"  Convex hull failed: {e}")

    # Save results
    result = {
        "method": "PCA + t-SNE on 115×115 similarity matrix",
        "pca": {
            "explained_variance": [float(v) for v in var_explained[:10]],
            "cumulative_variance": [float(v) for v in cumvar[:10]],
            "pcs_for_90pct": int(np.searchsorted(cumvar, 0.90)) + 1,
            "pcs_for_95pct": int(np.searchsorted(cumvar, 0.95)) + 1,
        },
        "tsne": {
            "perplexity": min(30, n - 1),
            "kl_divergence": float(tsne.kl_divergence_),
            "centroid_distance": float(centroid_dist),
            "overall_spread": float(overall_spread),
            "normalized_separation": float(centroid_dist / overall_spread),
            "agentic_in_chatgpt_hull": float(ag_overlap) if ag_overlap is not None else None,
        },
        "class_separation": {
            "platform": float(platform_sep),
            "community": float(community_sep),
            "ratio": float(community_sep / platform_sep) if platform_sep != 0 else None,
        },
        "concept_coordinates_tsne": [
            {
                "name": concepts[i]["name"],
                "platform": concepts[i]["platform"],
                "community": communities[i],
                "x": float(tsne_coords[i, 0]),
                "y": float(tsne_coords[i, 1]),
            }
            for i in range(n)
        ],
        "concept_coordinates_pca": [
            {
                "name": concepts[i]["name"],
                "platform": concepts[i]["platform"],
                "community": communities[i],
                "pc1": float(pca_coords[i, 0]),
                "pc2": float(pca_coords[i, 1]),
            }
            for i in range(n)
        ],
    }
    with open(RESULTS_DIR / "concept_dimensionality_reduction.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_dimensionality_reduction.json")


def run_f84(concepts, sim, threshold=0.70):
    """F84: Hierarchical community structure via agglomerative clustering.

    Build dendrogram from similarity matrix, cut at various levels,
    compare with Louvain communities.
    """
    print("\n" + "=" * 60)
    print("F84: Hierarchical Community Structure")
    print("=" * 60)

    n = len(concepts)
    # Convert to condensed distance matrix
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, 0.0)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist)

    # Hierarchical clustering with different linkage methods
    methods = ["average", "complete", "ward"]
    results_by_method = {}

    G = build_network(concepts, sim, threshold)
    louvain_partition = get_louvain_communities(G)
    louvain_labels = [louvain_partition.get(i, -1) for i in range(n)]

    for method in methods:
        if method == "ward":
            # Ward needs Euclidean distance — use PCA coordinates
            pca = PCA(n_components=20)
            coords = pca.fit_transform(sim)
            Z = linkage(coords, method="ward")
        else:
            Z = linkage(condensed, method=method)

        # Cut at various numbers of clusters
        cuts = {}
        for k in [3, 5, 7, 9, 11, 15]:
            labels = fcluster(Z, t=k, criterion="maxclust")
            nmi = normalized_mutual_info_score(louvain_labels, labels)
            # Platform mixing
            mixed = 0
            for c in set(labels):
                members = [i for i in range(n) if labels[i] == c]
                plats = set(concepts[i]["platform"] for i in members)
                if len(plats) > 1:
                    mixed += 1
            total_clusters = len(set(labels))
            cuts[k] = {
                "nmi_vs_louvain": float(nmi),
                "n_clusters": total_clusters,
                "mixed_clusters": mixed,
                "mixed_pct": float(mixed / total_clusters) if total_clusters > 0 else 0,
            }

        results_by_method[method] = cuts
        # Best NMI
        best_k = max(cuts, key=lambda k: cuts[k]["nmi_vs_louvain"])
        best_nmi = cuts[best_k]["nmi_vs_louvain"]
        print(f"\n{method.upper()} linkage:")
        for k, info in sorted(cuts.items()):
            print(f"  k={k:2d}: NMI={info['nmi_vs_louvain']:.3f}, "
                  f"mixed={info['mixed_clusters']}/{info['n_clusters']} ({info['mixed_pct']:.0%})")
        print(f"  Best match to Louvain: k={best_k} (NMI={best_nmi:.3f})")

    # Cophenetic correlation (how well dendrogram preserves pairwise distances)
    for method in ["average", "complete"]:
        Z = linkage(condensed, method=method)
        from scipy.cluster.hierarchy import cophenet
        coph_corr, _ = cophenet(Z, condensed)
        print(f"\nCophenetic correlation ({method}): {coph_corr:.4f}")
        results_by_method[method]["cophenetic_correlation"] = float(coph_corr)

    # Optimal cut: which k gives highest NMI?
    print(f"\nOverall best hierarchical match to Louvain:")
    for method in methods:
        best_k = max(results_by_method[method],
                     key=lambda k: results_by_method[method][k]["nmi_vs_louvain"]
                     if isinstance(results_by_method[method][k], dict) else -1)
        if isinstance(results_by_method[method][best_k], dict):
            print(f"  {method}: k={best_k}, NMI={results_by_method[method][best_k]['nmi_vs_louvain']:.3f}")

    result = {
        "method": "Agglomerative clustering (average, complete, ward) on 115×115 distance matrix",
        "louvain_n_communities": len(set(louvain_labels)),
        "linkage_results": results_by_method,
    }
    with open(RESULTS_DIR / "concept_hierarchical.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_hierarchical.json")


def run_f85(concepts, sim, threshold=0.70, n_random=100):
    """F85: Triad significance profile (motif census vs null models).

    Count all triads by type, compare to Erdős-Rényi and configuration model nulls.
    Compute z-scores and significance profile.
    """
    print("\n" + "=" * 60)
    print("F85: Triad Significance Profile")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"Network: {n_nodes} nodes, {n_edges} edges")

    # Count triads by platform composition
    platforms = {c["index"]: c["platform"] for c in concepts}
    nodes = list(G.nodes())

    # Count triangles
    triangles = []
    for u in nodes:
        for v in G.neighbors(u):
            if v > u:
                for w in G.neighbors(u):
                    if w > v and G.has_edge(v, w):
                        triangles.append((u, v, w))

    n_triangles = len(triangles)
    print(f"Triangles: {n_triangles}")

    # Platform composition census
    def triangle_type(u, v, w):
        plats = sorted([platforms[u], platforms[v], platforms[w]])
        n_ag = sum(1 for p in plats if p == "agentic")
        types = {0: "CCC", 1: "CCA", 2: "CAA", 3: "AAA"}
        return types[n_ag]

    type_counts = Counter(triangle_type(u, v, w) for u, v, w in triangles)
    print(f"\nTriangle type census:")
    for t in ["CCC", "CCA", "CAA", "AAA"]:
        print(f"  {t}: {type_counts.get(t, 0)}")

    # Connected triples (open triads = path of length 2)
    open_triads = 0
    for v in nodes:
        d = G.degree(v)
        open_triads += d * (d - 1) // 2
    open_triads -= n_triangles  # subtract closed triads
    print(f"Open triads (paths): {open_triads}")
    print(f"Global clustering: {3 * n_triangles / (3 * n_triangles + open_triads):.4f}" if (3 * n_triangles + open_triads) > 0 else "")

    # --- Null model comparison ---
    # Erdős-Rényi null
    p_edge = 2 * n_edges / (n_nodes * (n_nodes - 1))
    er_expected_triangles = (n_nodes * (n_nodes - 1) * (n_nodes - 2) / 6) * (p_edge ** 3)
    print(f"\nErdős-Rényi expected triangles: {er_expected_triangles:.1f}")
    print(f"Observed/expected ratio: {n_triangles / er_expected_triangles:.2f}")

    # Configuration model null (degree-preserving randomization)
    config_triangle_counts = []
    config_type_counts = defaultdict(list)
    degree_seq = [d for _, d in G.degree()]

    for trial in range(n_random):
        # Create configuration model graph
        Gr = nx.configuration_model(degree_seq, seed=trial)
        Gr = nx.Graph(Gr)  # remove multi-edges
        Gr.remove_edges_from(nx.selfloop_edges(Gr))

        # Assign platform labels (same node mapping)
        trial_triangles = []
        r_nodes = list(Gr.nodes())
        for u in r_nodes:
            if u >= n_nodes:
                continue
            for v in Gr.neighbors(u):
                if v >= n_nodes:
                    continue
                if v > u:
                    for w in Gr.neighbors(u):
                        if w >= n_nodes:
                            continue
                        if w > v and Gr.has_edge(v, w):
                            trial_triangles.append((u, v, w))

        config_triangle_counts.append(len(trial_triangles))
        trial_type_counts = Counter(triangle_type(u, v, w) for u, v, w in trial_triangles)
        for t in ["CCC", "CCA", "CAA", "AAA"]:
            config_type_counts[t].append(trial_type_counts.get(t, 0))

    config_mean = np.mean(config_triangle_counts)
    config_std = np.std(config_triangle_counts)
    config_z = (n_triangles - config_mean) / config_std if config_std > 0 else 0

    print(f"\nConfiguration model ({n_random} realizations):")
    print(f"  Expected triangles: {config_mean:.1f} ± {config_std:.1f}")
    print(f"  Observed: {n_triangles}")
    print(f"  z-score: {config_z:.2f}")

    print(f"\nType-specific z-scores (vs configuration model):")
    type_z = {}
    for t in ["CCC", "CCA", "CAA", "AAA"]:
        obs = type_counts.get(t, 0)
        null_mean = np.mean(config_type_counts[t])
        null_std = np.std(config_type_counts[t])
        z = (obs - null_mean) / null_std if null_std > 0 else 0
        type_z[t] = z
        print(f"  {t}: obs={obs}, null={null_mean:.1f}±{null_std:.1f}, z={z:.2f}")

    # Significance profile (normalized z-scores)
    z_vals = [type_z[t] for t in ["CCC", "CCA", "CAA", "AAA"]]
    z_norm = np.sqrt(sum(z ** 2 for z in z_vals)) if any(z != 0 for z in z_vals) else 1
    sig_profile = {t: type_z[t] / z_norm for t in ["CCC", "CCA", "CAA", "AAA"]}
    print(f"\nSignificance profile (normalized):")
    for t in ["CCC", "CCA", "CAA", "AAA"]:
        print(f"  {t}: {sig_profile[t]:+.3f}")

    result = {
        "method": f"Triad census + config model null ({n_random} realizations), θ={threshold}",
        "n_triangles": n_triangles,
        "n_open_triads": open_triads,
        "global_clustering": float(3 * n_triangles / (3 * n_triangles + open_triads)) if (3 * n_triangles + open_triads) > 0 else 0,
        "type_census": dict(type_counts),
        "er_expected_triangles": float(er_expected_triangles),
        "er_ratio": float(n_triangles / er_expected_triangles) if er_expected_triangles > 0 else None,
        "config_model": {
            "n_realizations": n_random,
            "mean_triangles": float(config_mean),
            "std_triangles": float(config_std),
            "z_score": float(config_z),
        },
        "type_z_scores": {t: float(type_z[t]) for t in ["CCC", "CCA", "CAA", "AAA"]},
        "significance_profile": {t: float(sig_profile[t]) for t in ["CCC", "CCA", "CAA", "AAA"]},
    }
    with open(RESULTS_DIR / "concept_triad_profile.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_triad_profile.json")


def run_f86(concepts, sim, threshold=0.70):
    """F86: Community semantic coherence validation.

    For each community, compute:
    - Intra-community mean similarity
    - Inter-community mean similarity to each other community
    - Coherence score = intra / mean(inter)
    - Topic label via highest-degree member
    """
    print("\n" + "=" * 60)
    print("F86: Community Semantic Coherence")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    partition = get_louvain_communities(G)
    n = len(concepts)

    # Group by community
    comm_members = defaultdict(list)
    for i in range(n):
        c = partition.get(i, -1)
        comm_members[c].append(i)

    # Filter to communities with ≥2 members
    comm_ids = sorted([c for c, members in comm_members.items() if len(members) >= 2])

    print(f"Communities with ≥2 members: {len(comm_ids)}")

    community_stats = []
    for c in comm_ids:
        members = comm_members[c]
        # Intra-community similarity
        intra_sims = []
        for i, m1 in enumerate(members):
            for m2 in members[i + 1:]:
                intra_sims.append(sim[m1, m2])
        mean_intra = np.mean(intra_sims) if intra_sims else 0

        # Inter-community similarity to each other community
        inter_sims = []
        for other_c in comm_ids:
            if other_c == c:
                continue
            other_members = comm_members[other_c]
            for m1 in members:
                for m2 in other_members:
                    inter_sims.append(sim[m1, m2])
        mean_inter = np.mean(inter_sims) if inter_sims else 0

        coherence = mean_intra / mean_inter if mean_inter > 0 else float("inf")

        # Platform composition
        plats = [concepts[m]["platform"] for m in members]
        n_cg = sum(1 for p in plats if p == "chatgpt")
        n_ag = len(plats) - n_cg

        # Topic: highest-degree member's name
        degrees = [(G.degree(m), concepts[m]["name"]) for m in members if m in G]
        degrees.sort(reverse=True)
        topic_label = degrees[0][1] if degrees else "unknown"
        top3 = [d[1] for d in degrees[:3]]

        community_stats.append({
            "community": c,
            "size": len(members),
            "n_chatgpt": n_cg,
            "n_agentic": n_ag,
            "mean_intra_sim": float(mean_intra),
            "mean_inter_sim": float(mean_inter),
            "coherence": float(coherence),
            "topic_label": topic_label,
            "top3_concepts": top3,
        })

        print(f"\n  C{c} ({len(members)} nodes, {n_cg}CG+{n_ag}AG):")
        print(f"    Intra-sim: {mean_intra:.4f}, Inter-sim: {mean_inter:.4f}")
        print(f"    Coherence: {coherence:.3f}")
        print(f"    Top concepts: {', '.join(top3)}")

    # Summary stats
    coherences = [s["coherence"] for s in community_stats if s["coherence"] != float("inf")]
    print(f"\nOverall coherence: mean={np.mean(coherences):.3f}, "
          f"range=[{min(coherences):.3f}, {max(coherences):.3f}]")

    # Coherence vs platform mixing correlation
    mixing = []
    coh = []
    for s in community_stats:
        if s["coherence"] != float("inf") and s["size"] > 1:
            frac_minor = min(s["n_chatgpt"], s["n_agentic"]) / s["size"]
            mixing.append(frac_minor)
            coh.append(s["coherence"])

    if len(mixing) > 3:
        rho, p = stats.spearmanr(mixing, coh)
        print(f"\nCoherence vs platform mixing: ρ={rho:.3f}, p={p:.4f}")
    else:
        rho, p = None, None

    result = {
        "method": "Intra/inter community similarity analysis at θ=0.70",
        "n_communities": len(comm_ids),
        "communities": community_stats,
        "overall_coherence": {
            "mean": float(np.mean(coherences)),
            "min": float(min(coherences)),
            "max": float(max(coherences)),
        },
        "coherence_vs_mixing": {
            "spearman_rho": float(rho) if rho is not None else None,
            "p_value": float(p) if p is not None else None,
        },
    }
    with open(RESULTS_DIR / "concept_community_coherence.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_community_coherence.json")


def run_f87(concepts, sim, threshold=0.70):
    """F87: Network backbone extraction via MST and graduated thresholding.

    Build maximum spanning tree (strongest connections), then add edges
    in decreasing weight order. Track when communities first emerge and
    when cross-platform bridges first appear.
    """
    print("\n" + "=" * 60)
    print("F87: Backbone Extraction (MST + Graduated)")
    print("=" * 60)

    n = len(concepts)
    platforms = {c["index"]: c["platform"] for c in concepts}

    # Build complete weighted graph
    G_full = nx.Graph()
    for c in concepts:
        G_full.add_node(c["index"], name=c["name"], platform=c["platform"])
    for i in range(n):
        for j in range(i + 1, n):
            G_full.add_edge(i, j, weight=float(sim[i, j]))

    # Maximum spanning tree (negate weights for minimum spanning tree algo)
    G_neg = G_full.copy()
    for u, v, d in G_neg.edges(data=True):
        d["weight"] = -d["weight"]
    mst = nx.minimum_spanning_tree(G_neg, weight="weight")

    # Restore positive weights
    for u, v, d in mst.edges(data=True):
        d["weight"] = -d["weight"]

    mst_edges = list(mst.edges(data=True))
    mst_weights = [d["weight"] for _, _, d in mst_edges]
    cross_plat_mst = sum(1 for u, v, _ in mst_edges if platforms[u] != platforms[v])

    print(f"Maximum Spanning Tree: {mst.number_of_edges()} edges")
    print(f"  Weight range: [{min(mst_weights):.4f}, {max(mst_weights):.4f}]")
    print(f"  Mean weight: {np.mean(mst_weights):.4f}")
    print(f"  Cross-platform edges: {cross_plat_mst}/{mst.number_of_edges()} "
          f"({cross_plat_mst / mst.number_of_edges():.1%})")

    # MST community structure
    mst_partition = get_louvain_communities(mst)
    mst_communities = len(set(mst_partition.values()))
    print(f"  MST communities (Louvain): {mst_communities}")

    # Compare MST partition with full network partition
    G_thresh = build_network(concepts, sim, threshold)
    full_partition = get_louvain_communities(G_thresh)
    mst_labels = [mst_partition.get(i, -1) for i in range(n)]
    full_labels = [full_partition.get(i, -1) for i in range(n)]
    nmi = normalized_mutual_info_score(mst_labels, full_labels)
    print(f"  NMI(MST vs full θ=0.70): {nmi:.3f}")

    # --- Graduated backbone: add edges from strongest to weakest ---
    # Sort all edges by weight (descending)
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j, float(sim[i, j])))
    all_edges.sort(key=lambda x: x[2], reverse=True)

    # Track when first cross-platform edge appears, community emergence, GC
    G_grow = nx.Graph()
    for c in concepts:
        G_grow.add_node(c["index"], platform=c["platform"])

    milestones = []
    first_cross = None
    first_gc50 = None

    # Sample at intervals to keep output manageable
    checkpoints = set([1, 5, 10, 25, 50, 100, 200, 500, 1000, 1280, 2000, 3000, len(all_edges)])

    for idx, (u, v, w) in enumerate(all_edges, 1):
        G_grow.add_edge(u, v, weight=w)

        is_cross = platforms[u] != platforms[v]
        if is_cross and first_cross is None:
            first_cross = {"edge_rank": idx, "weight": w, "nodes": [u, v],
                           "names": [concepts[u]["name"], concepts[v]["name"]]}
            print(f"\n  First cross-platform edge: rank {idx}, weight {w:.4f}")
            print(f"    {concepts[u]['name']} ({platforms[u]}) -- {concepts[v]['name']} ({platforms[v]})")

        # Check GC fraction
        if first_gc50 is None:
            gc = max(nx.connected_components(G_grow), key=len)
            if len(gc) >= n * 0.5:
                first_gc50 = {"edge_rank": idx, "weight": w, "gc_size": len(gc)}
                print(f"  GC reaches 50% at edge {idx}, weight {w:.4f}")

        if idx in checkpoints:
            gc = max(nx.connected_components(G_grow), key=len)
            n_cross = sum(1 for a, b in G_grow.edges() if platforms[a] != platforms[b])
            milestones.append({
                "edges": idx,
                "min_weight": w,
                "gc_fraction": len(gc) / n,
                "n_components": nx.number_connected_components(G_grow),
                "cross_platform_edges": n_cross,
                "cross_platform_fraction": n_cross / idx if idx > 0 else 0,
            })

    # Strongest cross-platform pairs (top 10)
    cross_edges = [(u, v, w) for u, v, w in all_edges if platforms[u] != platforms[v]]
    top_cross = cross_edges[:10]
    print(f"\n  Top 10 strongest cross-platform connections:")
    for u, v, w in top_cross:
        print(f"    {w:.4f}: {concepts[u]['name']} ({platforms[u]}) -- {concepts[v]['name']} ({platforms[v]})")

    result = {
        "method": "Maximum spanning tree + graduated backbone addition",
        "mst": {
            "n_edges": mst.number_of_edges(),
            "weight_range": [float(min(mst_weights)), float(max(mst_weights))],
            "mean_weight": float(np.mean(mst_weights)),
            "cross_platform_edges": cross_plat_mst,
            "cross_platform_fraction": float(cross_plat_mst / mst.number_of_edges()),
            "n_communities": mst_communities,
            "nmi_vs_full": float(nmi),
        },
        "graduated_backbone": {
            "first_cross_platform_edge": first_cross,
            "first_gc50": first_gc50,
            "milestones": milestones,
        },
        "top_cross_platform_edges": [
            {
                "weight": w,
                "node1": concepts[u]["name"],
                "platform1": platforms[u],
                "node2": concepts[v]["name"],
                "platform2": platforms[v],
            }
            for u, v, w in top_cross
        ],
    }
    with open(RESULTS_DIR / "concept_backbone.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_backbone.json")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Concept experiments F83-F87")
    parser.add_argument("--f83", action="store_true", help="Dimensionality reduction")
    parser.add_argument("--f84", action="store_true", help="Hierarchical communities")
    parser.add_argument("--f85", action="store_true", help="Triad significance profile")
    parser.add_argument("--f86", action="store_true", help="Community semantic coherence")
    parser.add_argument("--f87", action="store_true", help="Backbone extraction")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    if not any([args.f83, args.f84, args.f85, args.f86, args.f87, args.all]):
        parser.print_help()
        sys.exit(1)

    concepts, sim = load_concept_data()
    print(f"Loaded {len(concepts)} concepts, similarity matrix {sim.shape}")

    if args.f83 or args.all:
        run_f83(concepts, sim, args.threshold)
    if args.f84 or args.all:
        run_f84(concepts, sim, args.threshold)
    if args.f85 or args.all:
        run_f85(concepts, sim, args.threshold, n_random=100)
    if args.f86 or args.all:
        run_f86(concepts, sim, args.threshold)
    if args.f87 or args.all:
        run_f87(concepts, sim, args.threshold)


if __name__ == "__main__":
    main()
