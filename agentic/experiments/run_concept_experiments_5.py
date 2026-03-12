"""Concept-level network experiments (F75-F77).

Runs three analyses:
  F75: Concept similarity distribution analysis
  F76: Cross-platform concept prediction (logistic regression)
  F77: Concept network community detection comparison (Louvain vs Label Propagation vs Spectral)
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


# ─── Helpers ──────────────────────────────────────────────────────────────

def load_concept_network(threshold=0.70):
    """Build concept network from similarity matrix at given threshold."""
    with open(RESULTS_DIR / "concept_embeddings.json") as f:
        meta = json.load(f)
    sim = np.load(RESULTS_DIR / "concept_similarity_matrix.npy")
    concepts = meta["concepts"]

    G = nx.Graph()
    for c in concepts:
        G.add_node(c["index"], name=c["name"], platform=c["platform"],
                    source_community=c["source_community"], id=c["id"])

    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                G.add_edge(i, j, weight=float(sim[i, j]))

    return G, concepts, sim


def louvain_partition(G):
    """Run Louvain community detection."""
    import community.community_louvain as community_louvain
    return community_louvain.best_partition(G, random_state=42)


# ─── F75: Similarity Distribution Analysis ────────────────────────────────

def run_similarity_distribution():
    """F75: Analyze the full pairwise similarity distribution."""
    print("\n" + "=" * 60)
    print("F75: Concept Similarity Distribution Analysis")
    print("=" * 60)

    with open(RESULTS_DIR / "concept_embeddings.json") as f:
        meta = json.load(f)
    sim = np.load(RESULTS_DIR / "concept_similarity_matrix.npy")
    concepts = meta["concepts"]
    n = len(concepts)

    # Extract upper triangle (all pairwise similarities)
    triu_idx = np.triu_indices(n, k=1)
    all_sims = sim[triu_idx]

    # Categorize by platform pair type
    within_cg = []
    within_ag = []
    cross_plat = []

    for idx in range(len(triu_idx[0])):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        s = all_sims[idx]
        pi = concepts[i]["platform"]
        pj = concepts[j]["platform"]
        if pi == pj == "chatgpt":
            within_cg.append(s)
        elif pi == pj == "agentic":
            within_ag.append(s)
        else:
            cross_plat.append(s)

    print(f"Total pairs: {len(all_sims)}")
    print(f"  ChatGPT↔ChatGPT: {len(within_cg)}")
    print(f"  Agentic↔Agentic: {len(within_ag)}")
    print(f"  Cross-platform: {len(cross_plat)}")

    # Distribution statistics
    dist_stats = {}
    for name, vals in [("all", all_sims), ("within_chatgpt", within_cg),
                       ("within_agentic", within_ag), ("cross_platform", cross_plat)]:
        arr = np.array(vals)
        dist_stats[name] = {
            "n": len(arr),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "skewness": round(float(stats.skew(arr)), 4),
            "kurtosis": round(float(stats.kurtosis(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "q25": round(float(np.percentile(arr, 25)), 4),
            "q75": round(float(np.percentile(arr, 75)), 4),
        }
        print(f"\n{name}:")
        for k, v in dist_stats[name].items():
            print(f"  {k}: {v}")

    # Normality test
    _, p_normal = stats.normaltest(all_sims)
    _, p_cg_normal = stats.normaltest(within_cg)
    _, p_ag_normal = stats.normaltest(within_ag)
    _, p_cross_normal = stats.normaltest(cross_plat)

    normality = {
        "all": round(float(p_normal), 6),
        "within_chatgpt": round(float(p_cg_normal), 6),
        "within_agentic": round(float(p_ag_normal), 6),
        "cross_platform": round(float(p_cross_normal), 6),
    }
    print(f"\nNormality test p-values (D'Agostino-Pearson):")
    for k, v in normality.items():
        print(f"  {k}: p={v}")

    # KS test: are within-platform and cross-platform distributions different?
    ks_cg_cross = stats.ks_2samp(within_cg, cross_plat)
    ks_ag_cross = stats.ks_2samp(within_ag, cross_plat)
    ks_cg_ag = stats.ks_2samp(within_cg, within_ag)

    print(f"\nKS tests:")
    print(f"  CG vs cross: D={ks_cg_cross.statistic:.4f}, p={ks_cg_cross.pvalue:.4g}")
    print(f"  AG vs cross: D={ks_ag_cross.statistic:.4f}, p={ks_ag_cross.pvalue:.4g}")
    print(f"  CG vs AG: D={ks_cg_ag.statistic:.4f}, p={ks_cg_ag.pvalue:.4g}")

    # Histogram bins for threshold analysis
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    threshold_stats = []
    for theta in thresholds:
        n_above = int(np.sum(all_sims >= theta))
        n_cross_above = int(np.sum(np.array(cross_plat) >= theta))
        cross_frac = n_cross_above / n_above if n_above > 0 else 0
        threshold_stats.append({
            "threshold": theta,
            "pairs_above": n_above,
            "cross_platform_above": n_cross_above,
            "cross_platform_fraction": round(cross_frac, 4),
        })
        print(f"  θ≥{theta}: {n_above} pairs, {n_cross_above} cross-platform ({cross_frac:.1%})")

    # Bimodality test (Hartigan's dip test approximation)
    # Use the dip in the histogram
    hist, bin_edges = np.histogram(all_sims, bins=50)
    # Check for valley between peaks
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append((i, hist[i], round(float(bin_edges[i]), 3)))

    print(f"\nHistogram peaks (50 bins): {len(peaks)} peaks found")
    for i, (idx, count, center) in enumerate(peaks[:5]):
        print(f"  Peak {i+1}: sim≈{center}, count={count}")

    output = {
        "experiment": "F75: Similarity Distribution Analysis",
        "distribution_stats": dist_stats,
        "normality_p_values": normality,
        "ks_tests": {
            "chatgpt_vs_cross": {
                "D": round(float(ks_cg_cross.statistic), 4),
                "p": round(float(ks_cg_cross.pvalue), 6),
            },
            "agentic_vs_cross": {
                "D": round(float(ks_ag_cross.statistic), 4),
                "p": round(float(ks_ag_cross.pvalue), 6),
            },
            "chatgpt_vs_agentic": {
                "D": round(float(ks_cg_ag.statistic), 4),
                "p": round(float(ks_cg_ag.pvalue), 6),
            },
        },
        "threshold_analysis": threshold_stats,
        "histogram_peaks": [{"center": p[2], "count": int(p[1])} for p in peaks[:5]],
    }

    with open(RESULTS_DIR / "concept_similarity_distribution.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_similarity_distribution.json")
    return output


# ─── F76: Cross-Platform Concept Prediction ───────────────────────────────

def run_concept_prediction(threshold=0.70):
    """F76: Predict cross-platform bridging from network features."""
    print("\n" + "=" * 60)
    print("F76: Cross-Platform Concept Prediction")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    G, concepts, sim = load_concept_network(threshold)
    partition = louvain_partition(G)

    # For each concept, compute features
    pr = nx.pagerank(G, weight="weight")
    bc = nx.betweenness_centrality(G, weight="weight")
    cc = nx.closeness_centrality(G)
    core_numbers = nx.core_number(G)
    clustering = nx.clustering(G)

    features = []
    labels = []  # 1 = high cross-platform fraction, 0 = low
    concept_names = []

    for node in G.nodes():
        k = G.degree(node)
        if k < 2:
            continue

        cross = sum(1 for nbr in G.neighbors(node)
                    if concepts[nbr]["platform"] != concepts[node]["platform"])
        cross_frac = cross / k

        # Features
        feat = [
            k,                       # degree
            pr[node],                # pagerank
            bc[node],                # betweenness
            cc[node],                # closeness
            core_numbers[node],      # k-core number
            clustering[node],        # local clustering
            1 if concepts[node]["platform"] == "agentic" else 0,  # platform
        ]
        features.append(feat)

        # Binary label: above-median cross-platform fraction
        labels.append(cross_frac)
        concept_names.append(concepts[node]["name"])

    X = np.array(features)
    y_continuous = np.array(labels)
    median_cross = np.median(y_continuous)
    y_binary = (y_continuous >= median_cross).astype(int)

    feature_names = ["degree", "pagerank", "betweenness", "closeness",
                     "core_number", "clustering", "is_agentic"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic regression with cross-validation
    lr = LogisticRegression(random_state=42, max_iter=1000)
    cv_scores = cross_val_score(lr, X_scaled, y_binary, cv=5, scoring="accuracy")
    print(f"\nLogistic regression (5-fold CV):")
    print(f"  Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    # Fit on full data for feature importance
    lr.fit(X_scaled, y_binary)
    coefs = lr.coef_[0]

    print(f"\nFeature importance (logistic regression coefficients):")
    importance = {}
    for name, coef in sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1])):
        importance[name] = round(float(coef), 4)
        print(f"  {name}: {coef:+.4f}")

    # Spearman correlations with continuous cross-platform fraction
    correlations = {}
    for i, name in enumerate(feature_names):
        rho, p = stats.spearmanr(X[:, i], y_continuous)
        correlations[name] = {"rho": round(rho, 4), "p": round(p, 6)}

    print(f"\nSpearman correlations with cross-platform fraction:")
    for name, v in sorted(correlations.items(), key=lambda x: -abs(x[1]["rho"])):
        print(f"  {name}: ρ={v['rho']:+.3f}, p={v['p']:.4g}")

    # Ablation: which feature matters most?
    ablation = {}
    for i, name in enumerate(feature_names):
        X_ablated = np.delete(X_scaled, i, axis=1)
        scores = cross_val_score(
            LogisticRegression(random_state=42, max_iter=1000),
            X_ablated, y_binary, cv=5, scoring="accuracy")
        ablation[name] = {
            "accuracy_without": round(float(np.mean(scores)), 4),
            "drop": round(float(np.mean(cv_scores) - np.mean(scores)), 4),
        }

    print(f"\nFeature ablation (accuracy drop when removed):")
    for name, v in sorted(ablation.items(), key=lambda x: -x[1]["drop"]):
        print(f"  {name}: drop={v['drop']:+.4f} (acc={v['accuracy_without']:.3f})")

    output = {
        "experiment": "F76: Cross-Platform Concept Prediction",
        "threshold": threshold,
        "n_concepts": len(features),
        "median_cross_frac": round(float(median_cross), 4),
        "cv_accuracy": {
            "mean": round(float(np.mean(cv_scores)), 4),
            "std": round(float(np.std(cv_scores)), 4),
            "folds": [round(float(s), 4) for s in cv_scores],
        },
        "feature_coefficients": importance,
        "feature_correlations": correlations,
        "feature_ablation": ablation,
    }

    with open(RESULTS_DIR / "concept_prediction.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_prediction.json")
    return output


# ─── F77: Community Detection Comparison ──────────────────────────────────

def run_community_comparison(threshold=0.70):
    """F77: Compare Louvain, Label Propagation, and spectral clustering."""
    print("\n" + "=" * 60)
    print("F77: Community Detection Method Comparison")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Method 1: Louvain
    import community.community_louvain as community_louvain
    part_louvain = community_louvain.best_partition(G, random_state=42)
    Q_louvain = community_louvain.modularity(part_louvain, G)
    n_louvain = len(set(part_louvain.values()))

    # Method 2: Label Propagation
    lp_comms = list(nx.community.label_propagation_communities(G))
    part_lp = {}
    for i, comm in enumerate(lp_comms):
        for node in comm:
            part_lp[node] = i
    n_lp = len(lp_comms)
    Q_lp = community_louvain.modularity(part_lp, G)

    # Method 3: Greedy modularity
    greedy_comms = list(nx.community.greedy_modularity_communities(G, weight="weight"))
    part_greedy = {}
    for i, comm in enumerate(greedy_comms):
        for node in comm:
            part_greedy[node] = i
    n_greedy = len(greedy_comms)
    Q_greedy = community_louvain.modularity(part_greedy, G)

    # Method 4: Spectral (use sklearn)
    from sklearn.cluster import SpectralClustering
    A = nx.adjacency_matrix(G).toarray()
    # Try several k values
    spectral_results = []
    for k in [3, 5, 7, 9, 11]:
        if k >= G.number_of_nodes():
            continue
        sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                random_state=42, assign_labels="kmeans")
        labels = sc.fit_predict(A)
        part_spec = {node: int(labels[i]) for i, node in enumerate(sorted(G.nodes()))}
        Q_spec = community_louvain.modularity(part_spec, G)
        spectral_results.append({"k": k, "modularity": round(Q_spec, 4)})

    # Best spectral
    best_spec = max(spectral_results, key=lambda x: x["modularity"])
    k_best = best_spec["k"]
    sc = SpectralClustering(n_clusters=k_best, affinity="precomputed",
                            random_state=42, assign_labels="kmeans")
    labels = sc.fit_predict(A)
    part_spectral = {node: int(labels[i]) for i, node in enumerate(sorted(G.nodes()))}
    Q_spectral = best_spec["modularity"]

    print(f"\nMethod comparison:")
    methods = [
        ("Louvain", n_louvain, Q_louvain, part_louvain),
        ("Label Propagation", n_lp, Q_lp, part_lp),
        ("Greedy Modularity", n_greedy, Q_greedy, part_greedy),
        (f"Spectral (k={k_best})", k_best, Q_spectral, part_spectral),
    ]

    for name, n_c, Q, part in methods:
        # Platform mixing
        mixed = 0
        for comm_id in set(part.values()):
            members = [n for n, c in part.items() if c == comm_id]
            platforms = set(concepts[n]["platform"] for n in members)
            if len(platforms) > 1:
                mixed += 1
        mixed_frac = mixed / len(set(part.values())) if set(part.values()) else 0
        print(f"  {name}: {n_c} communities, Q={Q:.4f}, mixed={mixed_frac:.1%}")

    # Pairwise NMI between all methods
    nmi_matrix = {}
    method_names = ["Louvain", "Label Propagation", "Greedy", f"Spectral(k={k_best})"]
    partitions = [part_louvain, part_lp, part_greedy, part_spectral]

    nodes = sorted(G.nodes())
    for i, (n1, p1) in enumerate(zip(method_names, partitions)):
        for j, (n2, p2) in enumerate(zip(method_names, partitions)):
            if i >= j:
                continue
            l1 = [p1.get(n, -1) for n in nodes]
            l2 = [p2.get(n, -1) for n in nodes]
            nmi = normalized_mutual_info_score(l1, l2)
            nmi_matrix[f"{n1}_vs_{n2}"] = round(nmi, 4)

    print(f"\nPairwise NMI:")
    for pair, nmi in sorted(nmi_matrix.items()):
        print(f"  {pair}: {nmi:.3f}")

    # Stability test: run Louvain 10 times with different seeds
    louvain_partitions = []
    for seed in range(10):
        p = community_louvain.best_partition(G, random_state=seed)
        louvain_partitions.append(p)

    louvain_nmis = []
    for i in range(10):
        for j in range(i + 1, 10):
            l1 = [louvain_partitions[i].get(n, -1) for n in nodes]
            l2 = [louvain_partitions[j].get(n, -1) for n in nodes]
            louvain_nmis.append(normalized_mutual_info_score(l1, l2))

    print(f"\nLouvain stability (10 seeds):")
    print(f"  Mean pairwise NMI: {np.mean(louvain_nmis):.3f} ± {np.std(louvain_nmis):.3f}")
    print(f"  Min: {min(louvain_nmis):.3f}, Max: {max(louvain_nmis):.3f}")

    output = {
        "experiment": "F77: Community Detection Comparison",
        "threshold": threshold,
        "methods": {
            "louvain": {"communities": n_louvain, "modularity": round(Q_louvain, 4)},
            "label_propagation": {"communities": n_lp, "modularity": round(Q_lp, 4)},
            "greedy_modularity": {"communities": n_greedy, "modularity": round(Q_greedy, 4)},
            "spectral": {"k": k_best, "modularity": round(Q_spectral, 4),
                         "k_sweep": spectral_results},
        },
        "pairwise_nmi": nmi_matrix,
        "louvain_stability": {
            "n_seeds": 10,
            "mean_nmi": round(float(np.mean(louvain_nmis)), 4),
            "std_nmi": round(float(np.std(louvain_nmis)), 4),
            "min_nmi": round(float(min(louvain_nmis)), 4),
            "max_nmi": round(float(max(louvain_nmis)), 4),
        },
    }

    with open(RESULTS_DIR / "concept_community_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_community_comparison.json")
    return output


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Concept-level experiments F75-F77")
    parser.add_argument("--f75", action="store_true", help="Similarity distribution")
    parser.add_argument("--f76", action="store_true", help="Concept prediction")
    parser.add_argument("--f77", action="store_true", help="Community comparison")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Concept network threshold (default: 0.70)")
    args = parser.parse_args()

    if not any([args.f75, args.f76, args.f77, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.f75 or args.all:
        run_similarity_distribution()
    if args.f76 or args.all:
        run_concept_prediction(args.threshold)
    if args.f77 or args.all:
        run_community_comparison(args.threshold)
