"""Concept-level network experiments (F88-F92).

Runs five analyses:
  F88: Edge prediction — can we predict cross-platform links from node features?
  F89: Borgatti-Everett continuous core-periphery decomposition
  F90: Community bridge anatomy — which specific concepts bridge which communities?
  F91: Network embedding similarity vs concept name similarity
  F92: Threshold robustness — how stable are key findings across θ?
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score

RESULTS_DIR = Path(__file__).parent / "results"


def load_concept_data():
    """Load concept embeddings and similarity matrix."""
    with open(RESULTS_DIR / "concept_embeddings.json") as f:
        meta = json.load(f)
    sim = np.load(RESULTS_DIR / "concept_similarity_matrix.npy")
    return meta["concepts"], sim


def build_network(concepts, sim, threshold=0.70):
    """Build concept network at given threshold."""
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
        return community_louvain.best_partition(G, random_state=seed)
    except ImportError:
        return {n: 0 for n in G.nodes()}


def run_f88(concepts, sim, threshold=0.70):
    """F88: Edge prediction — can node features predict cross-platform links?

    For each node pair, predict whether they share an edge.
    Features: |degree_i - degree_j|, common_neighbors, Jaccard, Adamic-Adar,
    same_platform, sim_i_j (cheating baseline), product of PageRanks.
    """
    print("=" * 60)
    print("F88: Cross-Platform Edge Prediction")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    n = len(concepts)
    nodes = list(G.nodes())

    # Precompute node features
    degree = dict(G.degree())
    pr = nx.pagerank(G)
    cc = nx.clustering(G)
    bc = nx.betweenness_centrality(G)
    platforms = {c["index"]: c["platform"] for c in concepts}

    # Build edge feature matrix for all cross-platform pairs
    # (Predicting whether a cross-platform pair is connected)
    cross_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if platforms[i] != platforms[j]:
                cross_pairs.append((i, j))

    print(f"Cross-platform pairs: {len(cross_pairs)}")
    connected = sum(1 for i, j in cross_pairs if G.has_edge(i, j))
    print(f"Connected: {connected} ({connected/len(cross_pairs):.1%})")

    # Features for each pair
    X = []
    y = []
    for i, j in cross_pairs:
        cn = len(list(nx.common_neighbors(G, i, j)))
        # Jaccard coefficient
        ni = set(G.neighbors(i))
        nj = set(G.neighbors(j))
        jaccard = len(ni & nj) / len(ni | nj) if len(ni | nj) > 0 else 0
        # Adamic-Adar
        aa = sum(1 / np.log(G.degree(w)) for w in ni & nj if G.degree(w) > 1)

        features = [
            abs(degree.get(i, 0) - degree.get(j, 0)),  # degree difference
            degree.get(i, 0) + degree.get(j, 0),         # degree sum
            cn,                                            # common neighbors
            jaccard,                                       # Jaccard index
            aa,                                            # Adamic-Adar
            pr.get(i, 0) * pr.get(j, 0),                 # PageRank product
            cc.get(i, 0) * cc.get(j, 0),                 # clustering product
            abs(bc.get(i, 0) - bc.get(j, 0)),            # betweenness diff
        ]
        X.append(features)
        y.append(1 if G.has_edge(i, j) else 0)

    X = np.array(X)
    y = np.array(y)

    feature_names = ["degree_diff", "degree_sum", "common_neighbors", "jaccard",
                     "adamic_adar", "pagerank_product", "clustering_product",
                     "betweenness_diff"]

    # Logistic regression with CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc")
    print(f"\n5-fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit full model for coefficients
    lr.fit(X_scaled, y)
    y_prob = lr.predict_proba(X_scaled)[:, 1]
    full_auc = roc_auc_score(y, y_prob)
    print(f"Full model AUC: {full_auc:.3f}")

    # Feature importance
    coefs = lr.coef_[0]
    print(f"\nFeature coefficients:")
    for name, coef in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:25s}: {coef:+.4f}")

    # Ablation: which single feature is best?
    print(f"\nSingle-feature AUC:")
    single_aucs = {}
    for idx, name in enumerate(feature_names):
        X_single = X_scaled[:, idx:idx+1]
        try:
            cv = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                                 X_single, y, cv=5, scoring="roc_auc")
            single_aucs[name] = float(cv.mean())
            print(f"  {name:25s}: {cv.mean():.3f}")
        except Exception:
            single_aucs[name] = 0.5

    # Baseline: similarity alone
    sim_features = np.array([[sim[i, j]] for i, j in cross_pairs])
    sim_cv = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                             sim_features, y, cv=5, scoring="roc_auc")
    print(f"\n  similarity_alone:           {sim_cv.mean():.3f} (cheating baseline)")

    result = {
        "method": "Logistic regression on cross-platform node pairs, 5-fold CV",
        "n_cross_pairs": len(cross_pairs),
        "n_connected": connected,
        "connected_fraction": float(connected / len(cross_pairs)),
        "cv_auc": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "full_auc": float(full_auc),
        "feature_coefficients": {name: float(coef) for name, coef in zip(feature_names, coefs)},
        "single_feature_auc": single_aucs,
        "similarity_baseline_auc": float(sim_cv.mean()),
    }
    with open(RESULTS_DIR / "concept_edge_prediction.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_edge_prediction.json")


def run_f89(concepts, sim, threshold=0.70, n_iter=100):
    """F89: Borgatti-Everett continuous core-periphery decomposition.

    Fit continuous core-periphery model: each node gets a coreness score c_i ∈ [0,1].
    Ideal structure: edge probability = c_i * c_j.
    Optimize via gradient ascent on correlation with adjacency matrix.
    """
    print("\n" + "=" * 60)
    print("F89: Core-Periphery Decomposition")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    n = len(concepts)
    nodes = sorted(G.nodes())

    # Adjacency matrix
    A = nx.to_numpy_array(G, nodelist=nodes)

    # Initialize coreness from degree centrality
    deg_cent = nx.degree_centrality(G)
    c = np.array([deg_cent.get(i, 0) for i in nodes])
    c = (c - c.min()) / (c.max() - c.min() + 1e-10)  # normalize to [0,1]

    # Optimize: maximize correlation between c_i * c_j and A_ij
    best_corr = -1
    best_c = c.copy()

    for iteration in range(n_iter):
        # Compute ideal matrix
        ideal = np.outer(c, c)
        np.fill_diagonal(ideal, 0)

        # Compute correlation
        upper = np.triu_indices(n, k=1)
        corr = np.corrcoef(A[upper], ideal[upper])[0, 1]

        if corr > best_corr:
            best_corr = corr
            best_c = c.copy()

        # Gradient step: for each node, adjust c_i to increase correlation
        for i in range(n):
            # Try small increase and decrease
            for delta in [0.05, -0.05]:
                c_test = c.copy()
                c_test[i] = np.clip(c_test[i] + delta, 0, 1)
                ideal_test = np.outer(c_test, c_test)
                np.fill_diagonal(ideal_test, 0)
                corr_test = np.corrcoef(A[upper], ideal_test[upper])[0, 1]
                if corr_test > corr:
                    c = c_test
                    corr = corr_test
                    break

    c = best_c
    print(f"Core-periphery correlation: {best_corr:.4f}")

    # Classify: core (c > 0.5) vs periphery
    core_nodes = [nodes[i] for i in range(n) if c[i] > 0.5]
    periph_nodes = [nodes[i] for i in range(n) if c[i] <= 0.5]

    platforms = {co["index"]: co["platform"] for co in concepts}
    core_ag = sum(1 for i in core_nodes if platforms[i] == "agentic")
    core_cg = len(core_nodes) - core_ag
    periph_ag = sum(1 for i in periph_nodes if platforms[i] == "agentic")
    periph_cg = len(periph_nodes) - periph_ag

    print(f"\nCore (c > 0.5): {len(core_nodes)} nodes ({core_cg}CG + {core_ag}AG)")
    print(f"Periphery (c ≤ 0.5): {len(periph_nodes)} nodes ({periph_cg}CG + {periph_ag}AG)")
    print(f"Core agentic fraction: {core_ag / len(core_nodes):.1%}" if core_nodes else "")
    print(f"Periphery agentic fraction: {periph_ag / len(periph_nodes):.1%}" if periph_nodes else "")

    # Coreness by platform
    cg_coreness = [c[i] for i in range(n) if platforms[nodes[i]] == "chatgpt"]
    ag_coreness = [c[i] for i in range(n) if platforms[nodes[i]] == "agentic"]
    mw_stat, mw_p = stats.mannwhitneyu(cg_coreness, ag_coreness, alternative="two-sided")
    print(f"\nMean coreness: CG={np.mean(cg_coreness):.3f}, AG={np.mean(ag_coreness):.3f}")
    print(f"Mann-Whitney p={mw_p:.4e}")

    # Coreness vs cross-platform degree
    cross_deg = {}
    for i in nodes:
        cross_deg[i] = sum(1 for j in G.neighbors(i) if platforms[j] != platforms[i])
    coreness_vals = [c[idx] for idx, i in enumerate(nodes)]
    cross_deg_vals = [cross_deg[i] for i in nodes]
    rho, p = stats.spearmanr(coreness_vals, cross_deg_vals)
    print(f"Coreness vs cross-platform degree: ρ={rho:.3f}, p={p:.4e}")

    # Top 10 core and bottom 10 peripheral concepts
    ranked = sorted(range(n), key=lambda i: c[i], reverse=True)
    print(f"\nTop 10 core concepts:")
    for i in ranked[:10]:
        print(f"  c={c[i]:.3f}: {concepts[nodes[i]]['name']} ({platforms[nodes[i]]})")
    print(f"\nBottom 10 peripheral concepts:")
    for i in ranked[-10:]:
        print(f"  c={c[i]:.3f}: {concepts[nodes[i]]['name']} ({platforms[nodes[i]]})")

    result = {
        "method": f"Borgatti-Everett continuous CP ({n_iter} iterations)",
        "correlation": float(best_corr),
        "core_count": len(core_nodes),
        "periphery_count": len(periph_nodes),
        "core_composition": {"chatgpt": core_cg, "agentic": core_ag},
        "periphery_composition": {"chatgpt": periph_cg, "agentic": periph_ag},
        "mean_coreness": {
            "chatgpt": float(np.mean(cg_coreness)),
            "agentic": float(np.mean(ag_coreness)),
            "mann_whitney_p": float(mw_p),
        },
        "coreness_vs_cross_degree": {"rho": float(rho), "p": float(p)},
        "top_core": [
            {"name": concepts[nodes[i]]["name"], "platform": platforms[nodes[i]],
             "coreness": float(c[i])}
            for i in ranked[:15]
        ],
        "bottom_periphery": [
            {"name": concepts[nodes[i]]["name"], "platform": platforms[nodes[i]],
             "coreness": float(c[i])}
            for i in ranked[-15:]
        ],
    }
    with open(RESULTS_DIR / "concept_core_periphery.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_core_periphery.json")


def run_f90(concepts, sim, threshold=0.70):
    """F90: Community bridge anatomy — which specific concepts bridge which communities?

    For each inter-community edge, identify the bridging concept pair.
    Build a community-to-community bridge map.
    """
    print("\n" + "=" * 60)
    print("F90: Community Bridge Anatomy")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    partition = get_louvain_communities(G)
    n = len(concepts)
    platforms = {c["index"]: c["platform"] for c in concepts}

    # Participation coefficient: fraction of links to other communities
    def participation_coefficient(G, partition):
        pc = {}
        for v in G.nodes():
            ki = G.degree(v)
            if ki == 0:
                pc[v] = 0.0
                continue
            comm_links = Counter(partition[u] for u in G.neighbors(v))
            pc[v] = 1.0 - sum((count / ki) ** 2 for count in comm_links.values())
        return pc

    pc = participation_coefficient(G, partition)

    # Within-module degree z-score
    comm_members = defaultdict(list)
    for node, comm in partition.items():
        comm_members[comm].append(node)

    z_scores = {}
    for comm, members in comm_members.items():
        if len(members) < 2:
            for m in members:
                z_scores[m] = 0.0
            continue
        intra_degrees = []
        for m in members:
            intra_deg = sum(1 for nb in G.neighbors(m) if partition.get(nb) == comm)
            intra_degrees.append(intra_deg)
        mean_d = np.mean(intra_degrees)
        std_d = np.std(intra_degrees)
        for m, d in zip(members, intra_degrees):
            z_scores[m] = (d - mean_d) / std_d if std_d > 0 else 0.0

    # Guimerà-Amaral role classification
    def classify_role(z, p):
        if z < 2.5:  # Non-hub
            if p < 0.05: return "R1: Ultra-peripheral"
            elif p < 0.625: return "R2: Peripheral"
            elif p < 0.80: return "R3: Satellite connector"
            else: return "R4: Kinless"
        else:  # Hub
            if p < 0.30: return "R5: Provincial hub"
            elif p < 0.75: return "R6: Connector hub"
            else: return "R7: Kinless hub"

    roles = {v: classify_role(z_scores.get(v, 0), pc.get(v, 0)) for v in G.nodes()}
    role_counts = Counter(roles.values())
    print(f"\nGuimerà-Amaral roles:")
    for role in sorted(role_counts.keys()):
        members = [v for v, r in roles.items() if r == role]
        ag_count = sum(1 for m in members if platforms[m] == "agentic")
        print(f"  {role}: {role_counts[role]} nodes ({role_counts[role] - ag_count}CG + {ag_count}AG)")

    # Inter-community edges
    bridge_edges = []
    for u, v in G.edges():
        if partition[u] != partition[v]:
            bridge_edges.append({
                "node1": concepts[u]["name"],
                "platform1": platforms[u],
                "community1": partition[u],
                "node2": concepts[v]["name"],
                "platform2": platforms[v],
                "community2": partition[v],
                "weight": float(sim[u, v]),
                "cross_platform": platforms[u] != platforms[v],
            })

    # Community pair flow matrix
    comm_ids = sorted(set(partition.values()))
    flow_matrix = defaultdict(lambda: {"total": 0, "cross_platform": 0, "strongest": 0})
    for e in bridge_edges:
        key = tuple(sorted([e["community1"], e["community2"]]))
        flow_matrix[key]["total"] += 1
        if e["cross_platform"]:
            flow_matrix[key]["cross_platform"] += 1
        flow_matrix[key]["strongest"] = max(flow_matrix[key]["strongest"], e["weight"])

    print(f"\nInter-community edges: {len(bridge_edges)}")
    cross_bridge = sum(1 for e in bridge_edges if e["cross_platform"])
    print(f"Cross-platform bridges: {cross_bridge} ({cross_bridge / len(bridge_edges):.1%})")

    print(f"\nCommunity pair flow:")
    for key in sorted(flow_matrix.keys()):
        f = flow_matrix[key]
        print(f"  C{key[0]}-C{key[1]}: {f['total']} edges ({f['cross_platform']} cross-platform), "
              f"strongest={f['strongest']:.3f}")

    # Top bridge concepts (highest participation coefficient)
    top_bridges = sorted(G.nodes(), key=lambda v: pc.get(v, 0), reverse=True)[:15]
    print(f"\nTop 15 bridge concepts (by participation coefficient):")
    for v in top_bridges:
        role = roles[v]
        print(f"  P={pc[v]:.3f}, z={z_scores.get(v, 0):.2f}: "
              f"{concepts[v]['name']} ({platforms[v]}) [{role}]")

    result = {
        "method": "Participation coefficient + Guimerà-Amaral roles + bridge census",
        "n_bridge_edges": len(bridge_edges),
        "n_cross_platform_bridges": cross_bridge,
        "cross_platform_bridge_fraction": float(cross_bridge / len(bridge_edges)) if bridge_edges else 0,
        "role_distribution": {
            role: {
                "count": count,
                "chatgpt": count - sum(1 for v, r in roles.items() if r == role and platforms[v] == "agentic"),
                "agentic": sum(1 for v, r in roles.items() if r == role and platforms[v] == "agentic"),
            }
            for role, count in role_counts.items()
        },
        "community_pair_flow": {
            f"C{k[0]}-C{k[1]}": v for k, v in flow_matrix.items()
        },
        "top_bridges": [
            {
                "name": concepts[v]["name"],
                "platform": platforms[v],
                "participation": float(pc[v]),
                "z_score": float(z_scores.get(v, 0)),
                "role": roles[v],
                "community": partition[v],
            }
            for v in top_bridges
        ],
    }
    with open(RESULTS_DIR / "concept_bridge_anatomy.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_bridge_anatomy.json")


def run_f91(concepts, sim, threshold=0.70):
    """F91: Embedding similarity vs concept name lexical similarity.

    Are concepts with similar names actually close in embedding space?
    Uses Jaccard similarity on character n-grams as lexical proxy.
    """
    print("\n" + "=" * 60)
    print("F91: Embedding vs Lexical Similarity")
    print("=" * 60)

    n = len(concepts)

    def ngram_set(s, n_val=3):
        """Character n-gram set for a string."""
        s = s.lower().strip()
        return set(s[i:i+n_val] for i in range(len(s) - n_val + 1))

    def jaccard(s1, s2):
        ng1 = ngram_set(s1)
        ng2 = ngram_set(s2)
        if not ng1 or not ng2:
            return 0.0
        return len(ng1 & ng2) / len(ng1 | ng2)

    # Compute lexical similarity for all pairs
    lex_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            lex = jaccard(concepts[i]["name"], concepts[j]["name"])
            lex_sim[i, j] = lex
            lex_sim[j, i] = lex

    # Correlation between embedding sim and lexical sim
    upper = np.triu_indices(n, k=1)
    emb_flat = sim[upper]
    lex_flat = lex_sim[upper]

    rho, p_val = stats.spearmanr(emb_flat, lex_flat)
    print(f"Embedding vs lexical similarity: ρ={rho:.3f}, p={p_val:.4e}")

    # Cross-platform only
    platforms = [c["platform"] for c in concepts]
    cross_mask = np.array([platforms[i] != platforms[j] for i, j in zip(*upper)])
    if cross_mask.any():
        rho_cross, p_cross = stats.spearmanr(emb_flat[cross_mask], lex_flat[cross_mask])
        print(f"Cross-platform only: ρ={rho_cross:.3f}, p={p_cross:.4e}")
    else:
        rho_cross, p_cross = 0, 1

    # High lexical + low embedding (concept pairs that sound similar but aren't)
    # and low lexical + high embedding (concept pairs that don't sound similar but are)
    disagreements = []
    for idx in range(len(emb_flat)):
        i, j = upper[0][idx], upper[1][idx]
        if lex_flat[idx] > 0.3 and emb_flat[idx] < 0.65:
            disagreements.append({
                "type": "lexical_high_embed_low",
                "concept1": concepts[i]["name"],
                "platform1": platforms[i],
                "concept2": concepts[j]["name"],
                "platform2": platforms[j],
                "lexical_sim": float(lex_flat[idx]),
                "embed_sim": float(emb_flat[idx]),
            })
        elif lex_flat[idx] < 0.1 and emb_flat[idx] > 0.85:
            disagreements.append({
                "type": "lexical_low_embed_high",
                "concept1": concepts[i]["name"],
                "platform1": platforms[i],
                "concept2": concepts[j]["name"],
                "platform2": platforms[j],
                "lexical_sim": float(lex_flat[idx]),
                "embed_sim": float(emb_flat[idx]),
            })

    print(f"\nDisagreements: {len(disagreements)}")
    for d in disagreements[:10]:
        print(f"  [{d['type']}] {d['concept1']} ({d['platform1']}) vs "
              f"{d['concept2']} ({d['platform2']}): lex={d['lexical_sim']:.3f}, "
              f"emb={d['embed_sim']:.3f}")

    # Identical-name cross-platform pairs
    name_map = defaultdict(list)
    for c in concepts:
        name_map[c["name"].lower()].append(c)

    identical_pairs = []
    for name, cs in name_map.items():
        plats = set(c["platform"] for c in cs)
        if len(plats) > 1:
            for i, c1 in enumerate(cs):
                for c2 in cs[i+1:]:
                    if c1["platform"] != c2["platform"]:
                        identical_pairs.append({
                            "name": name,
                            "embed_sim": float(sim[c1["index"], c2["index"]]),
                        })

    print(f"\nIdentical-name cross-platform pairs: {len(identical_pairs)}")
    for pair in identical_pairs:
        print(f"  '{pair['name']}': embed_sim={pair['embed_sim']:.4f}")

    result = {
        "method": "Character 3-gram Jaccard vs embedding cosine similarity",
        "overall_correlation": {"rho": float(rho), "p": float(p_val)},
        "cross_platform_correlation": {"rho": float(rho_cross), "p": float(p_cross)},
        "n_disagreements": len(disagreements),
        "disagreements": disagreements[:20],
        "identical_name_pairs": identical_pairs,
    }
    with open(RESULTS_DIR / "concept_lexical_vs_embedding.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_lexical_vs_embedding.json")


def run_f92(concepts, sim):
    """F92: Threshold robustness — how stable are key metrics across θ?

    Sweep θ from 0.55 to 0.85 and compute: modularity, cross-platform %,
    clustering, small-world σ, community count, NMI vs θ=0.70.
    """
    print("\n" + "=" * 60)
    print("F92: Threshold Robustness Analysis")
    print("=" * 60)

    thresholds = [0.55, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.80, 0.85]

    # Reference partition at θ=0.70
    G_ref = build_network(concepts, sim, 0.70)
    ref_partition = get_louvain_communities(G_ref)

    platforms = {c["index"]: c["platform"] for c in concepts}
    results = []

    for theta in thresholds:
        G = build_network(concepts, sim, theta)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_edges == 0:
            continue

        # Basic metrics
        partition = get_louvain_communities(G)
        n_comm = len(set(partition.values()))

        try:
            import community as community_louvain
            mod = community_louvain.modularity(partition, G)
        except ImportError:
            mod = 0

        # Cross-platform edges
        cross = sum(1 for u, v in G.edges() if platforms.get(u) != platforms.get(v))
        cross_pct = cross / n_edges if n_edges > 0 else 0

        # Clustering
        clustering = nx.average_clustering(G)

        # NMI vs reference
        labels = [partition.get(i, -1) for i in range(len(concepts))]
        ref_labels = [ref_partition.get(i, -1) for i in range(len(concepts))]
        nmi = normalized_mutual_info_score(ref_labels, labels)

        # Density
        density = nx.density(G)

        # Giant component fraction
        if G.number_of_nodes() > 0:
            gc = max(nx.connected_components(G), key=len)
            gc_frac = len(gc) / n_nodes
        else:
            gc_frac = 0

        # Assortativity (degree)
        try:
            assort = nx.degree_assortativity_coefficient(G)
        except Exception:
            assort = 0

        results.append({
            "threshold": theta,
            "edges": n_edges,
            "density": float(density),
            "communities": n_comm,
            "modularity": float(mod),
            "cross_platform_pct": float(cross_pct),
            "clustering": float(clustering),
            "gc_fraction": float(gc_frac),
            "degree_assortativity": float(assort),
            "nmi_vs_070": float(nmi),
        })

        print(f"  θ={theta:.3f}: {n_edges:5d} edges, {n_comm:2d} comm, "
              f"Q={mod:.3f}, cross={cross_pct:.1%}, "
              f"NMI(0.70)={nmi:.3f}, CC={clustering:.3f}")

    # Stability analysis: which metrics are most/least stable?
    metrics = ["modularity", "cross_platform_pct", "clustering", "communities",
               "gc_fraction", "degree_assortativity"]
    print(f"\nMetric stability (coefficient of variation):")
    stability = {}
    for m in metrics:
        vals = [r[m] for r in results if r[m] is not None]
        if vals and np.mean(vals) != 0:
            cv = np.std(vals) / abs(np.mean(vals))
            stability[m] = float(cv)
            print(f"  {m:25s}: CV={cv:.3f}")

    # NMI stability: how fast does structure diverge from θ=0.70?
    print(f"\nNMI decay from θ=0.70:")
    for r in results:
        delta = abs(r["threshold"] - 0.70)
        print(f"  Δθ={delta:.3f}: NMI={r['nmi_vs_070']:.3f}")

    output = {
        "method": f"Threshold sweep {thresholds[0]}-{thresholds[-1]} with {len(thresholds)} points",
        "reference_threshold": 0.70,
        "results": results,
        "stability_cv": stability,
    }
    with open(RESULTS_DIR / "concept_threshold_robustness.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_threshold_robustness.json")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Concept experiments F88-F92")
    parser.add_argument("--f88", action="store_true", help="Edge prediction")
    parser.add_argument("--f89", action="store_true", help="Core-periphery decomposition")
    parser.add_argument("--f90", action="store_true", help="Community bridge anatomy")
    parser.add_argument("--f91", action="store_true", help="Lexical vs embedding similarity")
    parser.add_argument("--f92", action="store_true", help="Threshold robustness")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    if not any([args.f88, args.f89, args.f90, args.f91, args.f92, args.all]):
        parser.print_help()
        sys.exit(1)

    concepts, sim = load_concept_data()
    print(f"Loaded {len(concepts)} concepts, similarity matrix {sim.shape}")

    if args.f88 or args.all:
        run_f88(concepts, sim, args.threshold)
    if args.f89 or args.all:
        run_f89(concepts, sim, args.threshold)
    if args.f90 or args.all:
        run_f90(concepts, sim, args.threshold)
    if args.f91 or args.all:
        run_f91(concepts, sim, args.threshold)
    if args.f92 or args.all:
        run_f92(concepts, sim)


if __name__ == "__main__":
    main()
