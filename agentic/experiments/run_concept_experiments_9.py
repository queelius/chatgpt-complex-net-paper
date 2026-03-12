"""Concept-level network experiments (F93-F97).

Runs five analyses:
  F93: Comprehensive null model battery (ER, config, geometric random graph)
  F94: Network visualization export (GEXF for Gephi with all attributes)
  F95: Concept network assortativity profiles (degree, platform, community)
  F96: Local bridge detection (edges whose removal increases shortest paths)
  F97: Network communicability and Estrada index
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats
from scipy.linalg import expm

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


def run_f93(concepts, sim, threshold=0.70, n_random=100):
    """F93: Comprehensive null model battery.

    Compare concept network to three null models:
    1. Erdős-Rényi (same n, p)
    2. Configuration model (same degree sequence)
    3. Geometric random graph (same n, edges in d-dimensional space)

    For each: compare clustering, path length, modularity, degree distribution.
    """
    print("=" * 60)
    print("F93: Comprehensive Null Model Battery")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = 2 * m / (n * (n - 1))

    # Observed metrics
    obs = {
        "clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "density": nx.density(G),
    }

    # Path length on largest component
    gc_nodes = max(nx.connected_components(G), key=len)
    gc = G.subgraph(gc_nodes)
    obs["avg_path_length"] = nx.average_shortest_path_length(gc)

    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, random_state=42)
        obs["modularity"] = community_louvain.modularity(partition, G)
    except ImportError:
        obs["modularity"] = 0

    obs["degree_variance"] = float(np.var([d for _, d in G.degree()]))
    obs["max_degree"] = max(d for _, d in G.degree())
    obs["assortativity"] = nx.degree_assortativity_coefficient(G)

    print(f"\nObserved network ({n} nodes, {m} edges):")
    for k, v in obs.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Null Model 1: Erdős-Rényi ---
    er_metrics = defaultdict(list)
    for trial in range(n_random):
        Gr = nx.gnp_random_graph(n, p, seed=trial)
        er_metrics["clustering"].append(nx.average_clustering(Gr))
        er_metrics["transitivity"].append(nx.transitivity(Gr))
        if nx.is_connected(Gr):
            er_metrics["avg_path_length"].append(nx.average_shortest_path_length(Gr))
        else:
            gc_r = Gr.subgraph(max(nx.connected_components(Gr), key=len))
            if gc_r.number_of_nodes() > 1:
                er_metrics["avg_path_length"].append(nx.average_shortest_path_length(gc_r))
        try:
            import community as community_louvain
            part = community_louvain.best_partition(Gr, random_state=42)
            er_metrics["modularity"].append(community_louvain.modularity(part, Gr))
        except ImportError:
            pass
        er_metrics["assortativity"].append(
            nx.degree_assortativity_coefficient(Gr) if Gr.number_of_edges() > 0 else 0)

    print(f"\nErdős-Rényi (n={n}, p={p:.4f}, {n_random} realizations):")
    er_z = {}
    for metric in ["clustering", "transitivity", "avg_path_length", "modularity", "assortativity"]:
        if metric in er_metrics and er_metrics[metric]:
            null_mean = np.mean(er_metrics[metric])
            null_std = np.std(er_metrics[metric])
            z = (obs.get(metric, 0) - null_mean) / null_std if null_std > 0 else 0
            er_z[metric] = z
            print(f"  {metric}: obs={obs.get(metric, 0):.4f}, null={null_mean:.4f}±{null_std:.4f}, z={z:.2f}")

    # --- Null Model 2: Configuration model ---
    deg_seq = [d for _, d in G.degree()]
    cm_metrics = defaultdict(list)
    for trial in range(n_random):
        Gr = nx.configuration_model(deg_seq, seed=trial)
        Gr = nx.Graph(Gr)  # remove multi-edges
        Gr.remove_edges_from(nx.selfloop_edges(Gr))
        cm_metrics["clustering"].append(nx.average_clustering(Gr))
        cm_metrics["transitivity"].append(nx.transitivity(Gr))
        if Gr.number_of_edges() > 0:
            cm_metrics["assortativity"].append(nx.degree_assortativity_coefficient(Gr))
        try:
            import community as community_louvain
            part = community_louvain.best_partition(Gr, random_state=42)
            cm_metrics["modularity"].append(community_louvain.modularity(part, Gr))
        except ImportError:
            pass

    print(f"\nConfiguration model ({n_random} realizations):")
    cm_z = {}
    for metric in ["clustering", "transitivity", "modularity", "assortativity"]:
        if metric in cm_metrics and cm_metrics[metric]:
            null_mean = np.mean(cm_metrics[metric])
            null_std = np.std(cm_metrics[metric])
            z = (obs.get(metric, 0) - null_mean) / null_std if null_std > 0 else 0
            cm_z[metric] = z
            print(f"  {metric}: obs={obs.get(metric, 0):.4f}, null={null_mean:.4f}±{null_std:.4f}, z={z:.2f}")

    # --- Null Model 3: Geometric random graph ---
    # Find radius that gives approximately m edges
    # In d dimensions, G(n,r) has E[edges] ≈ n^2 * Vol(ball_r) / 2
    # For d=2, use radius search
    from scipy.optimize import brentq

    def expected_edges_geo(r, n_nodes, dim=2):
        """Expected edges in geometric random graph."""
        # Approximate: place n_nodes uniformly in [0,1]^dim
        # Edge if distance < r
        Gr = nx.random_geometric_graph(n_nodes, r, dim=dim, seed=0)
        return Gr.number_of_edges()

    # Binary search for radius
    target_edges = m
    r_low, r_high = 0.01, 1.0
    for _ in range(20):
        r_mid = (r_low + r_high) / 2
        Gr = nx.random_geometric_graph(n, r_mid, dim=2, seed=0)
        if Gr.number_of_edges() < target_edges:
            r_low = r_mid
        else:
            r_high = r_mid
    r_geo = (r_low + r_high) / 2

    geo_metrics = defaultdict(list)
    for trial in range(n_random):
        Gr = nx.random_geometric_graph(n, r_geo, dim=2, seed=trial)
        geo_metrics["clustering"].append(nx.average_clustering(Gr))
        geo_metrics["transitivity"].append(nx.transitivity(Gr))
        geo_metrics["n_edges"].append(Gr.number_of_edges())
        if Gr.number_of_edges() > 0:
            geo_metrics["assortativity"].append(nx.degree_assortativity_coefficient(Gr))

    print(f"\nGeometric random graph (r≈{r_geo:.4f}, mean edges={np.mean(geo_metrics['n_edges']):.0f}):")
    geo_z = {}
    for metric in ["clustering", "transitivity", "assortativity"]:
        if metric in geo_metrics and geo_metrics[metric]:
            null_mean = np.mean(geo_metrics[metric])
            null_std = np.std(geo_metrics[metric])
            z = (obs.get(metric, 0) - null_mean) / null_std if null_std > 0 else 0
            geo_z[metric] = z
            print(f"  {metric}: obs={obs.get(metric, 0):.4f}, null={null_mean:.4f}±{null_std:.4f}, z={z:.2f}")

    result = {
        "method": f"3 null models × {n_random} realizations",
        "observed": {k: float(v) for k, v in obs.items()},
        "erdos_renyi": {
            "p": float(p),
            "z_scores": {k: float(v) for k, v in er_z.items()},
            "means": {k: float(np.mean(v)) for k, v in er_metrics.items()},
        },
        "configuration_model": {
            "z_scores": {k: float(v) for k, v in cm_z.items()},
            "means": {k: float(np.mean(v)) for k, v in cm_metrics.items()},
        },
        "geometric_random": {
            "radius": float(r_geo),
            "z_scores": {k: float(v) for k, v in geo_z.items()},
            "means": {k: float(np.mean(v)) for k, v in geo_metrics.items()},
        },
    }
    with open(RESULTS_DIR / "concept_null_models.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_null_models.json")


def run_f94(concepts, sim, threshold=0.70):
    """F94: Export concept network as GEXF for Gephi visualization."""
    print("\n" + "=" * 60)
    print("F94: Network Visualization Export")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    partition = get_louvain_communities(G)

    # Add all node attributes
    platforms = {c["index"]: c["platform"] for c in concepts}
    bc = nx.betweenness_centrality(G)
    pr = nx.pagerank(G)
    cc = nx.clustering(G)
    kcore = nx.core_number(G)

    # Cross-platform degree
    cross_deg = {}
    for v in G.nodes():
        cross_deg[v] = sum(1 for u in G.neighbors(v) if platforms.get(u) != platforms.get(v))

    for v in G.nodes():
        G.nodes[v]["community"] = partition.get(v, -1)
        G.nodes[v]["betweenness"] = bc.get(v, 0)
        G.nodes[v]["pagerank"] = pr.get(v, 0)
        G.nodes[v]["clustering"] = cc.get(v, 0)
        G.nodes[v]["kcore"] = kcore.get(v, 0)
        G.nodes[v]["cross_platform_degree"] = cross_deg.get(v, 0)
        G.nodes[v]["degree"] = G.degree(v)

    # Mark cross-platform edges
    for u, v, d in G.edges(data=True):
        d["cross_platform"] = 1 if platforms.get(u) != platforms.get(v) else 0

    output_path = RESULTS_DIR / "concept_network.gexf"
    nx.write_gexf(G, output_path)
    print(f"Exported to {output_path}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Node attributes: name, platform, source_community, community, betweenness, "
          f"pagerank, clustering, kcore, cross_platform_degree, degree")
    print(f"  Edge attributes: weight, cross_platform")

    # Also export GraphML
    graphml_path = RESULTS_DIR / "concept_network.graphml"
    nx.write_graphml(G, graphml_path)
    print(f"Also exported to {graphml_path}")

    result = {
        "gexf_path": str(output_path),
        "graphml_path": str(graphml_path),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "threshold": threshold,
    }
    with open(RESULTS_DIR / "concept_export_meta.json", "w") as f:
        json.dump(result, f, indent=2)


def run_f95(concepts, sim, threshold=0.70):
    """F95: Assortativity profiles — degree, platform, community mixing patterns."""
    print("\n" + "=" * 60)
    print("F95: Assortativity Profiles")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    partition = get_louvain_communities(G)
    platforms = {c["index"]: c["platform"] for c in concepts}
    n = len(concepts)

    # Degree assortativity
    r_deg = nx.degree_assortativity_coefficient(G)
    print(f"Degree assortativity: r = {r_deg:.4f}")

    # Platform assortativity (nominal)
    nx.set_node_attributes(G, platforms, "platform")
    r_plat = nx.attribute_assortativity_coefficient(G, "platform")
    print(f"Platform assortativity: r = {r_plat:.4f}")

    # Community assortativity
    nx.set_node_attributes(G, partition, "community")
    r_comm = nx.attribute_assortativity_coefficient(G, "community")
    print(f"Community assortativity: r = {r_comm:.4f}")

    # Degree-degree correlation profile by platform type
    edge_types = {
        "CG-CG": [], "CG-AG": [], "AG-AG": []
    }
    for u, v in G.edges():
        du, dv = G.degree(u), G.degree(v)
        pu, pv = platforms[u], platforms[v]
        if pu == pv == "chatgpt":
            edge_types["CG-CG"].append((du, dv))
        elif pu == pv == "agentic":
            edge_types["AG-AG"].append((du, dv))
        else:
            edge_types["CG-AG"].append((du, dv))

    print(f"\nDegree-degree correlation by edge type:")
    type_correlations = {}
    for etype, pairs in edge_types.items():
        if len(pairs) > 2:
            d1, d2 = zip(*pairs)
            rho, p = stats.spearmanr(d1, d2)
            print(f"  {etype}: ρ={rho:.3f}, p={p:.4e}, n={len(pairs)}")
            type_correlations[etype] = {"rho": float(rho), "p": float(p), "n": len(pairs)}
        else:
            print(f"  {etype}: too few edges ({len(pairs)})")

    # Neighbor degree distribution: do high-degree nodes connect to high-degree?
    # Average neighbor degree
    avg_nn_deg = nx.average_neighbor_degree(G)
    degrees = dict(G.degree())

    # knn(k): mean neighbor degree for nodes of degree k
    knn = defaultdict(list)
    for v in G.nodes():
        knn[degrees[v]].append(avg_nn_deg[v])
    knn_mean = {k: np.mean(v) for k, v in sorted(knn.items()) if len(v) >= 2}

    print(f"\nknn(k) (mean neighbor degree by degree):")
    for k in sorted(knn_mean.keys()):
        if k >= 10:  # only show high-degree
            print(f"  k={k}: knn={knn_mean[k]:.1f}")

    # Correlation knn vs k
    ks = list(knn_mean.keys())
    knns = [knn_mean[k] for k in ks]
    if len(ks) > 3:
        rho_knn, p_knn = stats.spearmanr(ks, knns)
        print(f"\nknn vs k correlation: ρ={rho_knn:.3f}, p={p_knn:.4e}")
    else:
        rho_knn, p_knn = 0, 1

    result = {
        "method": "Degree, platform, and community assortativity + knn profiles",
        "degree_assortativity": float(r_deg),
        "platform_assortativity": float(r_plat),
        "community_assortativity": float(r_comm),
        "edge_type_correlations": type_correlations,
        "knn_vs_k": {"rho": float(rho_knn), "p": float(p_knn)},
        "knn_profile": {str(k): float(v) for k, v in knn_mean.items()},
    }
    with open(RESULTS_DIR / "concept_assortativity.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_assortativity.json")


def run_f96(concepts, sim, threshold=0.70):
    """F96: Local bridge detection — edges whose removal most increases shortest paths.

    An edge (u,v) is a local bridge if it connects nodes whose neighborhoods
    don't overlap (span > 2). Find edges that are critical for cross-community
    and cross-platform connectivity.
    """
    print("\n" + "=" * 60)
    print("F96: Local Bridge Detection")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    partition = get_louvain_communities(G)
    platforms = {c["index"]: c["platform"] for c in concepts}

    # Compute bridge span for each edge
    # span(u,v) = shortest path from u to v if edge (u,v) were removed
    # True bridges have infinite span (removing them disconnects)
    # Local bridges have span > 2

    bridges = []
    for u, v in G.edges():
        # Common neighbors
        cn = set(G.neighbors(u)) & set(G.neighbors(v))
        span = 2 if cn else 3  # approximate: if no common neighbors, span ≥ 3

        is_cross_plat = platforms[u] != platforms[v]
        is_cross_comm = partition[u] != partition[v]

        bridges.append({
            "u": u, "v": v,
            "name_u": concepts[u]["name"],
            "name_v": concepts[v]["name"],
            "platform_u": platforms[u],
            "platform_v": platforms[v],
            "community_u": partition[u],
            "community_v": partition[v],
            "common_neighbors": len(cn),
            "span": span,
            "cross_platform": is_cross_plat,
            "cross_community": is_cross_comm,
            "weight": float(sim[u, v]),
        })

    # True local bridges (no common neighbors)
    local_bridges = [b for b in bridges if b["common_neighbors"] == 0]
    print(f"Total edges: {len(bridges)}")
    print(f"Local bridges (span>2): {len(local_bridges)} ({len(local_bridges)/len(bridges):.1%})")

    # Breakdown
    lb_cross_plat = sum(1 for b in local_bridges if b["cross_platform"])
    lb_cross_comm = sum(1 for b in local_bridges if b["cross_community"])
    all_cross_plat = sum(1 for b in bridges if b["cross_platform"])
    all_cross_comm = sum(1 for b in bridges if b["cross_community"])

    print(f"\nLocal bridges by type:")
    print(f"  Cross-platform: {lb_cross_plat}/{len(local_bridges)} "
          f"({lb_cross_plat/len(local_bridges):.1%}) vs "
          f"{all_cross_plat/len(bridges):.1%} overall")
    print(f"  Cross-community: {lb_cross_comm}/{len(local_bridges)} "
          f"({lb_cross_comm/len(local_bridges):.1%}) vs "
          f"{all_cross_comm/len(bridges):.1%} overall")

    # Actual NetworkX bridges (removing disconnects the graph)
    nx_bridges = list(nx.bridges(G))
    print(f"\nTrue bridges (disconnecting): {len(nx_bridges)}")

    bridge_details = []
    for u, v in nx_bridges:
        bridge_details.append({
            "name_u": concepts[u]["name"],
            "name_v": concepts[v]["name"],
            "platform_u": platforms[u],
            "platform_v": platforms[v],
            "cross_platform": platforms[u] != platforms[v],
            "weight": float(sim[u, v]),
        })
        print(f"  {concepts[u]['name']} ({platforms[u]}) -- {concepts[v]['name']} ({platforms[v]})")

    # Edge embeddedness: common_neighbors / (deg_u + deg_v - 2 - common_neighbors)
    embeddedness = []
    for b in bridges:
        u, v = b["u"], b["v"]
        cn = b["common_neighbors"]
        denom = G.degree(u) + G.degree(v) - 2 - cn
        emb = cn / denom if denom > 0 else 0
        embeddedness.append({
            "embeddedness": emb,
            "cross_platform": b["cross_platform"],
            "cross_community": b["cross_community"],
        })

    cross_emb = [e["embeddedness"] for e in embeddedness if e["cross_platform"]]
    same_emb = [e["embeddedness"] for e in embeddedness if not e["cross_platform"]]
    if cross_emb and same_emb:
        mw, mw_p = stats.mannwhitneyu(cross_emb, same_emb, alternative="two-sided")
        print(f"\nEdge embeddedness:")
        print(f"  Cross-platform: mean={np.mean(cross_emb):.3f}")
        print(f"  Same-platform: mean={np.mean(same_emb):.3f}")
        print(f"  Mann-Whitney p={mw_p:.4e}")
    else:
        mw_p = 1

    result = {
        "method": "Local bridge detection and edge embeddedness analysis",
        "total_edges": len(bridges),
        "local_bridges": len(local_bridges),
        "local_bridge_fraction": float(len(local_bridges) / len(bridges)),
        "local_bridge_cross_platform": {
            "count": lb_cross_plat,
            "fraction": float(lb_cross_plat / len(local_bridges)) if local_bridges else 0,
            "overall_cross_fraction": float(all_cross_plat / len(bridges)),
        },
        "true_bridges": len(nx_bridges),
        "true_bridge_details": bridge_details,
        "embeddedness": {
            "cross_platform_mean": float(np.mean(cross_emb)) if cross_emb else 0,
            "same_platform_mean": float(np.mean(same_emb)) if same_emb else 0,
            "mann_whitney_p": float(mw_p),
        },
    }
    with open(RESULTS_DIR / "concept_bridges.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_bridges.json")


def run_f97(concepts, sim, threshold=0.70):
    """F97: Network communicability and Estrada index.

    Communicability G_pq = (e^A)_pq measures how easily information flows
    between nodes p and q through all possible walks. The Estrada index
    EE = Tr(e^A) = sum of e^λ_i is a global measure of network folding.
    """
    print("\n" + "=" * 60)
    print("F97: Communicability and Estrada Index")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    nodes = sorted(G.nodes())
    n = len(nodes)
    platforms = {c["index"]: c["platform"] for c in concepts}
    partition = get_louvain_communities(G)

    # Adjacency matrix
    A = nx.to_numpy_array(G, nodelist=nodes)

    # Matrix exponential
    expA = expm(A)

    # Estrada index
    estrada = np.trace(expA)
    print(f"Estrada index: {estrada:.2f}")

    # Subgraph centrality (diagonal of e^A)
    subgraph_cent = {nodes[i]: float(expA[i, i]) for i in range(n)}

    # Top 10 by subgraph centrality
    top_sg = sorted(subgraph_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 by subgraph centrality:")
    for node, cent in top_sg:
        print(f"  {cent:.2f}: {concepts[node]['name']} ({platforms[node]})")

    # Platform comparison of subgraph centrality
    cg_cent = [subgraph_cent[v] for v in nodes if platforms[v] == "chatgpt"]
    ag_cent = [subgraph_cent[v] for v in nodes if platforms[v] == "agentic"]
    mw, mw_p = stats.mannwhitneyu(cg_cent, ag_cent, alternative="two-sided")
    print(f"\nSubgraph centrality: CG mean={np.mean(cg_cent):.2f}, AG mean={np.mean(ag_cent):.2f}")
    print(f"Mann-Whitney p={mw_p:.4e}")

    # Cross-platform communicability
    # Average communicability between CG-CG, AG-AG, CG-AG pairs
    cg_idx = [i for i, v in enumerate(nodes) if platforms[v] == "chatgpt"]
    ag_idx = [i for i, v in enumerate(nodes) if platforms[v] == "agentic"]

    def mean_off_diag(matrix, rows, cols):
        vals = []
        for i in rows:
            for j in cols:
                if i != j:
                    vals.append(matrix[i, j])
        return np.mean(vals) if vals else 0

    comm_cg_cg = mean_off_diag(expA, cg_idx, cg_idx)
    comm_ag_ag = mean_off_diag(expA, ag_idx, ag_idx)
    comm_cross = mean_off_diag(expA, cg_idx, ag_idx)

    print(f"\nMean communicability:")
    print(f"  CG-CG: {comm_cg_cg:.4f}")
    print(f"  AG-AG: {comm_ag_ag:.4f}")
    print(f"  CG-AG: {comm_cross:.4f}")
    print(f"  Cross/same ratio: {comm_cross / ((comm_cg_cg + comm_ag_ag) / 2):.3f}")

    # Community communicability matrix
    comm_ids = sorted(set(partition.values()))
    comm_comm = {}
    for c1 in comm_ids:
        idx1 = [i for i, v in enumerate(nodes) if partition.get(v) == c1]
        for c2 in comm_ids:
            idx2 = [i for i, v in enumerate(nodes) if partition.get(v) == c2]
            if c1 <= c2:
                val = mean_off_diag(expA, idx1, idx2) if c1 != c2 else mean_off_diag(expA, idx1, idx1)
                comm_comm[f"C{c1}-C{c2}"] = float(val)

    print(f"\nCommunity communicability (top pairs):")
    for pair in sorted(comm_comm, key=comm_comm.get, reverse=True)[:10]:
        print(f"  {pair}: {comm_comm[pair]:.4f}")

    # Communicability distance: sqrt(G_pp + G_qq - 2*G_pq)
    # Average cross-platform vs same-platform communicability distance
    cross_dists = []
    same_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(max(0, expA[i, i] + expA[j, j] - 2 * expA[i, j]))
            if platforms[nodes[i]] != platforms[nodes[j]]:
                cross_dists.append(d)
            else:
                same_dists.append(d)

    print(f"\nCommunicability distance:")
    print(f"  Cross-platform: mean={np.mean(cross_dists):.4f}")
    print(f"  Same-platform: mean={np.mean(same_dists):.4f}")
    cd_mw, cd_p = stats.mannwhitneyu(cross_dists, same_dists, alternative="two-sided")
    print(f"  Mann-Whitney p={cd_p:.4e}")

    result = {
        "method": "Matrix exponential communicability and Estrada index",
        "estrada_index": float(estrada),
        "subgraph_centrality": {
            "top10": [{"name": concepts[v]["name"], "platform": platforms[v],
                       "centrality": float(c)} for v, c in top_sg],
            "chatgpt_mean": float(np.mean(cg_cent)),
            "agentic_mean": float(np.mean(ag_cent)),
            "mann_whitney_p": float(mw_p),
        },
        "mean_communicability": {
            "cg_cg": float(comm_cg_cg),
            "ag_ag": float(comm_ag_ag),
            "cross": float(comm_cross),
            "cross_same_ratio": float(comm_cross / ((comm_cg_cg + comm_ag_ag) / 2)),
        },
        "community_communicability": comm_comm,
        "communicability_distance": {
            "cross_platform_mean": float(np.mean(cross_dists)),
            "same_platform_mean": float(np.mean(same_dists)),
            "mann_whitney_p": float(cd_p),
        },
    }
    with open(RESULTS_DIR / "concept_communicability.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_communicability.json")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Concept experiments F93-F97")
    parser.add_argument("--f93", action="store_true", help="Null model battery")
    parser.add_argument("--f94", action="store_true", help="GEXF/GraphML export")
    parser.add_argument("--f95", action="store_true", help="Assortativity profiles")
    parser.add_argument("--f96", action="store_true", help="Local bridge detection")
    parser.add_argument("--f97", action="store_true", help="Communicability + Estrada")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    if not any([args.f93, args.f94, args.f95, args.f96, args.f97, args.all]):
        parser.print_help()
        sys.exit(1)

    concepts, sim = load_concept_data()
    print(f"Loaded {len(concepts)} concepts, similarity matrix {sim.shape}")

    if args.f93 or args.all:
        run_f93(concepts, sim, args.threshold)
    if args.f94 or args.all:
        run_f94(concepts, sim, args.threshold)
    if args.f95 or args.all:
        run_f95(concepts, sim, args.threshold)
    if args.f96 or args.all:
        run_f96(concepts, sim, args.threshold)
    if args.f97 or args.all:
        run_f97(concepts, sim, args.threshold)


if __name__ == "__main__":
    main()
