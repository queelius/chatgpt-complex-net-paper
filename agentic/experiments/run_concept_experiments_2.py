"""Concept-level network experiments (F62-F66).

Runs five analyses on the concept network:
  F62: K-core decomposition (backbone alternative to disparity filter)
  F63: Concept network motif census (triangles, higher-order patterns)
  F64: Concept-episode projection alignment
  F65: Concept degree assortativity and degree-degree correlations
  F66: Concept rich-club analysis
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats

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


# ─── F62: K-Core Decomposition ───────────────────────────────────────────

def run_kcore(threshold=0.70):
    """F62: K-core decomposition as backbone extraction."""
    print("\n" + "=" * 60)
    print("F62: K-Core Decomposition (Concept Network Backbone)")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute core numbers
    core_numbers = nx.core_number(G)
    max_k = max(core_numbers.values())
    print(f"Maximum core: k={max_k}")

    # Distribution of core numbers
    core_dist = Counter(core_numbers.values())
    print("\nCore distribution:")
    for k in sorted(core_dist):
        print(f"  k={k}: {core_dist[k]} nodes")

    # Platform composition per core level
    core_platform = defaultdict(lambda: {"chatgpt": 0, "agentic": 0})
    for node, k in core_numbers.items():
        plat = concepts[node]["platform"]
        core_platform[k][plat] += 1

    # Analyze innermost cores (top 3 k values)
    partition = louvain_partition(G)
    inner_cores = []
    for k in sorted(core_dist.keys(), reverse=True)[:4]:
        core_nodes = [n for n, c in core_numbers.items() if c >= k]
        subgraph = G.subgraph(core_nodes)

        # Cross-platform edges in this core
        cross = sum(1 for u, v in subgraph.edges()
                    if concepts[u]["platform"] != concepts[v]["platform"])
        total_e = subgraph.number_of_edges()

        # Community span
        communities_in_core = set(partition[n] for n in core_nodes)

        # Platform breakdown
        cg = sum(1 for n in core_nodes if concepts[n]["platform"] == "chatgpt")
        ag = sum(1 for n in core_nodes if concepts[n]["platform"] == "agentic")

        core_info = {
            "k": k,
            "n_nodes": len(core_nodes),
            "n_edges": total_e,
            "chatgpt": cg,
            "agentic": ag,
            "cross_platform_frac": round(cross / total_e, 3) if total_e > 0 else 0,
            "communities_spanned": len(communities_in_core),
            "density": round(nx.density(subgraph), 4),
        }
        inner_cores.append(core_info)
        print(f"\n  k≥{k}: {len(core_nodes)} nodes ({cg} CG + {ag} AG), "
              f"{total_e} edges, cross-platform={core_info['cross_platform_frac']:.1%}, "
              f"spans {len(communities_in_core)} communities")

    # Concept details for max core
    max_core_nodes = [n for n, c in core_numbers.items() if c == max_k]
    max_core_concepts = []
    for n in sorted(max_core_nodes, key=lambda x: G.degree(x), reverse=True):
        c = concepts[n]
        max_core_concepts.append({
            "name": c["name"],
            "platform": c["platform"],
            "source_community": c["source_community"],
            "degree": G.degree(n),
            "core_number": max_k,
        })

    # Compare cross-platform fraction across core shells
    shell_cross_platform = {}
    for k in sorted(core_dist.keys()):
        shell_nodes = [n for n, c in core_numbers.items() if c == k]
        if len(shell_nodes) < 2:
            continue
        shell_edges = [(u, v) for u, v in G.edges() if u in shell_nodes and v in shell_nodes
                       and core_numbers[u] == k and core_numbers[v] == k]
        if not shell_edges:
            # Use edges among nodes AT this core level (not just between same-shell)
            shell_sub = G.subgraph(shell_nodes)
            cross_e = sum(1 for u, v in shell_sub.edges()
                          if concepts[u]["platform"] != concepts[v]["platform"])
            tot_e = shell_sub.number_of_edges()
            if tot_e > 0:
                shell_cross_platform[k] = round(cross_e / tot_e, 3)

    output = {
        "experiment": "F62: K-Core Decomposition",
        "threshold": threshold,
        "network": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "max_core": max_k,
        "core_distribution": {str(k): v for k, v in sorted(core_dist.items())},
        "core_platform_composition": {
            str(k): dict(v) for k, v in sorted(core_platform.items())
        },
        "inner_cores": inner_cores,
        "max_core_concepts": max_core_concepts,
        "shell_cross_platform_frac": shell_cross_platform,
    }

    with open(RESULTS_DIR / "concept_kcore.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_kcore.json")
    return output


# ─── F63: Concept Network Motifs ─────────────────────────────────────────

def run_motifs(threshold=0.70):
    """F63: Triangle census and higher-order patterns in concept network."""
    print("\n" + "=" * 60)
    print("F63: Concept Network Motif Census")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Triangle census
    triangles_per_node = nx.triangles(G)
    total_triangles = sum(triangles_per_node.values()) // 3
    print(f"\nTotal triangles: {total_triangles}")

    # Classify triangles by platform composition
    # Enumerate all triangles
    tri_types = Counter()  # (n_chatgpt, n_agentic) -> count
    triangle_list = []

    # Use networkx to find all triangles
    for node in G.nodes():
        nbrs = set(G.neighbors(node))
        for nbr in nbrs:
            # Only count each triangle once (node < nbr)
            if nbr <= node:
                continue
            common = nbrs & set(G.neighbors(nbr))
            for third in common:
                if third <= nbr:
                    continue
                # Triangle: node, nbr, third
                platforms = sorted([concepts[node]["platform"],
                                   concepts[nbr]["platform"],
                                   concepts[third]["platform"]])
                n_cg = platforms.count("chatgpt")
                n_ag = platforms.count("agentic")
                tri_types[(n_cg, n_ag)] += 1
                triangle_list.append((node, nbr, third))

    print("\nTriangle types by platform composition:")
    type_labels = {
        (3, 0): "CCC (all-ChatGPT)",
        (2, 1): "CCA (2 ChatGPT + 1 agentic)",
        (1, 2): "CAA (1 ChatGPT + 2 agentic)",
        (0, 3): "AAA (all-agentic)",
    }
    tri_type_results = {}
    for (cg, ag), count in sorted(tri_types.items()):
        label = type_labels.get((cg, ag), f"({cg}CG+{ag}AG)")
        frac = count / total_triangles if total_triangles > 0 else 0
        print(f"  {label}: {count} ({frac:.1%})")
        tri_type_results[label] = {"count": count, "fraction": round(frac, 4)}

    # Expected triangle types under random platform assignment
    n_cg = sum(1 for c in concepts if c["platform"] == "chatgpt")
    n_ag = sum(1 for c in concepts if c["platform"] == "agentic")
    n = len(concepts)
    p_cg = n_cg / n
    p_ag = n_ag / n

    expected = {
        "CCC": p_cg ** 3,
        "CCA": 3 * p_cg ** 2 * p_ag,
        "CAA": 3 * p_cg * p_ag ** 2,
        "AAA": p_ag ** 3,
    }
    print(f"\nExpected under random assignment (p_CG={p_cg:.2f}, p_AG={p_ag:.2f}):")
    for label, prob in expected.items():
        print(f"  {label}: {prob:.3f} ({prob * total_triangles:.0f} triangles)")

    # Transitivity / clustering coefficient
    transitivity = nx.transitivity(G)
    avg_clustering = nx.average_clustering(G)
    print(f"\nTransitivity: {transitivity:.4f}")
    print(f"Average clustering: {avg_clustering:.4f}")

    # Per-platform clustering
    cg_nodes = [i for i, c in enumerate(concepts) if c["platform"] == "chatgpt"]
    ag_nodes = [i for i, c in enumerate(concepts) if c["platform"] == "agentic"]

    cg_clustering = np.mean([nx.clustering(G, n) for n in cg_nodes if G.degree(n) >= 2])
    ag_clustering = np.mean([nx.clustering(G, n) for n in ag_nodes if G.degree(n) >= 2])
    print(f"ChatGPT concept clustering: {cg_clustering:.4f}")
    print(f"Agentic concept clustering: {ag_clustering:.4f}")

    # Nodes with most triangles
    top_triangle_nodes = sorted(triangles_per_node.items(), key=lambda x: -x[1])[:10]
    top_nodes = []
    for node, tri_count in top_triangle_nodes:
        c = concepts[node]
        top_nodes.append({
            "name": c["name"],
            "platform": c["platform"],
            "triangles": tri_count,
            "degree": G.degree(node),
            "clustering": round(nx.clustering(G, node), 4),
        })

    output = {
        "experiment": "F63: Concept Network Motifs",
        "threshold": threshold,
        "network": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "total_triangles": total_triangles,
        "triangle_types": tri_type_results,
        "expected_random": {k: round(v, 4) for k, v in expected.items()},
        "transitivity": round(transitivity, 4),
        "average_clustering": round(avg_clustering, 4),
        "platform_clustering": {
            "chatgpt": round(cg_clustering, 4),
            "agentic": round(ag_clustering, 4),
        },
        "top_triangle_nodes": top_nodes,
    }

    with open(RESULTS_DIR / "concept_motifs.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_motifs.json")
    return output


# ─── F64: Concept-Episode Projection Alignment ───────────────────────────

def run_projection_alignment(threshold=0.70):
    """F64: Project concept communities onto episodes, compare with E-E Louvain."""
    print("\n" + "=" * 60)
    print("F64: Concept-Episode Projection Alignment")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    concept_partition = louvain_partition(G)
    print(f"Concept network: {G.number_of_nodes()} nodes, "
          f"{len(set(concept_partition.values()))} communities")

    # Load episode-level results for ChatGPT
    chatgpt_data_path = RESULTS_DIR.parent.parent / "data" / "chatgpt_analysis.json"
    if not chatgpt_data_path.exists():
        # Try alternative path
        chatgpt_data_path = RESULTS_DIR / "chatgpt_analysis.json"

    # Build concept→concept_community mapping
    concept_to_community = {}
    for node, comm in concept_partition.items():
        concept_to_community[node] = comm

    # Build episode→concept_community via source_community
    # Each concept belongs to a source community of episodes
    # We need the episode-level Louvain communities from the ChatGPT E-E network
    # Rebuild from edges
    edges_path = RESULTS_DIR.parent.parent / "data" / "network" / "edges.json"
    if not edges_path.exists():
        edges_path = RESULTS_DIR.parent.parent / "data" / "edges.json"

    # Try to load episode communities from the existing analysis
    analysis_files = list(RESULTS_DIR.glob("*multiplex*.json"))
    episode_communities = {}

    # Use the hierarchical multiplex which has episode community data
    hier_path = RESULTS_DIR / "hierarchical_multiplex.json"
    if hier_path.exists():
        with open(hier_path) as f:
            hier = json.load(f)

    # Build the mapping: source_community (episode-level) → concept_community
    source_to_concept_comm = defaultdict(set)
    for node in G.nodes():
        src = concepts[node]["source_community"]
        cc = concept_partition[node]
        source_to_concept_comm[src].add(cc)

    print(f"\nSource communities → concept communities mapping:")
    # How many episode-level communities map to each concept community?
    concept_comm_sources = defaultdict(set)
    for src, ccs in source_to_concept_comm.items():
        for cc in ccs:
            concept_comm_sources[cc].add(src)

    alignment_data = []
    for cc in sorted(concept_comm_sources.keys()):
        sources = concept_comm_sources[cc]
        # How many concepts in this concept community?
        cc_nodes = [n for n, c in concept_partition.items() if c == cc]
        cc_platforms = Counter(concepts[n]["platform"] for n in cc_nodes)

        entry = {
            "concept_community": cc,
            "n_concepts": len(cc_nodes),
            "platforms": dict(cc_platforms),
            "source_communities": sorted(sources),
            "n_source_communities": len(sources),
        }
        alignment_data.append(entry)
        print(f"  CC{cc}: {len(cc_nodes)} concepts from {len(sources)} "
              f"source communities ({dict(cc_platforms)})")

    # Compute many-to-many alignment metrics
    # How spread are concepts across concept communities?
    # For each source community, how many concept communities do its concepts belong to?
    src_spread = {}
    for src, ccs in source_to_concept_comm.items():
        n_concepts_from_src = sum(1 for c in concepts
                                  if c["source_community"] == src
                                  and c["index"] in concept_partition)
        src_spread[src] = {
            "n_concept_communities": len(ccs),
            "n_concepts": n_concepts_from_src,
            "concept_communities": sorted(ccs),
        }

    # Entropy of source→concept_community mapping
    # Higher entropy = more spread across concept communities
    entropies = []
    for src, ccs in source_to_concept_comm.items():
        src_concepts = [n for n in G.nodes()
                        if concepts[n]["source_community"] == src]
        if len(src_concepts) < 2:
            continue
        cc_counts = Counter(concept_partition[n] for n in src_concepts)
        total = sum(cc_counts.values())
        probs = [c / total for c in cc_counts.values()]
        h = -sum(p * np.log2(p) for p in probs if p > 0)
        entropies.append(h)

    mean_entropy = float(np.mean(entropies)) if entropies else 0
    print(f"\nMean source→concept community entropy: {mean_entropy:.3f}")
    print(f"  (0 = all concepts in one CC; high = spread across many)")

    # NMI between source community assignment and concept community assignment
    # Treat each concept as a data point
    src_labels = [concepts[n]["source_community"] for n in G.nodes()]
    cc_labels = [concept_partition[n] for n in G.nodes()]
    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(src_labels, cc_labels)
    print(f"NMI (source community vs concept community): {nmi:.3f}")

    output = {
        "experiment": "F64: Concept-Episode Projection Alignment",
        "threshold": threshold,
        "n_concept_communities": len(set(concept_partition.values())),
        "n_source_communities": len(source_to_concept_comm),
        "alignment": alignment_data,
        "source_spread": {str(k): v for k, v in sorted(src_spread.items())},
        "mean_mapping_entropy": round(mean_entropy, 4),
        "nmi_source_vs_concept": round(nmi, 4),
    }

    with open(RESULTS_DIR / "concept_projection_alignment.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_projection_alignment.json")
    return output


# ─── F65: Concept Degree Assortativity ────────────────────────────────────

def run_degree_assortativity(threshold=0.70):
    """F65: Degree-degree correlations and assortativity at concept level."""
    print("\n" + "=" * 60)
    print("F65: Concept Network Degree Assortativity")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Overall degree assortativity
    r = nx.degree_assortativity_coefficient(G)
    print(f"\nDegree assortativity coefficient: r = {r:.4f}")

    # Assortativity by platform attribute
    platform_assort = nx.attribute_assortativity_coefficient(G, "platform")
    print(f"Platform assortativity: r = {platform_assort:.4f}")

    # Source community assortativity
    src_assort = nx.attribute_assortativity_coefficient(G, "source_community")
    print(f"Source community assortativity: r = {src_assort:.4f}")

    # Degree correlations: for each edge, compare degree(u) vs degree(v)
    degree_pairs = [(G.degree(u), G.degree(v)) for u, v in G.edges()]
    d1 = [d[0] for d in degree_pairs]
    d2 = [d[1] for d in degree_pairs]
    rho, p_val = stats.spearmanr(d1, d2)
    print(f"Spearman ρ (edge degree-degree): {rho:.4f}, p={p_val:.4g}")

    # Average neighbor degree (knn)
    knn = nx.average_neighbor_degree(G)
    degrees = dict(G.degree())

    # knn(k): average neighbor degree as function of node degree
    degree_to_knn = defaultdict(list)
    for node, k in degrees.items():
        degree_to_knn[k].append(knn[node])

    knn_by_k = {}
    for k in sorted(degree_to_knn.keys()):
        vals = degree_to_knn[k]
        knn_by_k[k] = {
            "mean_knn": round(float(np.mean(vals)), 2),
            "n_nodes": len(vals),
        }

    # Cross-platform degree analysis
    # Do high-degree concepts have more cross-platform edges?
    cross_frac_by_degree = {}
    for node in G.nodes():
        k = G.degree(node)
        if k == 0:
            continue
        cross = sum(1 for nbr in G.neighbors(node)
                    if concepts[nbr]["platform"] != concepts[node]["platform"])
        cross_frac_by_degree.setdefault(k, []).append(cross / k)

    cross_frac_summary = {}
    for k in sorted(cross_frac_by_degree.keys()):
        vals = cross_frac_by_degree[k]
        cross_frac_summary[k] = round(float(np.mean(vals)), 3)

    # Spearman between degree and cross-platform fraction
    node_degrees = []
    node_cross_fracs = []
    for node in G.nodes():
        k = G.degree(node)
        if k == 0:
            continue
        cross = sum(1 for nbr in G.neighbors(node)
                    if concepts[nbr]["platform"] != concepts[node]["platform"])
        node_degrees.append(k)
        node_cross_fracs.append(cross / k)

    rho_cross, p_cross = stats.spearmanr(node_degrees, node_cross_fracs)
    print(f"\nDegree vs cross-platform fraction: ρ={rho_cross:.4f}, p={p_cross:.4g}")

    # Platform-specific degree distributions
    cg_degrees = [G.degree(i) for i, c in enumerate(concepts)
                  if c["platform"] == "chatgpt" and i in G]
    ag_degrees = [G.degree(i) for i, c in enumerate(concepts)
                  if c["platform"] == "agentic" and i in G]

    U_stat, mw_p = stats.mannwhitneyu(cg_degrees, ag_degrees, alternative='two-sided')
    print(f"\nPlatform degree comparison (Mann-Whitney):")
    print(f"  ChatGPT: mean={np.mean(cg_degrees):.1f}, median={np.median(cg_degrees):.0f}")
    print(f"  Agentic: mean={np.mean(ag_degrees):.1f}, median={np.median(ag_degrees):.0f}")
    print(f"  U={U_stat:.0f}, p={mw_p:.4g}")

    output = {
        "experiment": "F65: Concept Degree Assortativity",
        "threshold": threshold,
        "network": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "degree_assortativity": round(r, 4),
        "platform_assortativity": round(platform_assort, 4),
        "source_community_assortativity": round(src_assort, 4),
        "edge_degree_spearman": {"rho": round(rho, 4), "p": round(p_val, 6)},
        "knn_by_degree": knn_by_k,
        "degree_vs_cross_platform": {
            "spearman_rho": round(rho_cross, 4),
            "p_value": round(p_cross, 6),
        },
        "platform_degree_comparison": {
            "chatgpt": {
                "mean": round(float(np.mean(cg_degrees)), 2),
                "median": float(np.median(cg_degrees)),
                "std": round(float(np.std(cg_degrees)), 2),
            },
            "agentic": {
                "mean": round(float(np.mean(ag_degrees)), 2),
                "median": float(np.median(ag_degrees)),
                "std": round(float(np.std(ag_degrees)), 2),
            },
            "mann_whitney_U": round(float(U_stat), 1),
            "p_value": round(float(mw_p), 6),
        },
    }

    with open(RESULTS_DIR / "concept_degree_assortativity.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_degree_assortativity.json")
    return output


# ─── F66: Rich-Club Analysis ─────────────────────────────────────────────

def run_rich_club(threshold=0.70):
    """F66: Rich-club coefficient analysis of concept network."""
    print("\n" + "=" * 60)
    print("F66: Concept Network Rich-Club Analysis")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Rich-club coefficient φ(k)
    rc = nx.rich_club_coefficient(G, normalized=False)
    print(f"\nRich-club coefficient φ(k):")
    for k in sorted(rc.keys()):
        if rc[k] > 0:
            print(f"  k≥{k}: φ={rc[k]:.4f}")

    # Normalized rich-club: compare to random graphs
    # Generate N random graphs preserving degree sequence
    N_random = 100
    degree_seq = [d for n, d in G.degree()]

    print(f"\nNormalized rich-club (vs {N_random} configuration model realizations)...")
    random_rc = defaultdict(list)
    for i in range(N_random):
        try:
            R = nx.configuration_model(degree_seq, seed=i)
            R = nx.Graph(R)  # Remove multi-edges
            R.remove_edges_from(nx.selfloop_edges(R))
            rc_r = nx.rich_club_coefficient(R, normalized=False)
            for k, v in rc_r.items():
                random_rc[k].append(v)
        except Exception:
            continue

    normalized_rc = {}
    for k in sorted(rc.keys()):
        if k in random_rc and random_rc[k]:
            mean_random = np.mean(random_rc[k])
            if mean_random > 0:
                rho_norm = rc[k] / mean_random
                std_random = np.std(random_rc[k])
                z_score = (rc[k] - mean_random) / std_random if std_random > 0 else 0
                normalized_rc[k] = {
                    "phi_observed": round(rc[k], 4),
                    "phi_random_mean": round(mean_random, 4),
                    "phi_random_std": round(std_random, 4),
                    "rho_normalized": round(rho_norm, 4),
                    "z_score": round(z_score, 2),
                }

    print("\nNormalized rich-club ρ(k) = φ(k)/φ_random(k):")
    for k in sorted(normalized_rc.keys()):
        v = normalized_rc[k]
        sig = "***" if v["z_score"] > 2.58 else "**" if v["z_score"] > 1.96 else "*" if v["z_score"] > 1.64 else ""
        print(f"  k≥{k}: ρ={v['rho_normalized']:.3f} (z={v['z_score']:.1f}) {sig}")

    # Platform composition of rich club at various k thresholds
    rich_club_platform = {}
    for k_thresh in sorted(rc.keys()):
        if k_thresh > max(d for _, d in G.degree()):
            break
        rich_nodes = [n for n in G.nodes() if G.degree(n) >= k_thresh]
        if len(rich_nodes) < 2:
            break
        cg = sum(1 for n in rich_nodes if concepts[n]["platform"] == "chatgpt")
        ag = sum(1 for n in rich_nodes if concepts[n]["platform"] == "agentic")
        rich_club_platform[k_thresh] = {
            "n_nodes": len(rich_nodes),
            "chatgpt": cg,
            "agentic": ag,
            "chatgpt_frac": round(cg / len(rich_nodes), 3),
        }

    # Identify the "rich club" members (top 10% by degree)
    top_pct = 0.10
    n_top = max(1, int(len(concepts) * top_pct))
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:n_top]

    rich_members = []
    for n in top_nodes:
        c = concepts[n]
        # Cross-platform edges among rich club
        rich_set = set(top_nodes)
        rich_connections = sum(1 for nbr in G.neighbors(n) if nbr in rich_set)
        cross_rich = sum(1 for nbr in G.neighbors(n)
                         if nbr in rich_set and concepts[nbr]["platform"] != c["platform"])
        rich_members.append({
            "name": c["name"],
            "platform": c["platform"],
            "degree": G.degree(n),
            "rich_club_connections": rich_connections,
            "cross_platform_rich": cross_rich,
        })

    output = {
        "experiment": "F66: Rich-Club Analysis",
        "threshold": threshold,
        "network": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "raw_rich_club": {str(k): round(v, 4) for k, v in rc.items()},
        "normalized_rich_club": {str(k): v for k, v in normalized_rc.items()},
        "rich_club_platform_composition": {str(k): v for k, v in rich_club_platform.items()},
        "top_10pct_members": rich_members,
    }

    with open(RESULTS_DIR / "concept_rich_club.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_rich_club.json")
    return output


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Concept-level experiments F62-F66")
    parser.add_argument("--f62", action="store_true", help="K-core decomposition")
    parser.add_argument("--f63", action="store_true", help="Motif census")
    parser.add_argument("--f64", action="store_true", help="Projection alignment")
    parser.add_argument("--f65", action="store_true", help="Degree assortativity")
    parser.add_argument("--f66", action="store_true", help="Rich-club analysis")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Concept network threshold (default: 0.70)")
    args = parser.parse_args()

    if not any([args.f62, args.f63, args.f64, args.f65, args.f66, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.f62 or args.all:
        run_kcore(args.threshold)
    if args.f63 or args.all:
        run_motifs(args.threshold)
    if args.f64 or args.all:
        run_projection_alignment(args.threshold)
    if args.f65 or args.all:
        run_degree_assortativity(args.threshold)
    if args.f66 or args.all:
        run_rich_club(args.threshold)
