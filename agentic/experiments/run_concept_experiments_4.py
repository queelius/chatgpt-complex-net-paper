"""Concept-level network experiments (F71-F74).

Runs four analyses on the concept network:
  F71: Random walk information flow (PageRank, hitting times)
  F72: Ego networks of bridge concepts
  F73: Community role evolution across thresholds
  F74: Concept network modularity optimization landscape
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


# ─── F71: Random Walk Information Flow ────────────────────────────────────

def run_information_flow(threshold=0.70):
    """F71: PageRank, random walk centrality, cross-platform flow."""
    print("\n" + "=" * 60)
    print("F71: Random Walk Information Flow")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # PageRank (weighted)
    pr = nx.pagerank(G, weight="weight", alpha=0.85)

    # Top PageRank nodes
    pr_sorted = sorted(pr.items(), key=lambda x: -x[1])
    print("\nTop 15 by PageRank:")
    top_pr = []
    for node, score in pr_sorted[:15]:
        c = concepts[node]
        top_pr.append({
            "name": c["name"],
            "platform": c["platform"],
            "pagerank": round(score, 5),
            "degree": G.degree(node),
        })
        print(f"  {score:.5f}  {c['name']} ({c['platform']})")

    # Platform-aggregate PageRank
    cg_pr = sum(pr[n] for n in G.nodes() if concepts[n]["platform"] == "chatgpt")
    ag_pr = sum(pr[n] for n in G.nodes() if concepts[n]["platform"] == "agentic")
    n_cg = sum(1 for n in G.nodes() if concepts[n]["platform"] == "chatgpt")
    n_ag = sum(1 for n in G.nodes() if concepts[n]["platform"] == "agentic")

    print(f"\nPlatform-aggregate PageRank:")
    print(f"  ChatGPT: total={cg_pr:.3f}, mean={cg_pr/n_cg:.5f} ({n_cg} nodes)")
    print(f"  Agentic: total={ag_pr:.3f}, mean={ag_pr/n_ag:.5f} ({n_ag} nodes)")

    # Betweenness centrality (weighted)
    bc = nx.betweenness_centrality(G, weight="weight")
    bc_sorted = sorted(bc.items(), key=lambda x: -x[1])

    print("\nTop 10 by betweenness centrality:")
    top_bc = []
    for node, score in bc_sorted[:10]:
        c = concepts[node]
        top_bc.append({
            "name": c["name"],
            "platform": c["platform"],
            "betweenness": round(score, 5),
            "pagerank": round(pr[node], 5),
        })
        print(f"  {score:.5f}  {c['name']} ({c['platform']})")

    # Closeness centrality
    cc = nx.closeness_centrality(G)

    # Correlation between centrality measures
    nodes = list(G.nodes())
    pr_vals = [pr[n] for n in nodes]
    bc_vals = [bc[n] for n in nodes]
    cc_vals = [cc[n] for n in nodes]
    deg_vals = [G.degree(n) for n in nodes]

    correlations = {}
    for name, vals in [("pagerank", pr_vals), ("betweenness", bc_vals),
                       ("closeness", cc_vals), ("degree", deg_vals)]:
        for name2, vals2 in [("pagerank", pr_vals), ("betweenness", bc_vals),
                             ("closeness", cc_vals), ("degree", deg_vals)]:
            if name >= name2:
                continue
            rho, p = stats.spearmanr(vals, vals2)
            correlations[f"{name}_vs_{name2}"] = {
                "spearman_rho": round(rho, 4),
                "p_value": round(p, 6),
            }

    print("\nCentrality correlations:")
    for pair, v in sorted(correlations.items()):
        print(f"  {pair}: ρ={v['spearman_rho']:.3f}")

    # PageRank vs cross-platform fraction
    cross_fracs = []
    pr_for_corr = []
    for node in G.nodes():
        k = G.degree(node)
        if k == 0:
            continue
        cross = sum(1 for nbr in G.neighbors(node)
                    if concepts[nbr]["platform"] != concepts[node]["platform"])
        cross_fracs.append(cross / k)
        pr_for_corr.append(pr[node])

    rho_pr_cross, p_pr_cross = stats.spearmanr(pr_for_corr, cross_fracs)
    print(f"\nPageRank vs cross-platform fraction: ρ={rho_pr_cross:.4f}, p={p_pr_cross:.4g}")

    # Random walk: expected number of steps from ChatGPT cloud to agentic cloud
    # Use random walk simulations
    rng = np.random.default_rng(42)
    n_walks = 1000
    cg_nodes = [n for n in G.nodes() if concepts[n]["platform"] == "chatgpt"]
    ag_nodes_set = set(n for n in G.nodes() if concepts[n]["platform"] == "agentic")

    steps_to_agentic = []
    for _ in range(n_walks):
        start = rng.choice(cg_nodes)
        current = start
        steps = 0
        max_steps = 100
        while current not in ag_nodes_set and steps < max_steps:
            nbrs = list(G.neighbors(current))
            if not nbrs:
                break
            # Weighted random walk
            weights = [G[current][nbr]["weight"] for nbr in nbrs]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            current = rng.choice(nbrs, p=probs)
            steps += 1
        if current in ag_nodes_set:
            steps_to_agentic.append(steps)

    mean_steps = np.mean(steps_to_agentic) if steps_to_agentic else float('inf')
    print(f"\nRandom walk: ChatGPT → Agentic")
    print(f"  {len(steps_to_agentic)}/{n_walks} walks reached agentic node")
    print(f"  Mean steps: {mean_steps:.2f}")
    print(f"  Median steps: {np.median(steps_to_agentic):.0f}")

    # Reverse: agentic → chatgpt
    ag_nodes = [n for n in G.nodes() if concepts[n]["platform"] == "agentic"]
    cg_nodes_set = set(cg_nodes)
    steps_to_chatgpt = []
    for _ in range(n_walks):
        start = rng.choice(ag_nodes)
        current = start
        steps = 0
        while current not in cg_nodes_set and steps < 100:
            nbrs = list(G.neighbors(current))
            if not nbrs:
                break
            weights = [G[current][nbr]["weight"] for nbr in nbrs]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            current = rng.choice(nbrs, p=probs)
            steps += 1
        if current in cg_nodes_set:
            steps_to_chatgpt.append(steps)

    print(f"\nRandom walk: Agentic → ChatGPT")
    print(f"  {len(steps_to_chatgpt)}/{n_walks} walks reached ChatGPT node")
    print(f"  Mean steps: {np.mean(steps_to_chatgpt):.2f}")

    output = {
        "experiment": "F71: Random Walk Information Flow",
        "threshold": threshold,
        "top_pagerank": top_pr,
        "top_betweenness": top_bc,
        "platform_pagerank": {
            "chatgpt": {"total": round(cg_pr, 4), "mean": round(cg_pr/n_cg, 5), "n": n_cg},
            "agentic": {"total": round(ag_pr, 4), "mean": round(ag_pr/n_ag, 5), "n": n_ag},
        },
        "centrality_correlations": correlations,
        "pagerank_vs_cross_platform": {
            "spearman_rho": round(rho_pr_cross, 4),
            "p_value": round(p_pr_cross, 6),
        },
        "random_walk_chatgpt_to_agentic": {
            "n_walks": n_walks,
            "reached": len(steps_to_agentic),
            "mean_steps": round(mean_steps, 2),
            "median_steps": float(np.median(steps_to_agentic)) if steps_to_agentic else None,
        },
        "random_walk_agentic_to_chatgpt": {
            "n_walks": n_walks,
            "reached": len(steps_to_chatgpt),
            "mean_steps": round(float(np.mean(steps_to_chatgpt)), 2) if steps_to_chatgpt else None,
            "median_steps": float(np.median(steps_to_chatgpt)) if steps_to_chatgpt else None,
        },
    }

    with open(RESULTS_DIR / "concept_information_flow.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_information_flow.json")
    return output


# ─── F72: Ego Networks of Bridge Concepts ─────────────────────────────────

def run_ego_bridges(threshold=0.70):
    """F72: Local structure around top cross-platform bridge concepts."""
    print("\n" + "=" * 60)
    print("F72: Ego Networks of Bridge Concepts")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    partition = louvain_partition(G)

    # Identify top bridge concepts: highest cross-platform edge fraction × degree
    bridge_scores = {}
    for node in G.nodes():
        k = G.degree(node)
        if k < 5:  # Skip low-degree
            continue
        cross = sum(1 for nbr in G.neighbors(node)
                    if concepts[nbr]["platform"] != concepts[node]["platform"])
        bridge_scores[node] = (cross / k) * k  # cross-platform degree
    top_bridges = sorted(bridge_scores.items(), key=lambda x: -x[1])[:10]

    print(f"\nTop 10 bridge concepts:")
    ego_results = []
    for node, score in top_bridges:
        c = concepts[node]
        ego = nx.ego_graph(G, node, radius=1)

        # Ego network composition
        ego_nodes = list(ego.nodes())
        ego_cg = sum(1 for n in ego_nodes if concepts[n]["platform"] == "chatgpt")
        ego_ag = sum(1 for n in ego_nodes if concepts[n]["platform"] == "agentic")

        # Community diversity in ego network
        ego_communities = set(partition[n] for n in ego_nodes)

        # Edge types in ego network
        ego_edges = list(ego.edges())
        cross_edges = sum(1 for u, v in ego_edges
                          if concepts[u]["platform"] != concepts[v]["platform"])
        internal_edges = len(ego_edges) - cross_edges

        # Mean edge weight
        weights = [ego[u][v]["weight"] for u, v in ego_edges]
        mean_weight = float(np.mean(weights)) if weights else 0

        # Neighbors by platform
        nbr_cg = [concepts[n]["name"] for n in G.neighbors(node)
                   if concepts[n]["platform"] == "chatgpt"][:5]
        nbr_ag = [concepts[n]["name"] for n in G.neighbors(node)
                   if concepts[n]["platform"] == "agentic"][:5]

        result = {
            "name": c["name"],
            "platform": c["platform"],
            "degree": G.degree(node),
            "cross_platform_degree": int(score),
            "cross_platform_frac": round(int(score) / G.degree(node), 3),
            "ego_size": len(ego_nodes),
            "ego_chatgpt": ego_cg,
            "ego_agentic": ego_ag,
            "ego_communities": len(ego_communities),
            "ego_cross_edges": cross_edges,
            "ego_internal_edges": internal_edges,
            "mean_edge_weight": round(mean_weight, 4),
            "sample_chatgpt_neighbors": nbr_cg,
            "sample_agentic_neighbors": nbr_ag,
        }
        ego_results.append(result)
        print(f"  {c['name']} ({c['platform']}, deg={G.degree(node)}): "
              f"ego={len(ego_nodes)} nodes ({ego_cg} CG + {ego_ag} AG), "
              f"{len(ego_communities)} communities")

    # Compare ego network properties by platform
    cg_egos = [r for r in ego_results if r["platform"] == "chatgpt"]
    ag_egos = [r for r in ego_results if r["platform"] == "agentic"]

    print(f"\nBridge concept ego network comparison:")
    if cg_egos:
        print(f"  ChatGPT bridges ({len(cg_egos)}): "
              f"mean ego size={np.mean([r['ego_size'] for r in cg_egos]):.1f}, "
              f"mean communities={np.mean([r['ego_communities'] for r in cg_egos]):.1f}")
    if ag_egos:
        print(f"  Agentic bridges ({len(ag_egos)}): "
              f"mean ego size={np.mean([r['ego_size'] for r in ag_egos]):.1f}, "
              f"mean communities={np.mean([r['ego_communities'] for r in ag_egos]):.1f}")

    output = {
        "experiment": "F72: Ego Networks of Bridge Concepts",
        "threshold": threshold,
        "top_bridges": ego_results,
    }

    with open(RESULTS_DIR / "concept_ego_bridges.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_ego_bridges.json")
    return output


# ─── F73: Role Evolution Across Thresholds ────────────────────────────────

def run_role_evolution():
    """F73: How Guimerà-Amaral roles shift across thresholds."""
    print("\n" + "=" * 60)
    print("F73: Community Role Evolution Across Thresholds")
    print("=" * 60)

    from run_concept_experiments import within_module_degree_z, participation_coefficient, classify_role

    with open(RESULTS_DIR / "concept_embeddings.json") as f:
        meta = json.load(f)
    sim = np.load(RESULTS_DIR / "concept_similarity_matrix.npy")
    concepts = meta["concepts"]

    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    threshold_results = []

    for theta in thresholds:
        G = nx.Graph()
        for c in concepts:
            G.add_node(c["index"], name=c["name"], platform=c["platform"],
                        source_community=c["source_community"])
        n = len(concepts)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= theta:
                    G.add_edge(i, j, weight=float(sim[i, j]))

        if G.number_of_edges() == 0:
            continue

        partition = louvain_partition(G)
        n_communities = len(set(partition.values()))
        z_scores = within_module_degree_z(G, partition)
        p_coeffs = participation_coefficient(G, partition)

        role_counts = Counter()
        role_platform = defaultdict(lambda: {"chatgpt": 0, "agentic": 0})
        for node in G.nodes():
            z = z_scores.get(node, 0)
            p = p_coeffs.get(node, 0)
            role = classify_role(z, p)
            role_counts[role] += 1
            role_platform[role][concepts[node]["platform"]] += 1

        # Mean participation coefficient by platform
        cg_p = [p_coeffs.get(i, 0) for i, c in enumerate(concepts)
                if c["platform"] == "chatgpt" and i in G]
        ag_p = [p_coeffs.get(i, 0) for i, c in enumerate(concepts)
                if c["platform"] == "agentic" and i in G]

        result = {
            "threshold": theta,
            "edges": G.number_of_edges(),
            "communities": n_communities,
            "density": round(nx.density(G), 4),
            "role_distribution": {k: v for k, v in sorted(role_counts.items())},
            "role_platform": {k: dict(v) for k, v in sorted(role_platform.items())},
            "mean_participation": {
                "chatgpt": round(float(np.mean(cg_p)), 4) if cg_p else 0,
                "agentic": round(float(np.mean(ag_p)), 4) if ag_p else 0,
            },
        }
        threshold_results.append(result)

        print(f"\nθ={theta}: {G.number_of_edges()} edges, {n_communities} communities")
        for role, count in sorted(role_counts.items()):
            plat = role_platform[role]
            print(f"  {role}: {count} ({plat.get('chatgpt', 0)} CG + {plat.get('agentic', 0)} AG)")

    # Track individual concept role trajectories
    # Focus on top bridge concepts
    concept_trajectories = defaultdict(list)
    for result in threshold_results:
        theta = result["threshold"]
        G = nx.Graph()
        for c in concepts:
            G.add_node(c["index"])
        n = len(concepts)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= theta:
                    G.add_edge(i, j, weight=float(sim[i, j]))

        if G.number_of_edges() == 0:
            continue
        partition = louvain_partition(G)
        z_s = within_module_degree_z(G, partition)
        p_c = participation_coefficient(G, partition)

        for node in G.nodes():
            concept_trajectories[node].append({
                "threshold": theta,
                "role": classify_role(z_s.get(node, 0), p_c.get(node, 0)),
                "z": round(z_s.get(node, 0), 3),
                "p": round(p_c.get(node, 0), 3),
            })

    # Find concepts that change role class
    role_changers = []
    for node, trajectory in concept_trajectories.items():
        roles = set(t["role"] for t in trajectory)
        if len(roles) > 1:
            role_changers.append({
                "name": concepts[node]["name"],
                "platform": concepts[node]["platform"],
                "roles_seen": sorted(roles),
                "trajectory": trajectory,
            })

    print(f"\n{len(role_changers)} concepts change role across thresholds")

    output = {
        "experiment": "F73: Role Evolution Across Thresholds",
        "thresholds_analyzed": thresholds,
        "threshold_results": threshold_results,
        "n_role_changers": len(role_changers),
        "role_changers": role_changers[:20],  # Top 20
    }

    with open(RESULTS_DIR / "concept_role_evolution.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_role_evolution.json")
    return output


# ─── F74: Modularity Landscape ───────────────────────────────────────────

def run_modularity_landscape():
    """F74: Modularity optimization landscape across thresholds and resolutions."""
    print("\n" + "=" * 60)
    print("F74: Concept Network Modularity Landscape")
    print("=" * 60)

    import community.community_louvain as community_louvain

    with open(RESULTS_DIR / "concept_embeddings.json") as f:
        meta = json.load(f)
    sim = np.load(RESULTS_DIR / "concept_similarity_matrix.npy")
    concepts = meta["concepts"]

    # Sweep thresholds
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    threshold_results = []

    for theta in thresholds:
        G = nx.Graph()
        for c in concepts:
            G.add_node(c["index"], platform=c["platform"])
        n = len(concepts)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= theta:
                    G.add_edge(i, j, weight=float(sim[i, j]))

        if G.number_of_edges() == 0:
            threshold_results.append({
                "threshold": theta, "edges": 0,
                "communities": 0, "modularity": 0, "gc_fraction": 0,
            })
            continue

        partition = community_louvain.best_partition(G, random_state=42)
        Q = community_louvain.modularity(partition, G)
        gc = max(len(c) for c in nx.connected_components(G)) / n

        # Resolution sweep at this threshold
        resolutions = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        res_results = []
        for res in resolutions:
            part_r = community_louvain.best_partition(G, resolution=res, random_state=42)
            Q_r = community_louvain.modularity(part_r, G)
            n_c = len(set(part_r.values()))
            res_results.append({
                "resolution": res,
                "communities": n_c,
                "modularity": round(Q_r, 4),
            })

        # Platform mixing per community
        comm_sizes = Counter(partition.values())
        mixed_comms = 0
        for comm_id in set(partition.values()):
            members = [n for n, c in partition.items() if c == comm_id]
            platforms = set(concepts[n]["platform"] for n in members)
            if len(platforms) > 1:
                mixed_comms += 1

        result = {
            "threshold": theta,
            "edges": G.number_of_edges(),
            "communities": len(set(partition.values())),
            "modularity": round(Q, 4),
            "gc_fraction": round(gc, 3),
            "mixed_community_fraction": round(mixed_comms / len(set(partition.values())), 3),
            "resolution_sweep": res_results,
        }
        threshold_results.append(result)
        print(f"θ={theta}: {G.number_of_edges()} edges, {len(set(partition.values()))} communities, "
              f"Q={Q:.3f}, GC={gc:.1%}, mixed={result['mixed_community_fraction']:.1%}")

    # Find optimal threshold (max modularity)
    best = max(threshold_results, key=lambda x: x["modularity"])
    print(f"\nBest modularity: Q={best['modularity']} at θ={best['threshold']}")

    output = {
        "experiment": "F74: Modularity Landscape",
        "threshold_results": threshold_results,
        "optimal_threshold": best["threshold"],
        "optimal_modularity": best["modularity"],
    }

    with open(RESULTS_DIR / "concept_modularity_landscape.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_modularity_landscape.json")
    return output


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Concept-level experiments F71-F74")
    parser.add_argument("--f71", action="store_true", help="Information flow")
    parser.add_argument("--f72", action="store_true", help="Ego bridges")
    parser.add_argument("--f73", action="store_true", help="Role evolution")
    parser.add_argument("--f74", action="store_true", help="Modularity landscape")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Concept network threshold (default: 0.70)")
    args = parser.parse_args()

    if not any([args.f71, args.f72, args.f73, args.f74, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.f71 or args.all:
        run_information_flow(args.threshold)
    if args.f72 or args.all:
        run_ego_bridges(args.threshold)
    if args.f73 or args.all:
        run_role_evolution()
    if args.f74 or args.all:
        run_modularity_landscape()
