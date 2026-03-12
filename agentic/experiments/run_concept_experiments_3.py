"""Concept-level network experiments (F67-F70).

Runs four analyses on the concept network:
  F67: Network resilience (targeted vs random attack)
  F68: Cross-platform concept flow (temporal ordering)
  F69: Spectral analysis (eigenvalue spectrum, Fiedler gap)
  F70: Concept network small-world properties
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


# ─── F67: Network Resilience ─────────────────────────────────────────────

def run_resilience(threshold=0.70, n_random_trials=50):
    """F67: Targeted vs random node removal — how fragile is the core?"""
    print("\n" + "=" * 60)
    print("F67: Concept Network Resilience Analysis")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    N = G.number_of_nodes()
    print(f"Network: {N} nodes, {G.number_of_edges()} edges")

    # Helper: compute largest connected component fraction
    def gc_fraction(H):
        if H.number_of_nodes() == 0:
            return 0.0
        return max(len(c) for c in nx.connected_components(H)) / N

    # Removal fractions to track
    fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    # 1) Targeted attack: remove highest-degree nodes first
    H_targeted = G.copy()
    targeted_gc = []
    removed_targeted = []
    for frac in fracs:
        n_to_remove = int(frac * N)
        # Reset and remove top n_to_remove by degree from original
        H_targeted = G.copy()
        degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        to_remove = [n for n, d in degrees[:n_to_remove]]
        H_targeted.remove_nodes_from(to_remove)

        gc_f = gc_fraction(H_targeted)
        targeted_gc.append({"removed_fraction": frac, "gc_fraction": round(gc_f, 4)})

        # Track what we removed
        platforms = Counter(concepts[n]["platform"] for n in to_remove)
        removed_targeted.append({
            "frac": frac,
            "n_removed": n_to_remove,
            "chatgpt": platforms.get("chatgpt", 0),
            "agentic": platforms.get("agentic", 0),
        })

    print("\nTargeted attack (highest-degree first):")
    for r in targeted_gc:
        print(f"  Remove {r['removed_fraction']:.0%}: GC = {r['gc_fraction']:.1%}")

    # 2) Targeted by betweenness
    betw = nx.betweenness_centrality(G)
    betw_sorted = sorted(betw.items(), key=lambda x: x[1], reverse=True)
    betw_gc = []
    for frac in fracs:
        n_to_remove = int(frac * N)
        H_betw = G.copy()
        to_remove = [n for n, b in betw_sorted[:n_to_remove]]
        H_betw.remove_nodes_from(to_remove)
        betw_gc.append({"removed_fraction": frac, "gc_fraction": round(gc_fraction(H_betw), 4)})

    print("\nTargeted attack (highest-betweenness first):")
    for r in betw_gc:
        print(f"  Remove {r['removed_fraction']:.0%}: GC = {r['gc_fraction']:.1%}")

    # 3) Random attack (average over n_random_trials)
    rng = np.random.default_rng(42)
    random_gc_trials = {frac: [] for frac in fracs}
    for trial in range(n_random_trials):
        nodes = list(G.nodes())
        rng.shuffle(nodes)
        for frac in fracs:
            n_to_remove = int(frac * N)
            H_random = G.copy()
            H_random.remove_nodes_from(nodes[:n_to_remove])
            random_gc_trials[frac].append(gc_fraction(H_random))

    random_gc = []
    for frac in fracs:
        vals = random_gc_trials[frac]
        random_gc.append({
            "removed_fraction": frac,
            "gc_fraction_mean": round(np.mean(vals), 4),
            "gc_fraction_std": round(np.std(vals), 4),
        })

    print(f"\nRandom attack (mean of {n_random_trials} trials):")
    for r in random_gc:
        print(f"  Remove {r['removed_fraction']:.0%}: GC = {r['gc_fraction_mean']:.1%} "
              f"± {r['gc_fraction_std']:.1%}")

    # 4) Platform-targeted: remove only ChatGPT concepts
    cg_by_degree = sorted(
        [(n, G.degree(n)) for n in G.nodes() if concepts[n]["platform"] == "chatgpt"],
        key=lambda x: x[1], reverse=True)
    cg_gc = []
    for frac in fracs:
        n_to_remove = int(frac * len(cg_by_degree))
        H_cg = G.copy()
        to_remove = [n for n, d in cg_by_degree[:n_to_remove]]
        H_cg.remove_nodes_from(to_remove)
        cg_gc.append({
            "removed_fraction": frac,
            "n_removed": n_to_remove,
            "gc_fraction": round(gc_fraction(H_cg), 4),
        })

    # Remove only agentic concepts
    ag_by_degree = sorted(
        [(n, G.degree(n)) for n in G.nodes() if concepts[n]["platform"] == "agentic"],
        key=lambda x: x[1], reverse=True)
    ag_gc = []
    for frac in fracs:
        n_to_remove = int(frac * len(ag_by_degree))
        H_ag = G.copy()
        to_remove = [n for n, d in ag_by_degree[:n_to_remove]]
        H_ag.remove_nodes_from(to_remove)
        ag_gc.append({
            "removed_fraction": frac,
            "n_removed": n_to_remove,
            "gc_fraction": round(gc_fraction(H_ag), 4),
        })

    print("\nPlatform-specific attack:")
    print("  Removing ChatGPT concepts:")
    for r in cg_gc:
        print(f"    {r['removed_fraction']:.0%} ({r['n_removed']}): GC = {r['gc_fraction']:.1%}")
    print("  Removing agentic concepts:")
    for r in ag_gc:
        print(f"    {r['removed_fraction']:.0%} ({r['n_removed']}): GC = {r['gc_fraction']:.1%}")

    # Critical removal threshold: fraction to halve GC
    def find_halving_frac(results, key="gc_fraction"):
        initial = results[0][key] if isinstance(results[0][key], float) else results[0].get("gc_fraction_mean", 1.0)
        target = initial / 2
        for r in results:
            val = r[key] if isinstance(r[key], float) else r.get("gc_fraction_mean", 1.0)
            if val <= target:
                return r["removed_fraction"]
        return 1.0

    halving = {
        "targeted_degree": find_halving_frac(targeted_gc),
        "targeted_betweenness": find_halving_frac(betw_gc),
        "random": find_halving_frac(random_gc, key="gc_fraction_mean"),
        "chatgpt_only": find_halving_frac(cg_gc),
        "agentic_only": find_halving_frac(ag_gc),
    }
    print(f"\nFraction to halve GC:")
    for strategy, frac in halving.items():
        print(f"  {strategy}: {frac:.0%}")

    output = {
        "experiment": "F67: Concept Network Resilience",
        "threshold": threshold,
        "network": {"nodes": N, "edges": G.number_of_edges()},
        "targeted_degree": targeted_gc,
        "targeted_betweenness": betw_gc,
        "random": random_gc,
        "chatgpt_removal": cg_gc,
        "agentic_removal": ag_gc,
        "halving_fractions": halving,
        "removed_composition": removed_targeted,
    }

    with open(RESULTS_DIR / "concept_resilience.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_resilience.json")
    return output


# ─── F68: Cross-Platform Concept Flow ────────────────────────────────────

def run_concept_flow(threshold=0.70):
    """F68: Temporal ordering — which platform 'discovers' concepts first?"""
    print("\n" + "=" * 60)
    print("F68: Cross-Platform Concept Flow")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)

    # Load ChatGPT episode dates
    chatgpt_results_path = RESULTS_DIR.parent.parent / "data" / "network" / "edges.json"
    # We need session dates. Try to get from the edge files or analysis results.
    # Use the concept's source_community and the episode dates from each analysis.

    # Load temporal concept evolution data for era assignments
    temporal_path = RESULTS_DIR / "temporal_concept_evolution.json"
    era_path = RESULTS_DIR / "model_era_concept_bridging.json"

    if not era_path.exists():
        print("Need model_era_concept_bridging.json — skipping")
        return None

    with open(era_path) as f:
        era_data = json.load(f)

    # Build concept→era mapping from F60 data
    concept_era = {}
    for c in era_data.get("concept_details", []):
        concept_era[c["name"]] = c["era"]

    # For each cross-platform concept pair above threshold, check if ChatGPT
    # concept is from earlier or later era than agentic concept
    # Agentic sessions are all from late 2024-early 2025 (Claude era)

    # All agentic concepts are implicitly "late" era
    cross_pairs = []
    for i in range(len(concepts)):
        if concepts[i]["platform"] != "chatgpt":
            continue
        cg_name = concepts[i]["name"]
        cg_era = concept_era.get(cg_name)
        if not cg_era:
            continue

        for j in range(len(concepts)):
            if concepts[j]["platform"] != "agentic":
                continue
            if sim[i, j] >= 0.80:  # High similarity threshold for concept matches
                cross_pairs.append({
                    "chatgpt_concept": cg_name,
                    "chatgpt_era": cg_era,
                    "agentic_concept": concepts[j]["name"],
                    "similarity": round(float(sim[i, j]), 4),
                })

    # Count by era
    era_counts = Counter(p["chatgpt_era"] for p in cross_pairs)
    total = sum(era_counts.values())

    print(f"\nCross-platform concept pairs (sim ≥ 0.80): {len(cross_pairs)}")
    print(f"\nChatGPT era of matched concepts:")
    for era in ["early", "middle", "late"]:
        count = era_counts.get(era, 0)
        frac = count / total if total > 0 else 0
        print(f"  {era}: {count} ({frac:.1%})")

    # Are late-era ChatGPT concepts more likely to match agentic concepts?
    # Compare matching rates per era
    era_concepts = Counter(concept_era.values())
    matching_rate = {}
    for era in ["early", "middle", "late"]:
        n_total = era_concepts.get(era, 0)
        n_matched = len(set(p["chatgpt_concept"] for p in cross_pairs
                           if p["chatgpt_era"] == era))
        rate = n_matched / n_total if n_total > 0 else 0
        matching_rate[era] = {
            "total_concepts": n_total,
            "concepts_with_agentic_match": n_matched,
            "rate": round(rate, 3),
        }
        print(f"  {era}: {n_matched}/{n_total} concepts match agentic ({rate:.1%})")

    # Top cross-platform pairs by similarity
    cross_pairs_sorted = sorted(cross_pairs, key=lambda x: -x["similarity"])[:20]

    # Mean similarity per era
    era_sims = defaultdict(list)
    for p in cross_pairs:
        era_sims[p["chatgpt_era"]].append(p["similarity"])

    era_sim_stats = {}
    for era in ["early", "middle", "late"]:
        vals = era_sims.get(era, [])
        if vals:
            era_sim_stats[era] = {
                "n_pairs": len(vals),
                "mean_sim": round(float(np.mean(vals)), 4),
                "max_sim": round(float(max(vals)), 4),
            }

    # Unique concept bridges per era
    unique_bridges = {}
    for era in ["early", "middle", "late"]:
        era_pairs = [p for p in cross_pairs if p["chatgpt_era"] == era]
        cg_unique = set(p["chatgpt_concept"] for p in era_pairs)
        ag_unique = set(p["agentic_concept"] for p in era_pairs)
        unique_bridges[era] = {
            "chatgpt_concepts": sorted(cg_unique),
            "agentic_concepts": sorted(ag_unique),
            "n_chatgpt": len(cg_unique),
            "n_agentic": len(ag_unique),
        }

    output = {
        "experiment": "F68: Cross-Platform Concept Flow",
        "threshold": threshold,
        "match_threshold": 0.80,
        "total_cross_pairs": len(cross_pairs),
        "era_distribution": {era: era_counts.get(era, 0) for era in ["early", "middle", "late"]},
        "matching_rates": matching_rate,
        "era_similarity_stats": era_sim_stats,
        "unique_bridges_per_era": unique_bridges,
        "top_cross_pairs": cross_pairs_sorted,
    }

    with open(RESULTS_DIR / "concept_flow.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_flow.json")
    return output


# ─── F69: Spectral Analysis ──────────────────────────────────────────────

def run_spectral(threshold=0.70):
    """F69: Eigenvalue spectrum and spectral gap analysis."""
    print("\n" + "=" * 60)
    print("F69: Concept Network Spectral Analysis")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Laplacian spectrum
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))

    print(f"\nLaplacian eigenvalues (first 15):")
    for i, ev in enumerate(eigenvalues[:15]):
        print(f"  λ_{i} = {ev:.6f}")

    # Number of zero eigenvalues = number of connected components
    n_zero = sum(1 for ev in eigenvalues if ev < 1e-10)
    print(f"\nZero eigenvalues: {n_zero} (= {n_zero} connected components)")

    # Fiedler value (algebraic connectivity) = second-smallest eigenvalue
    fiedler = eigenvalues[n_zero] if n_zero < len(eigenvalues) else 0
    print(f"Fiedler value (algebraic connectivity): λ₂ = {fiedler:.6f}")

    # Spectral gap (ratio of first two non-zero eigenvalues)
    if n_zero + 1 < len(eigenvalues) and eigenvalues[n_zero] > 1e-10:
        spectral_gap = eigenvalues[n_zero + 1] / eigenvalues[n_zero]
        print(f"Spectral gap (λ₃/λ₂): {spectral_gap:.4f}")
    else:
        spectral_gap = None

    # Number of communities suggested by spectral gap
    # Look for largest gaps between consecutive eigenvalues
    gaps = []
    for i in range(n_zero, min(len(eigenvalues) - 1, 20)):
        if eigenvalues[i] > 1e-10:
            gap = eigenvalues[i + 1] - eigenvalues[i]
            ratio = eigenvalues[i + 1] / eigenvalues[i] if eigenvalues[i] > 0 else 0
            gaps.append({"k": i + 1, "lambda_k": round(eigenvalues[i], 6),
                         "lambda_k1": round(eigenvalues[i + 1], 6),
                         "gap": round(gap, 6), "ratio": round(ratio, 4)})

    print(f"\nLargest spectral gaps (suggests k communities):")
    sorted_gaps = sorted(gaps, key=lambda x: -x["gap"])[:5]
    for g in sorted_gaps:
        print(f"  k={g['k']}: gap={g['gap']:.4f} (λ={g['lambda_k']:.4f} → {g['lambda_k1']:.4f})")

    # Normalized Laplacian spectrum
    L_norm = nx.normalized_laplacian_matrix(G).toarray().astype(float)
    eigenvalues_norm = np.sort(np.linalg.eigvalsh(L_norm))

    print(f"\nNormalized Laplacian eigenvalues (first 10):")
    for i, ev in enumerate(eigenvalues_norm[:10]):
        print(f"  λ_{i} = {ev:.6f}")

    # Adjacency spectrum (for comparison)
    A = nx.adjacency_matrix(G).toarray().astype(float)
    adj_eigenvalues = np.sort(np.linalg.eigvalsh(A))[::-1]  # Descending

    print(f"\nAdjacency eigenvalues (top 10):")
    for i, ev in enumerate(adj_eigenvalues[:10]):
        print(f"  μ_{i} = {ev:.4f}")

    # Spectral radius
    spectral_radius = adj_eigenvalues[0]
    print(f"\nSpectral radius: {spectral_radius:.4f}")

    # Compare Louvain communities with spectral suggestion
    partition = louvain_partition(G)
    n_louvain = len(set(partition.values()))
    print(f"\nLouvain communities: {n_louvain}")
    print(f"Spectral suggestion (largest gap): k={sorted_gaps[0]['k'] if sorted_gaps else 'N/A'}")

    output = {
        "experiment": "F69: Spectral Analysis",
        "threshold": threshold,
        "network": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "n_components": n_zero,
        "fiedler_value": round(float(fiedler), 6),
        "spectral_gap_ratio": round(float(spectral_gap), 4) if spectral_gap else None,
        "spectral_radius": round(float(spectral_radius), 4),
        "laplacian_top_15": [round(float(ev), 6) for ev in eigenvalues[:15]],
        "normalized_laplacian_top_10": [round(float(ev), 6) for ev in eigenvalues_norm[:10]],
        "adjacency_top_10": [round(float(ev), 4) for ev in adj_eigenvalues[:10]],
        "spectral_gaps": sorted_gaps,
        "louvain_communities": n_louvain,
    }

    with open(RESULTS_DIR / "concept_spectral.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_spectral.json")
    return output


# ─── F70: Small-World Properties ─────────────────────────────────────────

def run_small_world(threshold=0.70, n_random=100):
    """F70: Small-world coefficient and comparison to random/lattice."""
    print("\n" + "=" * 60)
    print("F70: Concept Network Small-World Properties")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    N = G.number_of_nodes()
    E = G.number_of_edges()
    print(f"Network: {N} nodes, {E} edges")

    # Use the largest connected component for path-based metrics
    gc_nodes = max(nx.connected_components(G), key=len)
    GC = G.subgraph(gc_nodes).copy()
    print(f"Giant component: {GC.number_of_nodes()} nodes, {GC.number_of_edges()} edges")

    # Observed metrics
    C_obs = nx.average_clustering(GC)
    L_obs = nx.average_shortest_path_length(GC)
    diameter = nx.diameter(GC)
    print(f"\nObserved:")
    print(f"  Clustering coefficient C = {C_obs:.4f}")
    print(f"  Average path length L = {L_obs:.4f}")
    print(f"  Diameter = {diameter}")

    # Random graph comparison (Erdős-Rényi with same n, m)
    n_gc = GC.number_of_nodes()
    m_gc = GC.number_of_edges()

    C_random_vals = []
    L_random_vals = []
    for i in range(n_random):
        R = nx.gnm_random_graph(n_gc, m_gc, seed=i)
        if nx.is_connected(R):
            C_random_vals.append(nx.average_clustering(R))
            L_random_vals.append(nx.average_shortest_path_length(R))

    C_random = np.mean(C_random_vals) if C_random_vals else 0
    L_random = np.mean(L_random_vals) if L_random_vals else 1

    print(f"\nRandom graph (ER, same n={n_gc}, m={m_gc}, {len(C_random_vals)} connected):")
    print(f"  C_random = {C_random:.4f}")
    print(f"  L_random = {L_random:.4f}")

    # Small-world coefficient σ = (C/C_random) / (L/L_random)
    sigma = (C_obs / C_random) / (L_obs / L_random) if C_random > 0 and L_random > 0 else 0
    print(f"\nSmall-world coefficient σ = {sigma:.4f}")
    print(f"  C/C_random = {C_obs/C_random:.4f}" if C_random > 0 else "")
    print(f"  L/L_random = {L_obs/L_random:.4f}" if L_random > 0 else "")

    # Omega coefficient (Telesford et al. 2011): ω = L_random/L - C/C_lattice
    # ω ≈ 0: small-world, ω → -1: lattice-like, ω → 1: random-like
    # For lattice comparison, use a ring lattice with same average degree
    avg_k = 2 * m_gc / n_gc
    k_lattice = int(round(avg_k))
    if k_lattice % 2 != 0:
        k_lattice += 1
    k_lattice = max(2, min(k_lattice, n_gc - 1))

    try:
        lattice = nx.watts_strogatz_graph(n_gc, k_lattice, 0, seed=42)
        C_lattice = nx.average_clustering(lattice)
        omega = (L_random / L_obs) - (C_obs / C_lattice) if C_lattice > 0 else None
        print(f"\n  C_lattice (WS p=0, k={k_lattice}) = {C_lattice:.4f}")
        print(f"  ω (omega) = {omega:.4f}" if omega is not None else "")
    except Exception:
        C_lattice = None
        omega = None

    # Degree distribution analysis
    degrees = [d for n, d in GC.degree()]
    mean_k = np.mean(degrees)
    std_k = np.std(degrees)
    max_k = max(degrees)
    min_k = min(degrees)

    # Check for power-law (or truncated power-law)
    degree_counts = Counter(degrees)
    sorted_degrees = sorted(degree_counts.keys())

    print(f"\nDegree distribution:")
    print(f"  Mean: {mean_k:.1f}, Std: {std_k:.1f}, Min: {min_k}, Max: {max_k}")

    # Platform-specific path lengths
    cg_nodes = [n for n in GC.nodes() if concepts[n]["platform"] == "chatgpt"]
    ag_nodes = [n for n in GC.nodes() if concepts[n]["platform"] == "agentic"]

    # Mean path length within and between platforms
    # Sample to avoid O(n²) for all pairs
    rng = np.random.default_rng(42)
    n_sample = min(200, len(cg_nodes) * len(ag_nodes))

    within_cg_paths = []
    within_ag_paths = []
    cross_paths = []

    # Sample within-platform paths
    for _ in range(min(200, len(cg_nodes) * (len(cg_nodes) - 1) // 2)):
        i, j = rng.choice(len(cg_nodes), 2, replace=False)
        try:
            within_cg_paths.append(nx.shortest_path_length(GC, cg_nodes[i], cg_nodes[j]))
        except nx.NetworkXNoPath:
            pass

    for _ in range(min(200, len(ag_nodes) * (len(ag_nodes) - 1) // 2)):
        i, j = rng.choice(len(ag_nodes), 2, replace=False)
        try:
            within_ag_paths.append(nx.shortest_path_length(GC, ag_nodes[i], ag_nodes[j]))
        except nx.NetworkXNoPath:
            pass

    # Sample cross-platform paths
    for _ in range(min(200, len(cg_nodes) * len(ag_nodes))):
        i = rng.choice(len(cg_nodes))
        j = rng.choice(len(ag_nodes))
        try:
            cross_paths.append(nx.shortest_path_length(GC, cg_nodes[i], ag_nodes[j]))
        except nx.NetworkXNoPath:
            pass

    print(f"\nPlatform-specific path lengths (sampled):")
    if within_cg_paths:
        print(f"  ChatGPT↔ChatGPT: {np.mean(within_cg_paths):.2f}")
    if within_ag_paths:
        print(f"  Agentic↔Agentic: {np.mean(within_ag_paths):.2f}")
    if cross_paths:
        print(f"  Cross-platform: {np.mean(cross_paths):.2f}")

    output = {
        "experiment": "F70: Small-World Properties",
        "threshold": threshold,
        "network": {
            "nodes": N, "edges": E,
            "gc_nodes": GC.number_of_nodes(),
            "gc_edges": GC.number_of_edges(),
        },
        "observed": {
            "clustering": round(C_obs, 4),
            "avg_path_length": round(L_obs, 4),
            "diameter": diameter,
        },
        "random_baseline": {
            "clustering": round(C_random, 4),
            "avg_path_length": round(L_random, 4),
            "n_connected": len(C_random_vals),
        },
        "small_world_sigma": round(sigma, 4),
        "C_over_C_random": round(C_obs / C_random, 4) if C_random > 0 else None,
        "L_over_L_random": round(L_obs / L_random, 4) if L_random > 0 else None,
        "omega": round(omega, 4) if omega is not None else None,
        "degree_stats": {
            "mean": round(mean_k, 2),
            "std": round(std_k, 2),
            "min": int(min_k),
            "max": int(max_k),
        },
        "platform_path_lengths": {
            "chatgpt_chatgpt": round(float(np.mean(within_cg_paths)), 3) if within_cg_paths else None,
            "agentic_agentic": round(float(np.mean(within_ag_paths)), 3) if within_ag_paths else None,
            "cross_platform": round(float(np.mean(cross_paths)), 3) if cross_paths else None,
        },
    }

    with open(RESULTS_DIR / "concept_small_world.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to concept_small_world.json")
    return output


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Concept-level experiments F67-F70")
    parser.add_argument("--f67", action="store_true", help="Network resilience")
    parser.add_argument("--f68", action="store_true", help="Cross-platform concept flow")
    parser.add_argument("--f69", action="store_true", help="Spectral analysis")
    parser.add_argument("--f70", action="store_true", help="Small-world properties")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Concept network threshold (default: 0.70)")
    args = parser.parse_args()

    if not any([args.f67, args.f68, args.f69, args.f70, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.f67 or args.all:
        run_resilience(args.threshold)
    if args.f68 or args.all:
        run_concept_flow(args.threshold)
    if args.f69 or args.all:
        run_spectral(args.threshold)
    if args.f70 or args.all:
        run_small_world(args.threshold)
