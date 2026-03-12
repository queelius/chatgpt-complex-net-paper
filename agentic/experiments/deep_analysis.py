"""Deep network analysis: null models, edge distributions, rich clubs, spectral properties.

Runs analyses that go beyond compute_network_metrics() to find structurally
interesting phenomena in the semantic similarity networks.

Usage:
    python -m experiments.deep_analysis [--dataset chatgpt|agentic|both]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats

EXPERIMENTS_DIR = Path(__file__).parent
AGENTIC_DIR = EXPERIMENTS_DIR.parent
REPO_ROOT = AGENTIC_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENTIC_DIR))

ABLATION_EDGES = REPO_ROOT / "dev" / "ablation_study" / "data" / "edges"


def load_edges(path):
    with open(path) as f:
        return json.load(f)


def build_graph(edges, threshold, total_nodes=None):
    G = nx.Graph()
    for src, dst, w in edges:
        if w >= threshold:
            G.add_edge(src, dst, weight=w)
    if total_nodes and total_nodes > G.number_of_nodes():
        for i in range(total_nodes - G.number_of_nodes()):
            G.add_node(f"__iso_{i}")
    return G


# ---------------------------------------------------------------------------
# 1. Edge weight distribution analysis
# ---------------------------------------------------------------------------
def analyze_edge_weight_distribution(edges, label=""):
    """Analyze the full distribution of cosine similarity scores."""
    weights = np.array([w for _, _, w in edges])

    result = {
        "label": label,
        "total_pairs": len(weights),
        "mean": float(np.mean(weights)),
        "median": float(np.median(weights)),
        "std": float(np.std(weights)),
        "skewness": float(stats.skew(weights)),
        "kurtosis": float(stats.kurtosis(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
    }

    # Percentiles
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        result[f"p{p}"] = float(np.percentile(weights, p))

    # Fraction of pairs above each threshold
    for t in [0.80, 0.85, 0.875, 0.90, 0.925, 0.95]:
        frac = float(np.sum(weights >= t) / len(weights))
        count = int(np.sum(weights >= t))
        result[f"frac_above_{t}"] = frac
        result[f"count_above_{t}"] = count

    # Test for normality of the weight distribution
    if len(weights) > 5000:
        sample = np.random.default_rng(42).choice(weights, 5000, replace=False)
    else:
        sample = weights
    ks_stat, ks_p = stats.kstest(sample, 'norm', args=(np.mean(sample), np.std(sample)))
    result["ks_normality_stat"] = float(ks_stat)
    result["ks_normality_p"] = float(ks_p)

    return result


# ---------------------------------------------------------------------------
# 2. Null model comparison (configuration model)
# ---------------------------------------------------------------------------
def null_model_comparison(G, n_rewirings=10, seed=42):
    """Compare network properties against configuration model (degree-preserving randomization).

    The configuration model preserves the degree sequence but randomizes connections.
    Properties that differ significantly from the null model are structurally meaningful.
    """
    if G.number_of_edges() == 0:
        return {"error": "no edges"}

    # Get giant component for path length comparisons
    gc_nodes = max(nx.connected_components(G), key=len)
    gc = G.subgraph(gc_nodes).copy()

    real_metrics = {
        "clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "assortativity": nx.degree_assortativity_coefficient(G),
    }
    if gc.number_of_nodes() > 1:
        try:
            real_metrics["avg_path_length"] = nx.average_shortest_path_length(gc)
        except nx.NetworkXError:
            real_metrics["avg_path_length"] = None

    # Generate random graphs with same degree sequence
    rng = np.random.default_rng(seed)
    null_metrics = {k: [] for k in real_metrics}

    for i in range(n_rewirings):
        # Configuration model preserves degree sequence
        degree_seq = [d for _, d in G.degree()]
        try:
            R = nx.configuration_model(degree_seq, seed=int(rng.integers(1e9)))
            R = nx.Graph(R)  # Remove multi-edges
            R.remove_edges_from(nx.selfloop_edges(R))  # Remove self-loops
        except Exception:
            continue

        null_metrics["clustering"].append(nx.average_clustering(R))
        null_metrics["transitivity"].append(nx.transitivity(R))
        try:
            null_metrics["assortativity"].append(nx.degree_assortativity_coefficient(R))
        except (nx.NetworkXError, ValueError):
            pass

        if real_metrics.get("avg_path_length") is not None:
            r_components = sorted(nx.connected_components(R), key=len, reverse=True)
            r_gc = R.subgraph(r_components[0])
            if r_gc.number_of_nodes() > 1:
                try:
                    null_metrics["avg_path_length"].append(
                        nx.average_shortest_path_length(r_gc)
                    )
                except nx.NetworkXError:
                    pass

    result = {}
    for metric, real_val in real_metrics.items():
        if real_val is None:
            continue
        null_vals = null_metrics[metric]
        if not null_vals:
            continue
        null_mean = np.mean(null_vals)
        null_std = np.std(null_vals)
        z_score = (real_val - null_mean) / null_std if null_std > 0 else float('inf')
        result[metric] = {
            "real": float(real_val),
            "null_mean": float(null_mean),
            "null_std": float(null_std),
            "z_score": float(z_score),
            "ratio": float(real_val / null_mean) if null_mean != 0 else float('inf'),
            "significant": abs(z_score) > 2.0,
        }

    return result


# ---------------------------------------------------------------------------
# 3. Rich club analysis
# ---------------------------------------------------------------------------
def rich_club_analysis(G):
    """Rich club coefficient: do high-degree nodes preferentially connect to each other?

    φ(k) = fraction of edges between nodes of degree ≥ k out of maximum possible.
    φ(k) > φ_random(k) indicates a rich club.
    """
    if G.number_of_edges() == 0:
        return {"error": "no edges"}

    # Compute normalized rich club coefficient
    rc = nx.rich_club_coefficient(G, normalized=False)

    # Compute for random reference (degree-preserving)
    degree_seq = [d for _, d in G.degree()]
    rc_random_samples = []
    rng = np.random.default_rng(42)

    for _ in range(5):
        try:
            R = nx.configuration_model(degree_seq, seed=int(rng.integers(1e9)))
            R = nx.Graph(R)
            R.remove_edges_from(nx.selfloop_edges(R))
            rc_r = nx.rich_club_coefficient(R, normalized=False)
            rc_random_samples.append(rc_r)
        except Exception:
            continue

    # Normalize: φ_norm(k) = φ(k) / φ_random(k)
    result = {}
    for k in sorted(rc.keys()):
        null_vals = [s.get(k, 0) for s in rc_random_samples if k in s]
        null_mean = np.mean(null_vals) if null_vals else 0
        result[k] = {
            "phi": float(rc[k]),
            "phi_null": float(null_mean),
            "phi_normalized": float(rc[k] / null_mean) if null_mean > 0 else None,
        }

    # Summary: at what degree range is the rich club strongest?
    normalized_vals = [(k, v["phi_normalized"]) for k, v in result.items()
                       if v["phi_normalized"] is not None and v["phi_normalized"] > 0]

    if normalized_vals:
        max_k, max_phi = max(normalized_vals, key=lambda x: x[1])
        summary = {
            "max_normalized_phi": float(max_phi),
            "max_phi_at_degree": int(max_k),
            "has_rich_club": max_phi > 1.0,
            "degree_range": [int(normalized_vals[0][0]), int(normalized_vals[-1][0])],
        }
    else:
        summary = {"has_rich_club": False}

    return {"coefficients": result, "summary": summary}


# ---------------------------------------------------------------------------
# 4. Spectral analysis
# ---------------------------------------------------------------------------
def spectral_analysis(G, n_eigenvalues=20):
    """Analyze the Laplacian spectrum: spectral gap, algebraic connectivity, etc."""
    if G.number_of_nodes() < 3:
        return {"error": "too few nodes"}

    # Use giant component for spectral analysis
    gc_nodes = max(nx.connected_components(G), key=len)
    gc = G.subgraph(gc_nodes).copy()

    if gc.number_of_nodes() < 3:
        return {"error": "giant component too small"}

    # Normalized Laplacian eigenvalues
    try:
        eigenvalues = np.sort(nx.laplacian_spectrum(gc))
    except Exception as e:
        return {"error": str(e)}

    # Algebraic connectivity (second smallest eigenvalue of Laplacian)
    algebraic_connectivity = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0

    # Spectral gap (λ_n - λ_{n-1} for normalized, or λ_2 for regular)
    spectral_gap = algebraic_connectivity

    # Spectral radius (largest eigenvalue of adjacency matrix)
    try:
        adj_eigenvalues = np.sort(nx.adjacency_spectrum(gc))[::-1]
        spectral_radius = float(np.real(adj_eigenvalues[0]))
    except Exception:
        spectral_radius = None

    # Eigenvalue distribution statistics
    result = {
        "gc_nodes": gc.number_of_nodes(),
        "gc_edges": gc.number_of_edges(),
        "algebraic_connectivity": algebraic_connectivity,
        "spectral_gap": spectral_gap,
        "spectral_radius": spectral_radius,
        "laplacian_max": float(eigenvalues[-1]),
        "eigenvalue_mean": float(np.mean(eigenvalues)),
        "eigenvalue_std": float(np.std(eigenvalues)),
        # First few non-zero eigenvalues (reveal community structure)
        "smallest_nonzero_eigenvalues": [float(e) for e in eigenvalues[1:min(n_eigenvalues+1, len(eigenvalues))]],
    }

    # Spectral gap analysis: large gap after k-th eigenvalue suggests k natural communities
    if len(eigenvalues) > 2:
        diffs = np.diff(eigenvalues[1:min(30, len(eigenvalues))])
        if len(diffs) > 0:
            max_gap_idx = int(np.argmax(diffs))
            result["max_spectral_gap_after_eigenvalue"] = max_gap_idx + 2  # +2 because we skip λ_0=0 and use 1-indexing
            result["max_spectral_gap_size"] = float(diffs[max_gap_idx])
            result["suggested_num_communities"] = max_gap_idx + 1  # Number of eigenvalues before the gap

    return result


# ---------------------------------------------------------------------------
# 5. Degree-degree correlation profile
# ---------------------------------------------------------------------------
def degree_correlation_profile(G):
    """Analyze how degree correlations vary: k_nn(k) — average neighbor degree as a function of degree."""
    if G.number_of_edges() == 0:
        return {"error": "no edges"}

    knn = nx.average_neighbor_degree(G)
    degree_dict = dict(G.degree())

    # Group by degree
    k_to_knn = {}
    for node, nn_deg in knn.items():
        k = degree_dict[node]
        k_to_knn.setdefault(k, []).append(nn_deg)

    # Average knn per degree
    profile = {}
    for k in sorted(k_to_knn.keys()):
        vals = k_to_knn[k]
        profile[k] = {
            "mean_knn": float(np.mean(vals)),
            "std_knn": float(np.std(vals)),
            "count": len(vals),
        }

    # Fit power law: knn ~ k^μ
    # μ > 0: assortative, μ < 0: disassortative, μ = 0: neutral
    ks = np.array([k for k in profile.keys() if k > 0])
    knns = np.array([profile[k]["mean_knn"] for k in ks])

    if len(ks) > 2:
        log_k = np.log(ks)
        log_knn = np.log(knns[knns > 0])
        if len(log_knn) == len(log_k):
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_knn)
            fit = {
                "mu": float(slope),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "interpretation": "assortative" if slope > 0 else "disassortative" if slope < 0 else "neutral",
            }
        else:
            fit = {"error": "log transform failed (zeros in knn)"}
    else:
        fit = {"error": "too few degree classes"}

    return {"profile": profile, "power_law_fit": fit}


# ---------------------------------------------------------------------------
# 6. Community quality beyond modularity
# ---------------------------------------------------------------------------
def community_quality_analysis(G, seed=42):
    """Multiple measures of community quality."""
    try:
        import community.community_louvain as community_louvain
    except ImportError:
        return {"error": "python-louvain not installed"}

    if G.number_of_edges() == 0:
        return {"error": "no edges"}

    partition = community_louvain.best_partition(G, random_state=seed)
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)

    modularity = community_louvain.modularity(partition, G)

    # Community size distribution
    sizes = sorted([len(c) for c in communities.values()], reverse=True)

    # Internal density per community
    internal_densities = []
    conductances = []
    for comm_nodes in communities.values():
        if len(comm_nodes) < 2:
            continue
        subG = G.subgraph(comm_nodes)
        internal_edges = subG.number_of_edges()
        max_edges = len(comm_nodes) * (len(comm_nodes) - 1) / 2
        internal_densities.append(internal_edges / max_edges if max_edges > 0 else 0)

        # Conductance: fraction of edges leaving the community
        boundary_edges = 0
        total_edges = 0
        for node in comm_nodes:
            for neighbor in G.neighbors(node):
                total_edges += 1
                if neighbor not in comm_nodes:
                    boundary_edges += 1
        conductance = boundary_edges / total_edges if total_edges > 0 else 0
        conductances.append(conductance)

    # Resolution sweep: how many communities at different resolutions?
    resolution_sweep = {}
    for res in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        p = community_louvain.best_partition(G, resolution=res, random_state=seed)
        n_comm = len(set(p.values()))
        mod = community_louvain.modularity(p, G)
        resolution_sweep[str(res)] = {"communities": n_comm, "modularity": float(mod)}

    return {
        "modularity": float(modularity),
        "num_communities": len(communities),
        "community_sizes": sizes,
        "size_gini": float(gini_coefficient(sizes)),
        "mean_internal_density": float(np.mean(internal_densities)) if internal_densities else 0,
        "mean_conductance": float(np.mean(conductances)) if conductances else 0,
        "resolution_sweep": resolution_sweep,
    }


def gini_coefficient(values):
    """Compute Gini coefficient (inequality measure) for a list of values."""
    values = np.array(sorted(values), dtype=float)
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_deep_analysis(edges, label, total_nodes, threshold=0.9):
    """Run all deep analyses on one edge set."""
    print(f"\n{'='*60}")
    print(f"Deep analysis: {label} @ θ={threshold}")
    print(f"{'='*60}")

    results = {"label": label, "threshold": threshold}

    # 1. Edge weight distribution
    print("  [1/6] Edge weight distribution...")
    results["edge_distribution"] = analyze_edge_weight_distribution(edges, label)

    # 2. Build graph
    G = build_graph(edges, threshold, total_nodes)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 3. Null model comparison
    print("  [2/6] Null model comparison (10 rewirings)...")
    results["null_model"] = null_model_comparison(G, n_rewirings=10)

    # 4. Rich club
    print("  [3/6] Rich club analysis...")
    results["rich_club"] = rich_club_analysis(G)

    # 5. Spectral analysis (only feasible for smaller graphs)
    if G.number_of_nodes() <= 2000:
        print("  [4/6] Spectral analysis...")
        results["spectral"] = spectral_analysis(G)
    else:
        print(f"  [4/6] Spectral analysis SKIPPED (n={G.number_of_nodes()} > 2000)")
        results["spectral"] = {"skipped": True, "reason": f"n={G.number_of_nodes()} too large"}

    # 6. Degree correlation profile
    print("  [5/6] Degree-degree correlation profile...")
    results["degree_correlation"] = degree_correlation_profile(G)

    # 7. Community quality
    print("  [6/6] Community quality analysis...")
    results["community_quality"] = community_quality_analysis(G)

    return results


def print_summary(results):
    """Print a human-readable summary of deep analysis results."""
    label = results["label"]

    print(f"\n{'─'*60}")
    print(f"  SUMMARY: {label}")
    print(f"{'─'*60}")

    # Edge distribution
    ed = results.get("edge_distribution", {})
    if ed:
        print(f"\n  Edge weight distribution:")
        print(f"    Mean={ed['mean']:.4f}, Median={ed['median']:.4f}, Std={ed['std']:.4f}")
        print(f"    Skewness={ed['skewness']:.3f}, Kurtosis={ed['kurtosis']:.3f}")
        print(f"    P90={ed['p90']:.4f}, P95={ed['p95']:.4f}, P99={ed['p99']:.4f}")
        for t in [0.85, 0.90, 0.95]:
            frac = ed.get(f'frac_above_{t}', 0)
            count = ed.get(f'count_above_{t}', 0)
            print(f"    Above θ={t}: {count:,} edges ({frac:.2%})")

    # Null model
    nm = results.get("null_model", {})
    if nm and "error" not in nm:
        print(f"\n  Null model comparison (vs configuration model):")
        for metric, vals in nm.items():
            if isinstance(vals, dict) and "z_score" in vals:
                star = " ***" if vals["significant"] else ""
                print(f"    {metric}: real={vals['real']:.4f}, null={vals['null_mean']:.4f}, "
                      f"ratio={vals['ratio']:.2f}x, z={vals['z_score']:+.1f}{star}")

    # Rich club
    rc = results.get("rich_club", {})
    rc_summary = rc.get("summary", {})
    if rc_summary:
        has_rc = rc_summary.get("has_rich_club", False)
        print(f"\n  Rich club: {'YES' if has_rc else 'NO'}")
        if has_rc:
            print(f"    Max normalized φ = {rc_summary['max_normalized_phi']:.3f} at degree {rc_summary['max_phi_at_degree']}")

    # Spectral
    sp = results.get("spectral", {})
    if sp and "error" not in sp and not sp.get("skipped"):
        print(f"\n  Spectral properties:")
        print(f"    Algebraic connectivity: {sp['algebraic_connectivity']:.6f}")
        print(f"    Spectral radius: {sp.get('spectral_radius', 'N/A')}")
        if "suggested_num_communities" in sp:
            print(f"    Spectral gap suggests {sp['suggested_num_communities']} communities "
                  f"(gap size={sp['max_spectral_gap_size']:.4f})")

    # Degree correlation
    dc = results.get("degree_correlation", {})
    fit = dc.get("power_law_fit", {})
    if fit and "mu" in fit:
        print(f"\n  Degree-degree correlation: knn ~ k^{fit['mu']:.3f} (R²={fit['r_squared']:.3f})")
        print(f"    Interpretation: {fit['interpretation']}")

    # Community quality
    cq = results.get("community_quality", {})
    if cq and "error" not in cq:
        print(f"\n  Community quality:")
        print(f"    Modularity: {cq['modularity']:.3f}")
        print(f"    Communities: {cq['num_communities']}")
        print(f"    Size Gini: {cq['size_gini']:.3f} ({'unequal' if cq['size_gini'] > 0.4 else 'balanced'})")
        print(f"    Mean internal density: {cq['mean_internal_density']:.3f}")
        print(f"    Mean conductance: {cq['mean_conductance']:.3f}")
        print(f"    Resolution sweep:")
        for res, vals in cq.get("resolution_sweep", {}).items():
            print(f"      γ={res}: {vals['communities']} communities, mod={vals['modularity']:.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deep network analysis")
    parser.add_argument("--dataset", choices=["chatgpt", "agentic", "both"], default="both")
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    all_results = []

    if args.dataset in ("chatgpt", "both"):
        # Run on key ChatGPT configs
        configs = [
            ("user2.0-ai1.0", "ChatGPT 2:1 (optimal)", 1908),
            ("user1.618-ai1.0", "ChatGPT φ:1 (golden ratio)", 1908),
            ("user1.0-ai1.0", "ChatGPT 1:1 (equal)", 1908),
            ("user100.0-ai1.0", "ChatGPT user-only", 1908),
            ("user1.0-ai100.0", "ChatGPT AI-only", 1908),
        ]
        for label, name, total in configs:
            path = ABLATION_EDGES / f"edges_chatgpt-json-llm-{label}.json"
            if path.exists():
                edges = load_edges(str(path))
                result = run_deep_analysis(edges, name, total, args.threshold)
                print_summary(result)
                all_results.append(result)

    if args.dataset in ("agentic", "both"):
        agentic_edges = AGENTIC_DIR / "data" / "edges-all.json"
        if agentic_edges.exists():
            edges = load_edges(str(agentic_edges))
            result = run_deep_analysis(edges, "Agentic text-only 2:1", 3945, args.threshold)
            print_summary(result)
            all_results.append(result)

    # Save
    output_path = EXPERIMENTS_DIR / "results" / "deep_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
