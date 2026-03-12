"""Null model baselines for semantic and multi-layer network analysis.

Provides configuration model baselines for the semantic layer and
permutation tests for inter-layer correlation and community NMI.
"""
import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score

try:
    import community.community_louvain as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False


def configuration_model_baseline(
    G: nx.Graph,
    n_samples: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compare observed network metrics against configuration model ensembles.

    Generates `n_samples` random graphs with the same degree sequence as G
    and computes modularity, assortativity, clustering, and transitivity
    for each. Returns observed values, null distribution statistics, and
    z-scores.

    Args:
        G: The observed network.
        n_samples: Number of configuration model samples.
        random_state: Seed for reproducibility.

    Returns:
        Dict with observed values, null means/stds, z-scores, and p-values.
    """
    rng = np.random.RandomState(random_state)

    if G.number_of_nodes() == 0:
        empty = {"observed": 0.0, "null_mean": None, "null_std": None,
                 "z_score": None, "p_value": None, "n_samples": 0}
        return {k: dict(empty) for k in
                ["clustering", "transitivity", "assortativity", "modularity"]}

    # Observed metrics
    obs_clustering = nx.average_clustering(G)
    obs_transitivity = nx.transitivity(G)
    try:
        obs_assortativity = nx.degree_assortativity_coefficient(G)
    except (nx.NetworkXError, ValueError):
        obs_assortativity = None

    obs_modularity = None
    if HAS_COMMUNITY and G.number_of_edges() > 0:
        partition = community_louvain.best_partition(G, random_state=42)
        obs_modularity = community_louvain.modularity(partition, G)

    # Generate configuration model samples
    degree_sequence = [d for _, d in G.degree()]
    null_clustering = []
    null_transitivity = []
    null_assortativity = []
    null_modularity = []

    import sys
    for i in range(n_samples):
        print(f"\r    config model sample {i+1}/{n_samples}", end="", file=sys.stderr)
        seed = rng.randint(0, 2**31)
        try:
            R = nx.configuration_model(degree_sequence, seed=seed)
            # Remove parallel edges and self-loops
            R = nx.Graph(R)
            R.remove_edges_from(nx.selfloop_edges(R))
        except nx.NetworkXError:
            continue

        if R.number_of_edges() == 0:
            continue

        null_clustering.append(nx.average_clustering(R))
        null_transitivity.append(nx.transitivity(R))

        try:
            null_assortativity.append(nx.degree_assortativity_coefficient(R))
        except (nx.NetworkXError, ValueError):
            pass

        if HAS_COMMUNITY:
            part = community_louvain.best_partition(R, random_state=42)
            null_modularity.append(community_louvain.modularity(part, R))
    print("", file=sys.stderr)  # newline after progress

    def _zscore(observed, null_values):
        if observed is None or len(null_values) < 2:
            return {"observed": observed, "null_mean": None, "null_std": None,
                    "z_score": None, "p_value": None, "n_samples": len(null_values)}
        mean = np.mean(null_values)
        std = np.std(null_values, ddof=1)
        if std < 1e-12:
            z = float("inf") if observed != mean else 0.0
            p = 0.0 if observed != mean else 1.0
        else:
            z = (observed - mean) / std
            p = 2 * (1 - stats.norm.cdf(abs(z)))  # two-tailed
        return {
            "observed": observed,
            "null_mean": float(mean),
            "null_std": float(std),
            "z_score": float(z),
            "p_value": float(p),
            "n_samples": len(null_values),
        }

    return {
        "clustering": _zscore(obs_clustering, null_clustering),
        "transitivity": _zscore(obs_transitivity, null_transitivity),
        "assortativity": _zscore(obs_assortativity, null_assortativity),
        "modularity": _zscore(obs_modularity, null_modularity),
    }


def permutation_interlayer_test(
    semantic_G: nx.Graph,
    delegation_G: nx.DiGraph,
    n_perms: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Test whether inter-layer degree correlation exceeds a permutation null.

    Permutes node labels in the delegation layer and recomputes Spearman rho
    for each permutation. The p-value is the fraction of permutations with
    |rho| >= |observed rho|.

    Args:
        semantic_G: Semantic similarity graph.
        delegation_G: Delegation directed graph.
        n_perms: Number of permutations.
        random_state: Seed for reproducibility.

    Returns:
        Dict with observed rho, null distribution stats, and permutation p-value.
    """
    shared = sorted(set(semantic_G.nodes()) & set(delegation_G.nodes()))
    if len(shared) < 3:
        return {"observed_rho": None, "perm_p_value": None, "n_shared": len(shared)}

    sem_degrees = np.array([semantic_G.degree(n) for n in shared])
    del_degrees = np.array([delegation_G.degree(n) for n in shared])

    obs_rho, _ = stats.spearmanr(sem_degrees, del_degrees)
    if np.isnan(obs_rho):
        return {"observed_rho": None, "perm_p_value": None, "n_shared": len(shared)}

    rng = np.random.RandomState(random_state)
    null_rhos = []
    for _ in range(n_perms):
        perm_del = rng.permutation(del_degrees)
        rho, _ = stats.spearmanr(sem_degrees, perm_del)
        if not np.isnan(rho):
            null_rhos.append(rho)

    null_rhos = np.array(null_rhos)
    perm_p = np.mean(np.abs(null_rhos) >= abs(obs_rho)) if len(null_rhos) > 0 else None

    return {
        "observed_rho": float(obs_rho),
        "null_mean": float(np.mean(null_rhos)) if len(null_rhos) > 0 else None,
        "null_std": float(np.std(null_rhos, ddof=1)) if len(null_rhos) > 1 else None,
        "perm_p_value": float(perm_p) if perm_p is not None else None,
        "n_perms": len(null_rhos),
        "n_shared": len(shared),
    }


def permutation_nmi_test(
    partition_a: Dict[str, int],
    partition_b: Dict[str, int],
    n_perms: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Test whether observed NMI exceeds a permutation null.

    Permutes labels in partition_b and recomputes NMI for each permutation.
    The p-value is the fraction of permutations with NMI >= observed NMI.

    Args:
        partition_a: Node -> community label (e.g., Louvain semantic).
        partition_b: Node -> community label (e.g., delegation components).
        n_perms: Number of permutations.
        random_state: Seed for reproducibility.

    Returns:
        Dict with observed NMI, null distribution stats, and permutation p-value.
    """
    shared = sorted(set(partition_a.keys()) & set(partition_b.keys()))
    if len(shared) < 2:
        return {"observed_nmi": 0.0, "perm_p_value": None, "n_shared": len(shared)}

    labels_a = np.array([partition_a[n] for n in shared])
    labels_b = np.array([partition_b[n] for n in shared])
    obs_nmi = normalized_mutual_info_score(labels_a, labels_b)

    rng = np.random.RandomState(random_state)
    null_nmis = []
    for _ in range(n_perms):
        perm_b = rng.permutation(labels_b)
        null_nmis.append(normalized_mutual_info_score(labels_a, perm_b))

    null_nmis = np.array(null_nmis)
    perm_p = np.mean(null_nmis >= obs_nmi)

    return {
        "observed_nmi": float(obs_nmi),
        "null_mean": float(np.mean(null_nmis)),
        "null_std": float(np.std(null_nmis, ddof=1)) if len(null_nmis) > 1 else None,
        "perm_p_value": float(perm_p),
        "n_perms": len(null_nmis),
        "n_shared": len(shared),
    }
