"""Two-layer multiplex analysis: semantic + delegation.

Computes inter-layer correlations and cross-layer community comparisons
following De Domenico et al. (2013) and Mucha et al. (2010).
"""
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score

try:
    import community.community_louvain as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False


def compute_interlayer_degree_correlation(
    semantic_G: nx.Graph,
    delegation_G: nx.DiGraph,
) -> Dict[str, Any]:
    """Compute Spearman correlation of node degrees across layers.

    For nodes present in both layers, correlate semantic degree with
    delegation degree (in + out).

    Returns:
        Dict with correlation, p_value, n_shared_nodes.
    """
    shared = set(semantic_G.nodes()) & set(delegation_G.nodes())
    if len(shared) < 3:
        return {"correlation": None, "p_value": None, "n_shared_nodes": len(shared)}

    sem_degrees = [semantic_G.degree(n) for n in shared]
    del_degrees = [delegation_G.degree(n) for n in shared]

    corr, p_val = stats.spearmanr(sem_degrees, del_degrees)
    # spearmanr returns NaN for constant inputs
    if np.isnan(corr):
        corr = None
        p_val = None
    return {
        "correlation": corr,
        "p_value": p_val,
        "n_shared_nodes": len(shared),
    }


def compute_participation_coefficient(
    semantic_G: nx.Graph,
    delegation_G: nx.DiGraph,
) -> Dict[str, float]:
    """Compute participation coefficient for nodes across two layers.

    P_i = 1 - sum_l (k_il / k_i)^2

    where k_il is degree in layer l, k_i is total degree across layers.
    P=0 means all connections in one layer; P=0.5 means equal across both.

    Returns:
        Dict mapping node_id -> participation coefficient.
    """
    all_nodes = set(semantic_G.nodes()) | set(delegation_G.nodes())
    participation = {}

    for node in all_nodes:
        k_sem = semantic_G.degree(node) if node in semantic_G else 0
        k_del = delegation_G.degree(node) if node in delegation_G else 0
        k_total = k_sem + k_del
        if k_total == 0:
            participation[node] = 0.0
            continue

        participation[node] = 1.0 - (k_sem / k_total) ** 2 - (k_del / k_total) ** 2

    return participation


def compare_community_assignments(
    semantic_partition: Dict[str, int],
    delegation_partition: Dict[str, int],
) -> float:
    """Compare community structures across layers using NMI.

    Only considers nodes present in both partitions.

    Returns:
        Normalized Mutual Information score (0 = independent, 1 = identical).
    """
    shared = set(semantic_partition.keys()) & set(delegation_partition.keys())
    if len(shared) < 2:
        return 0.0

    nodes = sorted(shared)
    labels_sem = [semantic_partition[n] for n in nodes]
    labels_del = [delegation_partition[n] for n in nodes]

    return normalized_mutual_info_score(labels_sem, labels_del)
