"""Semantic network metrics computation.

Computes global, node-level, and community metrics for a similarity network.
Designed to produce output comparable to the ChatGPT temporal analysis.
"""
from typing import Any, Dict

import networkx as nx
import numpy as np

try:
    import community.community_louvain as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False


def compute_network_metrics(
    G: nx.Graph, random_state: int = 42
) -> Dict[str, Any]:
    """Compute comprehensive network metrics.

    Returns a flat dict of scalar metrics, compatible with CSV/DataFrame output.
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()

    if n == 0:
        return {"node_count": 0, "edge_count": 0, "density": 0.0}

    degrees = [d for _, d in G.degree()]

    # Giant component
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    gc = components[0] if components else set()
    gc_size = len(gc)

    metrics = {
        "node_count": n,
        "edge_count": e,
        "density": nx.density(G),
        "num_components": len(components),
        "giant_component_size": gc_size,
        "giant_component_fraction": gc_size / n if n else 0.0,
        "avg_degree": np.mean(degrees) if degrees else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
    }

    # Shortest paths (giant component only)
    if gc_size > 1:
        gc_sub = G.subgraph(gc)
        try:
            metrics["avg_shortest_path"] = nx.average_shortest_path_length(gc_sub)
        except nx.NetworkXError:
            metrics["avg_shortest_path"] = None
    else:
        metrics["avg_shortest_path"] = None

    # Community detection (requires at least one edge)
    if HAS_COMMUNITY and e > 0:
        partition = community_louvain.best_partition(G, random_state=random_state)
        metrics["modularity"] = community_louvain.modularity(partition, G)
        metrics["num_communities"] = len(set(partition.values()))
    else:
        metrics["modularity"] = 0.0
        metrics["num_communities"] = 1 if n > 0 else 0

    # Assortativity
    try:
        metrics["assortativity"] = nx.degree_assortativity_coefficient(G)
    except (nx.NetworkXError, ValueError):
        metrics["assortativity"] = None

    return metrics
