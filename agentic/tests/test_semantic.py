"""Tests for semantic network metrics computation."""
import networkx as nx
import numpy as np
from agentic.semantic import compute_network_metrics


def test_compute_metrics_basic():
    G = nx.Graph()
    G.add_edges_from([("a", "b", {"weight": 0.95}),
                      ("b", "c", {"weight": 0.92}),
                      ("a", "c", {"weight": 0.91})])
    metrics = compute_network_metrics(G)
    assert metrics["node_count"] == 3
    assert metrics["edge_count"] == 3
    assert metrics["density"] > 0
    assert "modularity" in metrics
    assert "avg_clustering" in metrics
    assert "giant_component_size" in metrics


def test_empty_graph():
    G = nx.Graph()
    metrics = compute_network_metrics(G)
    assert metrics["node_count"] == 0
    assert metrics["edge_count"] == 0


def test_disconnected_graph():
    G = nx.Graph()
    G.add_edges_from([("a", "b"), ("c", "d")])
    metrics = compute_network_metrics(G)
    assert metrics["num_components"] == 2
    assert metrics["giant_component_size"] == 2
    assert metrics["giant_component_fraction"] == 0.5


def test_single_node():
    G = nx.Graph()
    G.add_node("a")
    metrics = compute_network_metrics(G)
    assert metrics["node_count"] == 1
    assert metrics["edge_count"] == 0
    assert metrics["avg_shortest_path"] is None


def test_metrics_deterministic():
    """Same graph produces same metrics (random_state controls Louvain)."""
    G = nx.Graph()
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
    m1 = compute_network_metrics(G, random_state=42)
    m2 = compute_network_metrics(G, random_state=42)
    # NaN != NaN, so compare key by key handling None/NaN
    for key in m1:
        v1, v2 = m1[key], m2[key]
        if v1 is None:
            assert v2 is None, f"{key}: {v1} != {v2}"
        elif isinstance(v1, float) and np.isnan(v1):
            assert isinstance(v2, float) and np.isnan(v2), f"{key}: {v1} != {v2}"
        else:
            assert v1 == v2, f"{key}: {v1} != {v2}"
