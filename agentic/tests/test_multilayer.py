"""Tests for two-layer multiplex analysis."""
import networkx as nx
from agentic.multilayer import (
    compute_interlayer_degree_correlation,
    compute_participation_coefficient,
    compare_community_assignments,
)


def _make_layers():
    """Create sample semantic and delegation graphs with shared nodes."""
    sem = nx.Graph()
    sem.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])

    deleg = nx.DiGraph()
    deleg.add_edges_from([("a", "b"), ("a", "c"), ("a", "d")])

    return sem, deleg


def test_interlayer_degree_correlation():
    # Use a graph where degrees vary across layers
    sem = nx.Graph()
    sem.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])  # degrees: a=1, b=2, c=2, d=1
    deleg = nx.DiGraph()
    deleg.add_edges_from([("a", "b"), ("a", "c"), ("a", "d")])  # degrees: a=3, b=1, c=1, d=1
    result = compute_interlayer_degree_correlation(sem, deleg)
    assert result["n_shared_nodes"] == 4
    assert result["correlation"] is not None
    assert -1.0 <= result["correlation"] <= 1.0


def test_interlayer_too_few_nodes():
    sem = nx.Graph()
    sem.add_edge("a", "b")
    deleg = nx.DiGraph()
    deleg.add_edge("c", "d")  # No overlap
    result = compute_interlayer_degree_correlation(sem, deleg)
    assert result["correlation"] is None


def test_participation_coefficient():
    sem, deleg = _make_layers()
    pc = compute_participation_coefficient(sem, deleg)
    assert len(pc) == 4
    # All nodes have connections in both layers -> P > 0
    for node in ["a", "b", "c", "d"]:
        assert 0.0 <= pc[node] <= 0.5


def test_participation_single_layer_node():
    """Node in only one layer has P=0."""
    sem = nx.Graph()
    sem.add_edge("a", "b")
    deleg = nx.DiGraph()
    deleg.add_edge("c", "d")
    pc = compute_participation_coefficient(sem, deleg)
    # a and b only in semantic -> P=0
    assert pc["a"] == 0.0
    assert pc["b"] == 0.0


def test_compare_community_assignments_identical():
    partition = {"a": 0, "b": 0, "c": 1, "d": 1}
    nmi = compare_community_assignments(partition, partition)
    assert nmi == 1.0


def test_compare_community_assignments_different():
    sem_part = {"a": 0, "b": 0, "c": 1, "d": 1}
    del_part = {"a": 0, "b": 1, "c": 0, "d": 1}
    nmi = compare_community_assignments(sem_part, del_part)
    assert 0.0 <= nmi < 1.0


def test_compare_community_no_shared_nodes():
    nmi = compare_community_assignments({"a": 0}, {"b": 1})
    assert nmi == 0.0
