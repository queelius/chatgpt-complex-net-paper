"""Tests for null model baselines."""
import networkx as nx
import numpy as np
from agentic.null_models import (
    configuration_model_baseline,
    permutation_interlayer_test,
    permutation_nmi_test,
)


def _make_clustered_graph():
    """Two cliques connected by a single bridge — high modularity by design."""
    G = nx.Graph()
    # Clique 1: nodes 0-4
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(i, j)
    # Clique 2: nodes 5-9
    for i in range(5, 10):
        for j in range(i + 1, 10):
            G.add_edge(i, j)
    # Bridge
    G.add_edge(4, 5)
    return G


def test_configuration_model_returns_all_metrics():
    G = _make_clustered_graph()
    result = configuration_model_baseline(G, n_samples=10, random_state=42)
    for key in ["clustering", "transitivity", "assortativity", "modularity"]:
        assert key in result
        assert "observed" in result[key]
        assert "null_mean" in result[key]
        assert "z_score" in result[key]


def test_configuration_model_modularity_significant():
    """A two-clique graph should have modularity well above configuration model."""
    G = _make_clustered_graph()
    result = configuration_model_baseline(G, n_samples=50, random_state=42)
    mod = result["modularity"]
    # Observed modularity should be clearly higher than null
    assert mod["observed"] > mod["null_mean"]
    assert mod["z_score"] > 1.0


def test_configuration_model_empty_graph():
    G = nx.Graph()
    result = configuration_model_baseline(G, n_samples=5)
    assert result["clustering"]["observed"] == 0.0


def test_permutation_interlayer_random_layers():
    """Independent Erdos-Renyi layers should produce rho near zero."""
    nodes = list(range(80))
    # Two independent ER graphs with separate seeds — no structural correlation
    sem = nx.erdos_renyi_graph(80, 0.08, seed=111)
    deleg = nx.DiGraph(nx.erdos_renyi_graph(80, 0.08, seed=999))

    result = permutation_interlayer_test(sem, deleg, n_perms=200, random_state=42)
    assert result["observed_rho"] is not None
    # p-value should be relatively high (not significant) for independent layers
    assert result["perm_p_value"] > 0.01


def test_permutation_interlayer_too_few_shared():
    sem = nx.Graph()
    sem.add_edge("a", "b")
    deleg = nx.DiGraph()
    deleg.add_edge("c", "d")
    result = permutation_interlayer_test(sem, deleg, n_perms=100)
    assert result["observed_rho"] is None


def test_permutation_nmi_identical_partitions():
    """Identical partitions should have NMI=1 and low p-value."""
    part = {str(i): i % 3 for i in range(30)}
    result = permutation_nmi_test(part, part, n_perms=200, random_state=42)
    assert result["observed_nmi"] == 1.0
    assert result["perm_p_value"] < 0.05


def test_permutation_nmi_random_partitions():
    """Random partitions should have low NMI and high p-value."""
    # Use independent RNGs with many categories to minimize chance agreement
    rng_a = np.random.RandomState(111)
    rng_b = np.random.RandomState(999)
    part_a = {str(i): rng_a.randint(0, 10) for i in range(200)}
    part_b = {str(i): rng_b.randint(0, 10) for i in range(200)}
    result = permutation_nmi_test(part_a, part_b, n_perms=200, random_state=42)
    assert result["observed_nmi"] < 0.3
    # p-value should be high since partitions are independent
    assert result["perm_p_value"] > 0.05


def test_permutation_nmi_no_shared():
    result = permutation_nmi_test({"a": 0}, {"b": 1}, n_perms=100)
    assert result["observed_nmi"] == 0.0
