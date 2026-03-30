"""Tests for hierarchy.py — geometric hierarchical memory network."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hierarchy import (
    compute_similarity_matrix,
    build_dendrogram,
    cut_dendrogram,
    find_optimal_cuts,
    build_hierarchy,
    build_dag,
    build_intralevel_graph,
    analyze_hierarchy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_embeddings():
    """5 episodes in 2 clear clusters (3 + 2)."""
    rng = np.random.RandomState(42)
    # Cluster A: centered around [1, 0, 0, ...]
    a = rng.randn(3, 10) * 0.1
    a[:, 0] += 3.0
    # Cluster B: centered around [0, 1, 0, ...]
    b = rng.randn(2, 10) * 0.1
    b[:, 1] += 3.0
    emb = np.vstack([a, b])
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    ids = [f"ep_{i}" for i in range(5)]
    return ids, emb


@pytest.fixture
def medium_embeddings():
    """30 episodes in 3 clear clusters (10 + 10 + 10)."""
    rng = np.random.RandomState(42)
    clusters = []
    for i in range(3):
        c = rng.randn(10, 20) * 0.1
        c[:, i * 5] += 3.0  # offset each cluster in a different dimension
        clusters.append(c)
    emb = np.vstack(clusters)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    ids = [f"ep_{i}" for i in range(30)]
    return ids, emb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_similarity_matrix_shape(small_embeddings):
    ids, emb = small_embeddings
    sim = compute_similarity_matrix(emb)
    assert sim.shape == (5, 5)
    assert np.allclose(np.diag(sim), 1.0)


def test_similarity_matrix_symmetric(small_embeddings):
    ids, emb = small_embeddings
    sim = compute_similarity_matrix(emb)
    assert np.allclose(sim, sim.T)


def test_similarity_within_cluster_higher(small_embeddings):
    """Within-cluster similarity should be higher than between-cluster."""
    ids, emb = small_embeddings
    sim = compute_similarity_matrix(emb)
    within_a = sim[:3, :3][np.triu_indices(3, k=1)].mean()
    between = sim[:3, 3:].mean()
    assert within_a > between


def test_dendrogram_shape(small_embeddings):
    ids, emb = small_embeddings
    sim = compute_similarity_matrix(emb)
    Z = build_dendrogram(sim, method="average")
    assert Z.shape == (4, 4)  # N-1 merges


def test_ward_linkage_requires_embeddings(small_embeddings):
    ids, emb = small_embeddings
    sim = compute_similarity_matrix(emb)
    with pytest.raises(ValueError, match="Ward linkage requires"):
        build_dendrogram(sim, method="ward")


def test_ward_linkage_works(small_embeddings):
    ids, emb = small_embeddings
    sim = compute_similarity_matrix(emb)
    Z = build_dendrogram(sim, method="ward", embeddings=emb)
    assert Z.shape == (4, 4)


def test_cut_dendrogram_k2(small_embeddings):
    """Cutting into 2 clusters should separate the two groups."""
    ids, emb = small_embeddings
    sim = compute_similarity_matrix(emb)
    Z = build_dendrogram(sim, method="ward", embeddings=emb)
    labels = cut_dendrogram(Z, n_clusters=2)
    assert len(labels) == 5
    # First 3 should be in same cluster, last 2 in another
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def test_find_optimal_cuts(medium_embeddings):
    ids, emb = medium_embeddings
    sim = compute_similarity_matrix(emb)
    Z = build_dendrogram(sim, method="ward", embeddings=emb)
    results = find_optimal_cuts(Z, emb, k_range=(2, 10))
    assert len(results) > 0
    # k=3 should score high since data has 3 clusters
    k3 = [r for r in results if r["k"] == 3]
    assert len(k3) == 1
    assert k3[0]["silhouette"] > 0


def test_build_hierarchy(medium_embeddings):
    ids, emb = medium_embeddings
    sim = compute_similarity_matrix(emb)
    Z = build_dendrogram(sim, method="ward", embeddings=emb)
    levels = build_hierarchy(Z, emb, ids, [6, 3], ["concepts", "domains"])
    assert len(levels) == 2
    assert levels[0].n_clusters == 6
    assert levels[1].n_clusters == 3
    assert len(levels[0].labels) == 30
    assert levels[0].centroids.shape == (6, 20)


def test_centroids_normalized(medium_embeddings):
    ids, emb = medium_embeddings
    sim = compute_similarity_matrix(emb)
    Z = build_dendrogram(sim, method="ward", embeddings=emb)
    levels = build_hierarchy(Z, emb, ids, [3], ["concepts"])
    norms = np.linalg.norm(levels[0].centroids, axis=1)
    assert np.allclose(norms, 1.0, atol=0.01)


def test_build_dag(medium_embeddings):
    ids, emb = medium_embeddings
    sim = compute_similarity_matrix(emb)
    Z = build_dendrogram(sim, method="ward", embeddings=emb)
    levels = build_hierarchy(Z, emb, ids, [6, 3], ["concepts", "domains"])
    dag = build_dag(ids, levels)

    assert dag.number_of_nodes() > 30  # 30 episodes + cluster nodes
    assert nx.is_directed_acyclic_graph(dag)

    # All episodes should be in the DAG
    for eid in ids:
        assert eid in dag.nodes


def test_build_intralevel_graph(medium_embeddings):
    ids, emb = medium_embeddings
    sim = compute_similarity_matrix(emb)
    Z = build_dendrogram(sim, method="ward", embeddings=emb)
    levels = build_hierarchy(Z, emb, ids, [3], ["concepts"])
    G = build_intralevel_graph(levels[0].centroids, level=1, threshold=-1.0)
    # With threshold=-1, all pairs should be connected (cosine ≥ -1 always)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3  # 3 choose 2


def test_analyze_hierarchy(medium_embeddings):
    ids, emb = medium_embeddings
    sim = compute_similarity_matrix(emb)
    Z = build_dendrogram(sim, method="ward", embeddings=emb)
    levels = build_hierarchy(Z, emb, ids, [6, 3], ["concepts", "domains"])
    dag = build_dag(ids, levels)
    results = analyze_hierarchy(ids, emb, levels, dag)

    assert results["n_episodes"] == 30
    assert results["n_levels"] == 2
    assert len(results["levels"]) == 2
    assert results["dag"]["is_dag"] is True


def test_cluster_sizes_sum_to_n(medium_embeddings):
    """All episodes should be assigned to exactly one cluster per level."""
    ids, emb = medium_embeddings
    sim = compute_similarity_matrix(emb)
    Z = build_dendrogram(sim, method="ward", embeddings=emb)
    levels = build_hierarchy(Z, emb, ids, [6, 3], ["concepts", "domains"])
    for lvl in levels:
        assert sum(lvl.cluster_sizes) == 30


# Need networkx for DAG tests
import networkx as nx


def test_dag_episode_to_top_path(medium_embeddings):
    """Every episode should have a path to the top level."""
    ids, emb = medium_embeddings
    sim = compute_similarity_matrix(emb)
    Z = build_dendrogram(sim, method="ward", embeddings=emb)
    levels = build_hierarchy(Z, emb, ids, [6, 3], ["concepts", "domains"])
    dag = build_dag(ids, levels)

    top_nodes = [n for n, d in dag.nodes(data=True) if d.get("level") == 2]
    for eid in ids:
        reachable = False
        for top in top_nodes:
            if nx.has_path(dag, eid, top):
                reachable = True
                break
        assert reachable, f"Episode {eid} has no path to top level"
