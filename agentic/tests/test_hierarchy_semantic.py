"""Tests for hierarchy_semantic.py — semantic concept hierarchy pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hierarchy_semantic import (
    build_episode_concept_matrix,
    cluster_concepts,
    build_bipartite_graph,
    build_semantic_dag,
    analyze_semantic_hierarchy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_state():
    return {
        'vocabulary': ['bayesian inference', 'python programming', 'mcmc sampling',
                       'neural networks', 'data visualization'],
        'vocabulary_size': 5,
        'total_extractions': 4,
        'total_concepts_raw': 12,
        'episodes': [
            {'episode_id': 'ep1', 'title': 'Bayesian Stats',
             'concepts': ['bayesian inference', 'mcmc sampling', 'data visualization'],
             'message_count': 4, 'extraction_time': 1.0},
            {'episode_id': 'ep2', 'title': 'Python Basics',
             'concepts': ['python programming', 'data visualization'],
             'message_count': 6, 'extraction_time': 0.8},
            {'episode_id': 'ep3', 'title': 'Deep Learning',
             'concepts': ['neural networks', 'python programming', 'bayesian inference'],
             'message_count': 10, 'extraction_time': 1.2},
            {'episode_id': 'ep4', 'title': 'MCMC Tutorial',
             'concepts': ['mcmc sampling', 'bayesian inference',
                          'python programming', 'data visualization'],
             'message_count': 8, 'extraction_time': 1.5},
        ],
    }


@pytest.fixture
def sample_embeddings():
    """Generate deterministic pseudo-embeddings for 5 concepts."""
    rng = np.random.RandomState(42)
    embs = rng.randn(5, 32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


# ---------------------------------------------------------------------------
# Tests: Episode-concept matrix
# ---------------------------------------------------------------------------

def test_matrix_shape(sample_state):
    episode_ids, concepts, matrix = build_episode_concept_matrix(sample_state)
    assert matrix.shape == (4, 5)
    assert len(episode_ids) == 4
    assert len(concepts) == 5


def test_matrix_is_binary(sample_state):
    _, _, matrix = build_episode_concept_matrix(sample_state)
    assert set(np.unique(matrix)) <= {0.0, 1.0}


def test_matrix_row_sums(sample_state):
    episode_ids, _, matrix = build_episode_concept_matrix(sample_state)
    row_sums = matrix.sum(axis=1)
    ep_concept_counts = {ep['episode_id']: len(ep['concepts'])
                        for ep in sample_state['episodes']}
    for i, eid in enumerate(episode_ids):
        assert row_sums[i] == ep_concept_counts[eid]


def test_matrix_case_insensitive(sample_state):
    """Concepts differing only in case should map to same column."""
    state = sample_state.copy()
    state['episodes'] = [
        {'episode_id': 'ep1', 'title': 'Test',
         'concepts': ['Bayesian Inference', 'MCMC Sampling'],
         'message_count': 4, 'extraction_time': 1.0},
    ]
    _, concepts, matrix = build_episode_concept_matrix(state)
    assert matrix.sum() == 2  # both should match


# ---------------------------------------------------------------------------
# Tests: Concept clustering
# ---------------------------------------------------------------------------

def test_cluster_returns_levels(sample_embeddings):
    concepts = ['c1', 'c2', 'c3', 'c4', 'c5']
    levels = cluster_concepts(sample_embeddings, concepts, [3, 2])
    assert len(levels) == 2
    assert levels[0]['n_clusters'] == 3
    assert levels[1]['n_clusters'] == 2


def test_cluster_labels_cover_all_concepts(sample_embeddings):
    concepts = ['c1', 'c2', 'c3', 'c4', 'c5']
    levels = cluster_concepts(sample_embeddings, concepts, [3])
    all_members = []
    for members in levels[0]['cluster_membership'].values():
        all_members.extend(members)
    assert set(all_members) == set(concepts)


def test_cluster_centroids_normalized(sample_embeddings):
    concepts = ['c1', 'c2', 'c3', 'c4', 'c5']
    levels = cluster_concepts(sample_embeddings, concepts, [3])
    centroids = levels[0]['centroids']
    for i in range(centroids.shape[0]):
        norm = np.linalg.norm(centroids[i])
        if norm > 0:
            assert abs(norm - 1.0) < 1e-6


def test_cluster_silhouette_in_range(sample_embeddings):
    concepts = ['c1', 'c2', 'c3', 'c4', 'c5']
    levels = cluster_concepts(sample_embeddings, concepts, [3])
    assert -1.0 <= levels[0]['silhouette'] <= 1.0


def test_cluster_too_few_concepts():
    embs = np.random.randn(2, 10)
    levels = cluster_concepts(embs, ['a', 'b'], [2])
    assert levels == []


# ---------------------------------------------------------------------------
# Tests: Bipartite graph
# ---------------------------------------------------------------------------

def test_bipartite_node_types(sample_state):
    episode_ids, concepts, matrix = build_episode_concept_matrix(sample_state)
    B = build_bipartite_graph(episode_ids, concepts, matrix)

    ep_nodes = [n for n, d in B.nodes(data=True) if d['node_type'] == 'episode']
    c_nodes = [n for n, d in B.nodes(data=True) if d['node_type'] == 'concept']
    assert len(ep_nodes) == 4
    assert len(c_nodes) == 5


def test_bipartite_edge_count(sample_state):
    episode_ids, concepts, matrix = build_episode_concept_matrix(sample_state)
    B = build_bipartite_graph(episode_ids, concepts, matrix)
    assert B.number_of_edges() == int(matrix.sum())


# ---------------------------------------------------------------------------
# Tests: DAG construction
# ---------------------------------------------------------------------------

def test_dag_is_acyclic(sample_state, sample_embeddings):
    episode_ids, concepts, matrix = build_episode_concept_matrix(sample_state)
    levels = cluster_concepts(sample_embeddings, concepts, [3, 2])
    dag = build_semantic_dag(episode_ids, concepts, matrix, levels)
    assert dag.is_directed()
    import networkx as nx
    assert nx.is_directed_acyclic_graph(dag)


def test_dag_has_episode_nodes(sample_state, sample_embeddings):
    episode_ids, concepts, matrix = build_episode_concept_matrix(sample_state)
    levels = cluster_concepts(sample_embeddings, concepts, [3, 2])
    dag = build_semantic_dag(episode_ids, concepts, matrix, levels)
    for eid in episode_ids:
        assert eid in dag


def test_dag_episode_to_concept_edges(sample_state, sample_embeddings):
    episode_ids, concepts, matrix = build_episode_concept_matrix(sample_state)
    levels = cluster_concepts(sample_embeddings, concepts, [3, 2])
    dag = build_semantic_dag(episode_ids, concepts, matrix, levels)

    # Check ep1 has edges to its concepts
    ep1_successors = list(dag.successors('ep1'))
    ep1_concepts = [s for s in ep1_successors if s.startswith('C:')]
    assert len(ep1_concepts) == 3  # ep1 has 3 concepts


# ---------------------------------------------------------------------------
# Tests: Analysis
# ---------------------------------------------------------------------------

def test_analysis_returns_expected_keys(sample_state, sample_embeddings):
    episode_ids, concepts, matrix = build_episode_concept_matrix(sample_state)
    levels = cluster_concepts(sample_embeddings, concepts, [3, 2])
    dag = build_semantic_dag(episode_ids, concepts, matrix, levels)
    bipartite = build_bipartite_graph(episode_ids, concepts, matrix)

    results = analyze_semantic_hierarchy(
        episode_ids, concepts, matrix, levels, dag, bipartite
    )

    assert 'n_episodes' in results
    assert 'n_concepts' in results
    assert 'episode_concept_stats' in results
    assert 'levels' in results
    assert 'dag' in results
    assert 'top_concepts' in results
    assert 'concept_co_occurrence' in results


def test_analysis_dag_is_confirmed_acyclic(sample_state, sample_embeddings):
    episode_ids, concepts, matrix = build_episode_concept_matrix(sample_state)
    levels = cluster_concepts(sample_embeddings, concepts, [3, 2])
    dag = build_semantic_dag(episode_ids, concepts, matrix, levels)
    bipartite = build_bipartite_graph(episode_ids, concepts, matrix)

    results = analyze_semantic_hierarchy(
        episode_ids, concepts, matrix, levels, dag, bipartite
    )
    assert results['dag']['is_dag'] is True
