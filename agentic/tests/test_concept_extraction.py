"""Tests for concept extraction pipeline."""

import json
import networkx as nx
import pytest

from agentic.concept_extraction import (
    detect_communities,
    get_community_members,
    parse_concepts,
    build_extraction_prompt,
    ConceptNode,
)


def _make_graph():
    """Create a small test graph with two obvious communities."""
    G = nx.Graph()
    # Community A: nodes 0-4 fully connected
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(str(i), str(j), weight=0.95)
    # Community B: nodes 5-9 fully connected
    for i in range(5, 10):
        for j in range(i + 1, 10):
            G.add_edge(str(i), str(j), weight=0.95)
    # One bridge edge
    G.add_edge("4", "5", weight=0.91)
    return G


def test_detect_communities():
    G = _make_graph()
    partition = detect_communities(G)
    assert len(partition) == 10
    # Nodes 0-4 should be in one community, 5-9 in another
    comm_a = partition["0"]
    comm_b = partition["5"]
    assert comm_a != comm_b
    for i in range(5):
        assert partition[str(i)] == comm_a
    for i in range(5, 10):
        assert partition[str(i)] == comm_b


def test_get_community_members():
    partition = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 1}
    members = get_community_members(partition)
    assert len(members) == 2
    assert set(members[0]) == {"a", "b"}
    assert set(members[1]) == {"c", "d", "e"}


def test_parse_concepts_valid_json():
    response = """Here are the concepts:
[
  {"name": "Bayesian Inference", "description": "Using priors to update beliefs", "example_episodes": ["ep1", "ep2"]},
  {"name": "Pattern Recognition", "description": "Finding structure in data", "example_episodes": ["ep3"]}
]"""
    concepts = parse_concepts(response, community_id=5)
    assert len(concepts) == 2
    assert concepts[0].name == "Bayesian Inference"
    assert concepts[0].source_community == 5
    assert concepts[1].name == "Pattern Recognition"


def test_parse_concepts_no_json():
    response = "I couldn't identify any concepts."
    concepts = parse_concepts(response, community_id=0)
    assert concepts == []


def test_parse_concepts_malformed_json():
    response = "[{broken json"
    concepts = parse_concepts(response, community_id=0)
    assert concepts == []


def test_parse_concepts_missing_fields():
    response = '[{"name": "", "description": "valid"}, {"name": "valid", "description": ""}]'
    concepts = parse_concepts(response, community_id=0)
    assert len(concepts) == 0  # Both have empty required fields


def test_build_extraction_prompt():
    sessions = [
        {"title": "Session 1", "content": "Some content about ML"},
        {"title": "Session 2", "content": "More content about statistics"},
    ]
    prompt = build_extraction_prompt(sessions, community_id=3, max_concepts=5)
    assert "2 AI conversations" in prompt
    assert "Community 3" in prompt
    assert "Session 1" in prompt
    assert "Session 2" in prompt
    assert "2-5" in prompt


def test_concept_id_uniqueness():
    """Different names produce different IDs; same name+community produces same ID."""
    r1 = '[{"name": "Concept A", "description": "desc", "example_episodes": []}]'
    r2 = '[{"name": "Concept B", "description": "desc", "example_episodes": []}]'

    c1 = parse_concepts(r1, community_id=0)
    c2 = parse_concepts(r2, community_id=0)
    c1_again = parse_concepts(r1, community_id=0)

    assert c1[0].id != c2[0].id  # Different names → different IDs
    assert c1[0].id == c1_again[0].id  # Same name+community → same ID
