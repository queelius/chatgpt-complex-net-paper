"""Tests for delegation network construction and analysis."""
import networkx as nx
from agentic.extract import Session
from agentic.delegation import (
    build_delegation_graph,
    compute_fan_out_distribution,
    compute_delegation_ratio,
    agent_type_counts,
    classify_agent_type,
)


def _make_family():
    """Parent with 3 subagents."""
    parent = Session(
        id="p1", title="Build feature", source="claude_code",
        model="claude-opus-4-6", message_count=100,
        created_at="2025-12-01T10:00:00", updated_at="2025-12-01T12:00:00",
        metadata={}, parent_conversation_id=None, messages=[],
    )
    children = [
        Session(
            id=f"p1:agent-a{i}", title="subagent", source="claude_code",
            model="claude-haiku-4-5-20251001", message_count=20 + i * 10,
            created_at=f"2025-12-01T10:{i}0:00", updated_at=f"2025-12-01T10:{i}5:00",
            metadata={"agent_id": f"agent-a{i}"}, parent_conversation_id="p1",
            messages=[],
        )
        for i in range(3)
    ]
    return [parent] + children


def test_build_delegation_graph():
    sessions = _make_family()
    G = build_delegation_graph(sessions)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3
    assert G.is_directed()
    for u, v in G.edges():
        assert u == "p1"


def test_fan_out_distribution():
    sessions = _make_family()
    G = build_delegation_graph(sessions)
    dist = compute_fan_out_distribution(G)
    assert dist["p1"] == 3


def test_delegation_ratio():
    sessions = _make_family()
    G = build_delegation_graph(sessions)
    ratios = compute_delegation_ratio(G)
    # Children have 20+30+40=90 messages, parent has 100
    assert ratios["p1"] == 0.9


def test_agent_type_counts():
    sessions = _make_family()
    G = build_delegation_graph(sessions)
    counts = agent_type_counts(G)
    assert counts["user_spawned"] == 3


def test_classify_agent_type():
    assert classify_agent_type("agent-acompact-abc123") == "compact"
    assert classify_agent_type("agent-aprompt_suggestion-xyz") == "prompt_suggestion"
    assert classify_agent_type("agent-a1b2c3d") == "user_spawned"


def test_empty_sessions():
    G = build_delegation_graph([])
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_orphan_child_no_edge():
    """Child whose parent isn't in the session list gets no edge."""
    child = Session(
        id="p99:agent-a1", title="orphan", source="claude_code",
        model="claude-opus-4-6", message_count=10,
        created_at="2025-12-01T10:00:00", updated_at="2025-12-01T10:05:00",
        metadata={"agent_id": "agent-a1"}, parent_conversation_id="p99",
        messages=[],
    )
    G = build_delegation_graph([child])
    assert G.number_of_nodes() == 1
    assert G.number_of_edges() == 0
