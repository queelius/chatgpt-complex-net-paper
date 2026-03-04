"""Tests for temporal analysis: daily snapshots and densification."""
from agentic.extract import Session
from agentic.temporal import (
    build_daily_snapshots,
    fit_densification_law,
    compute_preferential_attachment,
)


def _make_temporal_sessions():
    """Sessions spread over 3 days."""
    sessions = []
    # Day 1: 2 parents
    for i in range(2):
        sessions.append(Session(
            id=f"d1-p{i}", title=f"Day 1 session {i}", source="claude_code",
            model="claude-opus-4-6", message_count=50,
            created_at=f"2025-12-01T{10+i}:00:00",
            updated_at=f"2025-12-01T{11+i}:00:00",
            metadata={}, parent_conversation_id=None, messages=[],
        ))
    # Day 2: 3 more parents
    for i in range(3):
        sessions.append(Session(
            id=f"d2-p{i}", title=f"Day 2 session {i}", source="claude_code",
            model="claude-opus-4-6", message_count=50,
            created_at=f"2025-12-02T{10+i}:00:00",
            updated_at=f"2025-12-02T{11+i}:00:00",
            metadata={}, parent_conversation_id=None, messages=[],
        ))
    # Day 3: 2 parents + 1 subagent
    for i in range(2):
        sessions.append(Session(
            id=f"d3-p{i}", title=f"Day 3 session {i}", source="claude_code",
            model="claude-opus-4-6", message_count=50,
            created_at=f"2025-12-03T{10+i}:00:00",
            updated_at=f"2025-12-03T{11+i}:00:00",
            metadata={}, parent_conversation_id=None, messages=[],
        ))
    sessions.append(Session(
        id="d3-p0:agent-a1", title="subagent", source="claude_code",
        model="claude-haiku-4-5-20251001", message_count=20,
        created_at="2025-12-03T10:30:00", updated_at="2025-12-03T10:35:00",
        metadata={"agent_id": "agent-a1"}, parent_conversation_id="d3-p0",
        messages=[],
    ))
    return sessions


def _make_edges(sessions):
    """Create some edges between sessions."""
    ids = [s.id for s in sessions]
    edges = []
    # Connect sequential pairs
    for i in range(len(ids) - 1):
        edges.append((ids[i], ids[i + 1], 0.92))
    # Some cross-connections
    if len(ids) > 3:
        edges.append((ids[0], ids[3], 0.91))
    return edges


def test_build_daily_snapshots():
    sessions = _make_temporal_sessions()
    edges = _make_edges(sessions)
    snapshots = build_daily_snapshots(sessions, edges)
    assert len(snapshots) == 3
    # Cumulative: day 1 has 2 nodes, day 2 has 5, day 3 has 8
    assert snapshots[0]["node_count"] == 2
    assert snapshots[1]["node_count"] == 5
    assert snapshots[2]["node_count"] == 8


def test_snapshots_have_delegation_metrics():
    sessions = _make_temporal_sessions()
    edges = _make_edges(sessions)
    snapshots = build_daily_snapshots(sessions, edges)
    for snap in snapshots:
        assert "num_parents" in snap
        assert "num_children" in snap
        assert "mean_fan_out" in snap
    # Day 3 has 1 child
    assert snapshots[2]["num_children"] == 1


def test_snapshots_cumulative_edges():
    sessions = _make_temporal_sessions()
    edges = _make_edges(sessions)
    snapshots = build_daily_snapshots(sessions, edges)
    # Edge count should be non-decreasing
    for i in range(1, len(snapshots)):
        assert snapshots[i]["edge_count"] >= snapshots[i - 1]["edge_count"]


def test_fit_densification_law():
    sessions = _make_temporal_sessions()
    edges = _make_edges(sessions)
    snapshots = build_daily_snapshots(sessions, edges)
    result = fit_densification_law(snapshots)
    assert "gamma" in result
    # With 3 data points, we should get a fit
    assert result["gamma"] is not None
    assert result["r_squared"] is not None


def test_fit_densification_insufficient_data():
    result = fit_densification_law([])
    assert result["gamma"] is None


def test_preferential_attachment_insufficient_data():
    result = compute_preferential_attachment([], [], [])
    assert result["beta"] is None
