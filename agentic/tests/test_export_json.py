"""Tests for exporting sessions to pipeline JSON format."""
import json
from agentic.extract import Session, Message
from agentic.export_json import export_sessions_to_json


def _make_session():
    return Session(
        id="p1", title="Fix auth", source="claude_code",
        model="claude-opus-4-6", message_count=2,
        created_at="2025-12-01T10:00:00", updated_at="2025-12-01T11:00:00",
        metadata={}, parent_conversation_id=None,
        messages=[
            Message(id="m1", role="user", content=[{"type": "text", "text": "Fix login"}]),
            Message(id="m2", role="assistant", content=[{"type": "text", "text": "Done"}]),
        ],
    )


def test_export_creates_json_files(tmp_path):
    sessions = [_make_session()]
    export_sessions_to_json(sessions, str(tmp_path))
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].name == "p1.json"


def test_exported_json_has_required_fields(tmp_path):
    sessions = [_make_session()]
    export_sessions_to_json(sessions, str(tmp_path))
    with open(tmp_path / "p1.json") as f:
        data = json.load(f)
    assert "messages" in data
    assert "metadata" in data
    assert data["metadata"]["id"] == "p1"
    assert data["metadata"]["created_at"] == "2025-12-01T10:00:00"
    assert data["metadata"]["model"] == "claude-opus-4-6"


def test_exported_messages_have_role_and_content(tmp_path):
    sessions = [_make_session()]
    export_sessions_to_json(sessions, str(tmp_path))
    with open(tmp_path / "p1.json") as f:
        data = json.load(f)
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "Fix login"


def test_colon_in_id_sanitized_to_underscore(tmp_path):
    """Subagent IDs use colons (sessionId:agentId) — must be filesystem-safe."""
    session = Session(
        id="abc-123:agent-a1", title="subagent", source="claude_code",
        model="claude-haiku-4-5-20251001", message_count=1,
        created_at="2025-12-01T10:00:00", updated_at="2025-12-01T11:00:00",
        metadata={}, parent_conversation_id="abc-123",
        messages=[
            Message(id="m1", role="user", content=[{"type": "text", "text": "Hello"}]),
        ],
    )
    export_sessions_to_json([session], str(tmp_path))
    # Filename should use underscore, not colon
    assert (tmp_path / "abc-123_agent-a1.json").exists()
    assert not (tmp_path / "abc-123:agent-a1.json").exists()
    # The stored metadata.id should preserve the original colon
    with open(tmp_path / "abc-123_agent-a1.json") as f:
        data = json.load(f)
    assert data["metadata"]["id"] == "abc-123:agent-a1"


def test_id_mapping_roundtrip():
    """Verify the safe_to_id mapping in run.py correctly reverses the sanitization."""
    original_ids = {"abc-123", "abc-123:agent-a1", "def-456:agent-acompact-xyz"}
    # This is the mapping logic from run.py do_analyze()
    safe_to_id = {sid.replace(":", "_"): sid for sid in original_ids}
    # Simulate edge file IDs (underscores)
    edge_ids = ["abc-123_agent-a1", "def-456_agent-acompact-xyz", "abc-123"]
    for edge_id in edge_ids:
        restored = safe_to_id.get(edge_id, edge_id)
        assert restored in original_ids, f"{edge_id} -> {restored} not in original IDs"
