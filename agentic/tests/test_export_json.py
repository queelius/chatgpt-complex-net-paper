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
