"""Tests for session extraction from memex DB."""
import json
import sqlite3
import tempfile
from pathlib import Path
import pytest

from agentic.extract import extract_sessions, Session


@pytest.fixture
def sample_db(tmp_path):
    """Create a minimal memex-like DB with 2 parents and 1 subagent."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            model TEXT,
            message_count INTEGER,
            created_at TEXT,
            updated_at TEXT,
            metadata TEXT DEFAULT '{}',
            parent_conversation_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE messages (
            conversation_id TEXT,
            id TEXT,
            role TEXT,
            content TEXT,
            parent_id TEXT,
            PRIMARY KEY (conversation_id, id)
        )
    """)
    # Parent 1
    conn.execute("""
        INSERT INTO conversations VALUES
        ('p1', 'Fix auth bug', 'claude_code', 'claude-opus-4-6', 4,
         '2025-12-01T10:00:00', '2025-12-01T11:00:00', '{}', NULL)
    """)
    conn.execute("INSERT INTO messages VALUES ('p1', 'm1', 'user', '[{\"type\":\"text\",\"text\":\"Fix the login bug\"}]', NULL)")
    conn.execute("INSERT INTO messages VALUES ('p1', 'm2', 'assistant', '[{\"type\":\"text\",\"text\":\"I will fix it\"},{\"type\":\"tool_use\",\"name\":\"Read\",\"input\":{\"file_path\":\"/src/auth.py\"}}]', 'm1')")
    conn.execute("INSERT INTO messages VALUES ('p1', 'm3', 'user', '[{\"type\":\"tool_result\",\"content\":\"def login():...\"}]', 'm2')")
    conn.execute("INSERT INTO messages VALUES ('p1', 'm4', 'assistant', '[{\"type\":\"text\",\"text\":\"Fixed the null check\"}]', 'm3')")
    # Subagent of p1
    conn.execute("""
        INSERT INTO conversations VALUES
        ('p1:agent-a1', 'subagent', 'claude_code', 'claude-haiku-4-5-20251001', 2,
         '2025-12-01T10:30:00', '2025-12-01T10:35:00',
         '{"agent_id":"agent-a1"}', 'p1')
    """)
    conn.execute("INSERT INTO messages VALUES ('p1:agent-a1', 'm1', 'assistant', '[{\"type\":\"tool_use\",\"name\":\"Grep\",\"input\":{\"pattern\":\"login\"}}]', NULL)")
    conn.execute("INSERT INTO messages VALUES ('p1:agent-a1', 'm2', 'user', '[{\"type\":\"tool_result\",\"content\":\"auth.py:42\"}]', 'm1')")
    # Parent 2 (no subagents)
    conn.execute("""
        INSERT INTO conversations VALUES
        ('p2', 'Add tests', 'claude_code', 'claude-opus-4-6', 2,
         '2025-12-02T09:00:00', '2025-12-02T10:00:00', '{}', NULL)
    """)
    conn.execute("INSERT INTO messages VALUES ('p2', 'm1', 'user', '[{\"type\":\"text\",\"text\":\"Write tests for auth\"}]', NULL)")
    conn.execute("INSERT INTO messages VALUES ('p2', 'm2', 'assistant', '[{\"type\":\"text\",\"text\":\"Here are the tests\"}]', 'm1')")
    conn.commit()
    conn.close()
    return db_path


def test_extract_all_sessions(sample_db):
    sessions = extract_sessions(str(sample_db))
    assert len(sessions) == 3  # 2 parents + 1 subagent


def test_extract_parents_only(sample_db):
    sessions = extract_sessions(str(sample_db), include_subagents=False)
    assert len(sessions) == 2
    assert all(s.parent_conversation_id is None for s in sessions)


def test_session_has_messages(sample_db):
    sessions = extract_sessions(str(sample_db))
    p1 = next(s for s in sessions if s.id == 'p1')
    assert len(p1.messages) == 4


def test_session_metadata(sample_db):
    sessions = extract_sessions(str(sample_db))
    p1 = next(s for s in sessions if s.id == 'p1')
    assert p1.model == 'claude-opus-4-6'
    assert p1.source == 'claude_code'
    assert p1.title == 'Fix auth bug'


def test_subagent_has_parent_id(sample_db):
    sessions = extract_sessions(str(sample_db))
    sub = next(s for s in sessions if s.id == 'p1:agent-a1')
    assert sub.parent_conversation_id == 'p1'
