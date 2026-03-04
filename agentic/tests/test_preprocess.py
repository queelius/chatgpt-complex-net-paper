"""Tests for content preprocessing variants."""
from agentic.extract import Message
from agentic.preprocess import (
    extract_text_full,
    extract_text_only,
    extract_user_only,
    extract_tool_names_only,
    extract_by_role,
)


def _make_messages():
    return [
        Message(id="m1", role="user", content=[
            {"type": "text", "text": "Fix the login bug in auth.py"}
        ]),
        Message(id="m2", role="assistant", content=[
            {"type": "text", "text": "I'll read the file first"},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/src/auth.py"}},
        ]),
        Message(id="m3", role="user", content=[
            {"type": "tool_result", "content": "def login():\n    return None"},
        ]),
        Message(id="m4", role="assistant", content=[
            {"type": "thinking", "thinking": "The bug is a null check"},
            {"type": "text", "text": "Fixed the null check"},
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "/src/auth.py", "old_string": "return None", "new_string": "return user"}},
        ]),
    ]


def test_extract_text_full():
    msgs = _make_messages()
    text = extract_text_full(msgs)
    assert "Fix the login bug" in text
    assert "Read" in text
    assert "def login" in text
    assert "Edit" in text


def test_extract_text_only():
    msgs = _make_messages()
    text = extract_text_only(msgs)
    assert "Fix the login bug" in text
    assert "Fixed the null check" in text
    assert "def login" not in text
    assert "Read" not in text


def test_extract_user_only():
    msgs = _make_messages()
    text = extract_user_only(msgs)
    assert "Fix the login bug" in text
    assert "Fixed the null check" not in text
    assert "def login" not in text


def test_extract_tool_names_only():
    msgs = _make_messages()
    text = extract_tool_names_only(msgs)
    assert "Read" in text
    assert "Edit" in text
    assert "Fix the login bug" not in text


def test_extract_by_role():
    msgs = _make_messages()
    by_role = extract_by_role(msgs)
    assert "user" in by_role
    assert "assistant" in by_role
    assert "Fix the login bug" in by_role["user"]
    assert "Fixed the null check" in by_role["assistant"]
