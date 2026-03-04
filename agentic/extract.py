"""Extract Claude Code sessions from a memex database into analysis-ready structures."""
import json
import sqlite3
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    id: str
    role: str
    content: List[dict]  # parsed content blocks
    parent_id: Optional[str] = None


@dataclass
class Session:
    id: str
    title: str
    source: str
    model: Optional[str]
    message_count: int
    created_at: str
    updated_at: str
    metadata: dict
    parent_conversation_id: Optional[str]
    messages: List[Message] = field(default_factory=list)


def extract_sessions(
    db_path: str,
    source: str = "claude_code",
    include_subagents: bool = True,
) -> List[Session]:
    """Extract sessions from a memex database.

    Args:
        db_path: Path to the memex SQLite database.
        source: Filter conversations by source (default: "claude_code").
        include_subagents: If False, exclude conversations with a parent_conversation_id.

    Returns:
        List of Session objects with their messages populated.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    where = "WHERE c.source = ?"
    params: list = [source]
    if not include_subagents:
        where += " AND c.parent_conversation_id IS NULL"

    rows = conn.execute(
        f"""
        SELECT c.id, c.title, c.source, c.model, c.message_count,
               c.created_at, c.updated_at, c.metadata, c.parent_conversation_id
        FROM conversations c
        {where}
        ORDER BY c.created_at
        """,
        params,
    ).fetchall()

    sessions = []
    for r in rows:
        meta = json.loads(r["metadata"]) if r["metadata"] else {}
        session = Session(
            id=r["id"],
            title=r["title"] or "",
            source=r["source"],
            model=r["model"],
            message_count=r["message_count"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            metadata=meta,
            parent_conversation_id=r["parent_conversation_id"],
        )

        msg_rows = conn.execute(
            """
            SELECT id, role, content, parent_id
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (r["id"],),
        ).fetchall()

        for m in msg_rows:
            content = json.loads(m["content"]) if m["content"] else []
            session.messages.append(
                Message(
                    id=m["id"],
                    role=m["role"],
                    content=content,
                    parent_id=m["parent_id"],
                )
            )
        sessions.append(session)

    conn.close()
    return sessions
