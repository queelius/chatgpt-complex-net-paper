# Agentic Cognitive MRI — Analysis Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the analysis pipeline to characterize agentic AI session networks, directly comparing with published ChatGPT conversational results.

**Architecture:** Extract sessions from memex DB → preprocess content → generate embeddings (Ollama) → compute edges → build semantic/delegation/temporal networks → compute metrics → generate figures. Extends the existing `code/` pipeline; new code lives in `agentic/` subdirectory within the papers repo.

**Tech Stack:** Python 3.12, networkx, numpy, scipy, scikit-learn, python-louvain, matplotlib, seaborn, powerlaw (Clauset et al.), sqlite3, Ollama (nomic-embed-text + code models)

---

### Task 1: Scaffold the agentic analysis directory

**Files:**
- Create: `agentic/extract.py` (data extraction from memex)
- Create: `agentic/preprocess.py` (content preprocessing variants)
- Create: `agentic/requirements.txt`
- Create: `agentic/README.md`
- Create: `agentic/tests/test_extract.py`
- Create: `agentic/tests/test_preprocess.py`

**Step 1: Create directory structure**

```bash
mkdir -p agentic/tests agentic/data agentic/output
touch agentic/__init__.py agentic/tests/__init__.py
```

**Step 2: Write requirements.txt**

```
networkx>=3.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
python-louvain==0.16
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
powerlaw>=1.5
tqdm>=4.65.0
python-dotenv>=1.0.0
requests>=2.30.0
```

**Step 3: Commit**

```bash
git add agentic/
git commit -m "scaffold: agentic analysis directory structure"
```

---

### Task 2: Data extraction — export memex sessions to JSON

**Files:**
- Create: `agentic/extract.py`
- Test: `agentic/tests/test_extract.py`

**Step 1: Write failing test for extract_sessions()**

```python
# agentic/tests/test_extract.py
"""Tests for session extraction from memex DB."""
import json
import sqlite3
import tempfile
from pathlib import Path
import pytest

# We'll test against a small in-memory DB that mimics memex schema
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
    conn.execute("""
        INSERT INTO messages VALUES
        ('p1', 'm1', 'user', '[{"type":"text","text":"Fix the login bug"}]', NULL)
    """)
    conn.execute("""
        INSERT INTO messages VALUES
        ('p1', 'm2', 'assistant', '[{"type":"text","text":"I will fix it"},{"type":"tool_use","name":"Read","input":{"file_path":"/src/auth.py"}}]', 'm1')
    """)
    conn.execute("""
        INSERT INTO messages VALUES
        ('p1', 'm3', 'user', '[{"type":"tool_result","content":"def login():..."}]', 'm2')
    """)
    conn.execute("""
        INSERT INTO messages VALUES
        ('p1', 'm4', 'assistant', '[{"type":"text","text":"Fixed the null check"}]', 'm3')
    """)
    # Subagent of p1
    conn.execute("""
        INSERT INTO conversations VALUES
        ('p1:agent-a1', 'subagent', 'claude_code', 'claude-haiku-4-5-20251001', 2,
         '2025-12-01T10:30:00', '2025-12-01T10:35:00',
         '{"agent_id":"agent-a1"}', 'p1')
    """)
    conn.execute("""
        INSERT INTO messages VALUES
        ('p1:agent-a1', 'm1', 'assistant', '[{"type":"tool_use","name":"Grep","input":{"pattern":"login"}}]', NULL)
    """)
    conn.execute("""
        INSERT INTO messages VALUES
        ('p1:agent-a1', 'm2', 'user', '[{"type":"tool_result","content":"auth.py:42"}]', 'm1')
    """)
    # Parent 2 (no subagents)
    conn.execute("""
        INSERT INTO conversations VALUES
        ('p2', 'Add tests', 'claude_code', 'claude-opus-4-6', 2,
         '2025-12-02T09:00:00', '2025-12-02T10:00:00', '{}', NULL)
    """)
    conn.execute("""
        INSERT INTO messages VALUES
        ('p2', 'm1', 'user', '[{"type":"text","text":"Write tests for auth"}]', NULL)
    """)
    conn.execute("""
        INSERT INTO messages VALUES
        ('p2', 'm2', 'assistant', '[{"type":"text","text":"Here are the tests"}]', 'm1')
    """)
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
```

**Step 2: Run test to verify it fails**

Run: `cd agentic && python -m pytest tests/test_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic.extract'`

**Step 3: Write minimal extraction implementation**

```python
# agentic/extract.py
"""Extract Claude Code sessions from a memex database into analysis-ready structures."""
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
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
        source: Filter by conversation source.
        include_subagents: If False, exclude conversations with parent_conversation_id.

    Returns:
        List of Session objects with parsed messages.
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
```

**Step 4: Run tests**

Run: `cd agentic && python -m pytest tests/test_extract.py -v`
Expected: PASS (5/5)

**Step 5: Commit**

```bash
git add agentic/extract.py agentic/tests/test_extract.py
git commit -m "feat: session extraction from memex DB"
```

---

### Task 3: Content preprocessing — multiple views of session content

**Files:**
- Create: `agentic/preprocess.py`
- Test: `agentic/tests/test_preprocess.py`

This implements the ablation dimension of content preprocessing: extracting different textual representations from the same session.

**Step 1: Write failing tests**

```python
# agentic/tests/test_preprocess.py
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
    """Create sample messages with mixed content types."""
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
    """Full extraction includes text, tool names, and tool content."""
    msgs = _make_messages()
    text = extract_text_full(msgs)
    assert "Fix the login bug" in text
    assert "Read" in text
    assert "def login" in text
    assert "Edit" in text


def test_extract_text_only():
    """Text-only strips tool_use and tool_result blocks."""
    msgs = _make_messages()
    text = extract_text_only(msgs)
    assert "Fix the login bug" in text
    assert "Fixed the null check" in text
    assert "def login" not in text  # tool_result stripped
    assert "Read" not in text  # tool_use stripped


def test_extract_user_only():
    """User-only extracts text blocks from user role only."""
    msgs = _make_messages()
    text = extract_user_only(msgs)
    assert "Fix the login bug" in text
    assert "Fixed the null check" not in text  # assistant
    assert "def login" not in text  # tool_result


def test_extract_tool_names_only():
    """Tool-names-only extracts just the tool invocation sequence."""
    msgs = _make_messages()
    text = extract_tool_names_only(msgs)
    assert "Read" in text
    assert "Edit" in text
    assert "Fix the login bug" not in text


def test_extract_by_role():
    """By-role returns separate text for each role."""
    msgs = _make_messages()
    by_role = extract_by_role(msgs)
    assert "user" in by_role
    assert "assistant" in by_role
    assert "Fix the login bug" in by_role["user"]
    assert "Fixed the null check" in by_role["assistant"]
```

**Step 2: Run to verify failure**

Run: `cd agentic && python -m pytest tests/test_preprocess.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement preprocessing**

```python
# agentic/preprocess.py
"""Content preprocessing: extract different textual views from session messages.

Each function takes a list of Message objects and returns a string suitable
for embedding generation. The different extractors correspond to ablation
dimensions in the research design.
"""
from typing import Dict, List

from agentic.extract import Message


def _text_blocks(messages: List[Message], roles: set | None = None) -> List[str]:
    """Extract text content blocks, optionally filtered by role."""
    parts = []
    for msg in messages:
        if roles and msg.role not in roles:
            continue
        for block in msg.content:
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
    return parts


def extract_text_full(messages: List[Message]) -> str:
    """Full content: text blocks + tool names + tool_result content.

    Preserves code semantics from tool outputs while including natural language.
    """
    parts = []
    for msg in messages:
        for block in msg.content:
            btype = block.get("type")
            if btype == "text" and block.get("text"):
                parts.append(block["text"])
            elif btype == "tool_use":
                name = block.get("name", "")
                parts.append(f"[Tool: {name}]")
            elif btype == "tool_result":
                content = block.get("content", "")
                if isinstance(content, str) and content.strip():
                    parts.append(content)
            elif btype == "thinking" and block.get("thinking"):
                parts.append(block["thinking"])
    return "\n".join(parts)


def extract_text_only(messages: List[Message]) -> str:
    """Text blocks only — strips tool_use, tool_result, thinking.

    Comparable to ChatGPT methodology (natural language only).
    """
    return "\n".join(_text_blocks(messages))


def extract_user_only(messages: List[Message]) -> str:
    """Text blocks from user role only.

    Maximum comparability with ChatGPT user-weighted approach.
    Note: in subagent sessions, 'user' role contains tool_result blocks
    from the parent agent, not human input.
    """
    return "\n".join(_text_blocks(messages, roles={"user"}))


def extract_tool_names_only(messages: List[Message]) -> str:
    """Tool invocation names only — behavioral signature without content.

    Captures the *strategy* of the session (what tools were used in what order)
    rather than the semantic content.
    """
    names = []
    for msg in messages:
        for block in msg.content:
            if block.get("type") == "tool_use" and block.get("name"):
                names.append(block["name"])
    return " ".join(names)


def extract_by_role(messages: List[Message]) -> Dict[str, str]:
    """Separate text per role — for role-weighted embedding generation.

    Returns dict with keys 'user' and 'assistant' (and any other roles found).
    """
    by_role: Dict[str, List[str]] = {}
    for msg in messages:
        for block in msg.content:
            if block.get("type") == "text" and block.get("text"):
                by_role.setdefault(msg.role, []).append(block["text"])
    return {role: "\n".join(texts) for role, texts in by_role.items()}
```

**Step 4: Run tests**

Run: `cd agentic && python -m pytest tests/test_preprocess.py -v`
Expected: PASS (5/5)

**Step 5: Commit**

```bash
git add agentic/preprocess.py agentic/tests/test_preprocess.py
git commit -m "feat: content preprocessing variants for embedding ablation"
```

---

### Task 4: Export sessions to pipeline-compatible JSON format

**Files:**
- Create: `agentic/export_json.py`
- Test: `agentic/tests/test_export_json.py`

The existing pipeline (`code/cli.py`) expects a directory of JSON files, each with a `messages` array. This task creates an exporter that writes sessions in that format, with the preprocessing variant baked in.

**Step 1: Write failing test**

```python
# agentic/tests/test_export_json.py
"""Tests for exporting sessions to pipeline JSON format."""
import json
import tempfile
from pathlib import Path

from agentic.extract import Session, Message
from agentic.export_json import export_sessions_to_json


def _make_session():
    return Session(
        id="p1",
        title="Fix auth",
        source="claude_code",
        model="claude-opus-4-6",
        message_count=2,
        created_at="2025-12-01T10:00:00",
        updated_at="2025-12-01T11:00:00",
        metadata={},
        parent_conversation_id=None,
        messages=[
            Message(id="m1", role="user", content=[
                {"type": "text", "text": "Fix login"}
            ]),
            Message(id="m2", role="assistant", content=[
                {"type": "text", "text": "Done"}
            ]),
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
```

**Step 2: Run to verify failure**

Run: `cd agentic && python -m pytest tests/test_export_json.py -v`

**Step 3: Implement exporter**

```python
# agentic/export_json.py
"""Export sessions to JSON format compatible with the existing analysis pipeline.

The pipeline (code/cli.py) expects:
- Directory of .json files (one per conversation)
- Each file has {"messages": [{"role": "...", "content": "..."}], ...}
- Optional metadata fields for node attributes
"""
import json
from pathlib import Path
from typing import List

from agentic.extract import Session
from agentic.preprocess import extract_text_only


def export_sessions_to_json(
    sessions: List[Session],
    output_dir: str,
    content_extractor=None,
) -> None:
    """Export sessions as individual JSON files for the embedding pipeline.

    Args:
        sessions: List of Session objects.
        output_dir: Directory to write JSON files.
        content_extractor: Function(List[Message]) -> str. Defaults to extract_text_only.
    """
    if content_extractor is None:
        content_extractor = extract_text_only

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for session in sessions:
        # Build simplified messages for embedding
        # The pipeline expects {"role": ..., "content": <string>}
        messages = []
        for msg in session.messages:
            text_parts = []
            for block in msg.content:
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])
            if text_parts:
                messages.append({
                    "role": msg.role,
                    "content": "\n".join(text_parts),
                })

        doc = {
            "messages": messages,
            "metadata": {
                "id": session.id,
                "title": session.title,
                "source": session.source,
                "model": session.model,
                "message_count": session.message_count,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "parent_conversation_id": session.parent_conversation_id,
                **session.metadata,
            },
        }

        # Use session ID as filename (sanitize colons for filesystem)
        filename = session.id.replace(":", "_") + ".json"
        with open(out / filename, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
```

**Step 4: Run tests**

Run: `cd agentic && python -m pytest tests/test_export_json.py -v`
Expected: PASS (3/3)

**Step 5: Commit**

```bash
git add agentic/export_json.py agentic/tests/test_export_json.py
git commit -m "feat: export sessions to pipeline-compatible JSON"
```

---

### Task 5: Delegation network analysis

**Files:**
- Create: `agentic/delegation.py`
- Test: `agentic/tests/test_delegation.py`

**Step 1: Write failing tests**

```python
# agentic/tests/test_delegation.py
"""Tests for delegation network construction and analysis."""
import networkx as nx
from agentic.extract import Session
from agentic.delegation import (
    build_delegation_graph,
    compute_fan_out_distribution,
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
            model="claude-haiku-4-5-20251001", message_count=20+i*10,
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
    # All edges point from parent to child
    for u, v in G.edges():
        assert u == "p1"


def test_fan_out_distribution():
    sessions = _make_family()
    G = build_delegation_graph(sessions)
    dist = compute_fan_out_distribution(G)
    assert dist["p1"] == 3


def test_classify_agent_type():
    assert classify_agent_type("agent-acompact-abc123") == "compact"
    assert classify_agent_type("agent-aprompt_suggestion-xyz") == "prompt_suggestion"
    assert classify_agent_type("agent-a1b2c3d") == "user_spawned"
```

**Step 2: Run to verify failure**

Run: `cd agentic && python -m pytest tests/test_delegation.py -v`

**Step 3: Implement delegation analysis**

```python
# agentic/delegation.py
"""Delegation network construction and analysis.

Builds a directed graph from parent→child session relationships
and computes delegation-specific metrics.
"""
import re
from collections import Counter
from typing import Dict, List

import networkx as nx

from agentic.extract import Session


def classify_agent_type(agent_id: str) -> str:
    """Classify a subagent by its ID prefix."""
    if "acompact" in agent_id:
        return "compact"
    if "aprompt_suggestion" in agent_id:
        return "prompt_suggestion"
    return "user_spawned"


def build_delegation_graph(sessions: List[Session]) -> nx.DiGraph:
    """Build a directed delegation graph from sessions.

    Nodes are session IDs. Edges point from parent to child.
    Node attributes include session metadata for downstream analysis.
    """
    G = nx.DiGraph()

    for s in sessions:
        G.add_node(
            s.id,
            title=s.title,
            model=s.model,
            message_count=s.message_count,
            created_at=s.created_at,
            is_parent=s.parent_conversation_id is None,
            agent_type=(
                classify_agent_type(s.metadata.get("agent_id", ""))
                if s.parent_conversation_id
                else "parent"
            ),
        )

    for s in sessions:
        if s.parent_conversation_id and s.parent_conversation_id in G:
            G.add_edge(s.parent_conversation_id, s.id)

    return G


def compute_fan_out_distribution(G: nx.DiGraph) -> Dict[str, int]:
    """Compute fan-out (out-degree) for each parent node."""
    return {
        node: G.out_degree(node)
        for node in G.nodes()
        if G.nodes[node].get("is_parent", False)
    }


def compute_delegation_ratio(G: nx.DiGraph) -> Dict[str, float]:
    """Compute delegation ratio: total child messages / parent messages."""
    ratios = {}
    for node in G.nodes():
        if not G.nodes[node].get("is_parent"):
            continue
        parent_msgs = G.nodes[node].get("message_count", 0)
        if parent_msgs == 0:
            continue
        child_msgs = sum(
            G.nodes[child].get("message_count", 0)
            for child in G.successors(node)
        )
        ratios[node] = child_msgs / parent_msgs
    return ratios


def agent_type_counts(G: nx.DiGraph) -> Counter:
    """Count subagents by type across the entire graph."""
    return Counter(
        G.nodes[n].get("agent_type", "unknown")
        for n in G.nodes()
        if not G.nodes[n].get("is_parent")
    )
```

**Step 4: Run tests**

Run: `cd agentic && python -m pytest tests/test_delegation.py -v`
Expected: PASS (3/3)

**Step 5: Commit**

```bash
git add agentic/delegation.py agentic/tests/test_delegation.py
git commit -m "feat: delegation network construction and metrics"
```

---

### Task 6: Semantic network analysis — metrics computation

**Files:**
- Create: `agentic/semantic.py`
- Test: `agentic/tests/test_semantic.py`

Wraps networkx metric computation into a structured output compatible with temporal analysis.

**Step 1: Write failing test**

```python
# agentic/tests/test_semantic.py
"""Tests for semantic network metrics computation."""
import networkx as nx
from agentic.semantic import compute_network_metrics


def test_compute_metrics_basic():
    G = nx.Graph()
    G.add_edges_from([("a", "b", {"weight": 0.95}),
                      ("b", "c", {"weight": 0.92}),
                      ("a", "c", {"weight": 0.91})])
    metrics = compute_network_metrics(G)
    assert metrics["node_count"] == 3
    assert metrics["edge_count"] == 3
    assert metrics["density"] > 0
    assert "modularity" in metrics
    assert "avg_clustering" in metrics
    assert "giant_component_size" in metrics


def test_empty_graph():
    G = nx.Graph()
    metrics = compute_network_metrics(G)
    assert metrics["node_count"] == 0
    assert metrics["edge_count"] == 0
```

**Step 2: Implement**

```python
# agentic/semantic.py
"""Semantic network metrics computation.

Computes global, node-level, and community metrics for a similarity network.
Designed to produce output comparable to the ChatGPT temporal analysis.
"""
from typing import Any, Dict

import networkx as nx
import numpy as np

try:
    import community.community_louvain as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False


def compute_network_metrics(
    G: nx.Graph, random_state: int = 42
) -> Dict[str, Any]:
    """Compute comprehensive network metrics.

    Returns a flat dict of scalar metrics, compatible with CSV/DataFrame output.
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()

    if n == 0:
        return {"node_count": 0, "edge_count": 0, "density": 0.0}

    degrees = [d for _, d in G.degree()]

    # Giant component
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    gc = components[0] if components else set()
    gc_size = len(gc)

    metrics = {
        "node_count": n,
        "edge_count": e,
        "density": nx.density(G),
        "num_components": len(components),
        "giant_component_size": gc_size,
        "giant_component_fraction": gc_size / n if n else 0.0,
        "avg_degree": np.mean(degrees) if degrees else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
    }

    # Shortest paths (giant component only)
    if gc_size > 1:
        gc_sub = G.subgraph(gc)
        try:
            metrics["avg_shortest_path"] = nx.average_shortest_path_length(gc_sub)
        except nx.NetworkXError:
            metrics["avg_shortest_path"] = None
    else:
        metrics["avg_shortest_path"] = None

    # Community detection
    if HAS_COMMUNITY and n > 0:
        partition = community_louvain.best_partition(G, random_state=random_state)
        metrics["modularity"] = community_louvain.modularity(partition, G)
        metrics["num_communities"] = len(set(partition.values()))
    else:
        metrics["modularity"] = 0.0
        metrics["num_communities"] = 0

    # Assortativity
    try:
        metrics["assortativity"] = nx.degree_assortativity_coefficient(G)
    except (nx.NetworkXError, ValueError):
        metrics["assortativity"] = None

    return metrics
```

**Step 3: Run tests, commit**

Run: `cd agentic && python -m pytest tests/test_semantic.py -v`

```bash
git add agentic/semantic.py agentic/tests/test_semantic.py
git commit -m "feat: semantic network metrics computation"
```

---

### Task 7: Temporal analysis — cumulative snapshots

**Files:**
- Create: `agentic/temporal.py`
- Test: `agentic/tests/test_temporal.py`

Adapts the existing `temporal_snapshots.py` for agentic data with daily resolution and delegation tracking.

**Step 1–5:** Follow same TDD pattern. Key functions:
- `build_daily_snapshots(sessions, edges, random_state)` → list of snapshot metrics dicts
- `fit_densification_law(snapshots)` → (gamma, r_squared, p_value)
- `compute_preferential_attachment(snapshots, edges)` → (beta, r_squared, z_scores)

This reuses the logic from `comp-net-2025-journal/code/temporal_snapshots.py` but operates on daily resolution and adds delegation columns (mean_fan_out, mean_delegation_ratio per snapshot).

**Commit message:** `feat: temporal snapshot builder with daily resolution`

---

### Task 8: Multi-layer analysis

**Files:**
- Create: `agentic/multilayer.py`
- Test: `agentic/tests/test_multilayer.py`

**Key functions:**
- `compute_interlayer_degree_correlation(semantic_G, delegation_G)` → Spearman correlation
- `compute_participation_coefficient(semantic_G, delegation_G)` → dict of per-node P values
- `compare_community_assignments(semantic_partition, delegation_partition)` → NMI score

**Commit message:** `feat: two-layer multiplex analysis (semantic + delegation)`

---

### Task 9: CLI runner — orchestrate the full pipeline

**Files:**
- Create: `agentic/run.py`
- No test (integration script)

CLI that chains: extract → preprocess → export JSON → (external: embeddings via Ollama) → load edges → build networks → compute metrics → save results.

```bash
# Full pipeline
python -m agentic.run \
    --db ~/.memex/default/conversations.db \
    --output-dir agentic/output/run-001 \
    --content-mode text-only \
    --include-subagents

# Just extraction + export (for manual embedding generation)
python -m agentic.run --extract-only \
    --db ~/.memex/default/conversations.db \
    --output-dir agentic/output/sessions-text-only
```

**Commit message:** `feat: CLI runner for agentic analysis pipeline`

---

### Task 10: Run initial experiment and verify

**Step 1:** Extract sessions from memex DB
```bash
python -m agentic.run --extract-only \
    --db ~/.memex/default/conversations.db \
    --output-dir agentic/data/sessions-text-only \
    --content-mode text-only
```

**Step 2:** Generate embeddings via Ollama (reuse existing pipeline)
```bash
cd ../code
python cli.py node-embeddings \
    -i ../agentic/data/sessions-text-only \
    -m role-aggregate \
    -u 2.0 -a 1.0 \
    -e llm
```

**Step 3:** Generate edges
```bash
python cli.py edges \
    -i ../agentic/data/sessions-text-only \
    -o ../agentic/data/edges-all.json
```

**Step 4:** Filter at threshold 0.9
```bash
python cli.py cut-off \
    -i ../agentic/data/edges-all.json \
    -o ../agentic/data/edges-t0.9.json \
    -c 0.9
```

**Step 5:** Verify output
```bash
python -c "
import json
with open('agentic/data/edges-t0.9.json') as f:
    edges = json.load(f)
nodes = set()
for s, d, w in edges:
    nodes.add(s)
    nodes.add(d)
print(f'Nodes: {len(nodes)}, Edges: {len(edges)}')
"
```

**Step 6:** Run analysis
```bash
python -m agentic.run --analyze-only \
    --edges-file agentic/data/edges-t0.9.json \
    --sessions-dir agentic/data/sessions-text-only \
    --output-dir agentic/output/analysis-001
```

**Step 7:** Commit results
```bash
git add agentic/output/analysis-001/
git commit -m "data: initial agentic network analysis results"
```

---

## Execution Order and Dependencies

```
Task 1 (scaffold) ─→ Task 2 (extract) ─→ Task 3 (preprocess) ─→ Task 4 (export)
                                                                       ↓
Task 5 (delegation) ────────────────────────────────────────→ Task 9 (CLI runner)
Task 6 (semantic) ──────────────────────────────────────────→ Task 9
Task 7 (temporal) ──────────────────────────────────────────→ Task 9
Task 8 (multilayer) ────────────────────────────────────────→ Task 9
                                                                       ↓
                                                              Task 10 (run)
```

Tasks 5, 6, 7, 8 are independent and can be implemented in parallel.
