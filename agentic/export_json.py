"""Export sessions to JSON format compatible with the existing analysis pipeline.

The pipeline (code/cli.py) expects:
- Directory of .json files (one per conversation)
- Each file has {"messages": [{"role": "...", "content": "..."}], ...}
- Optional metadata fields for node attributes

Content extraction is always text-only (strips tool_use/tool_result blocks),
preserving per-message role+content structure for the embedding pipeline.
"""
import json
from pathlib import Path
from typing import List

from agentic.extract import Session


def export_sessions_to_json(
    sessions: List[Session],
    output_dir: str,
) -> None:
    """Export sessions as individual JSON files for the embedding pipeline.

    Extracts text-only content per message (strips tool blocks) to produce
    {"role": ..., "content": <string>} pairs for the embedding pipeline.

    Args:
        sessions: List of Session objects.
        output_dir: Directory to write JSON files.
    """
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
