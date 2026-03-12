"""Export agentic sessions in different content modes for embedding experiments.

Creates one directory per mode, each containing JSON files compatible
with the code/cli.py embedding pipeline.
"""
import json
import sys
from pathlib import Path

AGENTIC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(AGENTIC_DIR.parent))

from agentic.extract import extract_sessions
from agentic.preprocess import (
    extract_text_full,
    extract_text_only,
    extract_user_only,
    extract_assistant_only,
    extract_tool_names_only,
)

DB_PATH = Path.home() / ".memex" / "default" / "conversations.db"
DATA_DIR = AGENTIC_DIR / "data"


def export_single_content(sessions, mode: str, output_dir: Path):
    """Export sessions as JSON with a specific content extraction mode.

    For modes that produce a single text string per session, we create
    a simple {"messages": [{"role": "user", "content": TEXT}]} format
    so the embedding pipeline treats the whole session as one document.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    extractors = {
        "text-full": extract_text_full,
        "text-only": extract_text_only,
        "user-only": extract_user_only,
        "assistant-only": extract_assistant_only,
        "tool-names-only": extract_tool_names_only,
    }

    extract_fn = extractors[mode]
    exported = 0
    skipped = 0

    for session in sessions:
        text = extract_fn(session.messages)
        if not text or not text.strip():
            skipped += 1
            continue

        # Wrap as a single "user" message for the embedding pipeline
        doc = {
            "messages": [{"role": "user", "content": text}],
            "metadata": {
                "id": session.id,
                "title": session.title,
                "content_mode": mode,
                "message_count": session.message_count,
                "created_at": session.created_at,
            },
        }

        filename = session.id.replace(":", "_") + ".json"
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        exported += 1

    return exported, skipped


def export_by_role(sessions, output_dir: Path):
    """Export sessions with per-role messages preserved.

    This creates {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    so the embedding pipeline can apply role-aggregate weighting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0

    for session in sessions:
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

        if not messages:
            continue

        doc = {
            "messages": messages,
            "metadata": {
                "id": session.id,
                "title": session.title,
                "content_mode": "by-role",
                "message_count": session.message_count,
                "created_at": session.created_at,
            },
        }

        filename = session.id.replace(":", "_") + ".json"
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        exported += 1

    return exported, 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export sessions in multiple content modes")
    parser.add_argument("--modes", nargs="+",
                       default=["text-full", "user-only", "assistant-only", "tool-names-only"],
                       help="Content modes to export")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    args = parser.parse_args()

    print(f"Extracting sessions from {args.db}...")
    sessions = extract_sessions(args.db, include_subagents=True)
    print(f"  {len(sessions)} sessions extracted")

    for mode in args.modes:
        output_dir = DATA_DIR / f"sessions-{mode}"
        print(f"\nExporting mode: {mode} → {output_dir}")
        exported, skipped = export_single_content(sessions, mode, output_dir)
        print(f"  {exported} exported, {skipped} skipped (empty content)")


if __name__ == "__main__":
    main()
