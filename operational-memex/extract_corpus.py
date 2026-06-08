"""Extract conversation corpus from memex databases for analysis.

Reads all configured memex databases and exports conversations as JSON
for the embedding pipeline. Each conversation becomes one record with
its full text content and metadata.

Usage:
    python extract_corpus.py --output-dir data/corpus
    python extract_corpus.py --output-dir data/corpus --source chatgpt
    python extract_corpus.py --output-dir data/corpus --stats-only
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from memex.config import load_config
from memex.db import Database


def extract_conversations(db, source_filter=None):
    """Extract all conversations from a database with text content."""
    where = ""
    params = []
    if source_filter:
        where = "WHERE c.source = ?"
        params = [source_filter]

    rows = db.conn.execute(f"""
        SELECT c.id, c.title, c.source, c.model, c.message_count,
               c.created_at, c.updated_at, c.parent_conversation_id,
               c.metadata
        FROM conversations c
        {where}
        ORDER BY c.created_at
    """, params).fetchall()

    conversations = []
    for row in rows:
        conv_id = row["id"]

        # Get all messages ordered by tree structure
        messages = db.conn.execute("""
            SELECT id, role, content, parent_id, model, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at
        """, [conv_id]).fetchall()

        # Extract text content from all messages
        texts = []
        msg_count = 0
        for msg in messages:
            msg_count += 1
            content = msg["content"]
            if not content:
                continue
            try:
                blocks = json.loads(content)
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            texts.append(f"[{msg['role']}]: {text}")
            except (json.JSONDecodeError, TypeError):
                if isinstance(content, str) and content.strip():
                    texts.append(f"[{msg['role']}]: {content.strip()}")

        if not texts:
            continue

        full_text = "\n\n".join(texts)

        # Get tags
        tags = [r["tag"] for r in db.conn.execute(
            "SELECT tag FROM tags WHERE conversation_id = ?", [conv_id]
        ).fetchall()]

        # Get enrichments
        enrichments = {}
        for e in db.conn.execute(
            "SELECT type, value FROM enrichments WHERE conversation_id = ?",
            [conv_id]
        ).fetchall():
            enrichments.setdefault(e["type"], []).append(e["value"])

        # Get notes
        notes = [r["text"] for r in db.conn.execute(
            "SELECT text FROM notes WHERE conversation_id = ?", [conv_id]
        ).fetchall()]

        metadata = {}
        raw_meta = row["metadata"]
        if raw_meta:
            try:
                metadata = json.loads(raw_meta)
            except (json.JSONDecodeError, TypeError):
                pass

        conversations.append({
            "id": conv_id,
            "title": row["title"],
            "source": row["source"],
            "model": row["model"],
            "message_count": msg_count,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "parent_conversation_id": row["parent_conversation_id"],
            "tags": tags,
            "enrichments": enrichments,
            "notes": notes,
            "has_notes": len(notes) > 0,
            "text": full_text,
            "text_length": len(full_text),
            "metadata": metadata,
        })

    return conversations


def print_stats(all_convs):
    """Print corpus statistics."""
    print(f"\n{'='*60}")
    print(f"CORPUS STATISTICS")
    print(f"{'='*60}")
    print(f"Total conversations: {len(all_convs)}")

    # By source
    by_source = Counter(c["source"] for c in all_convs)
    print(f"\nBy source:")
    for source, count in by_source.most_common():
        print(f"  {source or 'unknown':20s} {count:6d}")

    # By database
    by_db = Counter(c["_db_name"] for c in all_convs)
    print(f"\nBy database:")
    for db_name, count in by_db.most_common():
        print(f"  {db_name:20s} {count:6d}")

    # Text stats
    lengths = [c["text_length"] for c in all_convs]
    print(f"\nText length (chars):")
    print(f"  min:    {min(lengths):>10,}")
    print(f"  median: {sorted(lengths)[len(lengths)//2]:>10,}")
    print(f"  mean:   {sum(lengths)//len(lengths):>10,}")
    print(f"  max:    {max(lengths):>10,}")
    print(f"  total:  {sum(lengths):>10,}")

    # Message counts
    msg_counts = [c["message_count"] for c in all_convs]
    print(f"\nMessages per conversation:")
    print(f"  min:    {min(msg_counts):>6}")
    print(f"  median: {sorted(msg_counts)[len(msg_counts)//2]:>6}")
    print(f"  mean:   {sum(msg_counts)//len(msg_counts):>6}")
    print(f"  max:    {max(msg_counts):>6}")

    # Temporal range
    dates = sorted(c["created_at"] for c in all_convs if c["created_at"])
    if dates:
        print(f"\nTemporal range: {dates[0][:10]} to {dates[-1][:10]}")

    # Notes
    with_notes = sum(1 for c in all_convs if c["has_notes"])
    total_notes = sum(len(c["notes"]) for c in all_convs)
    print(f"\nMarginalia: {total_notes} notes across {with_notes} conversations")

    # Parent conversations (delegation)
    with_parent = sum(1 for c in all_convs if c["parent_conversation_id"])
    print(f"Delegation: {with_parent} subagent conversations")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract memex corpus for analysis")
    parser.add_argument("--output-dir", default="data/corpus",
                        help="Output directory for JSON files")
    parser.add_argument("--source", help="Filter by source (chatgpt, anthropic, gemini, claude_code)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print statistics without writing files")
    parser.add_argument("--config", default=os.path.expanduser("~/.memex/config.yaml"),
                        help="Path to memex config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    all_convs = []

    # Skip claude_code_full — subagent conversations are machine-generated
    # scaffolding, not personal cognitive artifacts
    skip_dbs = {"claude_code_full"}

    for db_name, db_config in config.get("databases", {}).items():
        if db_name in skip_dbs:
            print(f"  skip {db_name}: excluded (subagent-heavy)")
            continue
        db_dir = os.path.expanduser(db_config["path"])
        db_file = os.path.join(db_dir, "conversations.db")
        if not os.path.exists(db_file):
            print(f"  skip {db_name}: {db_file} not found")
            continue

        print(f"  reading {db_name} ({db_file})...")
        db = Database(db_dir, readonly=True)
        convs = extract_conversations(db, source_filter=args.source)
        for c in convs:
            c["_db_name"] = db_name
        all_convs.extend(convs)
        db.close()
        print(f"    {len(convs)} conversations extracted")

    if not all_convs:
        print("No conversations found.")
        sys.exit(1)

    # Deduplicate by conversation ID (same conv may appear in multiple DBs)
    # Prefer the version from the most detailed database
    DB_PRIORITY = {"claude_code_full": 0, "claude_code": 1, "conversations": 2}
    seen = {}
    for conv in all_convs:
        cid = conv["id"]
        if cid not in seen:
            seen[cid] = conv
        else:
            existing_priority = DB_PRIORITY.get(seen[cid]["_db_name"], 99)
            new_priority = DB_PRIORITY.get(conv["_db_name"], 99)
            if new_priority < existing_priority:
                seen[cid] = conv
    all_convs = sorted(seen.values(), key=lambda c: c.get("created_at") or "")

    print_stats(all_convs)

    if args.stats_only:
        return

    # Write corpus
    os.makedirs(args.output_dir, exist_ok=True)

    # One JSON per conversation (matches agentic pipeline convention)
    for conv in all_convs:
        safe_id = conv["id"].replace(":", "_")
        path = os.path.join(args.output_dir, f"{safe_id}.json")
        with open(path, "w") as f:
            json.dump(conv, f, ensure_ascii=False, indent=2)

    # Also write a manifest for the embedding pipeline
    manifest = [{
        "id": c["id"],
        "title": c["title"],
        "source": c["source"],
        "model": c["model"],
        "created_at": c["created_at"],
        "message_count": c["message_count"],
        "text_length": c["text_length"],
        "has_notes": c["has_notes"],
        "parent_conversation_id": c["parent_conversation_id"],
        "db_name": c["_db_name"],
    } for c in all_convs]

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(all_convs)} conversations to {args.output_dir}/")
    print(f"Manifest: {args.output_dir}/manifest.json")


if __name__ == "__main__":
    main()
