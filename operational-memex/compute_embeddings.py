"""Compute per-message embeddings and store in SQLite with sqlite-vec.

Reads conversations from memex databases directly, embeds each message
individually, and stores vectors with full provenance. Conversation-level
embeddings are composed at query time from message vectors.

Incremental: skips messages whose content_hash already exists for the
given model. Re-embeds messages whose content has changed.

Usage:
    python compute_embeddings.py --db data/analysis.db
    python compute_embeddings.py --db data/analysis.db --model text-embedding-3-large --dims 1024
    python compute_embeddings.py --db data/analysis.db --stats-only
"""
import argparse
import hashlib
import json
import os
import sqlite3
import struct
import sys
from datetime import datetime, timezone

import sqlite_vec
from openai import OpenAI
from tqdm import tqdm

from memex.config import load_config
from memex.db import Database

import tiktoken

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMS = 256
MAX_TOKENS = 8191
MIN_CHARS_FOR_EMBEDDING = 20  # below this, flag as short

_tokenizer = None

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.encoding_for_model(DEFAULT_MODEL)
    return _tokenizer


def create_db(db_path, dims):
    """Create analysis database with per-message vector storage."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript("""
        -- Provenance: what model produced these embeddings?
        CREATE TABLE IF NOT EXISTS embedding_models (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            registered_at TEXT NOT NULL
        );

        -- Conversation metadata (mirror from memex)
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            model TEXT,
            message_count INTEGER,
            created_at TEXT,
            updated_at TEXT,
            parent_conversation_id TEXT,
            has_notes BOOLEAN,
            db_name TEXT,
            tags TEXT,
            enrichments TEXT,
            notes TEXT
        );

        -- Per-message embedding metadata
        CREATE TABLE IF NOT EXISTS message_embeddings (
            conversation_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            is_short BOOLEAN DEFAULT FALSE,
            truncated BOOLEAN DEFAULT FALSE,
            embedded_at TEXT NOT NULL,
            PRIMARY KEY (conversation_id, message_id, model_id)
        );

        -- Graph edges (populated by build_graph.py)
        CREATE TABLE IF NOT EXISTS edges (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            weight REAL NOT NULL,
            PRIMARY KEY (src, dst)
        );

        CREATE INDEX IF NOT EXISTS idx_me_conv ON message_embeddings(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_me_role ON message_embeddings(role);
        CREATE INDEX IF NOT EXISTS idx_me_model ON message_embeddings(model_id);
        CREATE INDEX IF NOT EXISTS idx_me_hash ON message_embeddings(content_hash);
        CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
        CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
        CREATE INDEX IF NOT EXISTS idx_conv_source ON conversations(source);
        CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at);
    """)

    # sqlite-vec virtual table for vectors
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS message_vectors USING vec0(
            conversation_id TEXT,
            message_id TEXT,
            model_id TEXT,
            embedding float[{dims}]
        )
    """)

    conn.commit()
    return conn


def serialize_f32(vec):
    """Serialize a float list to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def content_hash(text):
    """SHA-256 of the text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate_text(text):
    """Truncate text to fit within OpenAI's token limit using tiktoken."""
    enc = _get_tokenizer()
    tokens = enc.encode(text)
    if len(tokens) <= MAX_TOKENS:
        return text, False
    truncated = enc.decode(tokens[:MAX_TOKENS])
    return truncated, True


def register_model(conn, provider, model, dims):
    """Register an embedding model and return its ID."""
    model_id = f"{provider}:{model}:{dims}"
    conn.execute("""
        INSERT OR IGNORE INTO embedding_models (id, provider, model, dimensions, registered_at)
        VALUES (?, ?, ?, ?, ?)
    """, [model_id, provider, model, dims, datetime.now(timezone.utc).isoformat()])
    conn.commit()
    return model_id


def extract_messages_from_memex(config_path, skip_dbs=None):
    """Extract all messages from memex databases with conversation metadata.

    Returns (conversations_meta, messages) where messages is a list of
    (conversation_id, message_id, role, text) tuples.
    """
    skip_dbs = skip_dbs or {"claude_code_full"}
    config = load_config(config_path)
    conversations_meta = []
    messages = []

    for db_name, db_config in config.get("databases", {}).items():
        if db_name in skip_dbs:
            print(f"  skip {db_name}: excluded")
            continue

        db_dir = os.path.expanduser(db_config["path"])
        db_file = os.path.join(db_dir, "conversations.db")
        if not os.path.exists(db_file):
            print(f"  skip {db_name}: {db_file} not found")
            continue

        print(f"  reading {db_name}...")
        db = Database(db_dir, readonly=True)

        # Get conversations
        convs = db.conn.execute("""
            SELECT id, title, source, model, message_count,
                   created_at, updated_at, parent_conversation_id, metadata
            FROM conversations
            ORDER BY created_at
        """).fetchall()

        for conv in convs:
            conv_id = conv["id"]

            # Get tags, enrichments, notes
            tags = [r["tag"] for r in db.conn.execute(
                "SELECT tag FROM tags WHERE conversation_id = ?", [conv_id]
            ).fetchall()]
            enrichments = {}
            for e in db.conn.execute(
                "SELECT type, value FROM enrichments WHERE conversation_id = ?",
                [conv_id]
            ).fetchall():
                enrichments.setdefault(e["type"], []).append(e["value"])
            notes = [r["text"] for r in db.conn.execute(
                "SELECT text FROM notes WHERE conversation_id = ?", [conv_id]
            ).fetchall()]

            conversations_meta.append({
                "id": conv_id,
                "title": conv["title"],
                "source": conv["source"],
                "model": conv["model"],
                "message_count": conv["message_count"],
                "created_at": conv["created_at"],
                "updated_at": conv["updated_at"],
                "parent_conversation_id": conv["parent_conversation_id"],
                "has_notes": len(notes) > 0,
                "db_name": db_name,
                "tags": json.dumps(tags),
                "enrichments": json.dumps(enrichments),
                "notes": json.dumps(notes),
            })

            # Get messages
            msgs = db.conn.execute("""
                SELECT id, role, content
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at
            """, [conv_id]).fetchall()

            for msg in msgs:
                raw_content = msg["content"]
                if not raw_content:
                    continue
                # Extract text from content blocks
                try:
                    blocks = json.loads(raw_content)
                    text_parts = []
                    for block in blocks:
                        if isinstance(block, dict) and block.get("type") == "text":
                            t = block.get("text", "").strip()
                            if t:
                                text_parts.append(t)
                    text = "\n".join(text_parts)
                except (json.JSONDecodeError, TypeError):
                    text = raw_content.strip() if isinstance(raw_content, str) else ""

                if text:
                    messages.append((conv_id, msg["id"], msg["role"], text))

        db.close()
        print(f"    {len(convs)} conversations, messages so far: {len(messages)}")

    # Deduplicate conversations (prefer claude_code over conversations DB)
    DB_PRIORITY = {"claude_code": 0, "conversations": 1}
    seen = {}
    for meta in conversations_meta:
        cid = meta["id"]
        if cid not in seen:
            seen[cid] = meta
        else:
            existing_pri = DB_PRIORITY.get(seen[cid]["db_name"], 99)
            new_pri = DB_PRIORITY.get(meta["db_name"], 99)
            if new_pri < existing_pri:
                seen[cid] = meta
    conversations_meta = list(seen.values())

    # Deduplicate messages (same message may appear in multiple DBs)
    seen_msgs = {}
    for conv_id, msg_id, role, text in messages:
        key = (conv_id, msg_id)
        if key not in seen_msgs:
            seen_msgs[key] = (conv_id, msg_id, role, text)
    messages = list(seen_msgs.values())

    return conversations_meta, messages


def get_existing_hashes(conn, model_id):
    """Get (conversation_id, message_id) -> content_hash for already-embedded messages."""
    rows = conn.execute("""
        SELECT conversation_id, message_id, content_hash
        FROM message_embeddings
        WHERE model_id = ?
    """, [model_id]).fetchall()
    return {(r[0], r[1]): r[2] for r in rows}


def compute_and_store(conn, messages, model_id, model_name, dims, batch_size=200):
    """Compute per-message embeddings in batches with incremental updates."""
    client = OpenAI()
    existing = get_existing_hashes(conn, model_id)
    now = datetime.now(timezone.utc).isoformat()

    # Determine which messages need embedding
    to_embed = []
    for conv_id, msg_id, role, text in messages:
        h = content_hash(text)
        existing_hash = existing.get((conv_id, msg_id))
        if existing_hash == h:
            continue  # already embedded with same content
        to_embed.append((conv_id, msg_id, role, text, h))

    if not to_embed:
        print("All messages already embedded.")
        return

    skipped_existing = len(messages) - len(to_embed)
    changed = sum(1 for c, m, r, t, h in to_embed if (c, m) in existing)
    new = len(to_embed) - changed
    print(f"Messages: {len(messages)} total, {skipped_existing} up-to-date, "
          f"{new} new, {changed} changed")

    # Process in batches
    for i in tqdm(range(0, len(to_embed), batch_size), desc="Embedding"):
        batch = to_embed[i:i + batch_size]

        # Prepare texts (truncate long ones)
        texts = []
        truncation_flags = []
        for _, _, _, text, _ in batch:
            t, was_truncated = truncate_text(text)
            texts.append(t)
            truncation_flags.append(was_truncated)

        # API call with retry on rate limit
        import time
        for attempt in range(5):
            try:
                response = client.embeddings.create(
                    model=model_name,
                    input=texts,
                    dimensions=dims,
                )
                break
            except Exception as e:
                if "rate_limit" in str(type(e).__name__).lower() or "429" in str(e):
                    wait = 2 ** attempt
                    tqdm.write(f"  rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        else:
            tqdm.write(f"  skipping batch after 5 retries")
            continue

        for (conv_id, msg_id, role, text, h), emb_data, was_truncated in zip(
            batch, response.data, truncation_flags
        ):
            vec = emb_data.embedding
            char_count = len(text)
            is_short = char_count < MIN_CHARS_FOR_EMBEDDING

            # Delete old embedding if content changed
            if (conv_id, msg_id) in existing:
                conn.execute("""
                    DELETE FROM message_vectors
                    WHERE conversation_id = ? AND message_id = ? AND model_id = ?
                """, [conv_id, msg_id, model_id])
                conn.execute("""
                    DELETE FROM message_embeddings
                    WHERE conversation_id = ? AND message_id = ? AND model_id = ?
                """, [conv_id, msg_id, model_id])

            # Store metadata
            conn.execute("""
                INSERT INTO message_embeddings
                (conversation_id, message_id, model_id, role, content_hash,
                 char_count, is_short, truncated, embedded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [conv_id, msg_id, model_id, role, h,
                  char_count, is_short, was_truncated, now])

            # Store vector
            conn.execute("""
                INSERT INTO message_vectors
                (conversation_id, message_id, model_id, embedding)
                VALUES (?, ?, ?, ?)
            """, [conv_id, msg_id, model_id, serialize_f32(vec)])

        conn.commit()

    print(f"Done. {len(to_embed)} message embeddings stored.")


def store_conversation_meta(conn, conversations_meta):
    """Store conversation metadata in analysis DB."""
    for meta in conversations_meta:
        conn.execute("""
            INSERT OR REPLACE INTO conversations
            (id, title, source, model, message_count, created_at,
             updated_at, parent_conversation_id, has_notes, db_name,
             tags, enrichments, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            meta["id"], meta["title"], meta["source"], meta["model"],
            meta["message_count"], meta["created_at"], meta["updated_at"],
            meta["parent_conversation_id"], meta["has_notes"], meta["db_name"],
            meta["tags"], meta["enrichments"], meta["notes"],
        ])
    conn.commit()


def print_stats(conn, model_id):
    """Print embedding statistics."""
    total = conn.execute(
        "SELECT COUNT(*) FROM message_embeddings WHERE model_id = ?",
        [model_id]
    ).fetchone()[0]

    by_role = conn.execute("""
        SELECT role, COUNT(*), SUM(is_short), SUM(truncated)
        FROM message_embeddings WHERE model_id = ?
        GROUP BY role ORDER BY COUNT(*) DESC
    """, [model_id]).fetchall()

    convs = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    by_source = conn.execute("""
        SELECT source, COUNT(*) FROM conversations
        GROUP BY source ORDER BY COUNT(*) DESC
    """).fetchall()

    # Storage size
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    db_size = os.path.getsize(db_path) if db_path and os.path.exists(db_path) else 0

    print(f"\n{'='*60}")
    print(f"EMBEDDING STATISTICS")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Conversations: {convs}")
    print(f"Message embeddings: {total}")
    print(f"\nBy role:")
    for role, count, short, trunc in by_role:
        print(f"  {role:15s} {count:>7d}  (short: {short or 0}, truncated: {trunc or 0})")
    print(f"\nBy source:")
    for source, count in by_source:
        print(f"  {source or 'unknown':15s} {count:>7d}")
    print(f"\nDB size: {db_size / 1024 / 1024:.1f} MB")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Compute per-message embeddings")
    parser.add_argument("--db", default="data/analysis.db",
                        help="Path to analysis SQLite database")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenAI embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dims", type=int, default=DEFAULT_DIMS,
                        help=f"Embedding dimensions (default: {DEFAULT_DIMS})")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Messages per API call (max 2048)")
    parser.add_argument("--config", default=os.path.expanduser("~/.memex/config.yaml"),
                        help="Path to memex config.yaml")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print statistics without embedding")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    conn = create_db(args.db, args.dims)
    model_id = register_model(conn, "openai", args.model, args.dims)

    if args.stats_only:
        print_stats(conn, model_id)
        conn.close()
        return

    # Extract from memex
    print("Extracting messages from memex databases...")
    conversations_meta, messages = extract_messages_from_memex(args.config)
    print(f"  {len(conversations_meta)} conversations, {len(messages)} messages")

    # Store conversation metadata
    store_conversation_meta(conn, conversations_meta)

    # Compute and store embeddings
    compute_and_store(conn, messages, model_id, args.model, args.dims,
                      batch_size=args.batch_size)

    print_stats(conn, model_id)
    conn.close()


if __name__ == "__main__":
    main()
