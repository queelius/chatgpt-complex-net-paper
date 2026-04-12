"""Experiment 2: Episode detection via changepoint analysis.

Decomposes conversations into episodes (coherent sub-conversations)
and analyzes the results both quantitatively and qualitatively.

Outputs detailed examples for human review.
"""
import json
import os
import sqlite3
import struct
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.expanduser("~/github/beta/embflow"))
sys.path.insert(0, os.path.expanduser("~/github/beta/memex"))
import embflow as ef

ANALYSIS_DB = os.path.expanduser(
    "~/github/papers/cognitive-mri-ai-conversations/operational-memex/data/analysis.db"
)


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def load_data():
    import sqlite_vec
    conn = sqlite3.connect(ANALYSIS_DB)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    model_id = conn.execute("SELECT id FROM embedding_models").fetchone()[0]

    meta = {}
    for r in conn.execute(
        "SELECT id, title, source, message_count, created_at, has_notes FROM conversations"
    ).fetchall():
        meta[r[0]] = {"title": r[1], "source": r[2], "msgs": r[3],
                       "date": (r[4] or "")[:10], "notes": bool(r[5])}

    # Load per-conversation message embeddings with roles and char counts
    rows = conn.execute("""
        SELECT me.conversation_id, me.message_id, me.role, me.is_short,
               me.char_count, mv.embedding
        FROM message_embeddings me
        JOIN message_vectors mv ON mv.conversation_id = me.conversation_id
          AND mv.message_id = me.message_id AND mv.model_id = me.model_id
        WHERE me.model_id = ?
        ORDER BY me.conversation_id, me.embedded_at
    """, [model_id]).fetchall()

    conv_msgs = defaultdict(list)
    for r in rows:
        conv_msgs[r[0]].append({
            "msg_id": r[1],
            "role": r[2],
            "short": r[3],
            "chars": r[4],
            "vec": np.array(deserialize_f32(r[5])),
        })

    return conv_msgs, meta, conn


def get_message_text(conn_memex, conv_id, msg_id):
    """Fetch message text from memex DB."""
    row = conn_memex.execute(
        "SELECT content FROM messages WHERE conversation_id = ? AND id = ?",
        [conv_id, msg_id]
    ).fetchone()
    if not row or not row["content"]:
        return ""
    try:
        blocks = json.loads(row["content"])
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                return b.get("text", "")[:300]
        return ""
    except (json.JSONDecodeError, TypeError):
        return str(row["content"])[:300]


def analyze_conversation(msgs, alpha=0.85):
    """Run full episode detection on a single conversation."""
    non_short = [m for m in msgs if not m["short"]]
    if len(non_short) < 10:
        return None

    vecs = np.array([m["vec"] for m in non_short])
    lens = ef.Exponential(alpha)

    # Trajectory
    traj = lens.trajectory(vecs)

    # Angular velocity (where the trajectory turns)
    ang = ef.angular_velocity(traj)

    # Detect changepoints
    changepoint_indices = ef.peaks(ang, threshold="auto")
    # Shift: angular_velocity[k] is between traj[k] and traj[k+1]
    boundaries = [cp + 1 for cp in changepoint_indices]

    # Segment
    segments = ef.segment(vecs, boundaries)

    # For each segment, compute its embedding and characterize
    seg_info = []
    for seg in segments:
        seg_vecs = seg["vectors"]
        seg_emb = ef.Uniform().project(seg_vecs)
        seg_msgs = non_short[seg["start"]:seg["end"]]

        # Roles in this segment
        roles = [m["role"] for m in seg_msgs]
        user_count = roles.count("user")
        asst_count = roles.count("assistant")

        # Internal drift
        if len(seg_vecs) >= 3:
            seg_traj = lens.trajectory(seg_vecs)
            seg_drift = ef.drift(seg_traj)
        else:
            seg_drift = 0.0

        seg_info.append({
            "start": seg["start"],
            "end": seg["end"],
            "n_messages": len(seg_vecs),
            "user_msgs": user_count,
            "asst_msgs": asst_count,
            "embedding": seg_emb,
            "drift": seg_drift,
            "msg_ids": [m["msg_id"] for m in seg_msgs],
            "first_msg_id": seg_msgs[0]["msg_id"],
            "last_msg_id": seg_msgs[-1]["msg_id"],
        })

    # Inter-segment similarity
    if len(seg_info) >= 2:
        seg_embs = np.array([s["embedding"] for s in seg_info])
        inter_sim = cosine_similarity(seg_embs)
    else:
        inter_sim = None

    return {
        "n_messages": len(non_short),
        "n_segments": len(seg_info),
        "segments": seg_info,
        "angular_velocity": ang,
        "changepoints": boundaries,
        "inter_segment_sim": inter_sim,
    }


def main():
    from memex.config import load_config
    from memex.db import Database

    print("Loading embeddings...")
    conv_msgs, meta, analysis_conn = load_data()

    # Open memex DB for message text
    config = load_config(os.path.expanduser("~/.memex/config.yaml"))
    memex_dbs = {}
    for db_name, db_config in config.get("databases", {}).items():
        db_dir = os.path.expanduser(db_config["path"])
        db_file = os.path.join(db_dir, "conversations.db")
        if os.path.exists(db_file):
            memex_dbs[db_name] = Database(db_dir, readonly=True)

    # Select conversations to analyze in detail
    # Focus on: long conversations, annotated conversations, and diverse examples
    candidates = []
    for cid, msgs in conv_msgs.items():
        non_short = [m for m in msgs if not m["short"]]
        if len(non_short) < 20:
            continue
        md = meta.get(cid, {})
        candidates.append({
            "cid": cid,
            "n_msgs": len(non_short),
            "title": md.get("title", "?"),
            "source": md.get("source", "?"),
            "date": md.get("date", ""),
            "notes": md.get("notes", False),
        })

    # Sort by message count to get a range
    candidates.sort(key=lambda x: -x["n_msgs"])

    # Pick a diverse sample:
    # - Top 5 longest
    # - Any annotated
    # - A few medium-length ChatGPT conversations
    # - A few medium Claude Code sessions
    selected = []

    # Longest 5
    for c in candidates[:5]:
        selected.append(c)

    # Annotated
    for c in candidates:
        if c["notes"] and c["cid"] not in {s["cid"] for s in selected}:
            selected.append(c)

    # Medium ChatGPT (around rank 50-60)
    openai_cands = [c for c in candidates if c["source"] == "openai" and c["cid"] not in {s["cid"] for s in selected}]
    for c in openai_cands[20:25]:
        selected.append(c)

    # Medium Claude Code
    cc_cands = [c for c in candidates if c["source"] == "claude_code" and c["cid"] not in {s["cid"] for s in selected}]
    for c in cc_cands[10:15]:
        selected.append(c)

    print(f"\nAnalyzing {len(selected)} conversations in detail...\n")

    # Run episode detection on all candidates for statistics
    print("=== Corpus-Wide Statistics ===")
    all_results = []
    for c in candidates:
        result = analyze_conversation(conv_msgs[c["cid"]])
        if result:
            all_results.append({
                "cid": c["cid"],
                "source": c["source"],
                "n_messages": result["n_messages"],
                "n_segments": result["n_segments"],
            })

    seg_counts = [r["n_segments"] for r in all_results]
    print(f"  {len(all_results)} conversations analyzed")
    print(f"  Mean segments: {np.mean(seg_counts):.1f}")
    print(f"  Median: {np.median(seg_counts):.0f}")
    print(f"  Distribution: 1={sum(1 for s in seg_counts if s==1)}, "
          f"2={sum(1 for s in seg_counts if s==2)}, "
          f"3={sum(1 for s in seg_counts if s==3)}, "
          f"4={sum(1 for s in seg_counts if s==4)}, "
          f"5+={sum(1 for s in seg_counts if s>=5)}")

    for source in ["openai", "claude_code", "anthropic"]:
        src_segs = [r["n_segments"] for r in all_results if r["source"] == source]
        if src_segs:
            print(f"  {source:>12s}: mean={np.mean(src_segs):.1f}, n={len(src_segs)}")

    # Detailed analysis of selected conversations
    print(f"\n{'='*80}")
    print(f"  DETAILED EPISODE ANALYSIS")
    print(f"{'='*80}")

    for sel in selected:
        cid = sel["cid"]
        result = analyze_conversation(conv_msgs[cid])
        if not result:
            continue

        md = meta.get(cid, {})
        ann = " [ANNOTATED]" if md.get("notes") else ""

        print(f"\n{'='*70}")
        print(f"  {md.get('title', '?')}")
        print(f"  {cid}")
        print(f"  Source: {md.get('source')}, Date: {md.get('date')}{ann}")
        print(f"  {result['n_messages']} messages -> {result['n_segments']} episodes")
        print(f"{'='*70}")

        # Find the right memex DB for this conversation
        conn_memex = None
        for db_name, db in memex_dbs.items():
            row = db.conn.execute(
                "SELECT id FROM conversations WHERE id = ?", [cid]
            ).fetchone()
            if row:
                conn_memex = db.conn
                break

        for i, seg in enumerate(result["segments"]):
            # Inter-segment similarity context
            sim_str = ""
            if result["inter_segment_sim"] is not None and result["n_segments"] > 1:
                sims = result["inter_segment_sim"][i]
                other_sims = [f"ep{j}:{sims[j]:.2f}" for j in range(len(sims)) if j != i]
                sim_str = f"  sim: {', '.join(other_sims)}"

            print(f"\n  Episode {i+1}/{result['n_segments']} "
                  f"(msgs {seg['start']+1}-{seg['end']}, "
                  f"user={seg['user_msgs']}, asst={seg['asst_msgs']}, "
                  f"drift={seg['drift']:.3f}){sim_str}")
            print(f"  {'~'*60}")

            # Show first 2 user messages of this segment
            shown = 0
            for msg_id in seg["msg_ids"]:
                msg = next((m for m in conv_msgs[cid] if m["msg_id"] == msg_id), None)
                if msg and msg["role"] == "user" and not msg["short"]:
                    text = ""
                    if conn_memex:
                        text = get_message_text(conn_memex, cid, msg_id)
                    if not text:
                        text = f"[{msg['chars']} chars]"
                    # Truncate
                    text = text[:200].replace("\n", " ")
                    print(f"    [user] {text}")
                    shown += 1
                    if shown >= 2:
                        break

        # Angular velocity profile (show where the big jumps are)
        ang = result["angular_velocity"]
        if len(ang) > 0:
            print(f"\n  Angular velocity: mean={np.mean(ang):.4f}, "
                  f"max={np.max(ang):.4f} at msg {np.argmax(ang)+1}, "
                  f"changepoints at msgs: {result['changepoints']}")

    # Close
    for db in memex_dbs.values():
        db.close()
    analysis_conn.close()

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
