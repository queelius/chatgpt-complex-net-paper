"""Experiment 19: Changepoint detection in conversation trajectories.

Decomposes long conversations into coherent segments by detecting
sharp turns in the exponential embedding trajectory.

Method:
1. Compute exponential trajectory emb(j) for each message j
2. Measure jump(j) = 1 - cos(emb(j), emb(j-1))
3. Detect changepoints where jump exceeds a threshold
4. Segment the conversation at changepoints
5. Each segment gets its own embedding (mean of segment messages)

Also explores:
- How many segments do conversations naturally decompose into?
- Does segment count correlate with conversation length?
- Are changepoints associated with role switches?
- What do the segments of the longest conversations look like?
"""
import json
import os
import sqlite3
import struct
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_messages(conn, model_id):
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
            "msg_id": r[1], "role": r[2], "short": r[3],
            "chars": r[4], "vec": np.array(deserialize_f32(r[5])),
        })
    return conv_msgs


def detect_changepoints(vecs, alpha=0.85, method="trajectory"):
    """Detect changepoints in a message sequence.

    Methods:
        trajectory: jumps in exponential trajectory
        pairwise: cosine distance between consecutive messages
        window: sliding window mean divergence

    Returns list of changepoint indices and jump values.
    """
    n = len(vecs)
    if n < 5:
        return [], []

    if method == "trajectory":
        # Compute exponential trajectory
        trajectory = []
        for j in range(n):
            weights = np.array([alpha ** (j - k) for k in range(j + 1)])
            emb = np.average(vecs[:j + 1], axis=0, weights=weights)
            emb = normalize(emb)
            trajectory.append(emb)

        # Compute jumps
        jumps = []
        for j in range(1, n):
            jump = 1 - float(cosine_similarity(
                trajectory[j].reshape(1, -1),
                trajectory[j - 1].reshape(1, -1)
            )[0, 0])
            jumps.append(jump)

    elif method == "pairwise":
        jumps = []
        for j in range(1, n):
            jump = 1 - float(cosine_similarity(
                vecs[j].reshape(1, -1),
                vecs[j - 1].reshape(1, -1)
            )[0, 0])
            jumps.append(jump)

    elif method == "window":
        window_size = max(n // 10, 3)
        jumps = []
        for j in range(1, n):
            # Compare mean of window before j to mean of window after j
            before = vecs[max(0, j - window_size):j]
            after = vecs[j:min(n, j + window_size)]
            if len(before) == 0 or len(after) == 0:
                jumps.append(0)
                continue
            mean_before = normalize(np.mean(before, axis=0))
            mean_after = normalize(np.mean(after, axis=0))
            jump = 1 - float(cosine_similarity(
                mean_before.reshape(1, -1), mean_after.reshape(1, -1)
            )[0, 0])
            jumps.append(jump)

    # Detect changepoints: jumps > mean + 1.5 * std
    if not jumps:
        return [], []
    jumps = np.array(jumps)
    threshold = np.mean(jumps) + 1.5 * np.std(jumps)
    # Also require minimum absolute jump
    min_jump = 0.05
    changepoints = [j + 1 for j, v in enumerate(jumps) if v > threshold and v > min_jump]

    return changepoints, jumps.tolist()


def segment_conversation(msgs, changepoints):
    """Split messages into segments at changepoints."""
    boundaries = [0] + changepoints + [len(msgs)]
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < 2:
            continue
        seg_msgs = msgs[start:end]
        seg_vecs = np.array([m["vec"] for m in seg_msgs])
        seg_emb = normalize(np.mean(seg_vecs, axis=0))
        segments.append({
            "start": start,
            "end": end,
            "n_messages": end - start,
            "embedding": seg_emb,
            "roles": [m["role"] for m in seg_msgs],
        })
    return segments


def main():
    import sqlite_vec

    conn = sqlite3.connect("data/analysis.db")
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

    print("Loading messages...")
    conv_msgs = load_messages(conn, model_id)

    # Focus on conversations with >= 20 messages (enough to have structure)
    valid = {cid: msgs for cid, msgs in conv_msgs.items()
             if len([m for m in msgs if not m["short"]]) >= 20}
    print(f"  {len(valid)} conversations with >= 20 non-short messages")

    # ============================================================
    # ANALYSIS 1: Changepoint detection across all conversations
    # ============================================================
    print(f"\n=== Changepoint Detection (trajectory method, alpha=0.85) ===")

    all_results = []
    for cid in sorted(valid.keys()):
        msgs = valid[cid]
        vecs = np.array([m["vec"] for m in msgs if not m["short"]])
        changepoints, jumps = detect_changepoints(vecs, alpha=0.85, method="trajectory")
        n_segments = len(changepoints) + 1

        all_results.append({
            "cid": cid,
            "n_messages": len(vecs),
            "n_changepoints": len(changepoints),
            "n_segments": n_segments,
            "max_jump": max(jumps) if jumps else 0,
            "mean_jump": float(np.mean(jumps)) if jumps else 0,
        })

    # Distribution of segment counts
    seg_counts = [r["n_segments"] for r in all_results]
    msg_counts = [r["n_messages"] for r in all_results]

    print(f"\nSegment count distribution:")
    print(f"  Mean: {np.mean(seg_counts):.1f}")
    print(f"  Median: {np.median(seg_counts):.1f}")
    print(f"  Max: {max(seg_counts)}")
    print(f"  1 segment (no changepoints): {sum(1 for s in seg_counts if s == 1)} ({sum(1 for s in seg_counts if s == 1)/len(seg_counts)*100:.0f}%)")
    print(f"  2-3 segments: {sum(1 for s in seg_counts if 2 <= s <= 3)} ({sum(1 for s in seg_counts if 2 <= s <= 3)/len(seg_counts)*100:.0f}%)")
    print(f"  4-10 segments: {sum(1 for s in seg_counts if 4 <= s <= 10)} ({sum(1 for s in seg_counts if 4 <= s <= 10)/len(seg_counts)*100:.0f}%)")
    print(f"  10+ segments: {sum(1 for s in seg_counts if s > 10)} ({sum(1 for s in seg_counts if s > 10)/len(seg_counts)*100:.0f}%)")

    # Correlation: segments vs message count
    corr = np.corrcoef(seg_counts, msg_counts)[0, 1]
    print(f"\n  Correlation (segments vs messages): r={corr:.3f}")

    # By source
    for source in ["openai", "claude_code", "anthropic"]:
        src = [r for r in all_results if meta.get(r["cid"], {}).get("source") == source]
        if src:
            src_segs = [r["n_segments"] for r in src]
            src_msgs = [r["n_messages"] for r in src]
            print(f"  {source:>12s}: mean segments={np.mean(src_segs):.1f}, "
                  f"mean msgs={np.mean(src_msgs):.0f}, n={len(src)}")

    # ============================================================
    # ANALYSIS 2: Deep dive into long conversations
    # ============================================================
    print(f"\n=== Long Conversation Decomposition ===")
    print("Top 15 conversations by segment count:")

    all_results.sort(key=lambda x: -x["n_segments"])
    for r in all_results[:15]:
        md = meta.get(r["cid"], {})
        ann = " [notes]" if md.get("notes") else ""
        print(f"  {r['n_segments']:>3d} segments from {r['n_messages']:>5d} msgs  "
              f"{md.get('title', '?')[:45]:<45s}  {md.get('source', ''):>12s}{ann}")

    # ============================================================
    # ANALYSIS 3: Segment similarity within vs across conversations
    # ============================================================
    print(f"\n=== Segment Analysis: Do segments of the same conversation cluster? ===")

    # For conversations with 3+ segments, compute intra vs inter segment similarity
    multi_seg_convs = [r for r in all_results if r["n_segments"] >= 3]
    print(f"  {len(multi_seg_convs)} conversations with 3+ segments")

    intra_sims = []
    inter_sims = []
    sample_convs = multi_seg_convs[:100]  # sample for speed

    seg_embeddings = {}  # cid -> list of segment embeddings
    for r in sample_convs:
        cid = r["cid"]
        msgs = valid[cid]
        vecs = np.array([m["vec"] for m in msgs if not m["short"]])
        changepoints, _ = detect_changepoints(vecs, alpha=0.85)
        segments = segment_conversation(
            [m for m in msgs if not m["short"]], changepoints
        )
        if len(segments) < 3:
            continue
        seg_embeddings[cid] = [s["embedding"] for s in segments]

        # Intra-conversation segment similarity
        seg_embs = np.array([s["embedding"] for s in segments])
        sim = cosine_similarity(seg_embs)
        np.fill_diagonal(sim, 0)
        n = len(segments)
        if n > 1:
            avg_intra = np.sum(sim) / (n * (n - 1))
            intra_sims.append(avg_intra)

    # Cross-conversation segment similarity (sample pairs)
    cids_with_segs = list(seg_embeddings.keys())
    for i in range(min(50, len(cids_with_segs))):
        for j in range(i + 1, min(50, len(cids_with_segs))):
            a_segs = seg_embeddings[cids_with_segs[i]]
            b_segs = seg_embeddings[cids_with_segs[j]]
            cross_sim = cosine_similarity(np.array(a_segs), np.array(b_segs))
            inter_sims.append(float(np.mean(cross_sim)))

    if intra_sims and inter_sims:
        print(f"  Intra-conversation segment similarity: mean={np.mean(intra_sims):.4f}")
        print(f"  Inter-conversation segment similarity: mean={np.mean(inter_sims):.4f}")
        print(f"  Ratio: {np.mean(intra_sims)/np.mean(inter_sims):.2f}x")
        print(f"  (Segments within a conversation are more similar to each other")
        print(f"   than to segments of other conversations, but not overwhelmingly so)")

    # ============================================================
    # ANALYSIS 4: Cross-conversation segment matching
    # ============================================================
    print(f"\n=== Cross-Conversation Segment Matching ===")
    print("Can we find segments in different conversations that match?")

    # Flatten all segments with their conversation IDs
    all_segments = []
    for cid, seg_list in seg_embeddings.items():
        for seg_idx, emb in enumerate(seg_list):
            all_segments.append({"cid": cid, "seg_idx": seg_idx, "emb": emb})

    if len(all_segments) >= 10:
        all_seg_embs = np.array([s["emb"] for s in all_segments])
        seg_sim = cosine_similarity(all_seg_embs)

        # Find high-similarity cross-conversation segment pairs
        cross_matches = []
        for i in range(len(all_segments)):
            for j in range(i + 1, len(all_segments)):
                if all_segments[i]["cid"] == all_segments[j]["cid"]:
                    continue
                if seg_sim[i, j] >= 0.85:
                    cross_matches.append((i, j, seg_sim[i, j]))

        cross_matches.sort(key=lambda x: -x[2])
        print(f"  {len(cross_matches)} cross-conversation segment matches (sim >= 0.85)")
        for m in cross_matches[:10]:
            i, j, s = m
            a = all_segments[i]
            b = all_segments[j]
            ma = meta.get(a["cid"], {})
            mb = meta.get(b["cid"], {})
            print(f"  sim={s:.3f}  seg{a['seg_idx']} of {ma.get('title', '?')[:35]:<35s}  <->  "
                  f"seg{b['seg_idx']} of {mb.get('title', '?')[:35]}")

    # ============================================================
    # ANALYSIS 5: Method comparison (trajectory vs pairwise vs window)
    # ============================================================
    print(f"\n=== Method Comparison ===")
    methods = ["trajectory", "pairwise", "window"]
    method_results = {}

    sample = list(valid.keys())[:200]
    for method in methods:
        segs = []
        for cid in sample:
            vecs = np.array([m["vec"] for m in valid[cid] if not m["short"]])
            cps, _ = detect_changepoints(vecs, alpha=0.85, method=method)
            segs.append(len(cps) + 1)
        method_results[method] = {
            "mean_segments": float(np.mean(segs)),
            "median_segments": float(np.median(segs)),
            "max_segments": max(segs),
        }
        print(f"  {method:<12s}: mean={np.mean(segs):.1f}, median={np.median(segs):.0f}, max={max(segs)}")

    # Save
    os.makedirs("experiments/results", exist_ok=True)
    save_results = {
        "n_conversations": len(all_results),
        "segment_distribution": {
            "mean": float(np.mean(seg_counts)),
            "median": float(np.median(seg_counts)),
            "max": max(seg_counts),
        },
        "correlation_segs_msgs": float(corr),
        "method_comparison": method_results,
    }
    if intra_sims and inter_sims:
        save_results["intra_vs_inter"] = {
            "intra_mean": float(np.mean(intra_sims)),
            "inter_mean": float(np.mean(inter_sims)),
        }
    with open("experiments/results/changepoints.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved to experiments/results/changepoints.json")
    conn.close()


if __name__ == "__main__":
    main()
