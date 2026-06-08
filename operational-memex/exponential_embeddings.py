"""Exponentially weighted message embeddings.

For a conversation with messages m1, m2, ..., mn, computes:
    emb_exp(j) = sum_{k=1}^{j} alpha^(j-k) * emb(mk)
    (normalized to unit length)

At j=n (end of conversation), this captures "where the conversation
ended up" with recency bias. At j=1, it's just the first message.

This enables a new kind of trail link: compare the ENDING of
conversation A with the BEGINNING of conversation B.

Usage:
    python exponential_embeddings.py --db data/analysis.db --alpha 0.85
"""
import argparse
import json
import os
import sqlite3
import struct
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def compute_exponential_embeddings(conn, model_id, alpha=0.85):
    """Compute start and end exponential embeddings per conversation.

    Returns dict: conv_id -> {
        'start': normalized exp embedding of first 20% of messages,
        'end': normalized exp embedding at final message,
        'mean': standard mean embedding,
        'n_messages': int,
    }
    """
    # Get all message embeddings ordered by conversation
    rows = conn.execute("""
        SELECT me.conversation_id, me.message_id, me.role, mv.embedding
        FROM message_embeddings me
        JOIN message_vectors mv
          ON mv.conversation_id = me.conversation_id
          AND mv.message_id = me.message_id
          AND mv.model_id = me.model_id
        WHERE me.model_id = ?
          AND me.is_short = 0
        ORDER BY me.conversation_id, me.embedded_at
    """, [model_id]).fetchall()

    # Group by conversation, preserve order
    conv_msgs = defaultdict(list)
    for r in rows:
        conv_msgs[r[0]].append({
            "role": r[2],
            "vec": np.array(deserialize_f32(r[3])),
        })

    results = {}
    for conv_id, msgs in conv_msgs.items():
        if len(msgs) < 3:
            continue

        vecs = np.array([m["vec"] for m in msgs])
        n = len(vecs)

        # Exponential embedding at end (recency-biased)
        weights_end = np.array([alpha ** (n - 1 - k) for k in range(n)])
        exp_end = np.average(vecs, axis=0, weights=weights_end)
        norm = np.linalg.norm(exp_end)
        if norm > 0:
            exp_end = exp_end / norm

        # Exponential embedding of first 20% (what the conversation started as)
        first_n = max(n // 5, 1)
        first_vecs = vecs[:first_n]
        if len(first_vecs) == 1:
            exp_start = first_vecs[0]
        else:
            w = np.array([alpha ** (first_n - 1 - k) for k in range(first_n)])
            exp_start = np.average(first_vecs, axis=0, weights=w)
        norm = np.linalg.norm(exp_start)
        if norm > 0:
            exp_start = exp_start / norm

        # Standard mean
        mean_emb = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        # Drift: how different is start from end?
        drift = 1 - float(cosine_similarity(
            exp_start.reshape(1, -1), exp_end.reshape(1, -1)
        )[0, 0])

        results[conv_id] = {
            "start": exp_start,
            "end": exp_end,
            "mean": mean_emb,
            "n_messages": n,
            "drift": drift,
        }

    return results


def main():
    import sqlite_vec
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/analysis.db")
    parser.add_argument("--alpha", type=float, default=0.85)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    model_id = conn.execute("SELECT id FROM embedding_models").fetchone()[0]

    meta = {}
    for r in conn.execute(
        "SELECT id, title, source, message_count, created_at, has_notes FROM conversations"
    ).fetchall():
        meta[r[0]] = {
            "title": r[1], "source": r[2], "msgs": r[3],
            "date": (r[4] or "")[:10], "notes": bool(r[5]),
        }

    print(f"Computing exponential embeddings (alpha={args.alpha})...")
    embs = compute_exponential_embeddings(conn, model_id, alpha=args.alpha)
    print(f"  {len(embs)} conversations with >= 3 non-short messages")

    # ============================================================
    # ANALYSIS 1: End-to-start links (continuation detection)
    # ============================================================
    print(f"\n=== End-to-Start Links (Intellectual Continuation) ===")
    print("Where conversation A ended is where conversation B began.")

    # Sort by date
    dated = []
    for cid in embs:
        d = meta.get(cid, {}).get("date", "")
        if d:
            try:
                dated.append((cid, datetime.strptime(d, "%Y-%m-%d")))
            except:
                pass
    dated.sort(key=lambda x: x[1])

    # Find best end-to-start links forward in time
    continuations = []
    for i, (cid_a, dt_a) in enumerate(dated):
        end_a = embs[cid_a]["end"]
        best = None
        for j in range(i + 1, len(dated)):
            cid_b, dt_b = dated[j]
            gap = (dt_b - dt_a).days
            if gap < 1:
                continue
            if gap > 365:
                break
            start_b = embs[cid_b]["start"]
            s = float(cosine_similarity(
                end_a.reshape(1, -1), start_b.reshape(1, -1)
            )[0, 0])
            if s >= 0.80:
                if best is None or s > best[1]:
                    best = (cid_b, s, gap)
        if best:
            continuations.append((cid_a, best[0], best[1], best[2]))

    print(f"Found {len(continuations)} end-to-start links (sim >= 0.80, gap >= 1d)")

    # Compare with mean-to-mean links at same threshold
    mean_links = 0
    for i, (cid_a, dt_a) in enumerate(dated):
        mean_a = embs[cid_a]["mean"]
        for j in range(i + 1, len(dated)):
            cid_b, dt_b = dated[j]
            gap = (dt_b - dt_a).days
            if gap < 1:
                continue
            if gap > 365:
                break
            mean_b = embs[cid_b]["mean"]
            s = float(cosine_similarity(
                mean_a.reshape(1, -1), mean_b.reshape(1, -1)
            )[0, 0])
            if s >= 0.80:
                mean_links += 1
                break  # just count one per source
    print(f"(For comparison: {mean_links} mean-to-mean links at same threshold)")

    # Show top end-to-start continuations with long gaps
    continuations.sort(key=lambda x: (-x[2] * np.log1p(x[3])))  # score by sim * log(gap)
    print(f"\nTop end-to-start continuations (scored by similarity * log(gap)):")
    for rank, (cid_a, cid_b, s, gap) in enumerate(continuations[:15], 1):
        ma = meta.get(cid_a, {})
        mb = meta.get(cid_b, {})
        score = s * np.log1p(gap)
        ann_a = " [notes]" if ma.get("notes") else ""
        ann_b = " [notes]" if mb.get("notes") else ""
        print(f"  {rank:>3d}. score={score:.2f}  sim={s:.3f}  gap={gap:>4d}d")
        print(f"       END: {ma.get('title', '?')[:50]:<50s}  {ma.get('date', '')} {ma.get('source', '')}{ann_a}")
        print(f"     START: {mb.get('title', '?')[:50]:<50s}  {mb.get('date', '')} {mb.get('source', '')}{ann_b}")

    # ============================================================
    # ANALYSIS 2: Convergent vs divergent conversations
    # ============================================================
    print(f"\n=== Convergence Analysis ===")
    print("Comparing mean embedding vs exponential end embedding.")

    # For each pair: is mean-similarity or end-similarity higher?
    # If end > mean: conversations converged to similar conclusions
    # If mean > end: conversations share topic but went different directions

    # Sample 500 high-similarity pairs (mean sim > 0.80)
    sample_pairs = []
    cid_list = [cid for cid, _ in dated]
    for i in range(min(1000, len(cid_list))):
        for j in range(i + 1, min(1000, len(cid_list))):
            a, b = cid_list[i], cid_list[j]
            if a not in embs or b not in embs:
                continue
            mean_sim = float(cosine_similarity(
                embs[a]["mean"].reshape(1, -1), embs[b]["mean"].reshape(1, -1)
            )[0, 0])
            if mean_sim >= 0.80:
                end_sim = float(cosine_similarity(
                    embs[a]["end"].reshape(1, -1), embs[b]["end"].reshape(1, -1)
                )[0, 0])
                sample_pairs.append({
                    "a": a, "b": b,
                    "mean_sim": mean_sim, "end_sim": end_sim,
                    "delta": end_sim - mean_sim,
                })
    sample_pairs.sort(key=lambda x: -abs(x["delta"]))

    convergent = [p for p in sample_pairs if p["delta"] > 0.05]
    divergent = [p for p in sample_pairs if p["delta"] < -0.05]
    neutral = [p for p in sample_pairs if abs(p["delta"]) <= 0.05]

    print(f"Among {len(sample_pairs)} high-similarity pairs (mean >= 0.80):")
    print(f"  Convergent (end_sim > mean_sim + 0.05): {len(convergent)} ({len(convergent)/len(sample_pairs)*100:.0f}%)")
    print(f"  Divergent  (end_sim < mean_sim - 0.05): {len(divergent)} ({len(divergent)/len(sample_pairs)*100:.0f}%)")
    print(f"  Neutral    (|delta| <= 0.05):           {len(neutral)} ({len(neutral)/len(sample_pairs)*100:.0f}%)")

    if convergent:
        print(f"\nMost convergent pairs (started different, ended similar):")
        for p in convergent[:5]:
            ma = meta.get(p["a"], {})
            mb = meta.get(p["b"], {})
            print(f"  delta=+{p['delta']:.3f}  mean={p['mean_sim']:.3f}  end={p['end_sim']:.3f}")
            print(f"    A: {ma.get('title', '?')[:50]}  ({ma.get('date', '')})")
            print(f"    B: {mb.get('title', '?')[:50]}  ({mb.get('date', '')})")

    if divergent:
        print(f"\nMost divergent pairs (same topic, ended differently):")
        for p in divergent[:5]:
            ma = meta.get(p["a"], {})
            mb = meta.get(p["b"], {})
            print(f"  delta={p['delta']:.3f}  mean={p['mean_sim']:.3f}  end={p['end_sim']:.3f}")
            print(f"    A: {ma.get('title', '?')[:50]}  ({ma.get('date', '')})")
            print(f"    B: {mb.get('title', '?')[:50]}  ({mb.get('date', '')})")

    # ============================================================
    # ANALYSIS 3: Intra-conversation drift distribution
    # ============================================================
    print(f"\n=== Drift Distribution ===")
    drifts = [(cid, embs[cid]["drift"], embs[cid]["n_messages"]) for cid in embs]
    drift_vals = [d[1] for d in drifts]
    print(f"  Mean drift:   {np.mean(drift_vals):.4f}")
    print(f"  Median drift: {np.median(drift_vals):.4f}")
    print(f"  Std drift:    {np.std(drift_vals):.4f}")

    # By source
    for source in ["openai", "claude_code", "anthropic"]:
        src_drifts = [d[1] for d in drifts if meta.get(d[0], {}).get("source") == source]
        if src_drifts:
            print(f"  {source:>12s}: mean={np.mean(src_drifts):.4f}, median={np.median(src_drifts):.4f} (n={len(src_drifts)})")

    conn.close()
    print(f"\nDone.")


if __name__ == "__main__":
    main()
