"""Refined trail detection with surprise scoring.

The null model shows that random temporal ordering produces MORE trails
than real data (Z=-4.6), because real conversations are temporally
bursty. This means each real trail is a genuine intellectual return,
not an artifact. We refine trail detection to:

1. Score each link by "surprise": similarity * log(gap_days)
   (longer gaps with high similarity are rarer and more meaningful)

2. Support end-to-start linking via exponential embeddings
   (where conversation A ended is where B began)

3. Report per-trail surprise scores for ranking

Usage:
    python detect_trails.py --db data/analysis.db
    python detect_trails.py --db data/analysis.db --mode end-to-start --alpha 0.85
    python detect_trails.py --db data/analysis.db --mode mean  # original mode
"""
import argparse
import json
import os
import sqlite3
import struct
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def get_conversation_embeddings(conn, model_id, mode="mean", alpha=0.85, uw=3.0, aw=1.0):
    """Get embeddings for trail detection.

    Modes:
        mean: standard weighted role aggregation (original method)
        end-to-start: exponential embeddings, returning (end_emb, start_emb) per conv
        end: just the exponential end embedding
    """
    rows = conn.execute("""
        SELECT me.conversation_id, me.message_id, me.role, me.is_short, mv.embedding
        FROM message_embeddings me
        JOIN message_vectors mv
          ON mv.conversation_id = me.conversation_id
          AND mv.message_id = me.message_id
          AND mv.model_id = me.model_id
        WHERE me.model_id = ?
        ORDER BY me.conversation_id, me.embedded_at
    """, [model_id]).fetchall()

    # Group by conversation, preserve order
    conv_msgs = defaultdict(list)
    for r in rows:
        conv_msgs[r[0]].append({
            "role": r[2],
            "is_short": r[3],
            "vec": np.array(deserialize_f32(r[4])),
        })

    if mode == "mean":
        ids, vecs = [], []
        for cid, msgs in conv_msgs.items():
            role_vecs = defaultdict(list)
            for m in msgs:
                role_vecs[m["role"]].append(m["vec"])
            components, weights = [], []
            if "user" in role_vecs:
                components.append(np.mean(role_vecs["user"], axis=0))
                weights.append(uw)
            if "assistant" in role_vecs:
                components.append(np.mean(role_vecs["assistant"], axis=0))
                weights.append(aw)
            if not components:
                continue
            emb = np.average(components, axis=0, weights=weights)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            ids.append(cid)
            vecs.append(emb)
        return ids, np.array(vecs), None

    elif mode in ("end-to-start", "end"):
        ids, end_vecs, start_vecs = [], [], []
        for cid, msgs in conv_msgs.items():
            non_short = [m for m in msgs if not m["is_short"]]
            if len(non_short) < 3:
                continue
            all_vecs = np.array([m["vec"] for m in non_short])
            n = len(all_vecs)

            # End embedding (recency-biased)
            w_end = np.array([alpha ** (n - 1 - k) for k in range(n)])
            exp_end = np.average(all_vecs, axis=0, weights=w_end)
            norm = np.linalg.norm(exp_end)
            if norm > 0:
                exp_end = exp_end / norm

            # Start embedding (first 20%)
            first_n = max(n // 5, 1)
            first = all_vecs[:first_n]
            if first_n == 1:
                exp_start = first[0]
            else:
                w_s = np.array([alpha ** (first_n - 1 - k) for k in range(first_n)])
                exp_start = np.average(first, axis=0, weights=w_s)
            norm = np.linalg.norm(exp_start)
            if norm > 0:
                exp_start = exp_start / norm

            ids.append(cid)
            end_vecs.append(exp_end)
            start_vecs.append(exp_start)

        return ids, np.array(end_vecs), np.array(start_vecs)


def detect_trails(ids, end_vecs, start_vecs, dates_map,
                  min_gap=14, threshold=0.82, max_gap=365,
                  score_fn="surprise"):
    """Detect trails with surprise scoring.

    If start_vecs is provided, links are end(A) -> start(B).
    Otherwise, links are mean(A) -> mean(B).

    Surprise score per link: similarity * log(1 + gap_days)
    Trail score: mean surprise across all links.
    """
    dated = []
    for cid in ids:
        d = dates_map.get(cid)
        if d:
            dated.append((cid, d))
    dated.sort(key=lambda x: x[1])
    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    # Precompute the right similarity matrix
    if start_vecs is not None:
        # end(A) vs start(B): not a square symmetric matrix
        # We compute on-the-fly per pair
        use_end_start = True
    else:
        # mean vs mean: precompute full matrix
        sim = cosine_similarity(end_vecs)
        use_end_start = False

    # Build forward links
    trails_graph = {}  # cid -> (next_cid, similarity, gap, surprise)
    for i, (cid_a, dt_a) in enumerate(dated):
        idx_a = id_to_idx.get(cid_a)
        if idx_a is None:
            continue
        best = None
        for j in range(i + 1, len(dated)):
            cid_b, dt_b = dated[j]
            gap = (dt_b - dt_a).days
            if gap < min_gap:
                continue
            if gap > max_gap:
                break
            idx_b = id_to_idx.get(cid_b)
            if idx_b is None:
                continue

            if use_end_start:
                s = float(cosine_similarity(
                    end_vecs[idx_a].reshape(1, -1),
                    start_vecs[idx_b].reshape(1, -1)
                )[0, 0])
            else:
                s = sim[idx_a, idx_b]

            if s >= threshold:
                surprise = s * np.log1p(gap)
                if best is None or surprise > best[3]:
                    best = (cid_b, s, gap, surprise)

        if best:
            trails_graph[cid_a] = best

    # Follow chains
    visited = set()
    trails = []
    for start_cid in [d[0] for d in dated]:
        if start_cid in visited:
            continue
        chain = [(start_cid, None, None, None)]
        visited.add(start_cid)
        current = start_cid
        while current in trails_graph:
            next_cid, s, gap, surprise = trails_graph[current]
            if next_cid in visited:
                break
            chain.append((next_cid, s, gap, surprise))
            visited.add(next_cid)
            current = next_cid
        if len(chain) >= 3:
            # Trail-level surprise: mean of link surprises
            link_surprises = [step[3] for step in chain[1:] if step[3] is not None]
            trail_surprise = np.mean(link_surprises) if link_surprises else 0
            total_gap = sum(step[2] for step in chain[1:] if step[2] is not None)
            trails.append({
                "chain": chain,
                "surprise": float(trail_surprise),
                "total_gap_days": total_gap,
                "length": len(chain),
            })

    trails.sort(key=lambda x: -x["surprise"])
    return trails, trails_graph


def main():
    import sqlite_vec

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/analysis.db")
    parser.add_argument("--mode", default="both", choices=["mean", "end-to-start", "both"])
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--min-gap", type=int, default=14)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    model_id = conn.execute("SELECT id FROM embedding_models").fetchone()[0]

    meta = {}
    dates_map = {}
    for r in conn.execute(
        "SELECT id, title, source, message_count, created_at, has_notes FROM conversations"
    ).fetchall():
        meta[r[0]] = {
            "title": r[1], "source": r[2], "msgs": r[3],
            "date": (r[4] or "")[:10], "notes": bool(r[5]),
        }
        if r[4]:
            try:
                dates_map[r[0]] = datetime.strptime(r[4][:10], "%Y-%m-%d")
            except:
                pass

    def show_trails(trails, label, top_n):
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"  {len(trails)} trails found")
        print(f"{'='*70}")
        for t_idx, trail in enumerate(trails[:top_n]):
            chain = trail["chain"]
            print(f"\nTrail {t_idx+1} (len={trail['length']}, "
                  f"surprise={trail['surprise']:.2f}, "
                  f"span={trail['total_gap_days']}d):")
            for step, (cid, s, gap, surprise) in enumerate(chain):
                md = meta.get(cid, {})
                ann = " [notes]" if md.get("notes") else ""
                if s is not None:
                    link = f"  (sim={s:.3f} gap={gap}d surprise={surprise:.2f})"
                else:
                    link = ""
                print(f"  {step+1:>2d}. {md.get('date',''):10s}  "
                      f"{md.get('title','?')[:50]:<50s}  "
                      f"{md.get('source',''):>12s}{ann}{link}")

    modes = [args.mode] if args.mode != "both" else ["mean", "end-to-start"]

    all_results = {}
    for mode in modes:
        print(f"\nComputing embeddings (mode={mode})...")
        ids, end_vecs, start_vecs = get_conversation_embeddings(
            conn, model_id, mode=mode, alpha=args.alpha
        )
        print(f"  {len(ids)} conversations")

        trails, tg = detect_trails(
            ids, end_vecs, start_vecs, dates_map,
            min_gap=args.min_gap, threshold=args.threshold
        )

        show_trails(trails, f"Mode: {mode}", args.top)

        # Compare with the other mode: which trails are unique to each?
        trail_members = set()
        for t in trails:
            for step in t["chain"]:
                trail_members.add(step[0])
        all_results[mode] = {
            "trails": trails,
            "members": trail_members,
            "count": len(trails),
        }

    # If both modes, compare
    if len(all_results) == 2:
        mean_members = all_results["mean"]["members"]
        ets_members = all_results["end-to-start"]["members"]
        shared = mean_members & ets_members
        mean_only = mean_members - ets_members
        ets_only = ets_members - mean_members

        print(f"\n{'='*70}")
        print(f"  Comparison: mean vs end-to-start")
        print(f"{'='*70}")
        print(f"  Mean trails:          {all_results['mean']['count']}")
        print(f"  End-to-start trails:  {all_results['end-to-start']['count']}")
        print(f"  Shared members:       {len(shared)}")
        print(f"  Mean-only members:    {len(mean_only)}")
        print(f"  End-to-start-only:    {len(ets_only)}")
        print(f"  Jaccard similarity:   {len(shared) / len(shared | mean_only | ets_only):.3f}")

    conn.close()


if __name__ == "__main__":
    main()
