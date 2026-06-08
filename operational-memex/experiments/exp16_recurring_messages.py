"""Experiment 16: Recurring messages (idea-level patterns).

Instead of conversation-level trails, look for recurring *messages*:
specific ideas, questions, or insights that appear across different
conversations. These are the atomic units of intellectual recurrence.

Clusters message embeddings to find recurring semantic patterns,
then traces which conversations contain instances of each pattern.
"""
import json
import os
import sqlite3
import struct
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def main():
    import sqlite_vec

    conn = sqlite3.connect("data/analysis.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    model_id = conn.execute("SELECT id FROM embedding_models").fetchone()[0]

    meta = {}
    for r in conn.execute("SELECT id, title, source, created_at, has_notes FROM conversations").fetchall():
        meta[r[0]] = {"title": r[1], "source": r[2], "date": (r[3] or "")[:10], "notes": bool(r[4])}

    # Load user messages only (user ideas are what recur)
    print("Loading user message embeddings...")
    rows = conn.execute("""
        SELECT me.conversation_id, me.message_id, me.char_count, mv.embedding
        FROM message_embeddings me
        JOIN message_vectors mv ON mv.conversation_id = me.conversation_id
          AND mv.message_id = me.message_id AND mv.model_id = me.model_id
        WHERE me.model_id = ? AND me.role = 'user' AND me.is_short = 0
        ORDER BY me.conversation_id
    """, [model_id]).fetchall()

    msg_data = []
    msg_vecs = []
    for r in rows:
        msg_data.append({"conv_id": r[0], "msg_id": r[1], "chars": r[2]})
        msg_vecs.append(deserialize_f32(r[3]))

    msg_vecs = np.array(msg_vecs)
    print(f"  {len(msg_data)} user messages (non-short)")

    # Cluster messages to find recurring semantic patterns
    # Use MiniBatchKMeans for speed
    n_clusters = 200  # aiming for ~150-250 semantic "ideas"
    print(f"\nClustering into {n_clusters} semantic clusters...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=3)
    labels = kmeans.fit_predict(msg_vecs)

    # Analyze clusters
    cluster_info = []
    for k in range(n_clusters):
        members = [i for i, l in enumerate(labels) if l == k]
        if not members:
            continue

        # Which conversations contain this cluster?
        conv_ids = set(msg_data[i]["conv_id"] for i in members)
        # How many unique conversations?
        n_convs = len(conv_ids)
        # Temporal span
        dates = sorted(meta.get(cid, {}).get("date", "") for cid in conv_ids if meta.get(cid, {}).get("date"))
        sources = defaultdict(int)
        for cid in conv_ids:
            sources[meta.get(cid, {}).get("source", "?")] += 1

        # Get representative message text (closest to centroid)
        centroid = kmeans.cluster_centers_[k]
        cluster_vecs = msg_vecs[members]
        dists = 1 - cosine_similarity(centroid.reshape(1, -1), cluster_vecs)[0]
        closest_idx = members[np.argmin(dists)]

        # Use the representative message's conversation title as proxy
        # (actual message text is in the memex DB, not analysis DB)
        rep_msg = msg_data[closest_idx]
        rep_title = meta.get(rep_msg["conv_id"], {}).get("title", "?")
        snippet = f"[from: {rep_title}] ({rep_msg['chars']} chars)"

        # Intra-cluster similarity (cohesion)
        if len(members) >= 3:
            sample = members[:min(50, len(members))]
            sample_vecs = msg_vecs[sample]
            sims = cosine_similarity(sample_vecs)
            np.fill_diagonal(sims, 0)
            cohesion = float(np.mean(sims))
        else:
            cohesion = 1.0

        cluster_info.append({
            "cluster_id": k,
            "n_messages": len(members),
            "n_conversations": n_convs,
            "cohesion": round(cohesion, 4),
            "date_range": f"{dates[0]} to {dates[-1]}" if dates else "",
            "span_days": (datetime.strptime(dates[-1], "%Y-%m-%d") - datetime.strptime(dates[0], "%Y-%m-%d")).days if len(dates) >= 2 else 0,
            "sources": dict(sources),
            "cross_source": len(sources) > 1,
            "representative": snippet[:150],
        })

    # Sort by number of conversations (recurring ideas span many conversations)
    cluster_info.sort(key=lambda x: -x["n_conversations"])

    # Report
    print(f"\n=== Most Recurring Ideas (by conversation spread) ===\n")
    for i, c in enumerate(cluster_info[:30]):
        cross = " [CROSS-SOURCE]" if c["cross_source"] else ""
        print(f"  Cluster {c['cluster_id']:>3d}: {c['n_conversations']:>4d} convs, "
              f"{c['n_messages']:>5d} msgs, cohesion={c['cohesion']:.3f}, "
              f"span={c['span_days']:>4d}d{cross}")
        print(f"    Sources: {c['sources']}")
        print(f"    Range: {c['date_range']}")
        print(f'    Rep: "{c["representative"][:100]}..."')
        print()

    # How many clusters span 2+ years?
    long_lived = [c for c in cluster_info if c["span_days"] >= 730]
    print(f"\n--- Long-lived ideas (span >= 2 years): {len(long_lived)} ---")
    for c in long_lived[:10]:
        print(f"  Cluster {c['cluster_id']}: {c['n_conversations']} convs, "
              f"span={c['span_days']}d, cohesion={c['cohesion']:.3f}")
        print(f'    "{c["representative"][:100]}"')

    # Cross-source recurring ideas
    cross_source = [c for c in cluster_info if c["cross_source"] and c["n_conversations"] >= 5]
    print(f"\n--- Cross-source recurring ideas (>= 5 convs, 2+ sources): {len(cross_source)} ---")
    for c in cross_source[:10]:
        print(f"  Cluster {c['cluster_id']}: {c['n_conversations']} convs, sources={c['sources']}")
        print(f'    "{c["representative"][:100]}"')

    # Do annotated conversations contain more diverse clusters?
    annotated_cids = set(cid for cid in meta if meta[cid].get("notes"))
    if annotated_cids:
        ann_clusters = set()
        non_clusters = set()
        for i, d in enumerate(msg_data):
            if d["conv_id"] in annotated_cids:
                ann_clusters.add(labels[i])
            else:
                non_clusters.add(labels[i])

        ann_msg_count = sum(1 for d in msg_data if d["conv_id"] in annotated_cids)
        non_msg_count = len(msg_data) - ann_msg_count

        print(f"\n--- Annotated vs Non-Annotated Cluster Diversity ---")
        print(f"  Annotated: {len(ann_clusters)} unique clusters across {ann_msg_count} messages")
        print(f"  Non-annotated: {len(non_clusters)} unique clusters across {non_msg_count} messages")
        print(f"  Annotated cluster density: {len(ann_clusters)/ann_msg_count:.4f} clusters/msg" if ann_msg_count else "")
        print(f"  Non-annotated cluster density: {len(non_clusters)/non_msg_count:.4f} clusters/msg" if non_msg_count else "")

    # Save
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/recurring_messages.json", "w") as f:
        json.dump(cluster_info[:50], f, indent=2, default=str)
    print(f"\nSaved top 50 clusters to experiments/results/recurring_messages.json")
    conn.close()


if __name__ == "__main__":
    main()
