"""Experiment 3: Strengthening experiments.

a) Episode density by platform
b) Session continuation markers as episode boundary ground truth
c) DTW trajectory comparison: same path, different topic
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
from memex.db import Database
from memex.config import load_config

ANALYSIS_DB = os.path.expanduser(
    "~/github/papers/cognitive-mri-ai-conversations/operational-memex/data/analysis.db"
)


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def load_all():
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

    return conv_msgs, meta, conn


def main():
    print("Loading data...")
    conv_msgs, meta, analysis_conn = load_all()

    # Open memex DBs for message text
    config = load_config(os.path.expanduser("~/.memex/config.yaml"))
    memex_dbs = {}
    for db_name, db_config in config.get("databases", {}).items():
        db_dir = os.path.expanduser(db_config["path"])
        db_file = os.path.join(db_dir, "conversations.db")
        if os.path.exists(db_file):
            memex_dbs[db_name] = Database(db_dir, readonly=True)

    # ============================================================
    # EXPERIMENT 3a: Episode density by platform
    # ============================================================
    print("\n=== 3a: Episode Density by Platform ===")

    platform_results = defaultdict(list)
    for cid, msgs in conv_msgs.items():
        non_short = [m for m in msgs if not m["short"]]
        if len(non_short) < 20:
            continue
        vecs = np.array([m["vec"] for m in non_short])
        segs = ef.auto_segment(vecs, threshold="window", min_segment_size=5,
                               window_size=10, sensitivity=0.8)
        source = meta.get(cid, {}).get("source", "?")
        platform_results[source].append({
            "n_msgs": len(non_short),
            "n_segs": len(segs),
            "density": len(segs) / len(non_short),
        })

    print(f"\n{'Platform':>12s} {'Convs':>6s} {'Mean segs':>10s} {'Mean msgs':>10s} {'Msgs/ep':>8s}")
    print("-" * 55)
    for source in ["openai", "claude_code", "anthropic"]:
        if source not in platform_results:
            continue
        data = platform_results[source]
        mean_segs = np.mean([d["n_segs"] for d in data])
        mean_msgs = np.mean([d["n_msgs"] for d in data])
        msgs_per_ep = mean_msgs / mean_segs if mean_segs > 0 else 0
        print(f"{source:>12s} {len(data):>6d} {mean_segs:>10.1f} {mean_msgs:>10.0f} {msgs_per_ep:>8.1f}")

    # By length bin and platform
    print(f"\nEpisode density by length and platform:")
    print(f"{'Length':>10s} {'ChatGPT msgs/ep':>16s} {'CC msgs/ep':>12s}")
    print("-" * 45)
    bins = [(20, 50), (51, 100), (101, 200), (201, 500), (500, 99999)]
    for lo, hi in bins:
        for source in ["openai", "claude_code"]:
            data = [d for d in platform_results.get(source, []) if lo <= d["n_msgs"] <= hi]
            if data:
                mean_mpe = np.mean([d["n_msgs"] / d["n_segs"] for d in data])
            else:
                mean_mpe = 0
        gpt = [d for d in platform_results.get("openai", []) if lo <= d["n_msgs"] <= hi]
        cc = [d for d in platform_results.get("claude_code", []) if lo <= d["n_msgs"] <= hi]
        gpt_mpe = np.mean([d["n_msgs"] / d["n_segs"] for d in gpt]) if gpt else 0
        cc_mpe = np.mean([d["n_msgs"] / d["n_segs"] for d in cc]) if cc else 0
        label = f"{lo}-{hi}" if hi < 99999 else f"{lo}+"
        print(f"{label:>10s} {gpt_mpe:>16.1f} {cc_mpe:>12.1f}")

    # ============================================================
    # EXPERIMENT 3b: Session continuation markers as ground truth
    # ============================================================
    print("\n=== 3b: Session Continuation Markers vs Episode Boundaries ===")

    continuation_phrase = "This session is being continued from a previous conversation"
    hits = 0
    total_checked = 0
    boundary_distances = []

    for cid, msgs in conv_msgs.items():
        non_short = [m for m in msgs if not m["short"]]
        if len(non_short) < 20:
            continue
        source = meta.get(cid, {}).get("source", "")
        if source != "claude_code":
            continue

        vecs = np.array([m["vec"] for m in non_short])
        segs = ef.auto_segment(vecs, threshold="window", min_segment_size=5,
                               window_size=10, sensitivity=0.8)
        seg_boundaries = set()
        for seg in segs:
            seg_boundaries.add(seg["start"])

        # Find continuation markers in message text
        for i, m in enumerate(non_short):
            if m["role"] != "user":
                continue
            # Check text via memex DB
            text = ""
            for db in memex_dbs.values():
                row = db.conn.execute(
                    "SELECT content FROM messages WHERE conversation_id = ? AND id = ?",
                    [cid, m["msg_id"]]
                ).fetchone()
                if row and row["content"]:
                    try:
                        blocks = json.loads(row["content"])
                        for b in blocks:
                            if isinstance(b, dict) and b.get("type") == "text":
                                text = b.get("text", "")
                                break
                    except:
                        pass
                    break

            if continuation_phrase.lower() in text.lower():
                total_checked += 1
                # How close is the nearest episode boundary?
                if seg_boundaries:
                    min_dist = min(abs(i - b) for b in seg_boundaries)
                    boundary_distances.append(min_dist)
                    if min_dist <= 3:
                        hits += 1

    if total_checked > 0:
        print(f"  Continuation markers found: {total_checked}")
        print(f"  Within 3 messages of episode boundary: {hits} ({hits/total_checked*100:.0f}%)")
        print(f"  Mean distance to nearest boundary: {np.mean(boundary_distances):.1f} messages")
        print(f"  Median distance: {np.median(boundary_distances):.0f} messages")
    else:
        print("  No continuation markers found")

    # ============================================================
    # EXPERIMENT 3c: DTW trajectory comparison
    # ============================================================
    print("\n=== 3c: Trajectory Similarity (DTW) ===")
    print("Finding conversations with similar trajectories but different topics...")

    # Sample 300 conversations, compute trajectories
    valid = {cid: msgs for cid, msgs in conv_msgs.items()
             if len([m for m in msgs if not m["short"]]) >= 15}
    sample_cids = sorted(valid.keys())[:300]

    lens = ef.Exponential(0.85)
    trajectories = {}
    mean_embs = {}
    for cid in sample_cids:
        vecs = np.array([m["vec"] for m in valid[cid] if not m["short"]])
        trajectories[cid] = lens.trajectory(vecs)
        mean_embs[cid] = ef.Uniform().project(vecs)

    # Find pairs with high DTW similarity but low mean similarity
    print("Computing DTW distances (300 x 300, sampling 2000 pairs)...")
    import random
    random.seed(42)
    pairs = []
    all_pairs = [(i, j) for i in range(len(sample_cids)) for j in range(i + 1, len(sample_cids))]
    sampled = random.sample(all_pairs, min(2000, len(all_pairs)))

    for i, j in sampled:
        a, b = sample_cids[i], sample_cids[j]
        mean_sim = float(cosine_similarity(
            mean_embs[a].reshape(1, -1), mean_embs[b].reshape(1, -1)
        )[0, 0])
        dtw_dist = ef.trajectory_distance(trajectories[a], trajectories[b], method="dtw")
        pairs.append({
            "a": a, "b": b,
            "mean_sim": mean_sim,
            "dtw_dist": dtw_dist,
        })

    # Pairs where DTW is low (similar trajectory) but mean sim is also low (different topic)
    # Normalize: dtw_dist is a distance, mean_sim is a similarity
    pairs.sort(key=lambda p: p["dtw_dist"])

    print("\nMost similar trajectories (lowest DTW distance):")
    for p in pairs[:10]:
        ma = meta.get(p["a"], {})
        mb = meta.get(p["b"], {})
        print(f"  dtw={p['dtw_dist']:.4f}  mean_sim={p['mean_sim']:.3f}  "
              f"{ma.get('title', '?')[:30]:<30s} ({ma.get('source', '')}) <-> "
              f"{mb.get('title', '?')[:30]:<30s} ({mb.get('source', '')})")

    # Specifically: low DTW (similar path) AND low mean_sim (different topic)
    diff_topic_same_path = [p for p in pairs if p["mean_sim"] < 0.5 and p["dtw_dist"] < np.percentile([pp["dtw_dist"] for pp in pairs], 20)]
    diff_topic_same_path.sort(key=lambda p: p["dtw_dist"])
    print(f"\nSame path, different topic (dtw < p20, mean_sim < 0.5): {len(diff_topic_same_path)} pairs")
    for p in diff_topic_same_path[:10]:
        ma = meta.get(p["a"], {})
        mb = meta.get(p["b"], {})
        print(f"  dtw={p['dtw_dist']:.4f}  mean_sim={p['mean_sim']:.3f}  "
              f"{ma.get('title', '?')[:30]:<30s} ({ma.get('source', '')}) <-> "
              f"{mb.get('title', '?')[:30]:<30s} ({mb.get('source', '')})")

    # Correlation between DTW distance and mean similarity
    from scipy.stats import spearmanr
    dtw_vals = [p["dtw_dist"] for p in pairs]
    sim_vals = [p["mean_sim"] for p in pairs]
    rho, p_val = spearmanr(dtw_vals, sim_vals)
    print(f"\nCorrelation (DTW dist vs mean sim): Spearman rho={rho:.3f}, p={p_val:.4f}")
    print(f"  (Negative rho = similar topics have similar trajectories, expected)")
    print(f"  (Weak rho = trajectories are partially independent of topic)")

    # Save
    os.makedirs("experiments/results", exist_ok=True)
    results = {
        "episode_density_by_platform": {
            source: {
                "mean_segments": float(np.mean([d["n_segs"] for d in data])),
                "mean_msgs": float(np.mean([d["n_msgs"] for d in data])),
                "msgs_per_episode": float(np.mean([d["n_msgs"]/d["n_segs"] for d in data])),
                "n_conversations": len(data),
            }
            for source, data in platform_results.items()
        },
        "continuation_validation": {
            "total_markers": total_checked,
            "within_3_msgs": hits,
            "precision": hits / total_checked if total_checked > 0 else None,
            "mean_distance": float(np.mean(boundary_distances)) if boundary_distances else None,
        },
        "dtw_correlation": {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "n_pairs": len(pairs),
        },
    }
    with open("experiments/results/exp03_strengthen.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/results/exp03_strengthen.json")

    for db in memex_dbs.values():
        db.close()
    analysis_conn.close()


if __name__ == "__main__":
    main()
