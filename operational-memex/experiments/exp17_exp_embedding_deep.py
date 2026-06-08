"""Experiment 17: Deep investigation of exponential embeddings.

Explores several novel angles:
1. Trajectory similarity: do two conversations TRAVEL the same path
   through embedding space, even if they start/end differently?
2. Message-level exponential curves: plot the embedding trajectory
   for selected conversations to visualize drift patterns.
3. Role-specific exponential embeddings: separate user and assistant
   trajectory. Does the user's thinking evolve differently from the
   assistant's responses?
4. Adaptive alpha: estimate the optimal alpha per conversation based
   on how quickly the topic actually shifts.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


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


def exp_trajectory(vecs, alpha):
    """Compute the full exponential trajectory (embedding at each step)."""
    trajectory = []
    for j in range(len(vecs)):
        weights = np.array([alpha ** (j - k) for k in range(j + 1)])
        emb = np.average(vecs[:j + 1], axis=0, weights=weights)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        trajectory.append(emb)
    return np.array(trajectory)


def trajectory_similarity(traj_a, traj_b):
    """Compare two trajectories by aligning them (resample to same length)
    and computing mean cosine similarity of aligned points."""
    n_a, n_b = len(traj_a), len(traj_b)
    # Resample both to 20 evenly spaced points
    n_pts = 20
    idx_a = np.linspace(0, n_a - 1, n_pts).astype(int)
    idx_b = np.linspace(0, n_b - 1, n_pts).astype(int)
    aligned_a = traj_a[idx_a]
    aligned_b = traj_b[idx_b]

    sims = [float(cosine_similarity(aligned_a[i:i+1], aligned_b[i:i+1])[0, 0])
            for i in range(n_pts)]
    return np.mean(sims), sims


def estimate_adaptive_alpha(vecs):
    """Estimate the natural decay rate of a conversation.

    Fits alpha to minimize reconstruction error: at each step j,
    the exponential embedding should predict the next message.
    Returns the alpha that best predicts the conversation's evolution.
    """
    if len(vecs) < 5:
        return 0.85  # default

    best_alpha = 0.85
    best_score = -1

    for alpha in np.arange(0.3, 0.99, 0.05):
        traj = exp_trajectory(vecs, alpha)
        # Score: how well does traj[j] predict vecs[j+1]?
        pred_sims = []
        for j in range(len(vecs) - 1):
            s = float(cosine_similarity(traj[j:j+1], vecs[j+1:j+2])[0, 0])
            pred_sims.append(s)
        score = np.mean(pred_sims)
        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha


def main():
    import sqlite_vec

    conn = sqlite3.connect("data/analysis.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    model_id = conn.execute("SELECT id FROM embedding_models").fetchone()[0]

    meta = {}
    for r in conn.execute("SELECT id, title, source, message_count, created_at, has_notes FROM conversations").fetchall():
        meta[r[0]] = {"title": r[1], "source": r[2], "msgs": r[3],
                       "date": (r[4] or "")[:10], "notes": bool(r[5])}

    print("Loading messages...")
    conv_msgs = load_messages(conn, model_id)
    print(f"  {len(conv_msgs)} conversations")

    # Filter to conversations with enough messages
    valid = {cid: msgs for cid, msgs in conv_msgs.items()
             if len([m for m in msgs if not m["short"]]) >= 10}
    print(f"  {len(valid)} with >= 10 non-short messages")

    alpha = 0.85

    # ============================================================
    # ANALYSIS 1: Trajectory similarity between conversations
    # ============================================================
    print(f"\n=== ANALYSIS 1: Trajectory Similarity ===")
    print("Do conversations travel the same PATH through embedding space?")

    # Sample 500 conversations, compute trajectories
    sample_cids = sorted(valid.keys())[:500]
    trajectories = {}
    for cid in sample_cids:
        vecs = np.array([m["vec"] for m in valid[cid] if not m["short"]])
        trajectories[cid] = exp_trajectory(vecs, alpha)

    # Find pairs with high trajectory similarity but low mean similarity
    print("\nComputing trajectory vs mean similarities (500 x 500)...")
    mean_embs = {}
    for cid in sample_cids:
        vecs = np.array([m["vec"] for m in valid[cid] if not m["short"]])
        m = np.mean(vecs, axis=0)
        norm = np.linalg.norm(m)
        if norm > 0: m /= norm
        mean_embs[cid] = m

    interesting_pairs = []
    for i in range(len(sample_cids)):
        for j in range(i + 1, len(sample_cids)):
            a, b = sample_cids[i], sample_cids[j]
            mean_sim = float(cosine_similarity(mean_embs[a].reshape(1, -1), mean_embs[b].reshape(1, -1))[0, 0])
            traj_sim, _ = trajectory_similarity(trajectories[a], trajectories[b])
            delta = traj_sim - mean_sim
            if abs(delta) > 0.1:
                interesting_pairs.append({
                    "a": a, "b": b, "mean_sim": mean_sim,
                    "traj_sim": traj_sim, "delta": delta,
                })

    # Pairs where trajectory sim >> mean sim (same journey, different topic)
    interesting_pairs.sort(key=lambda x: -x["delta"])
    print(f"\nPairs where trajectory similarity >> mean similarity:")
    print("(Same journey through embedding space, different average topic)")
    for p in interesting_pairs[:10]:
        ma, mb = meta.get(p["a"], {}), meta.get(p["b"], {})
        print(f"  traj={p['traj_sim']:.3f}  mean={p['mean_sim']:.3f}  delta=+{p['delta']:.3f}")
        print(f"    A: {ma.get('title', '?')[:50]}  ({ma.get('date', '')} {ma.get('source', '')})")
        print(f"    B: {mb.get('title', '?')[:50]}  ({mb.get('date', '')} {mb.get('source', '')})")

    # Pairs where mean sim >> trajectory sim (same topic, different journey)
    interesting_pairs.sort(key=lambda x: x["delta"])
    print(f"\nPairs where mean similarity >> trajectory similarity:")
    print("(Same average topic, but took very different paths)")
    for p in interesting_pairs[:10]:
        ma, mb = meta.get(p["a"], {}), meta.get(p["b"], {})
        print(f"  traj={p['traj_sim']:.3f}  mean={p['mean_sim']:.3f}  delta={p['delta']:.3f}")
        print(f"    A: {ma.get('title', '?')[:50]}  ({ma.get('date', '')} {ma.get('source', '')})")
        print(f"    B: {mb.get('title', '?')[:50]}  ({mb.get('date', '')} {mb.get('source', '')})")

    # ============================================================
    # ANALYSIS 2: Role-specific trajectories
    # ============================================================
    print(f"\n=== ANALYSIS 2: User vs Assistant Trajectory Divergence ===")

    role_divergences = []
    for cid in sorted(valid.keys()):
        msgs = valid[cid]
        user_vecs = [m["vec"] for m in msgs if m["role"] == "user" and not m["short"]]
        asst_vecs = [m["vec"] for m in msgs if m["role"] == "assistant" and not m["short"]]
        if len(user_vecs) < 5 or len(asst_vecs) < 5:
            continue

        user_traj = exp_trajectory(np.array(user_vecs), alpha)
        asst_traj = exp_trajectory(np.array(asst_vecs), alpha)

        # Compare final user embedding vs final assistant embedding
        user_end = user_traj[-1]
        asst_end = asst_traj[-1]
        divergence = 1 - float(cosine_similarity(user_end.reshape(1, -1), asst_end.reshape(1, -1))[0, 0])

        role_divergences.append({
            "cid": cid, "divergence": divergence,
            "n_user": len(user_vecs), "n_asst": len(asst_vecs),
        })

    role_divergences.sort(key=lambda x: -x["divergence"])
    divs = [r["divergence"] for r in role_divergences]
    print(f"  {len(role_divergences)} conversations analyzed")
    print(f"  Mean user/assistant divergence: {np.mean(divs):.4f}")
    print(f"  Median: {np.median(divs):.4f}")

    print(f"\nHighest user/assistant divergence (user and AI ended up in different places):")
    for r in role_divergences[:10]:
        md = meta.get(r["cid"], {})
        print(f"  div={r['divergence']:.3f}  {md.get('title', '?')[:50]}  ({md.get('date', '')} {md.get('source', '')})")

    print(f"\nLowest divergence (user and AI stayed aligned throughout):")
    for r in role_divergences[-10:]:
        md = meta.get(r["cid"], {})
        print(f"  div={r['divergence']:.3f}  {md.get('title', '?')[:50]}  ({md.get('date', '')} {md.get('source', '')})")

    # By source
    for source in ["openai", "claude_code", "anthropic"]:
        src_divs = [r["divergence"] for r in role_divergences
                    if meta.get(r["cid"], {}).get("source") == source]
        if src_divs:
            print(f"  {source:>12s}: mean divergence={np.mean(src_divs):.4f} (n={len(src_divs)})")

    # ============================================================
    # ANALYSIS 3: Adaptive alpha estimation
    # ============================================================
    print(f"\n=== ANALYSIS 3: Adaptive Alpha (Per-Conversation Decay Rate) ===")

    # Sample 300 conversations
    sample = sorted(valid.keys())[:300]
    adaptive_alphas = []
    for cid in sample:
        vecs = np.array([m["vec"] for m in valid[cid] if not m["short"]])
        est_alpha = estimate_adaptive_alpha(vecs)
        adaptive_alphas.append({
            "cid": cid, "alpha": est_alpha,
            "n_msgs": len(vecs),
            "source": meta.get(cid, {}).get("source", "?"),
        })

    alphas_vals = [a["alpha"] for a in adaptive_alphas]
    print(f"  {len(adaptive_alphas)} conversations analyzed")
    print(f"  Mean optimal alpha: {np.mean(alphas_vals):.3f}")
    print(f"  Median: {np.median(alphas_vals):.3f}")
    print(f"  Std: {np.std(alphas_vals):.3f}")

    # Distribution
    from collections import Counter
    alpha_counts = Counter(round(a, 2) for a in alphas_vals)
    print(f"\n  Alpha distribution:")
    for a in sorted(alpha_counts.keys()):
        bar = "#" * alpha_counts[a]
        print(f"    {a:.2f}: {bar} ({alpha_counts[a]})")

    # By source
    for source in ["openai", "claude_code"]:
        src_alphas = [a["alpha"] for a in adaptive_alphas if a["source"] == source]
        if src_alphas:
            print(f"\n  {source}: mean alpha={np.mean(src_alphas):.3f}, median={np.median(src_alphas):.3f} (n={len(src_alphas)})")

    # Correlation with conversation length
    lengths = [a["n_msgs"] for a in adaptive_alphas]
    corr = np.corrcoef(alphas_vals, lengths)[0, 1]
    print(f"\n  Correlation (alpha vs msg count): r={corr:.3f}")
    print(f"  (Positive = longer conversations need longer memory)")

    # Save
    os.makedirs("experiments/results", exist_ok=True)
    results = {
        "trajectory_pairs_found": len(interesting_pairs),
        "role_divergence_mean": float(np.mean(divs)),
        "role_divergence_median": float(np.median(divs)),
        "adaptive_alpha_mean": float(np.mean(alphas_vals)),
        "adaptive_alpha_median": float(np.median(alphas_vals)),
        "alpha_length_correlation": float(corr),
    }
    with open("experiments/results/exp_embedding_deep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/results/exp_embedding_deep.json")
    conn.close()


if __name__ == "__main__":
    main()
