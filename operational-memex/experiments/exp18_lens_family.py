"""Experiment 18: Conversation lens family.

Compares different weighting schemes for aggregating message embeddings.
Each "lens" reveals different structure in the same conversation data.

Lenses implemented:
1. Mean (uniform weights, baseline)
2. Exponential (recency, alpha=0.85)
3. Reverse exponential (primacy, alpha=0.85)
4. Surprise (weight = deviation from running mean)
5. Gaussian focal (peaked at a focal point k0)
6. First message only (extreme primacy)
7. Last message only (extreme recency)

For each lens, we compute conversation-level embeddings and measure:
- How different is this lens from the mean lens? (divergence)
- How many trail links does this lens find?
- Do annotated conversations look different under this lens?
"""
import json
import os
import sqlite3
import struct
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.stats import mannwhitneyu
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
        SELECT me.conversation_id, me.role, me.is_short, mv.embedding
        FROM message_embeddings me
        JOIN message_vectors mv ON mv.conversation_id = me.conversation_id
          AND mv.message_id = me.message_id AND mv.model_id = me.model_id
        WHERE me.model_id = ?
        ORDER BY me.conversation_id, me.embedded_at
    """, [model_id]).fetchall()
    conv_msgs = defaultdict(list)
    for r in rows:
        conv_msgs[r[0]].append({"role": r[1], "short": r[2], "vec": np.array(deserialize_f32(r[3]))})
    return conv_msgs


def apply_lens(vecs, lens_name, **params):
    """Apply a weighting lens to a sequence of vectors."""
    n = len(vecs)
    if n == 0:
        return None

    alpha = params.get("alpha", 0.85)

    if lens_name == "mean":
        weights = np.ones(n)

    elif lens_name == "exponential":
        weights = np.array([alpha ** (n - 1 - k) for k in range(n)])

    elif lens_name == "reverse_exponential":
        weights = np.array([alpha ** k for k in range(n)])

    elif lens_name == "surprise":
        # Weight = cosine distance from running mean
        weights = np.ones(n)
        running_sum = np.zeros_like(vecs[0])
        for k in range(n):
            if k > 0:
                running_mean = running_sum / k
                rm_norm = np.linalg.norm(running_mean)
                if rm_norm > 0:
                    running_mean = running_mean / rm_norm
                    sim = float(cosine_similarity(
                        vecs[k].reshape(1, -1), running_mean.reshape(1, -1)
                    )[0, 0])
                    weights[k] = max(1 - sim, 0.01)  # surprise = 1 - similarity
            running_sum = running_sum + vecs[k]

    elif lens_name == "gaussian":
        k0 = params.get("k0", n // 2)  # focal point
        sigma = params.get("sigma", max(n / 6, 1))
        weights = np.array([np.exp(-((k - k0) ** 2) / (2 * sigma ** 2)) for k in range(n)])

    elif lens_name == "first_only":
        weights = np.zeros(n)
        weights[0] = 1.0

    elif lens_name == "last_only":
        weights = np.zeros(n)
        weights[-1] = 1.0

    elif lens_name == "bookend":
        # First 10% and last 10% weighted, middle ignored
        first_n = max(n // 10, 1)
        last_n = max(n // 10, 1)
        weights = np.zeros(n)
        weights[:first_n] = 1.0
        weights[-last_n:] = 1.0

    else:
        weights = np.ones(n)

    emb = np.average(vecs, axis=0, weights=weights)
    return normalize(emb)


def count_trail_links(ids, embs, dates_map, threshold=0.82, min_gap=14, max_gap=365):
    """Count forward links at a given threshold."""
    dated = sorted([(c, dates_map[c]) for c in ids if c in dates_map], key=lambda x: x[1])
    id_to_idx = {c: i for i, c in enumerate(ids)}

    link_count = 0
    for i, (cid_a, dt_a) in enumerate(dated):
        idx_a = id_to_idx.get(cid_a)
        if idx_a is None:
            continue
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
            s = float(cosine_similarity(
                embs[idx_a].reshape(1, -1), embs[idx_b].reshape(1, -1)
            )[0, 0])
            if s >= threshold:
                link_count += 1
                break  # count one per source for speed
    return link_count


def main():
    import sqlite_vec

    conn = sqlite3.connect("data/analysis.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    model_id = conn.execute("SELECT id FROM embedding_models").fetchone()[0]

    meta = {}
    dates_map = {}
    for r in conn.execute(
        "SELECT id, title, source, message_count, created_at, has_notes FROM conversations"
    ).fetchall():
        meta[r[0]] = {"title": r[1], "source": r[2], "msgs": r[3],
                       "date": (r[4] or "")[:10], "notes": bool(r[5])}
        if r[4]:
            try:
                dates_map[r[0]] = datetime.strptime(r[4][:10], "%Y-%m-%d")
            except:
                pass

    annotated = set(cid for cid in meta if meta[cid].get("notes"))

    print("Loading messages...")
    conv_msgs = load_messages(conn, model_id)
    valid = {cid: msgs for cid, msgs in conv_msgs.items()
             if len([m for m in msgs if not m["short"]]) >= 5}
    print(f"  {len(valid)} conversations with >= 5 non-short messages")

    lenses = [
        ("mean", {}),
        ("exponential", {"alpha": 0.85}),
        ("reverse_exponential", {"alpha": 0.85}),
        ("surprise", {}),
        ("gaussian", {"k0": "mid", "sigma": "auto"}),  # focal at midpoint
        ("first_only", {}),
        ("last_only", {}),
        ("bookend", {}),
    ]

    # Compute embeddings under each lens
    all_embeddings = {}
    valid_ids = sorted(valid.keys())

    for lens_name, params in lenses:
        print(f"\nComputing lens: {lens_name}...")
        embs = []
        for cid in valid_ids:
            vecs = np.array([m["vec"] for m in valid[cid] if not m["short"]])
            n = len(vecs)
            # Handle parameterized values
            p = dict(params)
            if p.get("k0") == "mid":
                p["k0"] = n // 2
            if p.get("sigma") == "auto":
                p["sigma"] = max(n / 6, 1)
            emb = apply_lens(vecs, lens_name, **p)
            embs.append(emb)
        all_embeddings[lens_name] = np.array(embs)

    # ============================================================
    # ANALYSIS 1: Inter-lens divergence
    # ============================================================
    print(f"\n=== Inter-Lens Divergence ===")
    print("How different is each lens from the mean lens?")

    mean_embs = all_embeddings["mean"]
    print(f"\n{'Lens':<25s} {'Mean div':>10s} {'Med div':>10s} {'Max div':>10s} {'Trail links':>12s}")
    print("-" * 72)

    results = {}
    for lens_name, _ in lenses:
        lens_embs = all_embeddings[lens_name]
        # Per-conversation divergence from mean
        divs = []
        for i in range(len(valid_ids)):
            d = 1 - float(cosine_similarity(
                mean_embs[i].reshape(1, -1), lens_embs[i].reshape(1, -1)
            )[0, 0])
            divs.append(d)

        # Trail links
        links = count_trail_links(valid_ids, lens_embs, dates_map)

        results[lens_name] = {
            "mean_divergence": float(np.mean(divs)),
            "median_divergence": float(np.median(divs)),
            "max_divergence": float(np.max(divs)),
            "trail_links": links,
        }
        print(f"{lens_name:<25s} {np.mean(divs):>10.4f} {np.median(divs):>10.4f} {np.max(divs):>10.4f} {links:>12d}")

    # ============================================================
    # ANALYSIS 2: Which lens best distinguishes annotated convos?
    # ============================================================
    print(f"\n=== Annotated vs Non-Annotated Under Each Lens ===")
    print("Which lens makes annotated conversations most distinctive?")

    ann_idx = [i for i, cid in enumerate(valid_ids) if cid in annotated]
    non_idx = [i for i, cid in enumerate(valid_ids) if cid not in annotated]

    if len(ann_idx) >= 2:
        print(f"\nAnnotated: {len(ann_idx)}, Non-annotated: {len(non_idx)}")
        print(f"\n{'Lens':<25s} {'Ann avg_sim':>12s} {'Non avg_sim':>12s} {'Ratio':>8s} {'p-value':>10s}")
        print("-" * 72)

        for lens_name, _ in lenses:
            lens_embs = all_embeddings[lens_name]

            # Average pairwise similarity within annotated vs within non-annotated
            if len(ann_idx) >= 2:
                ann_embs = lens_embs[ann_idx]
                ann_sim = cosine_similarity(ann_embs)
                np.fill_diagonal(ann_sim, 0)
                ann_avg = float(np.sum(ann_sim) / (len(ann_idx) * (len(ann_idx) - 1)))
            else:
                ann_avg = 0

            # Sample non-annotated for speed
            sample_non = non_idx[:min(200, len(non_idx))]
            non_embs = lens_embs[sample_non]
            non_sim = cosine_similarity(non_embs)
            np.fill_diagonal(non_sim, 0)
            non_avg = float(np.sum(non_sim) / (len(sample_non) * (len(sample_non) - 1)))

            ratio = ann_avg / non_avg if non_avg > 0 else 0

            # Are annotated conversations more spread out (lower internal similarity)?
            ann_self_sims = [ann_sim[i, j] for i in range(len(ann_idx)) for j in range(i + 1, len(ann_idx))]
            non_self_sims = [non_sim[i, j] for i in range(len(sample_non)) for j in range(i + 1, len(sample_non))]
            if ann_self_sims and non_self_sims:
                U, p = mannwhitneyu(ann_self_sims, non_self_sims, alternative="two-sided")
            else:
                p = 1.0

            results[lens_name]["ann_internal_sim"] = ann_avg
            results[lens_name]["non_internal_sim"] = non_avg
            results[lens_name]["p_internal_sim"] = float(p)

            print(f"{lens_name:<25s} {ann_avg:>12.4f} {non_avg:>12.4f} {ratio:>8.2f} {p:>10.4f}")

    # ============================================================
    # ANALYSIS 3: Lens agreement matrix
    # ============================================================
    print(f"\n=== Lens Agreement: Nearest Neighbor Overlap ===")
    print("For each conversation, do different lenses agree on who the neighbors are?")

    # For 200 sampled conversations, get top-5 neighbors under each lens
    sample_idx = list(range(min(200, len(valid_ids))))
    lens_names = [l[0] for l in lenses]

    overlap_matrix = np.zeros((len(lens_names), len(lens_names)))
    for s_idx in sample_idx:
        nn_per_lens = {}
        for l_idx, lens_name in enumerate(lens_names):
            embs = all_embeddings[lens_name]
            sims = cosine_similarity(embs[s_idx].reshape(1, -1), embs)[0]
            sims[s_idx] = -1
            top5 = set(np.argsort(sims)[-5:])
            nn_per_lens[l_idx] = top5

        for a in range(len(lens_names)):
            for b in range(len(lens_names)):
                overlap = len(nn_per_lens[a] & nn_per_lens[b]) / 5.0
                overlap_matrix[a, b] += overlap

    overlap_matrix /= len(sample_idx)

    print(f"\n{'':>25s}", end="")
    for name in lens_names:
        print(f"{name[:8]:>10s}", end="")
    print()
    for i, name in enumerate(lens_names):
        print(f"{name:<25s}", end="")
        for j in range(len(lens_names)):
            print(f"{overlap_matrix[i, j]:>10.2f}", end="")
        print()

    # ============================================================
    # ANALYSIS 4: Surprise lens deep dive
    # ============================================================
    print(f"\n=== Surprise Lens: Topic-Shifting Messages ===")
    print("Conversations where the surprise lens diverges most from mean:")

    surprise_embs = all_embeddings["surprise"]
    surprise_divs = []
    for i, cid in enumerate(valid_ids):
        d = 1 - float(cosine_similarity(
            mean_embs[i].reshape(1, -1), surprise_embs[i].reshape(1, -1)
        )[0, 0])
        surprise_divs.append((cid, d))

    surprise_divs.sort(key=lambda x: -x[1])
    print("\nHighest surprise divergence from mean (conversations dominated by topic shifts):")
    for cid, div in surprise_divs[:15]:
        md = meta.get(cid, {})
        ann = " [notes]" if md.get("notes") else ""
        print(f"  div={div:.3f}  {md.get('title', '?')[:50]:<50s}  {md.get('source', ''):>12s}  {md.get('date', '')}{ann}")

    print("\nLowest surprise divergence (surprise lens ≈ mean, no dominant topic shifts):")
    for cid, div in surprise_divs[-15:]:
        md = meta.get(cid, {})
        print(f"  div={div:.3f}  {md.get('title', '?')[:50]:<50s}  {md.get('source', ''):>12s}  {md.get('date', '')}")

    # Save
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/lens_family.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/results/lens_family.json")
    conn.close()


if __name__ == "__main__":
    main()
