"""Experiment 15: Structural properties of revisitation vs continuation links.

Do "revisitation" links (mean-to-mean, same topic returned to) and
"continuation" links (end-to-start, picked up where left off) connect
different kinds of conversations? Do they cross different community
boundaries? Do they have different temporal gap distributions?
"""
import json
import os
import sqlite3
import struct
import sys
from collections import defaultdict
from datetime import datetime

import community as community_louvain
import networkx as nx
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def load_and_compute(conn, model_id, alpha=0.85, uw=3.0, aw=1.0):
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

    embs = {}
    for cid, msgs in conv_msgs.items():
        non_short = [m for m in msgs if not m["short"]]
        if len(non_short) < 3: continue
        vecs = np.array([m["vec"] for m in non_short])
        n = len(vecs)

        rv = defaultdict(list)
        for m in non_short: rv[m["role"]].append(m["vec"])
        c, w = [], []
        if "user" in rv: c.append(np.mean(rv["user"], axis=0)); w.append(uw)
        if "assistant" in rv: c.append(np.mean(rv["assistant"], axis=0)); w.append(aw)
        if not c: continue
        mean_emb = np.average(c, axis=0, weights=w)
        norm = np.linalg.norm(mean_emb)
        if norm > 0: mean_emb /= norm

        we = np.array([alpha ** (n - 1 - k) for k in range(n)])
        exp_end = np.average(vecs, axis=0, weights=we)
        norm = np.linalg.norm(exp_end)
        if norm > 0: exp_end /= norm

        fn = max(n // 5, 1)
        fv = vecs[:fn]
        if fn == 1: exp_start = fv[0].copy()
        else:
            ws = np.array([alpha ** (fn - 1 - k) for k in range(fn)])
            exp_start = np.average(fv, axis=0, weights=ws)
        norm = np.linalg.norm(exp_start)
        if norm > 0: exp_start /= norm

        drift = 1 - float(cosine_similarity(exp_start.reshape(1, -1), exp_end.reshape(1, -1))[0, 0])
        embs[cid] = {"mean": mean_emb, "start": exp_start, "end": exp_end, "drift": drift, "n_msgs": n}

    return embs


def main():
    import sqlite_vec

    conn = sqlite3.connect("data/analysis.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    model_id = conn.execute("SELECT id FROM embedding_models").fetchone()[0]

    meta = {}
    dates_map = {}
    for r in conn.execute("SELECT id, title, source, message_count, created_at, has_notes FROM conversations").fetchall():
        meta[r[0]] = {"title": r[1], "source": r[2], "msgs": r[3], "date": (r[4] or "")[:10], "notes": bool(r[5])}
        if r[4]:
            try: dates_map[r[0]] = datetime.strptime(r[4][:10], "%Y-%m-%d")
            except: pass

    print("Computing embeddings...")
    embs = load_and_compute(conn, model_id)
    print(f"  {len(embs)} conversations")

    # Build all links (not limited to top-k)
    dated = sorted([(c, dates_map[c]) for c in embs if c in dates_map], key=lambda x: x[1])
    mean_t, e2s_t = 0.82, 0.80
    min_gap, max_gap = 14, 365

    revisitation_links = []
    continuation_links = []

    print("Finding all links...")
    for i, (cid_a, dt_a) in enumerate(dated):
        for j in range(i + 1, len(dated)):
            cid_b, dt_b = dated[j]
            gap = (dt_b - dt_a).days
            if gap < min_gap: continue
            if gap > max_gap: break
            if cid_b not in embs: continue

            mean_sim = float(cosine_similarity(embs[cid_a]["mean"].reshape(1, -1), embs[cid_b]["mean"].reshape(1, -1))[0, 0])
            e2s_sim = float(cosine_similarity(embs[cid_a]["end"].reshape(1, -1), embs[cid_b]["start"].reshape(1, -1))[0, 0])

            if mean_sim >= mean_t:
                revisitation_links.append({
                    "src": cid_a, "dst": cid_b, "sim": mean_sim, "gap": gap,
                    "src_source": meta.get(cid_a, {}).get("source"),
                    "dst_source": meta.get(cid_b, {}).get("source"),
                    "src_drift": embs[cid_a]["drift"],
                    "dst_drift": embs[cid_b]["drift"],
                })
            if e2s_sim >= e2s_t:
                continuation_links.append({
                    "src": cid_a, "dst": cid_b, "sim": e2s_sim, "gap": gap,
                    "src_source": meta.get(cid_a, {}).get("source"),
                    "dst_source": meta.get(cid_b, {}).get("source"),
                    "src_drift": embs[cid_a]["drift"],
                    "dst_drift": embs[cid_b]["drift"],
                })

    print(f"\nRevisitation links: {len(revisitation_links)}")
    print(f"Continuation links: {len(continuation_links)}")

    # Overlap: how many links appear in both?
    revis_set = set((l["src"], l["dst"]) for l in revisitation_links)
    conti_set = set((l["src"], l["dst"]) for l in continuation_links)
    overlap = revis_set & conti_set
    revis_only = revis_set - conti_set
    conti_only = conti_set - revis_set
    print(f"Overlap: {len(overlap)} links appear in both")
    print(f"Revisitation-only: {len(revis_only)}")
    print(f"Continuation-only: {len(conti_only)}")
    if revis_set | conti_set:
        print(f"Jaccard: {len(overlap) / len(revis_set | conti_set):.3f}")

    # Compare gap distributions
    revis_gaps = [l["gap"] for l in revisitation_links]
    conti_gaps = [l["gap"] for l in continuation_links]
    print(f"\n--- Gap Distribution ---")
    print(f"  Revisitation: mean={np.mean(revis_gaps):.0f}d, median={np.median(revis_gaps):.0f}d")
    print(f"  Continuation: mean={np.mean(conti_gaps):.0f}d, median={np.median(conti_gaps):.0f}d")
    if len(conti_gaps) >= 5:
        U, p = mannwhitneyu(conti_gaps, revis_gaps, alternative="two-sided")
        print(f"  Mann-Whitney U p={p:.4f}")

    # Cross-source rate
    revis_cross = sum(1 for l in revisitation_links if l["src_source"] != l["dst_source"])
    conti_cross = sum(1 for l in continuation_links if l["src_source"] != l["dst_source"])
    print(f"\n--- Cross-Source Links ---")
    print(f"  Revisitation: {revis_cross}/{len(revisitation_links)} ({revis_cross/len(revisitation_links)*100:.1f}%)")
    if continuation_links:
        print(f"  Continuation: {conti_cross}/{len(continuation_links)} ({conti_cross/len(continuation_links)*100:.1f}%)")

    # Drift of source conversations
    revis_src_drift = [l["src_drift"] for l in revisitation_links]
    conti_src_drift = [l["src_drift"] for l in continuation_links]
    print(f"\n--- Source Conversation Drift ---")
    print(f"  Revisitation sources: mean drift={np.mean(revis_src_drift):.4f}")
    if conti_src_drift:
        print(f"  Continuation sources: mean drift={np.mean(conti_src_drift):.4f}")
        U, p = mannwhitneyu(conti_src_drift, revis_src_drift, alternative="two-sided")
        print(f"  Mann-Whitney U p={p:.4f}")

    # Are continuation-only links between conversations with higher drift?
    conti_only_drifts = [embs[src]["drift"] for src, dst in conti_only if src in embs]
    revis_only_drifts = [embs[src]["drift"] for src, dst in revis_only if src in embs]
    if conti_only_drifts and revis_only_drifts:
        print(f"\n--- Exclusive Link Source Drift ---")
        print(f"  Continuation-only source drift: mean={np.mean(conti_only_drifts):.4f} (n={len(conti_only_drifts)})")
        print(f"  Revisitation-only source drift: mean={np.mean(revis_only_drifts):.4f} (n={len(revis_only_drifts)})")
        U, p = mannwhitneyu(conti_only_drifts, revis_only_drifts, alternative="greater")
        print(f"  Mann-Whitney U (conti > revis) p={p:.4f}")

    # Build undirected semantic graph for community context
    print(f"\n--- Community Context ---")
    mean_vecs = np.array([embs[cid]["mean"] for cid, _ in dated if cid in embs])
    mean_ids = [cid for cid, _ in dated if cid in embs]
    sim_matrix = cosine_similarity(mean_vecs)

    G = nx.Graph()
    for i, cid in enumerate(mean_ids): G.add_node(cid)
    for i in range(len(mean_ids)):
        for j in range(i + 1, len(mean_ids)):
            if sim_matrix[i, j] >= 0.78:
                G.add_edge(mean_ids[i], mean_ids[j])

    comps = list(nx.connected_components(G))
    if comps:
        gc = G.subgraph(max(comps, key=len)).copy()
        partition = community_louvain.best_partition(gc)

        # How often do links cross community boundaries?
        revis_cross_comm = 0
        revis_in_gc = 0
        for l in revisitation_links:
            if l["src"] in partition and l["dst"] in partition:
                revis_in_gc += 1
                if partition[l["src"]] != partition[l["dst"]]:
                    revis_cross_comm += 1

        conti_cross_comm = 0
        conti_in_gc = 0
        for l in continuation_links:
            if l["src"] in partition and l["dst"] in partition:
                conti_in_gc += 1
                if partition[l["src"]] != partition[l["dst"]]:
                    conti_cross_comm += 1

        print(f"  Revisitation cross-community: {revis_cross_comm}/{revis_in_gc} ({revis_cross_comm/revis_in_gc*100:.1f}%)" if revis_in_gc else "")
        print(f"  Continuation cross-community: {conti_cross_comm}/{conti_in_gc} ({conti_cross_comm/conti_in_gc*100:.1f}%)" if conti_in_gc else "")

    # Save
    results = {
        "revisitation_count": len(revisitation_links),
        "continuation_count": len(continuation_links),
        "overlap": len(overlap),
        "jaccard": len(overlap) / len(revis_set | conti_set) if revis_set | conti_set else 0,
        "gap_revis_mean": float(np.mean(revis_gaps)),
        "gap_conti_mean": float(np.mean(conti_gaps)) if conti_gaps else None,
        "cross_source_revis_pct": revis_cross / len(revisitation_links) if revisitation_links else 0,
        "cross_source_conti_pct": conti_cross / len(continuation_links) if continuation_links else 0,
    }
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/link_type_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/results/link_type_analysis.json")
    conn.close()


if __name__ == "__main__":
    main()
