"""Experiment 2: Marginalia as attention signal.

Tests whether user-annotated conversations occupy structurally significant
positions in the semantic graph compared to non-annotated conversations.

Metrics compared: degree centrality, betweenness centrality, bridge score
(inter-community edge fraction).

Statistical tests: Mann-Whitney U (one-sided), permutation test (10,000 iters).
"""
import json
import os
import sqlite3
import struct

import community as community_louvain
import networkx as nx
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def compose_embeddings(conn, model_id, user_weight=3.0, asst_weight=1.0):
    rows = conn.execute("""
        SELECT me.conversation_id, me.role, mv.embedding
        FROM message_embeddings me
        JOIN message_vectors mv
          ON mv.conversation_id = me.conversation_id
          AND mv.message_id = me.message_id
          AND mv.model_id = me.model_id
        WHERE me.model_id = ?
        ORDER BY me.conversation_id
    """, [model_id]).fetchall()

    conv_role_vecs = {}
    for row in rows:
        conv_id, role = row[0], row[1]
        vec = deserialize_f32(row[2])
        conv_role_vecs.setdefault(conv_id, {}).setdefault(role, []).append(vec)

    ids, vectors = [], []
    for conv_id, role_vecs in conv_role_vecs.items():
        components, weights = [], []
        if "user" in role_vecs:
            components.append(np.mean(role_vecs["user"], axis=0))
            weights.append(user_weight)
        if "assistant" in role_vecs:
            components.append(np.mean(role_vecs["assistant"], axis=0))
            weights.append(asst_weight)
        if not components:
            continue
        emb = np.average(components, axis=0, weights=weights)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        ids.append(conv_id)
        vectors.append(emb)
    return ids, np.array(vectors)


def main():
    import sqlite_vec

    conn = sqlite3.connect("data/analysis.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    model_id = conn.execute("SELECT id FROM embedding_models").fetchone()[0]

    # Get annotated conversation IDs
    annotated = set(
        r[0] for r in conn.execute(
            "SELECT DISTINCT id FROM conversations WHERE has_notes = 1"
        ).fetchall()
    )
    print(f"Annotated conversations: {len(annotated)}")

    # Compose conversation embeddings
    ids, vecs = compose_embeddings(conn, model_id, user_weight=3.0, asst_weight=1.0)
    sim = cosine_similarity(vecs)

    # Build graph at operational threshold
    theta = 0.78
    G = nx.Graph()
    for i, cid in enumerate(ids):
        G.add_node(cid)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim[i, j] >= theta:
                G.add_edge(ids[i], ids[j], weight=float(sim[i, j]))

    # Giant component
    comps = list(nx.connected_components(G))
    gc_nodes = max(comps, key=len)
    gc = G.subgraph(gc_nodes).copy()
    partition = community_louvain.best_partition(gc)

    # Compute metrics
    degree = dict(gc.degree())
    betweenness = nx.betweenness_centrality(gc)
    bridge_score = {}
    for node in gc.nodes():
        node_comm = partition[node]
        total = gc.degree(node)
        if total == 0:
            bridge_score[node] = 0
        else:
            inter = sum(1 for nb in gc.neighbors(node) if partition[nb] != node_comm)
            bridge_score[node] = inter / total

    # Split annotated vs non-annotated
    ann_in_graph = [cid for cid in ids if cid in annotated]
    ann_in_gc = [n for n in gc.nodes() if n in annotated]
    non_in_gc = [n for n in gc.nodes() if n not in annotated]

    ann_gc_rate = len(ann_in_gc) / len(ann_in_graph) if ann_in_graph else 0
    total_gc_rate = len(gc_nodes) / len(ids)

    print(f"\nGiant component: {gc.number_of_nodes()} nodes (theta={theta})")
    print(f"  Annotated in GC: {len(ann_in_gc)}/{len(ann_in_graph)} ({ann_gc_rate:.1%})")
    print(f"  Overall in GC:   {len(gc_nodes)}/{len(ids)} ({total_gc_rate:.1%})")
    print(f"  Enrichment:      {ann_gc_rate / total_gc_rate:.1f}x" if total_gc_rate > 0 else "")

    if len(ann_in_gc) < 2:
        print("\nToo few annotated nodes in GC for statistical comparison.")
        conn.close()
        return

    # Compare metrics
    metrics_data = {
        "degree": degree,
        "betweenness": betweenness,
        "bridge_score": bridge_score,
    }

    print(f"\n{'Metric':<16s} {'Ann mean':>10s} {'Non mean':>10s} {'Ann med':>10s} {'Non med':>10s} {'U':>8s} {'p':>8s} {'r':>8s}")
    print("-" * 90)

    results = {"theta": theta, "user_weight": 3.0, "n_annotated_total": len(annotated)}

    for name, vals in metrics_data.items():
        ann_vals = [vals[n] for n in ann_in_gc]
        non_vals = [vals[n] for n in non_in_gc]
        ann_mean = np.mean(ann_vals)
        non_mean = np.mean(non_vals)
        ann_med = np.median(ann_vals)
        non_med = np.median(non_vals)

        U, p = mannwhitneyu(ann_vals, non_vals, alternative="greater")
        r = 1 - (2 * U) / (len(ann_vals) * len(non_vals))

        print(f"{name:<16s} {ann_mean:>10.4f} {non_mean:>10.4f} {ann_med:>10.4f} {non_med:>10.4f} {U:>8.0f} {p:>8.4f} {r:>8.3f}")

        results[name] = {
            "annotated_mean": float(ann_mean),
            "non_annotated_mean": float(non_mean),
            "annotated_median": float(ann_med),
            "non_annotated_median": float(non_med),
            "U": float(U),
            "p_value": float(p),
            "effect_size_r": float(r),
            "n_annotated": len(ann_vals),
            "n_non_annotated": len(non_vals),
        }

    # Permutation test
    print(f"\nPermutation test (10,000 iterations):")
    n_ann = len(ann_in_gc)
    gc_list = list(gc.nodes())
    rng = np.random.default_rng(42)

    null_means = {name: [] for name in metrics_data}
    for _ in range(10000):
        fake_ann = set(rng.choice(gc_list, size=n_ann, replace=False))
        for name, vals in metrics_data.items():
            null_means[name].append(np.mean([vals[n] for n in fake_ann]))

    for name in metrics_data:
        observed = results[name]["annotated_mean"]
        null_dist = null_means[name]
        p_perm = np.mean([nm >= observed for nm in null_dist])
        pct_rank = np.mean([nm < observed for nm in null_dist]) * 100
        print(f"  {name:<16s} observed={observed:.4f}, null_mean={np.mean(null_dist):.4f}, "
              f"p_perm={p_perm:.4f}, percentile={pct_rank:.1f}%")
        results[name]["p_permutation"] = float(p_perm)
        results[name]["null_mean"] = float(np.mean(null_dist))
        results[name]["percentile"] = float(pct_rank)

    # Community distribution of annotated nodes
    print(f"\nCommunity distribution of annotated nodes:")
    ann_comms = {}
    for n in ann_in_gc:
        c = partition[n]
        ann_comms[c] = ann_comms.get(c, 0) + 1
    for c, count in sorted(ann_comms.items(), key=lambda x: -x[1]):
        comm_size = sum(1 for v in partition.values() if v == c)
        meta = conn.execute(
            "SELECT title FROM conversations WHERE id = ?",
            [next(n for n in ann_in_gc if partition[n] == c)]
        ).fetchone()
        print(f"  Community {c}: {count} annotated / {comm_size} total  "
              f"(e.g. {meta[0][:50] if meta else '?'})")

    n_communities_with_ann = len(ann_comms)
    total_communities = len(set(partition.values()))
    results["community_spread"] = {
        "communities_with_annotations": n_communities_with_ann,
        "total_communities": total_communities,
        "spread_ratio": n_communities_with_ann / total_communities,
    }

    # Individual annotated node details
    print(f"\nIndividual annotated nodes in GC:")
    for n in sorted(ann_in_gc, key=lambda x: -degree.get(x, 0)):
        meta = conn.execute(
            "SELECT title, source FROM conversations WHERE id = ?", [n]
        ).fetchone()
        title = meta[0][:45] if meta else "?"
        src = meta[1] if meta else "?"
        print(f"  deg={degree[n]:>4d}  btw={betweenness[n]:.4f}  "
              f"bridge={bridge_score[n]:.2f}  comm={partition[n]}  {title} ({src})")

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/marginalia_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/results/marginalia_analysis.json")
    conn.close()


if __name__ == "__main__":
    main()
