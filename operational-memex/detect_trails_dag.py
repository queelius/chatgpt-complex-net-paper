"""Branching trail detection (DAG structure).

Unlike the greedy forward-chain algorithm, this allows a conversation
to participate in multiple trails. A conversation like "Bayesian
Perspective on AI" might be the origin of both a statistics trail
and a consciousness trail.

The output is a DAG where:
- Nodes are conversations
- Edges are (source, target, similarity, gap_days, surprise, link_type)
- link_type: "mean" (topic similarity) or "continuation" (end-to-start)

We then extract interesting substructures:
- Fan-out nodes (one conversation spawns multiple threads)
- Convergence nodes (multiple threads arrive at the same conversation)
- Long paths through the DAG

Usage:
    python detect_trails_dag.py --db data/analysis.db
"""
import argparse
import json
import os
import sqlite3
import struct
from collections import defaultdict
from datetime import datetime

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def load_data(conn, model_id):
    """Load message embeddings and metadata."""
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

    return conv_msgs, meta, dates_map


def compute_embeddings(conv_msgs, alpha=0.85, uw=3.0, aw=1.0):
    """Compute mean, start, and end embeddings per conversation."""
    results = {}
    for cid, msgs in conv_msgs.items():
        non_short = [m for m in msgs if not m["short"]]
        if len(non_short) < 3:
            continue
        vecs = np.array([m["vec"] for m in non_short])
        n = len(vecs)

        # Mean (role-weighted)
        role_vecs = defaultdict(list)
        for m in non_short:
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
        mean_emb = np.average(components, axis=0, weights=weights)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        # Exponential end
        w_end = np.array([alpha ** (n - 1 - k) for k in range(n)])
        exp_end = np.average(vecs, axis=0, weights=w_end)
        norm = np.linalg.norm(exp_end)
        if norm > 0:
            exp_end = exp_end / norm

        # Exponential start (first 20%)
        fn = max(n // 5, 1)
        fv = vecs[:fn]
        if fn == 1:
            exp_start = fv[0].copy()
        else:
            ws = np.array([alpha ** (fn - 1 - k) for k in range(fn)])
            exp_start = np.average(fv, axis=0, weights=ws)
        norm = np.linalg.norm(exp_start)
        if norm > 0:
            exp_start = exp_start / norm

        results[cid] = {"mean": mean_emb, "start": exp_start, "end": exp_end}

    return results


def build_trail_dag(embs, dates_map, meta,
                    mean_threshold=0.82, e2s_threshold=0.80,
                    min_gap=14, max_gap=365, max_links_per_node=3):
    """Build a DAG of trail links.

    For each conversation, find the top-k forward neighbors by:
    1. Mean-to-mean similarity (topic revisitation)
    2. End-to-start similarity (intellectual continuation)

    Each conversation can have up to max_links_per_node outgoing edges
    (enabling branching).
    """
    dated = sorted(
        [(cid, dates_map[cid]) for cid in embs if cid in dates_map],
        key=lambda x: x[1]
    )

    dag = nx.DiGraph()
    for cid, dt in dated:
        dag.add_node(cid, date=dt.strftime("%Y-%m-%d"), **{
            k: v for k, v in meta.get(cid, {}).items() if k != "date"
        })

    for i, (cid_a, dt_a) in enumerate(dated):
        candidates = []
        for j in range(i + 1, len(dated)):
            cid_b, dt_b = dated[j]
            gap = (dt_b - dt_a).days
            if gap < min_gap:
                continue
            if gap > max_gap:
                break
            if cid_b not in embs:
                continue

            # Mean-to-mean
            mean_sim = float(cosine_similarity(
                embs[cid_a]["mean"].reshape(1, -1),
                embs[cid_b]["mean"].reshape(1, -1)
            )[0, 0])

            # End-to-start
            e2s_sim = float(cosine_similarity(
                embs[cid_a]["end"].reshape(1, -1),
                embs[cid_b]["start"].reshape(1, -1)
            )[0, 0])

            surprise_mean = mean_sim * np.log1p(gap) if mean_sim >= mean_threshold else 0
            surprise_e2s = e2s_sim * np.log1p(gap) if e2s_sim >= e2s_threshold else 0

            if surprise_mean > 0:
                candidates.append({
                    "target": cid_b, "sim": mean_sim, "gap": gap,
                    "surprise": surprise_mean, "link_type": "revisitation",
                })
            if surprise_e2s > 0:
                candidates.append({
                    "target": cid_b, "sim": e2s_sim, "gap": gap,
                    "surprise": surprise_e2s, "link_type": "continuation",
                })

        # Keep top-k by surprise (allowing both link types)
        candidates.sort(key=lambda x: -x["surprise"])
        seen_targets = set()
        added = 0
        for c in candidates:
            if c["target"] in seen_targets:
                continue
            dag.add_edge(cid_a, c["target"],
                         sim=c["sim"], gap=c["gap"],
                         surprise=c["surprise"], link_type=c["link_type"])
            seen_targets.add(c["target"])
            added += 1
            if added >= max_links_per_node:
                break

    return dag


def analyze_dag(dag, meta):
    """Analyze the trail DAG for interesting structures."""
    print(f"\nDAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")

    # Link type distribution
    link_types = defaultdict(int)
    for u, v, data in dag.edges(data=True):
        link_types[data.get("link_type", "unknown")] += 1
    print(f"Link types: {dict(link_types)}")

    # In/out degree distributions
    out_degrees = [dag.out_degree(n) for n in dag.nodes()]
    in_degrees = [dag.in_degree(n) for n in dag.nodes()]
    print(f"Out-degree: max={max(out_degrees)}, mean={np.mean(out_degrees):.2f}")
    print(f"In-degree:  max={max(in_degrees)}, mean={np.mean(in_degrees):.2f}")

    # Fan-out nodes (one conversation spawns multiple threads)
    print(f"\n--- Fan-Out Nodes (out_degree >= 2) ---")
    fan_outs = [(n, dag.out_degree(n)) for n in dag.nodes() if dag.out_degree(n) >= 2]
    fan_outs.sort(key=lambda x: -x[1])
    for n, deg in fan_outs[:15]:
        md = meta.get(n, {})
        ann = " [notes]" if md.get("notes") else ""
        successors = list(dag.successors(n))
        succ_types = [dag[n][s].get("link_type", "?") for s in successors]
        print(f"  out={deg}  {md.get('title', '?')[:45]:<45s}  {md.get('date', '')} {md.get('source', '')}{ann}")
        for s, lt in zip(successors, succ_types):
            ms = meta.get(s, {})
            edge = dag[n][s]
            print(f"    -> [{lt[:5]}] sim={edge['sim']:.3f} gap={edge['gap']}d  {ms.get('title', '?')[:45]}  ({ms.get('date', '')})")

    # Convergence nodes (multiple threads arrive at same conversation)
    print(f"\n--- Convergence Nodes (in_degree >= 2) ---")
    converges = [(n, dag.in_degree(n)) for n in dag.nodes() if dag.in_degree(n) >= 2]
    converges.sort(key=lambda x: -x[1])
    for n, deg in converges[:15]:
        md = meta.get(n, {})
        ann = " [notes]" if md.get("notes") else ""
        predecessors = list(dag.predecessors(n))
        print(f"  in={deg}  {md.get('title', '?')[:45]:<45s}  {md.get('date', '')} {md.get('source', '')}{ann}")
        for p in predecessors[:5]:
            mp = meta.get(p, {})
            edge = dag[p][n]
            lt = edge.get("link_type", "?")
            print(f"    <- [{lt[:5]}] sim={edge['sim']:.3f} gap={edge['gap']}d  {mp.get('title', '?')[:45]}  ({mp.get('date', '')})")
        if len(predecessors) > 5:
            print(f"    ... and {len(predecessors) - 5} more")

    # Longest paths through the DAG
    print(f"\n--- Longest Paths ---")
    longest_paths = []
    for node in dag.nodes():
        if dag.in_degree(node) == 0:  # start from roots
            for path in _dfs_paths(dag, node, max_depth=15):
                if len(path) >= 3:
                    path_surprise = np.mean([
                        dag[path[i]][path[i+1]].get("surprise", 0)
                        for i in range(len(path) - 1)
                    ])
                    longest_paths.append((path, path_surprise))

    longest_paths.sort(key=lambda x: (-len(x[0]), -x[1]))
    seen = set()
    shown = 0
    for path, surprise in longest_paths:
        path_key = tuple(path[:3])  # deduplicate by first 3 nodes
        if path_key in seen:
            continue
        seen.add(path_key)
        print(f"\nPath (len={len(path)}, avg_surprise={surprise:.2f}):")
        for i, cid in enumerate(path):
            md = meta.get(cid, {})
            ann = " [notes]" if md.get("notes") else ""
            if i < len(path) - 1:
                edge = dag[cid][path[i + 1]]
                link = f"  -> [{edge.get('link_type', '?')[:5]}] sim={edge['sim']:.3f} gap={edge['gap']}d"
            else:
                link = ""
            print(f"  {i+1:>2d}. {md.get('date', ''):10s}  {md.get('title', '?')[:45]:<45s}  {md.get('source', ''):>12s}{ann}{link}")
        shown += 1
        if shown >= 10:
            break

    # Annotated node enrichment in DAG
    annotated = set(n for n in dag.nodes() if meta.get(n, {}).get("notes"))
    dag_members = set(n for n in dag.nodes() if dag.degree(n) > 0)
    ann_in_dag = annotated & dag_members
    print(f"\n--- Annotated Node Analysis ---")
    print(f"Annotated in DAG: {len(ann_in_dag)}/{len(annotated)}")
    if ann_in_dag:
        for n in ann_in_dag:
            md = meta.get(n, {})
            print(f"  in={dag.in_degree(n)} out={dag.out_degree(n)}  {md.get('title', '?')[:50]}  ({md.get('date', '')})")

    return fan_outs, converges


def _dfs_paths(dag, start, max_depth=15):
    """Yield all paths from start, up to max_depth."""
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if len(path) >= max_depth:
            yield path
            continue
        successors = list(dag.successors(node))
        if not successors:
            yield path
        else:
            for succ in successors:
                if succ not in path:  # avoid cycles
                    stack.append((succ, path + [succ]))


def main():
    import sqlite_vec

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/analysis.db")
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--mean-threshold", type=float, default=0.82)
    parser.add_argument("--e2s-threshold", type=float, default=0.80)
    parser.add_argument("--max-links", type=int, default=3)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    model_id = conn.execute("SELECT id FROM embedding_models").fetchone()[0]

    conv_msgs, meta, dates_map = load_data(conn, model_id)
    print(f"Loaded {len(conv_msgs)} conversations")

    embs = compute_embeddings(conv_msgs, alpha=args.alpha)
    print(f"Computed embeddings for {len(embs)} conversations")

    print(f"\nBuilding trail DAG (mean_t={args.mean_threshold}, e2s_t={args.e2s_threshold}, max_links={args.max_links})...")
    dag = build_trail_dag(embs, dates_map, meta,
                          mean_threshold=args.mean_threshold,
                          e2s_threshold=args.e2s_threshold,
                          max_links_per_node=args.max_links)

    fan_outs, converges = analyze_dag(dag, meta)

    # Save DAG as edge list
    edges = []
    for u, v, data in dag.edges(data=True):
        edges.append({
            "source": u, "target": v,
            "sim": data["sim"], "gap": data["gap"],
            "surprise": data["surprise"], "link_type": data["link_type"],
        })
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/trail_dag.json", "w") as f:
        json.dump({"edges": edges, "node_count": dag.number_of_nodes(),
                    "edge_count": dag.number_of_edges()}, f, indent=2)
    print(f"\nSaved DAG to experiments/results/trail_dag.json")

    conn.close()


if __name__ == "__main__":
    main()
