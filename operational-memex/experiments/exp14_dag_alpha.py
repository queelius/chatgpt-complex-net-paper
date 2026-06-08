"""Experiment 14: Alpha sensitivity on the trail DAG.

How does the branching/convergence structure change as alpha varies?
At low alpha (short memory), continuation links capture tight handoffs.
At high alpha (long memory), continuation links degenerate to mean similarity.
Where does the DAG structure change qualitatively?
"""
import json
import os
import sqlite3
import struct
import sys
from collections import defaultdict
from datetime import datetime

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


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


def compute_embs(conv_msgs, alpha, uw=3.0, aw=1.0):
    results = {}
    for cid, msgs in conv_msgs.items():
        non_short = [m for m in msgs if not m["short"]]
        if len(non_short) < 3:
            continue
        vecs = np.array([m["vec"] for m in non_short])
        n = len(vecs)

        # Mean (role-weighted)
        rv = defaultdict(list)
        for m in non_short:
            rv[m["role"]].append(m["vec"])
        c, w = [], []
        if "user" in rv: c.append(np.mean(rv["user"], axis=0)); w.append(uw)
        if "assistant" in rv: c.append(np.mean(rv["assistant"], axis=0)); w.append(aw)
        if not c: continue
        mean_emb = np.average(c, axis=0, weights=w)
        norm = np.linalg.norm(mean_emb)
        if norm > 0: mean_emb /= norm

        # Exp end
        we = np.array([alpha ** (n - 1 - k) for k in range(n)])
        exp_end = np.average(vecs, axis=0, weights=we)
        norm = np.linalg.norm(exp_end)
        if norm > 0: exp_end /= norm

        # Exp start (first 20%)
        fn = max(n // 5, 1)
        fv = vecs[:fn]
        if fn == 1:
            exp_start = fv[0].copy()
        else:
            ws = np.array([alpha ** (fn - 1 - k) for k in range(fn)])
            exp_start = np.average(fv, axis=0, weights=ws)
        norm = np.linalg.norm(exp_start)
        if norm > 0: exp_start /= norm

        results[cid] = {"mean": mean_emb, "start": exp_start, "end": exp_end}
    return results


def build_dag(embs, dates_map, mean_t=0.82, e2s_t=0.80, min_gap=14, max_gap=365, max_links=3):
    dated = sorted([(c, dates_map[c]) for c in embs if c in dates_map], key=lambda x: x[1])
    dag = nx.DiGraph()
    for cid, _ in dated:
        dag.add_node(cid)

    link_types = {"revisitation": 0, "continuation": 0}
    for i, (cid_a, dt_a) in enumerate(dated):
        candidates = []
        for j in range(i + 1, len(dated)):
            cid_b, dt_b = dated[j]
            gap = (dt_b - dt_a).days
            if gap < min_gap: continue
            if gap > max_gap: break
            if cid_b not in embs: continue

            mean_sim = float(cosine_similarity(embs[cid_a]["mean"].reshape(1, -1), embs[cid_b]["mean"].reshape(1, -1))[0, 0])
            e2s_sim = float(cosine_similarity(embs[cid_a]["end"].reshape(1, -1), embs[cid_b]["start"].reshape(1, -1))[0, 0])

            if mean_sim >= mean_t:
                candidates.append({"t": cid_b, "s": mean_sim, "g": gap, "sr": mean_sim * np.log1p(gap), "lt": "revisitation"})
            if e2s_sim >= e2s_t:
                candidates.append({"t": cid_b, "s": e2s_sim, "g": gap, "sr": e2s_sim * np.log1p(gap), "lt": "continuation"})

        candidates.sort(key=lambda x: -x["sr"])
        seen = set()
        added = 0
        for c in candidates:
            if c["t"] in seen: continue
            dag.add_edge(cid_a, c["t"], link_type=c["lt"])
            link_types[c["lt"]] += 1
            seen.add(c["t"])
            added += 1
            if added >= max_links: break

    return dag, link_types


def dag_metrics(dag):
    out_deg = [dag.out_degree(n) for n in dag.nodes()]
    in_deg = [dag.in_degree(n) for n in dag.nodes()]
    connected = sum(1 for n in dag.nodes() if dag.degree(n) > 0)
    fan_out = sum(1 for n in dag.nodes() if dag.out_degree(n) >= 2)
    converge = sum(1 for n in dag.nodes() if dag.in_degree(n) >= 2)
    # Longest path (via DFS from roots)
    max_path = 0
    roots = [n for n in dag.nodes() if dag.in_degree(n) == 0 and dag.out_degree(n) > 0]
    for root in roots[:100]:
        stack = [(root, 1)]
        while stack:
            node, depth = stack.pop()
            max_path = max(max_path, depth)
            if depth < 15:
                for succ in dag.successors(node):
                    stack.append((succ, depth + 1))

    return {
        "nodes": dag.number_of_nodes(),
        "edges": dag.number_of_edges(),
        "connected": connected,
        "max_out_degree": max(out_deg) if out_deg else 0,
        "max_in_degree": max(in_deg) if in_deg else 0,
        "mean_out_degree": float(np.mean(out_deg)),
        "fan_out_nodes": fan_out,
        "convergence_nodes": converge,
        "longest_path": max_path,
    }


def main():
    import sqlite_vec

    conn = sqlite3.connect("data/analysis.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    model_id = conn.execute("SELECT id FROM embedding_models").fetchone()[0]

    dates_map = {}
    for r in conn.execute("SELECT id, created_at FROM conversations").fetchall():
        if r[1]:
            try: dates_map[r[0]] = datetime.strptime(r[1][:10], "%Y-%m-%d")
            except: pass

    print("Loading messages...")
    conv_msgs = load_messages(conn, model_id)
    print(f"  {len(conv_msgs)} conversations")

    alphas = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
    results = []

    print(f"\n{'alpha':>6s} {'half':>5s} {'edges':>6s} {'revis':>6s} {'conti':>6s} {'fan_out':>8s} {'converge':>9s} {'longest':>8s} {'connected':>10s}")
    print("-" * 75)

    for alpha in alphas:
        half_life = np.log(0.5) / np.log(alpha) if alpha < 1 else float("inf")
        embs = compute_embs(conv_msgs, alpha)
        dag, lt = build_dag(embs, dates_map)
        m = dag_metrics(dag)
        m["alpha"] = alpha
        m["half_life"] = round(half_life, 1)
        m["revisitation_edges"] = lt["revisitation"]
        m["continuation_edges"] = lt["continuation"]
        results.append(m)

        print(f"{alpha:>6.2f} {half_life:>5.1f} {m['edges']:>6d} {lt['revisitation']:>6d} {lt['continuation']:>6d} "
              f"{m['fan_out_nodes']:>8d} {m['convergence_nodes']:>9d} {m['longest_path']:>8d} {m['connected']:>10d}")

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/dag_alpha_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/results/dag_alpha_sensitivity.json")
    conn.close()


if __name__ == "__main__":
    main()
