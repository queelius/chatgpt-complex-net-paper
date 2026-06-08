"""Experiment 3: Temporal analysis.

Heaps' law (vocabulary growth), densification law, and temporal
community evolution across the conversation archive.
"""
import json
import os
import sqlite3
import struct
from collections import Counter
from datetime import datetime

import community as community_louvain
import networkx as nx
import numpy as np
from scipy.optimize import curve_fit
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


def power_law(x, a, b):
    return a * np.power(x, b)


def main():
    import sqlite_vec

    conn = sqlite3.connect("data/analysis.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    model_id = conn.execute("SELECT id FROM embedding_models").fetchone()[0]

    # Get conversation dates
    conv_dates = {}
    for r in conn.execute("SELECT id, created_at, source FROM conversations").fetchall():
        if r[1]:
            conv_dates[r[0]] = {"date": r[1][:10], "source": r[2]}

    # Compose embeddings
    ids, vecs = compose_embeddings(conn, model_id, user_weight=3.0, asst_weight=1.0)
    sim = cosine_similarity(vecs)
    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    # Sort by date
    dated_ids = [(cid, conv_dates[cid]["date"]) for cid in ids if cid in conv_dates]
    dated_ids.sort(key=lambda x: x[1])

    # =========================================
    # Densification law: e(t) ~ n(t)^gamma
    # =========================================
    theta = 0.78
    print("=== Densification Law ===")
    print(f"(theta={theta}, uw=3.0)")

    # Monthly snapshots
    monthly = {}
    for cid, date in dated_ids:
        month = date[:7]
        monthly.setdefault(month, []).append(cid)

    cumulative_ids = []
    snapshots = []

    for month in sorted(monthly.keys()):
        cumulative_ids.extend(monthly[month])
        n = len(cumulative_ids)
        if n < 5:
            continue

        # Count edges among cumulative nodes
        idx_list = [id_to_idx[cid] for cid in cumulative_ids if cid in id_to_idx]
        edges = 0
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                if sim[idx_list[i], idx_list[j]] >= theta:
                    edges += 1

        if edges > 0:
            snapshots.append({"month": month, "nodes": n, "edges": edges})

    print(f"  {len(snapshots)} monthly snapshots")

    # Fit power law: e = a * n^gamma
    if len(snapshots) >= 5:
        ns = np.array([s["nodes"] for s in snapshots], dtype=float)
        es = np.array([s["edges"] for s in snapshots], dtype=float)

        try:
            popt, pcov = curve_fit(power_law, ns, es, p0=[1, 1.4], maxfev=5000)
            a, gamma = popt
            perr = np.sqrt(np.diag(pcov))

            # R-squared
            predicted = power_law(ns, *popt)
            ss_res = np.sum((es - predicted) ** 2)
            ss_tot = np.sum((es - np.mean(es)) ** 2)
            r_squared = 1 - ss_res / ss_tot

            print(f"  Densification exponent gamma = {gamma:.3f} +/- {perr[1]:.3f}")
            print(f"  R-squared = {r_squared:.4f}")
            print(f"  (Paper 1: gamma=1.405, Paper 2: gamma=1.410)")
        except Exception as e:
            print(f"  Fit failed: {e}")
            gamma, r_squared = None, None
    else:
        print("  Too few snapshots for fitting")
        gamma, r_squared = None, None

    # =========================================
    # Heaps' law: vocabulary growth
    # =========================================
    print("\n=== Heaps' Law (Title Vocabulary Growth) ===")

    # Use conversation titles as a proxy for "concept vocabulary"
    titles = conn.execute(
        "SELECT title FROM conversations WHERE title IS NOT NULL ORDER BY created_at"
    ).fetchall()

    vocab = set()
    vocab_growth = []
    for i, (title,) in enumerate(titles, 1):
        words = set(title.lower().split())
        vocab.update(words)
        vocab_growth.append({"n": i, "vocab": len(vocab)})

    # Fit Heaps' law: V = a * n^beta
    if len(vocab_growth) >= 10:
        ns = np.array([v["n"] for v in vocab_growth], dtype=float)
        vs = np.array([v["vocab"] for v in vocab_growth], dtype=float)

        popt, pcov = curve_fit(power_law, ns, vs, p0=[1, 0.5], maxfev=5000)
        a_heaps, beta = popt
        perr = np.sqrt(np.diag(pcov))

        predicted = power_law(ns, *popt)
        ss_res = np.sum((vs - predicted) ** 2)
        ss_tot = np.sum((vs - np.mean(vs)) ** 2)
        r_sq_heaps = 1 - ss_res / ss_tot

        print(f"  Heaps exponent beta = {beta:.3f} +/- {perr[1]:.3f}")
        print(f"  R-squared = {r_sq_heaps:.4f}")
        print(f"  Final vocabulary size: {len(vocab)} unique words from {len(titles)} titles")
        print(f"  (Paper 1: beta=0.286 for LLM-extracted concepts)")
        print(f"  beta < 1 indicates sublinear growth (consolidation signature)")
    else:
        beta, r_sq_heaps = None, None

    # =========================================
    # Per-source densification
    # =========================================
    print("\n=== Per-Source Densification ===")
    sources = set(conv_dates[cid]["source"] for cid in ids if cid in conv_dates)
    per_source_gamma = {}

    for source in sorted(sources):
        source_dated = [(cid, d) for cid, d in dated_ids if conv_dates.get(cid, {}).get("source") == source]
        if len(source_dated) < 20:
            print(f"  {source}: too few conversations ({len(source_dated)})")
            continue

        monthly_src = {}
        for cid, date in source_dated:
            month = date[:7]
            monthly_src.setdefault(month, []).append(cid)

        cumul = []
        snaps = []
        for month in sorted(monthly_src.keys()):
            cumul.extend(monthly_src[month])
            n = len(cumul)
            idx_list = [id_to_idx[cid] for cid in cumul if cid in id_to_idx]
            edges = 0
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    if sim[idx_list[i], idx_list[j]] >= theta:
                        edges += 1
            if edges > 0:
                snaps.append({"nodes": n, "edges": edges})

        if len(snaps) >= 5:
            ns = np.array([s["nodes"] for s in snaps], dtype=float)
            es = np.array([s["edges"] for s in snaps], dtype=float)
            try:
                popt, _ = curve_fit(power_law, ns, es, p0=[1, 1.4], maxfev=5000)
                g = popt[1]
                print(f"  {source}: gamma = {g:.3f} ({len(snaps)} snapshots, {len(source_dated)} convs)")
                per_source_gamma[source] = float(g)
            except Exception as e:
                print(f"  {source}: fit failed ({e})")
        else:
            print(f"  {source}: too few snapshots ({len(snaps)})")

    # =========================================
    # Save results
    # =========================================
    results = {
        "densification": {
            "gamma": float(gamma) if gamma else None,
            "r_squared": float(r_squared) if r_squared else None,
            "theta": theta,
            "n_snapshots": len(snapshots),
            "snapshots": snapshots[-5:],  # last 5 for reference
        },
        "heaps": {
            "beta": float(beta) if beta else None,
            "r_squared": float(r_sq_heaps) if r_sq_heaps else None,
            "final_vocab": len(vocab) if vocab else None,
            "n_titles": len(titles),
        },
        "per_source_gamma": per_source_gamma,
        "prior_comparison": {
            "paper1_gamma": 1.405,
            "paper2_gamma": 1.410,
            "paper1_heaps_beta": 0.286,
        },
    }

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/temporal_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/results/temporal_analysis.json")
    conn.close()


if __name__ == "__main__":
    main()
