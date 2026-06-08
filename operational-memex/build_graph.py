"""Build semantic similarity graph from per-message embeddings.

Composes conversation-level embeddings from stored message vectors using
configurable role weighting, then constructs a similarity graph and
computes network metrics.

Usage:
    python build_graph.py --db data/analysis.db --threshold 0.9
    python build_graph.py --db data/analysis.db --threshold 0.9 --user-weight 3.0
    python build_graph.py --db data/analysis.db --sweep 0.80 0.95 0.01
    python build_graph.py --db data/analysis.db --per-source
    python build_graph.py --db data/analysis.db --weight-sweep 0.5 10.0 0.5
"""
import argparse
import json
import os
import sqlite3
import struct
import sys

import community as community_louvain
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def deserialize_f32(blob):
    """Deserialize bytes to float list from sqlite-vec."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def get_model_id(conn):
    """Get the (single) model ID from the database."""
    rows = conn.execute("SELECT id FROM embedding_models").fetchall()
    if not rows:
        print("No embedding models registered. Run compute_embeddings.py first.")
        sys.exit(1)
    if len(rows) > 1:
        print("Multiple models found. Use --model-id to select one:")
        for r in rows:
            print(f"  {r[0]}")
        sys.exit(1)
    return rows[0][0]


def compose_conversation_embeddings(conn, model_id, user_weight=1.0, asst_weight=1.0,
                                    skip_short=False, source_filter=None):
    """Compose conversation-level embeddings from per-message vectors.

    For each conversation, computes:
        user_emb  = mean(user message embeddings)
        asst_emb  = mean(assistant message embeddings)
        conv_emb  = normalize(user_weight * user_emb + asst_weight * asst_emb)

    Returns (ids, vectors, metadata) for conversations with at least one
    embeddable message.
    """
    # Get conversations
    query = "SELECT id, source, title, created_at, message_count, has_notes, parent_conversation_id FROM conversations"
    params = []
    if source_filter:
        query += " WHERE source = ?"
        params = [source_filter]
    convs = {r[0]: {
        "source": r[1], "title": r[2], "created_at": r[3],
        "message_count": r[4], "has_notes": bool(r[5]),
        "parent_conversation_id": r[6],
    } for r in conn.execute(query, params).fetchall()}

    if not convs:
        return [], np.array([]), []

    # Get all message embeddings grouped by conversation
    short_filter = "AND me.is_short = 0" if skip_short else ""
    rows = conn.execute(f"""
        SELECT me.conversation_id, me.message_id, me.role,
               mv.embedding
        FROM message_embeddings me
        JOIN message_vectors mv
          ON mv.conversation_id = me.conversation_id
          AND mv.message_id = me.message_id
          AND mv.model_id = me.model_id
        WHERE me.model_id = ?
        {short_filter}
        ORDER BY me.conversation_id
    """, [model_id]).fetchall()

    # Group by conversation and role
    conv_role_vecs = {}  # conv_id -> role -> [vectors]
    for row in rows:
        conv_id = row[0]
        if conv_id not in convs:
            continue
        role = row[2]
        vec = deserialize_f32(row[3])
        conv_role_vecs.setdefault(conv_id, {}).setdefault(role, []).append(vec)

    # Compose conversation embeddings
    ids = []
    vectors = []
    metadata = []

    for conv_id, role_vecs in conv_role_vecs.items():
        user_vecs = role_vecs.get("user", [])
        asst_vecs = role_vecs.get("assistant", [])

        components = []
        weights = []

        if user_vecs:
            user_emb = np.mean(user_vecs, axis=0)
            components.append(user_emb)
            weights.append(user_weight)

        if asst_vecs:
            asst_emb = np.mean(asst_vecs, axis=0)
            components.append(asst_emb)
            weights.append(asst_weight)

        if not components:
            continue

        # Weighted combination
        conv_emb = np.average(components, axis=0, weights=weights)
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(conv_emb)
        if norm > 0:
            conv_emb = conv_emb / norm

        ids.append(conv_id)
        vectors.append(conv_emb)
        meta = convs.get(conv_id, {})
        meta["user_msg_count"] = len(user_vecs)
        meta["asst_msg_count"] = len(asst_vecs)
        metadata.append(meta)

    return ids, np.array(vectors), metadata


def build_similarity_graph(ids, vectors, metadata, threshold=0.9):
    """Build graph from cosine similarity above threshold."""
    print(f"Computing pairwise similarity for {len(ids)} nodes...")
    sim_matrix = cosine_similarity(vectors)

    G = nx.Graph()
    for i, (cid, meta) in enumerate(zip(ids, metadata)):
        G.add_node(cid, **meta)

    edge_count = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim_matrix[i, j] >= threshold:
                G.add_edge(ids[i], ids[j], weight=float(sim_matrix[i, j]))
                edge_count += 1

    print(f"Graph: {G.number_of_nodes()} nodes, {edge_count} edges (theta={threshold})")
    return G, sim_matrix


def store_edges(conn, G):
    """Store edges in analysis database."""
    conn.execute("DELETE FROM edges")
    for u, v, data in G.edges(data=True):
        conn.execute(
            "INSERT INTO edges (src, dst, weight) VALUES (?, ?, ?)",
            [u, v, data["weight"]]
        )
    conn.commit()
    print(f"Stored {G.number_of_edges()} edges in database")


def compute_metrics(G, label=""):
    """Compute standard network metrics."""
    results = {"label": label}
    results["nodes"] = G.number_of_nodes()
    results["edges"] = G.number_of_edges()
    results["density"] = nx.density(G)

    # Connected components
    components = list(nx.connected_components(G))
    results["components"] = len(components)
    gc = G.subgraph(max(components, key=len)).copy() if components else G
    results["gc_nodes"] = gc.number_of_nodes()
    results["gc_edges"] = gc.number_of_edges()
    results["gc_fraction"] = gc.number_of_nodes() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

    # Degree distribution
    degrees = [d for _, d in G.degree()]
    if degrees:
        results["degree_mean"] = float(np.mean(degrees))
        results["degree_median"] = float(np.median(degrees))
        results["degree_max"] = max(degrees)

    results["isolates"] = nx.number_of_isolates(G)

    if gc.number_of_nodes() < 3:
        return results

    # Clustering
    results["clustering"] = nx.average_clustering(gc)

    # Average shortest path
    if gc.number_of_nodes() <= 5000:
        try:
            results["avg_path_length"] = nx.average_shortest_path_length(gc)
        except nx.NetworkXError:
            pass

    # Community detection (Louvain)
    partition = community_louvain.best_partition(gc)
    results["modularity"] = community_louvain.modularity(partition, gc)
    results["num_communities"] = len(set(partition.values()))

    comm_sizes = {}
    for node, comm in partition.items():
        comm_sizes[comm] = comm_sizes.get(comm, 0) + 1
    results["community_sizes"] = sorted(comm_sizes.values(), reverse=True)

    # Small-world coefficient
    if "avg_path_length" in results and results["clustering"] > 0:
        n = gc.number_of_nodes()
        m = gc.number_of_edges()
        k_mean = 2 * m / n
        C_rand = k_mean / n
        L_rand = np.log(n) / np.log(k_mean) if k_mean > 1 else float("inf")
        if C_rand > 0 and L_rand > 0 and L_rand != float("inf"):
            sigma = (results["clustering"] / C_rand) / (results["avg_path_length"] / L_rand)
            results["small_world_sigma"] = float(sigma)

    # Assortativity
    try:
        results["degree_assortativity"] = nx.degree_assortativity_coefficient(gc)
    except Exception:
        pass

    return results


def threshold_sweep(ids, vectors, metadata, low, high, step):
    """Sweep threshold values and report topology changes."""
    print(f"\nThreshold sweep: {low:.2f} to {high:.2f}, step {step:.2f}")
    sim_matrix = cosine_similarity(vectors)
    results = []
    thresholds = np.arange(low, high + step / 2, step)

    for theta in thresholds:
        G = nx.Graph()
        for i, (cid, meta) in enumerate(zip(ids, metadata)):
            G.add_node(cid, **meta)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if sim_matrix[i, j] >= theta:
                    G.add_edge(ids[i], ids[j], weight=float(sim_matrix[i, j]))

        components = list(nx.connected_components(G))
        gc = G.subgraph(max(components, key=len)).copy() if components else G

        entry = {
            "threshold": round(float(theta), 3),
            "edges": G.number_of_edges(),
            "components": len(components),
            "gc_nodes": gc.number_of_nodes(),
            "gc_fraction": round(gc.number_of_nodes() / G.number_of_nodes(), 3) if G.number_of_nodes() > 0 else 0,
            "isolates": nx.number_of_isolates(G),
        }
        results.append(entry)
        print(f"  theta={theta:.3f}: {entry['edges']:>6d} edges, "
              f"GC={entry['gc_nodes']:>5d} ({entry['gc_fraction']:.1%}), "
              f"{entry['components']} comp, {entry['isolates']} isolates")

    return results


def weight_sweep(conn, model_id, threshold, low, high, step, output_dir):
    """Sweep user/assistant weight ratio and measure modularity."""
    print(f"\nWeight sweep: user_weight {low:.1f} to {high:.1f}, step {step:.1f}")
    print(f"  (assistant_weight fixed at 1.0, threshold={threshold})")
    results = []

    for uw in np.arange(low, high + step / 2, step):
        ids, vectors, metadata = compose_conversation_embeddings(
            conn, model_id, user_weight=uw, asst_weight=1.0
        )
        G, _ = build_similarity_graph(ids, vectors, metadata, threshold)
        metrics = compute_metrics(G, label=f"uw={uw:.1f}")
        entry = {
            "user_weight": round(float(uw), 2),
            "modularity": metrics.get("modularity"),
            "num_communities": metrics.get("num_communities"),
            "edges": metrics["edges"],
            "gc_nodes": metrics["gc_nodes"],
            "clustering": metrics.get("clustering"),
        }
        results.append(entry)
        print(f"  uw={uw:.1f}: mod={entry['modularity']:.3f}, "
              f"comm={entry['num_communities']}, edges={entry['edges']}")

    out_path = os.path.join(output_dir, "weight_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWeight sweep results: {out_path}")
    return results


def per_source_analysis(conn, model_id, user_weight, asst_weight, threshold):
    """Run analysis per source platform."""
    sources = [r[0] for r in conn.execute(
        "SELECT DISTINCT source FROM conversations"
    ).fetchall()]
    results = {}

    for source in sorted(sources):
        ids, vectors, metadata = compose_conversation_embeddings(
            conn, model_id, user_weight=user_weight, asst_weight=asst_weight,
            source_filter=source
        )
        if len(ids) < 10:
            print(f"  {source}: only {len(ids)} nodes, skipping")
            continue

        G, _ = build_similarity_graph(ids, vectors, metadata, threshold)
        metrics = compute_metrics(G, label=source)
        results[source] = metrics
        print(f"\n  {source}: {metrics['gc_nodes']} GC nodes, "
              f"{metrics['edges']} edges, "
              f"mod={metrics.get('modularity', 'N/A')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Build semantic similarity graph")
    parser.add_argument("--db", default="data/analysis.db")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--user-weight", type=float, default=1.0,
                        help="Weight for user message embeddings (default: 1.0)")
    parser.add_argument("--asst-weight", type=float, default=1.0,
                        help="Weight for assistant message embeddings (default: 1.0)")
    parser.add_argument("--skip-short", action="store_true",
                        help="Exclude short messages from composition")
    parser.add_argument("--source", help="Analyze only this source")
    parser.add_argument("--sweep", nargs=3, type=float, metavar=("LOW", "HIGH", "STEP"),
                        help="Sweep threshold range")
    parser.add_argument("--weight-sweep", nargs=3, type=float, metavar=("LOW", "HIGH", "STEP"),
                        help="Sweep user weight (assistant=1.0)")
    parser.add_argument("--per-source", action="store_true")
    parser.add_argument("--output", default="experiments/results")
    args = parser.parse_args()

    import sqlite_vec
    conn = sqlite3.connect(args.db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    model_id = get_model_id(conn)
    os.makedirs(args.output, exist_ok=True)

    # Weight sweep mode
    if args.weight_sweep:
        weight_sweep(conn, model_id, args.threshold, *args.weight_sweep, args.output)
        conn.close()
        return

    # Compose conversation embeddings
    ids, vectors, metadata = compose_conversation_embeddings(
        conn, model_id,
        user_weight=args.user_weight,
        asst_weight=args.asst_weight,
        skip_short=args.skip_short,
        source_filter=args.source,
    )
    print(f"Composed {len(ids)} conversation embeddings "
          f"(uw={args.user_weight}, aw={args.asst_weight})")

    # Threshold sweep mode
    if args.sweep:
        results = threshold_sweep(ids, vectors, metadata, *args.sweep)
        out_path = os.path.join(args.output, "threshold_sweep.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSweep results: {out_path}")
        conn.close()
        return

    # Build graph
    G, sim_matrix = build_similarity_graph(ids, vectors, metadata, args.threshold)
    store_edges(conn, G)

    # Full corpus analysis
    print("\n--- Full corpus ---")
    full_metrics = compute_metrics(G, label="full_corpus")
    full_metrics["user_weight"] = args.user_weight
    full_metrics["asst_weight"] = args.asst_weight
    full_metrics["threshold"] = args.threshold
    print(json.dumps(full_metrics, indent=2, default=str))

    out_path = os.path.join(args.output, "full_corpus.json")
    with open(out_path, "w") as f:
        json.dump(full_metrics, f, indent=2, default=str)

    # Per-source analysis
    if args.per_source:
        print("\n--- Per-source analysis ---")
        source_results = per_source_analysis(
            conn, model_id, args.user_weight, args.asst_weight, args.threshold
        )
        out_path = os.path.join(args.output, "per_source.json")
        with open(out_path, "w") as f:
            json.dump(source_results, f, indent=2, default=str)

    conn.close()
    print(f"\nResults written to {args.output}/")


if __name__ == "__main__":
    main()
