"""Operational query tools for the memex semantic graph.

Semantic search, neighbor recommendation, community browsing, and
bridge detection. Uses per-message embeddings composed into
conversation-level vectors with configurable role weighting.

Usage:
    # Semantic search (free text query)
    python query.py search "bernoulli type theory" --db data/analysis.db
    python query.py search "consciousness and moral agency" -k 10

    # Find neighbors of a conversation
    python query.py neighbors <conversation_id> --db data/analysis.db
    python query.py neighbors <conversation_id> -k 10 --threshold 0.7

    # List communities
    python query.py communities --db data/analysis.db

    # Show a community's conversations
    python query.py community 0 --db data/analysis.db

    # Find bridge conversations (connect different communities)
    python query.py bridges --db data/analysis.db
"""
import argparse
import json
import os
import sqlite3
import struct
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Lazy imports for optional features
community_louvain = None


def _ensure_louvain():
    global community_louvain
    if community_louvain is None:
        import community as _cl
        community_louvain = _cl


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def serialize_f32(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def open_db(db_path):
    import sqlite_vec
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def get_model_id(conn):
    rows = conn.execute("SELECT id, dimensions FROM embedding_models").fetchall()
    if not rows:
        print("No embedding models found. Run compute_embeddings.py first.")
        sys.exit(1)
    return rows[0][0], rows[0][1]


def compose_conversation_embeddings(conn, model_id, user_weight=3.0, asst_weight=1.0):
    """Compose conversation-level embeddings from per-message vectors."""
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

    # Group by conversation and role
    conv_role_vecs = {}
    for row in rows:
        conv_id, role = row[0], row[1]
        vec = deserialize_f32(row[2])
        conv_role_vecs.setdefault(conv_id, {}).setdefault(role, []).append(vec)

    ids = []
    vectors = []
    for conv_id, role_vecs in conv_role_vecs.items():
        components, weights = [], []
        user_vecs = role_vecs.get("user", [])
        asst_vecs = role_vecs.get("assistant", [])
        if user_vecs:
            components.append(np.mean(user_vecs, axis=0))
            weights.append(user_weight)
        if asst_vecs:
            components.append(np.mean(asst_vecs, axis=0))
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


def get_conv_meta(conn, conv_id):
    """Get conversation metadata."""
    row = conn.execute("""
        SELECT title, source, model, message_count, created_at, has_notes
        FROM conversations WHERE id = ?
    """, [conv_id]).fetchone()
    if not row:
        return {}
    return {
        "title": row[0], "source": row[1], "model": row[2],
        "message_count": row[3], "created_at": row[4], "has_notes": bool(row[5]),
    }


def get_notes(conn, conv_id):
    """Get notes for a conversation from the memex DB (if accessible)."""
    try:
        rows = conn.execute(
            "SELECT text FROM notes WHERE conversation_id = ?", [conv_id]
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def format_result(conv_id, meta, score=None, community=None):
    """Format a single result for display."""
    parts = []
    if score is not None:
        parts.append(f"  sim={score:.3f}")
    if community is not None:
        parts.append(f"  community={community}")
    title = meta.get("title", "Untitled")
    source = meta.get("source", "?")
    msgs = meta.get("message_count", "?")
    date = (meta.get("created_at") or "")[:10]
    notes = " [notes]" if meta.get("has_notes") else ""
    score_str = "".join(parts)
    return f"  {conv_id[:12]}  {title[:50]:<50s}  {source:>12s}  {str(msgs):>4s} msgs  {date}{notes}{score_str}"


def cmd_search(args):
    """Semantic search: embed query text, find nearest conversations."""
    from openai import OpenAI
    import tiktoken

    conn = open_db(args.db)
    model_id, dims = get_model_id(conn)
    _, model_name, _ = model_id.split(":", 2)
    provider = model_id.split(":")[0]

    # Embed the query
    client = OpenAI()
    response = client.embeddings.create(
        model=model_name,
        input=[args.query],
        dimensions=dims,
    )
    query_vec = np.array(response.data[0].embedding).reshape(1, -1)

    # Compose conversation embeddings
    ids, vectors = compose_conversation_embeddings(
        conn, model_id, user_weight=args.user_weight, asst_weight=args.asst_weight
    )

    # Compute similarities
    sims = cosine_similarity(query_vec, vectors)[0]
    top_indices = np.argsort(sims)[::-1][:args.k]

    print(f"\nSearch: \"{args.query}\"")
    print(f"Top {args.k} results (uw={args.user_weight}, aw={args.asst_weight}):\n")

    for rank, idx in enumerate(top_indices, 1):
        conv_id = ids[idx]
        score = sims[idx]
        meta = get_conv_meta(conn, conv_id)
        print(f"{rank:>3d}. {format_result(conv_id, meta, score=score)}")

    conn.close()


def cmd_neighbors(args):
    """Find conversations most similar to a given conversation."""
    conn = open_db(args.db)
    model_id, dims = get_model_id(conn)

    ids, vectors = compose_conversation_embeddings(
        conn, model_id, user_weight=args.user_weight, asst_weight=args.asst_weight
    )

    # Find the target conversation
    target_id = args.conversation_id
    # Support prefix matching
    matches = [i for i, cid in enumerate(ids) if cid.startswith(target_id)]
    if not matches:
        print(f"Conversation {target_id} not found in embeddings.")
        conn.close()
        return
    target_idx = matches[0]
    target_full_id = ids[target_idx]

    # Compute similarities
    target_vec = vectors[target_idx].reshape(1, -1)
    sims = cosine_similarity(target_vec, vectors)[0]
    sims[target_idx] = -1  # exclude self

    # Filter by threshold if given
    top_indices = np.argsort(sims)[::-1]
    if args.threshold:
        top_indices = [i for i in top_indices if sims[i] >= args.threshold][:args.k]
    else:
        top_indices = top_indices[:args.k]

    target_meta = get_conv_meta(conn, target_full_id)
    print(f"\nNeighbors of: {target_meta.get('title', target_full_id)}")
    print(f"  ({target_full_id}, {target_meta.get('source')}, {target_meta.get('created_at', '')[:10]})\n")

    for rank, idx in enumerate(top_indices, 1):
        conv_id = ids[idx]
        score = sims[idx]
        meta = get_conv_meta(conn, conv_id)
        print(f"{rank:>3d}. {format_result(conv_id, meta, score=score)}")

    conn.close()


def cmd_communities(args):
    """Detect and list communities in the semantic graph."""
    import networkx as nx
    _ensure_louvain()

    conn = open_db(args.db)
    model_id, dims = get_model_id(conn)

    ids, vectors = compose_conversation_embeddings(
        conn, model_id, user_weight=args.user_weight, asst_weight=args.asst_weight
    )

    # Build graph
    sim_matrix = cosine_similarity(vectors)
    G = nx.Graph()
    for i, cid in enumerate(ids):
        G.add_node(cid)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim_matrix[i, j] >= args.threshold:
                G.add_edge(ids[i], ids[j], weight=float(sim_matrix[i, j]))

    # Only partition the giant component
    components = list(nx.connected_components(G))
    gc_nodes = max(components, key=len) if components else set()
    gc = G.subgraph(gc_nodes).copy()

    partition = community_louvain.best_partition(gc)
    mod = community_louvain.modularity(partition, gc)

    # Group by community
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)

    print(f"\nCommunities (theta={args.threshold}, uw={args.user_weight})")
    print(f"Giant component: {len(gc_nodes)} nodes, modularity={mod:.3f}")
    print(f"{len(communities)} communities detected\n")

    for comm_id, members in sorted(communities.items(), key=lambda x: -len(x[1])):
        print(f"--- Community {comm_id} ({len(members)} conversations) ---")
        # Show up to 8 members, prioritize those with notes
        meta_list = [(m, get_conv_meta(conn, m)) for m in members]
        meta_list.sort(key=lambda x: (not x[1].get("has_notes"), x[1].get("created_at") or ""))
        for conv_id, meta in meta_list[:8]:
            print(format_result(conv_id, meta, community=comm_id))
        if len(members) > 8:
            print(f"  ... and {len(members) - 8} more")
        print()

    conn.close()


def cmd_bridges(args):
    """Find conversations that bridge different communities."""
    import networkx as nx
    _ensure_louvain()

    conn = open_db(args.db)
    model_id, dims = get_model_id(conn)

    ids, vectors = compose_conversation_embeddings(
        conn, model_id, user_weight=args.user_weight, asst_weight=args.asst_weight
    )

    sim_matrix = cosine_similarity(vectors)
    G = nx.Graph()
    for i, cid in enumerate(ids):
        G.add_node(cid)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim_matrix[i, j] >= args.threshold:
                G.add_edge(ids[i], ids[j], weight=float(sim_matrix[i, j]))

    components = list(nx.connected_components(G))
    gc_nodes = max(components, key=len) if components else set()
    gc = G.subgraph(gc_nodes).copy()

    partition = community_louvain.best_partition(gc)

    # Find inter-community edges per node
    bridge_scores = {}
    for node in gc.nodes():
        node_comm = partition[node]
        total_edges = gc.degree(node)
        if total_edges == 0:
            continue
        inter_edges = sum(1 for neighbor in gc.neighbors(node)
                         if partition[neighbor] != node_comm)
        bridge_scores[node] = inter_edges / total_edges

    # Also compute betweenness centrality
    betweenness = nx.betweenness_centrality(gc)

    # Rank by bridge score
    ranked = sorted(bridge_scores.items(), key=lambda x: -x[1])

    print(f"\nBridge conversations (theta={args.threshold})")
    print(f"Conversations with highest inter-community edge fraction:\n")
    print(f"{'rank':>4s}  {'id':>12s}  {'title':<45s}  {'comm':>4s}  {'bridge%':>7s}  {'between':>7s}")
    print("-" * 95)

    for rank, (conv_id, score) in enumerate(ranked[:args.k], 1):
        meta = get_conv_meta(conn, conv_id)
        title = meta.get("title", "")[:45]
        comm = partition[conv_id]
        btw = betweenness.get(conv_id, 0)
        print(f"{rank:>4d}  {conv_id[:12]}  {title:<45s}  {comm:>4d}  {score:>6.1%}  {btw:>7.4f}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Operational memex query tools")
    parser.add_argument("--db", default="data/analysis.db")
    parser.add_argument("--user-weight", type=float, default=3.0)
    parser.add_argument("--asst-weight", type=float, default=1.0)

    sub = parser.add_subparsers(dest="command")

    # search
    p = sub.add_parser("search", help="Semantic search by free text")
    p.add_argument("query", help="Search query text")
    p.add_argument("-k", type=int, default=10, help="Number of results")

    # neighbors
    p = sub.add_parser("neighbors", help="Find similar conversations")
    p.add_argument("conversation_id", help="Conversation ID (prefix match supported)")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--threshold", type=float, default=None,
                   help="Minimum similarity threshold")

    # communities
    p = sub.add_parser("communities", help="List detected communities")
    p.add_argument("--threshold", type=float, default=0.78)

    # bridges
    p = sub.add_parser("bridges", help="Find bridge conversations")
    p.add_argument("--threshold", type=float, default=0.78)
    p.add_argument("-k", type=int, default=20)

    args = parser.parse_args()
    if args.command == "search":
        cmd_search(args)
    elif args.command == "neighbors":
        cmd_neighbors(args)
    elif args.command == "communities":
        cmd_communities(args)
    elif args.command == "bridges":
        cmd_bridges(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
