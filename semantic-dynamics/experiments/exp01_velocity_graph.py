"""Experiment 1: Velocity-space graph.

Build a graph where conversations are connected not by topic similarity
(semantic space) but by transition similarity (velocity space).

"These two conversations are changing in the same way."

Compares the velocity graph to the semantic graph:
- Do they find the same communities?
- Do they identify the same hubs?
- What does the velocity graph reveal that the semantic graph misses?

Also builds a curvature-space graph: "these conversations turn
in the same way at the same structural moments."
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
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, os.path.expanduser("~/github/beta/embflow"))
import embflow as ef

ANALYSIS_DB = os.path.expanduser(
    "~/github/papers/cognitive-mri-ai-conversations/operational-memex/data/analysis.db"
)


def deserialize_f32(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def load_data():
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

    # Load per-conversation message embeddings
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
        if not r[2]:  # skip short
            conv_msgs[r[0]].append(np.array(deserialize_f32(r[3])))

    conn.close()
    return conv_msgs, meta


def compute_representations(conv_msgs, alpha=0.85):
    """Compute semantic, velocity, and curvature representations per conversation."""
    results = {}
    lens = ef.Exponential(alpha)

    for cid, vecs_list in conv_msgs.items():
        if len(vecs_list) < 10:
            continue
        vecs = np.array(vecs_list)

        # Semantic: standard projection
        semantic = lens.project(vecs)

        # Trajectory
        traj = lens.trajectory(vecs)

        # Velocity: project the velocity vectors through a lens
        vel = ef.velocity(traj)
        if len(vel) < 3:
            continue
        vel_proj = ef.Uniform().project(vel)  # mean velocity

        # Also: exponential velocity (recent transitions weighted more)
        vel_exp = ef.Exponential(alpha).project(vel)

        # Curvature: project curvature vectors
        curv = ef.curvature(traj)
        if len(curv) < 2:
            continue
        curv_proj = ef.Uniform().project(curv)

        # Speed profile (scalar): mean and std of speed
        spd = ef.speed(traj)
        mean_speed = float(np.mean(spd))
        std_speed = float(np.std(spd))

        # Angular velocity profile
        ang = ef.angular_velocity(traj)
        mean_ang = float(np.mean(ang))

        # Drift
        d = ef.drift(traj)

        results[cid] = {
            "semantic": semantic,
            "velocity_mean": vel_proj,
            "velocity_exp": vel_exp,
            "curvature": curv_proj,
            "drift": d,
            "mean_speed": mean_speed,
            "std_speed": std_speed,
            "mean_angular_velocity": mean_ang,
            "n_messages": len(vecs),
        }

    return results


def build_graph(ids, embeddings, threshold):
    sim = cosine_similarity(embeddings)
    G = nx.Graph()
    for i, cid in enumerate(ids):
        G.add_node(cid)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim[i, j] >= threshold:
                G.add_edge(ids[i], ids[j], weight=float(sim[i, j]))
    return G, sim


def graph_metrics(G, label=""):
    comps = list(nx.connected_components(G))
    gc = G.subgraph(max(comps, key=len)).copy() if comps else G
    result = {
        "label": label,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "gc_nodes": gc.number_of_nodes(),
        "gc_fraction": gc.number_of_nodes() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        "isolates": nx.number_of_isolates(G),
    }
    if gc.number_of_nodes() >= 10:
        partition = community_louvain.best_partition(gc)
        result["modularity"] = community_louvain.modularity(partition, gc)
        result["num_communities"] = len(set(partition.values()))
        result["clustering"] = nx.average_clustering(gc)
        result["partition"] = partition
    return result


def main():
    print("Loading data...")
    conv_msgs, meta = load_data()
    print(f"  {len(conv_msgs)} conversations")

    print("Computing representations...")
    reps = compute_representations(conv_msgs, alpha=0.85)
    print(f"  {len(reps)} conversations with full representations")

    annotated = set(cid for cid in reps if meta.get(cid, {}).get("notes"))

    ids = sorted(reps.keys())
    semantic_embs = np.array([reps[c]["semantic"] for c in ids])
    velocity_embs = np.array([reps[c]["velocity_mean"] for c in ids])
    velocity_exp_embs = np.array([reps[c]["velocity_exp"] for c in ids])
    curvature_embs = np.array([reps[c]["curvature"] for c in ids])

    # ============================================================
    # ANALYSIS 1: Build graphs in each space
    # ============================================================
    spaces = {
        "semantic": semantic_embs,
        "velocity_mean": velocity_embs,
        "velocity_exp": velocity_exp_embs,
        "curvature": curvature_embs,
    }

    # Find appropriate thresholds for each space
    # (different spaces have different similarity distributions)
    print("\n=== Similarity Distributions ===")
    for name, embs in spaces.items():
        sim = cosine_similarity(embs)
        upper = sim[np.triu_indices_from(sim, k=1)]
        print(f"  {name:20s}: mean={np.mean(upper):.4f}, std={np.std(upper):.4f}, "
              f"p90={np.percentile(upper, 90):.4f}, p95={np.percentile(upper, 95):.4f}, "
              f"p99={np.percentile(upper, 99):.4f}")

    # Use p95 as threshold for each space (comparable density)
    print("\n=== Graph Construction (threshold = p95 per space) ===")
    graphs = {}
    partitions = {}

    for name, embs in spaces.items():
        sim = cosine_similarity(embs)
        upper = sim[np.triu_indices_from(sim, k=1)]
        threshold = np.percentile(upper, 95)

        G, _ = build_graph(ids, embs, threshold)
        m = graph_metrics(G, label=name)
        graphs[name] = G
        if "partition" in m:
            partitions[name] = m.pop("partition")

        print(f"  {name:20s}: theta={threshold:.3f}, edges={m['edges']}, "
              f"GC={m['gc_nodes']} ({m['gc_fraction']:.1%}), "
              f"mod={m.get('modularity', 'N/A')}, "
              f"comm={m.get('num_communities', 'N/A')}")

    # ============================================================
    # ANALYSIS 2: Do the spaces agree on communities?
    # ============================================================
    print("\n=== Community Agreement (NMI) ===")
    space_names = [n for n in spaces if n in partitions]
    for i, a in enumerate(space_names):
        for b in space_names[i + 1:]:
            # Align: only nodes in both GCs
            shared = set(partitions[a].keys()) & set(partitions[b].keys())
            if len(shared) < 20:
                print(f"  {a} vs {b}: too few shared nodes ({len(shared)})")
                continue
            shared = sorted(shared)
            labels_a = [partitions[a][n] for n in shared]
            labels_b = [partitions[b][n] for n in shared]
            nmi = normalized_mutual_info_score(labels_a, labels_b)
            print(f"  {a:20s} vs {b:20s}: NMI={nmi:.3f} ({len(shared)} shared nodes)")

    # ============================================================
    # ANALYSIS 3: Do the spaces agree on node importance?
    # ============================================================
    print("\n=== Degree Correlation Between Spaces ===")
    for i, a in enumerate(space_names):
        for b in space_names[i + 1:]:
            shared = set(graphs[a].nodes()) & set(graphs[b].nodes())
            if len(shared) < 20:
                continue
            shared = sorted(shared)
            deg_a = [graphs[a].degree(n) for n in shared]
            deg_b = [graphs[b].degree(n) for n in shared]
            rho, p = spearmanr(deg_a, deg_b)
            print(f"  {a:20s} vs {b:20s}: Spearman rho={rho:.3f}, p={p:.4f}")

    # ============================================================
    # ANALYSIS 4: What does velocity space reveal uniquely?
    # ============================================================
    print("\n=== Velocity Space: Unique Hubs ===")
    print("Conversations that are velocity hubs but NOT semantic hubs:")

    sem_deg = dict(graphs["semantic"].degree()) if "semantic" in graphs else {}
    vel_deg = dict(graphs["velocity_mean"].degree()) if "velocity_mean" in graphs else {}

    # Velocity hubs not in semantic top-50
    sem_top50 = set(sorted(sem_deg, key=sem_deg.get, reverse=True)[:50])
    vel_top50 = set(sorted(vel_deg, key=vel_deg.get, reverse=True)[:50])
    vel_unique = vel_top50 - sem_top50

    for cid in sorted(vel_unique, key=lambda c: -vel_deg.get(c, 0))[:15]:
        md = meta.get(cid, {})
        ann = " [notes]" if md.get("notes") else ""
        print(f"  vel_deg={vel_deg[cid]:>4d}  sem_deg={sem_deg.get(cid, 0):>4d}  "
              f"{md.get('title', '?')[:45]:<45s}  {md.get('source', ''):>12s}{ann}")

    print("\nConversations that are semantic hubs but NOT velocity hubs:")
    sem_unique = sem_top50 - vel_top50
    for cid in sorted(sem_unique, key=lambda c: -sem_deg.get(c, 0))[:15]:
        md = meta.get(cid, {})
        ann = " [notes]" if md.get("notes") else ""
        print(f"  sem_deg={sem_deg[cid]:>4d}  vel_deg={vel_deg.get(cid, 0):>4d}  "
              f"{md.get('title', '?')[:45]:<45s}  {md.get('source', ''):>12s}{ann}")

    # ============================================================
    # ANALYSIS 5: Velocity communities = conversation archetypes?
    # ============================================================
    if "velocity_mean" in partitions:
        print("\n=== Velocity Communities (Conversation Archetypes) ===")
        vel_partition = partitions["velocity_mean"]
        vel_gc_nodes = set(vel_partition.keys())

        comms = defaultdict(list)
        for n, c in vel_partition.items():
            comms[c].append(n)

        for comm_id, members in sorted(comms.items(), key=lambda x: -len(x[1])):
            if len(members) < 10:
                continue
            # Characterize: mean drift, mean speed, mean angular velocity
            drifts = [reps[m]["drift"] for m in members]
            speeds = [reps[m]["mean_speed"] for m in members]
            ang_vels = [reps[m]["mean_angular_velocity"] for m in members]
            sources = defaultdict(int)
            for m in members:
                sources[meta.get(m, {}).get("source", "?")] += 1

            print(f"\n  Community {comm_id} ({len(members)} conversations):")
            print(f"    Drift:    mean={np.mean(drifts):.3f}, median={np.median(drifts):.3f}")
            print(f"    Speed:    mean={np.mean(speeds):.4f}")
            print(f"    Angular:  mean={np.mean(ang_vels):.4f}")
            print(f"    Sources:  {dict(sources)}")
            # Show 5 representative members
            for m in sorted(members, key=lambda x: -reps[x]["drift"])[:5]:
                md = meta.get(m, {})
                ann = " [notes]" if md.get("notes") else ""
                print(f"    - drift={reps[m]['drift']:.3f}  "
                      f"{md.get('title', '?')[:45]:<45s}  {md.get('source', ''):>12s}{ann}")

    # ============================================================
    # ANALYSIS 6: Annotated nodes in velocity space
    # ============================================================
    if annotated and "velocity_mean" in partitions:
        print("\n=== Annotated Nodes in Velocity Space ===")
        vel_deg_ann = [vel_deg.get(c, 0) for c in annotated if c in vel_deg]
        vel_deg_non = [vel_deg.get(c, 0) for c in ids if c not in annotated and c in vel_deg]

        if vel_deg_ann:
            from scipy.stats import mannwhitneyu
            U, p = mannwhitneyu(vel_deg_ann, vel_deg_non, alternative="greater")
            print(f"  Annotated velocity degree: mean={np.mean(vel_deg_ann):.1f}")
            print(f"  Non-annotated:             mean={np.mean(vel_deg_non):.1f}")
            print(f"  Mann-Whitney U p={p:.4f}")

    # Save
    os.makedirs("experiments/results", exist_ok=True)
    save = {}
    for name in spaces:
        if name in graphs:
            m = graph_metrics(graphs[name], label=name)
            m.pop("partition", None)
            save[name] = m
    with open("experiments/results/velocity_graph.json", "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nSaved to experiments/results/velocity_graph.json")


if __name__ == "__main__":
    main()
