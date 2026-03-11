"""CLI runner for the agentic analysis pipeline.

Chains: extract -> preprocess -> export JSON -> (external: Ollama embeddings)
        -> load edges -> build networks -> compute metrics -> save results.

Usage:
    # Extract + export sessions for embedding (step 1)
    python -m agentic.run --extract-only \
        --db ~/.memex/default/conversations.db \
        --output-dir agentic/data/sessions-text-only \
        --content-mode text-only

    # Full analysis (after embeddings generated externally)
    python -m agentic.run \
        --edges-file agentic/data/edges-t0.9.json \
        --db ~/.memex/default/conversations.db \
        --output-dir agentic/output/analysis-001
"""
import argparse
import json
import sys
from pathlib import Path

import networkx as nx

from agentic.extract import extract_sessions
from agentic.export_json import export_sessions_to_json
from agentic.preprocess import (  # noqa: F401 — available for future ablation
    extract_text_full,
    extract_text_only,
    extract_user_only,
    extract_tool_names_only,
)
from agentic.delegation import (
    build_delegation_graph,
    compute_fan_out_distribution,
    compute_delegation_ratio,
    agent_type_counts,
)
from agentic.semantic import compute_network_metrics
from agentic.temporal import (
    build_daily_snapshots,
    fit_densification_law,
    compute_preferential_attachment,
)
from agentic.multilayer import (
    compute_interlayer_degree_correlation,
    compute_participation_coefficient,
    compare_community_assignments,
)
from agentic.null_models import (
    configuration_model_baseline,
    permutation_interlayer_test,
    permutation_nmi_test,
)

try:
    import community.community_louvain as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False


def do_extract(args):
    """Extract sessions and export to JSON."""
    print(f"Extracting sessions from {args.db}...")
    sessions = extract_sessions(
        args.db,
        include_subagents=args.include_subagents,
    )
    print(f"  {len(sessions)} sessions extracted")

    parents = [s for s in sessions if s.parent_conversation_id is None]
    children = [s for s in sessions if s.parent_conversation_id is not None]
    print(f"  {len(parents)} parents, {len(children)} subagents")

    print(f"Exporting to {args.output_dir} (text-only mode)...")
    export_sessions_to_json(sessions, args.output_dir)
    print(f"  {len(sessions)} JSON files written")

    return sessions


def do_analyze(args):
    """Run full analysis on extracted sessions + edges."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load sessions
    print(f"Extracting sessions from {args.db}...")
    sessions = extract_sessions(args.db, include_subagents=args.include_subagents)
    print(f"  {len(sessions)} sessions")

    # Load edges and translate filename-safe IDs back to session IDs
    # (export_json replaces : with _ for filesystem safety)
    print(f"Loading edges from {args.edges_file}...")
    with open(args.edges_file) as f:
        raw_edges = json.load(f)
    session_ids = {s.id for s in sessions}
    safe_to_id = {sid.replace(":", "_"): sid for sid in session_ids}
    edges = [
        (safe_to_id.get(s, s), safe_to_id.get(t, t), w)
        for s, t, w in raw_edges
    ]
    print(f"  {len(edges)} edges loaded")

    # --- Delegation network ---
    print("Building delegation network...")
    deleg_G = build_delegation_graph(sessions)
    fan_out = compute_fan_out_distribution(deleg_G)
    del_ratio = compute_delegation_ratio(deleg_G)
    type_counts = agent_type_counts(deleg_G)

    deleg_results = {
        "num_nodes": deleg_G.number_of_nodes(),
        "num_edges": deleg_G.number_of_edges(),
        "agent_type_counts": dict(type_counts),
        "fan_out_stats": {
            "mean": sum(fan_out.values()) / len(fan_out) if fan_out else 0,
            "max": max(fan_out.values()) if fan_out else 0,
            "min": min(fan_out.values()) if fan_out else 0,
        },
        "delegation_ratio_stats": {
            "mean": sum(del_ratio.values()) / len(del_ratio) if del_ratio else 0,
            "max": max(del_ratio.values()) if del_ratio else 0,
        },
    }
    with open(out / "delegation.json", "w") as f:
        json.dump(deleg_results, f, indent=2)
    print(f"  Delegation: {deleg_G.number_of_nodes()} nodes, {deleg_G.number_of_edges()} edges")

    # --- Semantic network ---
    print("Building semantic network...")
    sem_G = nx.Graph()
    for s, t, w in edges:
        if s in session_ids and t in session_ids:
            sem_G.add_edge(s, t, weight=w)
    # Add isolated nodes
    for sid in session_ids:
        if sid not in sem_G:
            sem_G.add_node(sid)

    sem_metrics = compute_network_metrics(sem_G)
    with open(out / "semantic.json", "w") as f:
        json.dump(sem_metrics, f, indent=2, default=str)
    print(f"  Semantic: {sem_metrics['node_count']} nodes, {sem_metrics['edge_count']} edges")
    print(f"  Modularity: {sem_metrics.get('modularity', 'N/A')}")

    # --- Temporal analysis ---
    print("Building temporal snapshots...")
    snapshots = build_daily_snapshots(sessions, edges)
    densification = fit_densification_law(snapshots)

    pref_attach = compute_preferential_attachment(snapshots, edges, sessions)

    temporal_results = {
        "num_snapshots": len(snapshots),
        "densification": densification,
        "preferential_attachment": pref_attach,
        "snapshots": snapshots,
    }
    with open(out / "temporal.json", "w") as f:
        json.dump(temporal_results, f, indent=2, default=str)
    print(f"  {len(snapshots)} daily snapshots")
    print(f"  Densification gamma: {densification.get('gamma', 'N/A')}")
    print(f"  Preferential attachment beta: {pref_attach.get('beta', 'N/A')}")

    # --- Multi-layer analysis ---
    print("Computing multi-layer metrics...")
    interlayer = compute_interlayer_degree_correlation(sem_G, deleg_G)
    participation = compute_participation_coefficient(sem_G, deleg_G)

    # Community comparison
    nmi = 0.0
    if HAS_COMMUNITY and sem_G.number_of_edges() > 0:
        sem_partition = community_louvain.best_partition(sem_G, random_state=42)
        # For delegation: use weakly connected components as communities
        deleg_undirected = deleg_G.to_undirected()
        del_partition = {}
        for i, comp in enumerate(nx.connected_components(deleg_undirected)):
            for node in comp:
                del_partition[node] = i
        nmi = compare_community_assignments(sem_partition, del_partition)

    multilayer_results = {
        "interlayer_correlation": interlayer,
        "participation_stats": {
            "mean": sum(participation.values()) / len(participation) if participation else 0,
            "max": max(participation.values()) if participation else 0,
        },
        "community_nmi": nmi,
    }
    with open(out / "multilayer.json", "w") as f:
        json.dump(multilayer_results, f, indent=2, default=str)
    print(f"  Interlayer correlation: {interlayer.get('correlation', 'N/A')}")
    print(f"  Community NMI: {nmi:.4f}")

    # --- Null model baselines ---
    print("Running null model baselines...")

    # Configuration model for semantic network
    print("  Configuration model baseline (semantic)...")
    config_result = configuration_model_baseline(sem_G, n_samples=20, random_state=42)
    for metric, vals in config_result.items():
        obs = vals.get("observed")
        z = vals.get("z_score")
        p = vals.get("p_value")
        if obs is not None and z is not None and p is not None:
            print(f"    {metric}: obs={obs:.4f}, z={z:.2f}, p={p:.4f}")

    # Permutation test for inter-layer degree correlation
    print("  Permutation test (interlayer degree correlation)...")
    perm_interlayer = permutation_interlayer_test(
        sem_G, deleg_G, n_perms=1000, random_state=42
    )
    if perm_interlayer["observed_rho"] is not None:
        print(f"    rho={perm_interlayer['observed_rho']:.4f}, "
              f"p={perm_interlayer['perm_p_value']:.4f}")
    else:
        print(f"    Too few shared nodes ({perm_interlayer['n_shared']})")

    # Permutation test for community NMI
    perm_nmi_result = {"observed_nmi": 0.0, "perm_p_value": None, "n_shared": 0}
    if HAS_COMMUNITY and sem_G.number_of_edges() > 0:
        print("  Permutation test (community NMI)...")
        perm_nmi_result = permutation_nmi_test(
            sem_partition, del_partition, n_perms=1000, random_state=42
        )
        print(f"    NMI={perm_nmi_result['observed_nmi']:.4f}, "
              f"p={perm_nmi_result['perm_p_value']:.4f}")

    null_results = {
        "configuration_model": config_result,
        "permutation_interlayer": perm_interlayer,
        "permutation_nmi": perm_nmi_result,
    }
    with open(out / "null_models.json", "w") as f:
        json.dump(null_results, f, indent=2, default=str)

    print(f"\nResults written to {out}/")


def main():
    parser = argparse.ArgumentParser(description="Agentic Cognitive MRI analysis pipeline")
    parser.add_argument("--db", required=True, help="Path to memex SQLite database")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--include-subagents", action="store_true", default=True,
                        help="Include subagent sessions (default: True)")
    parser.add_argument("--no-subagents", dest="include_subagents", action="store_false",
                        help="Exclude subagent sessions")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract and export sessions (no analysis)")
    parser.add_argument("--edges-file", help="Pre-computed edges JSON file (for analysis)")

    args = parser.parse_args()

    if args.extract_only:
        do_extract(args)
    else:
        if not args.edges_file:
            parser.error("--edges-file is required for analysis (run --extract-only first)")
        do_analyze(args)


if __name__ == "__main__":
    main()
