"""Temporal analysis: cumulative daily snapshots of network growth.

Adapts the methodology from the PLOS Complex Systems paper (temporal evolution
of ChatGPT network) for agentic workloads with daily resolution.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import stats

from agentic.extract import Session


def _parse_date(dt_str: str) -> Optional[datetime]:
    """Parse ISO-ish datetime string to date."""
    try:
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        return None


def _snapshot_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Lightweight metrics for temporal snapshots — O(n+e), no Louvain."""
    n = G.number_of_nodes()
    e = G.number_of_edges()
    components = list(nx.connected_components(G))
    giant = max(len(c) for c in components) if components else 0
    return {
        "node_count": n,
        "edge_count": e,
        "density": nx.density(G) if n > 1 else 0.0,
        "num_components": len(components),
        "giant_component_size": giant,
        "giant_component_fraction": giant / n if n > 0 else 0.0,
    }


def build_daily_snapshots(
    sessions: List[Session],
    edges: List[Tuple[str, str, float]],
) -> List[Dict[str, Any]]:
    """Build cumulative daily network snapshots.

    For each day in the observation period, constructs the network from all
    sessions and edges up to that day, then computes lightweight metrics
    (node/edge counts, density, components — no Louvain or clustering).

    Args:
        sessions: All sessions (parents + subagents).
        edges: List of (source_id, target_id, weight) tuples — semantic edges.

    Returns:
        List of dicts, one per day, with date + network growth metrics.
    """
    # Map session ID to date
    session_dates = {}
    for s in sessions:
        dt = _parse_date(s.created_at)
        if dt:
            session_dates[s.id] = dt.date()

    # Map sessions by ID for delegation metrics
    sessions_by_id = {s.id: s for s in sessions}

    # Get all unique dates, sorted
    all_dates = sorted(set(session_dates.values()))
    if not all_dates:
        return []

    # Group sessions by date for incremental construction
    nodes_by_date = {}
    for sid, d in session_dates.items():
        nodes_by_date.setdefault(d, []).append(sid)

    # Index edges by node for fast lookup
    node_edges = {}  # node -> list of (other_node, weight)
    for s, t, w in edges:
        node_edges.setdefault(s, []).append((t, w))
        node_edges.setdefault(t, []).append((s, w))

    snapshots = []
    G = nx.Graph()

    for day in all_dates:
        # Add new nodes for this day
        new_nodes = nodes_by_date.get(day, [])
        G.add_nodes_from(new_nodes)

        # Add edges where both endpoints are now in the graph
        for node in new_nodes:
            for other, w in node_edges.get(node, []):
                if other in G:
                    G.add_edge(node, other, weight=w)

        metrics = _snapshot_metrics(G)
        metrics["date"] = day.isoformat()

        # Delegation metrics for this snapshot
        parent_nodes = [
            sid for sid in G.nodes
            if sid in sessions_by_id
            and sessions_by_id[sid].parent_conversation_id is None
        ]
        child_nodes = [
            sid for sid in G.nodes
            if sid in sessions_by_id
            and sessions_by_id[sid].parent_conversation_id is not None
        ]
        metrics["num_parents"] = len(parent_nodes)
        metrics["num_children"] = len(child_nodes)

        # Mean fan-out for parents in this snapshot
        fan_outs = []
        for pid in parent_nodes:
            n_children = sum(
                1 for cid in child_nodes
                if sessions_by_id[cid].parent_conversation_id == pid
            )
            fan_outs.append(n_children)
        metrics["mean_fan_out"] = float(np.mean(fan_outs)) if fan_outs else 0.0

        snapshots.append(metrics)

    return snapshots


def fit_densification_law(
    snapshots: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Fit e(t) ~ n(t)^gamma via log-log OLS.

    The densification exponent gamma characterizes how edge count scales
    with node count. gamma > 1 indicates superlinear growth (densification).

    Returns:
        Dict with gamma, r_squared, p_value, intercept.
    """
    nodes = []
    edges = []
    for snap in snapshots:
        n = snap.get("node_count", 0)
        e = snap.get("edge_count", 0)
        if n > 1 and e > 0:
            nodes.append(np.log(n))
            edges.append(np.log(e))

    if len(nodes) < 3:
        return {"gamma": None, "r_squared": None, "p_value": None, "intercept": None}

    slope, intercept, r_value, p_value, _ = stats.linregress(nodes, edges)
    return {
        "gamma": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "intercept": intercept,
    }


def compute_preferential_attachment(
    snapshots: List[Dict[str, Any]],
    edges: List[Tuple[str, str, float]],
    sessions: List[Session],
) -> Dict[str, float]:
    """Estimate preferential attachment exponent beta.

    For each new edge (u, v) added in snapshot t, record the degree of u and v
    in the previous snapshot. Then fit P(k) ~ k^beta.

    Returns:
        Dict with beta, r_squared, p_value.
    """
    if len(snapshots) < 3:
        return {"beta": None, "r_squared": None, "p_value": None}

    # Map session ID to date
    session_dates = {}
    for s in sessions:
        dt = _parse_date(s.created_at)
        if dt:
            session_dates[s.id] = dt.date()

    all_dates = sorted(set(session_dates.values()))

    # Group sessions by date for incremental construction
    nodes_by_date = {}
    for sid, d in session_dates.items():
        nodes_by_date.setdefault(d, []).append(sid)

    # Index edges by node for fast lookup
    node_edges = {}  # node -> list of (other_node, weight)
    for s, t, w in edges:
        node_edges.setdefault(s, []).append((t, w))
        node_edges.setdefault(t, []).append((s, w))

    # Incrementally build the graph, recording degree of endpoints for new edges
    G = nx.Graph()
    degree_records = []

    for day in all_dates:
        new_nodes = nodes_by_date.get(day, [])

        # Record degrees of existing endpoints BEFORE adding new edges
        # For each new node, find edges to existing nodes
        new_edges_this_day = []
        for node in new_nodes:
            for other, w in node_edges.get(node, []):
                if other in G and not G.has_edge(node, other):
                    new_edges_this_day.append((node, other, w))

        # Record pre-attachment degree for endpoints of new edges
        for _, other, _ in new_edges_this_day:
            d = G.degree(other)
            if d > 0:
                degree_records.append(d)

        # Now add new nodes and their edges
        G.add_nodes_from(new_nodes)
        for node, other, w in new_edges_this_day:
            G.add_edge(node, other, weight=w)

        # Also add edges between new same-day nodes
        for i, node in enumerate(new_nodes):
            for other, w in node_edges.get(node, []):
                if other in G and not G.has_edge(node, other):
                    G.add_edge(node, other, weight=w)

    if len(degree_records) < 10:
        return {"beta": None, "r_squared": None, "p_value": None}

    # Bin degrees and fit log-log
    degree_counts = {}
    for d in degree_records:
        degree_counts[d] = degree_counts.get(d, 0) + 1

    total = sum(degree_counts.values())
    ks = sorted(degree_counts.keys())
    log_k = [np.log(k) for k in ks]
    log_p = [np.log(degree_counts[k] / total) for k in ks]

    if len(log_k) < 3:
        return {"beta": None, "r_squared": None, "p_value": None}

    slope, intercept, r_value, p_value, _ = stats.linregress(log_k, log_p)
    return {
        "beta": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
    }
