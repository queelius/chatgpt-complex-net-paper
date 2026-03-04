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
from agentic.semantic import compute_network_metrics


def _parse_date(dt_str: str) -> Optional[datetime]:
    """Parse ISO-ish datetime string to date."""
    try:
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        return None


def build_daily_snapshots(
    sessions: List[Session],
    edges: List[Tuple[str, str, float]],
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """Build cumulative daily network snapshots.

    For each day in the observation period, constructs the network from all
    sessions and edges up to that day, then computes metrics.

    Args:
        sessions: All sessions (parents + subagents).
        edges: List of (source_id, target_id, weight) tuples — semantic edges.
        random_state: For community detection reproducibility.

    Returns:
        List of dicts, one per day, with date + all network metrics.
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

    # Build edge index
    edge_set = [(s, t, w) for s, t, w in edges]

    snapshots = []
    cumulative_nodes = set()

    for day in all_dates:
        # Add nodes active on or before this day
        for sid, d in session_dates.items():
            if d <= day:
                cumulative_nodes.add(sid)

        # Build graph from cumulative edges
        G = nx.Graph()
        G.add_nodes_from(cumulative_nodes)
        for s, t, w in edge_set:
            if s in cumulative_nodes and t in cumulative_nodes:
                G.add_edge(s, t, weight=w)

        metrics = compute_network_metrics(G, random_state=random_state)
        metrics["date"] = day.isoformat()

        # Delegation metrics for this snapshot
        parent_nodes = [
            sid for sid in cumulative_nodes
            if sid in sessions_by_id
            and sessions_by_id[sid].parent_conversation_id is None
        ]
        child_nodes = [
            sid for sid in cumulative_nodes
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

    # Build daily graphs to track degree at each snapshot
    all_dates = sorted(set(d.isoformat() for d in session_dates.values()))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # For each date, collect cumulative node set
    cumulative_by_date = {}
    cumulative_nodes = set()
    for date_str in all_dates:
        date_obj = datetime.fromisoformat(date_str).date()
        for sid, d in session_dates.items():
            if d <= date_obj:
                cumulative_nodes.add(sid)
        cumulative_by_date[date_str] = cumulative_nodes.copy()

    # Build graph for each date and record degrees
    degree_records = []
    prev_edges = set()

    for date_str in all_dates:
        nodes = cumulative_by_date[date_str]
        # Current edges
        current_edges = set()
        for s, t, w in edges:
            if s in nodes and t in nodes:
                current_edges.add((min(s, t), max(s, t)))

        # New edges this snapshot
        new_edges = current_edges - prev_edges

        if prev_edges and new_edges:
            # Build graph from previous edges to get degree
            G_prev = nx.Graph()
            G_prev.add_nodes_from(nodes)
            for s, t in prev_edges:
                G_prev.add_edge(s, t)

            for s, t in new_edges:
                ds = G_prev.degree(s) if s in G_prev else 0
                dt_val = G_prev.degree(t) if t in G_prev else 0
                if ds > 0:
                    degree_records.append(ds)
                if dt_val > 0:
                    degree_records.append(dt_val)

        prev_edges = current_edges

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
