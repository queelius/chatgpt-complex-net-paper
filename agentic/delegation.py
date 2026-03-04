"""Delegation network construction and analysis.

Builds a directed graph from parent->child session relationships
and computes delegation-specific metrics.
"""
from collections import Counter
from typing import Dict, List

import networkx as nx

from agentic.extract import Session


def classify_agent_type(agent_id: str) -> str:
    """Classify a subagent by its ID prefix."""
    if "acompact" in agent_id:
        return "compact"
    if "aprompt_suggestion" in agent_id:
        return "prompt_suggestion"
    return "user_spawned"


def build_delegation_graph(sessions: List[Session]) -> nx.DiGraph:
    """Build a directed delegation graph from sessions.

    Nodes are session IDs. Edges point from parent to child.
    Node attributes include session metadata for downstream analysis.
    """
    G = nx.DiGraph()

    for s in sessions:
        G.add_node(
            s.id,
            title=s.title,
            model=s.model,
            message_count=s.message_count,
            created_at=s.created_at,
            is_parent=s.parent_conversation_id is None,
            agent_type=(
                classify_agent_type(s.metadata.get("agent_id", ""))
                if s.parent_conversation_id
                else "parent"
            ),
        )

    for s in sessions:
        if s.parent_conversation_id and s.parent_conversation_id in G:
            G.add_edge(s.parent_conversation_id, s.id)

    return G


def compute_fan_out_distribution(G: nx.DiGraph) -> Dict[str, int]:
    """Compute fan-out (out-degree) for each parent node."""
    return {
        node: G.out_degree(node)
        for node in G.nodes()
        if G.nodes[node].get("is_parent", False)
    }


def compute_delegation_ratio(G: nx.DiGraph) -> Dict[str, float]:
    """Compute delegation ratio: total child messages / parent messages."""
    ratios = {}
    for node in G.nodes():
        if not G.nodes[node].get("is_parent"):
            continue
        parent_msgs = G.nodes[node].get("message_count", 0)
        if parent_msgs == 0:
            continue
        child_msgs = sum(
            G.nodes[child].get("message_count", 0)
            for child in G.successors(node)
        )
        ratios[node] = child_msgs / parent_msgs
    return ratios


def agent_type_counts(G: nx.DiGraph) -> Counter:
    """Count subagents by type across the entire graph."""
    return Counter(
        G.nodes[n].get("agent_type", "unknown")
        for n in G.nodes()
        if not G.nodes[n].get("is_parent")
    )
