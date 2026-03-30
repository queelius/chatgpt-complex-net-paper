"""Semantic hierarchy construction from extracted concepts (Experiment 2).

Takes the output of concept_extraction_v2.py (per-episode noun-phrase concepts)
and builds a multi-level hierarchical memory network:

1. Embed all unique concepts via nomic-embed-text
2. Cluster concept embeddings (agglomerative, ward) → canonical concepts
3. Build bipartite graph: episodes ↔ concepts (many-to-many)
4. Cluster canonical concepts → meta-concepts → domains
5. Build full DAG with cross-links

Contrasts with hierarchy.py (Experiment 1) which builds the hierarchy
purely from episode embedding geometry. Here the hierarchy emerges from
*semantic concepts* extracted by an LLM.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx

try:
    import community.community_louvain as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
AGENTIC_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_extraction_state(state_file: Path) -> dict:
    """Load the extraction state JSON."""
    with open(state_file) as f:
        return json.load(f)


def build_episode_concept_matrix(state: dict) -> tuple[list[str], list[str], np.ndarray]:
    """Build a binary episode × concept matrix from extraction results.

    Returns:
        episode_ids: List of episode IDs
        concepts: List of unique concept strings (lowercased)
        matrix: (n_episodes, n_concepts) binary matrix
    """
    # Collect all concepts (case-insensitive dedup)
    concept_counter = Counter()
    for ep in state['episodes']:
        for c in ep['concepts']:
            concept_counter[c.lower()] += 1

    # Sort by frequency (most common first)
    concepts = [c for c, _ in concept_counter.most_common()]
    concept_to_idx = {c: i for i, c in enumerate(concepts)}

    episode_ids = [ep['episode_id'] for ep in state['episodes']]

    matrix = np.zeros((len(episode_ids), len(concepts)), dtype=np.float32)
    for i, ep in enumerate(state['episodes']):
        for c in ep['concepts']:
            j = concept_to_idx.get(c.lower())
            if j is not None:
                matrix[i, j] = 1.0

    return episode_ids, concepts, matrix


# ---------------------------------------------------------------------------
# Concept embedding
# ---------------------------------------------------------------------------

def embed_concepts_ollama(
    concepts: list[str],
    model: str = "nomic-embed-text",
) -> np.ndarray:
    """Embed concept strings using Ollama's embedding API.

    Returns:
        (n_concepts, dim) numpy array of L2-normalized embeddings
    """
    import requests

    embeddings = []
    for i, concept in enumerate(concepts):
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": concept},
            timeout=60,
        )
        r.raise_for_status()
        vec = r.json()["embedding"]
        embeddings.append(vec)

        if (i + 1) % 100 == 0:
            logger.info(f"  Embedded {i+1}/{len(concepts)} concepts")

    matrix = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    logger.info(f"Embedded {len(concepts)} concepts → shape {matrix.shape}")
    return matrix


# ---------------------------------------------------------------------------
# Bipartite graph construction
# ---------------------------------------------------------------------------

def build_bipartite_graph(
    episode_ids: list[str],
    concepts: list[str],
    ep_concept_matrix: np.ndarray,
    episode_embeddings: Optional[np.ndarray] = None,
    concept_embeddings: Optional[np.ndarray] = None,
) -> nx.Graph:
    """Build bipartite graph between episodes and concepts.

    If embeddings are provided, edge weights are cosine similarity between
    the episode embedding and concept embedding. Otherwise, binary edges.
    """
    B = nx.Graph()

    for eid in episode_ids:
        B.add_node(eid, bipartite=0, node_type="episode")
    for concept in concepts:
        B.add_node(f"C:{concept}", bipartite=1, node_type="concept")

    # Compute weights if both embeddings available
    if episode_embeddings is not None and concept_embeddings is not None:
        cross_sim = cosine_similarity(episode_embeddings, concept_embeddings)
    else:
        cross_sim = None

    for i, eid in enumerate(episode_ids):
        for j, concept in enumerate(concepts):
            if ep_concept_matrix[i, j] > 0:
                weight = float(cross_sim[i, j]) if cross_sim is not None else 1.0
                B.add_edge(eid, f"C:{concept}", weight=weight)

    return B


# ---------------------------------------------------------------------------
# Concept hierarchy
# ---------------------------------------------------------------------------

def cluster_concepts(
    concept_embeddings: np.ndarray,
    concepts: list[str],
    level_cuts: list[int],
    level_names: Optional[list[str]] = None,
) -> list[dict]:
    """Hierarchically cluster concept embeddings.

    Returns list of level dicts with cluster assignments and centroids.
    """
    n = len(concepts)
    if n < 3:
        return []

    # Ward linkage on concept embeddings
    Z = linkage(concept_embeddings, method="ward", metric="euclidean")

    if level_names is None:
        level_names = [f"level_{i}" for i in range(len(level_cuts))]

    levels = []
    for i, (k, name) in enumerate(zip(level_cuts, level_names)):
        if k >= n:
            k = n - 1
        labels = fcluster(Z, t=k, criterion="maxclust") - 1  # 0-indexed

        # Compute centroids
        centroids = np.zeros((k, concept_embeddings.shape[1]))
        cluster_sizes = []
        for c in range(k):
            mask = labels == c
            members = concept_embeddings[mask]
            if len(members) > 0:
                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                centroids[c] = centroid
            cluster_sizes.append(int(mask.sum()))

        # Silhouette
        n_unique = len(set(labels))
        if n_unique >= 2 and n_unique < n:
            sil = float(silhouette_score(concept_embeddings, labels, metric="cosine"))
        else:
            sil = 0.0

        # Map concepts to clusters
        cluster_membership = {}
        for idx, label in enumerate(labels):
            key = f"MC{i+1}_C{label}"
            cluster_membership.setdefault(key, []).append(concepts[idx])

        levels.append({
            'level': i + 1,
            'name': name,
            'n_clusters': k,
            'silhouette': round(sil, 4),
            'labels': labels.tolist(),
            'cluster_sizes': cluster_sizes,
            'centroids': centroids,
            'cluster_membership': cluster_membership,
        })

    return levels


# ---------------------------------------------------------------------------
# Full DAG construction
# ---------------------------------------------------------------------------

def build_semantic_dag(
    episode_ids: list[str],
    concepts: list[str],
    ep_concept_matrix: np.ndarray,
    concept_levels: list[dict],
) -> nx.DiGraph:
    """Build the full hierarchical DAG.

    Nodes: episodes + concepts + meta-concept clusters at each level
    Edges: episode→concept (many-to-many), concept→meta-concept, etc.
    """
    dag = nx.DiGraph()

    # Episode nodes
    for eid in episode_ids:
        dag.add_node(eid, level=0, node_type="episode")

    # Concept nodes
    for concept in concepts:
        dag.add_node(f"C:{concept}", level=1, node_type="concept")

    # Episode → concept edges (many-to-many)
    for i, eid in enumerate(episode_ids):
        for j, concept in enumerate(concepts):
            if ep_concept_matrix[i, j] > 0:
                dag.add_edge(eid, f"C:{concept}", relation="has_concept")

    # Higher-level cluster nodes and edges
    for lvl in concept_levels:
        for cluster_id, members in lvl['cluster_membership'].items():
            dag.add_node(cluster_id, level=lvl['level'] + 1,
                         node_type=lvl['name'],
                         size=len(members))
            # Connect concepts to their cluster
            for concept in members:
                child = f"C:{concept}" if lvl['level'] == 1 else concept
                if child in dag:
                    dag.add_edge(child, cluster_id, relation="member_of")

    # Connect between meta-concept levels
    if len(concept_levels) >= 2:
        for i in range(1, len(concept_levels)):
            child_lvl = concept_levels[i - 1]
            parent_lvl = concept_levels[i]

            for child_cluster, child_concepts in child_lvl['cluster_membership'].items():
                # Find parent cluster by majority vote
                parent_labels = []
                for concept in child_concepts:
                    concept_idx = concepts.index(concept) if concept in concepts else -1
                    if concept_idx >= 0:
                        parent_labels.append(parent_lvl['labels'][concept_idx])
                if parent_labels:
                    majority = max(set(parent_labels), key=parent_labels.count)
                    parent_cluster = f"MC{parent_lvl['level']}_C{majority}"
                    if not dag.has_edge(child_cluster, parent_cluster):
                        dag.add_edge(child_cluster, parent_cluster,
                                     relation="member_of")

    return dag


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_semantic_hierarchy(
    episode_ids: list[str],
    concepts: list[str],
    ep_concept_matrix: np.ndarray,
    concept_levels: list[dict],
    dag: nx.DiGraph,
    bipartite: nx.Graph,
) -> dict:
    """Comprehensive analysis of the semantic hierarchy."""

    n_episodes = len(episode_ids)
    n_concepts = len(concepts)

    # Episode statistics
    concepts_per_episode = ep_concept_matrix.sum(axis=1)
    episodes_per_concept = ep_concept_matrix.sum(axis=0)

    results = {
        'n_episodes': n_episodes,
        'n_concepts': n_concepts,
        'episode_concept_stats': {
            'concepts_per_episode': {
                'mean': round(float(concepts_per_episode.mean()), 2),
                'std': round(float(concepts_per_episode.std()), 2),
                'min': int(concepts_per_episode.min()),
                'max': int(concepts_per_episode.max()),
                'median': round(float(np.median(concepts_per_episode)), 1),
            },
            'episodes_per_concept': {
                'mean': round(float(episodes_per_concept.mean()), 2),
                'std': round(float(episodes_per_concept.std()), 2),
                'min': int(episodes_per_concept.min()),
                'max': int(episodes_per_concept.max()),
                'median': round(float(np.median(episodes_per_concept)), 1),
            },
            'total_links': int(ep_concept_matrix.sum()),
        },
        'levels': [],
        'dag': {
            'nodes': dag.number_of_nodes(),
            'edges': dag.number_of_edges(),
            'is_dag': nx.is_directed_acyclic_graph(dag),
        },
        'bipartite': {
            'nodes': bipartite.number_of_nodes(),
            'edges': bipartite.number_of_edges(),
        },
    }

    # Concept frequency distribution
    freq_dist = Counter()
    for count in episodes_per_concept:
        freq_dist[int(count)] += 1
    results['concept_frequency_distribution'] = dict(sorted(freq_dist.items()))

    # Top concepts by frequency
    top_concepts = sorted(zip(concepts, episodes_per_concept),
                          key=lambda x: -x[1])[:30]
    results['top_concepts'] = [
        {'concept': c, 'episodes': int(n)} for c, n in top_concepts
    ]

    # Level analysis
    for lvl in concept_levels:
        sizes = np.array(lvl['cluster_sizes'])
        level_info = {
            'level': lvl['level'],
            'name': lvl['name'],
            'n_clusters': lvl['n_clusters'],
            'silhouette': lvl['silhouette'],
            'cluster_sizes': {
                'min': int(sizes.min()) if len(sizes) > 0 else 0,
                'max': int(sizes.max()) if len(sizes) > 0 else 0,
                'mean': round(float(sizes.mean()), 1) if len(sizes) > 0 else 0,
            },
            'cluster_membership': {
                k: v for k, v in lvl['cluster_membership'].items()
            },
        }
        results['levels'].append(level_info)

    # Concept co-occurrence (how many episodes share concepts)
    # Jaccard similarity between concept pairs
    co_occurrence = ep_concept_matrix.T @ ep_concept_matrix
    np.fill_diagonal(co_occurrence, 0)
    n_co = int((co_occurrence > 0).sum() / 2)
    results['concept_co_occurrence'] = {
        'pairs_with_shared_episodes': n_co,
        'max_shared': int(co_occurrence.max()),
    }

    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_semantic_hierarchy(
    state_file: Path,
    episode_embeddings_dir: Optional[Path] = None,
    concept_level_cuts: Optional[list[int]] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run the full semantic hierarchy pipeline.

    Args:
        state_file: Path to extraction_state.json from concept_extraction_v2
        episode_embeddings_dir: Directory with episode embedding JSONs
        concept_level_cuts: Cluster counts for concept hierarchy levels
        output_dir: Where to save results
    """
    # Load extraction results
    logger.info(f"Loading extraction state from {state_file}")
    state = load_extraction_state(state_file)
    logger.info(f"  {state['total_extractions']} episodes, "
                f"{state['vocabulary_size']} unique concepts")

    # Build episode-concept matrix
    episode_ids, concepts, ep_matrix = build_episode_concept_matrix(state)
    logger.info(f"  Episode-concept matrix: {ep_matrix.shape}, "
                f"{int(ep_matrix.sum())} links")

    # Embed concepts
    logger.info("Embedding concepts via Ollama...")
    concept_embeddings = embed_concepts_ollama(concepts)

    # Load episode embeddings if available
    episode_embeddings = None
    if episode_embeddings_dir:
        from hierarchy import load_embeddings
        all_ids, all_embs = load_embeddings(episode_embeddings_dir)
        id_to_idx = {eid: i for i, eid in enumerate(all_ids)}
        ep_indices = [id_to_idx[eid] for eid in episode_ids if eid in id_to_idx]
        if len(ep_indices) == len(episode_ids):
            episode_embeddings = all_embs[ep_indices]

    # Build bipartite graph
    logger.info("Building bipartite graph...")
    bipartite = build_bipartite_graph(
        episode_ids, concepts, ep_matrix,
        episode_embeddings, concept_embeddings,
    )

    # Cluster concepts into hierarchy
    if concept_level_cuts is None:
        # Adaptive: scale with concept count
        n_c = len(concepts)
        if n_c > 100:
            concept_level_cuts = [min(30, n_c // 5), min(10, n_c // 15), 3]
        elif n_c > 30:
            concept_level_cuts = [min(15, n_c // 3), 5]
        else:
            concept_level_cuts = [min(8, n_c // 2)]

    level_names = ['meta-concepts', 'themes', 'domains'][:len(concept_level_cuts)]

    logger.info(f"Clustering {len(concepts)} concepts: {list(zip(level_names, concept_level_cuts))}")
    concept_levels = cluster_concepts(
        concept_embeddings, concepts, concept_level_cuts, level_names
    )

    # Build DAG
    logger.info("Building semantic DAG...")
    dag = build_semantic_dag(episode_ids, concepts, ep_matrix, concept_levels)

    # Analyze
    logger.info("Computing metrics...")
    results = analyze_semantic_hierarchy(
        episode_ids, concepts, ep_matrix, concept_levels, dag, bipartite
    )

    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "hierarchy_semantic.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_dir / 'hierarchy_semantic.json'}")

        # Save concept embeddings
        np.save(output_dir / "concept_embeddings.npy", concept_embeddings)

        # Save DAG edges
        dag_edges = [{"source": u, "target": v, **d}
                     for u, v, d in dag.edges(data=True)]
        with open(output_dir / "semantic_dag_edges.json", "w") as f:
            json.dump(dag_edges, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Build semantic hierarchy from extracted concepts"
    )
    parser.add_argument("--state-file", type=str, required=True,
                        help="Path to extraction_state.json")
    parser.add_argument("--embeddings-dir", type=str, default=None,
                        help="Episode embeddings directory")
    parser.add_argument("--levels", type=str, default=None,
                        help="Comma-separated cluster counts (e.g., 30,10,3)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else None
    level_cuts = [int(x) for x in args.levels.split(",")] if args.levels else None
    output_dir = Path(args.output_dir) if args.output_dir else (
        AGENTIC_DIR / "experiments" / "results" / "hierarchy"
    )

    results = run_semantic_hierarchy(
        state_file=Path(args.state_file),
        episode_embeddings_dir=emb_dir,
        concept_level_cuts=level_cuts,
        output_dir=output_dir,
    )

    print(f"\n{'='*60}")
    print(f"Semantic Hierarchy: {results['n_episodes']} episodes, "
          f"{results['n_concepts']} concepts")
    print(f"Concepts/episode: {results['episode_concept_stats']['concepts_per_episode']}")
    print(f"Episodes/concept: {results['episode_concept_stats']['episodes_per_concept']}")
    print(f"Total links: {results['episode_concept_stats']['total_links']}")
    print(f"DAG: {results['dag']['nodes']} nodes, {results['dag']['edges']} edges")
    print(f"\nTop 10 concepts:")
    for tc in results['top_concepts'][:10]:
        print(f"  {tc['concept']} ({tc['episodes']} episodes)")


if __name__ == "__main__":
    main()
