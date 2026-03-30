"""Hierarchical memory network construction from episode embeddings.

Experiment 1: Geometric hierarchy — hierarchical agglomerative clustering on
the full 1908×1908 cosine similarity matrix. Cuts the dendrogram at multiple
heights to produce a multi-level DAG (episodes → concepts → meta-concepts →
domains), inspired by Complementary Learning Systems theory.

The hierarchy is purely geometric: structure emerges from embedding proximity,
not LLM judgment. Post-hoc LLM labeling can be added for interpretability.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
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
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(
    embeddings_dir: Optional[Path] = None,
) -> tuple[list[str], np.ndarray]:
    """Load all episode embeddings from JSON files.

    Returns:
        ids: List of episode IDs (filename stems)
        matrix: (N, 768) numpy array of L2-normalized embeddings
    """
    emb_dir = embeddings_dir or EMBEDDINGS_DIR
    ids = []
    vectors = []

    for fpath in sorted(emb_dir.glob("*.json")):
        try:
            with open(fpath) as f:
                doc = json.load(f)
            vec = doc["embeddings"]["role_aggregate"]["vector"]
            if vec is None:
                logger.warning(f"Skipping {fpath.name}: null vector")
                continue
            ids.append(fpath.stem)
            vectors.append(vec)
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping {fpath.name}: {e}")
            continue

    matrix = np.array(vectors, dtype=np.float64)
    # Ensure L2 normalization (should already be, but be safe)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    logger.info(f"Loaded {len(ids)} embeddings, shape {matrix.shape}")
    return ids, matrix


# ---------------------------------------------------------------------------
# Similarity and clustering
# ---------------------------------------------------------------------------

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute full cosine similarity matrix.

    For L2-normalized vectors, cosine similarity = dot product.
    """
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, 1.0)
    return sim


def build_dendrogram(
    similarity_matrix: np.ndarray,
    method: str = "average",
    embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build hierarchical clustering dendrogram from similarity matrix.

    Args:
        similarity_matrix: (N, N) cosine similarity matrix
        method: Linkage method (average, complete, ward, single)
        embeddings: Required for ward linkage (operates on raw observations)

    Returns:
        linkage_matrix: scipy linkage matrix (N-1, 4)
    """
    if method == "ward":
        # Ward requires raw observations with Euclidean distance.
        # For L2-normalized vectors, Euclidean² = 2(1 - cos_sim),
        # so ward on normalized embeddings is equivalent.
        if embeddings is None:
            raise ValueError("Ward linkage requires embeddings parameter")
        Z = linkage(embeddings, method="ward", metric="euclidean")
    else:
        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)
        # Clip small negatives from floating point
        distance_matrix = np.clip(distance_matrix, 0.0, None)
        # Convert to condensed form for scipy
        condensed = squareform(distance_matrix, checks=False)
        Z = linkage(condensed, method=method)

    logger.info(f"Dendrogram built: {method} linkage, "
                f"height range [{Z[:, 2].min():.4f}, {Z[:, 2].max():.4f}]")
    return Z


def cut_dendrogram(
    Z: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Cut dendrogram to produce exactly n_clusters clusters.

    Returns:
        labels: (N,) array of cluster assignments (0-indexed)
    """
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    # fcluster returns 1-indexed labels; convert to 0-indexed
    return labels - 1


def find_optimal_cuts(
    Z: np.ndarray,
    embeddings: np.ndarray,
    k_range: Optional[tuple[int, int]] = None,
) -> list[dict]:
    """Evaluate silhouette score across different numbers of clusters.

    Returns list of {k, silhouette, height} dicts sorted by k.
    """
    n = embeddings.shape[0]
    if k_range is None:
        k_range = (2, min(50, n // 5))

    results = []
    for k in range(k_range[0], k_range[1] + 1):
        labels = cut_dendrogram(Z, k)
        n_unique = len(set(labels))
        if n_unique < 2 or n_unique >= n:
            continue
        try:
            score = silhouette_score(embeddings, labels, metric="cosine")
        except ValueError:
            continue

        # Find the merge height that corresponds to this cut
        # The (n-k)th merge gives k clusters
        merge_idx = n - k - 1
        if 0 <= merge_idx < len(Z):
            height = Z[merge_idx, 2]
        else:
            height = None

        results.append({
            "k": k,
            "silhouette": round(float(score), 4),
            "height": round(float(height), 6) if height is not None else None,
        })

    return sorted(results, key=lambda r: r["k"])


# ---------------------------------------------------------------------------
# Hierarchy construction
# ---------------------------------------------------------------------------

@dataclass
class HierarchyLevel:
    """One level of the hierarchical memory network."""
    level: int  # 0 = episodes, 1 = concepts, 2 = meta-concepts, ...
    name: str
    n_clusters: int
    labels: list[int]  # cluster assignment for each episode
    centroids: np.ndarray  # (n_clusters, dim) centroid embeddings
    silhouette: float
    cluster_sizes: list[int]
    merge_height: Optional[float] = None


def build_hierarchy(
    Z: np.ndarray,
    embeddings: np.ndarray,
    episode_ids: list[str],
    level_cuts: list[int],
    level_names: Optional[list[str]] = None,
) -> list[HierarchyLevel]:
    """Build multi-level hierarchy by cutting dendrogram at specified cluster counts.

    Args:
        Z: Linkage matrix from build_dendrogram
        embeddings: (N, dim) episode embeddings
        episode_ids: Episode ID strings
        level_cuts: Number of clusters at each level, descending
                    e.g., [15, 5, 2] for concepts(15) → meta-concepts(5) → domains(2)
        level_names: Human-readable names for each level

    Returns:
        List of HierarchyLevel objects, from finest to coarsest
    """
    n = embeddings.shape[0]
    dim = embeddings.shape[1]

    if level_names is None:
        default_names = ["concepts", "meta-concepts", "domains", "super-domains"]
        level_names = default_names[:len(level_cuts)]

    levels = []

    for i, (k, name) in enumerate(zip(level_cuts, level_names)):
        labels = cut_dendrogram(Z, k)

        # Compute centroids
        centroids = np.zeros((k, dim))
        cluster_sizes = []
        for c in range(k):
            mask = labels == c
            members = embeddings[mask]
            if len(members) > 0:
                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                centroids[c] = centroid
            cluster_sizes.append(int(mask.sum()))

        # Silhouette
        n_unique = len(set(labels))
        if n_unique >= 2:
            sil = float(silhouette_score(embeddings, labels, metric="cosine"))
        else:
            sil = 0.0

        # Merge height
        merge_idx = n - k - 1
        height = float(Z[merge_idx, 2]) if 0 <= merge_idx < len(Z) else None

        levels.append(HierarchyLevel(
            level=i + 1,  # 0 is reserved for episodes
            name=name,
            n_clusters=k,
            labels=labels.tolist(),
            centroids=centroids,
            silhouette=round(sil, 4),
            cluster_sizes=cluster_sizes,
            merge_height=round(height, 6) if height is not None else None,
        ))

    return levels


# ---------------------------------------------------------------------------
# DAG and network construction
# ---------------------------------------------------------------------------

def build_dag(
    episode_ids: list[str],
    levels: list[HierarchyLevel],
) -> nx.DiGraph:
    """Build directed acyclic graph from hierarchy levels.

    Nodes: episodes (level 0) + cluster nodes at each level.
    Edges: membership links (child → parent).
    """
    dag = nx.DiGraph()

    # Add episode nodes
    for eid in episode_ids:
        dag.add_node(eid, level=0, node_type="episode")

    # Add cluster nodes and membership edges
    for lvl in levels:
        for c in range(lvl.n_clusters):
            node_id = f"L{lvl.level}_C{c}"
            dag.add_node(node_id, level=lvl.level, node_type=lvl.name,
                         cluster_size=lvl.cluster_sizes[c])

        # Connect episodes to their clusters at this level
        for idx, (eid, cluster) in enumerate(zip(episode_ids, lvl.labels)):
            child = eid if lvl.level == 1 else f"L{lvl.level - 1}_C{levels[lvl.level - 2].labels[idx]}"
            parent = f"L{lvl.level}_C{cluster}"
            dag.add_edge(child, parent, relation="member_of")

    # Connect between adjacent hierarchy levels
    if len(levels) >= 2:
        for i in range(1, len(levels)):
            child_level = levels[i - 1]
            parent_level = levels[i]

            # For each child cluster, find its parent cluster
            # Use majority vote of episode assignments
            for child_c in range(child_level.n_clusters):
                child_episodes = [
                    idx for idx, l in enumerate(child_level.labels) if l == child_c
                ]
                if not child_episodes:
                    continue
                parent_labels = [parent_level.labels[idx] for idx in child_episodes]
                majority_parent = max(set(parent_labels), key=parent_labels.count)
                child_node = f"L{child_level.level}_C{child_c}"
                parent_node = f"L{parent_level.level}_C{majority_parent}"
                if not dag.has_edge(child_node, parent_node):
                    dag.add_edge(child_node, parent_node, relation="member_of")

    return dag


def build_intralevel_graph(
    centroids: np.ndarray,
    level: int,
    threshold: float = 0.5,
) -> nx.Graph:
    """Build similarity graph between clusters at the same hierarchy level.

    Args:
        centroids: (K, dim) centroid embeddings
        level: Hierarchy level number
        threshold: Similarity threshold for edges

    Returns:
        Undirected weighted graph of cluster similarities
    """
    G = nx.Graph()
    k = centroids.shape[0]

    for i in range(k):
        G.add_node(f"L{level}_C{i}")

    sim = cosine_similarity(centroids)
    for i in range(k):
        for j in range(i + 1, k):
            if sim[i, j] >= threshold:
                G.add_edge(f"L{level}_C{i}", f"L{level}_C{j}",
                           weight=float(sim[i, j]))

    return G


# ---------------------------------------------------------------------------
# Analysis metrics
# ---------------------------------------------------------------------------

def analyze_hierarchy(
    episode_ids: list[str],
    embeddings: np.ndarray,
    levels: list[HierarchyLevel],
    dag: nx.DiGraph,
) -> dict:
    """Compute comprehensive metrics for the hierarchical memory network."""

    results = {
        "n_episodes": len(episode_ids),
        "n_levels": len(levels),
        "embedding_dim": embeddings.shape[1],
        "levels": [],
        "dag": {},
        "intralevel_networks": [],
        "cross_level": {},
    }

    # Per-level metrics
    for lvl in levels:
        sizes = np.array(lvl.cluster_sizes)
        level_info = {
            "level": lvl.level,
            "name": lvl.name,
            "n_clusters": lvl.n_clusters,
            "silhouette": lvl.silhouette,
            "merge_height": lvl.merge_height,
            "cluster_sizes": {
                "min": int(sizes.min()),
                "max": int(sizes.max()),
                "mean": round(float(sizes.mean()), 1),
                "median": round(float(np.median(sizes)), 1),
                "std": round(float(sizes.std()), 1),
            },
            "singletons": int((sizes == 1).sum()),
        }

        # Intra-level similarity network
        intra_G = build_intralevel_graph(lvl.centroids, lvl.level, threshold=0.3)
        level_info["intralevel_network"] = {
            "nodes": intra_G.number_of_nodes(),
            "edges": intra_G.number_of_edges(),
            "density": round(nx.density(intra_G), 4),
            "avg_clustering": round(nx.average_clustering(intra_G), 4),
        }

        if HAS_COMMUNITY and intra_G.number_of_edges() > 0:
            partition = community_louvain.best_partition(intra_G, random_state=42)
            level_info["intralevel_network"]["modularity"] = round(
                community_louvain.modularity(partition, intra_G), 4
            )
            level_info["intralevel_network"]["communities"] = len(set(partition.values()))

        results["levels"].append(level_info)
        results["intralevel_networks"].append({
            "level": lvl.level,
            "graph": nx.node_link_data(intra_G),
        })

    # DAG metrics
    results["dag"] = {
        "nodes": dag.number_of_nodes(),
        "edges": dag.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(dag),
    }

    # Branching factors between adjacent levels
    if len(levels) >= 2:
        branching = []
        for i in range(1, len(levels)):
            child_level = levels[i - 1]
            parent_level = levels[i]
            ratio = child_level.n_clusters / parent_level.n_clusters
            branching.append({
                "from": child_level.name,
                "to": parent_level.name,
                "ratio": round(ratio, 2),
                "child_clusters": child_level.n_clusters,
                "parent_clusters": parent_level.n_clusters,
            })
        results["cross_level"]["branching_factors"] = branching

    # Episode-to-top path lengths
    episode_nodes = [n for n, d in dag.nodes(data=True) if d.get("level") == 0]
    top_nodes = [n for n, d in dag.nodes(data=True)
                 if d.get("level") == levels[-1].level]

    if episode_nodes and top_nodes:
        path_lengths = []
        for ep in episode_nodes[:100]:  # Sample for speed
            for top in top_nodes:
                try:
                    length = nx.shortest_path_length(dag, ep, top)
                    path_lengths.append(length)
                    break
                except nx.NetworkXNoPath:
                    continue
        if path_lengths:
            results["cross_level"]["avg_path_depth"] = round(np.mean(path_lengths), 2)
            results["cross_level"]["max_path_depth"] = max(path_lengths)

    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_geometric_hierarchy(
    embeddings_dir: Optional[Path] = None,
    level_cuts: Optional[list[int]] = None,
    level_names: Optional[list[str]] = None,
    linkage_method: str = "average",
    output_dir: Optional[Path] = None,
    scan_k_range: Optional[tuple[int, int]] = None,
) -> dict:
    """Run the full geometric hierarchy experiment.

    Args:
        embeddings_dir: Directory with episode embedding JSONs
        level_cuts: Number of clusters at each level (finest to coarsest)
        level_names: Names for each level
        linkage_method: Hierarchical clustering linkage method
        output_dir: Where to save results
        scan_k_range: If set, scan this range of k values for optimal cuts

    Returns:
        Full analysis results dict
    """
    # Load embeddings
    episode_ids, embeddings = load_embeddings(embeddings_dir)
    n = len(episode_ids)

    # Compute similarity matrix
    logger.info(f"Computing {n}x{n} cosine similarity matrix...")
    sim_matrix = compute_similarity_matrix(embeddings)

    # Summary stats on similarity distribution
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    sim_stats = {
        "mean": round(float(upper_tri.mean()), 4),
        "std": round(float(upper_tri.std()), 4),
        "min": round(float(upper_tri.min()), 4),
        "max": round(float(upper_tri.max()), 4),
        "median": round(float(np.median(upper_tri)), 4),
        "above_0.9": int((upper_tri >= 0.9).sum()),
        "above_0.8": int((upper_tri >= 0.8).sum()),
        "above_0.7": int((upper_tri >= 0.7).sum()),
    }
    logger.info(f"Similarity stats: mean={sim_stats['mean']}, "
                f"std={sim_stats['std']}, above_0.9={sim_stats['above_0.9']}")

    # Build dendrogram
    logger.info(f"Building dendrogram ({linkage_method} linkage)...")
    Z = build_dendrogram(sim_matrix, method=linkage_method, embeddings=embeddings)

    # Scan for optimal cuts if requested
    cut_scan = None
    if scan_k_range:
        logger.info(f"Scanning k={scan_k_range[0]}..{scan_k_range[1]} for optimal cuts...")
        cut_scan = find_optimal_cuts(Z, embeddings, k_range=scan_k_range)
        if cut_scan:
            best = max(cut_scan, key=lambda r: r["silhouette"])
            logger.info(f"Best silhouette: k={best['k']}, score={best['silhouette']}")

    # Build hierarchy
    if level_cuts is None:
        level_cuts = [15, 5, 2]  # Default: 15 concepts, 5 meta, 2 domains
    if level_names is None:
        level_names = ["concepts", "meta-concepts", "domains"][:len(level_cuts)]

    logger.info(f"Building hierarchy: {list(zip(level_names, level_cuts))}")
    levels = build_hierarchy(Z, embeddings, episode_ids, level_cuts, level_names)

    # Build DAG
    dag = build_dag(episode_ids, levels)

    # Analyze
    logger.info("Computing hierarchy metrics...")
    analysis = analyze_hierarchy(episode_ids, embeddings, levels, dag)
    analysis["similarity_distribution"] = sim_stats
    analysis["linkage_method"] = linkage_method
    analysis["level_cuts"] = level_cuts
    analysis["level_names"] = level_names

    if cut_scan:
        analysis["silhouette_scan"] = cut_scan

    # Add cluster membership details
    for lvl, level_data in zip(levels, analysis["levels"]):
        membership = {}
        for idx, (eid, cluster) in enumerate(zip(episode_ids, lvl.labels)):
            cluster_key = f"L{lvl.level}_C{cluster}"
            membership.setdefault(cluster_key, []).append(eid)
        level_data["cluster_membership"] = membership

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Main results (without numpy arrays)
        results_file = output_dir / "hierarchy_geometric.json"
        with open(results_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Results saved to {results_file}")

        # Save linkage matrix for reproducibility
        np.save(output_dir / "linkage_matrix.npy", Z)

        # Save centroids at each level
        for lvl in levels:
            np.save(output_dir / f"centroids_L{lvl.level}_{lvl.name}.npy", lvl.centroids)

        # Save DAG as edge list
        dag_edges = []
        for u, v, d in dag.edges(data=True):
            dag_edges.append({"source": u, "target": v, "relation": d.get("relation", "")})
        with open(output_dir / "dag_edges.json", "w") as f:
            json.dump(dag_edges, f, indent=2)

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Build geometric hierarchy from episode embeddings"
    )
    parser.add_argument("--embeddings-dir", type=str, default=None,
                        help="Directory with embedding JSONs")
    parser.add_argument("--levels", type=str, default="15,5,2",
                        help="Comma-separated cluster counts per level (finest to coarsest)")
    parser.add_argument("--level-names", type=str, default="concepts,meta-concepts,domains",
                        help="Comma-separated names for each level")
    parser.add_argument("--linkage", type=str, default="average",
                        choices=["average", "complete", "single", "ward"],
                        help="Linkage method for agglomerative clustering")
    parser.add_argument("--scan", action="store_true",
                        help="Scan k=2..50 for optimal silhouette scores")
    parser.add_argument("--scan-range", type=str, default="2,50",
                        help="k range for silhouette scan")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else None
    level_cuts = [int(x) for x in args.levels.split(",")]
    level_names = args.level_names.split(",")

    scan_range = None
    if args.scan:
        lo, hi = args.scan_range.split(",")
        scan_range = (int(lo), int(hi))

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "experiments" / "results" / "hierarchy"
    )

    results = run_geometric_hierarchy(
        embeddings_dir=emb_dir,
        level_cuts=level_cuts,
        level_names=level_names,
        linkage_method=args.linkage,
        output_dir=output_dir,
        scan_k_range=scan_range,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Geometric Hierarchy: {results['n_episodes']} episodes")
    print(f"Linkage: {results['linkage_method']}")
    print(f"Similarity: mean={results['similarity_distribution']['mean']}, "
          f"std={results['similarity_distribution']['std']}")
    print(f"\nLevels:")
    for lvl in results["levels"]:
        print(f"  L{lvl['level']} {lvl['name']}: {lvl['n_clusters']} clusters, "
              f"silhouette={lvl['silhouette']}, "
              f"sizes=[{lvl['cluster_sizes']['min']}..{lvl['cluster_sizes']['max']}]")
    print(f"\nDAG: {results['dag']['nodes']} nodes, {results['dag']['edges']} edges")
    if "avg_path_depth" in results.get("cross_level", {}):
        print(f"Avg path depth: {results['cross_level']['avg_path_depth']}")


if __name__ == "__main__":
    main()
