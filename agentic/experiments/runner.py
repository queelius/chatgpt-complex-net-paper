"""Experiment runner for the Cognitive MRI research program.

Drives the embed → edges → threshold → analyze pipeline with different
configurations and stores results in a standard format for comparison.

Usage:
    python -m experiments.runner --list          # Show all experiments
    python -m experiments.runner --run EXP_ID    # Run one experiment
    python -m experiments.runner --run-all       # Run all pending
    python -m experiments.runner --compare       # Compare all results
"""
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import networkx as nx
import numpy as np

# Add parent to path for agentic imports
EXPERIMENTS_DIR = Path(__file__).parent
AGENTIC_DIR = EXPERIMENTS_DIR.parent
REPO_ROOT = AGENTIC_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENTIC_DIR))

from agentic.semantic import compute_network_metrics


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Defines a single experiment."""
    id: str
    name: str
    description: str
    dataset: str              # "chatgpt" or "agentic"
    # Edge source: either a pre-computed edges file or will be computed
    edges_file: Optional[str] = None
    # If no edges_file, these define how to compute edges
    content_mode: Optional[str] = None  # text-only, text-full, user-only, tool-names-only
    user_weight: float = 2.0
    assistant_weight: float = 1.0
    embedding_method: str = "llm"
    # Threshold(s) to apply
    thresholds: List[float] = field(default_factory=lambda: [0.9])
    # Total sessions (for adding isolated nodes)
    total_nodes: Optional[int] = None
    # Tags for filtering/grouping
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Results from one experiment at one threshold."""
    experiment_id: str
    threshold: float
    metrics: Dict[str, Any]
    elapsed_seconds: float
    timestamp: str


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def build_graph_from_edges(
    edges: List,
    threshold: float,
    total_nodes: Optional[int] = None,
) -> nx.Graph:
    """Build a NetworkX graph from edge list, filtered at threshold.

    If total_nodes is provided, adds isolated nodes to match the full dataset
    size (important for GC fraction, density, etc.)
    """
    G = nx.Graph()
    node_set = set()
    for src, dst, weight in edges:
        if weight >= threshold:
            G.add_edge(src, dst, weight=weight)
            node_set.add(src)
            node_set.add(dst)

    # Add isolated nodes if we know the total
    if total_nodes and total_nodes > G.number_of_nodes():
        for i in range(total_nodes - G.number_of_nodes()):
            G.add_node(f"__isolated_{i}")

    return G


def run_single_experiment(
    config: ExperimentConfig,
    edges: List,
) -> List[ExperimentResult]:
    """Run analysis at each threshold for a given experiment config."""
    results = []
    for threshold in config.thresholds:
        t0 = time.time()
        G = build_graph_from_edges(edges, threshold, config.total_nodes)
        metrics = compute_network_metrics(G)
        elapsed = time.time() - t0

        result = ExperimentResult(
            experiment_id=config.id,
            threshold=threshold,
            metrics=metrics,
            elapsed_seconds=round(elapsed, 2),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        results.append(result)
        print(f"  θ={threshold:.2f}: {metrics['node_count']} nodes, "
              f"{metrics['edge_count']} edges, "
              f"mod={metrics.get('modularity', 0):.3f}, "
              f"GC={metrics.get('giant_component_fraction', 0):.3f} "
              f"({elapsed:.1f}s)")
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_results(results: List[ExperimentResult], output_dir: Path):
    """Save experiment results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        fname = f"{r.experiment_id}_t{r.threshold:.2f}.json"
        with open(output_dir / fname, "w") as f:
            json.dump(asdict(r), f, indent=2, default=str)


def load_all_results(output_dir: Path) -> List[Dict]:
    """Load all experiment result files from the output directory.

    Only loads files that match the runner's result schema (must have
    experiment_id, threshold, and metrics keys). Ad-hoc analysis files
    with different schemas are silently skipped.
    """
    results = []
    if not output_dir.exists():
        return results
    required_keys = {"experiment_id", "threshold", "metrics"}
    for f in sorted(output_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or not required_keys.issubset(data):
            continue
        results.append(data)
    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_results(results: List[Dict]) -> str:
    """Generate a comparison table from all experiment results."""
    if not results:
        return "No results to compare."

    # Key metrics to compare
    key_metrics = [
        "node_count", "edge_count", "density",
        "giant_component_fraction", "avg_degree", "max_degree",
        "avg_clustering", "transitivity", "modularity",
        "num_communities", "assortativity", "num_components",
    ]

    lines = []
    # Header
    header = f"{'Experiment':<40} {'θ':>4} {'Nodes':>6} {'Edges':>8} " \
             f"{'Density':>8} {'GC%':>6} {'AvgDeg':>7} {'Clust':>6} " \
             f"{'Trans':>6} {'Mod':>6} {'Comm':>6} {'Assort':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in sorted(results, key=lambda x: (x["experiment_id"], x["threshold"])):
        m = r["metrics"]
        gc_frac = m.get("giant_component_fraction", 0) * 100
        assort = m.get("assortativity")
        assort_str = f"{assort:+.3f}" if assort is not None else "  N/A"
        lines.append(
            f"{r['experiment_id']:<40} {r['threshold']:>4.2f} "
            f"{m['node_count']:>6} {m['edge_count']:>8} "
            f"{m.get('density', 0):>8.4f} {gc_frac:>5.1f}% "
            f"{m.get('avg_degree', 0):>7.1f} {m.get('avg_clustering', 0):>6.3f} "
            f"{m.get('transitivity', 0):>6.3f} {m.get('modularity', 0):>6.3f} "
            f"{m.get('num_communities', 0):>6} {assort_str:>7}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Edge loading helpers
# ---------------------------------------------------------------------------
def load_edges(path: str) -> List:
    """Load edges from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_ablation_edges(weight_label: str) -> Optional[List]:
    """Load unfiltered edges from the ChatGPT ablation study."""
    ablation_dir = REPO_ROOT / "dev" / "ablation_study" / "data" / "edges"
    fname = f"edges_chatgpt-json-llm-{weight_label}.json"
    path = ablation_dir / fname
    if path.exists():
        return load_edges(str(path))
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--run", type=str, help="Run a specific experiment by ID")
    parser.add_argument("--run-tag", type=str, help="Run all experiments with this tag")
    parser.add_argument("--run-all", action="store_true", help="Run all experiments")
    parser.add_argument("--compare", action="store_true", help="Compare all results")
    args = parser.parse_args()

    from experiments.registry import EXPERIMENTS

    output_dir = EXPERIMENTS_DIR / "results"

    if args.list:
        for exp in EXPERIMENTS:
            tags = ", ".join(exp.tags) if exp.tags else ""
            print(f"  {exp.id:<40} [{tags}] {exp.description}")
        return

    if args.compare:
        results = load_all_results(output_dir)
        print(compare_results(results))
        # Also save comparison
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # Determine which experiments to run
    to_run = []
    if args.run:
        to_run = [e for e in EXPERIMENTS if e.id == args.run]
        if not to_run:
            print(f"Unknown experiment: {args.run}")
            return
    elif args.run_tag:
        to_run = [e for e in EXPERIMENTS if args.run_tag in e.tags]
    elif args.run_all:
        to_run = EXPERIMENTS

    if not to_run:
        parser.print_help()
        return

    for exp in to_run:
        print(f"\n{'='*60}")
        print(f"Running: {exp.id}")
        print(f"  {exp.description}")
        print(f"{'='*60}")

        # Load edges
        if exp.edges_file:
            edges = load_edges(exp.edges_file)
        else:
            print(f"  ERROR: No edges file and embedding pipeline not yet integrated")
            continue

        results = run_single_experiment(exp, edges)
        save_results(results, output_dir)

    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON OF ALL RESULTS")
    print(f"{'='*60}\n")
    all_results = load_all_results(output_dir)
    print(compare_results(all_results))


if __name__ == "__main__":
    main()
