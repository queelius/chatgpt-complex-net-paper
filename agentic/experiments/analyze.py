"""Analyze and compare all experiment results.

Generates:
  1. Comparison tables (text + JSON)
  2. Phase transition curves (edges, GC%, modularity vs θ) per weight config
  3. Weight ratio effect at fixed θ=0.9
  4. Key findings summary
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"


def load_all_results():
    """Load all result JSON files, grouped by experiment ID."""
    by_experiment = defaultdict(list)
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name in ("comparison.json", "analysis.json"):
            continue
        with open(f) as fh:
            r = json.load(fh)
        by_experiment[r["experiment_id"]].append(r)
    # Sort each experiment's results by threshold
    for eid in by_experiment:
        by_experiment[eid].sort(key=lambda r: r["threshold"])
    return dict(by_experiment)


def extract_sweep_curves(results_by_exp):
    """Extract threshold sweep curves for plotting/analysis."""
    sweeps = {}
    for eid, runs in results_by_exp.items():
        if len(runs) < 3:  # Only consider multi-threshold sweeps
            continue
        thresholds = [r["threshold"] for r in runs]
        edges = [r["metrics"]["edge_count"] for r in runs]
        gc_frac = [r["metrics"].get("giant_component_fraction", 0) for r in runs]
        modularity = [r["metrics"].get("modularity", 0) for r in runs]
        clustering = [r["metrics"].get("avg_clustering", 0) for r in runs]
        assortativity = [r["metrics"].get("assortativity", 0) or 0 for r in runs]
        transitivity = [r["metrics"].get("transitivity", 0) for r in runs]

        sweeps[eid] = {
            "thresholds": thresholds,
            "edges": edges,
            "gc_fraction": gc_frac,
            "modularity": modularity,
            "clustering": clustering,
            "assortativity": assortativity,
            "transitivity": transitivity,
        }
    return sweeps


def find_phase_transition(thresholds, gc_fractions):
    """Find the threshold where the biggest GC drop occurs."""
    if len(thresholds) < 2:
        return None, None
    max_drop = 0
    drop_theta = None
    for i in range(len(thresholds) - 1):
        drop = gc_fractions[i] - gc_fractions[i + 1]
        if drop > max_drop:
            max_drop = drop
            drop_theta = (thresholds[i] + thresholds[i + 1]) / 2
    return drop_theta, max_drop


def analyze_weight_effect(results_by_exp, target_theta=0.9):
    """Analyze how weight ratio affects network properties at fixed θ."""
    weight_results = []
    for eid, runs in results_by_exp.items():
        if "chatgpt" not in eid:
            continue
        # Find result closest to target_theta
        for r in runs:
            if abs(r["threshold"] - target_theta) < 0.01:
                # Parse weight from experiment ID
                label = eid.replace("chatgpt-", "").split("-sweep")[0].split("-t0.9")[0]
                parts = label.split("-")
                uw = float(parts[0].replace("user", ""))
                aw = float(parts[1].replace("ai", ""))
                ratio = uw / aw
                weight_results.append({
                    "experiment_id": eid,
                    "label": label,
                    "user_weight": uw,
                    "ai_weight": aw,
                    "user_ai_ratio": ratio,
                    "log_ratio": np.log10(ratio) if ratio > 0 else 0,
                    "metrics": r["metrics"],
                })
                break

    weight_results.sort(key=lambda x: x["user_ai_ratio"])
    return weight_results


def compute_findings(sweeps, weight_results):
    """Identify the most interesting findings across all experiments."""
    findings = []

    # Finding 1: Phase transitions
    for eid, curve in sweeps.items():
        pt_theta, pt_drop = find_phase_transition(
            curve["thresholds"], curve["gc_fraction"]
        )
        if pt_theta and pt_drop > 0.1:
            label = eid.replace("chatgpt-", "")
            findings.append({
                "type": "phase_transition",
                "experiment": eid,
                "theta": round(pt_theta, 3),
                "gc_drop": round(pt_drop, 3),
                "description": f"{label}: GC drops by {pt_drop:.1%} around θ={pt_theta:.3f}",
            })

    # Finding 2: Weight ratio effect on edges
    if weight_results:
        max_edges = max(weight_results, key=lambda x: x["metrics"]["edge_count"])
        min_edges = min(weight_results, key=lambda x: x["metrics"]["edge_count"])
        ratio = max_edges["metrics"]["edge_count"] / max(
            min_edges["metrics"]["edge_count"], 1
        )
        findings.append({
            "type": "weight_edge_ratio",
            "max_config": max_edges["label"],
            "max_edges": max_edges["metrics"]["edge_count"],
            "min_config": min_edges["label"],
            "min_edges": min_edges["metrics"]["edge_count"],
            "fold_difference": round(ratio, 1),
            "description": (
                f"At θ=0.9, {max_edges['label']} produces {ratio:.1f}x more edges "
                f"than {min_edges['label']} ({max_edges['metrics']['edge_count']} vs "
                f"{min_edges['metrics']['edge_count']})"
            ),
        })

    # Finding 3: Modularity vs connectivity anti-correlation
    if weight_results:
        mods = [(w["metrics"].get("modularity", 0), w["metrics"]["edge_count"], w["label"])
                for w in weight_results]
        # Check if modularity decreases as edges increase
        from scipy.stats import spearmanr
        mod_vals = [m[0] for m in mods]
        edge_vals = [m[1] for m in mods]
        if len(mod_vals) >= 3:
            rho, p = spearmanr(edge_vals, mod_vals)
            findings.append({
                "type": "modularity_connectivity_correlation",
                "spearman_rho": round(rho, 3),
                "p_value": round(p, 4),
                "description": (
                    f"Modularity-connectivity Spearman ρ={rho:.3f} (p={p:.4f}): "
                    f"{'anti-correlated' if rho < 0 else 'correlated'}"
                ),
            })

    # Finding 4: Assortativity sign change across thresholds
    for eid, curve in sweeps.items():
        assorts = curve["assortativity"]
        if any(a < 0 for a in assorts) and any(a > 0 for a in assorts):
            # Find crossover point
            for i in range(len(assorts) - 1):
                if assorts[i] < 0 and assorts[i + 1] > 0:
                    crossover = (curve["thresholds"][i] + curve["thresholds"][i + 1]) / 2
                    label = eid.replace("chatgpt-", "")
                    findings.append({
                        "type": "assortativity_sign_change",
                        "experiment": eid,
                        "crossover_theta": round(crossover, 3),
                        "description": (
                            f"{label}: assortativity flips from disassortative to "
                            f"assortative around θ={crossover:.3f}"
                        ),
                    })
                    break

    # Finding 5: Optimal weight ratio (most edges at θ=0.9)
    if weight_results:
        best = max(weight_results, key=lambda x: x["metrics"]["edge_count"])
        findings.append({
            "type": "optimal_weight",
            "config": best["label"],
            "user_weight": best["user_weight"],
            "ai_weight": best["ai_weight"],
            "edges": best["metrics"]["edge_count"],
            "gc_fraction": best["metrics"].get("giant_component_fraction", 0),
            "description": (
                f"Optimal weight at θ=0.9: {best['label']} "
                f"({best['metrics']['edge_count']} edges, "
                f"GC={best['metrics'].get('giant_component_fraction', 0):.1%})"
            ),
        })

    return findings


def main():
    results = load_all_results()
    print(f"Loaded {sum(len(v) for v in results.values())} results from "
          f"{len(results)} experiments\n")

    # Sweep curves
    sweeps = extract_sweep_curves(results)
    print(f"Sweep experiments: {len(sweeps)}")
    for eid, curve in sorted(sweeps.items()):
        label = eid.replace("chatgpt-", "")
        thetas = ", ".join(f"{t:.2f}" for t in curve["thresholds"])
        print(f"  {label}: θ=[{thetas}]")

    # Weight effect at θ=0.9
    weight_results = analyze_weight_effect(results)
    print(f"\nWeight configs at θ=0.9: {len(weight_results)}")
    print(f"\n{'Config':<25} {'Ratio':>6} {'Edges':>7} {'GC%':>6} "
          f"{'Mod':>6} {'Clust':>6} {'Assort':>7}")
    print("-" * 75)
    for w in weight_results:
        m = w["metrics"]
        gc = m.get("giant_component_fraction", 0) * 100
        assort = m.get("assortativity") or 0
        print(f"{w['label']:<25} {w['user_ai_ratio']:>6.2f} "
              f"{m['edge_count']:>7} {gc:>5.1f}% "
              f"{m.get('modularity', 0):>6.3f} "
              f"{m.get('avg_clustering', 0):>6.3f} "
              f"{assort:>+7.3f}")

    # Key findings
    findings = compute_findings(sweeps, weight_results)
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    for i, f in enumerate(findings, 1):
        print(f"\n  [{f['type']}] {f['description']}")

    # Save analysis
    analysis = {
        "num_experiments": len(results),
        "num_results": sum(len(v) for v in results.values()),
        "sweep_experiments": list(sweeps.keys()),
        "weight_effect": [
            {k: v for k, v in w.items() if k != "metrics"}
            | {"edge_count": w["metrics"]["edge_count"],
               "gc_fraction": w["metrics"].get("giant_component_fraction", 0),
               "modularity": w["metrics"].get("modularity", 0)}
            for w in weight_results
        ],
        "findings": findings,
    }
    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved to {RESULTS_DIR / 'analysis.json'}")


if __name__ == "__main__":
    main()
