#!/usr/bin/env python3
"""
CLS (Complementary Learning Systems) exploration experiments.

Tests whether the CLS consolidation signature exists in the temporal
dynamics of the conversation archive, even though the global analysis
shows semantic categories resisting rather than facilitating consolidation.

Experiments:
  E1: Windowed beta -- does consolidation emerge over time?
  E2: Chronological vs alphabetical ordering effect on beta
  E3: Model-era segmented analysis
  E4: New-concept rate over time (most direct CLS test)
  E5: Domain expansion dynamics

Output:
  experiments/results/hierarchy_v2/cls_exploration.json
  experiments/figures/cls_*.pdf/png
"""

import json
import glob
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

RESULTS_DIR = Path(__file__).parent.parent / "results" / "hierarchy_v2"
FIGURES_DIR = Path(__file__).parent
CONV_DIR = Path.home() / "github" / "chatgpt-conversation-corpus" / "conversations"


def setup_style():
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif', 'savefig.dpi': 300,
        'figure.dpi': 100, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    })
    sns.set_style("whitegrid")


def heaps_law(n, K, beta):
    return K * np.power(n, beta)


def fit_beta(x, y):
    """Fit Heaps' law and return beta, K."""
    try:
        popt, _ = curve_fit(heaps_law, x, y, p0=[1.0, 0.5], maxfev=10000)
        return popt[1], popt[0]
    except RuntimeError:
        return np.nan, np.nan


def build_concept_to_mc(hierarchy):
    """Build raw concept -> meta-concept cluster ID mapping."""
    concept_to_mc = {}
    mc_level = hierarchy["levels"][0]
    for mc_id, raw_concepts in mc_level["cluster_membership"].items():
        for rc in raw_concepts:
            concept_to_mc[rc.lower()] = mc_id
    return concept_to_mc


def get_episode_metaconcepts(episodes, concept_to_mc):
    """For each episode, return the set of meta-concepts it contains."""
    result = []
    for ep in episodes:
        mcs = set()
        for c in ep["concepts"]:
            mc = concept_to_mc.get(c.lower())
            if mc is not None:
                mcs.add(mc)
        result.append(mcs)
    return result


def load_timestamps():
    """Load episode_id -> (created_date, model) from conversation corpus."""
    dates = {}
    models = {}
    for f in glob.glob(str(CONV_DIR / "*.json")):
        with open(f) as fh:
            d = json.load(fh)
        slug = os.path.splitext(os.path.basename(f))[0]
        dates[slug] = d.get("created", "")
        models[slug] = d.get("model") or "unknown"
    return dates, models


def get_domain_for_metaconcept(mc_id, hierarchy):
    """Given a meta-concept ID like MC1_C42, find its domain label."""
    # MC1 = level 0 (meta-concepts), mapped to level 2 (domains)
    # We need to trace the hierarchy: concept -> meta-concept -> theme -> domain
    mc_level = hierarchy["levels"][0]  # meta-concepts (k=500)
    domain_level = hierarchy["levels"][2]  # domains (k=8)

    # mc_id like "MC1_C42" -> cluster index 42 from level 0
    mc_idx_str = mc_id.split("_C")[1]
    mc_idx = int(mc_idx_str)

    # Get a representative concept from this meta-concept
    concepts = mc_level["cluster_membership"].get(mc_id, [])
    if not concepts:
        return "unknown"

    # Find the concept's index in the original concept list
    # and look up its domain label
    domain_labels = domain_level["labels"]
    mc_labels = mc_level["labels"]

    # All concepts in this meta-concept share the same mc label
    # Find one concept's original index
    all_concepts_flat = []
    for cid, members in mc_level["cluster_membership"].items():
        for m in members:
            all_concepts_flat.append(m)

    # Build concept -> domain mapping via labels
    # The labels arrays are parallel to the concept list (frequency-sorted)
    # We need the original concept order
    # Actually, domain_labels[i] gives the domain cluster for concept i
    # and mc_labels[i] gives the meta-concept cluster for concept i
    # We need: for mc_id (which corresponds to mc_labels == mc_idx),
    # what is the majority domain_label?
    mc_labels_arr = np.array(mc_labels)
    domain_labels_arr = np.array(domain_labels)
    mask = mc_labels_arr == mc_idx
    if mask.sum() == 0:
        return "unknown"
    domain_votes = domain_labels_arr[mask]
    majority_domain = int(Counter(domain_votes).most_common(1)[0][0])
    return majority_domain


# ─── E1: Windowed beta ───────────────────────────────────────────────

def experiment_windowed_beta(chrono_episodes, concept_to_mc,
                             window_size=300, step=50):
    """Compute Heaps' beta in sliding windows over chronological episodes."""
    n = len(chrono_episodes)
    results = []

    for start in range(0, n - window_size + 1, step):
        window = chrono_episodes[start:start + window_size]
        vocab = set()
        counts = []
        for ep in window:
            for c in ep["concepts"]:
                mc = concept_to_mc.get(c.lower())
                if mc is not None:
                    vocab.add(mc)
            counts.append(len(vocab))

        x = np.arange(1, len(counts) + 1, dtype=float)
        y = np.array(counts, dtype=float)
        beta, K = fit_beta(x, y)

        results.append({
            "start_idx": start,
            "end_idx": start + window_size,
            "midpoint_idx": start + window_size // 2,
            "beta": round(beta, 4) if not np.isnan(beta) else None,
            "K": round(K, 3) if not np.isnan(K) else None,
            "final_vocab": int(counts[-1]),
        })

    return results


# ─── E2: Chronological vs alphabetical beta ──────────────────────────

def experiment_ordering_comparison(episodes, chrono_episodes, concept_to_mc,
                                   n_random=200):
    """Compare beta under alphabetical, chronological, and random orderings."""
    rng = np.random.default_rng(42)

    def compute_beta_for_ordering(ep_list):
        vocab = set()
        counts = []
        for ep in ep_list:
            for c in ep["concepts"]:
                mc = concept_to_mc.get(c.lower())
                if mc is not None:
                    vocab.add(mc)
            counts.append(len(vocab))
        x = np.arange(1, len(counts) + 1, dtype=float)
        y = np.array(counts, dtype=float)
        beta, _ = fit_beta(x, y)
        return beta

    beta_alpha = compute_beta_for_ordering(episodes)
    beta_chrono = compute_beta_for_ordering(chrono_episodes)

    random_betas = []
    for _ in range(n_random):
        shuffled = list(chrono_episodes)
        rng.shuffle(shuffled)
        b = compute_beta_for_ordering(shuffled)
        if not np.isnan(b):
            random_betas.append(b)
    random_betas = np.array(random_betas)

    # Reverse chronological (anti-CLS: most recent first)
    beta_reverse_chrono = compute_beta_for_ordering(list(reversed(chrono_episodes)))

    return {
        "alphabetical": round(beta_alpha, 4),
        "chronological": round(beta_chrono, 4),
        "reverse_chronological": round(beta_reverse_chrono, 4),
        "random_mean": round(float(np.mean(random_betas)), 4),
        "random_std": round(float(np.std(random_betas)), 4),
        "random_ci_95": [
            round(float(np.percentile(random_betas, 2.5)), 4),
            round(float(np.percentile(random_betas, 97.5)), 4),
        ],
        "chrono_vs_random_z": round(
            (beta_chrono - float(np.mean(random_betas))) / float(np.std(random_betas)), 2
        ),
    }


# ─── E3: Model-era segmented analysis ────────────────────────────────

def experiment_model_era(chrono_episodes, ep_models, concept_to_mc):
    """Compute beta within each model era."""
    # Define eras by date ranges based on OpenAI release history
    # Rather than model field (which is "unknown" for early ones),
    # use date-based eras
    era_results = []

    # Group by rough era
    eras = {
        "GPT-3.5 era (Dec 2022-Mar 2023)": ("2022-12-01", "2023-03-31"),
        "Early GPT-4 (Apr-Sep 2023)": ("2023-04-01", "2023-09-30"),
        "Late GPT-4 (Oct 2023-May 2024)": ("2023-10-01", "2024-05-31"),
        "GPT-4o era (Jun-Nov 2024)": ("2024-06-01", "2024-11-30"),
        "o1/o3 era (Dec 2024-Apr 2025)": ("2024-12-01", "2025-04-30"),
    }

    for era_name, (start_date, end_date) in eras.items():
        era_eps = [
            ep for ep in chrono_episodes
            if hasattr(ep, '_date') and start_date <= ep._date[:10] <= end_date
        ]
        if len(era_eps) < 20:
            era_results.append({
                "era": era_name, "n_episodes": len(era_eps),
                "beta": None, "note": "too few episodes"
            })
            continue

        vocab = set()
        counts = []
        for ep in era_eps:
            for c in ep["concepts"]:
                mc = concept_to_mc.get(c.lower())
                if mc is not None:
                    vocab.add(mc)
            counts.append(len(vocab))

        x = np.arange(1, len(counts) + 1, dtype=float)
        y = np.array(counts, dtype=float)
        beta, K = fit_beta(x, y)

        era_results.append({
            "era": era_name,
            "date_range": [start_date, end_date],
            "n_episodes": len(era_eps),
            "beta": round(beta, 4) if not np.isnan(beta) else None,
            "final_vocab": int(counts[-1]),
        })

    return era_results


# ─── E4: New-concept rate over time ──────────────────────────────────

def experiment_new_concept_rate(chrono_episodes, concept_to_mc):
    """For each episode, compute fraction of its meta-concepts that are new."""
    seen = set()
    results = []

    for i, ep in enumerate(chrono_episodes):
        ep_mcs = set()
        for c in ep["concepts"]:
            mc = concept_to_mc.get(c.lower())
            if mc is not None:
                ep_mcs.add(mc)

        if len(ep_mcs) == 0:
            results.append({"idx": i, "new_fraction": None, "n_concepts": 0})
            continue

        new_count = sum(1 for mc in ep_mcs if mc not in seen)
        new_frac = new_count / len(ep_mcs)
        results.append({
            "idx": i,
            "new_fraction": round(new_frac, 4),
            "n_concepts": len(ep_mcs),
            "n_new": new_count,
        })
        seen.update(ep_mcs)

    return results


# ─── E5: Domain expansion dynamics ───────────────────────────────────

def experiment_domain_expansion(chrono_episodes, concept_to_mc, hierarchy):
    """Track domain activation and cross-domain spanning over time."""
    # Build mc -> domain mapping from cluster_membership overlap
    # Each concept appears in exactly one MC and one domain cluster.
    # Map: concept -> domain_id, then MC -> majority domain.
    domain_level = hierarchy["levels"][2]  # domains (k=8)
    concept_to_domain = {}
    for domain_id, concepts in domain_level["cluster_membership"].items():
        for c in concepts:
            concept_to_domain[c.lower()] = domain_id

    mc_level = hierarchy["levels"][0]  # meta-concepts (k=500)
    mc_to_domain = {}
    for mc_id, concepts in mc_level["cluster_membership"].items():
        domain_votes = [concept_to_domain.get(c.lower(), "unknown")
                        for c in concepts]
        domain_votes = [d for d in domain_votes if d != "unknown"]
        if domain_votes:
            mc_to_domain[mc_id] = Counter(domain_votes).most_common(1)[0][0]

    activated_domains = set()
    domain_counts_over_time = []
    spanning_rates = []  # fraction of recent episodes spanning 2+ domains

    window = 50  # for smoothed spanning rate

    for i, ep in enumerate(chrono_episodes):
        ep_mcs = set()
        for c in ep["concepts"]:
            mc = concept_to_mc.get(c.lower())
            if mc is not None:
                ep_mcs.add(mc)

        ep_domains = set(mc_to_domain.get(mc, -1) for mc in ep_mcs) - {-1}
        activated_domains.update(ep_domains)

        domain_counts_over_time.append({
            "idx": i,
            "activated_domains": len(activated_domains),
            "episode_domains": len(ep_domains),
            "spans_multiple": len(ep_domains) >= 2,
        })

    # Compute smoothed spanning rate
    spanning_bools = [d["spans_multiple"] for d in domain_counts_over_time]
    smoothed_spanning = []
    for i in range(len(spanning_bools)):
        start = max(0, i - window + 1)
        rate = sum(spanning_bools[start:i+1]) / (i - start + 1)
        smoothed_spanning.append(round(rate, 4))

    return {
        "domain_activation": [d["activated_domains"] for d in domain_counts_over_time],
        "smoothed_spanning_rate": smoothed_spanning,
        "window_size": window,
    }


# ─── Main ────────────────────────────────────────────────────────────

def main():
    setup_style()

    print("Loading data...")
    with open(RESULTS_DIR / "extraction_state.json") as f:
        extraction_state = json.load(f)
    with open(RESULTS_DIR / "hierarchy_semantic.json") as f:
        hierarchy = json.load(f)

    episodes = extraction_state["episodes"]
    concept_to_mc = build_concept_to_mc(hierarchy)

    # Load timestamps
    dates, models = load_timestamps()

    # Attach dates to episodes and sort chronologically
    # Use a simple namespace trick to attach dates
    class EpisodeWrapper(dict):
        pass

    chrono_episodes = []
    for ep in episodes:
        eid = ep["episode_id"]
        date = dates.get(eid, "")
        if date:
            w = EpisodeWrapper(ep)
            w._date = date
            chrono_episodes.append(w)

    chrono_episodes.sort(key=lambda e: e._date)
    print(f"  {len(chrono_episodes)}/{len(episodes)} episodes with timestamps")
    print(f"  Date range: {chrono_episodes[0]._date} to {chrono_episodes[-1]._date}")

    results = {}

    # ── E1: Windowed beta ────────────────────────────────────────────
    print("\n=== E1: Windowed beta ===")
    windowed = experiment_windowed_beta(chrono_episodes, concept_to_mc,
                                        window_size=300, step=50)
    betas = [w["beta"] for w in windowed if w["beta"] is not None]
    midpoints = [w["midpoint_idx"] for w in windowed if w["beta"] is not None]
    print(f"  Windows: {len(windowed)}")
    print(f"  Beta range: {min(betas):.4f} to {max(betas):.4f}")

    # Linear regression on beta vs time
    if len(betas) > 2:
        z = np.polyfit(midpoints, betas, 1)
        slope = z[0]
        print(f"  Beta trend slope: {slope:.6f} per episode")
        if slope < 0:
            print("  --> DECREASING beta over time (CLS consolidation signature!)")
        else:
            print("  --> INCREASING beta over time (anti-CLS: diversification)")
    results["E1_windowed_beta"] = {
        "window_size": 300, "step": 50,
        "windows": windowed,
        "trend_slope": round(slope, 6) if len(betas) > 2 else None,
        "trend_direction": "decreasing (CLS)" if slope < 0 else "increasing (anti-CLS)",
    }

    # ── E2: Ordering comparison ──────────────────────────────────────
    print("\n=== E2: Chronological vs alphabetical ===")
    ordering = experiment_ordering_comparison(
        episodes, chrono_episodes, concept_to_mc
    )
    print(f"  Alphabetical:   beta = {ordering['alphabetical']}")
    print(f"  Chronological:  beta = {ordering['chronological']}")
    print(f"  Rev. chrono:    beta = {ordering['reverse_chronological']}")
    print(f"  Random:         beta = {ordering['random_mean']} +/- {ordering['random_std']}")
    print(f"  Chrono z-score: {ordering['chrono_vs_random_z']}")
    if ordering["chronological"] < ordering["random_mean"]:
        print("  --> Chrono beta LOWER than random (CLS-consistent)")
    else:
        print("  --> Chrono beta HIGHER than random (anti-CLS)")
    results["E2_ordering"] = ordering

    # ── E3: Model-era segmented ──────────────────────────────────────
    print("\n=== E3: Model-era segmented analysis ===")
    ep_models = {ep["episode_id"]: models.get(ep["episode_id"], "unknown")
                 for ep in episodes}
    era_results = experiment_model_era(chrono_episodes, ep_models, concept_to_mc)
    for er in era_results:
        beta_str = f"{er['beta']:.4f}" if er.get('beta') is not None else "N/A"
        print(f"  {er['era']}: n={er['n_episodes']}, beta={beta_str}")
    results["E3_model_era"] = era_results

    # ── E4: New-concept rate ─────────────────────────────────────────
    print("\n=== E4: New-concept rate over time ===")
    ncr = experiment_new_concept_rate(chrono_episodes, concept_to_mc)
    valid_ncr = [r for r in ncr if r["new_fraction"] is not None]
    fracs = [r["new_fraction"] for r in valid_ncr]
    # Split into halves
    half = len(fracs) // 2
    first_half_mean = np.mean(fracs[:half])
    second_half_mean = np.mean(fracs[half:])
    print(f"  First half mean new-concept rate:  {first_half_mean:.4f}")
    print(f"  Second half mean new-concept rate: {second_half_mean:.4f}")
    if second_half_mean < first_half_mean:
        print("  --> DECREASING new-concept rate (CLS consolidation!)")
        pct_drop = (first_half_mean - second_half_mean) / first_half_mean * 100
        print(f"      Drop: {pct_drop:.1f}%")
    else:
        print("  --> Increasing or stable (no consolidation)")

    # Linear trend
    x_ncr = np.arange(len(fracs), dtype=float)
    z_ncr = np.polyfit(x_ncr, fracs, 1)
    print(f"  Linear trend slope: {z_ncr[0]:.6f}")

    results["E4_new_concept_rate"] = {
        "first_half_mean": round(first_half_mean, 4),
        "second_half_mean": round(second_half_mean, 4),
        "trend_slope": round(z_ncr[0], 6),
        "direction": "decreasing (CLS)" if z_ncr[0] < 0 else "increasing (anti-CLS)",
        "n_episodes": len(valid_ncr),
    }

    # ── E5: Domain expansion ─────────────────────────────────────────
    print("\n=== E5: Domain expansion dynamics ===")
    expansion = experiment_domain_expansion(chrono_episodes, concept_to_mc, hierarchy)
    activation = expansion["domain_activation"]
    print(f"  Domains activated by episode 100: {activation[99] if len(activation) > 99 else 'N/A'}")
    print(f"  Domains activated by episode 500: {activation[499] if len(activation) > 499 else 'N/A'}")
    print(f"  Final domains activated: {activation[-1]}")
    # When did all 8 domains activate?
    full_activation_idx = next((i for i, a in enumerate(activation) if a >= 8), None)
    print(f"  All 8 domains activated by episode: {full_activation_idx}")

    spanning = expansion["smoothed_spanning_rate"]
    first_quarter = np.mean(spanning[:len(spanning)//4])
    last_quarter = np.mean(spanning[3*len(spanning)//4:])
    print(f"  Cross-domain spanning rate (first quarter): {first_quarter:.3f}")
    print(f"  Cross-domain spanning rate (last quarter):  {last_quarter:.3f}")

    results["E5_domain_expansion"] = {
        "full_activation_episode": full_activation_idx,
        "spanning_first_quarter": round(first_quarter, 4),
        "spanning_last_quarter": round(last_quarter, 4),
    }

    # ── Save results ─────────────────────────────────────────────────
    print("\n=== Saving results ===")
    with open(RESULTS_DIR / "cls_exploration.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved cls_exploration.json")

    # ── Generate figures ─────────────────────────────────────────────
    print("\n=== Generating figures ===")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Windowed beta over time (E1)
    ax = axes[0, 0]
    ax.plot(midpoints, betas, 'o-', color='#2980b9', markersize=3, linewidth=1.5)
    if len(betas) > 2:
        trend_line = np.poly1d(z)
        ax.plot(midpoints, trend_line(midpoints), '--', color='#e74c3c', linewidth=2,
                label=f'Trend: slope={slope:.5f}')
    ax.set_xlabel('Episode index (midpoint)')
    ax.set_ylabel("Heaps' $\\beta$")
    ax.set_title('E1: Windowed $\\beta$ (CLS predicts decrease)', fontweight='bold')
    ax.legend()

    # Panel 2: Ordering comparison (E2)
    ax = axes[0, 1]
    labels = ['Alpha', 'Chrono', 'Rev.Chrono', 'Random']
    values = [ordering['alphabetical'], ordering['chronological'],
              ordering['reverse_chronological'], ordering['random_mean']]
    colors = ['#95a5a6', '#2980b9', '#e67e22', '#bdc3c7']
    bars = ax.bar(labels, values, color=colors, edgecolor='white')
    ax.errorbar(3, ordering['random_mean'], yerr=ordering['random_std']*1.96,
                fmt='none', color='black', capsize=5, linewidth=2)
    ax.set_ylabel("Heaps' $\\beta$")
    ax.set_title('E2: Ordering Effect on $\\beta$', fontweight='bold')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f'{v:.3f}',
                ha='center', fontsize=9)

    # Panel 3: Model-era beta (E3)
    ax = axes[0, 2]
    era_names = [er["era"].split("(")[0].strip() for er in era_results
                 if er.get("beta") is not None]
    era_betas = [er["beta"] for er in era_results if er.get("beta") is not None]
    era_ns = [er["n_episodes"] for er in era_results if er.get("beta") is not None]
    if era_betas:
        bars = ax.barh(range(len(era_names)), era_betas, color='#3498db',
                       edgecolor='white')
        ax.set_yticks(range(len(era_names)))
        ax.set_yticklabels(era_names, fontsize=8)
        ax.set_xlabel("Heaps' $\\beta$")
        ax.set_title('E3: $\\beta$ by Model Era', fontweight='bold')
        for i, (b, n) in enumerate(zip(era_betas, era_ns)):
            ax.text(b + 0.005, i, f'{b:.3f} (n={n})', va='center', fontsize=8)

    # Panel 4: New-concept rate (E4)
    ax = axes[1, 0]
    # Smooth with rolling mean
    window = 50
    smoothed = np.convolve(fracs, np.ones(window)/window, mode='valid')
    x_smooth = np.arange(window//2, window//2 + len(smoothed))
    ax.scatter(range(len(fracs)), fracs, alpha=0.1, s=3, color='#95a5a6')
    ax.plot(x_smooth, smoothed, '-', color='#e74c3c', linewidth=2,
            label=f'Rolling mean (w={window})')
    ax.axhline(first_half_mean, color='#2980b9', linestyle='--', alpha=0.7,
               label=f'First half: {first_half_mean:.3f}')
    ax.axhline(second_half_mean, color='#e67e22', linestyle='--', alpha=0.7,
               label=f'Second half: {second_half_mean:.3f}')
    ax.set_xlabel('Episode index (chronological)')
    ax.set_ylabel('Fraction of new meta-concepts')
    ax.set_title('E4: New-Concept Rate (CLS predicts decrease)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Panel 5: Domain activation curve (E5)
    ax = axes[1, 1]
    ax.plot(range(len(activation)), activation, '-', color='#2980b9', linewidth=2)
    if full_activation_idx is not None:
        ax.axvline(full_activation_idx, color='#e74c3c', linestyle='--',
                   label=f'All 8 domains by ep. {full_activation_idx}')
    ax.set_xlabel('Episode index (chronological)')
    ax.set_ylabel('Domains activated')
    ax.set_title('E5a: Domain Activation', fontweight='bold')
    ax.legend(fontsize=9)

    # Panel 6: Cross-domain spanning rate (E5)
    ax = axes[1, 2]
    ax.plot(range(len(spanning)), spanning, '-', color='#27ae60', linewidth=1.5)
    ax.axhline(first_quarter, color='#2980b9', linestyle='--', alpha=0.7,
               label=f'First quarter: {first_quarter:.3f}')
    ax.axhline(last_quarter, color='#e67e22', linestyle='--', alpha=0.7,
               label=f'Last quarter: {last_quarter:.3f}')
    ax.set_xlabel('Episode index (chronological)')
    ax.set_ylabel('Spanning rate (2+ domains)')
    ax.set_title('E5b: Cross-Domain Spanning Rate', fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'cls_exploration.{fmt}',
                    format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved cls_exploration.pdf/png")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLS EXPLORATION SUMMARY")
    print("=" * 60)
    print(f"E1 Windowed beta trend:      {results['E1_windowed_beta']['trend_direction']}")
    print(f"E2 Chrono vs random:         z = {ordering['chrono_vs_random_z']}")
    print(f"E4 New-concept rate:         {results['E4_new_concept_rate']['direction']}")
    print(f"E5 Spanning rate (1st/last): {first_quarter:.3f} / {last_quarter:.3f}")

    cls_signals = 0
    if slope < 0:
        cls_signals += 1
    if ordering["chronological"] < ordering["random_mean"]:
        cls_signals += 1
    if z_ncr[0] < 0:
        cls_signals += 1
    if last_quarter > first_quarter:
        cls_signals += 1  # more spanning = more integration = CLS-like

    print(f"\nCLS-consistent signals: {cls_signals}/4")
    if cls_signals >= 3:
        print("VERDICT: Strong CLS consolidation evidence in temporal dynamics")
    elif cls_signals >= 2:
        print("VERDICT: Mixed evidence -- some CLS consolidation present")
    elif cls_signals >= 1:
        print("VERDICT: Weak CLS evidence -- mostly anti-consolidation")
    else:
        print("VERDICT: No CLS consolidation evidence in temporal dynamics")


if __name__ == '__main__':
    main()
