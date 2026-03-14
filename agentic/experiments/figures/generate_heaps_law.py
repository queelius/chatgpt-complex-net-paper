#!/usr/bin/env python3
"""
Generate the Heaps' law vocabulary growth figure with both raw and meta-concept curves.

Outputs: experiments/figures/heaps_law.pdf and .png
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

RESULTS_DIR = Path(__file__).parent.parent / "results" / "hierarchy_v2"
FIGURES_DIR = Path(__file__).parent


def setup_style():
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'savefig.dpi': 300,
        'figure.dpi': 100,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })
    sns.set_style("whitegrid")


def heaps_law(n, K, beta):
    """Heaps' law: V(n) = K * n^beta."""
    return K * np.power(n, beta)


def main():
    setup_style()

    # ── Load data ────────────────────────────────────────────────────
    with open(RESULTS_DIR / "extraction_state.json") as f:
        extraction_state = json.load(f)

    with open(RESULTS_DIR / "hierarchy_semantic.json") as f:
        hierarchy = json.load(f)

    episodes = extraction_state["episodes"]
    n_episodes = len(episodes)

    # ── Build reverse map: raw concept -> meta-concept ID ────────────
    concept_to_mc = {}
    mc_level = hierarchy["levels"][0]  # level 1 = meta-concepts
    for mc_id, raw_concepts in mc_level["cluster_membership"].items():
        for rc in raw_concepts:
            concept_to_mc[rc.lower()] = mc_id

    # ── Process episodes in order, tracking cumulative vocabularies ───
    raw_vocab = set()
    mc_vocab = set()
    raw_counts = []
    mc_counts = []

    for ep in episodes:
        for concept in ep["concepts"]:
            c_lower = concept.lower()
            raw_vocab.add(c_lower)
            mc_id = concept_to_mc.get(c_lower)
            if mc_id is not None:
                mc_vocab.add(mc_id)
        raw_counts.append(len(raw_vocab))
        mc_counts.append(len(mc_vocab))

    x = np.arange(1, n_episodes + 1, dtype=float)
    raw_counts = np.array(raw_counts, dtype=float)
    mc_counts = np.array(mc_counts, dtype=float)

    print(f"Episodes: {n_episodes}")
    print(f"Final raw vocabulary: {int(raw_counts[-1])}")
    print(f"Final meta-concept vocabulary: {int(mc_counts[-1])}")

    # ── Fit Heaps' law to both curves ────────────────────────────────
    popt_raw, _ = curve_fit(heaps_law, x, raw_counts, p0=[1.0, 0.9], maxfev=10000)
    K_raw, beta_raw = popt_raw

    popt_mc, _ = curve_fit(heaps_law, x, mc_counts, p0=[1.0, 0.3], maxfev=10000)
    K_mc, beta_mc = popt_mc

    print(f"Raw concepts:   V(n) = {K_raw:.2f} * n^{beta_raw:.3f}")
    print(f"Meta-concepts:  V(n) = {K_mc:.2f} * n^{beta_mc:.3f}")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.8))

    # Raw concept curve (observed + fit)
    ax.plot(x, raw_counts, '-', color='#2c3e50', linewidth=1.5, alpha=0.9,
            label=f'Raw concepts ($\\beta = {beta_raw:.3f}$)')
    x_fit = np.linspace(1, n_episodes, 500)
    ax.plot(x_fit, heaps_law(x_fit, *popt_raw), '--', color='#e74c3c',
            linewidth=1.5, alpha=0.8,
            label=f'Heaps fit: $V(n) = {K_raw:.1f} \\cdot n^{{{beta_raw:.3f}}}$')

    # Meta-concept curve (observed + fit)
    ax.plot(x, mc_counts, '-', color='#2980b9', linewidth=1.5, alpha=0.9,
            label=f'Meta-concepts ($\\beta = {beta_mc:.3f}$)')
    ax.plot(x_fit, heaps_law(x_fit, *popt_mc), '--', color='#e67e22',
            linewidth=1.5, alpha=0.8,
            label=f'Heaps fit: $V(n) = {K_mc:.1f} \\cdot n^{{{beta_mc:.3f}}}$')

    # Linear reference line (beta = 1)
    max_val = raw_counts[-1]
    linear_slope = max_val / n_episodes
    ax.plot(x_fit, linear_slope * x_fit, ':', color='#bdc3c7', linewidth=1.2,
            alpha=0.7, label='Linear ($\\beta = 1$)')

    ax.set_xlabel('Number of episodes processed')
    ax.set_ylabel('Cumulative vocabulary size')
    ax.set_title("Vocabulary Growth: Heaps' Law", fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='#cccccc')
    ax.set_xlim(0, n_episodes + 20)
    ax.set_ylim(0, None)

    plt.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'heaps_law.{fmt}',
                    format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved heaps_law.pdf/png")


if __name__ == '__main__':
    main()
