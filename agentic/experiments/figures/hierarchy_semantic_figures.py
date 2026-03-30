#!/usr/bin/env python3
"""
Publication-quality figures for the semantic hierarchy (Experiment 2).

Generates:
  Figure 1: Icicle diagram — 4-level semantic hierarchy with interpretable labels
  Figure 2: Concept frequency distribution + bipartite degree distributions
  Figure 3: Domain composition panel — what's inside each domain
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results" / "hierarchy_v2"
FIGURES_DIR = Path(__file__).parent


# ─── Style Configuration ─────────────────────────────────────────────

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
    sns.set_style("white")


def load_data():
    with open(RESULTS_DIR / "hierarchy_semantic.json") as f:
        return json.load(f)


def load_extraction_state():
    with open(RESULTS_DIR / "extraction_state.json") as f:
        return json.load(f)


# ─── Domain Labels (derived from cluster inspection) ─────────────────

DOMAIN_LABELS = {
    'MC3_C0': 'DevOps & Integration',
    'MC3_C1': 'Software Engineering',
    'MC3_C2': 'Statistical Methods',
    'MC3_C3': 'LLM Engineering',
    'MC3_C4': 'Philosophy & AI Theory',
    'MC3_C5': 'AI Safety & Research',
    'MC3_C6': 'Mathematics & Optimization',
    'MC3_C7': 'ML & Network Science',
}

DOMAIN_COLORS = {
    'MC3_C0': '#E74C3C',   # red
    'MC3_C1': '#9B59B6',   # purple
    'MC3_C2': '#3498DB',   # blue
    'MC3_C3': '#2ECC71',   # green
    'MC3_C4': '#E67E22',   # orange
    'MC3_C5': '#1ABC9C',   # teal
    'MC3_C6': '#F39C12',   # gold
    'MC3_C7': '#34495E',   # dark grey-blue
}


# ─── Figure 1: Icicle Diagram ────────────────────────────────────────

def figure_icicle(data):
    """Top-down icicle: Domains → Themes → (meta-concepts implied) → concepts count."""
    setup_style()

    # Get domain level (L3 = domains with 8 clusters)
    domain_level = None
    theme_level = None
    for lvl in data['levels']:
        if lvl['name'] == 'domains':
            domain_level = lvl
        elif lvl['name'] == 'themes':
            theme_level = lvl

    if not domain_level or not theme_level:
        print("ERROR: Could not find domain/theme levels")
        return

    # Map themes to domains via majority vote on concept overlap
    theme_to_domain = {}
    for theme_id, theme_concepts in theme_level['cluster_membership'].items():
        theme_set = set(theme_concepts)
        best_domain = None
        best_overlap = 0
        for domain_id, domain_concepts in domain_level['cluster_membership'].items():
            overlap = len(theme_set & set(domain_concepts))
            if overlap > best_overlap:
                best_overlap = overlap
                best_domain = domain_id
        theme_to_domain[theme_id] = best_domain

    total_concepts = data['n_concepts']

    fig, ax = plt.subplots(figsize=(18, 8))

    row_height = 0.22
    gap_y = 0.06
    gap_x = 0.002
    y_positions = {
        'domain': 0.68,
        'theme': 0.68 - row_height - gap_y,
    }

    def draw_box(ax, x, y, w, h, color, label, count, fontsize=9, bold=False,
                 label_below=False, alpha=0.88):
        if w < 0.003:
            return
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.004",
            facecolor=color, edgecolor='white', linewidth=1.2,
            alpha=alpha,
        )
        ax.add_patch(rect)

        weight = 'bold' if bold else 'normal'
        text_color = '#FFFFFF' if bold else '#1a1a1a'

        if label_below:
            ax.text(x + w / 2, y - 0.012, label,
                    ha='center', va='top', fontsize=fontsize - 1.5,
                    color='#444444', rotation=45,
                    rotation_mode='anchor')
            if w > 0.015:
                ax.text(x + w / 2, y + h * 0.5, f'{count}',
                        ha='center', va='center', fontsize=fontsize - 2,
                        color='#555555', clip_on=True)
        elif w > 0.05:
            ax.text(x + w / 2, y + h * 0.58, label,
                    ha='center', va='center', fontsize=fontsize,
                    fontweight=weight, color=text_color, clip_on=True)
            ax.text(x + w / 2, y + h * 0.25, f'{count} concepts',
                    ha='center', va='center', fontsize=fontsize - 2,
                    color=text_color if bold else '#555555', clip_on=True)
        elif w > 0.03:
            # Abbreviated
            abbrev = label[:12] + '...' if len(label) > 14 else label
            ax.text(x + w / 2, y + h * 0.58, abbrev,
                    ha='center', va='center', fontsize=fontsize - 1.5,
                    fontweight=weight, color=text_color, clip_on=True)
            ax.text(x + w / 2, y + h * 0.25, f'{count}',
                    ha='center', va='center', fontsize=fontsize - 2,
                    color='#555555', clip_on=True)
        elif w > 0.015:
            ax.text(x + w / 2, y + h * 0.5, f'{count}',
                    ha='center', va='center', fontsize=fontsize - 2.5,
                    color='#555555', clip_on=True)

    # ── Domain row (top) ──
    domain_sorted = sorted(domain_level['cluster_membership'].items(),
                          key=lambda x: -len(x[1]))
    x = 0.0
    domain_spans = {}
    for domain_id, members in domain_sorted:
        w = len(members) / total_concepts * (1.0 - gap_x * 8)
        color = DOMAIN_COLORS.get(domain_id, '#888888')
        label = DOMAIN_LABELS.get(domain_id, domain_id)
        draw_box(ax, x, y_positions['domain'], w, row_height, color, label,
                 len(members), fontsize=11, bold=True)
        domain_spans[domain_id] = (x, w)
        x += w + gap_x

    # ── Theme row (bottom), grouped under domains ──
    for domain_id, (dx, dw) in sorted(domain_spans.items(), key=lambda x: x[1][0]):
        children = [(tid, len(members))
                    for tid, members in theme_level['cluster_membership'].items()
                    if theme_to_domain.get(tid) == domain_id]
        children.sort(key=lambda x: -x[1])

        if not children:
            continue

        child_total = sum(c[1] for c in children)
        x = dx
        for tid, size in children:
            w = (size / child_total) * dw - gap_x * 0.3
            w = max(w, 0.006)
            color = DOMAIN_COLORS.get(domain_id, '#888888')
            # Lighter version for themes
            import matplotlib.colors as mc
            rgb = mc.to_rgb(color)
            light_color = tuple(min(1.0, c * 0.6 + 0.4) for c in rgb)

            # Get representative concepts for this theme
            theme_concepts = theme_level['cluster_membership'][tid][:3]
            label = theme_concepts[0] if theme_concepts else tid

            draw_box(ax, x, y_positions['theme'], w, row_height,
                     light_color, label, size,
                     fontsize=7, label_below=True, alpha=0.75)

            # Connecting line
            ax.plot([x + w / 2, x + w / 2],
                    [y_positions['theme'] + row_height, y_positions['domain']],
                    color='#dddddd', linewidth=0.4, zorder=0)

            x += w + gap_x * 0.3

    # ── Level labels ──
    for label, y_key, sublabel in [
        ('Domains', 'domain', 'k = 8'),
        ('Themes', 'theme', 'k = 50'),
    ]:
        y = y_positions[y_key] + row_height / 2
        ax.text(-0.06, y, label, ha='right', va='center',
                fontsize=11, fontweight='bold', color='#333333')
        ax.text(-0.06, y - 0.035, sublabel, ha='right', va='center',
                fontsize=8, color='#888888')

    # ── Bottom annotation ──
    ax.text(0.5, y_positions['theme'] - 0.10,
            f'3,517 semantic concepts extracted from 1,908 episodes → '
            f'500 meta-concepts → 50 themes → 8 domains',
            ha='center', va='center', fontsize=10, color='#666666', style='italic')

    ax.set_xlim(-0.12, 1.05)
    ax.set_ylim(y_positions['theme'] - 0.22, y_positions['domain'] + row_height + 0.06)
    ax.set_aspect('auto')
    ax.axis('off')

    fig.suptitle('Semantic Concept Hierarchy: LLM-Extracted Knowledge Structure',
                 fontsize=14, fontweight='bold', y=0.97)
    fig.text(0.5, 0.01,
             'Concepts extracted by Claude Code (Sonnet) → embedded with nomic-embed-text → '
             'hierarchically clustered (Ward linkage)',
             ha='center', fontsize=8.5, color='#999999')

    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'semantic_hierarchy_icicle.{fmt}',
                    format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved semantic_hierarchy_icicle.pdf/png")


# ─── Figure 2: Concept Frequency + Degree Distributions ─────────────

def figure_frequency_distributions(data, extraction_state):
    """3-panel: concept frequency (Zipf), concepts/episode, episodes/concept."""
    setup_style()
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel A: Concept rank-frequency (Zipf's law test) ──
    ax = axes[0]

    # Get concept frequencies from extraction state
    concept_counter = Counter()
    for ep in extraction_state['episodes']:
        for c in ep['concepts']:
            concept_counter[c.lower()] += 1

    freqs = sorted(concept_counter.values(), reverse=True)
    ranks = np.arange(1, len(freqs) + 1)

    ax.loglog(ranks, freqs, 'o', color='#3498DB', markersize=2, alpha=0.4)

    # Fit power law: freq ~ rank^(-alpha)
    # Use linear regression in log-log space
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)
    # Fit only to concepts with freq > 1
    mask = np.array(freqs) > 1
    if mask.sum() > 5:
        coeffs = np.polyfit(log_ranks[mask], log_freqs[mask], 1)
        alpha = -coeffs[0]
        fit_line = 10**(coeffs[0] * log_ranks + coeffs[1])
        ax.loglog(ranks, fit_line, '--', color='#E74C3C', linewidth=1.5,
                  label=f'Zipf fit: α = {alpha:.2f}')
        ax.legend(loc='upper right', fontsize=8)

    ax.set_xlabel('Concept rank')
    ax.set_ylabel('Frequency (episodes)')
    ax.set_title('(a) Concept rank-frequency', fontweight='bold')

    # Annotate top concepts
    top_5 = concept_counter.most_common(5)
    for i, (concept, count) in enumerate(top_5):
        if i < 3:
            short = concept[:25] + '...' if len(concept) > 25 else concept
            ax.annotate(short, xy=(i+1, count),
                       xytext=(10, 5 + i*12), textcoords='offset points',
                       fontsize=6, color='#E74C3C',
                       arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=0.5))

    # ── Panel B: Concepts per episode ──
    ax = axes[1]
    concepts_per_ep = [len(ep['concepts']) for ep in extraction_state['episodes']]
    bins = np.arange(0.5, max(concepts_per_ep) + 1.5, 1)
    ax.hist(concepts_per_ep, bins=bins, color='#2ECC71', edgecolor='white',
            alpha=0.8, density=True)
    mean_cpe = np.mean(concepts_per_ep)
    ax.axvline(mean_cpe, color='#E74C3C', linewidth=1.5, linestyle='--',
               label=f'mean = {mean_cpe:.1f}')
    ax.set_xlabel('Concepts per episode')
    ax.set_ylabel('Density')
    ax.set_title('(b) Concepts per episode', fontweight='bold')
    ax.legend(fontsize=8)

    # ── Panel C: Episodes per concept (log-scaled) ──
    ax = axes[2]
    eps_per_concept = list(concept_counter.values())
    freq_of_freq = Counter(eps_per_concept)
    x_vals = sorted(freq_of_freq.keys())
    y_vals = [freq_of_freq[x] for x in x_vals]

    ax.bar(x_vals, y_vals, color='#9B59B6', edgecolor='white', alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('Episodes per concept')
    ax.set_ylabel('Number of concepts (log)')
    ax.set_title('(c) Episodes per concept', fontweight='bold')

    # Annotate singleton count
    singleton_count = freq_of_freq.get(1, 0)
    total = sum(y_vals)
    ax.annotate(f'{singleton_count} singletons\n({100*singleton_count/total:.0f}%)',
                xy=(1, singleton_count), xytext=(3, singleton_count * 0.5),
                fontsize=8, color='#9B59B6', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=1))

    plt.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'semantic_frequency_distributions.{fmt}',
                    format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved semantic_frequency_distributions.pdf/png")


# ─── Figure 3: Domain Composition ────────────────────────────────────

def figure_domain_composition(data):
    """Horizontal bar chart showing domain sizes with representative concepts."""
    setup_style()
    sns.set_style("whitegrid")

    domain_level = None
    for lvl in data['levels']:
        if lvl['name'] == 'domains':
            domain_level = lvl

    if not domain_level:
        print("ERROR: No domain level found")
        return

    # Sort domains by size
    domains = sorted(domain_level['cluster_membership'].items(),
                     key=lambda x: len(x[1]))

    fig, ax = plt.subplots(figsize=(14, 6))

    labels = []
    sizes = []
    colors = []
    representative_concepts = []

    for domain_id, members in domains:
        label = DOMAIN_LABELS.get(domain_id, domain_id)
        labels.append(label)
        sizes.append(len(members))
        colors.append(DOMAIN_COLORS.get(domain_id, '#888888'))
        representative_concepts.append(members[:4])

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, sizes, color=colors, edgecolor='white',
                   linewidth=1, alpha=0.85, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_xlabel('Number of concepts', fontsize=11)
    ax.set_title('Knowledge Domain Sizes with Representative Concepts',
                 fontsize=13, fontweight='bold')

    # Add representative concepts as text
    for i, (bar, reps) in enumerate(zip(bars, representative_concepts)):
        width = bar.get_width()
        rep_text = '  ' + ', '.join(reps[:3])
        if len(rep_text) > 80:
            rep_text = rep_text[:77] + '...'
        ax.text(width + 15, bar.get_y() + bar.get_height() / 2,
                rep_text, va='center', fontsize=7, color='#666666',
                style='italic')

    # Add count labels inside bars
    for bar, size in zip(bars, sizes):
        if size > 300:
            ax.text(bar.get_width() - 10, bar.get_y() + bar.get_height() / 2,
                    str(size), va='center', ha='right', fontsize=9,
                    color='white', fontweight='bold')

    ax.set_xlim(0, max(sizes) * 1.6)
    ax.invert_yaxis()

    plt.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'semantic_domain_composition.{fmt}',
                    format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved semantic_domain_composition.pdf/png")


# ─── Figure 4: Silhouette Scan ───────────────────────────────────────

def figure_silhouette_scan():
    """Scan silhouette score across different k values for the concept hierarchy."""
    setup_style()
    sns.set_style("whitegrid")

    # Load concept embeddings
    concept_emb = np.load(RESULTS_DIR / "concept_embeddings.npy")
    print(f"Loaded concept embeddings: {concept_emb.shape}")

    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import silhouette_score

    Z = linkage(concept_emb, method="ward", metric="euclidean")

    ks = list(range(2, 21)) + [25, 30, 40, 50, 75, 100, 150, 200, 300, 500]
    sils = []
    for k in ks:
        labels = fcluster(Z, t=k, criterion="maxclust")
        n_unique = len(set(labels))
        if n_unique >= 2:
            sil = silhouette_score(concept_emb, labels, metric="cosine",
                                   sample_size=min(5000, len(concept_emb)))
            sils.append(sil)
        else:
            sils.append(0.0)
        print(f"  k={k}: silhouette={sils[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, sils, '-o', color='#2c3e50', linewidth=1.5, markersize=3, alpha=0.7)

    # Highlight chosen levels
    chosen = {8: 'Domains', 50: 'Themes', 500: 'Meta-concepts'}
    marker_colors = {8: '#E74C3C', 50: '#3498DB', 500: '#2ECC71'}
    for k_val, label in chosen.items():
        idx = ks.index(k_val) if k_val in ks else -1
        if idx >= 0:
            s = sils[idx]
            c = marker_colors[k_val]
            ax.plot(k_val, s, 'D', color=c, markersize=10, zorder=5,
                    markeredgecolor='white', markeredgewidth=1.5)
            ax.annotate(f'{label}\n(k={k_val}, s={s:.3f})',
                        xy=(k_val, s),
                        xytext=(15, 10 if k_val != 50 else -25),
                        textcoords='offset points',
                        fontsize=8, color=c, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=c, lw=1),
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec=c, alpha=0.8))

    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette score (cosine)')
    ax.set_title('Silhouette Score vs. Cluster Count for Concept Embeddings',
                 fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.axhline(y=0, color='#cccccc', linestyle='--', linewidth=0.8)

    plt.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'semantic_silhouette_scan.{fmt}',
                    format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved semantic_silhouette_scan.pdf/png")


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    data = load_data()
    extraction_state = load_extraction_state()

    figure_icicle(data)
    figure_frequency_distributions(data, extraction_state)
    figure_domain_composition(data)
    figure_silhouette_scan()

    print("\nAll semantic hierarchy figures generated.")
