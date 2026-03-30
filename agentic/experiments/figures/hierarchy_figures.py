#!/usr/bin/env python3
"""
Publication-quality figures for the hierarchical memory network (Experiment 1).

Generates:
  Figure 1: Icicle diagram — top-down hierarchy with proportional widths
  Figure 2: Multi-panel analysis (silhouette scan, cluster sizes, cross-cluster bridges)
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results" / "hierarchy"
FIGURES_DIR = Path(__file__).parent
EMBEDDINGS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "embeddings"


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


# ─── Semantic Labels ─────────────────────────────────────────────────

L2_LABELS = {
    'L2_C0': 'DevOps & Packaging',
    'L2_C1': 'Academic Publishing',
    'L2_C2': 'Software Engineering',
    'L2_C3': 'ML & Deep Learning',
    'L2_C4': 'AI Philosophy',
    'L2_C5': 'Creative & Generative',
    'L2_C6': 'Security Research',
    'L2_C7': 'MLE & Bootstrap',
    'L2_C8': 'R Statistics',
    'L2_C9': 'AlgoTree & Data Struct.',
    'L2_C10': 'Stat. Visualization',
    'L2_C11': 'Numerical Computing',
    'L2_C12': 'Physics Simulations',
    'L2_C13': 'Optimization',
    'L2_C14': 'Algorithms & Search',
}

L3_LABELS = {
    'L3_C0': 'Software Infrastructure',
    'L3_C1': 'Machine Learning',
    'L3_C2': 'Ideas & Exploration',
    'L3_C3': 'Statistical Theory',
    'L3_C4': 'Algorithms & Computation',
}

L4_LABELS = {
    'L4_C0': 'Tooling & Publishing',
    'L4_C1': 'Research & Ideas',
}

# Color scheme: muted academic palette
L4_COLORS = {
    'L4_C0': '#4878A8',
    'L4_C1': '#D4764E',
}

L3_COLORS = {
    'L3_C0': '#6FA0C7',
    'L3_C1': '#E8A87C',
    'L3_C2': '#C49BC4',
    'L3_C3': '#7BC47B',
    'L3_C4': '#E88E8E',
}

L2_COLORS = {
    'L2_C0': '#8BB8D6', 'L2_C1': '#9DC8E2', 'L2_C2': '#78AACC',
    'L2_C3': '#F0BF9A', 'L2_C4': '#D4AAD4', 'L2_C5': '#D9B8D9',
    'L2_C6': '#C49BC4', 'L2_C7': '#8ED48E', 'L2_C8': '#A0DCA0',
    'L2_C9': '#F0A0A0', 'L2_C10': '#8ED48E', 'L2_C11': '#F0AAAA',
    'L2_C12': '#F0B0B0', 'L2_C13': '#ECA0A0', 'L2_C14': '#E89898',
}


def load_data():
    with open(RESULTS_DIR / "hierarchy_geometric.json") as f:
        return json.load(f)


def get_parent_mapping(data, child_level_idx, parent_level_idx):
    """Map each cluster at child level to its parent by majority vote."""
    child_membership = data['levels'][child_level_idx]['cluster_membership']
    parent_membership = data['levels'][parent_level_idx]['cluster_membership']

    mapping = {}
    for child_id, child_members in child_membership.items():
        child_set = set(child_members)
        best_parent = None
        best_overlap = 0
        for parent_id, parent_members in parent_membership.items():
            overlap = len(child_set & set(parent_members))
            if overlap > best_overlap:
                best_overlap = overlap
                best_parent = parent_id
        mapping[child_id] = best_parent
    return mapping


# ─── Figure 1: Icicle Diagram ────────────────────────────────────────

def figure_icicle(data):
    """Top-down icicle diagram: each level is a horizontal row of proportional-width boxes."""
    setup_style()

    l2_membership = data['levels'][1]['cluster_membership']
    l3_membership = data['levels'][2]['cluster_membership']
    l4_membership = data['levels'][3]['cluster_membership']

    l2_to_l3 = get_parent_mapping(data, 1, 2)
    l3_to_l4 = get_parent_mapping(data, 2, 3)

    total = 1905

    fig, ax = plt.subplots(figsize=(18, 9))

    row_height = 0.20
    gap_y = 0.08
    gap_x = 0.003
    y_positions = {
        'L4': 0.72,
        'L3': 0.72 - row_height - gap_y,
        'L2': 0.72 - 2 * (row_height + gap_y),
    }

    def draw_box(ax, x, y, w, h, color, label, count, fontsize=9, bold=False,
                 label_below=False):
        """Draw a rounded box with label and count."""
        if w < 0.005:
            return
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.005",
            facecolor=color, edgecolor='white', linewidth=1.5,
            alpha=0.88,
        )
        ax.add_patch(rect)

        weight = 'bold' if bold else 'normal'
        text_color = '#FFFFFF' if bold else '#1a1a1a'

        if label_below:
            # Put label below the box, rotated
            ax.text(x + w / 2, y - 0.015, label,
                    ha='center', va='top', fontsize=fontsize - 1,
                    color='#444444', rotation=45,
                    rotation_mode='anchor')
            # Count inside the box
            if w > 0.02:
                ax.text(x + w / 2, y + h * 0.5, f'{count}',
                        ha='center', va='center', fontsize=fontsize - 2,
                        color='#555555', clip_on=True)
        elif w > 0.04:
            ax.text(x + w / 2, y + h * 0.58, label,
                    ha='center', va='center', fontsize=fontsize,
                    fontweight=weight, color=text_color,
                    clip_on=True)
            ax.text(x + w / 2, y + h * 0.25, f'n={count}',
                    ha='center', va='center', fontsize=fontsize - 1.5,
                    color=text_color if bold else '#555555',
                    clip_on=True)
        elif w > 0.025:
            ax.text(x + w / 2, y + h * 0.5, f'{count}',
                    ha='center', va='center', fontsize=fontsize - 2,
                    color='#333333', clip_on=True)

    # ── L4 (top row): 2 domains ──
    l4_sorted = sorted(l4_membership.items(), key=lambda x: -len(x[1]))
    x = 0.0
    l4_spans = {}
    for l4_id, members in l4_sorted:
        w = len(members) / total * (1.0 - gap_x)
        color = L4_COLORS.get(l4_id, '#888888')
        label = L4_LABELS.get(l4_id, l4_id)
        draw_box(ax, x, y_positions['L4'], w, row_height, color, label,
                 len(members), fontsize=12, bold=True)
        l4_spans[l4_id] = (x, w)
        x += w + gap_x

    # ── L3 (middle row): 5 meta-concepts, grouped under L4 ──
    l3_spans = {}
    for l4_id in [x[0] for x in l4_sorted]:
        l4_x, l4_w = l4_spans[l4_id]
        children = [(l3_id, len(members))
                    for l3_id, members in l3_membership.items()
                    if l3_to_l4.get(l3_id) == l4_id]
        children.sort(key=lambda x: -x[1])
        child_total = sum(c[1] for c in children)

        x = l4_x
        for l3_id, size in children:
            w = (size / child_total) * l4_w - gap_x * 0.5
            w = max(w, 0.01)
            color = L3_COLORS.get(l3_id, '#AAAAAA')
            label = L3_LABELS.get(l3_id, l3_id)
            draw_box(ax, x, y_positions['L3'], w, row_height, color, label,
                     size, fontsize=9)
            l3_spans[l3_id] = (x, w)
            x += w + gap_x * 0.5

    # ── L2 (bottom row): 15 concepts, grouped under L3 ──
    for l3_id in [c[0] for c in sorted(l3_spans.items(), key=lambda x: x[1][0])]:
        l3_x, l3_w = l3_spans[l3_id]
        children = [(l2_id, len(members))
                    for l2_id, members in l2_membership.items()
                    if l2_to_l3.get(l2_id) == l3_id]
        children.sort(key=lambda x: -x[1])
        child_total = sum(c[1] for c in children)

        x = l3_x
        for l2_id, size in children:
            w = (size / child_total) * l3_w - gap_x * 0.3
            w = max(w, 0.008)
            color = L2_COLORS.get(l2_id, '#CCCCCC')
            label = L2_LABELS.get(l2_id, l2_id)
            draw_box(ax, x, y_positions['L2'], w, row_height, color, label,
                     size, fontsize=7.5, label_below=True)
            x += w + gap_x * 0.3

    # ── Connecting lines (subtle) ──
    for l3_id, (l3_x, l3_w) in l3_spans.items():
        l4_id = l3_to_l4.get(l3_id)
        if l4_id and l4_id in l4_spans:
            l4_x, l4_w = l4_spans[l4_id]
            ax.plot([l3_x + l3_w / 2, l3_x + l3_w / 2],
                    [y_positions['L3'] + row_height, y_positions['L4']],
                    color='#cccccc', linewidth=0.5, zorder=0)

    # ── Level labels on the left ──
    for label, y_key, sublabel in [
        ('Domains', 'L4', 'k = 2'),
        ('Meta-concepts', 'L3', 'k = 5'),
        ('Concepts', 'L2', 'k = 15'),
    ]:
        y = y_positions[y_key] + row_height / 2
        ax.text(-0.06, y, label, ha='right', va='center',
                fontsize=11, fontweight='bold', color='#333333')
        ax.text(-0.06, y - 0.04, sublabel, ha='right', va='center',
                fontsize=8, color='#888888')

    # ── Episode count at bottom ──
    ax.text(0.5, y_positions['L2'] - 0.07,
            '1,905 episodic memories (ChatGPT conversations, Dec 2022 – Apr 2025)',
            ha='center', va='center', fontsize=10, color='#666666', style='italic')

    # ── Branching annotation ──
    ax.annotate('', xy=(1.02, y_positions['L4'] + row_height / 2),
                xytext=(1.02, y_positions['L2'] + row_height / 2),
                arrowprops=dict(arrowstyle='<->', color='#999999', lw=1.2))
    ax.text(1.04, y_positions['L3'] + row_height / 2,
            '~3× branching\nat each level',
            ha='left', va='center', fontsize=8, color='#888888')

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(y_positions['L2'] - 0.20, y_positions['L4'] + row_height + 0.06)
    ax.set_aspect('auto')
    ax.axis('off')

    fig.suptitle('Hierarchical Memory Network: 4-Level Geometric Hierarchy',
                 fontsize=15, fontweight='bold', y=0.97)
    fig.text(0.5, 0.01,
             'Ward linkage on 768-dim nomic-embed-text embeddings  |  '
             'NMI = 0.705 vs. Louvain communities  |  '
             'Silhouette at k=15: 0.038',
             ha='center', fontsize=8.5, color='#999999')

    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'hierarchy_icicle.{fmt}',
                    format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved hierarchy_icicle.pdf/png")


# ─── Figure 2: Multi-panel Analysis ─────────────────────────────────

def figure_analysis_panels(data):
    """3-panel figure: silhouette scan, cluster sizes, cross-cluster bridge heatmap."""
    setup_style()
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(17, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.8, 1.2], wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # ── Panel A: Silhouette scan ──
    ax = axes[0]
    scan = data.get('silhouette_scan', [])
    if scan:
        ks = [r['k'] for r in scan]
        sils = [r['silhouette'] for r in scan]
        ax.plot(ks, sils, '-', color='#2c3e50', linewidth=1.8, alpha=0.7)
        ax.plot(ks, sils, 'o', color='#2c3e50', markersize=2.5, alpha=0.5)

        # Highlight chosen cut points
        chosen = {2: 'Domains', 5: 'Meta-concepts', 15: 'Concepts', 50: 'Fine-concepts'}
        marker_colors = {2: '#9b59b6', 5: '#e74c3c', 15: '#2ecc71', 50: '#3498db'}

        for k_val, label in chosen.items():
            matching = [r for r in scan if r['k'] == k_val]
            if matching:
                s = matching[0]['silhouette']
                c = marker_colors.get(k_val, '#e74c3c')
                ax.plot(k_val, s, 'D', color=c, markersize=9, zorder=5,
                        markeredgecolor='white', markeredgewidth=1.5)

                # Offset labels to avoid overlap
                offsets = {2: (6, 0.005), 5: (5, -0.008),
                           15: (4, 0.005), 50: (4, 0.005)}
                dx, dy = offsets.get(k_val, (4, 0.005))
                ax.annotate(f'{label}\n(k={k_val})',
                            xy=(k_val, s), xytext=(k_val + dx, s + dy),
                            fontsize=7.5, color=c, fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color=c, lw=1),
                            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                      ec=c, alpha=0.8))

    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette score')
    ax.set_title('(a) Silhouette score vs. cluster count', fontweight='bold',
                 fontsize=11)
    ax.set_xlim(0, 82)
    ax.axhline(y=0, color='#cccccc', linestyle='--', linewidth=0.8)

    # ── Panel B: Cluster size distribution at each level ──
    ax = axes[1]
    level_data = []
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    level_labels = []

    for lvl in data['levels']:
        sizes = [len(members) for members in lvl['cluster_membership'].values()]
        sizes.sort(reverse=True)
        level_data.append(sizes)
        level_labels.append(f"L{lvl['level']}\n{lvl['name']}\n(k={lvl['n_clusters']})")

    positions = np.arange(len(level_data))
    bp = ax.boxplot(level_data, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=4, alpha=0.5),
                    medianprops=dict(color='#2c3e50', linewidth=2))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(color)

    ax.set_xticks(positions)
    ax.set_xticklabels(level_labels, fontsize=7.5)
    ax.set_ylabel('Cluster size (episodes)')
    ax.set_title('(b) Cluster size distributions', fontweight='bold',
                 fontsize=11)
    ax.set_yscale('log')

    # Branching ratios as annotations between boxes
    branching = data.get('cross_level', {}).get('branching_factors', [])
    for i, bf in enumerate(branching):
        ax.annotate(f'{bf["ratio"]}×',
                    xy=(i + 0.5, ax.get_ylim()[1] * 0.5),
                    fontsize=10, ha='center', va='center',
                    color='#e74c3c', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='#fff5f5',
                              ec='#e74c3c', alpha=0.8))

    # ── Panel C: Cross-cluster bridge heatmap ──
    ax = axes[2]

    from concept_extraction import load_edges_and_build_graph

    edges_file = str(Path(__file__).parent.parent.parent.parent /
                     "data" / "network" / "edges_user2.0-ai1.0_t0.9.json")
    G = load_edges_and_build_graph(edges_file, threshold=0.9)

    # Map nodes to L2 clusters
    l2 = data['levels'][1]
    node_to_cluster = {}
    for cid, members in l2['cluster_membership'].items():
        for m in members:
            node_to_cluster[m] = cid

    # Sort clusters by size for better visual grouping
    cluster_ids = sorted(l2['cluster_membership'].keys(),
                         key=lambda c: -len(l2['cluster_membership'][c]))
    n_clusters = len(cluster_ids)
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}

    bridge_matrix = np.zeros((n_clusters, n_clusters))
    for u, v in G.edges():
        cu = node_to_cluster.get(u)
        cv = node_to_cluster.get(v)
        if cu and cv and cu != cv:
            i, j = cid_to_idx[cu], cid_to_idx[cv]
            bridge_matrix[i, j] += 1
            bridge_matrix[j, i] += 1

    short_labels = []
    for cid in cluster_ids:
        label = L2_LABELS.get(cid, cid)
        short_labels.append(label)

    # Mask upper triangle
    mask = np.triu(np.ones_like(bridge_matrix, dtype=bool), k=0)

    # Custom colormap: white → warm
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(bridge_matrix, mask=mask, ax=ax,
                xticklabels=short_labels, yticklabels=short_labels,
                cmap=cmap, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Bridge edges', 'shrink': 0.7},
                annot=True, fmt='.0f', annot_kws={'size': 7},
                vmin=0, square=True)

    ax.set_title('(c) Cross-concept bridges (episodic network, θ=0.9)',
                 fontweight='bold', fontsize=11)
    ax.tick_params(axis='x', rotation=55, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)

    plt.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES_DIR / f'hierarchy_analysis.{fmt}',
                    format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved hierarchy_analysis.pdf/png")


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    data = load_data()
    figure_icicle(data)
    figure_analysis_panels(data)
    print("\nAll figures generated.")
