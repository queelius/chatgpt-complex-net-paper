#!/usr/bin/env python3
"""
Create a cleaner weight ratio figure with just the 3 core metrics.
Focus on modularity, number of communities, and network size.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def create_focused_weight_ratio_figure(csv_path, output_dir):
    """Create a cleaner figure with just 3 core metrics at threshold 0.9."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter for threshold 0.9
    df_09 = df[df['threshold'] == 0.9].copy()
    
    # Sort by user:ai ratio for consistent ordering
    df_09['ratio'] = df_09['user_weight'] / df_09['ai_weight']
    df_09 = df_09.sort_values('ratio', ascending=False)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Prepare x-axis labels
    x_labels = []
    for _, row in df_09.iterrows():
        if row['user_weight'] == 100 and row['ai_weight'] == 1:
            x_labels.append('100:1')
        elif row['user_weight'] == 1 and row['ai_weight'] == 100:
            x_labels.append('1:100')
        else:
            x_labels.append(f"{row['user_weight']:.1f}:{row['ai_weight']:.1f}".replace('.0', ''))
    
    x_pos = range(len(x_labels))
    
    # Panel 1: Modularity (the star metric)
    ax = axes[0]
    modularity_values = df_09['modularity'].values
    bars = ax.bar(x_pos, modularity_values, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1)
    
    # Highlight the optimal 2:1 ratio
    optimal_idx = None
    for i, label in enumerate(x_labels):
        if label == '2:1':
            optimal_idx = i
            bars[i].set_color('#2ca02c')
            bars[i].set_alpha(1.0)
            bars[i].set_linewidth(2)
            bars[i].set_edgecolor('darkgreen')
            
            # Add annotation
            ax.annotate(f'Optimal\n{modularity_values[i]:.3f}', 
                       xy=(i, modularity_values[i]), 
                       xytext=(i, modularity_values[i] + 0.02),
                       ha='center', fontsize=10, fontweight='bold', color='darkgreen')
    
    ax.set_ylabel('Modularity', fontsize=11)
    ax.set_title('Community Separation Quality', fontsize=12, fontweight='bold')
    ax.set_ylim(0.68, 0.78)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('User:AI Weight Ratio', fontsize=10)
    
    # Add horizontal line at 0.75 for reference
    ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(len(x_pos)-0.5, 0.751, '0.75', color='red', fontsize=9)
    
    # Panel 2: Number of Communities
    ax = axes[1]
    communities = df_09['num_communities'].values
    bars = ax.bar(x_pos, communities, color='darkgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
    
    # Highlight 2:1 ratio
    if optimal_idx is not None:
        bars[optimal_idx].set_alpha(1.0)
        bars[optimal_idx].set_linewidth(2)
        ax.annotate(f'{int(communities[optimal_idx])}', 
                   xy=(optimal_idx, communities[optimal_idx]), 
                   xytext=(optimal_idx, communities[optimal_idx] + 0.3),
                   ha='center', fontsize=10, fontweight='bold', color='darkgreen')
    
    ax.set_ylabel('Number of Communities', fontsize=11)
    ax.set_title('Knowledge Granularity', fontsize=12, fontweight='bold')
    ax.set_ylim(8, 16)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('User:AI Weight Ratio', fontsize=10)
    
    # Panel 3: Network Size (dual axis for nodes and edges)
    ax = axes[2]
    ax2 = ax.twinx()
    
    # Plot nodes and edges
    nodes = df_09['num_nodes'].values
    edges = df_09['num_edges'].values
    
    line1 = ax.plot(x_pos, nodes, 'o-', color='blue', linewidth=2, markersize=6, label='Nodes')
    line2 = ax2.plot(x_pos, edges, 's-', color='red', linewidth=2, markersize=6, label='Edges')
    
    # Highlight optimal point
    if optimal_idx is not None:
        ax.plot(optimal_idx, nodes[optimal_idx], 'o', color='blue', 
               markersize=10, markeredgecolor='darkblue', markeredgewidth=2)
        ax2.plot(optimal_idx, edges[optimal_idx], 's', color='red', 
                markersize=10, markeredgecolor='darkred', markeredgewidth=2)
    
    ax.set_ylabel('Number of Nodes', fontsize=11, color='blue')
    ax2.set_ylabel('Number of Edges', fontsize=11, color='red')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_title('Network Size', fontsize=12, fontweight='bold')
    ax.set_ylim(440, 640)
    ax2.set_ylim(600, 1800)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('User:AI Weight Ratio', fontsize=10)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=9)
    
    # Main title
    fig.suptitle('Weight Ratio Effects at Threshold θ = 0.9', 
                fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig(output_dir / 'weight_ratio_analysis_clean.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'weight_ratio_analysis_clean.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved cleaner weight ratio figure to {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create cleaner weight ratio figure")
    parser.add_argument("--csv", 
                       default="../dev/ablation_study/analysis_2d/focused/fixed_threshold_0.9.csv",
                       help="Path to CSV with fixed threshold data")
    parser.add_argument("--output-dir",
                       default="../comp-net-2025-paper/images",
                       help="Output directory for figures")
    args = parser.parse_args()
    
    create_focused_weight_ratio_figure(args.csv, args.output_dir)

if __name__ == "__main__":
    main()