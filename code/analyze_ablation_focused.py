#!/usr/bin/env python3
"""
Focused ablation analyses:
1. Fixed threshold=0.9, vary weight ratios
2. Fixed ratio=2:1, vary thresholds
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set publication quality
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
sns.set_style("whitegrid")

def analyze_fixed_threshold(df, threshold=0.9, output_dir=Path(".")):
    """Detailed analysis at fixed threshold=0.9"""
    
    # Filter to threshold 0.9
    df_09 = df[df['threshold'] == threshold].copy()
    df_09 = df_09.sort_values('ratio', ascending=False)
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Weight Ratio Effects at Threshold={threshold}', fontsize=14, fontweight='bold')
    
    # Prepare x-axis labels
    x_labels = []
    for _, row in df_09.iterrows():
        if row['ratio'] >= 1:
            x_labels.append(f"{row['ratio']:.1f}:1")
        else:
            x_labels.append(f"1:{1/row['ratio']:.1f}")
    
    x_pos = np.arange(len(x_labels))
    
    # 1. Modularity (key metric)
    ax = axes[0, 0]
    bars = ax.bar(x_pos, df_09['modularity'], color='steelblue', edgecolor='black', linewidth=0.5)
    # Highlight 2:1 ratio
    for i, ratio in enumerate(df_09['ratio']):
        if abs(ratio - 2.0) < 0.01:
            bars[i].set_color('darkred')
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)
    ax.set_xlabel('User:AI Weight Ratio')
    ax.set_ylabel('Modularity')
    ax.set_title('Community Separation Quality')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='Optimal (0.75)')
    ax.legend()
    
    # 2. Number of communities
    ax = axes[0, 1]
    ax.bar(x_pos, df_09['num_communities'], color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('User:AI Weight Ratio')
    ax.set_ylabel('Number of Communities')
    ax.set_title('Community Detection')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 3. Giant component fraction
    ax = axes[0, 2]
    ax.plot(x_pos, df_09['giant_component_fraction'], 'o-', color='purple', linewidth=2, markersize=8)
    ax.set_xlabel('User:AI Weight Ratio')
    ax.set_ylabel('Giant Component Fraction')
    ax.set_title('Network Connectivity')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 4. Clustering coefficient
    ax = axes[1, 0]
    ax.plot(x_pos, df_09['avg_clustering'], 's-', color='orange', linewidth=2, markersize=8)
    ax.set_xlabel('User:AI Weight Ratio')
    ax.set_ylabel('Average Clustering')
    ax.set_title('Local Clustering Patterns')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 5. Network size (nodes and edges)
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(x_pos, df_09['num_nodes'], 'o-', color='blue', label='Nodes', linewidth=2, markersize=6)
    l2 = ax2.plot(x_pos, df_09['num_edges'], 's-', color='red', label='Edges', linewidth=2, markersize=6)
    ax.set_xlabel('User:AI Weight Ratio')
    ax.set_ylabel('Number of Nodes', color='blue')
    ax2.set_ylabel('Number of Edges', color='red')
    ax.set_title('Network Size')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 6. Density
    ax = axes[1, 2]
    ax.bar(x_pos, df_09['density'], color='teal', alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('User:AI Weight Ratio')
    ax.set_ylabel('Network Density')
    ax.set_title('Edge Density')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'weight_ratio_analysis_t{threshold}.png', bbox_inches='tight')
    plt.savefig(output_dir / f'weight_ratio_analysis_t{threshold}.pdf', bbox_inches='tight')
    plt.close()
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"WEIGHT RATIO ANALYSIS AT THRESHOLD = {threshold}")
    print(f"{'='*80}")
    
    summary_df = df_09[['ratio', 'user_weight', 'ai_weight', 'num_nodes', 'num_edges', 
                        'modularity', 'num_communities', 'avg_clustering', 'giant_component_fraction']]
    summary_df = summary_df.round(3)
    print(summary_df.to_string(index=False))
    
    # Find optimal ratio
    best_mod_idx = df_09['modularity'].idxmax()
    best_mod = df_09.loc[best_mod_idx]
    print(f"\nOptimal ratio for modularity: {best_mod['user_weight']}:{best_mod['ai_weight']}")
    print(f"  Modularity: {best_mod['modularity']:.3f}")
    print(f"  Communities: {int(best_mod['num_communities'])}")
    
    return df_09

def analyze_fixed_ratio(df, user_weight=2.0, ai_weight=1.0, output_dir=Path(".")):
    """Detailed analysis at fixed ratio=2:1"""
    
    # Filter to 2:1 ratio
    df_21 = df[(df['user_weight'] == user_weight) & (df['ai_weight'] == ai_weight)].copy()
    df_21 = df_21.sort_values('threshold')
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Threshold Effects at Weight Ratio {user_weight}:{ai_weight}', fontsize=14, fontweight='bold')
    
    x = df_21['threshold'].values
    
    # 1. Phase transition plot
    ax = axes[0, 0]
    ax.plot(x, df_21['giant_component_fraction'], 'o-', color='darkblue', linewidth=2, markersize=8, label='Giant Component')
    ax.axvline(x=0.875, color='red', linestyle='--', alpha=0.7, label='Phase Transition')
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.7, label='Selected Threshold')
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Giant Component Fraction')
    ax.set_title('Percolation Phase Transition')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])
    
    # 2. Modularity evolution
    ax = axes[0, 1]
    ax.plot(x, df_21['modularity'], 's-', color='purple', linewidth=2, markersize=8)
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Modularity')
    ax.set_title('Community Structure Quality')
    ax.grid(True, alpha=0.3)
    
    # 3. Edge count (log scale)
    ax = axes[0, 2]
    ax.semilogy(x, df_21['num_edges'], 'o-', color='red', linewidth=2, markersize=8)
    ax.axvline(x=0.875, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Number of Edges (log scale)')
    ax.set_title('Network Sparsification')
    ax.grid(True, alpha=0.3)
    
    # 4. Clustering coefficient
    ax = axes[1, 0]
    ax.plot(x, df_21['avg_clustering'], '^-', color='orange', linewidth=2, markersize=8)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Average Clustering')
    ax.set_title('Local Clustering Evolution')
    ax.grid(True, alpha=0.3)
    
    # 5. Number of communities
    ax = axes[1, 1]
    ax.bar(x, df_21['num_communities'], width=0.015, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Number of Communities')
    ax.set_title('Community Detection')
    ax.grid(True, alpha=0.3)
    
    # 6. Density
    ax = axes[1, 2]
    ax.semilogy(x, df_21['density'], 'd-', color='teal', linewidth=2, markersize=8)
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Network Density (log scale)')
    ax.set_title('Edge Density Evolution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'threshold_analysis_r{user_weight}-{ai_weight}.png', bbox_inches='tight')
    plt.savefig(output_dir / f'threshold_analysis_r{user_weight}-{ai_weight}.pdf', bbox_inches='tight')
    plt.close()
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"THRESHOLD ANALYSIS AT RATIO = {user_weight}:{ai_weight}")
    print(f"{'='*80}")
    
    summary_df = df_21[['threshold', 'num_nodes', 'num_edges', 'density',
                        'giant_component_fraction', 'modularity', 'num_communities', 'avg_clustering']]
    summary_df = summary_df.round(4)
    print(summary_df.to_string(index=False))
    
    # Identify phase transition
    if len(df_21) > 1:
        gc_diff = df_21['giant_component_fraction'].diff()
        max_drop_idx = gc_diff.idxmin()
        if pd.notna(max_drop_idx):
            transition = df_21.loc[max_drop_idx, 'threshold']
            print(f"\nPhase transition detected at threshold ≈ {transition:.3f}")
            print(f"  Giant component drops from {df_21.loc[max_drop_idx-1, 'giant_component_fraction']:.1%} to {df_21.loc[max_drop_idx, 'giant_component_fraction']:.1%}")
    
    return df_21

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Focused ablation analyses")
    parser.add_argument("--data-file", 
                       default="../dev/ablation_study/analysis_2d/ablation_2d_results.csv",
                       help="Path to 2D ablation results CSV")
    parser.add_argument("--output-dir", 
                       default="../dev/ablation_study/analysis_2d/focused",
                       help="Output directory for focused analyses")
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating focused ablation analyses...")
    
    # Analysis 1: Fixed threshold=0.9, vary ratios
    df_t09 = analyze_fixed_threshold(df, threshold=0.9, output_dir=output_dir)
    
    # Analysis 2: Fixed ratio=2:1, vary thresholds
    df_r21 = analyze_fixed_ratio(df, user_weight=2.0, ai_weight=1.0, output_dir=output_dir)
    
    # Save focused data
    df_t09.to_csv(output_dir / "fixed_threshold_0.9.csv", index=False)
    df_r21.to_csv(output_dir / "fixed_ratio_2-1.csv", index=False)
    
    print(f"\nFocused analyses saved to: {output_dir}")
    print("  - weight_ratio_analysis_t0.9.png/pdf")
    print("  - threshold_analysis_r2.0-1.0.png/pdf")
    print("  - fixed_threshold_0.9.csv")
    print("  - fixed_ratio_2-1.csv")

if __name__ == "__main__":
    main()