#!/usr/bin/env python3
"""
Analyze 2D Ablation Study Results: Weight Ratios × Thresholds
Analyzes network metrics across both user:AI weight ratios and similarity thresholds
"""

import os
import sys
import json
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import community detection
try:
    import community as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False
    print("Warning: python-louvain not installed. Install with: pip install python-louvain")

# Set publication-quality plot style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_network_statistics(edges_file):
    """Compute network statistics for a single edge file."""
    
    stats = {}
    
    try:
        # Load edges
        with open(edges_file, 'r') as f:
            edges_data = json.load(f)
        
        # Create NetworkX graph
        G = nx.Graph()
        for edge in edges_data:
            if len(edge) == 3:
                src, dst, weight = edge
                G.add_edge(src, dst, weight=weight)
        
        # Basic statistics
        stats['num_nodes'] = G.number_of_nodes()
        stats['num_edges'] = G.number_of_edges()
        stats['density'] = nx.density(G) if G.number_of_nodes() > 1 else 0
        
        # Handle empty graph
        if G.number_of_nodes() == 0:
            stats['giant_component_size'] = 0
            stats['giant_component_fraction'] = 0
            stats['num_components'] = 0
            stats['avg_degree'] = 0
            stats['avg_clustering'] = 0
            stats['modularity'] = 0
            stats['num_communities'] = 0
            return stats
        
        # Connected components
        components = list(nx.connected_components(G))
        stats['num_components'] = len(components)
        
        if components:
            giant_component = max(components, key=len)
            stats['giant_component_size'] = len(giant_component)
            stats['giant_component_fraction'] = len(giant_component) / G.number_of_nodes()
            
            # Get giant component subgraph for further analysis
            G_giant = G.subgraph(giant_component).copy()
            
            # Degree statistics
            degrees = dict(G.degree())
            stats['avg_degree'] = np.mean(list(degrees.values()))
            stats['std_degree'] = np.std(list(degrees.values()))
            
            # Clustering
            if G_giant.number_of_nodes() > 2:
                stats['avg_clustering'] = nx.average_clustering(G_giant)
                
                # Community detection (if available)
                if HAS_COMMUNITY and G_giant.number_of_edges() > 0:
                    partition = community_louvain.best_partition(G_giant)
                    stats['modularity'] = community_louvain.modularity(partition, G_giant)
                    stats['num_communities'] = len(set(partition.values()))
                else:
                    stats['modularity'] = 0
                    stats['num_communities'] = 1
            else:
                stats['avg_clustering'] = 0
                stats['modularity'] = 0
                stats['num_communities'] = 1
        else:
            stats['giant_component_size'] = 0
            stats['giant_component_fraction'] = 0
            stats['avg_degree'] = 0
            stats['std_degree'] = 0
            stats['avg_clustering'] = 0
            stats['modularity'] = 0
            stats['num_communities'] = 0
            
    except Exception as e:
        logging.error(f"Error computing statistics for {edges_file}: {e}")
        stats['error'] = str(e)
    
    return stats

def analyze_all_combinations(edges_dir, edges_filtered_dir, output_dir):
    """Analyze all combinations of weight ratios and thresholds."""
    
    # Weight ratios (sorted for consistent ordering)
    weight_ratios = [
        (100.0, 1.0),
        (4.0, 1.0),
        (2.0, 1.0),
        (1.618, 1.0),
        (1.0, 1.0),
        (1.0, 1.618),
        (1.0, 2.0),
        (1.0, 4.0),
        (1.0, 100.0)
    ]
    
    # Thresholds
    thresholds = [0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95]
    
    # Collect all statistics
    all_stats = []
    
    logging.info(f"Analyzing {len(weight_ratios)} weight ratios × {len(thresholds)} thresholds = {len(weight_ratios)*len(thresholds)} networks")
    
    # Process each combination
    for user_w, ai_w in tqdm(weight_ratios, desc="Weight ratios"):
        ratio_label = f"user{user_w}-ai{ai_w}"
        
        for threshold in thresholds:
            # Find filtered edge file
            edge_file = Path(edges_filtered_dir) / f"edges_chatgpt-json-llm-{ratio_label}_t{threshold}.json"
            
            if not edge_file.exists():
                logging.warning(f"Missing: {edge_file.name}")
                continue
            
            # Compute statistics
            stats = compute_network_statistics(edge_file)
            
            # Add metadata
            stats['user_weight'] = user_w
            stats['ai_weight'] = ai_w
            stats['ratio'] = user_w / ai_w
            stats['threshold'] = threshold
            stats['ratio_label'] = ratio_label
            stats['file'] = str(edge_file)
            
            all_stats.append(stats)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_stats)
    
    # Save raw data
    df.to_csv(output_dir / "ablation_2d_results.csv", index=False)
    with open(output_dir / "ablation_2d_results.json", 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    return df

def create_2d_heatmaps(df, output_dir):
    """Create 2D heatmaps for key metrics."""
    
    # Metrics to plot
    metrics = [
        ('giant_component_fraction', 'Giant Component Fraction', 'RdYlBu_r'),
        ('modularity', 'Modularity', 'viridis'),
        ('avg_clustering', 'Average Clustering', 'plasma'),
        ('density', 'Network Density', 'YlOrRd'),
        ('num_communities', 'Number of Communities', 'coolwarm'),
        ('avg_degree', 'Average Degree', 'magma')
    ]
    
    # Create pivot tables for heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Network Metrics: Weight Ratios × Similarity Thresholds', fontsize=16, y=1.02)
    
    for idx, (metric, title, cmap) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'{title}\n(not computed)', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Create pivot table
        pivot = df.pivot_table(values=metric, index='ratio', columns='threshold', aggfunc='mean')
        
        # Sort index (ratio values) in descending order for intuitive display
        pivot = pivot.sort_index(ascending=False)
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f' if metric != 'num_communities' else '.0f',
                   cmap=cmap, cbar_kws={'label': title}, ax=ax, 
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title(title)
        ax.set_xlabel('Similarity Threshold')
        ax.set_ylabel('User:AI Weight Ratio')
        
        # Format y-labels to show ratios nicely
        ylabels = [f"{r:.3f}:1" if r >= 1 else f"1:{1/r:.3f}" for r in pivot.index]
        ax.set_yticklabels(ylabels, rotation=0)
        ax.set_xticklabels([f"{t:.3f}" for t in pivot.columns], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_heatmaps_2d.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "metrics_heatmaps_2d.pdf", bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved heatmaps to {output_dir}")

def create_threshold_evolution_plots(df, output_dir):
    """Show how metrics change with threshold for each weight ratio."""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Metric Evolution Across Similarity Thresholds', fontsize=16)
    
    # Group by weight ratio
    ratios = sorted(df['ratio'].unique(), reverse=True)
    
    # Define colors for each ratio
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(ratios)))
    
    metrics = [
        ('num_edges', 'Number of Edges', True),
        ('giant_component_fraction', 'Giant Component Fraction', False),
        ('modularity', 'Modularity', False),
        ('avg_clustering', 'Average Clustering', False),
        ('num_communities', 'Number of Communities', False),
        ('density', 'Network Density', True),
        ('avg_degree', 'Average Degree', False),
        ('num_nodes', 'Number of Nodes', False),
        ('num_components', 'Number of Components', False)
    ]
    
    for idx, (metric, title, use_log) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        if metric not in df.columns:
            continue
        
        for i, ratio in enumerate(ratios):
            ratio_df = df[df['ratio'] == ratio].sort_values('threshold')
            
            # Create label
            if ratio >= 1:
                label = f"{ratio:.1f}:1"
            else:
                label = f"1:{1/ratio:.1f}"
            
            ax.plot(ratio_df['threshold'], ratio_df[metric], 
                   'o-', color=colors[i], label=label, 
                   linewidth=1.5, markersize=4, alpha=0.8)
        
        ax.set_xlabel('Similarity Threshold')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if use_log and df[metric].min() > 0:
            ax.set_yscale('log')
        
        # Only show legend for first plot
        if idx == 0:
            ax.legend(title='User:AI', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_evolution.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "threshold_evolution.pdf", bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved evolution plots to {output_dir}")

def create_summary_table(df, output_dir):
    """Create a summary table for the paper."""
    
    # Select key configurations
    key_configs = [
        (2.0, 1.0, 0.9),    # Paper's configuration
        (1.0, 1.0, 0.9),    # Balanced
        (100.0, 1.0, 0.9),  # User-heavy
        (1.0, 100.0, 0.9),  # AI-heavy
        (2.0, 1.0, 0.8),    # Paper's ratio, lower threshold
        (2.0, 1.0, 0.95),   # Paper's ratio, higher threshold
    ]
    
    summary_rows = []
    
    for user_w, ai_w, threshold in key_configs:
        row_data = df[(df['user_weight'] == user_w) & 
                     (df['ai_weight'] == ai_w) & 
                     (df['threshold'] == threshold)]
        
        if not row_data.empty:
            row = row_data.iloc[0]
            summary_rows.append({
                'Configuration': f"u:{user_w} a:{ai_w} t:{threshold}",
                'Nodes': int(row.get('num_nodes', 0)),
                'Edges': int(row.get('num_edges', 0)),
                'Density': f"{row.get('density', 0):.4f}",
                'Giant Component': f"{row.get('giant_component_fraction', 0):.3f}",
                'Modularity': f"{row.get('modularity', 0):.3f}",
                'Communities': int(row.get('num_communities', 0)),
                'Avg Clustering': f"{row.get('avg_clustering', 0):.3f}",
                'Avg Degree': f"{row.get('avg_degree', 0):.2f}"
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save as CSV and LaTeX
    summary_df.to_csv(output_dir / "key_configurations_summary.csv", index=False)
    summary_df.to_latex(output_dir / "key_configurations_summary.tex", index=False, escape=False)
    
    # Print for paper
    print("\n" + "="*80)
    print("KEY CONFIGURATIONS SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description="Analyze 2D ablation study results")
    parser.add_argument("--data-dir", default="../dev/ablation_study",
                       help="Base directory with ablation study data")
    parser.add_argument("--output-dir", default="../dev/ablation_study/analysis_2d",
                       help="Output directory for analysis results")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation")
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    edges_dir = data_dir / "data" / "edges"
    edges_filtered_dir = data_dir / "data" / "edges_filtered"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check directories exist
    if not edges_filtered_dir.exists():
        logging.error(f"Edges filtered directory not found: {edges_filtered_dir}")
        sys.exit(1)
    
    logging.info("Starting 2D ablation analysis...")
    
    # Analyze all combinations
    df = analyze_all_combinations(edges_dir, edges_filtered_dir, output_dir)
    
    if df.empty:
        logging.error("No data to analyze")
        sys.exit(1)
    
    logging.info(f"Analyzed {len(df)} network configurations")
    
    # Generate visualizations
    if not args.no_plots:
        logging.info("Generating visualizations...")
        create_2d_heatmaps(df, output_dir)
        create_threshold_evolution_plots(df, output_dir)
    
    # Create summary table
    summary_df = create_summary_table(df, output_dir)
    
    # Print overall statistics
    print("\n" + "="*80)
    print("2D ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"Weight ratios analyzed: {df['ratio'].nunique()}")
    print(f"Thresholds analyzed: {df['threshold'].nunique()}")
    print(f"Total configurations: {len(df)}")
    print(f"Results saved to: {output_dir}")
    
    # Find optimal configuration
    if 'modularity' in df.columns:
        best_modularity = df.loc[df['modularity'].idxmax()]
        print(f"\nBest modularity: {best_modularity['modularity']:.3f}")
        print(f"  Configuration: u:{best_modularity['user_weight']} a:{best_modularity['ai_weight']} t:{best_modularity['threshold']}")
    
    if 'giant_component_fraction' in df.columns:
        best_giant = df.loc[df['giant_component_fraction'].idxmax()]
        print(f"\nLargest giant component: {best_giant['giant_component_fraction']:.3f}")
        print(f"  Configuration: u:{best_giant['user_weight']} a:{best_giant['ai_weight']} t:{best_giant['threshold']}")

if __name__ == "__main__":
    main()