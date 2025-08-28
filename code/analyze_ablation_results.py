#!/usr/bin/env python3
"""
Analyze Ablation Study Results
This script analyzes previously generated embeddings and edges to produce
statistics, plots, and publication-ready figures.
Can be run multiple times with different parameters without regenerating data.
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
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from datetime import datetime

# Load environment variables
load_dotenv()

# Set publication-quality plot style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
sns.set_style("whitegrid")
sns.set_palette("husl")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_study_data(data_dir):
    """Load study configuration and data file locations."""
    config_file = data_dir / "study_config.json"
    summary_file = data_dir / "generation_summary.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Study configuration not found at {config_file}")
    
    if not summary_file.exists():
        raise FileNotFoundError(f"Generation summary not found at {summary_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    return config, summary

def compute_network_statistics(edges_file, nodes_dir, cache_dir=None):
    """Compute comprehensive network statistics with caching."""
    
    # Check cache if provided
    if cache_dir:
        cache_file = cache_dir / f"{Path(edges_file).stem}_stats.json"
        if cache_file.exists():
            logging.info(f"Loading cached statistics from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
    
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
        stats['density'] = nx.density(G) if G.number_of_nodes() > 0 else 0
        
        # Connected components
        components = list(nx.connected_components(G))
        stats['num_components'] = len(components)
        
        if components:
            giant_component = max(components, key=len)
            stats['giant_component_size'] = len(giant_component)
            stats['giant_component_fraction'] = len(giant_component) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
            
            # Analyze giant component
            G_giant = G.subgraph(giant_component).copy()
            
            # Degree statistics
            degrees = dict(G_giant.degree())
            if degrees:
                degree_values = list(degrees.values())
                stats['avg_degree'] = np.mean(degree_values)
                stats['std_degree'] = np.std(degree_values)
                stats['max_degree'] = max(degree_values)
                stats['min_degree'] = min(degree_values)
                stats['median_degree'] = np.median(degree_values)
            
            # Clustering
            stats['avg_clustering'] = nx.average_clustering(G_giant)
            stats['transitivity'] = nx.transitivity(G_giant)
            
            # Path lengths (sample if too large)
            if len(G_giant) < 500:
                try:
                    stats['avg_shortest_path'] = nx.average_shortest_path_length(G_giant)
                    stats['diameter'] = nx.diameter(G_giant)
                    stats['radius'] = nx.radius(G_giant)
                except:
                    pass
            
            # Centrality measures
            degree_cent = nx.degree_centrality(G_giant)
            stats['avg_degree_centrality'] = np.mean(list(degree_cent.values()))
            
            # Sample betweenness for large graphs
            k = min(100, len(G_giant))
            betweenness_cent = nx.betweenness_centrality(G_giant, k=k)
            stats['avg_betweenness_centrality'] = np.mean(list(betweenness_cent.values()))
            
            # Community detection
            try:
                import community.community_louvain as community_louvain
                partition = community_louvain.best_partition(G_giant)
                stats['modularity'] = community_louvain.modularity(partition, G_giant)
                stats['num_communities'] = len(set(partition.values()))
            except ImportError:
                logging.warning("python-louvain not installed, using NetworkX greedy modularity")
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(G_giant))
                stats['num_communities'] = len(communities)
                stats['modularity'] = nx.algorithms.community.modularity(G_giant, communities)
            
            # Assortativity
            stats['degree_assortativity'] = nx.degree_assortativity_coefficient(G_giant)
            
            # Store degree sequence for plotting
            stats['degree_sequence'] = degree_values
            
        else:
            stats['giant_component_size'] = 0
            stats['giant_component_fraction'] = 0
        
        # Save to cache if provided
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logging.info(f"Cached statistics to {cache_file}")
        
    except Exception as e:
        logging.error(f"Error computing statistics for {edges_file}: {e}")
        import traceback
        traceback.print_exc()
    
    return stats

def plot_degree_distribution(stats, ratio_label, output_dir):
    """Create publication-quality degree distribution plots."""
    try:
        degrees = stats.get('degree_sequence', [])
        
        if not degrees:
            logging.warning(f"No degree sequence for {ratio_label}")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Degree Distribution Analysis\nWeight Ratio: {ratio_label}', fontsize=16, y=1.02)
        
        # 1. Histogram (linear scale)
        ax1 = axes[0, 0]
        counts, bins, _ = ax1.hist(degrees, bins=min(30, len(set(degrees))), 
                                   edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Degree')
        ax1.set_ylabel('Count')
        ax1.set_title('Degree Distribution (Linear Scale)')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_deg = np.mean(degrees)
        std_deg = np.std(degrees)
        median_deg = np.median(degrees)
        ax1.text(0.65, 0.85, f'Mean: {mean_deg:.2f}\nStd: {std_deg:.2f}\nMedian: {median_deg:.1f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Log-log plot for power law analysis
        ax2 = axes[0, 1]
        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1
        
        if len(degree_counts) > 1:
            x = list(degree_counts.keys())
            y = list(degree_counts.values())
            
            # Remove zeros for log scale
            x_nonzero = [xi for xi, yi in zip(x, y) if xi > 0 and yi > 0]
            y_nonzero = [yi for xi, yi in zip(x, y) if xi > 0 and yi > 0]
            
            if x_nonzero and y_nonzero:
                ax2.scatter(x_nonzero, y_nonzero, alpha=0.6, s=50, color='darkblue')
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                ax2.set_xlabel('Degree (log scale)')
                ax2.set_ylabel('Frequency (log scale)')
                ax2.set_title('Degree Distribution (Log-Log Scale)')
                ax2.grid(True, alpha=0.3, which='both')
                
                # Fit power law (linear regression in log space)
                if len(x_nonzero) > 2:
                    log_x = np.log10(x_nonzero)
                    log_y = np.log10(y_nonzero)
                    slope, intercept, r_value, _, _ = scipy_stats.linregress(log_x, log_y)
                    
                    # Plot fit line
                    x_fit = np.logspace(np.log10(min(x_nonzero)), np.log10(max(x_nonzero)), 100)
                    y_fit = 10**intercept * x_fit**slope
                    ax2.plot(x_fit, y_fit, 'r--', alpha=0.8, linewidth=2,
                            label=f'Power law fit: γ={-slope:.2f}, R²={r_value**2:.3f}')
                    ax2.legend()
        
        # 3. Complementary CDF (P(X >= k))
        ax3 = axes[1, 0]
        sorted_degrees = np.sort(degrees)[::-1]
        ccdf = np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
        
        ax3.plot(sorted_degrees, ccdf, 'o-', markersize=4, linewidth=1.5, color='green')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Degree (log scale)')
        ax3.set_ylabel('P(Degree ≥ k)')
        ax3.set_title('Complementary Cumulative Distribution')
        ax3.grid(True, alpha=0.3, which='both')
        
        # 4. Violin plot with box plot overlay
        ax4 = axes[1, 1]
        parts = ax4.violinplot([degrees], positions=[1], widths=0.7, 
                              showmeans=True, showmedians=True, showextrema=True)
        
        # Customize violin plot colors
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        # Add box plot overlay
        bp = ax4.boxplot([degrees], positions=[1], widths=0.3, 
                        patch_artist=True, notch=True,
                        boxprops=dict(facecolor='white', alpha=0.8))
        
        ax4.set_ylabel('Degree')
        ax4.set_title('Degree Distribution (Violin & Box Plot)')
        ax4.set_xticks([1])
        ax4.set_xticklabels([ratio_label.replace('user', '').replace('ai', ':')])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add percentile annotations
        percentiles = [25, 50, 75, 90, 95, 99]
        if len(degrees) > 0:
            percs = np.percentile(degrees, percentiles)
            text = '\n'.join([f'{p}th: {v:.1f}' for p, v in zip(percentiles, percs)])
            ax4.text(1.3, 0.5, text, transform=ax4.transAxes, 
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_file = output_dir / f"degree_dist_{ratio_label}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.savefig(plot_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved degree distribution plot to {plot_file}")
        return str(plot_file)
        
    except Exception as e:
        logging.error(f"Failed to create degree distribution plot: {e}")
        plt.close()
        return None

def create_comparison_plots(all_stats, output_dir):
    """Create comprehensive comparison plots across all ratios."""
    try:
        # Filter successful stats
        df = pd.DataFrame([s for s in all_stats if s.get('num_nodes', 0) > 0])
        
        if df.empty:
            logging.warning("No successful ratios to compare")
            return
        
        # Sort by user weight for consistent ordering
        df = df.sort_values('user_weight', ascending=False)
        
        # 1. Key metrics comparison (bar plot grid)
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle('Network Metrics Comparison Across Weight Ratios', fontsize=18, y=1.02)
        
        metrics = [
            ('giant_component_fraction', 'Giant Component Size (%)', 100),
            ('avg_degree', 'Average Degree', 1),
            ('avg_clustering', 'Average Clustering', 1),
            ('modularity', 'Modularity', 1),
            ('num_communities', 'Number of Communities', 1),
            ('density', 'Network Density', 1),
            ('avg_degree_centrality', 'Avg Degree Centrality', 1),
            ('avg_betweenness_centrality', 'Avg Betweenness', 1),
            ('degree_assortativity', 'Degree Assortativity', 1)
        ]
        
        for idx, (metric, title, scale) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            if metric in df.columns:
                values = df[metric] * scale
                labels = df['ratio_label'].apply(lambda x: x.replace('user', '').replace('ai', ':'))
                
                bars = ax.bar(range(len(values)), values, color=sns.color_palette("husl", len(values)))
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel(title)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Highlight paper's ratio (2:1)
                for i, label in enumerate(df['ratio_label']):
                    if 'user2.0-ai1.0' in label:
                        bars[i].set_edgecolor('red')
                        bars[i].set_linewidth(3)
        
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "metrics_comparison_grid.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "metrics_comparison_grid.pdf", bbox_inches='tight')
        plt.close()
        
        # 2. Network evolution plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Network Evolution Across User:AI Weight Ratios', fontsize=16)
        
        # Calculate ratio value for x-axis
        df['ratio_value'] = df['user_weight'] / df['ai_weight']
        df = df.sort_values('ratio_value')
        
        # Plot 1: Giant component and density
        ax1 = axes[0, 0]
        if 'giant_component_fraction' in df.columns and 'density' in df.columns:
            ax1_twin = ax1.twinx()
            
            ax1.plot(df['ratio_value'], df['giant_component_fraction'], 
                    'o-', color='blue', linewidth=2, markersize=8, label='Giant Component')
            ax1_twin.plot(df['ratio_value'], df['density'], 
                         's-', color='red', linewidth=2, markersize=8, label='Density')
            
            ax1.set_xlabel('User:AI Weight Ratio (log scale)')
            ax1.set_xscale('log')
            ax1.set_ylabel('Giant Component Fraction', color='blue')
            ax1_twin.set_ylabel('Network Density', color='red')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1_twin.tick_params(axis='y', labelcolor='red')
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Network Connectivity')
            ax1.axvline(x=2.0, color='green', linestyle='--', alpha=0.5, label='Paper ratio (2:1)')
        
        # Plot 2: Clustering and modularity
        ax2 = axes[0, 1]
        if 'avg_clustering' in df.columns:
            ax2.plot(df['ratio_value'], df['avg_clustering'], 'o-', linewidth=2, 
                    markersize=8, label='Avg Clustering', color='purple')
        if 'modularity' in df.columns:
            ax2.plot(df['ratio_value'], df['modularity'], 's-', linewidth=2, 
                    markersize=8, label='Modularity', color='orange')
        
        ax2.set_xlabel('User:AI Weight Ratio (log scale)')
        ax2.set_xscale('log')
        ax2.set_ylabel('Value')
        ax2.set_title('Community Structure')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=2.0, color='green', linestyle='--', alpha=0.5)
        
        # Plot 3: Degree statistics
        ax3 = axes[1, 0]
        if 'avg_degree' in df.columns:
            ax3.plot(df['ratio_value'], df['avg_degree'], 'o-', linewidth=2, 
                    markersize=8, label='Mean', color='darkblue')
            
            if 'std_degree' in df.columns:
                ax3.fill_between(df['ratio_value'], 
                                df['avg_degree'] - df['std_degree'],
                                df['avg_degree'] + df['std_degree'],
                                alpha=0.3, color='lightblue', label='±1 std')
        
        ax3.set_xlabel('User:AI Weight Ratio (log scale)')
        ax3.set_xscale('log')
        ax3.set_ylabel('Degree')
        ax3.set_title('Degree Distribution Statistics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=2.0, color='green', linestyle='--', alpha=0.5)
        
        # Plot 4: Number of nodes and edges
        ax4 = axes[1, 1]
        if 'num_nodes' in df.columns and 'num_edges' in df.columns:
            ax4_twin = ax4.twinx()
            
            ax4.plot(df['ratio_value'], df['num_nodes'], 'o-', color='teal', 
                    linewidth=2, markersize=8, label='Nodes')
            ax4_twin.plot(df['ratio_value'], df['num_edges'], 's-', color='coral', 
                         linewidth=2, markersize=8, label='Edges')
            
            ax4.set_xlabel('User:AI Weight Ratio (log scale)')
            ax4.set_xscale('log')
            ax4.set_ylabel('Number of Nodes', color='teal')
            ax4_twin.set_ylabel('Number of Edges', color='coral')
            ax4.tick_params(axis='y', labelcolor='teal')
            ax4_twin.tick_params(axis='y', labelcolor='coral')
            ax4.grid(True, alpha=0.3)
            ax4.set_title('Network Size')
            ax4.axvline(x=2.0, color='green', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / "network_evolution.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "network_evolution.pdf", bbox_inches='tight')
        plt.close()
        
        # 3. Create LaTeX table
        summary_metrics = ['ratio_label', 'num_nodes', 'num_edges', 'giant_component_fraction',
                          'avg_degree', 'avg_clustering', 'modularity', 'num_communities']
        
        available_summary = [m for m in summary_metrics if m in df.columns]
        summary_df = df[available_summary].copy()
        
        # Format for LaTeX
        summary_df['ratio_label'] = summary_df['ratio_label'].apply(
            lambda x: x.replace('user', '').replace('ai', ':').replace('.0', ''))
        
        for col in summary_df.columns:
            if col != 'ratio_label' and col in summary_df.columns:
                if summary_df[col].dtype in [np.float64, np.float32]:
                    if col in ['num_nodes', 'num_edges', 'num_communities']:
                        summary_df[col] = summary_df[col].astype(int)
                    else:
                        summary_df[col] = summary_df[col].round(3)
        
        # Save tables
        latex_table = summary_df.to_latex(index=False, 
                                         caption="Network Statistics Across User:AI Weight Ratios",
                                         label="tab:ablation_results")
        
        with open(output_dir / "summary_table.tex", 'w') as f:
            f.write(latex_table)
        
        summary_df.to_csv(output_dir / "summary_table.csv", index=False)
        
        logging.info(f"Created comparison plots in {output_dir}")
        
    except Exception as e:
        logging.error(f"Failed to create comparison plots: {e}")
        import traceback
        traceback.print_exc()

def export_to_gexf(nodes_dir, edges_file, output_file):
    """Export network to GEXF format for Gephi."""
    if output_file.exists():
        logging.info(f"GEXF already exists at {output_file}")
        return
    
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
        
        # Export to GEXF
        nx.write_gexf(G, str(output_file))
        logging.info(f"Exported network to {output_file}")
        
    except Exception as e:
        logging.error(f"Failed to export GEXF: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("--data-dir", required=True,
                      help="Directory containing generated data")
    parser.add_argument("--output-dir", default=None,
                      help="Output directory for analysis results (default: data-dir/analysis)")
    parser.add_argument("--cutoff", type=float, default=0.9,
                      help="Similarity cutoff to analyze (default: 0.9)")
    parser.add_argument("--plot-individual", action="store_true",
                      help="Generate individual degree distribution plots")
    parser.add_argument("--export-gexf", action="store_true",
                      help="Export networks to GEXF format")
    parser.add_argument("--focus-ratio", type=str,
                      help="Focus on specific ratio (e.g., 'user2.0-ai1.0')")
    parser.add_argument("--recalculate", action="store_true",
                      help="Recalculate statistics even if cached")
    args = parser.parse_args()
    
    # Set up paths
    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    stats_dir = output_dir / "statistics"
    plots_dir = output_dir / "plots"
    gexf_dir = output_dir / "gexf"
    cache_dir = output_dir / "cache" if not args.recalculate else None
    
    for d in [stats_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    if args.export_gexf:
        gexf_dir.mkdir(parents=True, exist_ok=True)
    
    # Load study data
    logging.info(f"Loading study data from {data_dir}")
    config, summary = load_study_data(data_dir)
    
    # Process each successful ratio
    all_stats = []
    ratios_to_process = summary.get('data_files', {})
    
    if args.focus_ratio:
        ratios_to_process = {k: v for k, v in ratios_to_process.items() 
                            if args.focus_ratio in k}
    
    for ratio_label, data_files in tqdm(ratios_to_process.items(), desc="Analyzing ratios"):
        logging.info(f"\nAnalyzing {ratio_label}")
        
        # Get filtered edges file for the specified cutoff
        filtered_edges_file = data_files.get('filtered_edges', {}).get(str(args.cutoff))
        
        if not filtered_edges_file or not Path(filtered_edges_file).exists():
            logging.warning(f"No filtered edges found for {ratio_label} at cutoff {args.cutoff}")
            continue
        
        # Compute statistics
        embeddings_dir = data_files.get('embeddings')
        stats = compute_network_statistics(filtered_edges_file, embeddings_dir, cache_dir)
        
        # Add metadata
        user_w, ai_w = ratio_label.replace('user', '').replace('ai', '').split('-')
        stats['user_weight'] = float(user_w)
        stats['ai_weight'] = float(ai_w)
        stats['ratio_label'] = ratio_label
        stats['cutoff'] = args.cutoff
        
        all_stats.append(stats)
        
        # Save individual statistics
        stats_file = stats_dir / f"stats_{ratio_label}_cutoff_{args.cutoff}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate individual degree distribution plot if requested
        if args.plot_individual and stats.get('degree_sequence'):
            plot_degree_distribution(stats, ratio_label, plots_dir / "degree_distributions")
        
        # Export to GEXF if requested
        if args.export_gexf:
            gexf_file = gexf_dir / f"network_{ratio_label}_cutoff_{args.cutoff}.gexf"
            export_to_gexf(embeddings_dir, filtered_edges_file, gexf_file)
    
    # Generate comparison plots
    if len(all_stats) > 1:
        logging.info("\nGenerating comparison plots...")
        create_comparison_plots(all_stats, plots_dir / "comparisons")
    
    # Save combined statistics
    combined_stats_file = output_dir / f"combined_statistics_cutoff_{args.cutoff}.json"
    with open(combined_stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # Create summary DataFrame
    df_stats = pd.DataFrame(all_stats)
    if not df_stats.empty:
        df_stats = df_stats.sort_values('user_weight', ascending=False)
        df_stats.to_csv(output_dir / f"summary_cutoff_{args.cutoff}.csv", index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Analyzed {len(all_stats)} ratios")
    print(f"Cutoff: {args.cutoff}")
    print(f"Results saved to: {output_dir}")
    
    # Print key findings
    if df_stats is not None and not df_stats.empty:
        print("\nKey Metrics Summary:")
        print("-" * 40)
        
        key_metrics = ['ratio_label', 'num_nodes', 'num_edges', 
                      'giant_component_fraction', 'modularity']
        
        available = [m for m in key_metrics if m in df_stats.columns]
        print(df_stats[available].to_string(index=False))

if __name__ == "__main__":
    main()