#!/usr/bin/env python3
"""
Ablation Study: User-AI Embedding Weight Ratios
Generates embeddings with 9 different weight ratios and computes network statistics.
"""

import os
import sys
import json
import subprocess
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
sns.set_palette("husl")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define weight ratios for ablation study
WEIGHT_RATIOS = [
    (100.0, 1.0),   # Near-pure user perspective
    (4.0, 1.0),     # Strong user emphasis
    (2.0, 1.0),     # Paper's ratio
    (1.618, 1.0),   # Golden ratio (user-favored)
    (1.0, 1.0),     # Perfect balance
    (1.0, 1.618),   # Inverse golden ratio
    (1.0, 2.0),     # Inverse paper ratio
    (1.0, 4.0),     # Strong AI emphasis
    (1.0, 100.0),   # Near-pure AI perspective
]

# Load configuration from environment
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.9"))
MAX_NODES_EXACT_PATH = int(os.getenv("MAX_NODES_EXACT_PATH", "1000"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "100"))
BETWEENNESS_K = int(os.getenv("BETWEENNESS_K", "100"))

def run_command(cmd):
    """Execute shell command and handle errors."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {cmd}")
        logging.error(f"Error: {e.stderr}")
        return None

def generate_embeddings(input_dir, user_weight, ai_weight, base_output_dir):
    """Generate embeddings for a specific weight ratio."""
    output_dir = f"{base_output_dir}/chatgpt-json-llm-user{user_weight}-ai{ai_weight}"
    
    # Check if already exists
    if os.path.exists(output_dir):
        logging.info(f"Embeddings already exist at {output_dir}, skipping...")
        return output_dir
    
    cmd = f"""python cli.py node-embeddings \
        --input-dir {input_dir} \
        --method role-aggregate \
        --embedding-method llm \
        --user-weight {user_weight} \
        --assistant-weight {ai_weight} \
        --output-dir {output_dir}"""
    
    logging.info(f"Generating embeddings with user:{user_weight} ai:{ai_weight}")
    result = run_command(cmd)
    
    if result:
        logging.info(f"Successfully generated embeddings at {output_dir}")
        return output_dir
    else:
        logging.error(f"Failed to generate embeddings for ratio {user_weight}:{ai_weight}")
        return None

def generate_edges(embeddings_dir, output_file, use_gpu=False):
    """Generate edge list from embeddings."""
    if os.path.exists(output_file):
        logging.info(f"Edges already exist at {output_file}, skipping...")
        return output_file
    
    cmd_type = "edges-gpu" if use_gpu else "edges"
    cmd = f"""python cli.py {cmd_type} \
        --input-dir {embeddings_dir} \
        --output-file {output_file}"""
    
    logging.info(f"Generating edges from {embeddings_dir}")
    result = run_command(cmd)
    
    if result:
        logging.info(f"Successfully generated edges at {output_file}")
        return output_file
    else:
        logging.error(f"Failed to generate edges from {embeddings_dir}")
        return None

def filter_edges(input_file, output_file, cutoff=SIMILARITY_THRESHOLD):
    """Filter edges by similarity threshold."""
    if os.path.exists(output_file):
        logging.info(f"Filtered edges already exist at {output_file}, skipping...")
        return output_file
    
    cmd = f"""python cli.py cut-off \
        --input-file {input_file} \
        --output-file {output_file} \
        --cutoff {cutoff}"""
    
    logging.info(f"Filtering edges with cutoff {cutoff}")
    result = run_command(cmd)
    
    if result:
        logging.info(f"Successfully filtered edges at {output_file}")
        return output_file
    else:
        logging.error(f"Failed to filter edges")
        return None

def compute_network_statistics(edges_file, nodes_dir):
    """Compute comprehensive network statistics."""
    stats = {}
    
    try:
        # Load edges
        with open(edges_file, 'r') as f:
            edges_data = json.load(f)
        
        # Create NetworkX graph
        G = nx.Graph()
        for src, dst, weight in edges_data:
            if weight >= SIMILARITY_THRESHOLD:
                G.add_edge(src, dst, weight=weight)
        
        # Basic statistics
        stats['num_nodes'] = G.number_of_nodes()
        stats['num_edges'] = G.number_of_edges()
        stats['density'] = nx.density(G)
        
        # Connected components
        components = list(nx.connected_components(G))
        stats['num_components'] = len(components)
        
        if components:
            giant_component = max(components, key=len)
            stats['giant_component_size'] = len(giant_component)
            stats['giant_component_fraction'] = len(giant_component) / G.number_of_nodes()
            
            # Analyze giant component
            G_giant = G.subgraph(giant_component)
            
            # Degree statistics
            degrees = dict(G_giant.degree())
            stats['avg_degree'] = np.mean(list(degrees.values()))
            stats['std_degree'] = np.std(list(degrees.values()))
            stats['max_degree'] = max(degrees.values())
            stats['min_degree'] = min(degrees.values())
            
            # Clustering
            stats['avg_clustering'] = nx.average_clustering(G_giant)
            stats['transitivity'] = nx.transitivity(G_giant)
            
            # Path lengths (sample if too large)
            if len(G_giant) < 1000:
                stats['avg_shortest_path'] = nx.average_shortest_path_length(G_giant)
                stats['diameter'] = nx.diameter(G_giant)
            else:
                # Sample for large graphs
                sample_nodes = np.random.choice(list(G_giant.nodes()), 
                                              min(100, len(G_giant)), replace=False)
                G_sample = G_giant.subgraph(sample_nodes)
                if nx.is_connected(G_sample):
                    stats['avg_shortest_path_sample'] = nx.average_shortest_path_length(G_sample)
            
            # Centrality measures (top nodes)
            degree_cent = nx.degree_centrality(G_giant)
            stats['avg_degree_centrality'] = np.mean(list(degree_cent.values()))
            
            betweenness_cent = nx.betweenness_centrality(G_giant, k=min(100, len(G_giant)))
            stats['avg_betweenness_centrality'] = np.mean(list(betweenness_cent.values()))
            
            # Community detection (modularity)
            try:
                import community.community_louvain as community_louvain
                partition = community_louvain.best_partition(G_giant)
                stats['modularity'] = community_louvain.modularity(partition, G_giant)
                stats['num_communities'] = len(set(partition.values()))
            except ImportError:
                logging.warning("python-louvain not installed, skipping modularity calculation")
                # Alternative using NetworkX's greedy modularity
                from networkx.algorithms.community import greedy_modularity_communities
                communities = greedy_modularity_communities(G_giant)
                stats['num_communities'] = len(communities)
                stats['modularity'] = nx.algorithms.community.modularity(G_giant, communities)
            
            # Assortativity
            stats['degree_assortativity'] = nx.degree_assortativity_coefficient(G_giant)
            
        else:
            stats['giant_component_size'] = 0
            stats['giant_component_fraction'] = 0
        
        # Small world metrics
        if G.number_of_nodes() > 10:
            # Generate random graph for comparison
            n = G.number_of_nodes()
            m = G.number_of_edges()
            p = 2 * m / (n * (n - 1)) if n > 1 else 0
            
            G_random = nx.erdos_renyi_graph(n, p)
            if nx.is_connected(G_random):
                L_random = nx.average_shortest_path_length(G_random)
                C_random = nx.average_clustering(G_random)
                
                if stats.get('avg_shortest_path') and stats.get('avg_clustering'):
                    L_actual = stats['avg_shortest_path']
                    C_actual = stats['avg_clustering']
                    
                    # Small world coefficient (sigma)
                    if L_random > 0 and C_random > 0:
                        stats['small_world_sigma'] = (C_actual / C_random) / (L_actual / L_random)
                        stats['small_world_omega'] = L_random / L_actual - C_actual / C_random
        
    except Exception as e:
        logging.error(f"Error computing statistics: {e}")
        import traceback
        traceback.print_exc()
    
    return stats

def plot_degree_distribution(G, ratio_label, output_dir):
    """Create publication-quality degree distribution plots."""
    try:
        # Get degree sequence
        degrees = [d for n, d in G.degree()]
        
        if not degrees:
            logging.warning(f"No degrees to plot for {ratio_label}")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Degree Distribution Analysis\nRatio: {ratio_label}', fontsize=16, y=1.02)
        
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
        ax1.text(0.7, 0.9, f'Mean: {mean_deg:.2f}\nStd: {std_deg:.2f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
        
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
                    ax2.plot(x_fit, y_fit, 'r--', alpha=0.8, 
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
        ax4.set_xticklabels([ratio_label])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add percentile annotations
        percentiles = [25, 50, 75, 90, 95, 99]
        percs = np.percentile(degrees, percentiles)
        text = '\n'.join([f'{p}th: {v:.1f}' for p, v in zip(percentiles, percs)])
        ax4.text(1.3, 0.5, text, transform=ax4.transAxes, 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        plt.tight_layout()
        
        # Save figure
        plot_file = output_dir / f"degree_distribution_{ratio_label}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.savefig(plot_file.with_suffix('.pdf'), bbox_inches='tight')  # PDF version for publication
        plt.close()
        
        logging.info(f"Saved degree distribution plot to {plot_file}")
        return str(plot_file)
        
    except Exception as e:
        logging.error(f"Failed to create degree distribution plot: {e}")
        plt.close()
        return None

def create_comparison_plots(all_stats, base_dir):
    """Create comprehensive comparison plots across all ratios."""
    try:
        plots_dir = base_dir / "comparison_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Filter successful stats
        df = pd.DataFrame([s for s in all_stats if s.get('status') == 'success'])
        
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
                for i, label in enumerate(labels):
                    if '2.0:1.0' in label:
                        bars[i].set_edgecolor('red')
                        bars[i].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "metrics_comparison_grid.png", dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / "metrics_comparison_grid.pdf", bbox_inches='tight')
        plt.close()
        
        # 2. Network evolution plot (line plots showing how metrics change with ratio)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Network Evolution Across User:AI Weight Ratios', fontsize=16)
        
        # Calculate ratio value for x-axis (user_weight / ai_weight)
        df['ratio_value'] = df['user_weight'] / df['ai_weight']
        df = df.sort_values('ratio_value')
        
        # Plot 1: Giant component and density
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(df['ratio_value'], df['giant_component_fraction'], 
                        'o-', color='blue', linewidth=2, markersize=8, label='Giant Component')
        line2 = ax1_twin.plot(df['ratio_value'], df['density'], 
                             's-', color='red', linewidth=2, markersize=8, label='Density')
        
        ax1.set_xlabel('User:AI Weight Ratio (log scale)')
        ax1.set_xscale('log')
        ax1.set_ylabel('Giant Component Fraction', color='blue')
        ax1_twin.set_ylabel('Network Density', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Network Connectivity')
        
        # Mark paper's ratio
        ax1.axvline(x=2.0, color='green', linestyle='--', alpha=0.5, label='Paper ratio (2:1)')
        
        # Plot 2: Clustering and modularity
        ax2 = axes[0, 1]
        ax2.plot(df['ratio_value'], df['avg_clustering'], 'o-', linewidth=2, 
                markersize=8, label='Avg Clustering', color='purple')
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
        ax4_twin = ax4.twinx()
        
        line3 = ax4.plot(df['ratio_value'], df['num_nodes'], 'o-', color='teal', 
                        linewidth=2, markersize=8, label='Nodes')
        line4 = ax4_twin.plot(df['ratio_value'], df['num_edges'], 's-', color='coral', 
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
        plt.savefig(plots_dir / "network_evolution.png", dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / "network_evolution.pdf", bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap of key metrics
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Select metrics for heatmap
        heatmap_metrics = ['giant_component_fraction', 'avg_degree', 'avg_clustering', 
                          'modularity', 'density', 'num_communities']
        
        # Normalize each metric to 0-1 scale for comparison
        heatmap_data = []
        metric_names = []
        for metric in heatmap_metrics:
            if metric in df.columns:
                values = df[metric].values
                # Normalize
                if values.max() > values.min():
                    normalized = (values - values.min()) / (values.max() - values.min())
                else:
                    normalized = values
                heatmap_data.append(normalized)
                metric_names.append(metric.replace('_', ' ').title())
        
        if heatmap_data:
            heatmap_array = np.array(heatmap_data)
            
            sns.heatmap(heatmap_array, 
                       xticklabels=df['ratio_label'].apply(lambda x: x.replace('user', '').replace('ai', ':')),
                       yticklabels=metric_names,
                       cmap='YlOrRd', annot=True, fmt='.2f',
                       cbar_kws={'label': 'Normalized Value'},
                       ax=ax)
            
            ax.set_title('Normalized Network Metrics Heatmap', fontsize=16)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Highlight paper's ratio column
            paper_idx = None
            for i, label in enumerate(df['ratio_label']):
                if 'user2.0-ai1.0' in label:
                    paper_idx = i
                    break
            
            if paper_idx is not None:
                ax.add_patch(plt.Rectangle((paper_idx, 0), 1, len(metric_names), 
                                         fill=False, edgecolor='red', linewidth=3))
            
            plt.tight_layout()
            plt.savefig(plots_dir / "metrics_heatmap.png", dpi=300, bbox_inches='tight')
            plt.savefig(plots_dir / "metrics_heatmap.pdf", bbox_inches='tight')
            plt.close()
        
        # 4. Summary statistics table (LaTeX format)
        summary_metrics = ['ratio_label', 'num_nodes', 'num_edges', 'giant_component_fraction',
                          'avg_degree', 'avg_clustering', 'modularity', 'num_communities']
        
        available_summary = [m for m in summary_metrics if m in df.columns]
        summary_df = df[available_summary].copy()
        
        # Format ratio labels
        summary_df['ratio_label'] = summary_df['ratio_label'].apply(
            lambda x: x.replace('user', '').replace('ai', ':').replace('.0', ''))
        
        # Round numerical values
        for col in summary_df.columns:
            if col != 'ratio_label' and summary_df[col].dtype in [np.float64, np.float32]:
                if col in ['num_nodes', 'num_edges', 'num_communities']:
                    summary_df[col] = summary_df[col].astype(int)
                else:
                    summary_df[col] = summary_df[col].round(3)
        
        # Save as LaTeX table
        latex_table = summary_df.to_latex(index=False, 
                                         caption="Network Statistics Across User:AI Weight Ratios",
                                         label="tab:ablation_results")
        
        with open(plots_dir / "summary_table.tex", 'w') as f:
            f.write(latex_table)
        
        # Also save as CSV for easy access
        summary_df.to_csv(plots_dir / "summary_table.csv", index=False)
        
        logging.info(f"Created comparison plots in {plots_dir}")
        
    except Exception as e:
        logging.error(f"Failed to create comparison plots: {e}")
        import traceback
        traceback.print_exc()

def export_to_gexf(nodes_dir, edges_file, output_file):
    """Export network to GEXF format for Gephi."""
    if os.path.exists(output_file):
        logging.info(f"GEXF already exists at {output_file}, skipping...")
        return output_file
    
    cmd = f"""python cli.py export \
        --nodes-dir {nodes_dir} \
        --edges-file {edges_file} \
        --format gexf \
        --output-file {output_file}"""
    
    logging.info(f"Exporting to GEXF format")
    result = run_command(cmd)
    
    if result:
        logging.info(f"Successfully exported to {output_file}")
        return output_file
    else:
        logging.error(f"Failed to export to GEXF")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run ablation study on user-AI embedding weight ratios")
    parser.add_argument("--input-dir", default="../dev/chatgpt-4-11-2025_json_no_embeddings",
                      help="Input directory with raw conversation JSONs")
    parser.add_argument("--output-base", default="../dev/ablation_study",
                      help="Base output directory for all results")
    parser.add_argument("--use-gpu", action="store_true", 
                      help="Use GPU for edge generation")
    parser.add_argument("--skip-embeddings", action="store_true",
                      help="Skip embedding generation if already exists")
    args = parser.parse_args()
    
    # Create output directory structure
    base_dir = Path(args.output_base)
    base_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = base_dir / "embeddings"
    edges_dir = base_dir / "edges"
    gexf_dir = base_dir / "gexf"
    stats_dir = base_dir / "statistics"
    plots_dir = base_dir / "plots"
    
    for d in [embeddings_dir, edges_dir, gexf_dir, stats_dir, plots_dir]:
        d.mkdir(exist_ok=True)
    
    # Store all statistics and failures
    all_stats = []
    failures = []
    
    # Process each weight ratio
    for user_w, ai_w in tqdm(WEIGHT_RATIOS, desc="Processing weight ratios"):
        ratio_label = f"user{user_w}-ai{ai_w}"
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing ratio {ratio_label}")
        logging.info(f"{'='*60}")
        
        # Track any failures for this ratio
        ratio_failures = []
        
        # Step 1: Generate embeddings
        if not args.skip_embeddings:
            emb_dir = generate_embeddings(
                args.input_dir, user_w, ai_w, str(embeddings_dir)
            )
        else:
            emb_dir = f"{embeddings_dir}/chatgpt-json-llm-user{user_w}-ai{ai_w}"
        
        if not emb_dir or not os.path.exists(emb_dir):
            error_msg = f"Embedding generation failed for {ratio_label}"
            logging.error(error_msg)
            ratio_failures.append(error_msg)
            failures.append({"ratio": ratio_label, "stage": "embeddings", "error": error_msg})
            continue
        
        # Step 2: Generate edges
        edges_file = edges_dir / f"edges_{ratio_label}.json"
        edges_result = generate_edges(emb_dir, str(edges_file), args.use_gpu)
        
        if not edges_result:
            error_msg = f"Edge generation failed for {ratio_label}"
            logging.error(error_msg)
            failures.append({"ratio": ratio_label, "stage": "edges", "error": error_msg})
            continue
        
        # Step 3: Filter edges
        filtered_edges_file = edges_dir / f"edges_{ratio_label}_filtered_{SIMILARITY_THRESHOLD}.json"
        try:
            filtered_result = filter_edges(str(edges_file), str(filtered_edges_file))
            if not filtered_result:
                error_msg = f"Edge filtering failed for {ratio_label}"
                logging.error(error_msg)
                failures.append({"ratio": ratio_label, "stage": "filter", "error": error_msg})
                continue
        except Exception as e:
            error_msg = f"Edge filtering exception for {ratio_label}: {str(e)}"
            logging.error(error_msg)
            failures.append({"ratio": ratio_label, "stage": "filter", "error": error_msg})
            continue
        
        # Step 4: Compute network statistics and create plots
        logging.info(f"Computing network statistics for {ratio_label}")
        G = None
        try:
            stats = compute_network_statistics(str(filtered_edges_file), emb_dir)
            stats['user_weight'] = user_w
            stats['ai_weight'] = ai_w
            stats['ratio_label'] = ratio_label
            stats['status'] = 'success'
            all_stats.append(stats)
            
            # Save individual statistics
            stats_file = stats_dir / f"stats_{ratio_label}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Create NetworkX graph for plotting
            with open(filtered_edges_file, 'r') as f:
                edges_data = json.load(f)
            
            G = nx.Graph()
            for src, dst, weight in edges_data:
                if weight >= SIMILARITY_THRESHOLD:
                    G.add_edge(src, dst, weight=weight)
            
            # Generate degree distribution plots
            if G.number_of_nodes() > 0:
                logging.info(f"Creating degree distribution plots for {ratio_label}")
                plot_degree_distribution(G, ratio_label, plots_dir)
            
        except Exception as e:
            error_msg = f"Statistics computation failed for {ratio_label}: {str(e)}"
            logging.error(error_msg)
            failures.append({"ratio": ratio_label, "stage": "statistics", "error": error_msg})
            # Add partial stats entry
            all_stats.append({
                'user_weight': user_w,
                'ai_weight': ai_w,
                'ratio_label': ratio_label,
                'status': 'failed',
                'error': str(e)
            })
        
        # Step 5: Export to GEXF
        try:
            gexf_file = gexf_dir / f"network_{ratio_label}.gexf"
            export_to_gexf(emb_dir, str(filtered_edges_file), str(gexf_file))
        except Exception as e:
            error_msg = f"GEXF export failed for {ratio_label}: {str(e)}"
            logging.error(error_msg)
            failures.append({"ratio": ratio_label, "stage": "gexf", "error": error_msg})
            # Continue anyway - GEXF is not critical
    
    # Create comparison DataFrame
    df_stats = pd.DataFrame(all_stats)
    df_stats = df_stats.sort_values('user_weight', ascending=False)
    
    # Save comparison results
    comparison_file = base_dir / "ablation_comparison.csv"
    df_stats.to_csv(comparison_file, index=False)
    logging.info(f"\nSaved comparison results to {comparison_file}")
    
    # Generate comprehensive comparison plots
    logging.info("\nGenerating comparison plots...")
    create_comparison_plots(all_stats, base_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    # Select key metrics for display
    key_metrics = [
        'ratio_label', 'num_nodes', 'num_edges', 'giant_component_fraction',
        'avg_degree', 'avg_clustering', 'modularity', 'num_communities'
    ]
    
    available_metrics = [m for m in key_metrics if m in df_stats.columns]
    print(df_stats[available_metrics].to_string(index=False))
    
    # Save summary JSON including failures
    summary_file = base_dir / "ablation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'weight_ratios': WEIGHT_RATIOS,
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'statistics': all_stats,
            'failures': failures,
            'success_count': len([s for s in all_stats if s.get('status') == 'success']),
            'failure_count': len(failures)
        }, f, indent=2)
    
    # Save failure report if any
    if failures:
        failure_file = base_dir / "failure_report.json"
        with open(failure_file, 'w') as f:
            json.dump(failures, f, indent=2)
        
        print("\n" + "="*80)
        print("FAILURE REPORT")
        print("="*80)
        for failure in failures:
            print(f"Ratio {failure['ratio']}: Failed at {failure['stage']} stage")
            print(f"  Error: {failure['error'][:100]}...")
    
    logging.info(f"\nAblation study complete!")
    logging.info(f"  Successful ratios: {len([s for s in all_stats if s.get('status') == 'success'])}/{len(WEIGHT_RATIOS)}")
    logging.info(f"  Failed stages: {len(failures)}")
    logging.info(f"  Results saved to {base_dir}")

if __name__ == "__main__":
    main()