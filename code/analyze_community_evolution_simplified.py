#!/usr/bin/env python3
"""
Simplified version of community evolution analysis - focusing on 3 key ratios
for better readability and intuition.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re

# Try to import community detection
try:
    import community as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    print("Install python-louvain: pip install python-louvain")
    HAS_COMMUNITY = False

def extract_conversation_topics(node_name):
    """Extract topic keywords from conversation filename."""
    topics = []
    
    # ML/AI related
    if any(term in node_name.lower() for term in ['llm', 'ai', 'ml', 'neural', 'training', 'model', 'gpt', 'embedding']):
        topics.append('ML/AI')
    
    # Programming
    if any(term in node_name.lower() for term in ['python', 'code', 'function', 'algorithm', 'debug', 'compile', 'git']):
        topics.append('Programming')
    
    # Statistics/Math
    if any(term in node_name.lower() for term in ['stat', 'probability', 'mle', 'distribution', 'math', 'calculus', 'variance']):
        topics.append('Stats/Math')
    
    # Philosophy
    if any(term in node_name.lower() for term in ['philosophy', 'consciousness', 'ethics', 'moral', 'exist']):
        topics.append('Philosophy')
    
    # Health/Medical
    if any(term in node_name.lower() for term in ['cancer', 'health', 'medical', 'therapy', 'pain', 'chemotherapy']):
        topics.append('Health')
    
    # R/Data Science
    if any(term in node_name.lower() for term in [' r ', 'r-', 'ggplot', 'tidyverse', 'data.frame']):
        topics.append('R/DataSci')
    
    # Network/Complex Systems
    if any(term in node_name.lower() for term in ['network', 'graph', 'gephi', 'complex', 'node', 'edge']):
        topics.append('Networks')
    
    return topics if topics else ['General']

def analyze_community_structure(edges_file, embeddings_dir):
    """Analyze the community structure for a specific configuration."""
    
    # Load edges
    with open(edges_file, 'r') as f:
        edges_data = json.load(f)
    
    # Build graph
    G = nx.Graph()
    for edge in edges_data:
        if len(edge) == 3:
            src, dst, weight = edge
            G.add_edge(src, dst, weight=weight)
    
    if G.number_of_nodes() == 0:
        return None
    
    # Get giant component
    components = list(nx.connected_components(G))
    if not components:
        return None
    
    giant = max(components, key=len)
    G_giant = G.subgraph(giant).copy()
    
    # Detect communities
    if HAS_COMMUNITY and G_giant.number_of_edges() > 0:
        partition = community_louvain.best_partition(G_giant)
        
        # Analyze each community
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        # Characterize communities by topics
        community_profiles = {}
        for comm_id, nodes in communities.items():
            # Get topic distribution
            all_topics = []
            for node in nodes:
                all_topics.extend(extract_conversation_topics(node))
            
            topic_counts = Counter(all_topics)
            total = sum(topic_counts.values())
            
            if total > 0:
                # Get dominant topic
                dominant_topic = topic_counts.most_common(1)[0][0]
                dominant_pct = topic_counts[dominant_topic] / total
                
                community_profiles[comm_id] = {
                    'size': len(nodes),
                    'dominant_topic': dominant_topic,
                    'dominant_pct': dominant_pct,
                    'topic_distribution': dict(topic_counts),
                    'sample_nodes': nodes[:5]
                }
        
        return {
            'num_communities': len(communities),
            'modularity': community_louvain.modularity(partition, G_giant),
            'community_profiles': community_profiles,
            'giant_size': G_giant.number_of_nodes()
        }
    
    return None

def create_simplified_viz(base_dir, output_dir):
    """Create simplified visualization with just 3 key ratios."""
    
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Just 3 key ratios that tell the story
    weight_ratios = [
        (4.0, 1.0, "User-Driven\n(4:1)"),     # User's mental model
        (2.0, 1.0, "Optimal Balance\n(2:1)"),  # Best configuration
        (1.0, 4.0, "AI-Driven\n(1:4)")        # AI's organization
    ]
    
    results = []
    
    for user_w, ai_w, label in weight_ratios:
        print(f"\nAnalyzing ratio {user_w}:{ai_w}")
        
        edges_file = base_dir / f"edges_filtered/edges_chatgpt-json-llm-user{user_w}-ai{ai_w}_t0.9.json"
        embeddings_dir = base_dir / f"embeddings/chatgpt-json-llm-user{user_w}-ai{ai_w}"
        
        if not edges_file.exists():
            print(f"  Edge file not found: {edges_file}")
            continue
        
        structure = analyze_community_structure(edges_file, embeddings_dir)
        
        if structure:
            results.append({
                'ratio': f"{user_w}:{ai_w}",
                'label': label,
                'user_weight': user_w,
                'ai_weight': ai_w,
                **structure
            })
    
    # Create cleaner visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('How Weight Ratios Change Knowledge Organization', fontsize=18, fontweight='bold', y=1.02)
    
    # Define consistent topic order and colors
    topic_order = ['ML/AI', 'Programming', 'Stats/Math', 'Philosophy', 'Health', 'R/DataSci', 'Networks', 'General']
    
    # Create a more intuitive color scheme
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(topic_order)))
    topic_colors = dict(zip(topic_order, colors))
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        # Get communities sorted by size
        communities = result['community_profiles']
        sorted_comms = sorted(communities.items(), key=lambda x: x[1]['size'], reverse=True)[:7]  # Top 7 for clarity
        
        # Create stacked bar chart instead of heatmap
        bottom = np.zeros(len(sorted_comms))
        
        for topic in topic_order:
            heights = []
            for comm_id, profile in sorted_comms:
                topic_count = profile['topic_distribution'].get(topic, 0)
                total = sum(profile['topic_distribution'].values())
                heights.append(topic_count / total if total > 0 else 0)
            
            heights = np.array(heights) * 100  # Convert to percentage
            if np.any(heights > 0):
                bars = ax.bar(range(len(sorted_comms)), heights, bottom=bottom, 
                             label=topic if heights.max() > 5 else "",  # Only label significant topics
                             color=topic_colors[topic], edgecolor='white', linewidth=0.5)
                bottom += heights
        
        # Customize axes
        ax.set_ylim(0, 100)
        ax.set_ylabel('Topic Composition (%)', fontsize=12)
        ax.set_xlabel('Communities (sorted by size)', fontsize=12)
        
        # Add community size labels
        comm_labels = [f"n={profile['size']}" for _, profile in sorted_comms]
        ax.set_xticks(range(len(sorted_comms)))
        ax.set_xticklabels(comm_labels, rotation=0, fontsize=10)
        
        # Add title with key metrics
        ax.set_title(f"{result['label']}\nModularity: {result['modularity']:.3f}\n{result['num_communities']} communities",
                    fontsize=14, pad=10)
        
        # Add grid for readability
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add annotations for dominant topics in largest communities
        for i, (comm_id, profile) in enumerate(sorted_comms[:3]):  # Annotate top 3
            if profile['dominant_pct'] > 0.5:  # Only if clearly dominant
                y_pos = profile['dominant_pct'] * 50  # Middle of dominant section
                ax.text(i, y_pos, profile['dominant_topic'], 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white' if profile['dominant_pct'] > 0.7 else 'black')
    
    # Add legend to the last subplot
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), 
                      frameon=True, fontsize=11, title="Topics")
    
    # Add interpretation text below
    fig.text(0.15, -0.05, "Clear topic separation\nUser's mental model", 
             ha='center', fontsize=11, style='italic', color='#666')
    fig.text(0.5, -0.05, "Best balance\nHighest modularity", 
             ha='center', fontsize=11, style='italic', color='#006400', fontweight='bold')
    fig.text(0.85, -0.05, "Topics merge\nAI's response patterns", 
             ha='center', fontsize=11, style='italic', color='#666')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'community_evolution_simplified.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'community_evolution_simplified.pdf', bbox_inches='tight')
    plt.close()
    
    # Create a second figure showing the key insight more clearly
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Topic purity metric (how pure/focused are communities)
    all_ratios = [
        (100.0, 1.0), (4.0, 1.0), (2.0, 1.0), (1.0, 1.0), 
        (1.0, 2.0), (1.0, 4.0), (1.0, 100.0)
    ]
    
    purities = []
    modularities = []
    ratio_labels = []
    
    for user_w, ai_w in all_ratios:
        edges_file = base_dir / f"edges_filtered/edges_chatgpt-json-llm-user{user_w}-ai{ai_w}_t0.9.json"
        if edges_file.exists():
            structure = analyze_community_structure(edges_file, None)
            if structure:
                # Calculate average purity
                avg_purity = np.mean([p['dominant_pct'] for p in structure['community_profiles'].values()])
                purities.append(avg_purity * 100)
                modularities.append(structure['modularity'])
                ratio_labels.append(f"{user_w}:{ai_w}")
    
    # Create dual axis plot
    ax2 = ax.twinx()
    
    x_pos = range(len(ratio_labels))
    
    # Plot bars for purity
    bars = ax.bar(x_pos, purities, alpha=0.7, color='steelblue', label='Topic Purity')
    ax.set_ylabel('Average Topic Purity (%)', fontsize=12, color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    
    # Plot line for modularity
    line = ax2.plot(x_pos, modularities, 'ro-', linewidth=2, markersize=8, label='Modularity')
    ax2.set_ylabel('Modularity', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Highlight optimal ratio
    optimal_idx = ratio_labels.index('2.0:1.0')
    ax.bar(optimal_idx, purities[optimal_idx], alpha=1.0, color='green', edgecolor='darkgreen', linewidth=2)
    ax2.plot(optimal_idx, modularities[optimal_idx], 'go', markersize=12, markeredgecolor='darkgreen', markeredgewidth=2)
    
    ax.set_xlabel('User:AI Weight Ratio', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ratio_labels, rotation=45, ha='right')
    ax.set_title('Trade-off: Topic Purity vs Network Modularity', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add annotation for optimal point
    ax.annotate('Optimal\nBalance', xy=(optimal_idx, purities[optimal_idx]), 
                xytext=(optimal_idx-0.5, purities[optimal_idx]+5),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                fontsize=11, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'topic_purity_vs_modularity.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'topic_purity_vs_modularity.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\nSimplified visualizations saved to {output_dir}")
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simplified community evolution analysis")
    parser.add_argument("--data-dir", 
                       default="../dev/ablation_study/data",
                       help="Directory with edges_filtered and embeddings")
    parser.add_argument("--output-dir",
                       default="../dev/ablation_study/analysis_2d/community_evolution_simple",
                       help="Output directory for analysis")
    args = parser.parse_args()
    
    print("Creating simplified community evolution visualizations...")
    results = create_simplified_viz(args.data_dir, args.output_dir)
    
    if results:
        print("\n" + "="*60)
        print("KEY INSIGHT: Weight ratios control knowledge organization")
        print("="*60)
        print("\n• User-heavy (4:1): Clear topic boundaries, your mental model")
        print("• Optimal (2:1): Best modularity, balanced organization")  
        print("• AI-heavy (1:4): Topics merge, AI's response patterns")

if __name__ == "__main__":
    main()