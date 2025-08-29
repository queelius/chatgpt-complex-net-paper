#!/usr/bin/env python3
"""
Create a cleaner, more intuitive community evolution visualization.
Shows only key weight ratios with improved clarity.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter, defaultdict

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
    
    # Networks
    if any(term in node_name.lower() for term in ['network', 'graph', 'gephi', 'complex', 'node', 'edge']):
        topics.append('Networks')
    
    return topics if topics else ['General']

def load_community_data(base_dir, user_w, ai_w):
    """Load and analyze community structure for a specific weight ratio."""
    import networkx as nx
    try:
        import community as community_louvain
    except ImportError:
        print("Install python-louvain: pip install python-louvain")
        return None
    
    base_dir = Path(base_dir)
    edges_file = base_dir / f"edges_filtered/edges_chatgpt-json-llm-user{user_w}-ai{ai_w}_t0.9.json"
    
    if not edges_file.exists():
        print(f"Edge file not found: {edges_file}")
        return None
    
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
    if G_giant.number_of_edges() > 0:
        partition = community_louvain.best_partition(G_giant)
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        # Get topic distribution for each community
        community_topics = {}
        for comm_id, nodes in communities.items():
            all_topics = []
            for node in nodes:
                all_topics.extend(extract_conversation_topics(node))
            
            topic_counts = Counter(all_topics)
            total = sum(topic_counts.values())
            
            if total > 0:
                # Convert to proportions
                topic_props = {t: c/total for t, c in topic_counts.items()}
                community_topics[comm_id] = {
                    'size': len(nodes),
                    'topics': topic_props,
                    'dominant': max(topic_props, key=topic_props.get)
                }
        
        return {
            'partition': partition,
            'communities': community_topics,
            'modularity': community_louvain.modularity(partition, G_giant),
            'num_communities': len(communities)
        }
    
    return None

def create_cleaner_visualization(base_dir, output_dir):
    """Create a cleaner, more intuitive visualization."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Focus on 3 key configurations (like we did for other figures)
    configurations = [
        (100.0, 1.0, 'User-driven\n(100:1)'),
        (2.0, 1.0, 'Optimal\n(2:1)'),
        (1.0, 100.0, 'AI-driven\n(1:100)')
    ]
    
    # Define consistent topic colors
    topic_colors = {
        'ML/AI': '#1f77b4',        # Blue
        'Programming': '#ff7f0e',   # Orange
        'Stats/Math': '#2ca02c',    # Green
        'Philosophy': '#d62728',    # Red
        'Health': '#9467bd',        # Purple
        'Networks': '#8c564b',      # Brown
        'General': '#7f7f7f'        # Gray
    }
    
    fig = plt.figure(figsize=(14, 6))
    
    # Create grid with custom spacing
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.4, wspace=0.3)
    
    all_topics = set()
    data_by_config = []
    
    for idx, (user_w, ai_w, label) in enumerate(configurations):
        print(f"Processing {label}...")
        
        # Load data
        result = load_community_data(base_dir, user_w, ai_w)
        if not result:
            continue
        
        data_by_config.append((label, result))
        
        # Collect all topics
        for comm_data in result['communities'].values():
            all_topics.update(comm_data['topics'].keys())
        
        # Main panel: Stacked bar chart showing topic composition
        ax = fig.add_subplot(gs[0, idx])
        
        # Sort communities by size (largest first)
        sorted_comms = sorted(result['communities'].items(), 
                            key=lambda x: x[1]['size'], 
                            reverse=True)[:8]  # Top 8 communities
        
        # Prepare data for stacked bars
        community_labels = []
        topic_data = {topic: [] for topic in topic_colors.keys()}
        
        for comm_id, comm_data in sorted_comms:
            community_labels.append(f"C{comm_id}\n({comm_data['size']})")
            
            for topic in topic_colors.keys():
                prop = comm_data['topics'].get(topic, 0)
                topic_data[topic].append(prop)
        
        # Create stacked bars
        x = np.arange(len(community_labels))
        bottom = np.zeros(len(community_labels))
        
        for topic, color in topic_colors.items():
            if topic in topic_data and any(topic_data[topic]):
                ax.bar(x, topic_data[topic], bottom=bottom, 
                      color=color, label=topic, width=0.8)
                bottom += topic_data[topic]
        
        ax.set_ylabel('Topic Proportion', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(community_labels, fontsize=8)
        ax.set_ylim(0, 1)
        
        # Add modularity score
        ax.text(0.95, 0.95, f"Mod: {result['modularity']:.3f}", 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
        
        # Summary metrics panel below
        ax_summary = fig.add_subplot(gs[1, idx])
        ax_summary.axis('off')
        
        # Calculate summary statistics
        topic_diversity = np.mean([len(c['topics']) for c in result['communities'].values()])
        topic_purity = np.mean([max(c['topics'].values()) for c in result['communities'].values()])
        
        summary_text = (
            f"Communities: {result['num_communities']}\n"
            f"Avg. topics/comm: {topic_diversity:.1f}\n"
            f"Topic purity: {topic_purity:.1%}"
        )
        
        ax_summary.text(0.5, 0.5, summary_text, 
                       transform=ax_summary.transAxes,
                       ha='center', va='center',
                       fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Add shared legend
    handles = [mpatches.Patch(color=color, label=topic) 
              for topic, color in topic_colors.items() if topic in all_topics]
    fig.legend(handles=handles, loc='upper right', 
              bbox_to_anchor=(0.98, 0.98), ncol=1, fontsize=9)
    
    # Main title
    fig.suptitle('Knowledge Community Organization Across Weight Ratios\n(Similarity Threshold = 0.9)',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figures
    output_file = output_dir / 'community_evolution_clean'
    plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_file}.pdf", bbox_inches='tight')
    plt.close()
    
    # Create a second, even simpler visualization showing the key transition
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, ((label, result), ax) in enumerate(zip(data_by_config, axes)):
        # Create a pie chart showing overall topic distribution
        all_topics_combined = Counter()
        total_nodes = 0
        
        for comm_data in result['communities'].values():
            for topic, prop in comm_data['topics'].items():
                all_topics_combined[topic] += prop * comm_data['size']
            total_nodes += comm_data['size']
        
        # Normalize
        for topic in all_topics_combined:
            all_topics_combined[topic] /= total_nodes
        
        # Sort topics by proportion
        sorted_topics = sorted(all_topics_combined.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare data for pie chart
        labels = []
        sizes = []
        colors = []
        
        for topic, prop in sorted_topics:
            if prop > 0.05:  # Only show topics > 5%
                labels.append(f"{topic}\n{prop:.0%}")
                sizes.append(prop)
                colors.append(topic_colors.get(topic, '#7f7f7f'))
            else:
                # Group small topics
                if not labels or labels[-1] != 'Other':
                    labels.append('Other')
                    sizes.append(prop)
                    colors.append('#e0e0e0')
                else:
                    sizes[-1] += prop
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct=lambda p: f'{p:.0f}%' if p > 5 else '',
                                          startangle=90)
        
        # Adjust text size
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        ax.set_title(label.replace('\n', ' '), fontsize=11, fontweight='bold', pad=20)
        
        # Add metrics below
        metrics = f"Modularity: {result['modularity']:.3f}\nCommunities: {result['num_communities']}"
        ax.text(0, -1.3, metrics, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Overall Topic Distribution Shift', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    output_file2 = output_dir / 'topic_distribution_shift'
    plt.savefig(f"{output_file2}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_file2}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"\nCleaner visualizations saved to {output_dir}")
    print("- community_evolution_clean.pdf: Community-level topic composition")
    print("- topic_distribution_shift.pdf: Overall topic distribution changes")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", 
                       default="../dev/ablation_study/data",
                       help="Directory with edges_filtered")
    parser.add_argument("--output-dir",
                       default="../comp-net-2025-paper/images",
                       help="Output directory")
    args = parser.parse_args()
    
    create_cleaner_visualization(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()