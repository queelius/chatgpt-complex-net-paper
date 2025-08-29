#!/usr/bin/env python3
"""
Create the simplest possible community evolution visualization.
Shows the key insight with minimal visual complexity.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
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
        
        # Calculate overall topic distribution
        all_topics = []
        for node in G_giant.nodes():
            all_topics.extend(extract_conversation_topics(node))
        
        topic_counts = Counter(all_topics)
        total = sum(topic_counts.values())
        
        # Calculate topic purity for each community
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        purities = []
        for comm_id, nodes in communities.items():
            comm_topics = []
            for node in nodes:
                comm_topics.extend(extract_conversation_topics(node))
            
            if comm_topics:
                comm_counter = Counter(comm_topics)
                max_count = max(comm_counter.values())
                total_count = sum(comm_counter.values())
                purity = max_count / total_count
                purities.append(purity)
        
        return {
            'modularity': community_louvain.modularity(partition, G_giant),
            'num_communities': len(communities),
            'topic_distribution': {t: c/total for t, c in topic_counts.items()},
            'avg_purity': np.mean(purities) if purities else 0,
            'giant_size': G_giant.number_of_nodes()
        }
    
    return None

def create_simplest_visualization(base_dir, output_dir):
    """Create the simplest possible visualization showing the key insight."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Focus on 3 key configurations
    configurations = [
        (100.0, 1.0, 'User-driven', '#d62728'),     # Red
        (2.0, 1.0, 'Optimal (2:1)', '#2ca02c'),     # Green
        (1.0, 100.0, 'AI-driven', '#ff7f0e')        # Orange
    ]
    
    # Topic colors for pie charts
    topic_colors = {
        'ML/AI': '#1f77b4',
        'Programming': '#ff7f0e', 
        'Stats/Math': '#2ca02c',
        'Philosophy': '#d62728',
        'Health': '#9467bd',
        'Networks': '#8c564b',
        'General': '#7f7f7f',
        'Other': '#e0e0e0'  # For small grouped topics
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    metrics_data = []
    
    for idx, (user_w, ai_w, label, color) in enumerate(configurations):
        ax = axes[idx]
        
        # Load data
        result = load_community_data(base_dir, user_w, ai_w)
        if not result:
            continue
        
        metrics_data.append({
            'label': label,
            'modularity': result['modularity'],
            'communities': result['num_communities'],
            'purity': result['avg_purity']
        })
        
        # Create pie chart of topic distribution
        topics = result['topic_distribution']
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        
        # Group small topics
        pie_data = []
        pie_labels = []
        pie_colors = []
        other_total = 0
        has_general = False
        
        for topic, prop in sorted_topics:
            if prop > 0.08:  # Show topics > 8%
                pie_data.append(prop)
                pie_labels.append(topic)
                pie_colors.append(topic_colors.get(topic, '#7f7f7f'))
                if topic == 'General':
                    has_general = True
            else:
                other_total += prop
        
        # Add grouped small topics, but combine with General if it exists
        if other_total > 0:
            if has_general:
                # Find General and add to it
                for i, label in enumerate(pie_labels):
                    if label == 'General':
                        pie_data[i] += other_total
                        break
            else:
                # Add as 'Other' if no General category exists
                pie_data.append(other_total)
                pie_labels.append('Other')
                pie_colors.append('#e0e0e0')
        
        # Draw pie
        wedges, texts, autotexts = ax.pie(pie_data, labels=pie_labels, colors=pie_colors,
                                          autopct=lambda p: f'{p:.0f}%' if p > 10 else '',
                                          startangle=90)
        
        # Style adjustments
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        # Title with key metrics
        ax.set_title(f"{label}\nMod: {result['modularity']:.3f} | {result['num_communities']} communities",
                    fontsize=10, fontweight='bold', pad=15)
        
        # Add topic purity indicator
        purity_color = plt.cm.RdYlGn(result['avg_purity'])
        ax.add_patch(plt.Circle((0, -1.4), 0.1, color=purity_color, transform=ax.transData))
        ax.text(0, -1.6, f"Purity: {result['avg_purity']:.0%}", 
               ha='center', fontsize=8, transform=ax.transData)
    
    # Overall title
    fig.suptitle('Knowledge Organization Transition: User → Optimal → AI',
                fontsize=12, fontweight='bold', y=1.05)
    
    # Add explanation text
    fig.text(0.5, -0.05, 
            'Topic distribution shifts from distinct domains (user-driven) to merged categories (AI-driven).\n'
            'Optimal 2:1 ratio balances semantic coherence with knowledge specificity.',
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'community_simplest'
    plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_file}.pdf", bbox_inches='tight')
    plt.close()
    
    # Create a single summary metric plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if metrics_data:
        x = np.arange(len(metrics_data))
        width = 0.25
        
        # Modularity bars
        mod_bars = ax.bar(x - width, [m['modularity'] for m in metrics_data], 
                         width, label='Modularity', color='#2ca02c', alpha=0.8)
        
        # Purity bars
        purity_bars = ax.bar(x, [m['purity'] for m in metrics_data],
                           width, label='Topic Purity', color='#1f77b4', alpha=0.8)
        
        # Communities (normalized)
        max_comm = max(m['communities'] for m in metrics_data)
        comm_bars = ax.bar(x + width, [m['communities']/max_comm for m in metrics_data],
                          width, label=f'Communities (/{max_comm})', color='#ff7f0e', alpha=0.8)
        
        # Add value labels
        for bars in [mod_bars, purity_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Add community count labels
        for i, bar in enumerate(comm_bars):
            actual_count = metrics_data[i]['communities']
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   str(actual_count), ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Weight Ratio Configuration', fontsize=10)
        ax.set_ylabel('Metric Value', fontsize=10)
        ax.set_title('Key Metrics Across Weight Ratios', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m['label'] for m in metrics_data])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        # Add insight annotation
        ax.annotate('Optimal balance', xy=(1, metrics_data[1]['modularity']), 
                   xytext=(1.5, 0.6), fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
    
    plt.tight_layout()
    
    output_file2 = output_dir / 'community_metrics_comparison'
    plt.savefig(f"{output_file2}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_file2}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"\nSimplest visualizations saved to {output_dir}")
    print("- community_simplest.pdf: Minimal topic distribution view")
    print("- community_metrics_comparison.pdf: Key metrics bar chart")

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
    
    create_simplest_visualization(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()