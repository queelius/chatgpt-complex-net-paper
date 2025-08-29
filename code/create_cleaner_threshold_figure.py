#!/usr/bin/env python3
"""
Create a cleaner, more focused version of the threshold evolution figure.
Focus on the key metrics that tell the story of the phase transition.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def create_focused_threshold_figure(csv_path, output_dir):
    """Create a cleaner figure focusing on key metrics that show the phase transition."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure with just the essential metrics
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Define weight ratios to show (fewer lines for clarity)
    key_ratios = [
        (100.0, 1.0, 'User only', '#d62728'),     # Red
        (2.0, 1.0, 'Optimal (2:1)', '#2ca02c'),   # Green - our chosen
        (1.0, 1.0, 'Balanced', '#1f77b4'),        # Blue
        (1.0, 100.0, 'AI only', '#ff7f0e')        # Orange
    ]
    
    thresholds = sorted(df['threshold'].unique())
    
    # Panel 1: Giant Component Fraction (shows catastrophic fragmentation)
    ax = axes[0]
    for user_w, ai_w, label, color in key_ratios:
        df_ratio = df[(df['user_weight'] == user_w) & (df['ai_weight'] == ai_w)]
        if not df_ratio.empty:
            values = []
            for t in thresholds:
                df_t = df_ratio[df_ratio['threshold'] == t]
                if not df_t.empty:
                    values.append(df_t.iloc[0]['giant_component_fraction'])
                else:
                    values.append(0)
            
            # Make the optimal line thicker
            linewidth = 3 if user_w == 2.0 and ai_w == 1.0 else 2
            alpha = 1.0 if user_w == 2.0 and ai_w == 1.0 else 0.7
            
            ax.plot(thresholds, values, 'o-', label=label, color=color, 
                   linewidth=linewidth, markersize=6, alpha=alpha)
    
    # Add vertical line at critical threshold
    ax.axvline(x=0.875, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.875, 0.5, 'Critical\nThreshold', ha='center', va='center', 
            fontsize=10, color='gray', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_xlabel('Similarity Threshold', fontsize=11)
    ax.set_ylabel('Giant Component Fraction', fontsize=11)
    ax.set_title('Network Connectivity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    
    # Panel 2: Modularity (shows community structure quality)
    ax = axes[1]
    for user_w, ai_w, label, color in key_ratios:
        df_ratio = df[(df['user_weight'] == user_w) & (df['ai_weight'] == ai_w)]
        if not df_ratio.empty:
            values = []
            for t in thresholds:
                df_t = df_ratio[df_ratio['threshold'] == t]
                if not df_t.empty:
                    values.append(df_t.iloc[0]['modularity'])
                else:
                    values.append(0)
            
            linewidth = 3 if user_w == 2.0 and ai_w == 1.0 else 2
            alpha = 1.0 if user_w == 2.0 and ai_w == 1.0 else 0.7
            
            ax.plot(thresholds, values, 'o-', label=label, color=color,
                   linewidth=linewidth, markersize=6, alpha=alpha)
    
    # Highlight optimal point
    ax.plot(0.9, 0.750, 'o', color='#2ca02c', markersize=12, 
           markeredgecolor='darkgreen', markeredgewidth=2, zorder=5)
    ax.annotate('Maximum\n(0.750)', xy=(0.9, 0.750), xytext=(0.92, 0.72),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
                fontsize=9, color='darkgreen', fontweight='bold')
    
    ax.axvline(x=0.875, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Similarity Threshold', fontsize=11)
    ax.set_ylabel('Modularity', fontsize=11)
    ax.set_title('Community Quality', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 0.85)
    
    # Panel 3: Number of Edges (shows sparsification)
    ax = axes[2]
    for user_w, ai_w, label, color in key_ratios:
        df_ratio = df[(df['user_weight'] == user_w) & (df['ai_weight'] == ai_w)]
        if not df_ratio.empty:
            values = []
            for t in thresholds:
                df_t = df_ratio[df_ratio['threshold'] == t]
                if not df_t.empty:
                    values.append(df_t.iloc[0]['num_edges'])
                else:
                    values.append(0)
            
            linewidth = 3 if user_w == 2.0 and ai_w == 1.0 else 2
            alpha = 1.0 if user_w == 2.0 and ai_w == 1.0 else 0.7
            
            ax.plot(thresholds, values, 'o-', label=label, color=color,
                   linewidth=linewidth, markersize=6, alpha=alpha)
    
    ax.axvline(x=0.875, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Similarity Threshold', fontsize=11)
    ax.set_ylabel('Number of Edges', fontsize=11)
    ax.set_title('Network Density', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Main title
    fig.suptitle('Phase Transition at θ ≈ 0.875: Universal Across All Weight Ratios',
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Add interpretation text
    fig.text(0.5, -0.05, 
             'All weight ratios experience catastrophic fragmentation at the same critical threshold, '
             'confirming a fundamental boundary in semantic space',
             ha='center', fontsize=10, style='italic', wrap=True)
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig(output_dir / 'threshold_evolution_clean.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'threshold_evolution_clean.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved cleaner threshold evolution figure to {output_dir}")
    
    # Create an even simpler version with just giant component
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    for user_w, ai_w, label, color in key_ratios:
        df_ratio = df[(df['user_weight'] == user_w) & (df['ai_weight'] == ai_w)]
        if not df_ratio.empty:
            values = []
            for t in thresholds:
                df_t = df_ratio[df_ratio['threshold'] == t]
                if not df_t.empty:
                    values.append(df_t.iloc[0]['giant_component_fraction'])
                else:
                    values.append(0)
            
            linewidth = 3 if user_w == 2.0 and ai_w == 1.0 else 2
            alpha = 1.0 if user_w == 2.0 and ai_w == 1.0 else 0.8
            
            ax.plot(thresholds, values, 'o-', label=label, color=color,
                   linewidth=linewidth, markersize=7, alpha=alpha)
    
    # Shade the phase transition region
    ax.axvspan(0.875, 0.9, alpha=0.2, color='red', label='Phase transition')
    ax.axvline(x=0.875, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add text annotations
    ax.text(0.865, 0.95, 'Connected\nNetwork', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.91, 0.3, 'Fragmented\nNetwork', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.9, 0.85, 'Our choice:\nθ = 0.9', ha='center', fontsize=10, 
            color='green', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                     edgecolor='green', alpha=0.9))
    
    ax.set_xlabel('Similarity Threshold (θ)', fontsize=12)
    ax.set_ylabel('Fraction of Nodes in Giant Component', fontsize=12)
    ax.set_title('Percolation Phase Transition in Semantic Networks', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    ax.set_xlim(0.795, 0.955)
    ax.set_ylim(-0.05, 1.05)
    
    # Add arrow showing the "cliff"
    ax.annotate('', xy=(0.9, 0.2), xytext=(0.875, 0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.885, 0.55, 'Catastrophic\nfragmentation', ha='center', 
            rotation=-70, fontsize=10, color='red')
    
    plt.tight_layout()
    
    # Save simple version
    plt.savefig(output_dir / 'threshold_phase_transition_simple.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'threshold_phase_transition_simple.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved simple phase transition figure to {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create cleaner threshold evolution figures")
    parser.add_argument("--csv", 
                       default="../dev/ablation_study/analysis_2d/ablation_results.csv",
                       help="Path to CSV with ablation results")
    parser.add_argument("--output-dir",
                       default="../comp-net-2025-paper/images",
                       help="Output directory for figures")
    args = parser.parse_args()
    
    create_focused_threshold_figure(args.csv, args.output_dir)

if __name__ == "__main__":
    main()