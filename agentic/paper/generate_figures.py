#!/usr/bin/env python3
"""Generate all 8 figures for the agentic cognitive MRI paper.

Reads pre-computed analysis results from data/analysis-001/ and edge data from
data/edges-t0.9.json. Builds semantic and delegation graphs from the database
to compute per-node metrics for figures that need them.

Outputs PDFs to paper/figures/.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np

# Allow importing the agentic package
AGENTIC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENTIC_ROOT.parent))  # parent of agentic/ so `import agentic` works

from agentic.extract import extract_sessions
from agentic.delegation import (
    build_delegation_graph,
    compute_fan_out_distribution,
    compute_delegation_ratio,
)
from agentic.multilayer import (
    compute_interlayer_degree_correlation,
    compute_participation_coefficient,
)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = AGENTIC_ROOT / "data" / "analysis-001"
EDGES_FILE = AGENTIC_ROOT / "data" / "edges-t0.9.json"
DB_PATH = Path.home() / ".memex" / "default" / "conversations.db"
FIG_DIR = AGENTIC_ROOT / "paper" / "figures"

# ChatGPT comparison data from the parent repo
PARENT_REPO = AGENTIC_ROOT.parent
CHATGPT_EDGES_FILE = PARENT_REPO / "data" / "network" / "edges_user2.0-ai1.0_t0.9.json"
CHATGPT_DENSIFICATION_FILE = PARENT_REPO / "data" / "temporal" / "densification.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,       # TrueType fonts in PDF
    "ps.fonttype": 42,
})

COLOR_AGENTIC = "#1f77b4"   # blue
COLOR_CHATGPT = "#ff7f0e"   # orange


# ── Load pre-computed JSON data ────────────────────────────────────────────
def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


semantic_data = load_json(DATA_DIR / "semantic.json")
delegation_data = load_json(DATA_DIR / "delegation.json")
temporal_data = load_json(DATA_DIR / "temporal.json")
multilayer_data = load_json(DATA_DIR / "multilayer.json")
null_data = load_json(DATA_DIR / "null_models.json")


# ── Build graphs from raw data ────────────────────────────────────────────
print("Loading edges...")
with open(EDGES_FILE) as f:
    raw_edges = json.load(f)
print(f"  {len(raw_edges)} edges loaded")

print("Extracting sessions from database...")
sessions = extract_sessions(str(DB_PATH), include_subagents=True)
session_ids = {s.id for s in sessions}
print(f"  {len(sessions)} sessions extracted")

# Translate filename-safe IDs back to session IDs
safe_to_id = {sid.replace(":", "_"): sid for sid in session_ids}
edges = [
    (safe_to_id.get(s, s), safe_to_id.get(t, t), w)
    for s, t, w in raw_edges
]

# Build semantic graph
print("Building semantic graph...")
sem_G = nx.Graph()
for s, t, w in edges:
    if s in session_ids and t in session_ids:
        sem_G.add_edge(s, t, weight=w)
for sid in session_ids:
    if sid not in sem_G:
        sem_G.add_node(sid)
print(f"  {sem_G.number_of_nodes()} nodes, {sem_G.number_of_edges()} edges")

# Build delegation graph
print("Building delegation graph...")
deleg_G = build_delegation_graph(sessions)
print(f"  {deleg_G.number_of_nodes()} nodes, {deleg_G.number_of_edges()} edges")

# Load ChatGPT comparison data if available
chatgpt_G = None
chatgpt_densification = None
if CHATGPT_EDGES_FILE.exists():
    print("Loading ChatGPT comparison edges...")
    with open(CHATGPT_EDGES_FILE) as f:
        chatgpt_edges = json.load(f)
    chatgpt_G = nx.Graph()
    for s, t, w in chatgpt_edges:
        chatgpt_G.add_edge(s, t, weight=w)
    print(f"  ChatGPT: {chatgpt_G.number_of_nodes()} nodes, {chatgpt_G.number_of_edges()} edges")

if CHATGPT_DENSIFICATION_FILE.exists():
    print("Loading ChatGPT densification data...")
    chatgpt_densification = load_json(CHATGPT_DENSIFICATION_FILE)


# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Degree distribution (log-log)
# ══════════════════════════════════════════════════════════════════════════
def fig1_degree_distribution():
    print("Generating Figure 1: Degree distribution...")
    fig, ax = plt.subplots(figsize=(5, 4))

    # Agentic degree distribution
    degrees_a = [d for _, d in sem_G.degree()]
    deg_count_a = Counter(degrees_a)
    ks_a = sorted(deg_count_a.keys())
    # Filter out degree 0 for log-log
    ks_a = [k for k in ks_a if k > 0]
    pk_a = [deg_count_a[k] / len(degrees_a) for k in ks_a]

    ax.scatter(ks_a, pk_a, s=15, alpha=0.7, color=COLOR_AGENTIC,
               label=f"Agentic ($n={sem_G.number_of_nodes()}$)", zorder=3)

    # ChatGPT comparison
    if chatgpt_G is not None:
        degrees_c = [d for _, d in chatgpt_G.degree()]
        deg_count_c = Counter(degrees_c)
        ks_c = sorted(k for k in deg_count_c.keys() if k > 0)
        pk_c = [deg_count_c[k] / len(degrees_c) for k in ks_c]
        ax.scatter(ks_c, pk_c, s=15, alpha=0.7, color=COLOR_CHATGPT,
                   label=f"ChatGPT ($n={chatgpt_G.number_of_nodes()}$)", zorder=2,
                   marker="^")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("$P(k)$")
    ax.set_title("Degree Distribution")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.8")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_degree_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_degree_distribution.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Densification comparison (ChatGPT vs Agentic)
# ══════════════════════════════════════════════════════════════════════════
def fig2_densification_comparison():
    print("Generating Figure 2: Densification comparison...")
    fig, ax = plt.subplots(figsize=(5, 4))

    snapshots = temporal_data["snapshots"]
    # Filter snapshots with > 0 edges for log
    valid_snaps = [s for s in snapshots if s["node_count"] > 1 and s["edge_count"] > 0]

    log_n_a = [np.log10(s["node_count"]) for s in valid_snaps]
    log_e_a = [np.log10(s["edge_count"]) for s in valid_snaps]

    gamma_a = temporal_data["densification"]["gamma"]
    intercept_a = temporal_data["densification"]["intercept"]
    r2_a = temporal_data["densification"]["r_squared"]

    ax.scatter(log_n_a, log_e_a, s=25, color=COLOR_AGENTIC, alpha=0.8, zorder=3,
               label=f"Agentic ($\\gamma={gamma_a:.2f}$)")

    # Fit line for agentic
    x_fit = np.linspace(min(log_n_a), max(log_n_a), 100)
    y_fit = gamma_a * x_fit + intercept_a
    ax.plot(x_fit, y_fit, color=COLOR_AGENTIC, linestyle="--", linewidth=1.5, alpha=0.7)

    # ChatGPT comparison
    if chatgpt_densification is not None:
        nodes_c = chatgpt_densification["nodes"]
        edges_c = chatgpt_densification["edges"]
        # Filter out zeros
        valid_c = [(n, e) for n, e in zip(nodes_c, edges_c) if n > 1 and e > 0]
        log_n_c = [np.log10(n) for n, e in valid_c]
        log_e_c = [np.log10(e) for n, e in valid_c]

        gamma_c = chatgpt_densification["alpha"]
        intercept_c = chatgpt_densification["intercept"]

        ax.scatter(log_n_c, log_e_c, s=25, color=COLOR_CHATGPT, alpha=0.8, zorder=2,
                   marker="^", label=f"ChatGPT ($\\gamma={gamma_c:.2f}$)")

        x_fit_c = np.linspace(min(log_n_c), max(log_n_c), 100)
        y_fit_c = gamma_c * x_fit_c + intercept_c
        ax.plot(x_fit_c, y_fit_c, color=COLOR_CHATGPT, linestyle="--", linewidth=1.5,
                alpha=0.7)

    ax.set_xlabel("$\\log_{10}(n)$  [nodes]")
    ax.set_ylabel("$\\log_{10}(e)$  [edges]")
    ax.set_title("Densification Law Comparison")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.8")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_densification_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_densification_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Fan-out distribution (log-scale histogram)
# ══════════════════════════════════════════════════════════════════════════
def fig3_fanout_distribution():
    print("Generating Figure 3: Fan-out distribution...")
    fig, ax = plt.subplots(figsize=(5, 4))

    fan_out = compute_fan_out_distribution(deleg_G)
    values = list(fan_out.values())

    # Use log-spaced bins for the heavy tail
    max_val = max(values) if values else 1
    bins = np.logspace(0, np.log10(max_val + 1), 30)

    ax.hist(values, bins=bins, color=COLOR_AGENTIC, alpha=0.8, edgecolor="white",
            linewidth=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fan-out (children per parent)")
    ax.set_ylabel("Count")
    ax.set_title("Fan-out Distribution")

    # Annotate summary statistics
    mean_fo = np.mean(values) if values else 0
    max_fo = max(values) if values else 0
    ax.text(0.97, 0.97,
            f"$n = {len(values)}$ parents\n"
            f"mean $= {mean_fo:.1f}$\n"
            f"max $= {max_fo}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="0.8", alpha=0.9))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_fanout_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_fanout_distribution.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Figure 4: Delegation ratio vs fan-out scatter plot
# ══════════════════════════════════════════════════════════════════════════
def fig4_delegation_ratio_fanout():
    print("Generating Figure 4: Delegation ratio vs fan-out...")
    fig, ax = plt.subplots(figsize=(5, 4))

    fan_out = compute_fan_out_distribution(deleg_G)
    del_ratio = compute_delegation_ratio(deleg_G)

    # Only plot parents that have both metrics
    shared = set(fan_out.keys()) & set(del_ratio.keys())
    fo = [fan_out[n] for n in shared]
    dr = [del_ratio[n] for n in shared]

    ax.scatter(fo, dr, s=12, alpha=0.4, color=COLOR_AGENTIC, edgecolors="none")

    ax.set_xlabel("Fan-out (children per parent)")
    ax.set_ylabel("Delegation ratio (child msgs / parent msgs)")
    ax.set_title("Delegation Ratio vs Fan-out")

    # Use log scale on x if range is large
    if max(fo) > 20:
        ax.set_xscale("log")

    # Annotate correlation
    if len(fo) > 2:
        from scipy import stats
        rho, p_val = stats.spearmanr(fo, dr)
        ax.text(0.97, 0.97,
                f"Spearman $\\rho = {rho:.3f}$\n$p = {p_val:.2e}$\n$n = {len(fo)}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", edgecolor="0.8", alpha=0.9))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_delegation_ratio_fanout.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_delegation_ratio_fanout.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Temporal evolution (4-panel)
# ══════════════════════════════════════════════════════════════════════════
def fig5_temporal_evolution():
    print("Generating Figure 5: Temporal evolution (4 panels)...")
    snapshots = temporal_data["snapshots"]

    dates = [s["date"] for s in snapshots]
    nodes = [s["node_count"] for s in snapshots]
    edge_counts = [s["edge_count"] for s in snapshots]
    density = [s["density"] for s in snapshots]
    gc_frac = [s["giant_component_fraction"] for s in snapshots]

    # Convert dates to numeric for plotting
    from datetime import datetime
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

    # (a) Cumulative nodes
    ax = axes[0, 0]
    ax.plot(date_objs, nodes, color=COLOR_AGENTIC, linewidth=1.5, marker=".", markersize=3)
    ax.set_ylabel("Nodes")
    ax.set_title("(a) Cumulative sessions")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{int(x)}"))

    # (b) Cumulative edges
    ax = axes[0, 1]
    ax.plot(date_objs, edge_counts, color=COLOR_AGENTIC, linewidth=1.5, marker=".", markersize=3)
    ax.set_ylabel("Edges")
    ax.set_title("(b) Cumulative edges")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{int(x)}"))

    # (c) Density
    ax = axes[1, 0]
    ax.plot(date_objs, density, color=COLOR_AGENTIC, linewidth=1.5, marker=".", markersize=3)
    ax.set_ylabel("Density")
    ax.set_title("(c) Network density")

    # (d) Giant component fraction
    ax = axes[1, 1]
    ax.plot(date_objs, gc_frac, color=COLOR_AGENTIC, linewidth=1.5, marker=".", markersize=3)
    ax.set_ylabel("GC fraction")
    ax.set_title("(d) Giant component fraction")

    # Format x-axes
    import matplotlib.dates as mdates
    for ax in axes[1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        for label in ax.get_xticklabels():
            label.set_fontsize(8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_temporal_evolution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_temporal_evolution.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Figure 6: Densification law fit (agentic only, with annotation)
# ══════════════════════════════════════════════════════════════════════════
def fig6_densification_fit():
    print("Generating Figure 6: Densification law fit...")
    fig, ax = plt.subplots(figsize=(5, 4))

    snapshots = temporal_data["snapshots"]
    valid_snaps = [s for s in snapshots if s["node_count"] > 1 and s["edge_count"] > 0]

    log_n = [np.log10(s["node_count"]) for s in valid_snaps]
    log_e = [np.log10(s["edge_count"]) for s in valid_snaps]

    gamma = temporal_data["densification"]["gamma"]
    intercept = temporal_data["densification"]["intercept"]
    r2 = temporal_data["densification"]["r_squared"]

    ax.scatter(log_n, log_e, s=30, color=COLOR_AGENTIC, alpha=0.8, zorder=3,
               label="Daily snapshots")

    # Fit line
    x_fit = np.linspace(min(log_n) - 0.1, max(log_n) + 0.1, 100)
    y_fit = gamma * x_fit + intercept
    ax.plot(x_fit, y_fit, color="red", linestyle="--", linewidth=1.5, alpha=0.8,
            label=f"OLS fit: $\\gamma = {gamma:.3f}$")

    # ChatGPT comparison line (reference only)
    if chatgpt_densification is not None:
        gamma_c = chatgpt_densification["alpha"]
        intercept_c = chatgpt_densification["intercept"]
        y_fit_c = gamma_c * x_fit + intercept_c
        ax.plot(x_fit, y_fit_c, color=COLOR_CHATGPT, linestyle=":", linewidth=1.2,
                alpha=0.6, label=f"ChatGPT ref: $\\gamma = {gamma_c:.3f}$")

    ax.set_xlabel("$\\log_{10}(n)$  [nodes]")
    ax.set_ylabel("$\\log_{10}(e)$  [edges]")
    ax.set_title("Densification Law Fit")

    # Annotate R^2
    ax.text(0.03, 0.97,
            f"$R^2 = {r2:.4f}$\n$\\gamma = {gamma:.3f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", edgecolor="0.8", alpha=0.9))

    ax.legend(frameon=True, fancybox=False, edgecolor="0.8", loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_densification_fit.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_densification_fit.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Figure 7: Participation coefficient distribution
# ══════════════════════════════════════════════════════════════════════════
def fig7_participation_distribution():
    print("Generating Figure 7: Participation coefficient distribution...")
    fig, ax = plt.subplots(figsize=(5, 4))

    participation = compute_participation_coefficient(sem_G, deleg_G)
    values = list(participation.values())

    # Histogram
    bins = np.linspace(0, 0.5, 30)
    ax.hist(values, bins=bins, color=COLOR_AGENTIC, alpha=0.8, edgecolor="white",
            linewidth=0.5)

    ax.set_xlabel("Participation coefficient $P_i$")
    ax.set_ylabel("Count")
    ax.set_title("Participation Coefficient Distribution")

    # Annotate
    mean_p = np.mean(values)
    max_p = max(values)
    # Count how many are at exactly 0
    n_zero = sum(1 for v in values if v == 0.0)
    ax.text(0.97, 0.97,
            f"mean $= {mean_p:.3f}$\n"
            f"max $= {max_p:.3f}$\n"
            f"$P_i = 0$: {n_zero} nodes",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="0.8", alpha=0.9))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_participation_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig7_participation_distribution.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Figure 8: Semantic degree vs delegation degree scatter plot
# ══════════════════════════════════════════════════════════════════════════
def fig8_interlayer_scatter():
    print("Generating Figure 8: Interlayer degree scatter...")
    fig, ax = plt.subplots(figsize=(5, 4))

    shared = set(sem_G.nodes()) & set(deleg_G.nodes())
    sem_deg = [sem_G.degree(n) for n in shared]
    del_deg = [deleg_G.degree(n) for n in shared]

    ax.scatter(sem_deg, del_deg, s=8, alpha=0.25, color=COLOR_AGENTIC, edgecolors="none")

    ax.set_xlabel("Semantic degree")
    ax.set_ylabel("Delegation degree")
    ax.set_title("Semantic vs Delegation Degree")

    # Annotate correlation from pre-computed data
    corr_data = multilayer_data["interlayer_correlation"]
    rho = corr_data["correlation"]
    p_val = corr_data["p_value"]
    n_shared = corr_data["n_shared_nodes"]

    ax.text(0.97, 0.97,
            f"Spearman $\\rho = {rho:.3f}$\n$p = {p_val:.2e}$\n$n = {n_shared}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="0.8", alpha=0.9))

    # Log scale for x if range is large
    if max(sem_deg) > 50:
        ax.set_xscale("symlog", linthresh=1)
    if max(del_deg) > 20:
        ax.set_yscale("symlog", linthresh=1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_interlayer_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig8_interlayer_scatter.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig1_degree_distribution()
    fig2_densification_comparison()
    fig3_fanout_distribution()
    fig4_delegation_ratio_fanout()
    fig5_temporal_evolution()
    fig6_densification_fit()
    fig7_participation_distribution()
    fig8_interlayer_scatter()
    print(f"\nAll 8 figures saved to {FIG_DIR}/")
