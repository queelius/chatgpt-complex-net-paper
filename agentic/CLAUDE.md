# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

The `agentic/` package extends the Cognitive MRI methodology (see parent repo's CLAUDE.md) from ChatGPT conversations to Claude Code sessions. It constructs and analyzes a **multiplex network** with two layers:

- **Semantic layer** — undirected weighted graph of session similarity (cosine similarity of nomic-embed-text embeddings, θ=0.9 threshold), same methodology as the conference paper pipeline in `code/`
- **Delegation layer** — directed acyclic graph of parent→child session spawning via `parent_conversation_id`

The research question: do these layers encode the same structure, or reveal complementary dimensions? (Answer: largely orthogonal — ρ=0.115, NMI=0.581.)

## Two-Phase Pipeline

The pipeline has an **external embedding step** between phases:

```bash
# Phase 1: Extract sessions from memex DB → export as JSON for embedding
python -m agentic.run --extract-only \
    --db ~/.memex/default/conversations.db \
    --output-dir agentic/data/sessions-text-only \
    --content-mode text-only

# (External) Generate embeddings via code/cli.py, then compute edges

# Phase 2: Full analysis using pre-computed edges
python -m agentic.run \
    --edges-file agentic/data/edges-t0.9.json \
    --db ~/.memex/default/conversations.db \
    --output-dir agentic/output/analysis-001
```

Phase 2 produces four JSON result files: `delegation.json`, `semantic.json`, `temporal.json`, `multilayer.json`.

## Running Tests

```bash
# From the agentic/ directory
python -m pytest tests/ -v

# Single test file
python -m pytest tests/test_delegation.py -v

# Single test
python -m pytest tests/test_delegation.py::test_fan_out_distribution -v
```

38 tests total. Tests use in-memory SQLite fixtures (no external DB needed). No conftest.py — fixtures are per-file.

## Module Architecture

The package is structured as a linear pipeline with four analysis domains:

```
extract.py          Session/Message dataclasses; reads memex SQLite DB
    ↓
preprocess.py       Content extraction modes (text-full, text-only, user-only, tool-names-only)
    ↓
export_json.py      Writes one JSON per session for the embedding pipeline (code/cli.py)
    ↓  (external: embeddings + edge computation)
    ↓
semantic.py         Network metrics on the similarity graph (density, modularity, clustering, etc.)
delegation.py       Directed parent→child graph + fan-out, delegation ratio, agent type counts
temporal.py         Cumulative daily snapshots + densification law fitting (e ~ n^γ)
multilayer.py       Cross-layer analysis: Spearman degree correlation, participation coefficient, NMI
    ↓
run.py              CLI orchestrator (--extract-only for phase 1, default for phase 2)
```

## Key Design Decisions

- **Data source is memex SQLite**, not raw JSONL — sessions are extracted via SQL from `conversations` and `messages` tables
- **Filename-safe IDs**: `export_json.py` replaces `:` with `_` in session IDs for filesystem safety; `run.py` translates them back when loading edges
- **Agent type classification** is string-based on `agent_id` patterns: `acompact` → compact, `aprompt_suggestion` → prompt suggestion, everything else → user-spawned
- **Louvain community detection** is optional (`python-louvain` package) — code degrades gracefully if not installed
- **Temporal snapshots** use lightweight metrics only (no Louvain per snapshot) to keep O(n+e) per day

## Content Modes

The `--content-mode` flag controls what text gets exported for embedding:

| Mode | What's included |
|------|----------------|
| `text-only` (default) | Text blocks from all roles |
| `text-full` | Text + tool names + tool results + thinking blocks |
| `user-only` | Only user text blocks |
| `assistant-only` | Only assistant text blocks |
| `tool-names-only` | Space-separated tool names (e.g., "Read Edit Grep") |

**Critical content mode findings (F37-F38):**
- **Assistant-only produces no network** for parent sessions — only 1 edge at θ≥0.90 out of 22,366 pairs. AI responses cluster around μ=0.673 with σ=0.064, too tight for discrimination.
- **User-only is degenerate for subagents** (3.6% identical-embedding pairs from template prompts) but works well for parent-only analysis.
- **Text-only is synergistic** — user specificity creates peaks while AI scaffolding normalizes the embedding space. Neither alone suffices.
- **Cross-mode orthogonality**: text-only and user-only degree rankings are uncorrelated (ρ=-0.17), capturing independent structural dimensions.

## Experiment Infrastructure

Experiments are registered in `experiments/registry.py` (4 batches) and executed by `experiments/runner.py`. Results are stored as JSON in `experiments/results/` (~97 files). Findings are documented in `experiments/results/findings.md` (F1-F42).

## Dependencies

Defined in `requirements.txt`. Key packages: `networkx`, `numpy`, `scipy`, `scikit-learn`, `python-louvain`, `pandas`, `powerlaw`. Uses the parent repo's venv at `../code/venv/` or create a local one.

## Key Corrected Comparison Numbers

**IMPORTANT**: All comparisons must use **parent-only** agentic data. Including subagents inflates metrics (γ: 1.41→1.71, mod: 0.28→0.39, assort: -0.05→+0.90). The `experiments/results/findings.md` master comparison table is authoritative.

| Metric | ChatGPT (θ=0.90) | Agentic parents (θ=0.95) |
|:---|:---|:---|
| Nodes (GC) | 453 | 233 |
| Edges | 1,612 | 2,607 |
| Modularity | **0.749** | 0.278 |
| Densification γ | 1.405 | 1.410 |
| Small-world σ | **14.9** | 4.2 |
| Robustness (halve GC) | 7 nodes | **21 nodes** |
| Architecture | Knowledge Archipelago | Cognitive Web |
| Communities map to | Knowledge domains | Projects |

The universal constant: **γ ≈ 1.41** across both platforms.

## Paper Context

Results feed into Paper 3: "From Conversation to Delegation: Multi-Layer Network Analysis of Agentic AI Sessions" — outline in `paper/outline.md`.
