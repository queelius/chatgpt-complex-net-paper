# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Academic research repository for "Cognitive MRI of AI Conversations" — applying complex network analysis to ChatGPT conversation archives. Contains a published Springer conference paper (Complex Networks 2025) and an in-progress journal extension for PLOS Complex Systems.

The core idea: transform linear conversation logs into semantic similarity networks, revealing knowledge communities, bridge conversations, and cognitive structure in AI-assisted knowledge exploration.

## Active Work

**Journal extension**: Extending the conference paper for PLOS Complex Systems special issue. The extension adds temporal network evolution analysis (densification, preferential attachment, community dynamics, bridge persistence, model era effects). The paper is drafted and submission-ready in `comp-net-2025-journal/paper/PLOS/`.

**Dataset repo**: A standalone dataset repo at `~/github/chatgpt-conversation-corpus/` is being prepared for the sanitized conversation corpus. Pre-release — sanitization still in progress. See that repo's CLAUDE.md for full context.

## Project Structure

```
.
├── comp-net-2025-camera-ready/        # PUBLISHED conference paper (Springer)
│   ├── paper/paper.tex                # Camera-ready paper (svproc.cls)
│   ├── slides/slide-pretty.tex        # Conference presentation (Beamer)
│   └── paper/images/                  # Figures for the paper
├── comp-net-2025-journal/             # SUBMISSION-READY journal extension
│   ├── paper/PLOS/                    # PLOS Complex Systems submission files
│   │   ├── paper.tex                  # Submission version (no embedded figures)
│   │   ├── paper-with-figs.tex        # Review version (figures embedded)
│   │   ├── refs.bib                   # Bibliography
│   │   ├── plos2025.bst              # PLOS bibliography style
│   │   └── submission-info.txt        # Editorial Manager form values, reviewer suggestions
│   ├── paper/figures/temporal/        # Figure files for the paper
│   ├── code/                          # Multi-layer network implementation (earlier approach)
│   └── data/                          # Generated data artifacts
├── code/                              # Original pipeline implementation
│   ├── cli.py                         # Main CLI (embeddings, edges, export)
│   ├── networks.py                    # Network statistics & metrics
│   ├── rec-conv.py                    # Conversation recommendation REPL
│   ├── embedding/                     # LLM (Ollama) & TF-IDF embedding models
│   ├── graph/                         # Edge generation, GPU acceleration, export
│   ├── run_ablation_study.py          # 63-config ablation study runner
│   └── analyze_ablation_*.py          # Ablation analysis & visualization scripts
├── data/                              # Curated reproducibility data (tracked)
│   ├── embeddings/                    # 1908 conversation embeddings (2:1 ratio, no messages)
│   ├── network/                       # Primary edge list (601 nodes, 1718 edges)
│   ├── temporal/                      # Journal paper: monthly network snapshots
│   ├── ablation/                      # Conference paper: 63-config parameter study
│   └── conversations/                 # README only (corpus lives in dev/ and dataset repo)
└── dev/                               # Research data (gitignored)
    ├── chatgpt-4-11-2025_json_no_embeddings/  # 1908 raw conversation JSONs
    ├── conversations-sanitized/       # 1906 sanitized JSONs + metadata (WIP)
    ├── conversations-held/            # 2 excluded conversations
    └── ablation_study/                # Ablation study results & metadata
```

## Building Papers

```bash
# PLOS Complex Systems journal paper (submission version, no figures)
cd comp-net-2025-journal/paper/PLOS
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex

# PLOS Complex Systems journal paper (review version, figures embedded)
cd comp-net-2025-journal/paper/PLOS
pdflatex paper-with-figs.tex && bibtex paper-with-figs && pdflatex paper-with-figs.tex && pdflatex paper-with-figs.tex

# Camera-ready conference paper (Springer svproc class)
cd comp-net-2025-camera-ready/paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex

# Conference slides
cd comp-net-2025-camera-ready/slides
pdflatex slide-pretty.tex
```

## Python Pipeline (`code/`)

```bash
cd code
source venv/bin/activate   # Python 3.12 venv exists in code/venv/

# Full pipeline
python cli.py node-embeddings --input-dir ../dev/chatgpt-4-11-2025_json_no_embeddings \
    --method role-aggregate --embedding-method llm --output-dir ../dev/chatgpt-json-llm
python cli.py edges-gpu --input-dir ./embeddings_json --output-file edges.json
python cli.py cut-off --input-file edges.json --output-file filtered.json --cutoff 0.9
python cli.py export --nodes-dir ./embeddings_json --edges-file filtered.json --format gexf -o graph.gexf

# Ablation study (63 configs: 9 weight ratios x 7 thresholds)
python run_ablation_study.py
python analyze_ablation_results.py
```

Embeddings use Ollama API (`nomic-embed-text` model, 8192 token context). The pipeline weights user messages at 2:1 vs AI responses (validated by ablation study).

## Architecture: Conference Paper Pipeline

1. **Embedding** → per-message embeddings via Ollama, role-aggregated (user/AI separate means), weighted combination, L2-normalized
2. **Edge generation** → all-pairs cosine similarity, threshold filter (θ=0.9 optimal)
3. **Network analysis** → Louvain community detection, centrality metrics, core-periphery decomposition
4. **Export** → GEXF/GraphML for Gephi visualization

## Architecture: Journal Extension (Multi-Layer)

Adds a concept layer on top of the episodic (conversation) layer:
- **Episodic nodes (E)**: Conversations with embeddings (from conference paper)
- **Concept nodes (C)**: Abstract principles extracted via LLM from community clusters
- **E—E links**: Cosine similarity (existing)
- **C→E links**: Instantiation — concept exemplified by episode (LLM-scored)
- **C—C links**: Association — Jaccard overlap of episode sets
- **Hidden connections**: Episode pairs linked through shared concepts but not by direct embedding similarity

## Key Research Parameters

| Parameter | Optimal Value | Source |
|-----------|--------------|--------|
| User:AI weight ratio (α) | 2:1 | Ablation study (63 configs) |
| Similarity threshold (θ) | 0.9 | Phase transition analysis |
| Network size (full) | 601 nodes, 1718 edges | At θ=0.9 (journal paper) |
| Network size (giant component) | 449 nodes, 1615 edges | At θ=0.9 (conference paper) |
| Communities | 15 (modularity 0.750) | Louvain method |
| Embedding model | nomic-embed-text | 768-dim, 8192 token context |

## Key Dependencies

- Python: networkx, scikit-learn, numpy, pandas, requests (Ollama API), tqdm
- LaTeX: svproc.cls (Springer), llncs.cls (LNCS), beamer
- Ollama running locally for embedding generation

## External References

- Code repo: https://github.com/queelius/chatgpt-complex-net (DOI: 10.5281/zenodo.15314235)
- Dataset: 1908 ChatGPT conversations (Dec 2022 – Apr 2025), 601 connected at θ=0.9 (449 in giant component)
- Conference: Complex Networks 2025, published in Springer proceedings
