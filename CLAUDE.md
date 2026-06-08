# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Repository Overview

This is the **umbrella** for the "Cognitive MRI of AI Conversations" research program:
applying complex network analysis to ChatGPT conversation archives. The program began as one
conference paper and grew into a cluster of related papers, all built on the same corpus of
1,908 ChatGPT conversations (December 2022 to April 2025).

The core idea: transform linear conversation logs into semantic similarity networks,
revealing knowledge communities, bridge conversations, and cognitive structure in
AI-assisted knowledge exploration.

This repository is a landing page plus shared working data. Each paper is its own independent
git repository under `papers/` (gitignored here). The umbrella still carries the program's
published compendium DOI (10.5281/zenodo.18778376) and the full pre-split git history.

## Papers (each an independent repo under papers/)

| Local path | Repo | Status |
|------------|------|--------|
| `papers/knowledge-networks/` | cmri-knowledge-networks | Published (Complex Networks 2025, Springer). Holds the conference paper plus the shared pipeline (`code/`) and reproducibility data (`data/`). |
| `papers/temporal-networks/` | cmri-temporal-networks | Rejected at PLOS Complex Systems 2026-04-27, seeking a venue. Temporal-evolution extension. |
| `papers/hierarchical-memory/` | cmri-hierarchical-memory | ISCS 2026. Latent hierarchical memory (episodic/concept layers). Formerly `agentic/`. |
| `papers/embedding-dynamics/` | cmri-embedding-dynamics | In progress (spec stage). |
| `papers/semantic-dynamics/` | cmri-semantic-dynamics | In progress. |
| `papers/operational-memex/` | cmri-operational-memex | Early draft. |

Build instructions, pipeline usage, and per-paper notes live in each child repo's README and
CLAUDE.md. The shared embedding/network pipeline is in `papers/knowledge-networks/code/` (a
working mirror of the standalone `chatgpt-complex-net`).

## Umbrella layout

- `papers/`: independent paper repos (gitignored; clone each `cmri-*` repo into place).
- `dev/`: shared working corpus (gitignored). Raw conversation JSON the pipelines read.
  - `chatgpt-4-11-2025_json_no_embeddings/`: 1,908 raw conversation JSONs.
  - `conversations-sanitized/`: 1,906 sanitized JSONs and metadata (work in progress).
- `future-ideas/`: early-stage directions (compsac-2026 draft, percolation, turn-taking motifs).
- `docs/`: program-level design and planning documents (see `docs/plans/`).

Child code reads the corpus from `$CMRI_CORPUS_DIR`, which defaults to this repo's `dev/`.

## Architecture: the base pipeline (in papers/knowledge-networks/code/)

1. Embedding: per-message embeddings via Ollama, role-aggregated (user and AI separate means),
   weighted combination, L2-normalized.
2. Edge generation: all-pairs cosine similarity, threshold filter (theta = 0.9 optimal).
3. Network analysis: Louvain community detection, centrality metrics, core-periphery
   decomposition.
4. Export: GEXF/GraphML for Gephi visualization.

Embeddings use the Ollama API (`nomic-embed-text`, 8192 token context). The pipeline weights
user messages at 2:1 versus AI responses (validated by the ablation study).

## Key Research Parameters

| Parameter | Optimal Value | Source |
|-----------|--------------|--------|
| User:AI weight ratio (alpha) | 2:1 | Ablation study (63 configs) |
| Similarity threshold (theta) | 0.9 | Phase transition analysis |
| Network size (full) | 601 nodes, 1718 edges | At theta=0.9 (journal paper) |
| Network size (giant component) | 449 nodes, 1615 edges | At theta=0.9 (conference paper) |
| Communities | 15 (modularity 0.750) | Louvain method |
| Embedding model | nomic-embed-text | 768-dim, 8192 token context |

## Key Dependencies

- Python: networkx, scikit-learn, numpy, pandas, requests (Ollama API), tqdm
- LaTeX: svproc.cls (Springer), llncs.cls (LNCS), beamer
- Ollama running locally for embedding generation

## External References

- Pipeline of record: https://github.com/queelius/chatgpt-complex-net (DOI: 10.5281/zenodo.15314235)
- Corpus: https://github.com/queelius/chatgpt-conversation-corpus (1,908 conversations, Dec 2022 to Apr 2025)
- Compendium DOI (this umbrella): 10.5281/zenodo.18778376
- The 2026-06-08 split into a paper cluster is recorded in
  `docs/plans/2026-06-08-papers-cluster-reorg-design.md` and the paired `-plan.md`.
