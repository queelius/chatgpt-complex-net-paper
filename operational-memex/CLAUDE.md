# CLAUDE.md

## What This Is

The `operational-memex/` package is the third paper in the Cognitive MRI series. It extends the methodology from single-platform analysis to cross-platform replication and introduces marginalia (user annotations) as a structural signal.

Prior papers:
- **Paper 1** (comp-net-2025): Semantic graph on 1,908 ChatGPT conversations. Small-world σ=14.9, Heaps β=0.286, γ=1.41.
- **Paper 2** (agentic, ISCS2026): Multiplex network with delegation layer on Claude Code. γ=1.41 universality, orthogonal layers.
- **Paper 3** (this): Cross-source replication + marginalia + operational memex system.

## Pipeline

```bash
# 1. Compute per-message embeddings (reads memex DBs directly, incremental)
python compute_embeddings.py --db data/analysis.db

# 2. Threshold sweep (composes conversation-level embeddings on the fly)
python build_graph.py --db data/analysis.db --sweep 0.80 0.95 0.01

# 3. Build graph and full analysis
python build_graph.py --db data/analysis.db --threshold 0.90 --per-source

# 4. Weight sweep (optimize user/assistant weight ratio for modularity)
python build_graph.py --db data/analysis.db --weight-sweep 0.5 10.0 0.5

# Optional: extract corpus as JSON (for inspection, not needed for pipeline)
python extract_corpus.py --output-dir data/corpus
```

## Key Differences from Prior Papers

- **Per-message embeddings**: Atomic unit is the message, not the conversation. Conversation-level embeddings are composed at query time via weighted role aggregation.
- **Embedding model**: OpenAI text-embedding-3-small at 256 dims (Matryoshka truncation) instead of nomic-embed-text (768-dim)
- **Incremental updates**: Content-hash-based; only new/changed messages are re-embedded
- **Multi-source corpus**: ChatGPT + Claude + Claude Code (conversation-only)
- **Excludes**: claude_code_full subagent sessions (machine-generated, not personal cognitive artifacts)
- **Marginalia**: First study of user annotation behavior in conversation archives

## Architecture

```
compute_embeddings.py   Read memex DBs directly, embed per-message, store in sqlite-vec
build_graph.py          Compose conv embeddings, similarity graph, Louvain, sweeps
extract_corpus.py       (optional) Export per-conversation JSON + manifest
experiments/            Experiment scripts and results
paper/                  LaTeX source and outline
data/                   analysis.db + optional corpus JSON (gitignored)
```

## Data Model (analysis.db)

- `embedding_models`: provenance (provider, model, dimensions, registered_at)
- `conversations`: metadata mirror from memex
- `message_embeddings`: per-message metadata (role, content_hash, char_count, is_short, truncated)
- `message_vectors` (sqlite-vec virtual table): per-message float[256] vectors
- `edges`: (src, dst, weight) for the similarity graph at chosen theta

Conversation-level embeddings are composed at query time:
  user_emb = mean(user message vectors)
  asst_emb = mean(assistant message vectors)
  conv_emb = normalize(user_weight * user_emb + asst_weight * asst_emb)

## Dependencies

See `requirements.txt`. Key additions beyond the agentic paper: `openai`, `sqlite-vec`, `py-memex`.

## Operational Query Tools

```bash
# Semantic search
python query.py --db data/analysis.db search "bernoulli type theory"

# Neighbor recommendation
python query.py --db data/analysis.db neighbors <conv_id_prefix>

# Community detection and listing
python query.py --db data/analysis.db communities --threshold 0.78

# Bridge conversations (cross-community connectors)
python query.py --db data/analysis.db bridges --threshold 0.78
```

Default weights: uw=3.0, aw=1.0 (ChatGPT-optimal). Override with `--user-weight` and `--asst-weight`.

## Key Findings So Far

- **Cross-paper replication**: ChatGPT modularity 0.752 (this paper, OAI-small-256) vs 0.749 (Paper 1, nomic-768)
- **Platform-dependent weighting**: ChatGPT peaks at uw=3 (user-driven), Claude Code at uw=0.5 (assistant-driven)
- **Dual threshold**: analytical (theta=0.84, max modularity) vs operational (theta=0.78, 53% connected)
- **12 interpretable communities** at operational threshold: AI/ML theory, statistics/R, philosophy/consciousness, software dev, physics sim, career/academic, creative writing, etc.

## Conventions

- Results go to `experiments/results/` as JSON
- Figures go to `experiments/figures/`
