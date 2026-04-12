# CLAUDE.md

## What This Is

Paper B in the cognitive-MRI series. A standalone methods paper on embedding flow: treating conversation trajectories through embedding space as a dynamical system with velocity, curvature, and drift.

Uses the `embflow` package (~/github/beta/embflow) for all computations.
Uses the analysis database from `operational-memex/data/analysis.db` (149,623 per-message embeddings).

## Pipeline

```bash
# Experiments use the pre-computed embeddings from operational-memex
# No separate embedding step needed

# Run experiments
python experiments/exp01_velocity_graph.py
```

## Dependencies

- embflow (local, ~/github/beta/embflow)
- numpy, scikit-learn, networkx, community (python-louvain)
- sqlite3, sqlite-vec (for reading analysis.db)

## Relation to Other Papers

- Paper 1 (comp-net-2025): Semantic graph on ChatGPT conversations
- Paper 2 (agentic, ISCS2026): Multiplex network with delegation layer
- Paper 3 (operational-memex): Cross-source replication, trails, marginalia
- Paper B (this): Embedding flow calculus, velocity/curvature graphs, semantic dynamics
