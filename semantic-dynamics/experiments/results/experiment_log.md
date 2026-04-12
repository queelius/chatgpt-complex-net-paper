# Semantic Dynamics Experiment Log

All experiments use embflow package and pre-computed embeddings from
operational-memex/data/analysis.db (149,623 per-message embeddings,
text-embedding-3-small, 256 dims).

## Experiment 1: Velocity-Space Graph

**Script:** experiments/exp01_velocity_graph.py
**Results:** experiments/results/velocity_graph.json

Built graphs in four representation spaces (semantic, velocity_mean,
velocity_exp, curvature) at matched density (p95 threshold per space).

### Key findings:

1. **Spaces are nearly orthogonal.** Community assignments across spaces
   have NMI 0.09-0.21. Semantic, velocity, and curvature find different
   structure in the same data. These are genuinely independent dimensions
   of conversation structure.

2. **Velocity communities are conversation archetypes.**
   - Community 0 (784): Exploratory ChatGPT (high drift, moderate speed)
   - Community 1 (534): Focused ChatGPT (moderate drift)
   - Community 3 (329): Mixed technical (lower drift, Claude Code heavy)
   - Community 2 (132): High-drift Claude Code sessions (most dynamic)

3. **Degree correlates moderately across spaces** (rho=0.576 semantic vs velocity).
   Hub conversations tend to be hubs in both spaces, but with substantial disagreement.

4. **Unique velocity hubs exist.** "Gradient Descent for Low-Rank" has vel_deg=344
   but sem_deg=31. Conversations with common transition patterns but niche topics.

5. **Marginalia is semantic, not dynamic.** Annotated nodes are NOT significant
   in velocity space (p=0.74). Annotations track WHAT you think about, not HOW
   your thinking moves. This distinguishes marginalia from structural dynamics.

6. **Curvature and velocity are moderately related** (NMI=0.207) but both
   independent of semantics.

### Theoretical interpretation:

The four spaces capture orthogonal aspects of conversation:
- Semantic: WHAT the conversation is about (topic)
- Velocity: HOW the conversation changes (transition pattern)
- Curvature: WHERE the conversation turns (structural shape)
- Each is a valid basis for graph construction, community detection, and comparison.

This supports the embflow framework: the same calculus (projection, trajectory,
derivatives) applied in different spaces reveals different cognitive dimensions.
