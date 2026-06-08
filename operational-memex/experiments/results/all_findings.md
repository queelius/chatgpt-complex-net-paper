# Operational Memex: Experimental Findings

## Corpus

- 3,097 conversations (2,496 ChatGPT, 472 Claude Code, 129 Anthropic)
- 149,623 per-message embeddings (text-embedding-3-small, 256 dims)
- Temporal range: 2022-12-22 to 2026-04-09 (3.4 years)
- 17 marginalia annotations

## Finding 1: Cross-Source Replication

ChatGPT modularity 0.752 (this paper) vs 0.749 (Paper 1, nomic-768).
Structural laws replicate across embedding models despite different
dimensionality (256 vs 768) and threshold (0.84 vs 0.90).

## Finding 2: Platform-Dependent Role Weighting

Optimal user/assistant weight ratio differs by platform:
- ChatGPT: user_weight=3.0 (user text is more discriminative)
- Claude Code: user_weight=0.5 (assistant text is more discriminative)

Reflects conversational (user-driven) vs agentic (assistant-driven) modalities.

## Finding 3: Marginalia as Structural Signal

Annotated conversations exhibit:
- 3.9x higher betweenness centrality (p=0.003, permutation)
- 3.1x higher bridge score (p=0.003, permutation)
- 99.7th percentile of null distribution for both metrics
- No significant difference in degree (p=0.655)

Interpretation: annotated nodes are not hubs (high-degree), they are bridges
(high-betweenness, cross-community connectors).

## Finding 4: Latent Trails

76 implicit trails (chains of >= 3 semantically similar conversations
revisited over time, gap >= 14 days, similarity >= 0.82).

Longest trail: 10 conversations spanning 3 years, crossing from ChatGPT
to Claude Code. Trail 1 is the "memex genesis trail."

5 of 17 annotated conversations appear as trail waypoints (29% vs 9.2%
base rate). Fisher's exact test: odds ratio = 4.17, p = 0.016.

## Finding 5: Platform Isolation

Communities are 87-100% single-source. Only 7.2% of edges cross platform
boundaries. Knowledge domains are platform-segregated. One mixed-source
community (Community 6) bridges academic/network science ChatGPT conversations
with Claude Code implementation sessions.

## Finding 6: Composite Structural Significance

Combining betweenness (40%), bridge score (40%), and trail membership (20%):
- Annotated mean: 0.634 vs non-annotated: 0.433 (p=0.003)
- 13.8x enrichment in top 20 most significant conversations
- Top 20 is 100% trail members

## Finding 7: Topic Lifecycle

- Jan 2023: highest novelty (0.40), everything new
- Early 2023: consolidation phase (low novelty, deepening existing topics)
- Jun 2025: novelty spike (0.14), new exploration
- Mar 2026: most stable month (drift=0.009), focused Claude Code work

## Finding 8: Cross-Platform Migration in Trails

5 of top 10 trails cross from ChatGPT/Anthropic to Claude Code in final legs.
Pattern: explore conceptually in ChatGPT, implement in Claude Code.

## Finding 9: Intra-Conversation Semantic Drift

Most focused conversations (lowest drift): long Claude Code sessions (technical work).
Most divergent (highest drift): short exploratory ChatGPT conversations.
Biggest single-message pivots include "Discussing a Cancer Diagnosis" (annotated),
confirming that topic pivots correlate with emotional/cognitive significance.

## Finding 10: Densification

ChatGPT-only gamma = 1.329 (prior papers: 1.405, 1.410).
Near-universal densification constant replicates across embedding models.
Combined corpus gamma = 5.98 (inflated by Claude Code cluster entry).

## Methods

- Embeddings: OpenAI text-embedding-3-small, 256 dims (Matryoshka)
- Per-message storage with content-hash incremental updates
- Conversation-level: weighted role aggregation (compose at query time)
- Graph: cosine similarity thresholding
- Analytical threshold: 0.84 (max modularity)
- Operational threshold: 0.78 (53% connected, navigable)
- Community detection: Louvain
- Statistical tests: Mann-Whitney U, Fisher's exact, permutation (10,000 iters)
