# Papermill State — Emergent Hierarchical Memory in AI Conversations

> Last refreshed: 2026-03-14

## Project Identity

| Field | Value |
|-------|-------|
| Title | From Episodes to Abstractions: Latent Hierarchical Memory in 1,908 AI Conversations |
| Stage | **revising** — reviewed 2026-03-13 (major-revision) and 2026-03-14 (minor-revision); null models and sensitivity analysis added; CLS narrative needs reconciliation |
| Format | LaTeX (Springer `svproc.cls`) |
| Venue | Under consideration — originally targeted ISCS 2026 (8pp short paper), now expanded to 13pp; venue TBD |
| Base paper | Complex Networks 2025 conference paper (Springer, published) |

## Authors

| Name | Email | ORCID | Affiliation |
|------|-------|-------|-------------|
| Alexander Towell | atowell@siue.edu | 0000-0001-6443-9897 | Southern Illinois University Edwardsville |
| John Matta | jmatta@siue.edu | — | Southern Illinois University Edwardsville |

## Thesis

**Claim**: When AI conversation archives are analyzed through LLM-extracted semantic concepts rather than conversation-level embeddings, a bipartite episode--concept structure reveals latent hierarchical organization with three distinctive properties: small-world topology consistent with human semantic memory benchmarks, many-to-many episode--domain associations invisible to partition-based methods, and asymmetric cross-domain concept sharing confirming genuine semantic boundaries. The semantic clustering creates meaningful categorical distinctions (validated against a random-clustering null model), though semantic categories resist rather than facilitate consolidation — maintaining discriminable knowledge representations rather than following the consolidation-efficiency prediction of CLS theory.

**Novelty beyond conference paper**:
1. Bipartite episode--concept graph (many-to-many) vs. partition-based Louvain communities — 77% of episodes span multiple domains
2. Size-normalized asymmetric flow analysis revealing genuine semantic boundaries (mean obs/exp = 0.61) with targeted cross-domain dependencies
3. Null model showing semantic clustering creates meaningful categorical distinctions (β_real = 0.320 vs β_null = 0.268, p < 0.001) — the CLS direction-inversion is itself an interesting finding
4. Small-world topology (σ ≈ 6.6 against 100 ER graphs) consistent with human semantic memory benchmarks (Steyvers & Tenenbaum σ = 5.6–15.3)
5. Clustering sensitivity analysis (k = 50–1000) demonstrating qualitative robustness of all key findings

## Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Episodes | 1,908 | Full ChatGPT archive, Dec 2022 – Apr 2025 |
| Raw concepts extracted | 3,517 unique (7,555 mentions) | Claude Sonnet 3.5 v2, 20 parallel agents |
| Meta-concepts | 500 | Ward linkage on nomic-embed-text 768-dim |
| Themes | 50 | Ward linkage cut |
| Domains | 8 | Ward linkage cut |
| Heaps' β (raw) | 0.931 | Near-linear growth |
| Heaps' β (meta-concept) | 0.320 | Sublinear — CLS consolidation signature |
| Small-world σ | 5.69 | Meta-concept co-occurrence network |
| Clustering C/C_rand | 6.88 | High local clustering |
| Path length L/L_rand | 1.21 | Short global paths |
| Episodes spanning 2+ domains | 77.5% | Bipartite graph analysis |
| Episodes spanning 3+ domains | 35.4% | Cross-domain integration points |
| SE broadcast reach | 40.4% (7/7 domains) | Asymmetric flow analysis |
| ML/Net porosity | 29.9% | Highest absorption rate |
| Max flow asymmetry | Stats→LLM-Eng 10:1 | Strongest directional dependency |

## Manuscript Status

| Component | Status | Notes |
|-----------|--------|-------|
| Abstract | ✅ complete | 4 findings: hierarchy, Heaps', small-world, asymmetric flow |
| Introduction | ✅ complete | 5 contributions listed |
| Related Work | ✅ complete | 3 paragraphs: CLS, cognitive networks, text networks |
| Methods | ✅ complete | 5 subsections: dataset, extraction, embedding, clustering, bipartite |
| §4.1 Extraction & Dedup | ✅ complete | Table 1, Fig 1 |
| §4.2 Emergent Hierarchy | ✅ complete | Table 2, Fig 2 (co-occurrence heatmap) |
| §4.3 Vocabulary Growth | ✅ complete | Fig 3 (Heaps' law — both curves) |
| §4.4 Small-World | ✅ complete | Table 3 |
| §4.5 Asymmetric Flow | ✅ complete | Fig 4 (flow heatmap + broadcast/porosity) |
| §4.6 Bipartite Visualization | ✅ complete | Fig 5 (knowledge map), Fig 6 (zoom-in bridge) |
| Discussion | ✅ complete | 4 paragraphs: CLS, small-world, flow, limitations |
| Conclusion | ✅ complete | |
| Bibliography | ✅ complete | 20 references (splncs04.bst) |

## Figures

| # | File | Description | Status |
|---|------|-------------|--------|
| 1 | `semantic_frequency_distributions.pdf` | 3-panel: Zipf, concepts/episode, episodes/concept | ✅ |
| 2 | `domain_cooccurrence.pdf` | Symmetric domain co-occurrence heatmap | ✅ |
| 3 | `heaps_law.pdf` | Heaps' law — raw (β=0.931) + meta-concept (β=0.320) curves | ✅ |
| 4 | `domain_flow.pdf` | Asymmetric flow heatmap + broadcast/porosity bars | ✅ |
| 5 | `knowledge_map_bipartite.pdf` | Bipartite radial knowledge map (1,908 ep + 499 mc) | ✅ |
| 6 | `zoom_in_bridge.pdf` | Zoom-in: Stats↔Philosophy episode pair through shared concepts | ✅ |

## Tables

| # | Label | Description |
|---|-------|-------------|
| 1 | `tab:dedup` | Raw vs meta-concept deduplication effect |
| 2 | `tab:domains` | Eight domains with concept counts and examples |
| 3 | `tab:smallworld` | Small-world network metrics |

## Experiments & Data

| Artifact | Path | Description |
|----------|------|-------------|
| Extraction state | `experiments/results/hierarchy_v2/extraction_state.json` | 1,908 episodes × concepts |
| Hierarchy results | `experiments/results/hierarchy_v2/hierarchy_semantic.json` | Cluster memberships, DAG, bipartite |
| Network metrics | `experiments/results/hierarchy_v2/network_metrics_v2.json` | Heaps' law, co-occurrence, cross-domain |
| Domain flow | `experiments/results/hierarchy_v2/domain_flow_analysis.json` | Asymmetric flow matrix and derived metrics |
| Concept embeddings | `experiments/results/hierarchy_v2/concept_embeddings.npy` | (3517, 768) nomic-embed-text |
| Silhouette scan | `experiments/results/hierarchy_v2/silhouette_scan.json` | k=2..1000 silhouette scores |
| GEXF export | `experiments/figures/knowledge_map.gexf` | For Gephi visualization |

## Build

```bash
cd agentic/paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Relationship to Other Papers in This Repo

This repo contains three papers from the same research program:

1. **Conference paper** (`comp-net-2025-camera-ready/`) — Published, Springer LNCS. Semantic similarity network from conversation embeddings. Louvain communities. θ=0.9 threshold. This is the "Paper 1" baseline.

2. **Journal extension** (`comp-net-2025-journal/paper/PLOS/`) — Submission-ready for PLOS Complex Systems. Adds temporal evolution analysis (densification, preferential attachment, bridge persistence). Has its own `.papermill/` at the repo root.

3. **This paper** (`agentic/paper/`) — Concept extraction + hierarchical clustering. Bipartite graph, Heaps' law, small-world, asymmetric flow. Different methodology (concepts, not embeddings), complementary findings. Has its own `.papermill/` here.

## Next Steps

- [ ] Decide venue (ISCS 2026 at 8pp? Different venue at 13pp? Expand to full journal paper?)
- [ ] Review pass — run `/papermill:review` for structured feedback
- [ ] Polish figures for print quality (especially bipartite map at reduced width)
- [ ] Update `outline.md` to reflect current paper structure (outline has V1 numbers)
- [ ] Consider adding the silhouette analysis back (cut for 8pp, room now at 13pp)
- [ ] Consider adding episode-episode semantic similarity as a third edge type in the bipartite map
