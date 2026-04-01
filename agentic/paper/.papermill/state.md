# Papermill State -- Emergent Hierarchical Memory in AI Conversations

> Last refreshed: 2026-03-31

## Project Identity

| Field | Value |
|-------|-------|
| Title | From Episodes to Abstractions: Latent Hierarchical Memory in 1,908 AI Conversations |
| Stage | **accepted** -- ISCS 2026 (Paper ID 39). Camera-ready revisions complete. |
| Format | LaTeX (Springer `svproc.cls`) |
| Venue | ISCS 2026, La Rochelle, France, June 3-5. Springer proceedings. |
| Base paper | Complex Networks 2025 conference paper (Springer, published) |

## Authors

| Name | Email | ORCID | Affiliation |
|------|-------|-------|-------------|
| Alexander Towell | atowell@siue.edu | 0000-0001-6443-9897 | Southern Illinois University Edwardsville |
| John Matta | jmatta@siue.edu | 0000-0002-7666-1409 | Southern Illinois University Edwardsville |

## Thesis

**Claim**: A longitudinal AI conversation archive, analyzed through LLM-extracted concepts organized by hierarchical clustering into a bipartite episode-concept graph, exhibits structural properties characteristic of human semantic memory networks: small-world topology (sigma=6.6, within the 5-15 range for word association and thesaurus networks), sublinear vocabulary growth dynamics distinguishable from both random clustering and bipartite degree-sequence nulls (p < 0.001 for both), and many-to-many cross-domain associations invisible to partition-based methods.

**CLS relationship**: CLS consolidation operates at the vocabulary saturation level (new-concept rate drops 82% from first to second half of the archive), but semantic categories resist consolidation at the growth dynamics level (beta_observed > beta_null). These measure different things and are simultaneously true. See `reviewer-response.md` for full analysis.

**Novelty beyond conference paper**:
1. Bipartite episode-concept graph (many-to-many) vs. partition-based Louvain communities: 77% of episodes span multiple domains
2. Size-normalized asymmetric flow analysis revealing genuine semantic boundaries (mean obs/exp = 0.61) with targeted cross-domain dependencies
3. Dual null models: cluster-permutation (beta_null=0.268) and bipartite configuration (beta_null=0.254), both p < 0.001
4. Small-world topology (sigma = 6.57 against 100 ER graphs) consistent with human semantic memory benchmarks (Steyvers & Tenenbaum sigma = 5.6-15.3)
5. Clustering sensitivity analysis (k = 50-1000) demonstrating qualitative robustness

## Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Episodes | 1,908 | Full ChatGPT archive, Dec 2022 - Apr 2025 |
| Messages | 35,411 (16,503 user, 18,908 assistant) | Corpus statistics |
| Raw concepts extracted | 3,517 unique (7,555 mentions) | Claude Sonnet 3.5 v2, 20 parallel agents |
| Meta-concepts | 500 | Ward linkage on nomic-embed-text 768-dim |
| Themes | 50 | Ward linkage cut |
| Domains | 8 | Ward linkage cut |
| Heaps' beta (raw) | 0.931 | Near-linear growth |
| Heaps' beta (meta-concept) | 0.320 | Sublinear (alphabetical ordering) |
| Cluster-permutation null beta | 0.268 +/- 0.006 | 1,000 permutations, p < 0.001 |
| Bipartite config. null beta | 0.254 +/- 0.007 | 1,000 Curveball permutations, p < 0.001 |
| Small-world sigma | 6.57 | Meta-concept co-occurrence network |
| Clustering C/C_rand | 6.89 | High local clustering |
| Path length L/L_rand | 1.05 | Short global paths |
| Episodes spanning 2+ domains | 77.5% | Bipartite graph analysis |
| Episodes spanning 3+ domains | 35.4% | Cross-domain integration points |
| SE broadcast reach | 40.4% (7/7 domains) | Asymmetric flow analysis |
| ML/Net porosity | 29.9% | Highest absorption rate |
| New-concept rate (1st half) | 13.5% | CLS exploration E4 |
| New-concept rate (2nd half) | 2.5% | CLS exploration E4 (82% drop) |

## Reviews

Reviews received 2026-03-30. Full response in `reviewer-response.md`.

- R1: Moderately original, Limited but convincing, Low confidence
- R2: Moderately original, Limited but convincing, Medium confidence
- R3: Very original, Sufficient validation, High confidence
- R4: Moderately original, Sufficient validation, Medium confidence. Recommends acceptance.

All issues addressed in camera-ready (commit `72a678b`, tag `iscs2026-camera-ready`).

## Manuscript Status

| Component | Status | Notes |
|-----------|--------|-------|
| Abstract | done | Updated: dual null models, degree-sequence language |
| Introduction | done | 5 contributions listed |
| Related Work | done | CLS, cognitive networks, topic modeling, LLM extraction, corpus analysis |
| Methods | done | Updated: corpus stats (R1), embedding dim (R1), silhouette (R2) |
| Results: Extraction | done | Table 1, Fig 1 |
| Results: Hierarchy | done | Table 2, Fig 2 (cooccurrence at 0.85 textwidth per R1) |
| Results: Heaps' law | done | Updated: dual null models (R3), Fig 3 with both distributions |
| Results: Small-world | done | Table 3, Table 4 (sensitivity) |
| Results: Flow | done | Fig 4 |
| Results: Visualization | done | Fig 5 (bipartite map), Fig 6 (zoom-in bridge) |
| Discussion | done | Updated: dual nulls, human memory comparison (R1), CLS nuance |
| Limitations | done | Updated: proof-of-concept framing (R3), WildChat/LMSYS (R2), auto thresholds (R2) |
| Conclusion | done | Updated: dual null reference |
| Bibliography | done | 20 references (splncs04.bst) |

## Figures

| # | File | Description | Status |
|---|------|-------------|--------|
| 1 | `semantic_frequency_distributions.pdf` | 3-panel: Zipf, concepts/episode, episodes/concept | done |
| 2 | `domain_cooccurrence.pdf` | Symmetric domain co-occurrence heatmap | done |
| 3 | `heaps_law.pdf` | Heaps' law with dual null model distributions | done (updated) |
| 4 | `domain_flow.pdf` | Asymmetric flow heatmap + broadcast/porosity bars | done |
| 5 | `knowledge_map_bipartite.pdf` | Bipartite radial knowledge map | done |
| 6 | `zoom_in_bridge.pdf` | Cross-domain episode pair through shared concepts | done |

## Tables

| # | Label | Description |
|---|-------|-------------|
| 1 | `tab:dedup` | Raw vs meta-concept deduplication effect |
| 2 | `tab:domains` | Eight domains with concept counts and examples |
| 3 | `tab:smallworld` | Small-world network metrics |
| 4 | `tab:sensitivity` | Clustering sensitivity: beta and sigma across k |

## Experiments and Data

| Artifact | Path | Description |
|----------|------|-------------|
| Extraction state | `experiments/results/hierarchy_v2/extraction_state.json` | 1,908 episodes x concepts |
| Hierarchy results | `experiments/results/hierarchy_v2/hierarchy_semantic.json` | Cluster memberships, DAG |
| Network metrics | `experiments/results/hierarchy_v2/network_metrics_v2.json` | Heaps' law, co-occurrence |
| Null model results | `experiments/results/hierarchy_v2/heaps_null_model.json` | Both null models + bootstrap |
| Sensitivity | `experiments/results/hierarchy_v2/clustering_sensitivity.json` | k=50-1000 metrics |
| Domain flow | `experiments/results/hierarchy_v2/domain_flow_analysis.json` | Asymmetric flow matrix |
| Concept embeddings | `experiments/results/hierarchy_v2/concept_embeddings.npy` | (3517, 768) nomic-embed-text |
| Silhouette scan | `experiments/results/hierarchy_v2/silhouette_scan.json` | k=2..1000 |
| CLS exploration | `experiments/results/hierarchy_v2/cls_exploration.json` | E1-E5 temporal dynamics |
| GEXF export | `experiments/figures/knowledge_map.gexf` | For Gephi visualization |
| Reviewer response | `paper/reviewer-response.md` | Issue-by-issue response with provenance |

## Git Provenance

| Tag | Commit | Description |
|-----|--------|-------------|
| `iscs2026-camera-ready` | `72a678b` | Paper with all reviewer responses |
| `pre-cls-experiments` | `69ff9ff` | Checkpoint before CLS exploration |

## Build

```bash
cd agentic/paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Relationship to Other Papers in This Repo

1. **Conference paper** (`comp-net-2025-camera-ready/`) -- Published, Springer LNCS. Semantic similarity network from conversation embeddings.
2. **Journal extension** (`comp-net-2025-journal/paper/PLOS/`) -- Submitted to PLOS Complex Systems. Temporal evolution analysis.
3. **This paper** (`agentic/paper/`) -- Concept extraction + hierarchical clustering. Bipartite graph, Heaps' law, small-world, asymmetric flow.
4. **COMPSAC paper** (`compsac-2026/`) -- In progress.

## Next Steps

- [ ] Add 2-3 sentences to camera-ready Discussion noting E4 CLS finding (optional polish)
- [ ] Submit camera-ready to ISCS 2026
- [ ] Consider journal extension with: multi-user replication, prompt sensitivity, alternative embeddings, temporal CLS analysis
