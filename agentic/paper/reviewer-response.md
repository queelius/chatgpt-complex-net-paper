# Reviewer Response and Provenance Trail

**Paper:** From Episodes to Abstractions: Latent Hierarchical Memory in 1,908 AI Conversations
**Venue:** ISCS 2026 (Paper ID 39)
**Reviews received:** 2026-03-30
**Camera-ready tag:** `iscs2026-camera-ready` (commit `72a678b`)

## Reviewer Summary

| Reviewer | Significance | Validation | Confidence | Stance |
|----------|-------------|-----------|------------|--------|
| R1 | Moderately original | Limited but convincing | Low | Interested, wants figures + stats |
| R2 | Moderately original | Limited but convincing | Medium | Likes it, wants multi-user + auto thresholds |
| R3 | Very original | Sufficient | High | Strong advocate, suggests stricter null model |
| R4 | Moderately original | Sufficient | Medium | Recommends acceptance |

## Issue-by-Issue Response

### R1-1: Figure readability

> "The figures in the paper are so small and hard to see."

**Action:** Increased `domain_cooccurrence.pdf` from `0.65\textwidth` to `0.85\textwidth`. All other figures were already at `\textwidth`.
**File:** `paper.tex:324`
**Commit:** `72a678b`

### R1-2: Dataset statistics upfront

> "It would be better if the statistics of the dataset are shown at the beginning of section 3."

**Action:** Expanded Section 3.1 (Dataset) with: 35,411 messages (16,503 user, 18,908 assistant), conversation lengths (2-416, median 10, mean 18.6), topic areas, model generations.
**File:** `paper.tex:180-188`
**Commit:** `72a678b`

### R1-3: Embedding dimension validity

> "It is not clear whether the dimension of concept embedding (768) is valid or not."

**Action:** Added clarification that 768 is the native output dimension of `nomic-embed-text` (BERT-derived architecture), not a hyperparameter choice.
**File:** `paper.tex:211-215`
**Commit:** `72a678b`

### R1-4: Compare with human conversation

> "If the authors compare the analysis of AI conversation with those of human conversation, it would be more appealing."

**Action:** Expanded Discussion paragraph on small-world topology. Added explicit sigma range comparison (5-15) with human word association and category fluency networks. Renamed paragraph to "Small-World Topology and Comparison with Human Memory Networks."
**File:** `paper.tex:573-580`
**Commit:** `72a678b`
**Note:** Direct comparison with human conversation data (as opposed to human semantic memory benchmarks) would require a separate corpus and is noted as future work.

### R2-1: Single-user case study

> "The entire dataset is based on a single-user case study. This severely restricts the ability to claim these hierarchical patterns are universal."

**Action:** Reframed Limitations paragraph with R3's "proof-of-concept" language. Cited WildChat and LMSYS-Chat-1M as natural replication targets.
**File:** `paper.tex:604-610`
**Commit:** `72a678b`

### R2-2: Manual hierarchy cut points

> "The hierarchy's cut points (k=500,50,8) were manually chosen rather than being naturally discovered by the algorithm."

**Action:** Expanded Methods discussion. Noted silhouette scores are monotonically increasing for k >= 20 with no distinct peak, indicating no natural cluster boundaries. Referenced sensitivity analysis (Table 4) showing robustness across k=200-1000. Mentioned automated thresholding methods (gap statistic, stability-based selection) as future work.
**File:** `paper.tex:231-239, 611-616`
**Commit:** `72a678b`

### R2-3: LLM extraction dependency

> "The initial concept extraction relies heavily on the specific quirks of Claude 3.5 Sonnet and the exact prompt used."

**Action:** Acknowledged in Limitations: "Concept extraction depends on the specific LLM and prompt used; alternative models or prompts would produce different concept sets, though the downstream structural properties (small-world, Heaps' law) may be more robust than the specific concept vocabulary."
**File:** `paper.tex:616-619`
**Commit:** `72a678b`
**Note:** Testing with alternative LLMs deferred to journal extension.

### R3-1: Alternative null model (bipartite configuration)

> "A stricter baseline would also randomize meta-concept co-occurrence across episodes, allowing the authors to disentangle temporal exploration dynamics from clustering granularity effects."

**Action:** Implemented bipartite configuration null model using the Curveball algorithm for uniform sampling of binary matrices with fixed row and column sums. 1,000 permutations.
**Result:** beta_null = 0.254 +/- 0.007, p < 0.001. Observed beta = 0.320 lies entirely above this stricter null, confirming the finding is not an artifact of the bipartite degree sequence.
**Files:**
- `experiments/figures/heaps_null_model.py` (added `null_model_bipartite()` function)
- `experiments/results/hierarchy_v2/heaps_null_model.json` (added `bipartite_null_model` section)
- `experiments/figures/heaps_law.pdf` (updated: both null distributions shown)
- `paper.tex:349-371` (new "Null model 2" paragraph in Results)
- `paper.tex:553-564` (updated Discussion)
- `paper.tex:632-634` (updated Conclusion)
- `paper.tex:40-42` (updated Abstract)
**Commit:** `72a678b`

### R3-2: Single-user proof-of-concept framing

> "The present study is best understood as a proof-of-concept that motivates, rather than establishes, this generalization."

**Action:** Adopted this framing verbatim in Limitations.
**File:** `paper.tex:606-607`
**Commit:** `72a678b`

### R2/R3: Multi-user replication

> R2: "Expand the analysis to multi-user/multi-LLM datasets."
> R3: "Direct replication on independent archives would be needed."

**Action:** Cited WildChat and LMSYS-Chat-1M as replication targets. Noted as future work, not feasible for camera-ready.
**File:** `paper.tex:609-610`
**Commit:** `72a678b`

## Post-Review Exploration (not in camera-ready)

### CLS Consolidation Experiments (E1-E5)

**Motivation:** The paper invokes CLS theory but the main finding (beta_observed > beta_null) inverts the CLS consolidation prediction. Does CLS consolidation exist at a different level of analysis?

**Experiments:**

| ID | Experiment | Result | CLS-consistent? |
|----|-----------|--------|----------------|
| E1 | Windowed beta over time | Essentially flat (slope ~ 0) | Barely |
| E2 | Chronological vs random ordering | Chrono beta (0.328) > random (0.248), z=10.0 | No |
| E3 | Model-era segmented beta | Non-monotonic across eras | No |
| E4 | New-concept rate over time | **82% drop** (0.135 to 0.025) | **Yes (strong)** |
| E5 | Domain activation + spanning | All 8 domains by episode 2; spanning stable ~77% | Neutral |

**Key finding:** CLS consolidation IS present at the vocabulary saturation level (E4), but operates differently than the global beta analysis suggests. Later conversations overwhelmingly map to existing meta-concepts (CLS prediction confirmed), but temporal locality inflates beta by introducing related-but-distinct concepts in bursts (beta dynamics anti-CLS).

**Reconciliation:** E4 asks "is this episode's knowledge already in the archive?" (increasingly yes = CLS). E2/global beta asks "does temporal order make growth more/less sublinear?" (less = anti-CLS for beta, but only because of bursty topic exploration).

**Files:**
- `experiments/figures/cls_exploration.py`
- `experiments/figures/cls_exploration.pdf`
- `experiments/results/hierarchy_v2/cls_exploration.json`
**Commit:** `5a67e84`
**Tag:** `pre-cls-experiments` (checkpoint before these experiments)

## Git Provenance

| Tag | Commit | Description |
|-----|--------|-------------|
| `iscs2026-camera-ready` | `72a678b` | Paper with all reviewer responses |
| `pre-cls-experiments` | `69ff9ff` | Checkpoint before CLS exploration |
| (latest) | `5a67e84` | CLS experiments added |

To restore the exact camera-ready state: `git checkout iscs2026-camera-ready`
