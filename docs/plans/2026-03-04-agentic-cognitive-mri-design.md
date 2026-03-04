# Cognitive MRI of Agentic AI Sessions — Research Design

**Date**: 2026-03-04
**Status**: Draft
**Extends**: Towell & Matta (2025, Complex Networks), Towell & Matta (submitted, PLOS Complex Systems)

## 1. Thesis

Agentic AI sessions — where a human delegates tasks to an AI that autonomously reads, writes, and executes code through tool use, and which itself spawns child agents — produce fundamentally richer cognitive traces than passive conversation. We apply and extend the cognitive MRI methodology to characterize the network structure of agentic AI sessions, directly comparing with our prior conversational results to test whether the same macroscopic laws (densification, preferential attachment, community stabilization) hold when AI transitions from interlocutor to autonomous agent.

## 2. Datasets

| Dataset | Sessions | Messages | Timespan | Modality | Source |
|---------|----------|----------|----------|----------|--------|
| ChatGPT | 1,908 | ~60K | 29 months (Dec 2022 – Apr 2025) | Conversational (text only) | OpenAI export |
| Claude Code | 3,945 (1,061 parents + 2,884 subagents) | 374,376 | ~3 months (Nov 2025 – Mar 2026) | Agentic (tool_use, tool_result, text, thinking) | JSONL session logs |

Same human subject in both datasets. Claude Code sessions contain:
- **Parent sessions**: human-initiated, full tool access (Read, Edit, Bash, Agent, etc.)
- **Subagent sessions**: agent-spawned, reduced tool set (no Agent tool → flat hierarchy, max depth 2)
  - User-spawned agents: 2,077 (real delegated work, up to 1,057 messages)
  - Compact agents: 518 (context window compaction)
  - Prompt suggestion agents: 289 (auto-generated follow-ups)

Key structural differences from ChatGPT data:
- **Hierarchical**: parent → child relationships (mean fan-out 7.9, max 156)
- **Action-mediated**: ~90% of content blocks are tool_use/tool_result
- **Multi-model**: Opus 4.6 (2,117), Haiku 4.5 (940), Opus 4.5 (637), Sonnet variants
- **Dense in time**: 23 parent sessions/day average, 100 on peak day

## 3. Research Questions

### RQ1 (Semantic): Do agentic knowledge networks exhibit the same macroscopic laws as conversational ones?
- H1a: Densification exponent γ differs between modalities (expect γ_agentic > γ_conversational because tool use creates more cross-domain connections)
- H1b: Preferential attachment exponent β differs (agentic work is more project-driven, potentially higher exploitation)
- H1c: Community structure differs (project-based vs. topic-based communities)

### RQ2 (Delegation): What network properties characterize AI agent delegation?
- RQ2a: What distribution governs fan-out (subagents per parent)? Power law? Log-normal?
- RQ2b: Are parent–child families semantically coherent? (Do subagents within a family share more semantic similarity than random pairs?)
- RQ2c: Does delegation correlate with efficiency? (More subagents → more files touched per unit time? Shorter parent sessions?)

### RQ3 (Temporal): How does the agentic network evolve, and do delegation patterns change over time?
- H3a: Densification and preferential attachment hold at daily resolution (vs. monthly in PLOS paper)
- H3b: Community structure stabilizes early (replicating PLOS finding)
- H3c: Delegation intensity (fan-out, delegation ratio) increases over time as the human gains experience with agentic tools

### RQ4 (Multi-layer): Do semantic similarity and delegation structure reveal the same or different organization?
- RQ4a: Inter-layer degree correlation — are semantically central sessions also heavy delegators?
- RQ4b: Do multiplex communities differ from single-layer communities?
- RQ4c: Multiplex participation coefficient — do sessions play different structural roles in each layer?

## 4. Analysis A: Semantic Similarity Network

### 4.1 Embedding Generation

Replicate the ChatGPT methodology with ablation:

**Embedding models to test:**
1. `nomic-embed-text` (baseline, identical to ChatGPT work)
2. A code-aware model (e.g., `voyage-code-3` or `Codestral Embed`)
3. A general-purpose model (e.g., `text-embedding-3-large`)

**Content preprocessing options to ablate:**
- Full content (text + tool_use + tool_result) — preserves code semantics
- Text-only (strip tool blocks) — comparable to ChatGPT methodology
- User-messages-only — maximum comparability
- Tool-name sequences only — captures behavioral signature without content

**Weighting schemes to ablate:**
- 2:1 user:AI (established baseline)
- Equal weighting
- Tool-use-aware weighting (weight code blocks differently from natural language)
- User-only (zero AI weight)

**Ablation study**: Full sweep of (embedding model × content preprocessing × weighting × similarity threshold). Report which configuration produces the best community structure (highest modularity with connected giant component).

### 4.2 Network Construction

For each ablation configuration:
- Compute pairwise cosine similarities
- Threshold sweep: 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95
- Identify phase transition threshold (giant component collapse)
- Select optimal threshold for analysis

### 4.3 Network Properties to Compute

**Global:**
- Node count, edge count, density
- Connected components, giant component size/fraction
- Degree distribution (fit power law, exponential, log-normal)
- Average clustering coefficient, transitivity
- Average shortest path length, diameter (giant component)
- Modularity (Louvain), community count
- Assortativity (degree, model, parent/child status)
- Small-world coefficient (σ = C/C_rand × L_rand/L)

**Node-level:**
- Degree, betweenness centrality, closeness centrality, eigenvector centrality
- PageRank
- Community membership
- Local clustering coefficient
- Eccentricity
- Node attributes: parent/child, model, message_count, tool_count, fan-out, project

**Edge-level:**
- Edge betweenness
- Jaccard similarity of neighbor sets
- Bridge classification (evolutionary, integrative, pure — replicate taxonomy)

### 4.4 Comparisons

**ChatGPT vs. Claude Code (same methodology, same human):**

| Metric | ChatGPT (published) | Claude Code (this work) |
|--------|---------------------|------------------------|
| γ (densification) | 1.405 | ? |
| β (pref. attachment) | 0.763 | ? |
| Modularity | 0.750 | ? |
| Communities | 15 | ? |
| Clustering coeff. | 0.44 | ? |

**Subnetwork comparisons within Claude Code:**
- Parents-only vs. full network (with subagents)
- User-spawned agents only vs. compact agents vs. prompt suggestion agents
- By model (Opus vs. Haiku vs. Sonnet subnetworks)
- By project (game, memex, papers, etc.)

## 5. Analysis B: Delegation Network

### 5.1 Network Construction

Directed graph: parent → child edges from `parent_conversation_id`.
- 1,061 potential parent nodes, 2,884 child nodes
- 365 parents have at least one child

### 5.2 Fan-out Distribution Analysis

Fit the fan-out distribution (subagents per parent) to candidate models:
- Power law: P(k) ∝ k^(-α)
- Log-normal: P(k) ∝ (1/k) exp(-(ln k - μ)² / 2σ²)
- Exponential: P(k) ∝ exp(-λk)
- Stretched exponential: P(k) ∝ exp(-λk^β)

Use Clauset et al. (2009) methodology for rigorous power-law testing. Report AIC/BIC for model comparison.

Current data: mean=7.9, median=4, stdev=11.6, max=156. The high variance and right skew suggest log-normal or power law.

### 5.3 Semantic Coherence of Families

For each parent with ≥2 subagents:
- Compute mean pairwise semantic similarity among subagents (intra-family cohesion)
- Compare to null model: random groups of the same size drawn from the full subagent population
- Compute parent–child semantic similarity: is the parent semantically close to its children?
- Statistical test: are real families more semantically coherent than random groupings?

### 5.4 Delegation Efficiency

Correlate delegation metrics with session outcomes:
- Fan-out vs. parent message count (does delegating reduce parent workload?)
- Fan-out vs. total files touched (does delegating increase scope?)
- Fan-out vs. unique tools used (does delegating increase tool diversity?)
- Fan-out vs. session duration (time between first and last message)
- Model selection: do parents choose cheaper models (Haiku) for routine subagents and expensive models (Opus) for complex ones? Correlate subagent model with subagent message count.

### 5.5 Delegation Style Clustering

Cluster parents by their delegation signature vector:
- [n_user_spawned, n_compact, n_prompt_suggestion, mean_child_size, model_entropy, ...]
- Use k-means or DBSCAN
- Interpret clusters: "heavy delegator," "solo worker," "model-mixer," etc.
- Correlate clusters with semantic community membership

## 6. Analysis C: Temporal Evolution

### 6.1 Snapshot Construction

Cumulative snapshots at daily resolution (46 active days).
- V(t) = all sessions created by day t
- E(t) = all edges between nodes in V(t) (from pre-computed similarity matrix)

### 6.2 Replicate PLOS Methodology

Apply identically to agentic data:
1. **Densification law**: fit e(t) ∝ n(t)^γ. Compare γ_agentic to γ_conversational = 1.405.
2. **Preferential attachment**: monthly (or weekly) degree–attachment correlation, fit Π(k) ∝ k^β. Compare β_agentic to β_conversational = 0.763.
3. **Community lifecycle tracking**: Jaccard-based community alignment across snapshots. Track births, deaths, continuations, merges, splits. Compare to ChatGPT: 40 tracked communities, 0 merges/splits.
4. **Bridge formation dynamics**: identify bridge conversations (top-5% betweenness), track centrality over time. Do bridges persist once established (as in ChatGPT)?
5. **Structural evolution**: modularity, clustering coefficient, giant component fraction over time.

### 6.3 Novel Temporal Analyses

**Delegation dynamics over time:**
- Mean fan-out per day — does delegation increase with experience?
- Mean delegation ratio (child_msgs / parent_msgs) per day
- Model selection over time — does the human shift model preferences?

**Intra-day dynamics** (unique to this dataset — ChatGPT had monthly resolution):
- Network changes within a single high-activity day (100 sessions)
- Burst analysis: are sessions temporally clustered? Inter-event time distribution.

**Phase analysis** (analogous to ChatGPT's model eras):
- Divide by model availability or usage patterns
- Compare sub-network metrics across phases

## 7. Analysis D: Two-Layer Multiplex

### 7.1 Layer Definitions

| Layer | Nodes | Edges | Type |
|-------|-------|-------|------|
| Semantic | All sessions | Cosine similarity ≥ θ | Undirected, weighted |
| Delegation | All sessions | parent_conversation_id | Directed, unweighted |

### 7.2 Inter-layer Analysis

**Degree correlation**: Pearson/Spearman correlation between semantic degree and delegation degree (fan-out for parents, 0/1 for children). Are semantically central sessions also heavy delegators?

**Multiplex participation coefficient** (Battiston et al. 2014):
P_i = (M / (M-1)) × (1 - Σ_α (k_i^α / k_i)²)
where k_i^α is degree in layer α, k_i is total degree. P=0 means active in only one layer, P=1 means equally active in all layers.

**Community comparison**:
- Detect communities independently in each layer
- Compute NMI (Normalized Mutual Information) between layer-specific community assignments
- If NMI is high: layers encode redundant structure
- If NMI is low: layers reveal complementary organization

**Multiplex community detection** (Mucha et al. 2010):
- Apply multi-slice modularity maximization
- Compare multiplex communities to single-layer communities

### 7.3 Multiplex Roles

Classify sessions by their role across layers (Guimerà & Amaral 2005, adapted for multiplex):
- **Provincial**: high within-community connectivity in both layers
- **Connector**: bridges communities in semantic layer, peripheral in delegation (or vice versa)
- **Kinless**: low participation in both layers (isolated sessions)

## 8. Null Models and Controls

For each analysis, test against appropriate null models:

| Analysis | Null model | What it tests |
|----------|-----------|---------------|
| Semantic communities | Configuration model (preserve degree, randomize edges) | Are communities stronger than degree-structure alone? |
| Densification | Temporal permutation (shuffle creation timestamps) | Is temporal ordering necessary for densification? |
| Preferential attachment | Uniform random attachment | Is attachment truly preferential? |
| Family coherence | Random parent assignment (shuffle parent_conversation_id) | Is actual delegation more coherent than random? |
| Inter-layer correlation | Layer permutation (shuffle node labels in one layer) | Is the cross-layer structure real? |
| Delegation distribution | Poisson, exponential baselines | Is the heavy-tailed fan-out real or artifactual? |

## 9. Ablation Summary

| Parameter | Values | Purpose |
|-----------|--------|---------|
| Embedding model | nomic-embed-text, voyage-code-3, text-embedding-3-large | Best representation for agentic content |
| Content preprocessing | full, text-only, user-only, tool-names-only | What content drives similarity? |
| User:AI weight | 0:1, 1:2, 1:1, 2:1, 4:1, user-only | Role importance |
| Similarity threshold | 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95 | Phase transition identification |
| Subagent inclusion | parents-only, +user-spawned, +compact, +prompt-suggestion, all | Effect of delegation traces on network structure |
| Temporal resolution | daily, weekly, monthly | Effect of snapshot granularity on evolution metrics |

Total configurations (full sweep): 3 × 4 × 6 × 7 × 5 × 3 = 7,560
Practical approach: fix embedding model first (ablate across content/weight/threshold), then sweep remaining parameters at optimal embedding.

## 10. Expected Contributions

1. **Empirical**: First network characterization of agentic AI session archives, with direct comparison to conversational AI (same human, same methodology)
2. **Methodological**: Extension of cognitive MRI to multi-layer networks incorporating delegation structure; embedding ablation for code-heavy content
3. **Theoretical**: Test whether densification laws and preferential attachment generalize from conversational to agentic AI interaction; characterize delegation as computationally traceable distributed cognition
4. **Practical**: Identify what network properties distinguish "good" delegation patterns; inform design of agentic AI tools

## 11. Candidate Venues

- PLOS Complex Systems (if accepted for PLOS journal, natural follow-up)
- Complex Networks 2026 (conference paper, then journal extension)
- Nature Scientific Reports (broader audience)
- CSCW (Computer Supported Cooperative Work — the human-AI delegation angle)
- CHI (Human-Computer Interaction — cognitive aspects)

## 12. Open Questions

- How to handle sessions with zero user messages (some compact agents)? Exclude or treat as infrastructure?
- Should we weight subagent sessions by their parent's weight in the semantic network?
- The 3-month window may be too short for robust temporal analysis (vs. 29 months for ChatGPT). How to address this limitation?
- Privacy: can we release embeddings (as with ChatGPT data) given that Claude Code sessions contain file paths and code?
