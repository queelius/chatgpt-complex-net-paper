# COMPSAC 2026 AIML Workshop Paper: Outline

**Target:** 9th International Workshop on Advances in AI and Machine Learning (AIML 2026)
**Venue:** IEEE COMPSAC 2026, Madrid, Spain (July 7-10)
**Deadline:** April 15, 2026
**Format:** 6 pages, IEEE 2-column, double-blind review
**Submission:** EasyChair (compsac2026)

---

## Working Title

**Multi-Scale Network Organization in AI-Assisted Knowledge Exploration**

*Alternative titles:*
- From Conversations to Cognition: Three Network Perspectives on Human-AI Knowledge Co-Construction
- Cognitive MRI of AI Conversations: Topological, Temporal, and Semantic Organization

---

## Thesis (1 sentence)

A single individual's 29-month archive of 1,908 AI conversations, when analyzed as
a semantic similarity network, reveals multi-scale organization (community structure,
temporal growth laws matching collective knowledge systems, and a latent concept
hierarchy with small-world topology), suggesting that AI-assisted knowledge
exploration produces cognitive artifacts with deep structural regularity.

---

## Figures (3) + Table (1)

| # | Source | File | Shows | Why include |
|---|--------|------|-------|-------------|
| Fig 1 | Paper 1 | `cluster-vis-topics-better.png` | 15 Louvain communities at θ=0.9, hub-spoke vs tree-like topology | Visual hook: this IS the "cognitive MRI." Immediate comprehension. |
| Fig 2 | Paper 2 | `densification_law.pdf` | Log-log e(t) ∝ n(t)^1.405, R²=0.993 | Headline temporal result. One person's knowledge network follows same law as arXiv citations. |
| Fig 3 | Paper 3 | `heaps_law.pdf` | Vocabulary growth β=0.320 vs null β=0.268, p<0.001 | Headline semantic result. Concept consolidation follows quantitative law, exceeds random baseline. |
| Table 1 | New | (none) | Multi-scale summary: layer, method, key metric, comparison to literature | Synthesis artifact. Shows the three layers side-by-side with external benchmarks. |

**Figure file paths:**
- Fig 1: `../comp-net-2025-camera-ready/paper/images/cluster-vis-topics-better.png`
- Fig 2: `../comp-net-2025-journal/paper/figures/temporal/densification_law.pdf`
- Fig 3: `../agentic/experiments/figures/heaps_law.pdf`

---

## Table 1 Draft: Multi-Scale Organization Summary

| Scale | Method | Key Finding | Metric | Literature Comparison |
|-------|--------|-------------|--------|-----------------------|
| **Topological** | Louvain community detection on cosine-similarity network | 15 knowledge communities; heterogeneous hub-spoke (theory) vs tree-like (applied) topology | Q = 0.750; core 25.6% | Comparable to scientific collaboration networks |
| **Temporal** | 29 monthly cumulative snapshots; Leskovec densification model | Super-linear densification; sub-linear preferential attachment | γ = 1.405; β_PA = 0.763 | arXiv γ=1.69, Patents γ=1.26, AS γ=1.18 |
| **Semantic** | LLM concept extraction, hierarchical clustering, bipartite network | Small-world concept co-occurrence; sublinear vocabulary growth; 77% multi-domain episodes | σ ≈ 6.6; β_Heaps = 0.320 | Roget's σ≈13, WordNet σ≈15, word assoc. σ≈5.6 |

---

## Section-by-Section Outline

### Abstract (~150 words)

- **Problem:** AI conversation archives are treated as flat logs; their latent structure is invisible
- **Approach:** "Cognitive MRI" transforms 1,908 conversations (29 months) into a semantic similarity network and analyzes at three scales
- **Results:** (1) 15 communities with heterogeneous topology, (2) super-linear densification γ=1.405 matching collective knowledge systems, (3) small-world concept hierarchy σ≈6.6 matching human semantic memory
- **Implication:** AI-assisted knowledge exploration produces structurally regular cognitive artifacts amenable to principled analysis

### I. Introduction (~1 page, ~800 words)

**Opening hook:** Millions of people now conduct sustained intellectual exploration through AI assistants. These conversations accumulate into rich but opaque archives: sequential logs that hide the structural relationships between topics, the dynamics of knowledge growth, and the conceptual organization that emerges over time.

**Gap:** Existing work on AI conversations focuses on single-turn quality metrics (helpfulness, harmlessness) or prompt engineering. No framework treats multi-month conversation archives as cognitive artifacts with analyzable structure.

**Contribution:** We introduce a multi-scale network analysis framework that reveals three layers of organization:
1. **Topological:** Community structure distinguishing theoretical from applied knowledge domains
2. **Temporal:** Growth laws quantitatively matching collective knowledge systems (citation networks, patent networks)
3. **Semantic:** Latent concept hierarchy with small-world topology matching human semantic memory benchmarks

**Positioning:** Frame as contribution to understanding human-AI cognitive interaction, NOT as a chatbot evaluation method. This is about what sustained AI-assisted exploration *produces* structurally.

**Key references to include:**
- Leskovec et al. (2007), densification laws
- Steyvers & Tenenbaum (2005), small-world semantic memory
- Barabasi & Albert (1999), preferential attachment
- Blondel et al. (2008), Louvain community detection
- Heaps (1978), vocabulary growth

### II. Dataset and Network Construction (~0.75 page, ~600 words)

**Dataset:** 1,908 ChatGPT conversations, Dec 2022 to Apr 2025, single user (first author), spanning ML, statistics, philosophy, programming, cryptography.

**Network construction pipeline** (brief; point to prior work for details):
1. Embed each conversation using nomic-embed-text (768-dim, 8192-token context)
2. Weight user messages 2:1 over AI messages (validated by 63-config ablation)
3. Connect conversations with cosine similarity > θ = 0.9
4. Result: 601 connected nodes, 1,718 edges at final snapshot

**Methodological note:** θ = 0.9 sits just above a phase transition at θ ≈ 0.875 where the network catastrophically fragments, a 2.5% semantic distance separating distinct cognitive contexts.

**Double-blind note:** Anonymize as "a single user's archive"; do not identify first author. Reference conference paper as [anonymized].

### III. Multi-Scale Analysis (~2.5 pages, ~2000 words)

#### III-A. Topological Layer: Knowledge Communities (~0.75 page)

**[Fig 1 here]**

Key points:
- 15 Louvain communities with modularity Q = 0.750
- **Heterogeneous topology:** Theoretical domains (ML/AI, Stats, Philosophy, Math) form dense hub-and-spoke structures; applied domains (Programming, Metaprogramming) form tree-like hierarchies
- Core-periphery organization: 25.6% core nodes (avg degree 18.94) vs 74.4% periphery (avg degree 3.15)
- Three types of bridge conversations connect otherwise distant communities (evolutionary, integrative, pure-bridge)
- This layer reveals *what* the user knows and how knowledge domains relate

#### III-B. Temporal Layer: Growth Laws (~0.75 page)

**[Fig 2 here]**

Key points:
- 29 monthly cumulative snapshots reveal how the network *develops*
- **Super-linear densification:** e(t) ∝ n(t)^1.405 (R²=0.993). Edges grow faster than nodes; the network becomes denser over time. Comparable to arXiv (γ=1.69) and patent networks (γ=1.26).
- **Sub-linear preferential attachment:** β = 0.763. Popular topics attract connections but less aggressively than Barabasi-Albert (β=1). Reflects balanced exploration-exploitation.
- **Community stability:** Modularity stabilizes at 0.75 by late 2023. Zero merge/split events across 40 tracked communities: knowledge domains grow internally rather than fusing. First-mover advantage: earliest communities (Stats, Deep Learning) remain largest.
- This layer reveals *how* knowledge grows, and that individual knowledge development follows the same quantitative laws as collective systems.

#### III-C. Semantic Layer: Concept Hierarchy (~1 page)

**[Fig 3 here]**

Key points:
- LLM-based concept extraction: 3 to 10 noun-phrase concepts per conversation, yielding 6,275 raw concepts, then 500 meta-concepts (hierarchical clustering on embeddings), 50 themes, and 8 domains
- **Small-world topology:** Concept co-occurrence network has σ ≈ 6.6 (clustering 6.89x random, path length only 1.05x random). Comparable to Steyvers & Tenenbaum's (2005) benchmarks for human semantic memory (Roget's σ≈13, WordNet σ≈15, free association σ≈5.6).
- **Sublinear vocabulary growth:** Meta-concept vocabulary follows Heaps' law with β=0.320, significantly above random clustering baseline (β=0.268, p<0.001). Meaningful semantic categories resist consolidation.
- **Multi-domain integration:** 77% of conversations span 2+ knowledge domains. Asymmetric flow: Software Engineering concepts penetrate 40% of non-SE episodes; ML & Networks most porous (30% foreign concepts).
- This layer reveals *what concepts* organize knowledge and how they compose across domains.

**[Table 1 here: multi-scale summary with literature comparisons]**

### IV. Discussion (~0.75 page, ~600 words)

**Synthesis:** The three layers are not independent views but reveal a single organizing principle: AI-assisted exploration produces cognitive artifacts with structural regularities previously observed only in large-scale collective systems. A single individual's 29 months of AI conversation produces:
- Community structure comparable to scientific collaboration networks
- Growth dynamics comparable to citation and patent networks
- Semantic topology comparable to human semantic memory

**Implications for AI research:**
- Conversation archives are underexploited as data. They contain recoverable cognitive structure.
- The framework is applicable to any sustained human-AI interaction corpus (not just ChatGPT).
- Understanding the *structure* of AI-assisted cognition, not just its accuracy, matters for responsible AI deployment.

**Limitations:**
- Single-user case study (n=1). Structural regularities may reflect individual cognitive style, not universal laws. Multi-user replication needed.
- LLM-based concept extraction introduces model-dependent bias.
- Embedding-based similarity conflates topical similarity with methodological similarity.

**Future work:**
- Multi-user studies to test generalizability of growth laws
- Causal analysis: does network structure predict learning outcomes?
- Real-time "cognitive MRI" as a feedback tool for AI-assisted learning

### V. Conclusion (~0.25 page, ~200 words)

Restate thesis. Emphasize the surprising finding that individual-scale AI-assisted knowledge exploration exhibits the same structural laws as collective knowledge systems. Frame cognitive MRI as a general framework applicable beyond this dataset.

### References (~0.5 page, ~15-20 references)

**Core references (must include):**
- [Anonymized] Conference paper (Complex Networks 2025): self-citation, anonymized
- [Anonymized] Journal submission (PLOS Complex Systems): self-citation, anonymized
- Leskovec, Kleinberg, Faloutsos (2007): Graph evolution: Densification and shrinking diameters
- Steyvers & Tenenbaum (2005): The large-scale structure of semantic networks
- Barabasi & Albert (1999): Emergence of scaling in random networks
- Blondel et al. (2008): Fast unfolding of communities in large networks (Louvain)
- Heaps (1978): Information Retrieval: Computational and Theoretical Aspects
- Watts & Strogatz (1998): Collective dynamics of small-world networks
- Borgatti & Everett (2000): Models of core/periphery structures
- Newman (2006): Modularity and community structure in networks

**Domain references (AI interaction):**
- Ouyang et al. (2022): Training language models to follow instructions (InstructGPT)
- OpenAI (2023): GPT-4 Technical Report
- Bommasani et al. (2021): On the opportunities and risks of foundation models

**Embedding/methodology:**
- Nussbaum et al. (2024): nomic-embed-text

---

## Double-Blind Compliance Checklist

- [ ] Remove author names and affiliations from manuscript
- [ ] Replace self-citations with "[anonymized]" (affects conference paper and journal submission)
- [ ] Do not reference specific dataset details that identify the user (e.g., "first author's conversations")
- [ ] Use "a single user" or "the participant" throughout
- [ ] Remove acknowledgments section during review
- [ ] Strip metadata from figure files (author info in PDF metadata)
- [ ] Do not reference GitHub repository URL

---

## Writing Schedule

| Week | Dates | Deliverable |
|------|-------|-------------|
| 1 | Mar 20-27 | Outline finalized (this document). LaTeX skeleton with IEEE template. Figure selection confirmed. |
| 2 | Mar 27-Apr 3 | Full draft: all sections written. Table 1 populated. Figures placed. |
| 3 | Apr 3-10 | Revision pass. Co-author review (John Matta). Double-blind compliance check. |
| 4 | Apr 10-15 | Final polish. Format validation. EasyChair submission. |

---

## File Organization Plan

```
compsac-2026/
  OUTLINE.md          (this file)
  paper/
    paper.tex         (IEEE format manuscript)
    refs.bib          (bibliography)
    figures/
      fig1-network-topology.png    (from Paper 1)
      fig2-densification-law.pdf   (from Paper 2)
      fig3-heaps-law.pdf           (from Paper 3)
  submission/
    ...               (camera-ready artifacts)
```
