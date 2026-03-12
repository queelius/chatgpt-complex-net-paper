# Paper 3 Outline: From Conversation to Delegation

**Working Title:** *From Conversation to Delegation: Multi-Layer Network Analysis of Agentic AI Sessions*

**Target venue:** PLOS Complex Systems (follows Paper 2)
**Authors:** Alex Towell, John Matta — Southern Illinois University Edwardsville

---

## Narrative Arc

Papers 1–2 established that a single user's conversational AI archive self-organizes into a semantic similarity network with macroscopic laws: super-linear densification (γ = 1.405), sub-linear preferential attachment (β = 0.763), and early community stabilization. But conversational AI is passive—the human drives every turn.

Agentic AI changes the dynamic fundamentally: the AI reads files, writes code, executes commands, and spawns child agents that work autonomously. This creates *two* simultaneous network layers: a **semantic similarity layer** (what sessions are about) and a **delegation hierarchy** (who spawned whom). The central question is whether these layers encode the same structure or reveal complementary dimensions of cognitive activity.

**Critical methodological finding**: subagent sessions (compact, prompt-suggestion, and AI-spawned workers) fundamentally distort network metrics. All comparisons must use **parent-only** data to be meaningful. With subagents included: γ=1.707, modularity=0.391, assortativity=+0.90. Parent-only: γ=1.410, modularity=0.278, assortativity=-0.05.

Six themes emerge from 42 experimental findings:

1. **Universal densification (γ ≈ 1.41)**: Parent-only agentic densification (γ=1.410) matches ChatGPT (γ=1.405) almost exactly — a universal property of AI-assisted knowledge exploration regardless of interaction modality.

2. **Knowledge Archipelago vs Cognitive Web**: ChatGPT produces isolated knowledge islands (80% ultra-peripheral, 0.2% connectors, mod=0.749) while agentic coding creates a distributed web (36% ultra-peripheral, 18% connectors, mod=0.278). Communities map to knowledge domains in ChatGPT, to projects in agentic.

3. **Content mode synergy**: Neither user-only nor assistant-only embeddings produce viable networks alone. Text-only works synergistically — user prompts provide topical fingerprinting, AI responses provide semantic scaffolding. Cross-mode degree rankings are uncorrelated (ρ=-0.17).

4. **Robustness asymmetry**: ChatGPT's GC halves after removing 7 bridge conversations (1.5%); agentic requires 21 (9.0%). The Knowledge Archipelago has critical single points of failure; the Cognitive Web has redundant pathways.

5. **Small-world regimes**: Both are small-world (σ>>1) but ChatGPT is extreme (σ=14.9, diameter=18) while agentic is moderate (σ=4.2, diameter=9).

6. **Percolation threshold shift**: ChatGPT θ_c≈0.91 vs agentic θ_c≈0.97 — a 0.06 gap reflecting higher baseline similarity in coding sessions.

---

## Section-by-Section Outline

### Abstract (~250 words)
- **Context:** Prior work (cite Papers 1–2) established cognitive MRI for conversational AI
- **Gap:** Conversational AI is passive; agentic AI introduces autonomous delegation, creating richer cognitive traces
- **Method:** Multiplex network analysis of 3,945 Claude Code sessions (1,061 parents + 2,884 subagents) from the same human subject, constructing semantic similarity and delegation layers
- **Key findings:**
  - Universal densification law (γ ≈ 1.41) holds across both conversational and agentic platforms
  - Opposite cognitive architectures: ChatGPT "Knowledge Archipelago" vs agentic "Cognitive Web"
  - Modularity decreases (0.278 vs. 0.749) — project-driven rather than topic-driven organization
  - Agentic network is 3× more robust to targeted attack (21 vs 7 nodes to halve GC)
  - Content mode synergy: neither user-only nor assistant-only embeddings produce viable networks alone
  - Semantic and delegation layers are largely orthogonal (ρ = 0.115, NMI = 0.581)
- **Contribution:** First multiplex network analysis of agentic AI; discovery of universal densification law across AI interaction modalities; characterization of fundamentally different cognitive architectures

---

### 1. Introduction (~1,500 words)

**Purpose:** Motivate the transition from conversational to agentic AI analysis; frame the multiplex contribution.

**Key arguments:**
1. **Context:** Conversational AI archives are now understood to have meaningful network structure (cite Papers 1–2). Densification, preferential attachment, and community stabilization operate at individual scale.
2. **The agentic transition:** AI is evolving from interlocutor to autonomous agent. Claude Code, GitHub Copilot, Cursor—these tools don't just answer questions; they read files, write code, execute commands, and delegate to child agents. This fundamentally changes the cognitive trace left behind.
3. **The multiplex opportunity:** Agentic sessions create two simultaneous network layers. Semantic similarity captures *what* (topical relationships); delegation hierarchy captures *who did what for whom* (structural relationships). Neither alone tells the full story.
4. **Research questions (4 RQs):**
   - **RQ1 (Semantic):** Do agentic knowledge networks exhibit the same macroscopic laws as conversational ones?
   - **RQ2 (Delegation):** What network properties characterize AI agent delegation structure?
   - **RQ3 (Temporal):** Does the agentic network densify at daily resolution?
   - **RQ4 (Multi-layer):** Do semantic and delegation layers encode the same or different organization?
5. **Preview of findings:** Densification intensifies, modularity weakens, layers are orthogonal, delegation adds a complementary dimension.
6. **Roadmap:** Section 2 reviews related work; Section 3 describes methods; Section 4 presents results; Section 5 discusses implications; Section 6 concludes.

---

### 2. Related Work (~1,500 words)

**Purpose:** Position the contribution at the intersection of multiplex networks, distributed cognition, and AI tool analysis.

**2.1 Multiplex and Multi-Layer Networks**
- Boccaletti et al. (2014) comprehensive review of multiplex networks
- Battiston et al. (2014) participation coefficient, inter-layer degree correlation
- Mucha et al. (2010) multi-slice community detection
- De Domenico et al. (2013) mathematical formalism for multiplex networks
- Key insight: single-layer analysis can miss structure that emerges from inter-layer coupling

**2.2 Densification Laws and Temporal Network Evolution**
- Leskovec et al. (2005, 2007) densification power laws
- Barabási & Albert (1999) preferential attachment
- Brief summary of Paper 2's findings as direct comparison baseline

**2.3 Agentic AI and Tool Use**
- The emergence of agentic coding assistants (Claude Code, Copilot, etc.)
- Agent delegation patterns in distributed systems (multi-agent systems literature)
- Tree-structured delegation: prior work on task decomposition in distributed computing

**2.4 Cognitive MRI Framework**
- Paper 1: static semantic network analysis of AI conversations
- Paper 2: temporal evolution, densification, community stabilization
- Present work: multiplex extension incorporating delegation structure
- Distributed cognition (Hutchins 1995) and extended mind (Clark & Chalmers 1998) as theoretical framing

---

### 3. Methods (~2,500 words)

**Purpose:** Describe dataset, network construction, and analysis methods with enough detail for reproducibility.

**3.1 Dataset**
- Source: Claude Code JSONL session logs from a single human subject (same as Papers 1–2)
- Observation period: November 24, 2025 – March 4, 2026 (101 days, 49 active days)
- Session types and counts:
  - 1,061 parent sessions (human-initiated)
  - 2,884 subagent sessions (AI-spawned children)
  - Total: 3,945 sessions, 374,376 messages
- Subagent taxonomy:
  - User-spawned (2,077): real delegated work, up to 1,057 messages
  - Compact (518): context window compaction
  - Prompt suggestion (289): auto-generated follow-ups
- Content modalities: text, tool_use, tool_result, thinking blocks
- Table: ChatGPT vs. Claude Code dataset comparison (sessions, messages, timespan, modality, hierarchy)
- Ethics: single-subject self-study, same participant across both datasets

**3.2 Semantic Layer Construction**
- Embedding: nomic-embed-text via Ollama (identical to Papers 1–2 for comparability)
- Content preprocessing: text-only mode (strip tool blocks for maximal comparability with ChatGPT methodology)
- Chunking: 256-word chunks with mean aggregation for messages exceeding model context
- Pairwise cosine similarity: 7.78M pairs
- Threshold: θ = 0.9 (established by Paper 1's ablation study)
- Result: undirected weighted graph

**3.3 Delegation Layer Construction**
- Directed graph: edges from parent session → child session via `parent_conversation_id`
- Edge creation: deterministic from session metadata (no thresholding)
- Agent type classification from subagent ID patterns
- Result: directed acyclic graph (maximum depth 2: parent → child, no grandchildren)

**3.4 Temporal Snapshot Construction**
- Cumulative daily snapshots: V(t) = all sessions created by day t; E(t) = all edges between V(t)
- 49 snapshots at daily resolution (vs. 29 monthly snapshots in Paper 2)
- Lightweight metrics per snapshot: node count, edge count, density, components, giant component

**3.5 Network Metrics**
- **Semantic layer:** density, connected components, giant component fraction, average degree, clustering coefficient, transitivity, average shortest path, modularity (Louvain, random_state=42), number of communities, degree assortativity
- **Delegation layer:** node/edge counts, fan-out distribution, delegation ratio, agent type counts
- **Temporal:** densification law fitting (log-log OLS: e(t) ~ n(t)^γ)
- **Multi-layer:** inter-layer degree correlation (Spearman), multiplex participation coefficient (Battiston et al. 2014), community structure comparison (NMI)

**3.6 Comparison Framework**
- Direct comparison with Paper 2 ChatGPT results (same human, same embedding model, same threshold)
- Table of directly comparable metrics side-by-side

---

### 4. Results (~3,000 words)

**Purpose:** Present findings organized by research question.

**4.1 Semantic Network Structure (RQ1)**

*Table: Side-by-side comparison — parent-only agentic vs ChatGPT (GC metrics)*

| Metric | ChatGPT (θ=0.90) | Agentic parents (θ=0.95) |
|---|---|---|
| Nodes (GC) | 453 | 233 |
| Edges | 1,612 | 2,607 |
| ⟨k⟩ | 7.1 | 22.4 |
| Density | 0.016 | 0.097 |
| Clustering | 0.44 | 0.62 |
| Modularity | 0.749 | 0.278 |
| Communities | 14 | 7 |
| Assortativity | +0.09 | -0.05 |
| Densification γ | 1.405 | 1.410 |
| Small-world σ | 14.9 | 4.2 |
| Diameter | 18 | 9 |

**Note**: Agentic uses θ=0.95 (not 0.90) because agentic sessions have higher baseline similarity (μ=0.744 vs 0.633), shifting the percolation threshold by ~0.06.

Key findings:
- **Universal densification** (γ ≈ 1.41 for both, R² > 0.99): the same super-linear accumulation law governs both platforms — a universal property of AI-assisted knowledge exploration
- **Reduced modularity** (0.278 vs. 0.749): agentic communities map to projects, not knowledge domains. A single coding session spans debugging, testing, documentation, deployment
- **Opposite cognitive architectures**: ChatGPT = "Knowledge Archipelago" (80% ultra-peripheral, 0.2% connectors); Agentic = "Cognitive Web" (36% ultra-peripheral, 18% connectors)
- **Robustness asymmetry**: 7 nodes to halve ChatGPT GC vs 21 for agentic — redundant pathways in project-integrated work
- **Content mode synergy**: assistant-only embeddings produce no network (1 edge); user+AI combination is synergistic

*Figure: Degree distribution comparison (ChatGPT vs Claude Code)*
*Figure: Log-log densification plot with both datasets overlaid*

**4.2 Delegation Network Structure (RQ2)**

- 3,945 nodes, 2,884 directed edges
- Agent type breakdown: 2,077 user-spawned (72%), 518 compact (18%), 289 prompt suggestion (10%)
- Fan-out distribution:
  - Mean: 2.72 subagents per parent
  - Max: 156 (a power-user session)
  - Heavy-tailed: most parents have 0–3 children, but a long tail extends to hundreds
- Delegation ratio (child messages / parent messages):
  - Mean: 0.316 — subagents produce about 32% as many messages as their parents on average
  - Max: 7.0 — some subagents exceed their parent's message count

*Figure: Fan-out distribution (log-scale histogram)*
*Figure: Delegation ratio vs fan-out scatter plot*

**4.3 Temporal Evolution (RQ3)**

- 49 daily snapshots over 101-day period
- **Parent-only** densification law: γ = 1.410, R² = 0.994
  - Nearly identical to ChatGPT (1.405) — universal property of AI-assisted knowledge exploration
  - Full dataset (with subagents): γ = 1.707, but this is inflated by template-spawned subagents
  - Interpretation: the densification exponent is independent of interaction modality
- Giant component fraction stabilizes at ~62% by day 30 and remains stable
- Mean fan-out peaks early (~4.3 in late January) then stabilizes around 2.7

*Figure: Temporal evolution of node count, edge count, and density*
*Figure: Densification law fit (log n vs log e) with ChatGPT comparison*
*Figure: Giant component fraction over time*

**4.4 Multi-Layer Analysis (RQ4)**

- **Inter-layer degree correlation:** ρ = 0.115 (p < 10^-13)
  - Statistically significant but weak: semantic centrality and delegation structure are largely independent
  - Being a hub in the semantic network does not predict being a heavy delegator
- **Participation coefficient:** mean P = 0.142, max P = 0.5
  - Most sessions are specialized to one layer (low P)
  - Some sessions bridge both layers (high P) — these are semantically well-connected sessions that also involve heavy delegation
- **Community NMI:** 0.581
  - Moderate overlap: semantic communities and delegation clusters share some structure (both reflect the same human's work) but diverge substantially
  - Interpretation: the two layers capture genuinely complementary dimensions of cognitive activity

*Figure: Participation coefficient distribution (histogram)*
*Figure: Semantic degree vs delegation degree scatter plot colored by layer*

---

### 5. Discussion (~2,500 words)

**Purpose:** Interpret results, compare with prior work, identify implications and limitations.

**5.1 Universal Densification Across Interaction Modalities**
- Parent-only agentic γ=1.410 matches ChatGPT γ=1.405 almost exactly
- This universality is remarkable: two different AI platforms, interfaces, interaction patterns, and observation periods produce the same densification exponent
- The γ≈1.41 value places AI-assisted knowledge networks near citation networks (1.69) and above autonomous systems (1.18) in the Leskovec et al. (2007) taxonomy
- Subagent inflation (γ→1.707) demonstrates that system-generated sessions must be excluded for meaningful comparison
- Hypothesis: γ≈1.41 is a universal property of single-human AI-assisted knowledge exploration, not a platform-specific artifact

**5.2 Knowledge Archipelago vs Cognitive Web**
- The central architectural finding: conversational and agentic AI produce opposite cognitive structures
- ChatGPT: 14 cleanly separated knowledge communities (ML, statistics, physics, etc.) — an "archipelago" of isolated topic islands connected by rare bridge conversations
- Claude Code (parent-only): 7 weaker communities mapping to projects (texwatch, flexhaz, game engine, R packages) — a "cognitive web" where every session touches multiple domains
- Guimerà-Amaral roles quantify this: ChatGPT has 80% ultra-peripheral nodes (confined to single communities) vs 36% for agentic; agentic has 18% connectors vs 0.2% for ChatGPT
- Robustness follows from architecture: the Archipelago fragments after removing 7 bridge nodes (1.5%), the Web survives removing 21 (9.0%)
- This maps to real cognitive resilience: project-integrated work creates redundant knowledge pathways that topical exploration does not

**5.3 The Orthogonality of Semantic and Delegation Layers**
- Central finding: ρ = 0.115, NMI = 0.581
- **Why this matters:** single-layer analysis—the approach of Papers 1–2—systematically misses the organizational structure created by delegation. Conversely, delegation analysis alone misses semantic coherence across the hierarchy.
- **Interpretation through distributed cognition (Hutchins 1995):** the semantic layer captures *what knowledge is being explored*; the delegation layer captures *how cognitive labor is distributed*. These are genuinely different dimensions of intellectual activity.
- **Comparison with other multiplex systems:** social + economic layers in online networks (Szell et al. 2010), structural + functional layers in brain networks (Bullmore & Sporns 2009). The weak inter-layer correlation we observe is consistent with the finding that different relationship types in multiplex networks encode complementary rather than redundant information.

**5.4 Content Mode Synergy and the Embedding Space**
- Assistant-only embeddings produce no network for parent sessions (1 edge at θ≥0.90 out of 22,366 pairs) — AI responses cluster at μ=0.673 with σ=0.064, too homogeneous for discrimination
- User-only is viable for parent sessions but degenerate for subagents (3.6% identical embeddings from template prompts)
- The text-only (combined) mode works synergistically: user prompts provide topical specificity (variance), AI responses provide semantic scaffolding (baseline elevation)
- Cross-mode degree rankings are orthogonal (ρ=-0.17): text-only and user-only capture independent structural dimensions of the same session corpus
- This has methodological implications: content preprocessing is not a neutral choice — it determines which aspects of cognitive structure are visible

**5.5 The Densification Paradox Revisited**
- Paper 2 noted that densification in semantic networks reflects "progressive revelation" of latent structure rather than active tie formation
- The universal γ≈1.41 finding strengthens the case that this is a genuine property rather than an artifact: the same exponent across two different platforms, thresholds (0.90 vs 0.95), and interaction modalities suggests something fundamental about how a single human's AI-assisted knowledge accumulates
- The delegation layer, by contrast, represents actual tie formation (parent spawns child) — no threshold needed

**5.5 Implications for Agentic AI Design**
1. **Delegation is not just task decomposition:** the orthogonality of semantic and delegation layers suggests that delegation patterns reflect more than just dividing a task into subtopics. Structural roles (compaction, prompt suggestion, real work) create a richer hierarchy.
2. **Power-law delegation risks:** the heavy-tailed fan-out distribution (max 156 children from one parent) suggests some sessions attempt extreme delegation. Understanding whether this is efficient or pathological could inform agentic AI tool design.
3. **Multiplex view for conversation archival:** conversation management tools that organize by topic (semantic) would miss the delegation structure; tools that show only the hierarchy would miss semantic connections between unrelated sessions. Multiplex-aware organization could serve both.

**5.6 Limitations**
1. **Single subject:** same limitation as Papers 1–2, but now extended to a different modality
2. **Short observation period:** 101 days vs. 29 months. Community lifecycle and long-term stabilization patterns cannot be assessed. The densification comparison should be interpreted cautiously.
3. **Content mode sensitivity (now addressed):** F37-F38 show assistant-only produces no network, user-only is degenerate for subagents, and text-only is synergistic. Content mode choice is not arbitrary — it fundamentally determines what structure is visible.
4. **Threshold calibration:** ChatGPT θ_c≈0.91 vs agentic θ_c≈0.97 (F39). Direct same-threshold comparisons are misleading; matched-density or percolation-relative comparisons are needed.
5. **Subagent distortion (now addressed):** Including subagents inflates γ from 1.41→1.71, modularity from 0.28→0.39, assortativity from -0.05→+0.90. All cross-platform comparisons must use parent-only data. This is the single most important methodological finding.
6. **Pre-computed similarities:** the densification caveat from Paper 2 applies here
7. **No null models:** the current analysis lacks configuration model baselines, temporal permutation tests, and random assignment controls. Results are descriptive, not yet validated against null hypotheses.

---

### 6. Conclusion (~500 words)

**Purpose:** Synthesize the contribution and point forward.

- We extended the cognitive MRI methodology from single-layer conversational analysis to multiplex analysis of agentic AI sessions
- Key findings:
  1. **Universal densification** (γ ≈ 1.41): the same super-linear accumulation law governs both conversational and agentic AI, suggesting a fundamental property of AI-assisted knowledge exploration
  2. **Opposite cognitive architectures**: conversational AI produces a "Knowledge Archipelago" (modular, fragile topic islands); agentic AI produces a "Cognitive Web" (distributed, resilient project network). Lower modularity (0.278 vs 0.749), 3× more robust to targeted attack
  3. **Content mode synergy**: network structure requires both user and AI contributions; neither alone produces viable similarity networks
  4. **Layer orthogonality** (ρ = 0.115, NMI = 0.581): semantic and delegation layers encode complementary dimensions invisible to single-layer analysis
- These findings suggest that as AI transitions from interlocutor to autonomous agent, the cognitive architecture shifts from deep topical exploration to integrative project execution — the same human's knowledge organizes differently depending on interaction modality
- Future work: multi-user studies, concept extraction and hierarchical abstraction, null model validation, code-aware embeddings, longer observation periods

---

## Figures Required

| # | Description | Data source |
|---|---|---|
| 1 | Dataset comparison schematic (conversational vs. agentic, flat vs. hierarchical) | Conceptual diagram |
| 2 | Side-by-side network visualizations: Archipelago vs Web | NetworkX / Gephi |
| 3 | Degree distribution comparison (both datasets, log-log) | semantic.json |
| 4 | Densification law fit (log n vs. log e, both datasets overlaid, showing γ≈1.41 universal) | temporal.json |
| 5 | Temporal evolution (4-panel: nodes, edges, density, giant component over time) | temporal.json |
| 6 | Fan-out distribution (log-scale histogram) | delegation.json |
| 7 | Participation coefficient distribution | multilayer.json |
| 8 | Guimerà-Amaral role distribution: stacked bar, ChatGPT vs Agentic | role_comparison.json |
| 9 | Targeted attack robustness curves (GC% vs nodes removed) | robustness_comparison.json |
| 10 | Content mode similarity distributions (text-only, user-only, assistant-only) | content_mode_parent_comparison.json |
| 11 | Percolation curves: ChatGPT vs agentic parent-only | chatgpt_vs_agentic_percolation.json |

## Tables Required

| # | Description |
|---|---|
| 1 | Dataset comparison (ChatGPT vs. Claude Code: sessions, messages, timespan, etc.) |
| 2 | Master comparison table (findings.md authoritative version) |
| 3 | Delegation network summary (agent types, fan-out stats) |
| 4 | Multi-layer metrics summary |
| 5 | Densification law comparison with Leskovec et al. (2007) reference networks |
| 6 | Content mode discrimination power (F37-F38 summary) |
| 7 | Guimerà-Amaral role distributions comparison |

## Key References to Add (Beyond Papers 1–2)

- Boccaletti et al. (2014) — multiplex network review
- Battiston et al. (2014) — participation coefficient
- De Domenico et al. (2013) — multiplex formalism
- Szell et al. (2010) — multiplex social networks
- Mucha et al. (2010) — multi-slice community detection
- Bullmore & Sporns (2009) — brain as multiplex
- Guimerà & Amaral (2005) — network roles
- Leskovec et al. (2005, 2007) — densification laws
- Clauset et al. (2009) — power-law testing
- Hutchins (1995) — distributed cognition
- Clark & Chalmers (1998) — extended mind
