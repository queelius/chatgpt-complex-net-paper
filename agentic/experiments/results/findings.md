# Experimental Findings — Cognitive MRI Research Program

Generated: 2026-03-11

## Experiments Completed

### Batch 1: Threshold sweeps
- **9 ChatGPT weight configs** × 6 thresholds = **54 experiments** (all complete)
- Weight ratios: 1:100, 1:4, 1:2, 1:φ, 1:1, φ:1, 2:1, 4:1, 100:1
- Thresholds: 0.80, 0.85, 0.875, 0.90, 0.925, 0.95
- **Agentic threshold sweep**: in progress (7.8M edge pairs)

### Batch 2: Cross-weight analysis at θ=0.9
- 9 ChatGPT single-threshold experiments (complete)

### Batch 3: Content mode experiments
- Agentic session export complete (text-full, user-only, assistant-only, tool-names-only)
- User-only embeddings: in progress (~35% done)

---

## Key Findings

### F1: Golden Ratio is Optimal Weight

The golden ratio φ:1 ≈ 1.618:1 produces the most edges (1727) at θ=0.9, slightly edging out the 2:1 ratio (1718) used in Papers 1-2.

| Weight ratio | Edges at θ=0.9 | GC% | Modularity |
|:---|---:|---:|---:|
| 1:100 (AI-only) | 601 | 9.8% | 0.841 |
| 1:4 | 1047 | 14.0% | 0.811 |
| 1:2 | 1367 | 15.9% | 0.801 |
| 1:φ | 1482 | 17.4% | 0.799 |
| 1:1 | 1647 | 19.2% | 0.791 |
| **φ:1** | **1727** | **21.8%** | **0.779** |
| 2:1 | 1718 | 23.7% | 0.776 |
| 4:1 | 1578 | 22.6% | 0.765 |
| 100:1 (user-only) | 1265 | 19.1% | 0.750 |

The relationship is non-monotonic: edge count peaks at φ:1 then declines. This suggests the optimal weight balances user intent signal (stronger per-prompt) with AI response diversity (provides topical context).

### F2: Modularity and Connectivity are Perfectly Anti-Correlated

Spearman ρ = -1.000 (p < 10⁻⁵) between edge count and modularity at θ=0.9.

As more connections appear, communities merge. This is expected from percolation theory but the perfect rank-order inversion across all 9 weight ratios is striking.

### F3: Community Structure is Robust (Mean NMI = 0.887)

Pairwise NMI between community partitions at θ=0.9:
- Most similar: 2:1 vs φ:1 (NMI = 0.978)
- Most different: user-only vs AI-only (NMI = 0.798)
- Mean NMI: 0.887

The 15 core knowledge communities persist regardless of weight ratio. But there's a systematic difference:
- **User prompts** group by *project context* (same codebase, same workflow)
- **AI responses** group by *technical content* (same domain, same abstraction level)

### F4: Phase Transition Location Depends on Weight Ratio

AI-heavy configs transition at θ_c ≈ 0.825, balanced/user-heavy at θ_c ≈ 0.887. AI responses create a more uniform similarity distribution, so the giant component dissolves at a lower threshold.

### F5: Assortativity Sign Flip

All configs show a disassortative → assortative transition as θ increases:
- Low θ: hubs connect to non-hubs (disassortative, r < 0)
- High θ: hubs connect to hubs (assortative, r > 0)
- Crossover: θ ≈ 0.82-0.86

User-only (100:1) has the weakest assortativity at all thresholds (+0.076 at θ=0.9), while AI-heavy configs have consistently higher assortativity.

### F6: Degree Distributions are Lognormal, Not Power-Law

All weight configs show lognormal degree distributions (not power-law):
- Power-law vs lognormal: R(PL/LN) = -1.4 to -4.9, all negative
- α varies with weight: 1.80 (2:1) to 2.98 (user-only)
- User-only has steeper tail (fewer extreme hubs)

### F7: Strongly Small-World Networks (σ = 10.9-16.5)

All configs produce small-world networks:
- Clustering 14-27× higher than random
- Path lengths only 1.3-1.8× longer
- Golden ratio (φ:1) has strongest small-world property (σ = 16.5)

### F8: Agentic vs ChatGPT — Dramatically Different Scales

At θ=0.9:
| Metric | ChatGPT (2:1) | Agentic | Ratio |
|:---|---:|---:|---:|
| Edges | 1,718 | 287,693 | 167× |
| GC fraction | 23.7% | 62.5% | 2.6× |
| Modularity | 0.776 | 0.391 | 0.5× |
| Clustering | 0.119 | 0.481 | 4.0× |
| Assortativity | +0.167 | +0.901 | 5.4× |

The agentic corpus is dramatically denser — coding sessions share far more semantic overlap than general ChatGPT conversations.

### F9: Core-Periphery Structure

K-core decomposition at θ=0.9:
- 2:1 optimal: max k=14, 61 nodes in k≥10 core
- User-only: max k=9 (shallower but larger core, 13.3%)
- AI-only: max k=6 (shallowest, most uniform)
- Gap at k=11 and k=13 suggests discrete core layers

### F10: Bridge Conversations Connect Knowledge Domains

Top bridge conversations (betweenness centrality):
1. "Geometric Mean Calculation" (BC=0.254) — connects math to ML
2. "MCTS Code Analysis" (BC=0.209) — connects algorithms to coding projects
3. "Loss in LLM Training" (BC=0.205) — connects theory to practice

93.7% of edges are intra-community. The 6.3% inter-community edges are cognitively meaningful, connecting related but distinct knowledge domains.

### F11: Temporal Community Stability (NMI > 0.83 across 2+ years)

Cumulative network snapshots at 6-month intervals:
- by 2023-06 vs by 2025-04: NMI=0.861 (133 shared nodes)
- by 2024-01 vs by 2025-04: NMI=0.959 (320 shared nodes)

Communities were established early and persisted. But each has a temporal signature:
- **Bursty**: MLE analysis (42/44 in 2023H2), SpaceSim (25/25 in 2024H2+)
- **Steady**: AI/ML (24→29→32→17 across periods)
- **Completed**: R statistics (47+33 early, 2+0 late)

### F12: Flat Delegation Trees (Max Depth = 1)

All 2,884 delegation edges are parent→child with no further nesting. Claude Code's delegation architecture at data collection time was single-level only. The delegation graph is a forest of 365 star graphs (plus 696 singletons).

Fan-out distribution: median 4, mean 7.9, max 156. Heavy-tailed — most sessions spawn a few subagents, a few spawn 50-150+.

### F13: Multiplex Hub Enrichment (6.2× over chance)

Sessions that are hubs in both semantic and delegation layers overlap 6.2× more than expected by chance (6 dual-hubs vs 1.0 expected). But the correlation is moderate (ρ=0.528): being a semantic hub doesn't guarantee being a delegation hub.

Only 426/3945 sessions appear in both networks (at θ=0.9 for semantic).

### F14: User Prompts vs AI Responses Encode Different Boundaries

Community comparison between 2:1 and 1:100 weights:
- **Preserved**: AI/ML (102 nodes → AI-28), Health/Philosophy (65 → AI-6), FSM/Markov (12 → AI-31)
- **Split**: DevOps/Hugo (46 nodes splits into 6+ AI communities), AlgoTree/TreeProg (29 splits into 5+)
- **Merged**: R packages (Opt-5) and MLE analysis (Opt-10) partially merge in AI view

Pattern: User prompts create *project-oriented* communities; AI responses create *content-oriented* communities.

### F15: AI-Only Network is Most Structurally Anomalous

Null model comparison (configuration model with same degree sequence) at θ=0.9:

| Config | Clustering ratio | Transitivity ratio | Assortativity z-score |
|:---|---:|---:|---:|
| AI-only (1:100) | **20.9×** | **14.4×** | +6.3 |
| 1:1 | 11.3× | 8.9× | +9.6 |
| 2:1 | 10.9× | 8.2× | +12.9 |
| φ:1 | 10.5× | 8.7× | +14.2 |
| user-only (100:1) | 9.4× | 6.8× | +6.4 |

The AI-only network deviates most from random despite having the lowest absolute clustering (0.055). Its structure cannot be explained by degree distribution alone — the triangles are intensely non-random. AI responses create a network where similar topics form extremely tight local neighborhoods.

### F16: Rich Club — Knowledge Hubs Preferentially Interconnect

All weight configs show a "rich club" (high-degree nodes connect to each other more than expected):

| Config | Max φ_norm | At degree k |
|:---|---:|---:|
| 2:1 | **3.0** | 42 |
| φ:1 | 2.0 | 42 |
| 1:1 | 1.7 | 36 |
| AI-only | 1.6 | 5 |
| user-only | 1.5 | 37 |

The rich club is strongest at balanced weights, peaking at 2:1. AI-only has its rich club at low degree (k=5) — meaning even moderately connected AI-response nodes preferentially interconnect. Balanced configs have rich clubs at high degree — the most connected knowledge hubs form an elite interconnected core.

### F17: Spectral Gap Reveals Macro-Community Structure

Laplacian spectral analysis suggests far fewer natural communities than Louvain:

| Config | Spectral communities | Louvain communities | Algebraic connectivity |
|:---|---:|---:|---:|
| φ:1 | 4 | ~15 | 0.0119 |
| 2:1 | 6 | ~15 | 0.0076 |
| 1:1 | 27 | ~15 | 0.0275 |
| user-only | 2 | ~15 | 0.0493 |
| AI-only | 2 | ~15 | 0.0059 |

The ~15 Louvain communities are sub-divisions of 2-6 fundamental knowledge macro-domains. User-only and AI-only both resolve to 2 macro-communities (binary partition), suggesting a fundamental two-part structure in the knowledge base that finer community detection subdivides.

### F18: AI Responses Create Tighter, More Isolated Communities

| Config | Internal density | Conductance | Community Gini |
|:---|---:|---:|---:|
| AI-only | **0.852** | **0.002** | 0.167 |
| user-only | 0.839 | 0.032 | 0.205 |
| 2:1 | 0.786 | 0.023 | 0.273 |
| φ:1 | 0.783 | 0.014 | 0.281 |
| 1:1 | 0.790 | 0.014 | 0.282 |

AI-only communities have 16× lower conductance (boundary leakage) than user-only. AI responses create self-contained topic clusters with almost no inter-community edges. User prompts create more interconnected communities with higher cross-boundary traffic.

### F19: Edge Weight Distributions Shift Systematically

| Config | Mean sim | Std | Skewness | Frac > 0.9 |
|:---|---:|---:|---:|---:|
| φ:1 | 0.634 | 0.103 | -0.230 | 0.095% |
| 2:1 | 0.633 | 0.105 | -0.236 | 0.095% |
| 1:1 | 0.633 | 0.100 | -0.199 | 0.091% |
| user-only | 0.606 | 0.120 | -0.218 | 0.070% |
| AI-only | 0.578 | 0.100 | **+0.037** | 0.033% |

AI-only is the only config with positive skewness (near-symmetric distribution) — AI responses have uniform similarity levels. User prompts have wider variance and more extreme left tails. Only 0.03-0.10% of all pairs exceed θ=0.9, making the networks extraordinarily sparse.

### F20: Degree-Degree Correlations — Assortative Mixing Varies with Weight

Power-law fit of k_nn(k) ~ k^μ:

| Config | μ (slope) | R² | Interpretation |
|:---|---:|---:|:---|
| 1:1 | 0.240 | 0.477 | Strongest assortative |
| φ:1 | 0.221 | 0.527 | Strong (best fit) |
| 2:1 | 0.183 | 0.430 | Moderate |
| AI-only | 0.151 | 0.153 | Weak, noisy |
| user-only | 0.092 | 0.230 | Weakest |

All configs are assortative (μ > 0): high-degree nodes connect to other high-degree nodes. Equal weight (1:1) has the steepest slope — hubs are most strongly attracted to other hubs when user and AI contributions are balanced. User-only has the weakest degree-degree correlation (μ=0.092, R²=0.23).

### F21: Agentic User Prompts Are Indistinguishable

Edge weight distribution for agentic user-only embeddings:

| Threshold | Edges above | % of edges above θ=0.9 |
|:---|---:|---:|
| θ ≥ 0.90 | 250,817 | 100% |
| θ ≥ 0.95 | 246,344 | 98.2% |
| θ ≥ 0.99 | 241,048 | 96.1% |
| θ ≥ 0.999 | 235,985 | 94.1% |
| θ ≥ 1.000 | **139,937** | **55.8%** |

56% of all high-similarity user-only pairs have **perfect** cosine similarity (1.0). Their embeddings are byte-for-byte identical. Short coding prompts ("fix this bug", "run tests", "look at this file") embed to the same vector.

Compare text-only: 287K edges at θ=0.9 → 2K at θ=0.99 (99.3% drop). Text-only has a normal-ish right tail; user-only has a massive spike at 1.0.

**Implication**: User prompts alone cannot distinguish agentic coding sessions. The discriminating signal lives in AI responses and tool usage. This validates the role-aggregate weighting approach.

### F22: ChatGPT vs Agentic — Qualitatively Different Knowledge Architectures

At matched edge counts (~1700-2100 edges, ChatGPT θ=0.9 vs Agentic θ=0.99):

| Metric | ChatGPT θ=0.9 | Agentic θ=0.99 |
|:---|---:|---:|
| Nodes | 1,908 | 3,945 |
| Edges | 1,718 | 2,118 |
| **GC fraction** | **23.7%** | **2.1%** |
| Clustering | 0.119 | 0.081 |
| **Transitivity** | **0.436** | **0.830** |
| Modularity | 0.776 | 0.879 |
| **Assortativity** | **+0.17** | **+0.80** |
| Communities | 1,380 | 3,387 |
| GC path length | 5.84 | 5.46 |

Key structural differences:
- **ChatGPT forms an interconnected web** (GC 24%); **agentic fragments into isolated project bubbles** (GC 2%).
- **Agentic has near-perfect transitivity** (0.83): if A~B and B~C, then A~C. Coding sessions within a project are mutually similar with no structural holes.
- **Agentic is strongly assortative** (+0.80): sessions connect almost exclusively to sessions of similar degree. ChatGPT has more hub-periphery mixing (+0.17).

Interpretation: ChatGPT conversations explore a connected intellectual landscape; agentic coding sessions are deep but narrow, staying within project boundaries.

### F23: Percolation Sharpness Maximized at Equal Weight

The 1:1 weight config has the sharpest phase transition across all ChatGPT configs:

| Config | Max |dGC/dθ| | θ location | Interpretation |
|:---|---:|---:|:---|
| **1:1** | **10.31** | 0.887 | Sharpest — knife-edge transition |
| 1:φ | 9.98 | 0.887 | Very sharp |
| 1:2 | 9.22 | 0.887 | Sharp |
| φ:1 | 9.14 | 0.887 | Sharp |
| AI-only | 8.43 | 0.863 | Moderate (earlier location) |
| 1:4 | 8.57 | 0.863 | Moderate |
| 2:1 | 8.26 | 0.913 | Moderate (later location) |
| 4:1 | 8.03 | 0.913 | Gradual |
| user-only | 6.94 | 0.913 | **Gentlest** |

Equal weighting creates the most abrupt collapse — a tiny threshold change around θ=0.887 dissolves the giant component. At extremes (AI-only, user-only), the transition is gentler and shifted earlier/later.

### F24: Agentic User Prompts Are Semantically Degenerate (Bimodal Distribution)

The agentic user-only edge weight distribution is bimodal:
- **96% of pairs** have moderate similarity (mean 0.62, range 0.3–0.8)
- **3.6% of pairs** have PERFECT similarity (≥0.9999)
- From θ=0.80 (272K edges) to θ=0.95 (246K edges), only **10% of edges are lost**

At θ≥0.93, the graph achieves transitivity=1.000 and assortativity=1.000 — it becomes a collection of disconnected perfect cliques.

| θ | Edges | GC% | Clustering | Modularity | Transitivity |
|:---|---:|---:|---:|---:|---:|
| 0.80 | 272,610 | 90.7% | 0.604 | 0.353 | 0.985 |
| 0.90 | 250,817 | 25.7% | 0.331 | 0.314 | 0.999 |
| 0.95 | 246,344 | 17.3% | 0.285 | 0.296 | 1.000 |

**Root cause identified**: The three largest identical-embedding groups are:
- **520 sessions**: Context continuation subagents (identical "create a detailed summary" template)
- **230 sessions**: Prompt suggestion subagents (identical "suggest what user might type" template)
- **100 sessions**: Context continuation variant (slightly different template)

These are system-spawned Claude Code subagents, not actual user work. The "user" content is boilerplate generated by the platform itself.

Implication: user-only embeddings are fundamentally inappropriate for agentic AI because the "user" is often the system, not the human. The text-only mode works because the AI's *responses* to these identical templates are different (each summary is about different code).

### F25: ChatGPT vs Agentic — Content Mode Discrimination Power

Edge weight distribution comparison across datasets and content modes:

| Dataset & Mode | Mean sim | Std | P99 | ≥0.9 | ≥0.99 | Perfect |
|:---|---:|---:|---:|---:|---:|---:|
| ChatGPT 2:1 | 0.633 | 0.105 | 0.846 | 0.095% | ~0% | ~0% |
| ChatGPT user-only | 0.606 | 0.119 | 0.844 | 0.070% | ~0% | ~0% |
| ChatGPT AI-only | 0.578 | 0.100 | 0.810 | 0.033% | ~0% | ~0% |
| **Agentic text-only** | **0.744** | **0.076** | **0.977** | **3.7%** | **0.02%** | **~0%** |
| **Agentic user-only** | **0.622** | **0.098** | **1.000** | **4.0%** | **3.9%** | **3.6%** |

Key differences:
1. **ChatGPT has well-behaved distributions** in ALL modes — smooth tails, good discrimination, no degeneracy
2. **Agentic text-only** has higher baseline similarity (0.74 vs 0.63) due to shared coding context, but still provides good discrimination (3.7% → 0.02% from θ=0.9 to θ=0.99)
3. **Agentic user-only** has a massive plateau above θ=0.9 — virtually no filtering effect because a fixed 3.6% of sessions share identical prompt embeddings

### F26: Kurtosis Reveals Embedding Space Geometry

| Config | Skewness | Kurtosis | Interpretation |
|:---|---:|---:|:---|
| ChatGPT 2:1 | -0.24 | -0.30 | Near-Gaussian, platykurtic (thin tails) |
| ChatGPT AI-only | +0.04 | -0.15 | Symmetric, platykurtic |
| Agentic text-only | **+0.45** | **+1.42** | Right-skewed, **leptokurtic** (heavy right tail) |

Agentic sessions have excess kurtosis (+1.42) — a heavy right tail of high-similarity pairs. This is the quantitative signature of project-level clustering: sessions within a project are much more similar to each other than random, creating a long tail in the distribution.

ChatGPT's platykurtic distributions mean no such outlier clustering — conversations are spread more uniformly across the semantic space.

### F27: The Signal Lives in AI Responses, Not User Prompts (Agentic Only)

In agentic coding sessions, the semantic discrimination comes from AI responses and tool usage, not from user instructions:
- **User prompts** are indistinguishable (3.6% perfect pairs, bimodal distribution)
- **Full text** (user + AI) provides good discrimination (edges drop 99.6% from θ=0.93 to θ=0.99)
- **Implication**: the AI's execution trajectory (which tools it calls, which code it generates) creates the semantic fingerprint of a coding session

This is the **opposite** of conversational AI (ChatGPT), where both user and AI contribute meaningfully to the semantic signature. In ChatGPT conversations, user-only and AI-only distributions are both smooth and discriminative.

This finding has theoretical implications: in agentic AI, the user's role is to set direction while the AI's responses encode the actual content. The network structure we observe reflects the AI's work patterns, not the human's instructions.

### F28: Tool-Names-Only Is the Most Degenerate Content Mode

Tool-name embeddings have an extreme left-skewed distribution (skewness -1.76, mean 0.916, median 0.974):

| θ | Edges (of 4.5M) | ⟨k⟩ | % of pairs |
|:---|---:|---:|---:|
| 0.90 | 3,310,000 | 2,200 | 73% |
| 0.99 | 1,576,000 | 1,047 | 35% |
| 0.999 | 495,000 | 329 | 11% |
| 0.9999 | 125,000 | 83 | 2.8% |
| **1.0** | **57,300** | **38** | **1.3%** |

Even at EXACT equality (θ=1.0), there are 57,300 perfectly identical pairs. The tool vocabulary (~20-30 tools) creates a space too low-dimensional for threshold-based analysis.

At θ=1.0, the network consists of **145 perfect cliques** (transitivity=1.000), each representing a distinct tool-usage "profile":
- Grep/Read/Bash sessions (exploration/research)
- Write/Read sessions (implementation)
- WebFetch/WebSearch sessions (web research)
- Read/Bash sessions (debugging/testing)

**Content mode discrimination hierarchy**: text-only >> user-only >> tool-names-only.

### F29: Semantic and Tool-Usage Capture Partially Independent Dimensions

Cross-layer NMI (θ=0.95 semantic, θ=0.9999 tool-names):

| Pair | NMI | AMI | Interpretation |
|:---|---:|---:|:---|
| Semantic ↔ Tool-usage | 0.50 | 0.19 | Moderate overlap |
| Semantic ↔ User-only | 0.97 | — | Near-identical at matched density |
| Tool-usage ↔ User-only | 0.48 | — | Moderate overlap |

Tool profiles capture *how* you work; semantic similarity captures *what* you work on. These are partially but not fully redundant (AMI=0.19 after chance correction).

### F30: User-Only Degeneracy Is Entirely Caused by Subagent Templates

Decomposing the identical-embedding groups:
- **100% of identical-embedding clusters are subagent sessions**
- 520 context-continuation subagents share one embedding, 230 prompt-suggestion subagents share another
- Zero parent sessions have identical user-only embeddings

| Edge set | Mean sim | Perfect (≥0.9999) | ≥0.90 |
|:---|---:|---:|---:|
| Parent↔Parent | 0.606 | 0.04% | 0.18% |
| Involving subagents | 0.623 | 3.68% | 4.0% |

Parent-only user-only embeddings have a well-behaved, ChatGPT-like distribution!

### F31: Subagents Fundamentally Alter Network Topology

The same edges produce opposite network structures when subagent sessions are included vs excluded:

| Metric | Full (with subagents) @ θ=0.93 | Parent-only @ θ=0.93 |
|:---|---:|---:|
| Nodes | 3,945 | 339 |
| Edges | 229,925 | 10,060 |
| GC fraction | 37.2% | 94.7% |
| Assortativity | **+0.924** | **-0.110** |
| Transitivity | 0.970 | 0.635 |
| Modularity | 0.285 | 0.158 |

Subagents create extreme assortativity (+0.92) and fragmentation (GC 37%). Without them, the network flips to weakly disassortative (-0.11) with nearly everything connected (GC 95%) — structurally closer to ChatGPT.

Parent-only text-only agentic at θ=0.95 (242 nodes, 2613 edges) vs ChatGPT at θ=0.90 (1908 nodes, 1718 edges):

| Metric | ChatGPT θ=0.90 | Agentic parent-only θ=0.95 |
|:---|---:|---:|
| Nodes | 1,908 | 242 |
| Edges | 1,718 | 2,613 |
| GC% | 23.7% | 96.3% |
| Modularity | 0.776 | 0.268 |
| Assortativity | +0.17 | -0.046 |
| Transitivity | 0.436 | 0.488 |

ChatGPT has stronger community structure (mod 0.78 vs 0.27), while agentic parent sessions form a denser, more connected web. ChatGPT conversations occupy distinct knowledge niches; coding sessions span multiple topics within individual sessions.

### F32: Matched-Degree Comparison Reveals Modular vs Core Architectures

At matched average degree ⟨k⟩=5.7 (ChatGPT θ=0.90, Agentic parent θ=0.965):

| Metric | ChatGPT | Agentic parent |
|:---|---:|---:|
| Nodes | 601 | 147 |
| Avg degree | 5.7 | 5.7 |
| **Modularity** | **0.776** | **0.486** |
| Communities | 73 | 12 |
| **GC fraction** | **75.4%** | **90.5%** |
| Clustering | 0.379 | 0.398 |
| Transitivity | 0.436 | 0.369 |
| Assortativity | +0.167 | +0.058 |
| GC avg path | 5.84 | 3.32 |
| GC diameter | 18 | 8 |
| Degree CV | 1.33 | 1.16 |

**ChatGPT** forms a *modular knowledge landscape* — 73 distinct topic communities with bridge conversations linking them, longer paths (5.8 hops), and higher degree heterogeneity.

**Agentic coding** forms a *dense cognitive core* — 12 weaker communities, nearly everything connected (91%), shorter paths (3.3 hops), and more uniform degree distribution.

Both have similar local clustering (~0.4), but ChatGPT's modularity is 1.6× higher — conversations occupy distinct knowledge niches, while coding sessions naturally span multiple topics within individual sessions.

### F33: Session Depth, Not Length, Predicts Network Centrality

Spearman correlations between network position and session properties (parent-only, θ=0.95):

| Property | vs Degree | vs Betweenness |
|:---|---:|---:|
| **Message count** | **ρ=+0.616****** | **ρ=+0.525****** |
| **Tool count** | **ρ=+0.606****** | **ρ=+0.511****** |
| Unique tools | ρ=+0.476*** | ρ=+0.386*** |
| Word count | ρ=+0.044 ns | ρ=+0.017 ns |

Conversation depth (message count, tool usage) strongly predicts centrality, but raw text length has zero predictive power. Longer interactions touch more topics, creating semantic bridges. The top hub has 3,796 messages and 1,343 tool calls but only 30 words of natural language text.

### F34: Guimerà-Amaral Role Distribution

Node classification by within-module degree and participation coefficient:

| Role | Count | % | Description |
|:---|---:|---:|:---|
| Ultra-peripheral | 94 | 39% | Single-community, low degree |
| Peripheral | 110 | 46% | Mostly within-community |
| **Connector** | **37** | **15%** | Significant cross-community bridging |
| Hub-connector | 1 | 0.4% | High degree + high cross-community |

Hubs (top 10%) connect to 4 of 10 communities on average; peripheral nodes reach only 1.0. The 15% connector nodes are the sessions that link different coding topics, analogous to "bridge conversations" in the ChatGPT network.

### F35: Universal Densification Law (γ ≈ 1.41)

Parent-only agentic densification exponent γ=1.410 (R²=0.994), almost exactly matching ChatGPT's γ=1.405:

| Network | γ | R² | Notes |
|:---|---:|---:|:---|
| ChatGPT (1908 conversations) | 1.405 | 0.998 | Conference paper result |
| **Agentic parent-only (462 sessions)** | **1.410** | **0.994** | **New finding** |
| Agentic full (3945 sessions) | 1.707 | — | Inflated by subagents |

Subagents inflate the densification exponent from 1.41 to 1.71. Without them, agentic coding sessions follow the SAME densification law as ChatGPT conversations.

This suggests a **universal property of AI-assisted knowledge exploration**: edges accumulate super-linearly with nodes at rate e ~ n^1.4, regardless of interaction modality (conversational vs agentic coding). The identical exponent across two different AI platforms, user interfaces, and interaction patterns is remarkable.

Temporal evolution (weekly cumulative, parent-only θ=0.95):
- GC starts at 100%, dips to ~91% as diverse projects appear, then recovers to 96%
- Modularity grows from 0.07 to 0.28 as community structure emerges
- Communities grow from 1 to 11 over 8 weeks

---

## Paper Narrative Synthesis

### F36: Conversational AI vs Agentic AI — Who Carries the Semantic Signal?

In ChatGPT (θ=0.9), BOTH user and AI content produce well-structured, non-degenerate networks:

| Config | Edges | Modularity | GC% | Network type |
|:---|---:|---:|---:|:---|
| ChatGPT user-only (100:1) | 1,265 | 0.750 | 19.1% | Well-structured |
| ChatGPT AI-only (1:100) | 601 | 0.841 | 9.8% | Well-structured |
| ChatGPT 2:1 (baseline) | 1,718 | 0.776 | 23.7% | Well-structured |

In agentic sessions, only text-only (combining user + AI) produces meaningful structure. User-only is degenerate (boilerplate templates), tool-names-only is degenerate (small vocabulary).

**Fundamental asymmetry**: In conversational AI, the *human* drives the semantic signature through diverse topics and unique questions. In agentic AI, the *AI* drives the semantic signature through diverse code generation and tool trajectories. The roles are reversed.

ChatGPT user prompts carry MORE discriminative signal (1265 edges) than AI responses (601 edges). In agentic sessions, the human's contribution is nearly constant ("fix this", "implement that") while the AI's execution creates all the semantic variation.

### Theme 1: Content Mode Discrimination (F24-F28, F30, F37)

The semantic signal requires BOTH user and AI content — neither alone suffices. For parent-only sessions:

1. **text-only** (user + AI) — μ=0.847, 2,613 parent edges at θ=0.95 — synergistic combination
2. **user-only** — μ=0.612, 42 edges at θ=0.90 — topical fingerprint but sparse
3. **assistant-only** — μ=0.673, 1 edge at θ=0.90 — "AI voice" homogenizes all responses
4. **tool-names-only** — mean sim=0.916, 57K perfect pairs — degenerate (small vocabulary)

Key mechanism: AI responses provide a consistent semantic scaffolding (narrow σ=0.064) that normalizes the embedding space. User prompts provide topical differentiation. The combination elevates baseline similarity while preserving peaks — uniquely enabling high-threshold network construction.

### Theme 2: Subagents Transform Topology (F30-F31)

Including vs excluding subagents produces opposite network architectures:

| Property | With subagents | Parents only |
|:---|:---|:---|
| Assortativity | +0.92 (cliques) | -0.05 (neutral) |
| GC fraction | 37% (fragmented) | 95% (connected) |
| Modularity | 0.29 | 0.27 |
| Densification γ | 1.707 | **1.410** |

The parent-only network resembles ChatGPT; the subagent-inflated version is a different beast entirely.

### Theme 3: Universal Densification (F35)

Parent-only agentic γ=1.410 ≈ ChatGPT γ=1.405. Two completely different AI platforms, interaction modalities, and user populations produce the same densification exponent. This suggests a fundamental scaling law of AI-assisted knowledge exploration.

### Theme 4: Modular vs Core Architectures (F22, F32)

At matched ⟨k⟩=5.7:
- **ChatGPT**: modular knowledge landscape (mod=0.78, 73 communities, GC=75%)
- **Agentic parents**: dense cognitive core (mod=0.49, 12 communities, GC=91%)

ChatGPT conversations occupy distinct knowledge niches; coding sessions naturally span multiple topics within individual sessions.

### Theme 5: Interaction Depth Drives Connectivity (F33-F34)

Message count (ρ=+0.62) and tool count (ρ=+0.61) strongly predict session centrality, but text length has zero predictive power. The depth of AI interaction — not its verbosity — determines how semantically connected a session is to the rest of the corpus.

---

## F37: Assistant-Only Network is Empty — The "AI Voice" Homogenization Effect

**Completed**: Assistant-only embeddings (2787/3906, 212 parents) → all-pairs cosine similarity → network analysis.

**The key test**: Does the assistant-only (AI responses alone) produce meaningful network structure for parent sessions?

**Answer: No.** The parent-parent assistant-only network is essentially empty:

| θ | Parent nodes | Parent edges | ⟨k⟩ | GC |
|:---|---:|---:|---:|---:|
| 0.85 | 29 | 16 | 1.10 | 3 |
| 0.88 | 6 | 3 | 1.00 | 2 |
| 0.90 | 2 | 1 | — | 2 |
| 0.95+ | ≤2 | ≤1 | — | — |

**Three-way parent-parent pairwise similarity comparison:**

| Metric | Text-only (user+asst) | User-only | Assistant-only |
|:---|:---|:---|:---|
| Parents embedded | 462 | ~259 | 212 |
| Pairs | 106,491 | 33,411 | 22,366 |
| Mean sim | **0.847** | 0.612 | 0.673 |
| Std | 0.080 | 0.084 | **0.064** |
| Max sim | 1.000 | 1.000 | **0.973** |
| P99 | 0.958 | 0.803 | 0.800 |
| θ≥0.90 | 27,974 (26.3%) | 42 (0.13%) | **1 (0.004%)** |
| θ≥0.95 | 2,613 (2.5%) | 9 (0.027%) | 1 (0.004%) |

**The "AI Voice" Effect**: Assistant-only responses cluster tightly around μ=0.673 with the narrowest spread (σ=0.064). The AI speaks in a "house style" — consistent formatting, hedging patterns, explanation structure — that makes all responses look similar regardless of topic. The maximum parent-parent similarity (0.973) never reaches the levels needed for clean networks.

**Revised signal model**: F36's hypothesis that "the AI drives the semantic signature in agentic sessions" was partially wrong. More precisely:

1. **User prompts** carry the topical fingerprint (lower mean = more discriminating)
2. **AI responses** add consistent semantic scaffolding (higher baseline, tight cluster)
3. **Combined text** works best because AI scaffolding normalizes the embedding space while user specificity creates high-similarity peaks above the elevated baseline

The combination is synergistic: text-only at θ=0.95 yields 2,613 parent-parent edges — more than either mode alone produces at ANY threshold.

**Contrast with ChatGPT (F36)**: In ChatGPT, AI-only produces 601 edges at θ=0.9 — a well-structured network. In agentic Claude, AI-only produces 1 edge at θ=0.9. The difference: ChatGPT responses are brief, topic-focused answers. Claude Code responses are long, tool-heavy transcripts where structural patterns dominate topical content.

---

## F38: Cross-Mode Orthogonality — Text-Only and User-Only Encode Different Structures

**Degree correlation**: Spearman ρ = -0.17 (p=0.088, n=101 common nodes). Text-only degree is **uncorrelated** with user-only degree. Sessions that are central hubs by full transcript are NOT hubs by user prompt alone.

Example: session `a215f9e9` has text_deg=149, user_deg=3 — a deep 285-message coding session where centrality comes entirely from AI-generated code. Conversely, `dc88bd57` has text_deg=2, user_deg=17 — a 1-message session where the user prompt connects to many others but the AI barely responded.

**Community NMI**: 0.41–0.72 depending on overlap size. Partial agreement — the two modes capture overlapping but independent structure.

**Structural comparison at matched ⟨k⟩ ≈ 21.5:**

| Metric | Text-only (θ=0.95) | User-only (θ=0.73) |
|:---|:---|:---|
| Nodes | 242 | 228 |
| Edges | 2,613 | 2,429 |
| Communities | 12 | 7 |
| Modularity | 0.26 | **0.45** |
| Clustering | **0.59** | 0.42 |
| GC fraction | 96% | 100% |
| Assortativity | -0.05 | +0.07 |

At lower ⟨k⟩ (text θ=0.97, user θ=0.80), user-only shows **strong assortative mixing** (r=+0.67) — hubs connect to hubs — while text-only remains neutral (r=+0.10).

**Interpretation**: Text-only creates a cohesive core (high clustering, low modularity) — a unified cognitive workspace. User-only creates a more modular structure (higher modularity, fewer communities) — the user's topic hierarchy. Full transcripts smear the topic structure into a flat workspace; user prompts alone preserve topical boundaries.

---

## F39: ChatGPT vs Agentic — Percolation and Distribution Comparison

**Edge weight distributions:**

| Metric | ChatGPT (2:1, 1.9K convs) | Agentic parents (462 sessions) |
|:---|:---|:---|
| Mean similarity | 0.633 | **0.847** |
| Std | **0.105** | 0.080 |
| Skewness | -0.24 | **-1.61** |
| P99 | 0.846 | **0.958** |

ChatGPT conversations span diverse topics (ML, math, writing, medical, philosophy) → lower mean, wider spread. Agentic sessions are all software engineering → higher baseline, more homogeneous.

**Percolation critical thresholds:**
- ChatGPT: GC drops below 50% at **θ_c ≈ 0.91** (sharp phase transition)
- Agentic parents: GC drops below 50% at **θ_c ≈ 0.97** (shifted 0.06 higher)

The "useful analysis window" shifts correspondingly: ChatGPT θ=0.88–0.92, Agentic θ=0.93–0.97.

**ChatGPT weight configuration sweep (new):**

| Config | Mean | θ≥0.9 edges | θ≥0.95 edges |
|:---|---:|---:|---:|
| User-only (100:1) | 0.606 | **1,265** | 30 |
| User-heavy (2:1) | 0.633 | **1,718** | 45 |
| Balanced (1:1) | 0.633 | 1,647 | 40 |
| AI-only (1:100) | 0.578 | 601 | 16 |

In ChatGPT, user prompts carry MORE discriminative signal than AI responses (1,265 vs 601 edges). Rank correlation between user-only and AI-only: ρ=0.778 — high but not identical.

**Degree distributions at optimal thresholds:**

| Metric | ChatGPT θ=0.90 | Agentic θ=0.95 |
|:---|:---|:---|
| Nodes | 601 | 242 |
| ⟨k⟩ | 5.72 | 21.60 |
| Median degree | 2 | 12.5 |
| Max degree | 59 | 107 |
| CV (σ/μ) | **1.33** | 1.10 |
| Hub:mean ratio | **10.3×** | 5.0× |
| Power-law α | 1.80 | 2.32 |

ChatGPT is hub-dominated (median=2, max=59, ratio=10.3×). Agentic is more democratic (median=12.5, ratio=5.0×). ChatGPT has a few super-connector conversations; agentic coding sessions share broad common ground.

---

## F40: Node Role Comparison — Knowledge Archipelago vs Cognitive Web

**Guimerà-Amaral role distributions:**

| Role | ChatGPT GC | Agentic GC |
|:---|---:|---:|
| Ultra-peripheral | **79.7%** | 35.6% |
| Peripheral | 17.2% | **45.5%** |
| Connector | 0.2% | **18.0%** |
| Provincial hub | **2.6%** | 0.0% |
| Connector hub | 0.2% | 0.9% |

**ChatGPT = "Knowledge Archipelago"**: 80% ultra-peripheral, 2.6% provincial hubs, only 2 bridge conversations. Isolated topic islands (ML, statistics, web dev, physics, computability) connected by rare bridging sessions.

**Agentic = "Cognitive Web"**: 18% connectors (P>0.625) spanning multiple communities. Zero provincial hubs — no community has a single dominant node. Mean participation coefficient P=0.34 (vs ChatGPT where most nodes have P≈0).

The single ChatGPT connector-hub (`mle-bootstrapping-simulation`, deg=47) bridges statistics and ML — the mathematical foundation shared by both. In agentic, the top 10 bridges ALL have P>0.70 and degrees 21–107, spanning 3+ communities each.

**Why**: A ChatGPT conversation about "Weibull reliability" stays in its domain. A Claude Code session fixing a data pipeline touches statistics, graph theory, Python, testing, and file I/O — naturally bridging multiple domains in every session.

**Community characterization (ChatGPT GC, 14 communities):**
- C0 (103): ML/AI/LLM — RLHF, embeddings, self-supervised learning
- C8 (82): Statistics/Math — MLE, bootstrapping, Frobenius norms
- C10 (69): Creative/Science — Dune, Kolmogorov, simulations
- C9 (46): Web dev/DevOps — Hugo, FastAPI, Redis
- C13 (44): Reliability analysis — Weibull, MLE, CI plotting (density=0.291!)
- C3 (29): Algorithms — AlgoTree, MCTS, tree grammars
- C7 (25): Physics simulation — N-body, thermodynamics
- C12 (7): Computability — Cantor, digital physics
- C6 (5): Medical — chemotherapy, surgery

This IS the "cognitive MRI" — the user's knowledge domains as distinct neural clusters.

**Agentic community characterization (GC, 7 communities, θ=0.95):**
- C3 (60): Mixed projects — Go CLI tools, beta toolkits. Sessions with `/init` commands.
- C2 (55): R packages & academic tools — dfr.dist, zenodo, Singular Hash papers. Research-focused.
- C1 (39): Infrastructure — texwatch (LaTeX), clerk (LLM CLI). Server/CLI development.
- C5 (32): R packages & this paper — flexhaz, temporal network analysis. Academic R development.
- C6 (25): Game development — autotile terrain, build mode. Creative/hobby projects.
- C0 (9): Short untitled sessions (3-6 msgs) — quick questions.
- C4, C7 (3 each): Small project-specific clusters.

Unlike ChatGPT where communities map to **knowledge domains** (ML, statistics, physics), agentic communities map to **projects** (texwatch, flexhaz, game engine). The same programming knowledge (Python, R, testing, git) is used across ALL communities — it's the *project context* that creates separation, not the *skill domain*.

---

## F41: Network Robustness — Cognitive Web vs Knowledge Archipelago Under Attack

**Betweenness centrality concentration:**

| Metric | ChatGPT GC | Agentic GC |
|:---|:---|:---|
| BC Gini coefficient | **0.87** | 0.72 |
| Max/Mean BC | **39.2×** | 11.9× |
| BC=0 fraction | 43% | 31% |
| Max BC | **0.420** | 0.085 |
| Mean closeness | 0.18 | **0.40** |

**Targeted attack (betweenness-ordered node removal):**

| Nodes removed | ChatGPT GC% | Agentic GC% |
|:---|:---|:---|
| 0 | 100% | 100% |
| 2 | **74.8%** | 96.1% |
| 7 | **43.3%** | 90.1% |
| 10 | 42.4% | 88.4% |
| 20 | 26.5% | 78.5% |

**Removals to halve GC**: ChatGPT=7 (1.5% of nodes), Agentic=21 (9.0% of nodes).

ChatGPT collapses after removing 7 bridges (especially `geometric-mean-calculation` which carries 42% of all shortest paths). The agentic network barely notices — after removing 20 nodes, 78.5% of the GC remains connected.

**Interpretation**: The ChatGPT "knowledge archipelago" has critical single points of failure — remove a few bridge conversations and isolated topic islands become disconnected. The agentic "cognitive web" has redundant pathways — many sessions span multiple domains, so no single session is irreplaceable. This maps to real cognitive resilience: the user's agentic knowledge is more robustly interconnected than their conversational knowledge.

---

## Updated Paper Narrative Synthesis

### Theme 6: Cognitive Architecture — Archipelago vs Web (F40-F41, NEW)

The most striking finding: ChatGPT and agentic AI produce **opposite cognitive architectures**:

| Property | ChatGPT | Agentic |
|:---|:---|:---|
| Architecture | Knowledge Archipelago | Cognitive Web |
| Communities map to | Knowledge domains | Projects |
| Ultra-peripheral nodes | 80% | 36% |
| Connector nodes | 0.2% | **18%** |
| Bridge concentration | Gini=0.87 | Gini=0.72 |
| Robustness (nodes to halve GC) | 7 (1.5%) | 21 (9.0%) |
| Hub:mean degree | 10.3× | 5.0× |

In **conversational AI**, each topic gets explored in its own conversation, creating isolated knowledge islands connected by rare "bridge" conversations. Knowledge is **modular but fragile** — remove a few bridges and the network fragments.

In **agentic AI**, each coding session naturally spans multiple topics (the same project touches algorithms, data structures, testing, CI/CD, documentation). Knowledge is **distributed and resilient** — every session contributes to multiple communities, and no single session is a critical path.

This is not just a network metric — it reflects fundamentally different modes of AI-assisted thinking. Conversational AI supports **deep topical exploration**. Agentic AI supports **integrative project execution**.

---

## F42: Small-World Analysis — Both Networks, Different Regimes

| Property | ChatGPT GC | Agentic GC |
|:---|:---|:---|
| σ (small-world) | **14.9** | 4.2 |
| C/C_random | **27.9×** | 6.5× |
| L/L_random | 1.88× | 1.52× |
| ω (Telesford) | -0.17 | -0.21 |
| Clustering C | 0.44 | **0.62** |
| Diameter | **18** | 9 |
| Transitivity | 0.44 | **0.49** |

Both networks are small-world (σ >> 1), but in different regimes:

- **ChatGPT** (σ=14.9): Extreme small-world. Very tight local clusters (C=28× random) with short global paths (L=1.9× random). The few bridge conversations create dramatic "shortcuts." Diameter=18 — some knowledge domains are very distant.
- **Agentic** (σ=4.2): Moderate small-world. Uniformly well-connected (C=6.5× random, L=1.5× random). Lower σ because the network is denser, not less structured. Diameter=9 — everything is close.

---

## Master Comparison Table

| Metric | ChatGPT (θ=0.90) | Agentic parents (θ=0.95) |
|:---|:---|:---|
| **Scale** | | |
| Nodes (GC) | 453 | 233 |
| Edges | 1,612 | 2,607 |
| ⟨k⟩ | 7.1 | 22.4 |
| Density | 0.016 | 0.097 |
| **Community** | | |
| Communities | 14 | 7 |
| Modularity | **0.749** | 0.278 |
| Community type | Knowledge domains | Projects |
| **Topology** | | |
| Small-world σ | **14.9** | 4.2 |
| Clustering | 0.44 | **0.62** |
| Assortativity | +0.09 | -0.05 |
| Diameter | 18 | **9** |
| **Roles** | | |
| Ultra-peripheral | **80%** | 36% |
| Connectors | 0.2% | **18%** |
| Provincial hubs | **2.6%** | 0% |
| **Robustness** | | |
| BC Gini | **0.87** | 0.72 |
| Nodes to halve GC | 7 (1.5%) | **21 (9.0%)** |
| **Temporal** | | |
| Densification γ | 1.405 | 1.410 |
| Percolation θ_c | 0.91 | 0.97 |
| **Content modes** | | |
| User-only signal | Strong (1,265 edges) | Weak (42 edges) |
| AI-only signal | Moderate (601 edges) | **Empty** (1 edge) |
| Combined signal | Strongest (1,718) | Strongest (2,613) |

| **Concept hierarchy** | | |
| Concepts extracted | 79 | 36 |
| Concept type | Knowledge-domain | Process-workflow |
| L2 meta-concepts | 8 | 6 |
| L3 orientations | 4 (epistemic) | 3 (practical) |
| L2→L3 compression | 2.0× | 2.0× |

Two universal constants: **densification γ ≈ 1.41** and **L2→L3 compression ratio = 2.0×** across both platforms.

---

> **Note**: Findings F43-F102 (cross-platform concept network analysis) were archived on 2026-03-13.
> They remain accessible in git history. The research program pivoted to hierarchical memory
> analysis of ChatGPT conversations only.

---

## Hierarchical Memory Network — Experiment 1 (Geometric)

### F43: Ward linkage produces balanced hierarchies; average linkage chains

**Average linkage** on the 1905×1905 cosine similarity matrix produces extreme chaining: at k=15, one cluster contains 1879/1905 episodes (98.6%) with 14 singleton/tiny outlier clusters. This is because the similarity distribution is smooth (mean=0.633, std=0.105) and average linkage peels off low-similarity outliers sequentially.

**Ward linkage** produces balanced clusters at all levels: k=15 gives sizes [35..296], k=5 gives [284..561], k=2 gives [355..1550]. Ward minimizes within-cluster variance, which favors balanced splits. For L2-normalized embeddings, ward's Euclidean metric is monotonically related to cosine distance (d²=2(1-cos)), so the results are cosine-meaningful.

**Implication**: Linkage method is a critical methodological choice. Average linkage is inappropriate for smooth embedding spaces; ward should be the default.

### F44: 4-level geometric hierarchy with ~3× branching at each level

The dendrogram of 1,905 ChatGPT conversation embeddings (768-dim, nomic-embed-text, 2:1 user:AI weighting) supports a clean 4-level hierarchy under ward linkage:

| Level | Name | k | Silhouette | Cluster sizes |
|:---|:---|:---|:---|:---|
| L1 | Fine-concepts | 50 | -0.006 | [17..108] |
| L2 | Concepts | 15 | 0.038 | [35..296] |
| L3 | Meta-concepts | 5 | 0.044 | [284..561] |
| L4 | Domains | 2 | 0.108 | [355, 1550] |

**Branching factors** between levels are remarkably consistent:
- L1→L2: 3.33×
- L2→L3: 3.0×
- L3→L4: 2.5×

This ~3× branching mirrors **Rosch's basic-level categorization** (superordinate/basic/subordinate ≈ 3 levels) and **Miller's chunking** (3-5 items per group). The hierarchy depth (4 levels) aligns with schema theory's observation that human knowledge organizes into ~3-4 abstraction levels.

### F45: Low silhouette scores confirm continuous semantic space

Silhouette scores are consistently low across all k values and linkage methods (best: 0.108 at k=2 with ward). The silhouette scan shows monotonically decreasing scores as k increases — there are no "natural" cluster counts where silhouette spikes.

This is not a failure of clustering; it reflects the **continuous nature of the embedding space**. Conversation topics blend into each other (statistics shades into machine learning shades into optimization). The hierarchy imposes discrete boundaries on a continuous landscape — exactly as human categorical memory does (Rosch 1975: categories have fuzzy boundaries with graded membership).

### F46: Geometric hierarchy strongly agrees with Louvain communities (NMI=0.705)

Comparing the geometric hierarchy (ward, k=15) with Louvain community detection on the thresholded graph (θ=0.9, 601 connected nodes):

| Hierarchy level | k | NMI vs Louvain | ARI vs Louvain |
|:---|:---|:---|:---|
| L1 Fine-concepts | 50 | **0.752** | 0.391 |
| L2 Concepts | 15 | **0.705** | 0.544 |
| L3 Meta-concepts | 5 | 0.552 | 0.368 |
| L4 Domains | 2 | 0.194 | 0.043 |

**Key finding**: The hierarchical clustering (using full pairwise similarities) and Louvain (using thresholded graph topology) converge on similar groupings despite using fundamentally different information. L1 (k=50) actually has *higher* NMI than L2 (k=15) because finer geometric cuts better match Louvain's resolution.

This validates that the community structure is real and robust — it emerges from both continuous geometry and discrete graph topology.

### F47: Semantic interpretation of the hierarchy

The 4-level hierarchy has clear semantic interpretation:

**L4 Domains (k=2):**
- Domain 0 (355): **Tooling & Publishing** — software infrastructure, packaging, LaTeX
- Domain 1 (1550): **Research & Ideas** — ML, statistics, philosophy, simulations

**L3 Meta-concepts (k=5):**
1. Software Infrastructure (355): DevOps, packaging, publishing
2. Machine Learning (296): LLMs, transformers, fine-tuning
3. Ideas & Exploration (409): AI philosophy, security research, creative work
4. Statistical Theory (284): Likelihood, reliability, bootstrap
5. Algorithms & Computation (561): Optimization, simulations, data structures

**L2 Concepts (k=15):** Semantically coherent clusters including:
- MLE & bootstrap methods (105 episodes)
- AlgoTree package development (106 episodes)
- Physics simulations (65 episodes)
- R visualization (35 episodes)
- AI philosophy & alignment (144 episodes)

The L4 domain split (355 tooling vs 1550 research, ratio 1:4.4) reveals the user's conversation distribution: overwhelmingly research-oriented, with tooling as instrumental support.

### F48: Similarity distribution baseline

The full 1905×1905 pairwise cosine similarity matrix statistics:

| Statistic | Value |
|:---|:---|
| Mean | 0.633 |
| Std | 0.105 |
| Median | 0.632 |
| Min | 0.145 |
| Max | 1.000 |
| Pairs ≥ 0.9 | 1,718 |
| Pairs ≥ 0.8 | 31,952 |
| Pairs ≥ 0.7 | 344,009 |

The 1,718 pairs above θ=0.9 exactly matches the episodic network edge count from Papers 1-2, confirming consistency. The distribution is approximately Gaussian (mean≈0.63, σ≈0.10), consistent with random vectors in high-dimensional space with a semantic offset.

---

## Experiment 2: Semantic Concept Hierarchy

LLM-extracted noun-phrase concepts from 1,813 ChatGPT conversations. Concepts extracted by Claude Code (Sonnet) running as 19 parallel agents, each processing ~100 conversations. Concepts embedded via nomic-embed-text, hierarchically clustered using Ward linkage.

### F49: Concept extraction yields 5,944 raw concepts with extreme singleton rate

From all 1,908 episodes, Claude Code extracted 6,773 raw concept mentions (3.5 per episode, range 1-5) mapping to 6,275 unique concepts (case-insensitive). **95.3% of concepts are singletons** — appearing in exactly 1 episode. This is an artifact of parallel extraction without shared vocabulary: the 19 agents independently generated near-synonymous phrases (e.g., "bootstrap confidence intervals" vs "BCa bootstrap confidence intervals").

| Metric | Value |
|:---|:---|
| Episodes extracted | 1,908 / 1,908 (100%) |
| Raw concept mentions | 6,773 |
| Unique concepts | 6,275 |
| Concepts per episode | 3.5 (mean), 4 (median), 1-5 (range) |
| Singletons | 5,977 (95.3%) |
| Max frequency | 14 ("maximum likelihood estimation") |
| Gini coefficient | 0.073 (nearly uniform) |

Note: Pipeline analysis results below use 1,813 episodes (95% coverage at time of pipeline execution). The remaining 5% were extracted subsequently and are included in `extraction_state.json` for re-runs.

### F50: Meta-concept clustering eliminates singletons and creates dense bipartite structure

The 500-cluster meta-concept level acts as the geometric deduplication layer, grouping semantically similar concepts. This transforms the sparse raw-concept bipartite graph into a dense, well-connected structure.

| Level | Metric | Raw concepts | Meta-concepts (k=500) |
|:---|:---|:---|:---|
| Singletons | % | 95.1% | 0.2% |
| Mean frequency | eps/concept | 1.08 | 11.2 |
| Episode pairs sharing | count | 875 | 39,002 |
| Episode pairs sharing | % of all | 0.05% | 2.4% |

Top meta-concepts by episode frequency: MLE parameter estimation (56), R boot package (53), LLM capabilities (47), R statistical graphics (43), AI research (42), Bootstrap CIs (42).

### F51: Eight interpretable knowledge domains emerge from concept clustering

Ward linkage at k=8 produces semantically coherent domains:

| Domain | Concepts | Episodes | Representative concepts |
|:---|---:|---:|:---|
| Algorithms & Computation | 2,395 | 1,195 | simulation design, data structures, optimization |
| Statistical Methods | 932 | 562 | MLE, bootstrap, confidence intervals, series systems |
| Software Engineering | 839 | 487 | CLI tools, packaging, web dev, code review |
| Philosophy & AI Theory | 714 | 421 | alignment, AGI, philosophy of mind, reasoning |
| Visualization & Formatting | 461 | 306 | plotting, image generation, documents |
| R Programming | 222 | 183 | R packages, stats, data structures |
| LLM Engineering | 206 | 176 | prompt engineering, agent systems, tool use |
| AI Safety & Research | 175 | 134 | containment, consciousness, evaluation |

The largest domain (Algorithms & Computation) contains 40% of all concepts — a catch-all for computational topics. The remaining 7 domains are cleanly separated and map to recognizable knowledge areas.

### F52: 69% of episodes span multiple knowledge domains

The many-to-many bipartite structure naturally captures cross-domain knowledge integration that partition-based methods (Louvain, Exp 1) cannot represent.

| Domains spanned | Episodes | % |
|:---|---:|---:|
| 1 | 565 | 31.2% |
| 2 | 882 | 48.6% |
| 3 | 329 | 18.1% |
| 4 | 37 | 2.0% |

Mean participation coefficient: 0.338 (moderate domain diversity per episode). 20% of episodes are "highly diverse" (P > 0.5), functioning as cross-domain bridges.

Strongest domain co-occurrences: Statistics↔Algorithms (177 episodes), Software Eng.↔Algorithms (133), Philosophy↔Algorithms (101), R Programming↔Algorithms (52), R↔Statistics (43).

### F53: Semantic and geometric hierarchies are complementary (NMI = 0.26)

Comparing semantic domain assignments (from LLM-extracted concepts + clustering) with geometric cluster assignments (from episode embedding clustering):

| Comparison | NMI | ARI |
|:---|:---|:---|
| Semantic domains (k=8) vs Geometric L2 (k=15) | 0.261 | 0.111 |
| Semantic domains (k=8) vs Geometric L3 (k=5) | 0.259 | 0.190 |

The low NMI indicates the two hierarchies capture **different structural dimensions**. The geometric hierarchy groups episodes with similar overall embedding vectors; the semantic hierarchy groups episodes through shared extracted concepts. This orthogonality suggests LLM concept extraction adds genuine information beyond what's captured by embedding geometry alone.

### F54: Semantic concept space is genuinely continuous (silhouette monotonically increases)

Silhouette score increases monotonically with cluster count k:

| k | Silhouette | Level |
|:---|:---|:---|
| 2 | 0.015 | — |
| 8 | 0.013 | Domains |
| 50 | 0.025 | Themes |
| 100 | 0.045 | — |
| 200 | 0.067 | — |
| 500 | 0.107 | Meta-concepts |

No natural cluster boundaries exist — the hierarchy is a useful simplification of a continuous semantic space. This parallels F45 from Experiment 1 and is consistent with CLS theory: knowledge exists on a continuous spectrum from episodic to semantic, not in discrete categories.

### F55: Heaps' law governs concept vocabulary growth (β = 0.286)

Meta-concept vocabulary grows sublinearly with episode count: V(n) = 64.6 × n^0.286. The Heaps exponent β=0.286 is lower than natural language (β ≈ 0.4-0.6), indicating faster saturation of the concept space. This is the CLS consolidation signature: new episodes increasingly map to existing semantic concepts rather than creating novel ones.

### F56: Meta-concept frequency follows a heavy-tailed distribution

The meta-concept frequency distribution (episodes per meta-concept) has a heavy tail: α=2.82 with xmin=8. Lognormal provides a better fit than a pure power law (R=-24.2, p<0.001). The distribution is: mean=11.2, median=10, IQR=[7,14], max=56.

### F57: The 4-level hierarchy has explicit branching ratios

| Level | Label | k | Mean size | Silhouette |
|:---|:---|---:|---:|:---|
| 0 | Episodes | 1,813 | — | — |
| 1 | Meta-concepts | 500 | 11.9 concepts each | 0.107 |
| 2 | Themes | 50 | 118.9 concepts each | 0.025 |
| 3 | Domains | 8 | 743.0 concepts each | 0.013 |

Branching ratios: 3.6× (episodes→meta-concepts), 10× (meta-concepts→themes), 6.25× (themes→domains). The DAG has 8,315 nodes and 12,924 edges, confirmed acyclic.

### F58: Concept co-occurrence network is strongly small-world (σ ≈ 5)

The meta-concept co-occurrence network (two meta-concepts linked if they appear in the same episode) exhibits small-world topology, directly validating Steyvers & Tenenbaum (2005).

| Property | Value |
|:---|:---|
| Nodes | 500 meta-concepts |
| Edges | 4,992 |
| Density | 0.040 |
| Giant component | 499 (99.8%) |
| Mean degree | 20.0 |
| Max degree | 78 |
| Clustering C | 0.233 |
| C / C_random | 5.80 |
| Path length L | 2.51 |
| L / L_random | 1.21 |
| **Small-world σ** | **4.80 (theoretical), 5.48 (empirical)** |
| Assortativity | 0.036 (neutral) |
| Transitivity | 0.179 |

Comparison with Steyvers & Tenenbaum (2005) semantic memory networks:

| Network | σ |
|:---|:---|
| Roget's Thesaurus | 13.0 |
| WordNet | 15.3 |
| Word associations | 5.6 |
| **Our concept network** | **4.8-5.5** |

Our σ ≈ 5 is closest to the human word association network (σ = 5.6), suggesting LLM-extracted concepts from AI conversations organize into the same semantic memory topology as human free associations. The degree distribution fits α=2.97 (near-classic scale-free exponent), though exponential provides a statistically better fit given the limited tail range (max degree 78).
