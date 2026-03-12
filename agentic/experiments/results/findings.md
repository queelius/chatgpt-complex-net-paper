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

## F43: Concept Extraction — Knowledge Domains vs Work Processes

**Method**: LLM-based concept extraction (llama3.2 via Ollama) from Louvain community clusters. Each community's top 20 sessions (by message count) fed to LLM with prompt requesting 2-5 abstract concepts.

| Metric | ChatGPT (θ=0.9) | Agentic parent (θ=0.95) |
|:---|:---|:---|
| Communities extracted | 17 | 9 |
| Total concepts | **79** | 36 |
| Concepts/community | 4.6 | 4.0 |
| Cross-community associations | 42 | 40 |

**ChatGPT concept examples**: "Bayesian Reasoning for Model Selection", "Statistical Modeling and Estimation", "Philosophical Inquiry", "Symbolic Computation", "Medical Diagnosis Reasoning"

**Agentic concept examples**: "Iterative Refining", "Meta-Reflection", "Error Handling Protocol", "Debugging Iterative Refinement", "Verification-Driven Development", "Refactoring as Knowledge Discovery"

**Key finding**: ChatGPT concepts are **knowledge-domain** concepts (WHAT the user knows). Agentic concepts are **process-workflow** concepts (HOW the user works). The same user, analyzed through two AI modalities, reveals complementary cognitive dimensions.

**Shared bridge concepts** appear in both: "Knowledge Graphing", "Pattern Recognition", "Iterative Refinement", "Problem Decomposition". These are modality-independent cognitive habits.

**Top concept word frequencies**:
- ChatGPT-dominant: "reasoning" (11), "cognitive" (9), "statistical" (4), "optimization" (5)
- Agentic-dominant: "iterative" (9), "refining" (3), "debugging", "refactoring", "error"
- Shared: "problem" (6/5), "knowledge" (6/5), "pattern" (2/2), "analysis" (4/2)

---

## F44: Recursive Abstraction Hierarchy — Convergence to Fundamental Orientations

**Method**: Three-level LLM-based abstraction. L1: concepts from communities. L2: meta-concepts from concepts (themes spanning ≥3 communities). L3: fundamental orientations from meta-concepts.

| Level | ChatGPT | Agentic | Compression ChatGPT | Compression Agentic |
|:---|:---|:---|:---|:---|
| L1: Concepts | 79 | 36 | — | — |
| L2: Meta-concepts | 8 | 6 | 9.9× | 6.0× |
| L3: Orientations | 4 | 3 | 2.0× | 2.0× |

**ChatGPT L3 orientations**: Analytical Problem-Solving, Iterative Improvement, Data-Driven Decision Making, Computationalist Approach → **epistemic** orientations (how you KNOW)

**Agentic L3 orientations**: Systems Thinking, Analytical Problem-Solving, Holistic Approach → **practical** orientations (how you DO)

**The one shared L3 theme**: "Analytical Problem-Solving" appears in both — but its content differs:
- ChatGPT version: Bayesian inference, statistical methods, knowledge representation
- Agentic version: iterative refinement, error analysis, debugging loops

**Critical observation**: The L2→L3 compression ratio is **exactly 2.0× for both platforms**. This suggests a universal property of how concept hierarchies compress in human cognitive archives — regardless of whether the archive captures knowledge exploration (ChatGPT) or project execution (agentic).

The L1→L2 compression differs (9.9× vs 6.0×): ChatGPT has more diverse domain knowledge requiring more compression, while agentic has concentrated process patterns.

---

## Updated Paper Narrative Synthesis

### Theme 7: Epistemic vs Practical Cognitive Architectures (F43-F44, NEW)

The concept extraction reveals the deepest finding: **the same human mind, viewed through two AI modalities, reveals complementary cognitive dimensions**.

| Dimension | ChatGPT | Agentic |
|:---|:---|:---|
| Network architecture | Knowledge Archipelago | Cognitive Web |
| Community semantics | Knowledge domains | Projects |
| Concept types | Epistemic (WHAT you know) | Practical (HOW you do) |
| L3 orientations | Data-Driven, Computationalist | Systems Thinking, Holistic |
| Shared foundation | Analytical Problem-Solving | Analytical Problem-Solving |

Conversational AI captures the user's **knowledge structure** — a library of domains with bridges between them. Agentic AI captures the user's **work process** — an interconnected web of iterative refinement cycles. Neither alone gives a complete picture. Together, they form a bimodal cognitive MRI: one scan for knowledge, one for practice.

---

## F45: Concept-Level Network and Cross-Platform Alignment

**Method**: Embed all 115 concepts (79 ChatGPT + 36 agentic) via nomic-embed-text using "name: description" as input. Compute all-pairs cosine similarity. Build concept-level network.

| Similarity type | Mean | Std | Max | ≥0.8 | ≥0.9 |
|:---|:---|:---|:---|:---|:---|
| ChatGPT↔ChatGPT | 0.646 | 0.067 | 0.901 | 86 | 2 |
| Agentic↔Agentic | 0.652 | 0.082 | 0.944 | 66 | 10 |
| **Cross-platform** | **0.634** | **0.071** | **0.943** | **48** | **6** |

Cross-platform concept similarity (μ=0.634) is nearly as high as within-platform (μ=0.646/0.652). The top cross-platform pair: "Bayesian inference as model selection" (sim=0.943) — independently extracted from both platforms.

**Concept network (θ=0.8)**: 115 nodes, 124 edges, 48 components, GC=47 nodes (41%). **39% of all edges are cross-platform** — concepts bridge the ChatGPT/agentic divide more than episodes do.

**Concept-level communities (Louvain)** reveal mixed clusters:
- "Knowledge Graphing" cluster (11 members, 6 CG + 5 AG): knowledge graph concepts from both platforms
- "Iterative Refinement" cluster (11 members, 2 CG + 9 AG): iteration-dominant in agentic
- "Cognitive Frameworks" cluster (10 members, 9 CG + 1 AG): cognition-dominant in ChatGPT
- "Statistical/Bayesian" cluster (9 members, 8 CG + 1 AG): statistics-dominant in ChatGPT

**Top bridge concepts** (most cross-platform connections):
1. ChatGPT:"Iterative Development and Refining" — 8 of 9 edges are cross-platform (89%)
2. ChatGPT:"Iterative Inference" — 7 of 8 (88%)
3. Agentic:"Bayesian inference as model selection" — 3 of 3 (100%)

---

## F46: Hidden Connections — Cognitive Structure Invisible to Similarity Networks

**Method**: For each pair of semantically similar concepts (θ_c=0.85) across different episode communities, check if the episodes in those communities are connected by direct embedding similarity. "Hidden connections" are episode pairs linked through concepts but NOT by direct similarity.

| Metric | ChatGPT | Agentic parents |
|:---|:---|:---|
| Concept bridges (θ_c=0.85) | 6 | 12 |
| Concept-linked episode pairs | 6,626 | 20,937 |
| Also similarity-linked | 42 (0.6%) | 1,502 (7.2%) |
| **Hidden connections** | **6,584 (99.4%)** | **19,435 (92.8%)** |

**Key finding**: Concepts reveal **90-99%** of connections that are invisible to the episodic similarity network. The concept layer adds a massive amount of cognitive structure.

ChatGPT has a higher hidden fraction (99.4% vs 92.8%) because its Knowledge Archipelago architecture means fewer direct cross-community similarity edges. The agentic Cognitive Web has more direct connections (7.2% vs 0.6% overlap) because its denser, less modular structure creates more inter-community similarity bridges.

**Top concept bridges creating hidden connections**:
- ChatGPT: "Simulation and Monte Carlo Methods" ~ "Simulation-based analysis" (C5↔C10, 3567 pairs)
- Agentic: "Iterative Refining" ~ "Iterative Refinement" (C0↔C2, 3668 pairs)

---

## F47: Concept Extraction Stability

**Method**: Run extraction 3× on 5 ChatGPT communities, compare concept names across runs.

| Community | Unique names | Exact-match stable | Semantic concepts (θ=0.85) |
|:---|:---|:---|:---|
| C0 (102 sessions) | 9 | 2 (22%) | 7 |
| C5 (82 sessions) | 12 | 1 (8%) | 11 |
| C7 (65 sessions) | 14 | 0 (0%) | 13 |
| C4 (46 sessions) | 10 | 2 (20%) | 10 |
| C10 (44 sessions) | 13 | 1 (8%) | 11 |
| **Mean** | **11.6** | **10.3%** | **10.4** |

Exact-match stability is low (10.3%) because llama3.2 rephrases concepts across runs. Semantic matching (embedding similarity θ≥0.85) merges variants like "Bayesian inference as model selection" ↔ "Bayesian reasoning for model selection", improving effective stability.

**Implication**: Concept *themes* are stable (Bayesian reasoning always appears in C0, statistical modeling in C5), but specific *phrasing* varies. Production use should employ (1) semantic deduplication across runs, (2) consensus extraction, or (3) larger models.

---

## F48: Concept-Level Guimerà-Amaral Roles

**Method**: Compute within-module degree z-scores and participation coefficients on the concept network at multiple thresholds (θ=0.65, 0.70, 0.75, 0.80). Classify nodes into Guimerà-Amaral roles: R1-ultra-peripheral, R2-peripheral, R3-connector, R4-kinless, R5-R7 hub variants.

At θ=0.70 (7 communities, 1280 edges):

| Role | Total | ChatGPT | Agentic | Cross-platform edge % |
|:---|---:|---:|---:|---:|
| R1-ultra-peripheral | 18 | 11 | 7 | 22.7% |
| R2-peripheral | 59 | 41 | 18 | 36.3% |
| R3-connector | 38 | 27 | 11 | 41.3% |

**Key finding**: **Connectors disproportionately carry cross-platform edges.** At θ=0.70, connectors have 41.3% cross-platform edges vs 22.7% for ultra-peripheral — a consistent pattern across all thresholds (gap widens at higher θ: 43.2% vs 31.0% at θ=0.75).

No hubs (z≥2.5) appear at any threshold — the concept network lacks the degree heterogeneity for hub formation. This contrasts with the episode-level network which has provincial hubs. Concepts are a flatter, more egalitarian layer.

**Top connectors** (highest participation): "Modeling Complexity", "Problem-Solving Strategies", "Contextual Understanding" — generic cognitive primitives that bridge concept communities and platforms simultaneously.

---

## F49: Recursive Abstraction Convergence

**Method**: Run L2→L3 abstraction 5 times per platform via llama3.2 to test robustness of the 2.0× compression ratio. Also attempt L3→L4 to test hierarchy termination.

| Platform | L2 count | L3 counts (5 runs) | Mean compression | Std |
|:---|---:|:---|---:|---:|
| ChatGPT | 8 | 2, 3, 4, 5, 3 | 2.59× | 0.82 |
| Agentic | 6 | 2, 2, 2, 2, 2 | 3.00× | 0.00 |

**Key findings**:

1. **Agentic compression is perfectly deterministic**: 6→2 = 3.0× in all 5 runs. The LLM always finds exactly 2 themes for 6 meta-concepts.

2. **ChatGPT compression is variable**: 8→{2,3,4,5,3}, mean 2.59×. With more inputs, the LLM wavers between aggressive lumping and fine-grained splitting.

3. **L3→L4 always converges to 1**: Both platforms collapse to a single concept at L4 (ratio 2.0×). The hierarchy terminates.

4. **The "universal 2.0× compression" claim is not robust.** The original observation (both platforms showing 2.0×) was a coincidence of the first run. The actual pattern: compression varies with input size, but the **hierarchy always terminates at L4 with a single concept** — this is the true universal property.

---

## F50: Hierarchical Multiplex Network (E—E + C→E + C—C)

**Method**: Build three-relation multiplex: (1) E—E episode similarity (existing), (2) C→E concept-to-episode instantiation (concept links to all episodes in its source community), (3) C—C concept similarity (θ=0.70).

| Layer | Nodes | Edges | Communities |
|:---|---:|---:|---:|
| C—C (concept similarity) | 115 | 1,280 | 7 |
| C→E (instantiation) | 1,738 | 7,688 | — |
| ChatGPT E—E | 601 | 1,718 | 73 |
| Agentic E—E | 1,516 | 193,309 | 122 |

**Interlayer degree correlation (C—C vs C→E)**: ρ=0.235, p=0.011 — significant but weak. A concept's similarity to other concepts is weakly related to how many episodes it grounds. The layers carry complementary information.

**Community platform mixing**: 57% of concept-level communities contain both ChatGPT and agentic concepts. Largest mixed community has 41 members (23 ChatGPT + 18 agentic).

**C→E fan-out**: mean=66.9, median=12, max=600. Highly skewed — a few concepts from large communities anchor hundreds of episodes.

**Top multiplex participants** (balanced across C—C and C→E layers): "Iterative Development and Refining" (CC=29, CE=29), "Knowledge Graphs" (CC=25, CE=25), "Problem-Solving Frameworks" (CC=51, CE=46).

---

## F51: Consensus Concept Extraction

**Method**: Run N=5 concept extractions per community on top 3 ChatGPT communities (102, 82, 65 episodes). Embed all unique concept names via nomic-embed-text, cluster at θ≥0.85, keep clusters appearing in ≥3/5 runs.

| Community | Unique names | Exact consensus (≥3/5) | Semantic consensus (≥3/5) |
|:---|---:|---:|---:|
| C0 (102 episodes, ML/AI) | 22 | 1 (5%) | 4 |
| C5 (82 episodes, statistics) | 16 | 0 (0%) | 2 |
| C7 (65 episodes, philosophy) | 22 | 1 (5%) | 2 |
| **Mean** | **20** | **0.7** | **2.7** |

**Key finding**: Semantic consensus extracts **4× more stable concepts** than exact-match (mean 2.7 vs 0.7 per community). The most stable concepts:

- C0: "Fine-tuning and adaptation" (4/5 runs), "Neural Network Architectures" (3/5, 6 naming variants)
- C5: "Statistical Theory and Interpretation" (4/5 runs, 13 naming variants)
- C7: "The Human Condition and Complexity" (5/5 runs), "Existential Inquiry" (5/5 runs)

Community 7 (philosophy) is the most concept-stable — abstract philosophical themes are named more consistently than technical ones. Community 5 (statistics) has the most naming variance (13 variants for one concept).

**Implication**: Consensus extraction with semantic clustering is viable as a methodological improvement. The pipeline: extract N times → embed all names → cluster at θ≥0.85 → keep clusters in ≥60% of runs.

---

## Updated Paper Narrative Synthesis

### Theme 8: Concept Layer as Orthogonal Dimension (F48-F51)

The concept layer reveals structure invisible to both the episodic similarity and delegation layers:

| Property | Episode layer | Concept layer |
|:---|:---|:---|
| Degree distribution | Heterogeneous (has hubs) | Flat (no hubs, max z < 2.5) |
| Interlayer correlation | — | ρ=0.235 (weak, layers independent) |
| Platform mixing | Low (modularity → platform separation) | 57% mixed communities |
| Cross-platform bridging | Rare (39% at concept θ=0.80) | Connectors at 41-43% |
| Hidden connections | — | 92-99% of concept-linked pairs invisible to E—E |
| Abstraction hierarchy | — | Terminates at L4 (1 concept), deterministic for small inputs |
| Extraction stability | — | 10% exact-match, improved to 2.7/community via consensus |

The concept layer is genuinely orthogonal to the episodic layer — it reveals cognitive primitives (like "Iterative Refinement", "Statistical Theory") that organize knowledge across platforms and communities in ways that embedding similarity alone cannot detect.

---

## F52: Temporal Concept Evolution

**Method**: Split ChatGPT episodes into three eras — Early (Dec 2022 – Jun 2023, GPT-3.5 era, 157 episodes), Middle (Jul 2023 – Jun 2024, GPT-4 era, 291 episodes), Late (Jul 2024 – Apr 2025, GPT-4o/o1 era, 153 episodes). Extract concepts per era via LLM, embed, compare.

| Era | Episodes | Dominant themes | Within-era diversity |
|:---|---:|:---|---:|
| Early (Foundations) | 157 | ML, AI & Cognitive Science, Philosophy | 0.378 |
| Middle (Deepening) | 291 | Probability Theory, Programming, R&D | 0.353 |
| Late (Application) | 153 | Web Dev, Data Processing, ML for Science | 0.391 |

**Cross-era similarity decays with temporal distance**:
- Early ↔ Middle: 0.627
- Middle ↔ Late: 0.609
- Early ↔ Late: 0.605

**Key finding**: A clear **Foundations → Deepening → Application** cognitive arc. Early-era concepts are abstract and theoretical (Philosophy, Cognitive Science); middle-era concepts are methodological (Probability Theory, Programming); late-era concepts are applied and production-oriented (Web Development, Data Processing).

**Most era-unique concepts** (lowest similarity to other eras):
- Early: "General AI and Philosophy" (max sim to other eras = 0.609)
- Middle: "Research and Development" (0.674)
- Late: "Data Analysis and Processing" (0.615)

Within-era diversity is highest in the late era (0.391) — interests diversified after an initial deepening/specialization phase.

---

## F53: Cross-Platform 4-Layer Multiplex

**Method**: Build full 4-layer multiplex: (1) ChatGPT E—E (601 nodes, 1718 edges), (2) Agentic E—E (1516 nodes, 193K edges), (3) C—C concept similarity (115 nodes, 1280 edges at θ=0.70), (4) C→E instantiation (7688 edges). Compute interlayer metrics.

| Metric | Value |
|:---|:---|
| NMI (concept communities vs episode-projected) | 0.297 |
| Spearman ρ (C—C degree vs C→E fan-out) | 0.235 (p=0.011) |
| Community platform mixing (C—C layer) | 57% (4/7 mixed) |
| Largest mixed concept community | 41 members (23 CG + 18 AG) |

**Platform asymmetry in multiplex participation**:
- ChatGPT concepts: mean P=0.362, mean C→E fan-out=28.6
- Agentic concepts: mean P=0.200, mean C→E fan-out=150.9

Agentic concepts ground 5× more episodes but participate less in the C—C layer — they're dominated by their grounding rather than their conceptual relationships.

**Cross-layer spanning**: The largest concept community (41 members) spans 15 different episode-level communities. Concepts unify episodes that similarity keeps apart.

---

## F54: Concept Network Null Models

**Method**: Three null models test whether concept network properties are significant: (1) random platform shuffle (N=1000), (2) configuration model (degree-preserving, N=100), (3) community assignment shuffle (N=1000).

| Statistic | Observed | Null mean ± std | p-value | Significant? |
|:---|---:|:---|---:|:---|
| Cross-platform edge fraction | 0.391 | 0.433 ± 0.025 | 0.09 | Marginal |
| Connector cross-platform gap | 0.056 | -0.002 ± 0.052 | 0.13 | No |
| Mixed community fraction | 0.571 | 0.666 ± 0.068 | 1.00 | No (lower than null!) |
| Config model cross-platform | 0.391 | 0.435 ± 0.011 | <0.001 | Yes |

**Key finding — sobering correction**: Cross-platform concept mixing is **lower** than expected by chance, not higher. Same-platform concepts are more similar to each other than to cross-platform concepts. The configuration model (degree-preserving) confirms the structure is non-random (p<0.001), but the direction is opposite to the F48 narrative.

**Revised interpretation**: The concept layer bridges platforms *relative to the episode layer* (which can't compare cross-platform at all), but concepts retain strong platform identity. ChatGPT knowledge concepts cluster together; agentic process concepts cluster together. The bridges exist but are the minority, not the norm.

This is a more honest and nuanced finding: the concept layer enables cross-platform comparison but doesn't erase the fundamental epistemic vs. practical distinction.

---

## Updated Theme 8 (revised with null model corrections)

### Theme 8: Concept Layer as Complementary Dimension (F48-F54)

| Property | Episode layer | Concept layer |
|:---|:---|:---|
| Degree distribution | Heterogeneous (has hubs) | Flat (no hubs) |
| Interlayer correlation | — | ρ=0.235 (weak, complementary) |
| NMI concept vs episode communities | — | 0.297 (different groupings) |
| Platform mixing | N/A (separate networks) | 57% mixed, but *less* than random |
| Cross-platform edges | N/A | 39% (significant structure, but minority) |
| Temporal evolution | Densification γ≈1.41 | Foundations → Deepening → Application arc |
| Hierarchy termination | — | Always converges to 1 concept at L4 |
| Extraction stability | — | 2.7 stable concepts/community via consensus |

The concept layer is genuinely complementary to episodes — NMI=0.297 confirms different grouping structure. But the null models temper earlier enthusiasm: platforms still cluster separately at the concept level. The real value of the concept layer is not that it erases platform differences, but that it enables comparison across platforms and reveals a temporal cognitive arc invisible to static similarity networks.

---

## F55: Concept Stability Across LLM Models

**Method**: Extract concepts from same 3 ChatGPT communities using 3 different LLMs (llama3.2, phi4-mini, gemma3n). Embed all concept names via nomic-embed-text. Compute cross-model best-match similarity and semantic consensus.

| Community | llama3.2 vs phi4-mini | llama3.2 vs gemma3n | phi4-mini vs gemma3n | All-model consensus |
|:---|---:|---:|---:|---:|
| C0 (ML/AI) | 0.829 | 0.708 | 0.847 | 0 |
| C5 (Statistics) | — | 0.795 | — | 0 |
| C7 (Philosophy) | **0.958** | **0.977** | **0.966** | **1 (10 variants)** |

**Key finding**: **Abstract/philosophical concepts are the most model-robust.** Community 7 (philosophy/cognition) achieves near-perfect cross-model agreement (0.958-0.977), with all three LLMs independently converging on "existential inquiry" themes. Technical communities (ML, statistics) show more model variance because LLMs decompose the domain differently.

**Implication**: Philosophical/cognitive themes are the strongest candidates for universal cognitive primitives — they emerge regardless of which LLM performs the extraction.

---

## F56: Episode Centrality Does Not Predict Concept Bridging

**Method**: For each ChatGPT concept, compute its source community's mean episode degree, betweenness, and size. Correlate with concept's mean cross-platform similarity to agentic concepts.

| Correlation | ρ | p-value |
|:---|---:|---:|
| Community degree vs cross-platform sim | -0.050 | 0.659 |
| Community betweenness vs cross-platform sim | 0.073 | 0.520 |
| Community size vs cross-platform sim | -0.048 | 0.672 |
| Community degree vs concept C—C degree | 0.114 | 0.317 |

**Key finding**: **All correlations are non-significant.** Episode-level network centrality has zero predictive power for concept-level cross-platform bridging. A community's episodes being central in the E—E network tells you nothing about whether its concepts bridge to the agentic platform. This further confirms the concept layer is genuinely independent of the episode layer.

---

## F57: Structural vs Semantic Concept Networks

**Method**: Compare two concept-concept networks: (1) structural (Jaccard overlap of episode sets — concepts from same community share all episodes), (2) semantic (cosine similarity of concept embeddings).

| Pair type | Mean cosine | N pairs |
|:---|---:|---:|
| Same community (Jaccard=1.0) | 0.707 | 204 |
| Different community, same platform | 0.643 | 3,507 |
| Cross-platform | 0.634 | 2,844 |

**Correlation (Jaccard vs Cosine)**: ρ=0.151, p<10⁻⁶ — significant but weak. The structural network (based on episode co-membership) weakly predicts semantic similarity, but the semantic network adds substantial cross-community structure that the structural network cannot capture.

**Within-community concept diversity** varies from 0.199 (C43, most redundant) to 0.337 (C24, most diverse). Communities with diversity < 0.25 suggest over-extraction — concepts are too semantically similar to be distinct.

---

## Master Comparison Table (Updated)

| Metric | ChatGPT | Agentic (parents) |
|:---|:---|:---|
| Episodes | 601 (θ=0.9) | 449 (θ=0.95) |
| Edges | 1,718 | 4,316 |
| Density | 0.0095 | 0.0429 |
| Modularity | 0.750 | 0.278 |
| Communities | 15 | 12 |
| Densification γ | 1.405 | 1.410 |
| Assortativity | -0.13 | -0.05 |
| Architecture | Knowledge Archipelago | Cognitive Web |
| L1 concepts | 79 | 36 |
| L2 meta-concepts | 8 | 6 |
| L3 orientations | 4 | 3 |
| L2→L3 compression | 2.59× (mean, variable) | 3.00× (deterministic) |
| Concept types | Epistemic (WHAT) | Practical (HOW) |
| Hidden connections | 99.4% | 92.8% |
| Cross-platform edges (concept) | 39% (< 43% null) | |
| Temporal concept arc | Foundations → Deepening → Application | — |
| Cross-model stability | Technical: 0.71-0.85; Philosophical: 0.96-0.98 | |

---

## F58: Bridge Concept Deep Dive

**Method**: Identify top 10 cross-platform concept pairs by embedding similarity. Trace each to source episodes and descriptions. Classify bridge types.

**Top bridges** (sim ≥ 0.87):

| ChatGPT concept | Agentic concept | Sim | Type |
|:---|:---|---:|:---|
| Bayesian inference as model selection | Bayesian inference as model selection | 0.943 | identical |
| Knowledge Graphs and Network Analysis | Knowledge Graphing | 0.916 | identical |
| Knowledge Graphs | Knowledge Graphing | 0.913 | identical |
| Problem Decomposition | Problem Decomposition | 0.908 | identical |
| Iterative Development and Refining | Iterative Problem-Solving | 0.894 | shared-keyword |
| Pattern of Abstraction | Abstraction and Simplification | 0.877 | semantic |

**Bridge type distribution**: 6 identical-concept, 3 shared-keyword, 1 semantic-bridge (out of top 10).

**Key finding**: The strongest bridges are **genuine cognitive primitives** — independently extracted from both platforms with near-identical names. "Bayesian inference as model selection" emerges from MLE analysis conversations (ChatGPT) and statistical analysis sessions (agentic). These are the user's actual cognitive habits visible in both modalities.

---

## F59: Concept Redundancy Pruning

**Method**: Identify communities with within-concept diversity < 0.25 (over-extracted). Remove all but the most connected concept per redundant community. Recompute network metrics.

| Metric | Original | Pruned | Change |
|:---|---:|---:|:---|
| Concepts | 115 | 104 | -11 (3 communities) |
| ChatGPT concepts | 79 | 68 | -11 |
| Agentic concepts | 36 | 36 | unchanged |
| Edges (θ=0.70) | 1,280 | 1,047 | -18% |
| Communities | 8 | 8 | unchanged |
| Cross-platform edge % | 39.1% | **41.5%** | **+2.4%** |
| Density | 0.1953 | 0.1955 | unchanged |

**Key finding**: Pruning redundant concepts **increases** cross-platform fraction from 39.1% to 41.5%. Redundant concepts were same-platform duplicates that inflated within-platform edges. All 36 agentic concepts survived (only ChatGPT had over-extraction). The pruned network better represents the true cross-platform structure.

---

## Master Comparison Table (Updated with F58-F59)

| Metric | ChatGPT | Agentic (parents) |
|:---|:---|:---|
| Episodes | 601 (θ=0.9) | 449 (θ=0.95) |
| Edges | 1,718 | 4,316 |
| Density | 0.0095 | 0.0429 |
| Modularity | 0.750 | 0.278 |
| Communities | 15 | 12 |
| Densification γ | 1.405 | 1.410 |
| Assortativity | -0.13 | -0.05 |
| Architecture | Knowledge Archipelago | Cognitive Web |
| L1 concepts | 79 (68 after pruning) | 36 |
| L2 meta-concepts | 8 | 6 |
| L3 orientations | 4 | 3 |
| L2→L3 compression | 2.59× (variable) | 3.00× (deterministic) |
| Concept types | Epistemic (WHAT) | Practical (HOW) |
| Hidden connections | 99.4% | 92.8% |
| Cross-platform edges (pruned) | 41.5% | |
| Bridge type | 6/10 identical concepts | |
| Temporal arc | Foundations → Deepening → Application | — |
| Cross-model stability | Technical: 0.71-0.85; Philosophical: 0.96-0.98 | |

---

## F60: Model Era Effects on Concept Bridging

**Method**: Assign each ChatGPT concept to the era (Early/Middle/Late) of its source community's median episode date. Compute each concept's mean cross-platform similarity to all agentic concepts. Test era effect via Kruskal-Wallis.

| Era | N concepts | Mean cross-platform sim | Std |
|:---|---:|---:|---:|
| Early (GPT-3.5) | 14 | 0.627 | 0.035 |
| Middle (GPT-4) | 43 | 0.627 | 0.041 |
| Late (GPT-4o) | 22 | **0.654** | 0.031 |

**Kruskal-Wallis H=7.826, p=0.020** — significant era effect.

**Key finding**: Late-era (GPT-4o) concepts bridge significantly more to agentic concepts than early/middle-era concepts. The 4.3% gap (0.654 vs 0.627) is small but consistent. Early and middle eras are indistinguishable (both 0.627).

**Interpretation**: As the user's ChatGPT conversations shifted from foundational exploration to applied work (F52: Foundations → Application arc), the concepts became more similar to agentic process concepts. The late era's applied, production-oriented concepts ("Problem-Solving Frameworks", "Data Processing and Analysis") naturally align with agentic workflow patterns.

**Top bridging concepts by era**:
- Late: "Problem-Solving Frameworks" (0.717), "Iterative Inference" (0.707), "Reverse-Process Thinking" (0.689)
- Middle: "Iterative Development and Refining" (0.705), "Pattern of Abstraction" (0.700)
- Early: "Problem-Solving Strategies" (0.694), "Iterative Refinement" (0.683)

---

## F61: Disparity Filter Backbone (Negative Result)

**Method**: Apply Serrano et al. (2009) disparity filter to concept network (θ=0.70, 115 nodes, 1280 edges). The filter retains edges whose weight is statistically incompatible with a null model of uniform weight distribution: p_ij = (1 - w_ij/s_i)^(k_i - 1), keep if p < α.

| α threshold | Edges retained | Retention % | Cross-platform % |
|:---|---:|---:|---:|
| 0.05 | 6 | 0.5% | 0% |
| 0.10 | 6 | 0.5% | 0% |
| 0.20 | 6 | 0.5% | 0% |
| 0.50 | 1,280 | 100% | 39.1% |

**Key finding (negative result)**: The disparity filter fails on this network. At any meaningful α (≤0.20), only 6 edges survive — all same-platform, all from a single high-weight cluster. At α=0.50, all edges pass.

**Why it fails**: The concept network has insufficient weight heterogeneity. Most nodes have relatively uniform edge weight distributions (cosine similarities clustered in a narrow band above θ=0.70). The disparity filter is designed for networks with heavy-tailed weight distributions (e.g., trade networks, airport networks), not for thresholded similarity networks where the threshold already removes weak edges.

**Alternative approaches for backbone extraction**: k-core decomposition, minimum spanning tree, or edge betweenness could work better for this network topology.

---

## Master Comparison Table (Updated with F60-F61)

| Metric | ChatGPT | Agentic (parents) |
|:---|:---|:---|
| Episodes | 601 (θ=0.9) | 449 (θ=0.95) |
| Edges | 1,718 | 4,316 |
| Density | 0.0095 | 0.0429 |
| Modularity | 0.750 | 0.278 |
| Communities | 15 | 12 |
| Densification γ | 1.405 | 1.410 |
| Assortativity | -0.13 | -0.05 |
| Architecture | Knowledge Archipelago | Cognitive Web |
| L1 concepts | 79 (68 after pruning) | 36 |
| L2 meta-concepts | 8 | 6 |
| L3 orientations | 4 | 3 |
| L2→L3 compression | 2.59× (variable) | 3.00× (deterministic) |
| Concept types | Epistemic (WHAT) | Practical (HOW) |
| Hidden connections | 99.4% | 92.8% |
| Cross-platform edges (pruned) | 41.5% | |
| Bridge type | 6/10 identical concepts | |
| Temporal arc | Foundations → Deepening → Application | — |
| Era bridging effect | Late > Early/Middle (p=0.020) | |
| Cross-model stability | Technical: 0.71-0.85; Philosophical: 0.96-0.98 | |
| Disparity filter backbone | Fails (insufficient weight heterogeneity) | |

---

## F62: K-Core Decomposition — The Cross-Platform Backbone

**Method**: Compute k-core decomposition of concept network (θ=0.70). The k-core is the maximal subgraph where every node has degree ≥ k. Unlike the failed disparity filter (F61), k-cores work on any topology.

| Core level | Nodes | CG:AG | Edges | Cross-platform % | Density | Communities |
|:---|---:|:---|---:|---:|---:|---:|
| Full network | 115 | 79:36 | 1,280 | 39.1% | 0.20 | 7 |
| k ≥ 16 | 50 | 29:21 | 697 | **47.3%** | 0.57 | 4 |
| k ≥ 17 | 41 | 23:18 | 557 | **47.6%** | 0.68 | 4 |
| k ≥ 18 | 37 | 22:15 | 489 | **47.4%** | 0.73 | 4 |
| k ≥ 19 (max) | 31 | 19:12 | 382 | **46.3%** | 0.82 | 4 |

**Key finding**: The innermost core (k=19) is the cross-platform backbone. Cross-platform edge fraction increases from 39.1% (full network) to **47.3-47.6%** (inner cores) — the core is where platforms converge. With density 0.82, the 19-core is a near-complete subgraph of 31 concepts.

**Core composition reveals platform convergence**: The full network is 69% ChatGPT, but the 19-core is 61% ChatGPT (19:12). Agentic concepts are disproportionately represented in the core — they are more broadly connected to other concepts. At the very highest degrees (k≥52), the platform ratio reaches 50:50.

**Top 19-core concepts** (the cognitive backbone):
1. "Problem-Solving Strategies" (CG, degree=67)
2. "Iterative Refining" (AG, degree=62)
3. "Pattern Recognition" (AG, degree=57)
4. "Pattern of Abstraction" (CG, degree=55)
5. "Rule-Based Reasoning" (CG, degree=52)
6. "Iterative Refinement" (AG, degree=52)

These are genuine cognitive primitives — abstract enough to appear in both knowledge exploration and project execution, forming the common core of the user's cognitive style.

---

## F63: Concept Network Motif Census

**Method**: Enumerate all triangles in the concept network (θ=0.70). Classify by platform composition: CCC (all-ChatGPT), CCA, CAA, AAA (all-agentic). Compare to expected frequencies under random platform assignment.

| Triangle type | Count | Observed % | Expected % | Ratio |
|:---|---:|---:|---:|---:|
| CCC (all-ChatGPT) | 2,308 | 31.9% | 32.4% | 0.98 |
| CCA (2 CG + 1 AG) | 2,948 | 40.7% | 44.3% | 0.92 |
| CAA (1 CG + 2 AG) | 1,587 | 21.9% | 20.2% | **1.08** |
| AAA (all-agentic) | 402 | 5.5% | 3.1% | **1.81** |
| **Total** | **7,245** | | | |

**Key finding**: Agentic concept triangles are **1.8× overrepresented**. Agentic process concepts form closed triads more than expected — when two agentic concepts share a neighbor, they're likely connected to each other. This reflects the tighter, more interdependent structure of workflow concepts (debugging ↔ testing ↔ refactoring form natural triads). Mixed CCA triangles are underrepresented (0.92×), suggesting partial platform segregation at the triadic level.

**Clustering coefficients**:
- Overall: transitivity=0.531, avg clustering=0.572
- Agentic concepts: **0.629** (more clustered)
- ChatGPT concepts: 0.609

Agentic concepts are more locally clustered — their process patterns form tighter neighborhoods than ChatGPT's knowledge domains.

---

## F64: Concept-Episode Projection Alignment

**Method**: Compare concept-level Louvain communities with episode-level source communities. Compute NMI and mapping entropy.

| Metric | Value |
|:---|:---|
| Concept communities (Louvain) | 7 |
| Source communities mapped | 26 |
| NMI (source vs concept community) | **0.297** |
| Mean mapping entropy | 1.046 |

**Key finding**: NMI=0.297 confirms the concept layer **reorganizes** episode-level structure rather than mirroring it. Concepts from 26 different source communities are redistributed into 7 concept communities based on semantic similarity, not structural proximity.

**Largest concept community (CC5)**: 41 concepts from 15 source communities (23 CG + 18 AG) — a massive cross-platform, cross-community convergence. This single concept community spans more than half of all source communities, unifying knowledge that the episode layer keeps separate.

**Implication**: The concept layer is not merely a coarse-graining of the episode layer. It genuinely regroups knowledge by semantic affinity, pulling concepts from diverse episode communities into coherent thematic clusters.

---

## F65: Concept Degree Assortativity and Cross-Platform Bridging

**Method**: Compute degree-degree correlations, platform assortativity, and the relationship between degree and cross-platform edge fraction.

| Metric | Value | p-value |
|:---|:---|:---|
| Degree assortativity (r) | 0.027 | — |
| Platform assortativity | 0.096 | — |
| Source community assortativity | 0.033 | — |
| Spearman ρ (edge degree-degree) | 0.051 | 0.070 |
| **Spearman ρ (degree vs cross-platform frac)** | **0.496** | **< 10⁻⁸** |

**Key finding — degree predicts cross-platform bridging**: ρ=0.496 (p < 10⁻⁸) is the strongest structural correlation found in this study. **Higher-degree concepts have dramatically more cross-platform edges.** Generalist concepts that connect to many other concepts are the ones that bridge the ChatGPT/agentic divide.

**Platform degree parity**: ChatGPT mean degree=22.2, agentic mean=22.4 (Mann-Whitney p=0.957). Despite having 2.2× more ChatGPT concepts, the degree distributions are identical — neither platform dominates the concept layer's connectivity structure.

**Near-zero assortativity** (r=0.027) means the concept network has neutral degree mixing — high-degree nodes connect to both high- and low-degree nodes equally. This is unlike many real-world networks which show clear assortative or disassortative patterns.

---

## F66: Rich-Club Effect in Concept Network

**Method**: Compute rich-club coefficient φ(k) and normalized ρ(k) = φ(k)/φ_random(k) using 100 configuration model realizations. The rich-club coefficient measures whether high-degree nodes interconnect more than expected.

| k threshold | φ(k) observed | φ(k) random | ρ(k) normalized | z-score |
|:---|---:|---:|---:|---:|
| k ≥ 5 | 0.278 | 0.226 | **1.23** | 21.3 |
| k ≥ 15 | 0.373 | 0.315 | **1.18** | 8.1 |
| k ≥ 25 | 0.686 | 0.519 | **1.32** | 10.0 |
| k ≥ 30 | 0.793 | 0.604 | **1.31** | 6.1 |
| k ≥ 36 | 0.942 | 0.701 | **1.34** | 2.7 |

**Key finding**: The concept network has a **highly significant rich-club effect** (ρ > 1.2, z > 20 at low k thresholds). High-degree concepts interconnect ~20-34% more densely than expected by their degree sequence alone. The effect strengthens at higher k (ρ peaks at 1.34 for k≥36), meaning the most connected concepts form an especially tight elite club.

**Rich-club composition (top 10% by degree)**:
- 7 ChatGPT + 4 agentic concepts (ratio approaches 50:50 at very top)
- Agentic members: "Iterative Refining" (deg=62), "Pattern Recognition" (57), "Iterative Refinement" (52) — each with 8/10 cross-platform rich-club connections
- ChatGPT members: "Problem-Solving Strategies" (67), "Pattern of Abstraction" (55) — each with 3/10 cross-platform

**Critical observation**: Within the rich club, agentic concepts are **more cross-platform** (8/10 connections cross-platform) than ChatGPT concepts (3/10). Agentic process concepts are the universal connectors — they're similar to everything. ChatGPT knowledge concepts in the rich club connect mostly to other ChatGPT concepts.

---

## Updated Theme 9: Core-Periphery Structure of Cognitive Concepts (F62-F66)

The concept network reveals a layered architecture:

| Layer | Structure | Platform mixing | Function |
|:---|:---|:---|:---|
| **Core** (k≥19, 31 nodes) | Near-clique (density=0.82) | 47% cross-platform | Universal cognitive primitives |
| **Mid-shell** (k=13-18) | Moderately connected | 39-47% | Domain-bridging concepts |
| **Periphery** (k<13) | Loosely connected | < 39% | Platform-specific specializations |

**Three key structural properties**:

1. **Rich-club effect** (ρ=1.2-1.3, z>20): High-degree concepts preferentially interconnect, forming a dense elite club
2. **Degree predicts bridging** (ρ=0.50, p<10⁻⁸): Generalist concepts bridge platforms; specialist concepts stay within-platform
3. **Agentic concepts dominate the core**: Despite being 31% of all concepts, agentic concepts comprise 39% of the 19-core and have 8/10 cross-platform rich-club connections vs 3/10 for ChatGPT

**Interpretation**: The user's cognitive concept space has a clear core-periphery gradient. The core consists of abstract, modality-independent cognitive habits (problem-solving, iteration, abstraction, pattern recognition) that function identically in knowledge exploration and project execution. The periphery contains specialized domain knowledge (statistics, philosophy, debugging protocols) that retains platform identity. The concept layer's value is precisely this: it identifies which cognitive patterns are universal and which are modality-specific.

---

## Master Comparison Table (Updated with F62-F66)

| Metric | ChatGPT | Agentic (parents) |
|:---|:---|:---|
| Episodes | 601 (θ=0.9) | 449 (θ=0.95) |
| Edges | 1,718 | 4,316 |
| Density | 0.0095 | 0.0429 |
| Modularity | 0.750 | 0.278 |
| Communities | 15 | 12 |
| Densification γ | 1.405 | 1.410 |
| Assortativity | -0.13 | -0.05 |
| Architecture | Knowledge Archipelago | Cognitive Web |
| L1 concepts | 79 (68 after pruning) | 36 |
| L2 meta-concepts | 8 | 6 |
| L3 orientations | 4 | 3 |
| L2→L3 compression | 2.59× (variable) | 3.00× (deterministic) |
| Concept types | Epistemic (WHAT) | Practical (HOW) |
| Hidden connections | 99.4% | 92.8% |
| Cross-platform edges (pruned) | 41.5% | |
| Bridge type | 6/10 identical concepts | |
| Temporal arc | Foundations → Deepening → Application | — |
| Era bridging effect | Late > Early/Middle (p=0.020) | |
| Cross-model stability | Technical: 0.71-0.85; Philosophical: 0.96-0.98 | |
| **Concept 19-core** | 19 concepts (61%) | 12 concepts (39%) |
| **Core cross-platform frac** | 46-48% (vs 39% full) | |
| **Concept clustering** | 0.609 | 0.629 |
| **Triangle overrepresentation** | CCC: 0.98× | AAA: 1.81× |
| **Rich-club ρ** | 1.2-1.3 (z > 20, significant) | |
| **Degree → bridging ρ** | 0.496 (p < 10⁻⁸) | |
| **NMI (source vs concept comm)** | 0.297 (low — concepts reorganize) | |

---

## F67: Concept Network Resilience

**Method**: Compare giant component survival under four attack strategies: (1) targeted by degree, (2) targeted by betweenness, (3) random (50 trials), (4) platform-specific removal.

| Removed % | Targeted (degree) | Targeted (betw.) | Random (mean±std) |
|:---|---:|---:|:---|
| 0% | 95.7% | 95.7% | 95.7% |
| 10% | 86.1% | 80.9% | 85.8 ± 1.1% |
| 20% | 74.8% | 70.4% | 74.9 ± 1.5% |
| 30% | 65.2% | 60.9% | 65.1 ± 1.6% |
| 50% | **24.3%** | **37.4%** | **45.7 ± 1.5%** |
| 70% | 2.6% | 2.6% | 26.8 ± 1.5% |

**Fraction to halve GC**: Targeted(degree) = 50%, targeted(betweenness) = 50%, random = 50%.

**Key finding**: The concept network is remarkably resilient — it takes removing **50% of nodes** to halve the GC under any strategy. At low removal rates (≤30%), targeted and random are nearly identical. The network's density (0.20) provides enough path redundancy that individual hub removal doesn't create bottlenecks. The divergence appears only at 50%+ removal, where targeted attack collapses the GC to 24% while random only reduces to 46%.

**Platform-specific resilience**:
- Removing 100% of agentic concepts: GC = **66.1%** (doesn't halve!)
- Removing 70% of ChatGPT concepts: GC = **43.5%** (just halves)

The network can survive losing ALL agentic concepts without fragmenting. ChatGPT knowledge concepts alone form a connected backbone. Agentic concepts are embedded within this backbone but are not structurally necessary — they add cross-platform bridges but not structural integrity.

---

## F68: Cross-Platform Concept Flow (Temporal Convergence)

**Method**: For cross-platform concept pairs (sim ≥ 0.80), classify the ChatGPT concept by its era (Early/Middle/Late). Compute matching rates: what fraction of each era's concepts have an agentic match?

| Era | Total concepts | With agentic match | Matching rate |
|:---|---:|---:|---:|
| Early (GPT-3.5) | 14 | 1 | **7.1%** |
| Middle (GPT-4) | 42 | 9 | **21.4%** |
| Late (GPT-4o) | 21 | 9 | **42.9%** |

**Key finding — temporal convergence**: Late-era ChatGPT concepts are **6× more likely** to match an agentic concept than early-era (42.9% vs 7.1%). This is a clear temporal gradient: as the user's ChatGPT conversations matured from foundational exploration to applied work, concepts increasingly overlapped with agentic process patterns.

**Cross-platform pairs**: 48 total at sim ≥ 0.80. Distribution: 52% from late era, 46% from middle, only 2% from early.

**Interpretation**: The user's cognitive style converged across platforms over time. Early ChatGPT use explored abstract domains (philosophy, cognitive science) with no agentic parallel. Late ChatGPT use focused on applied problem-solving — exactly the domain where agentic sessions live. The concept layer captures this convergence quantitatively.

---

## F69: Spectral Analysis

**Method**: Compute Laplacian, normalized Laplacian, and adjacency eigenvalue spectra. Analyze Fiedler value (algebraic connectivity), spectral gaps, and community structure suggestions.

| Metric | Value |
|:---|:---|
| Connected components | 4 |
| Fiedler value (λ₂) | **0.206** |
| Spectral gap (λ₃/λ₂) | 1.33 |
| Spectral radius | 25.21 |
| Louvain communities | 7 |
| Spectral suggestion (largest gap) | k=9 |

**Spectral gap structure** (largest gaps suggest natural community counts):

| k | Gap (Δλ) | λ_k → λ_{k+1} |
|:---|---:|:---|
| 9 | **0.594** | 0.735 → 1.329 |
| 16 | 0.502 | 2.242 → 2.744 |
| 11 | 0.453 | 1.457 → 1.910 |
| 6 | 0.398 | 0.274 → 0.672 |

**Key finding**: The spectral analysis suggests k=9 communities (largest gap), close to Louvain's k=7. The secondary gap at k=6 also aligns with Louvain. The Fiedler value (0.206) indicates moderate algebraic connectivity — the network is neither barely connected nor highly integrated.

The spectral radius (25.21) is close to the mean degree (22.3), consistent with a relatively homogeneous degree distribution. In contrast, networks with strong hubs have spectral radius >> mean degree.

---

## F70: Concept Network Small-World Properties

**Method**: Compute clustering coefficient C, average path length L, and compare to Erdős-Rényi random graphs (same n,m) and ring lattices. Compute Humphries-Gurney σ and Telesford ω.

| Metric | Observed | ER Random | Lattice |
|:---|---:|---:|---:|
| Clustering (C) | **0.598** | 0.214 | 0.717 |
| Avg path length (L) | **2.074** | 1.792 | — |
| Diameter | **7** | — | — |

| Small-world metric | Value | Interpretation |
|:---|---:|:---|
| σ (Humphries-Gurney) | **2.42** | σ > 1 → small-world |
| C/C_random | **2.80** | 2.8× more clustered than random |
| L/L_random | **1.16** | Only 16% longer paths than random |
| ω (Telesford) | **0.03** | ≈ 0 → true small-world |

**Key finding**: The concept network is a **confirmed small-world** (σ=2.42, ω≈0). It has nearly 3× the clustering of a random graph but essentially random path lengths. Any concept can reach any other in ≤7 steps (diameter), with average path length just 2.07.

**Platform-specific path lengths**:
- ChatGPT↔ChatGPT: 2.04
- Agentic↔Agentic: **1.89** (shortest)
- Cross-platform: **2.15** (longest, but only ~5% longer)

Agentic concepts are closer to each other (1.89 hops) than ChatGPT concepts (2.04), consistent with their higher clustering (F63). Cross-platform distances (2.15) are barely longer than within-platform — the concept layer maintains a compact cross-platform distance despite platform identity.

---

## Updated Theme 10: The Concept Network as a Small-World with Resilient Core (F67-F70)

| Property | Value | Significance |
|:---|:---|:---|
| Small-world σ | 2.42 | Confirmed small-world |
| Small-world ω | 0.03 | Neither lattice nor random |
| Resilience | 50% removal to halve GC | Robust to both targeted and random attack |
| Agentic removal resilience | 100% removal → 66% GC survives | Network doesn't depend on agentic concepts structurally |
| Temporal convergence | 7% → 43% matching rate (early → late) | 6× concept convergence over time |
| Spectral communities | k=9 (vs Louvain k=7) | Reasonable agreement |
| Cross-platform distance | 2.15 hops (vs 2.04 within-platform) | Near-parity |

The concept network is a compact, resilient small-world where:
1. **Cross-platform concepts are just one extra hop away** (2.15 vs 2.04)
2. **The network survives complete agentic concept removal** — ChatGPT knowledge concepts form a self-sufficient backbone
3. **Temporal convergence accelerates** — the user's concept spaces merged over time as ChatGPT use became more applied
4. **Spectral structure agrees with community detection** — the 7-9 community range is a genuine structural feature

---

## Master Comparison Table (Updated with F67-F70)

| Metric | ChatGPT | Agentic (parents) |
|:---|:---|:---|
| Episodes | 601 (θ=0.9) | 449 (θ=0.95) |
| Edges | 1,718 | 4,316 |
| Density | 0.0095 | 0.0429 |
| Modularity | 0.750 | 0.278 |
| Communities | 15 | 12 |
| Densification γ | 1.405 | 1.410 |
| Assortativity | -0.13 | -0.05 |
| Architecture | Knowledge Archipelago | Cognitive Web |
| L1 concepts | 79 (68 after pruning) | 36 |
| L2 meta-concepts | 8 | 6 |
| L3 orientations | 4 | 3 |
| L2→L3 compression | 2.59× (variable) | 3.00× (deterministic) |
| Concept types | Epistemic (WHAT) | Practical (HOW) |
| Hidden connections | 99.4% | 92.8% |
| Cross-platform edges (pruned) | 41.5% | |
| Bridge type | 6/10 identical concepts | |
| Temporal arc | Foundations → Deepening → Application | — |
| Era bridging effect | Late > Early/Middle (p=0.020) | |
| Era matching rate | 7% → 43% (early → late) | |
| Cross-model stability | Technical: 0.71-0.85; Philosophical: 0.96-0.98 | |
| Concept 19-core | 19 concepts (61%) | 12 concepts (39%) |
| Core cross-platform frac | 46-48% (vs 39% full) | |
| Concept clustering | 0.609 | 0.629 |
| Triangle overrepresentation | CCC: 0.98× | AAA: 1.81× |
| Rich-club ρ | 1.2-1.3 (z > 20, significant) | |
| Degree → bridging ρ | 0.496 (p < 10⁻⁸) | |
| Small-world σ | 2.42 (concept network) | |
| Small-world ω | 0.03 (confirmed) | |
| Resilience to halve GC | 50% removal (any strategy) | |
| Spectral communities | k=9 (vs Louvain k=7) | |

---

## Pending Experiments

1. **Full consensus extraction**: All communities, N=5 runs (production-quality concept set)
2. **Agentic temporal evolution**: Apply temporal analysis to agentic sessions
3. **Concept network information flow**: Simulate random walks — where does information concentrate?
4. **Ego networks of bridge concepts**: Local structure around the top cross-platform bridges
5. **Community role evolution across thresholds**: How do Guimerà-Amaral roles shift from θ=0.65 to 0.80?
