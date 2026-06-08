# Experiment Log

All results generated on 2026-04-09/10. Analysis DB: data/analysis.db.
Embedding model: openai:text-embedding-3-small:256 (149,623 per-message embeddings).
Corpus: 3,097 conversations (2,496 ChatGPT, 472 Claude Code, 129 Anthropic).

## Experiment 1: Cross-Source Replication

**Script:** build_graph.py (threshold sweep, per-source analysis)
**Results:** experiments/results/full_analysis.json, threshold_sweep.json, per_source.json

- ChatGPT at optimal (theta=0.84, uw=5.0): modularity=0.752, sigma=12.3, 14 communities
- Claude Code at optimal (theta=0.88, uw=0.5): modularity=0.320, sigma=3.25, 7 communities
- Paper 1 comparison: modularity=0.749, sigma=14.9 (nomic-768, theta=0.90)
- Finding: structural laws replicate across embedding models

## Experiment 2: Platform-Dependent Role Weighting

**Script:** build_graph.py (weight sweep, per-source weight sweep)
**Results:** experiments/results/weight_sweep.json, per_source_weight_sweep.json

- ChatGPT optimal: uw=3.0 (user text discriminative)
- Claude Code optimal: uw=0.5 (assistant text discriminative)
- 2D sweep (theta x uw) for ChatGPT: experiments/results/chatgpt_2d_sweep.json

## Experiment 3: Marginalia as Structural Signal

**Script:** analyze_marginalia.py
**Results:** experiments/results/marginalia_analysis.json

- 17 annotated conversations, 12 in giant component
- Betweenness: 3.9x higher (p=0.003 permutation, 99.7th percentile)
- Bridge score: 3.1x higher (p=0.003 permutation, 99.7th percentile)
- Degree: no significant difference (p=0.655)
- Interpretation: annotated nodes are bridges, not hubs

## Experiment 4: Latent Trail Detection (Mean Similarity)

**Script:** detect_trails.py --mode mean
**Results:** experiments/results/robust_trails.json

- 72 trails (76 at slightly different params), longest=10
- 109 conversations robust across 5 parameter settings
- 20 robust trails saved with commentary template
- Trail R19 (10 steps, 3 years): simulation hypothesis -> consciousness -> pain -> Kahneman

## Experiment 5: Null Model (Temporal Shuffle)

**Script:** inline (1000 iterations in background, 200 iterations foreground)
**Results:** experiments/results/ (inline output, both runs consistent)

- Real: 72 trails, 268 total steps
- Null: 100.1 trails, 416.1 steps (mean of 1000 shuffles)
- Z-score: -4.60 (real has FEWER trails than random)
- Interpretation: real data is temporally bursty. Trails are rare genuine returns,
  not artifacts. Fewer trails = each one is more meaningful.

## Experiment 6: Trail-Marginalia Overlap

**Script:** inline analysis
**Results:** experiments/results/marginalia_analysis.json (extended)

- 5 of 17 annotated convs are trail waypoints (29% vs 9.2% base rate)
- Fisher exact test: odds ratio=4.17, p=0.016
- Composite significance score: annotated 1.5x higher (p=0.003)
- 13.8x enrichment in top 20 most significant conversations

## Experiment 7: Exploratory Probes

**Script:** inline analyses (probes 1-10)
**Key results:**

- K-core: innermost core (k=162) is 100% Claude Code
- Outliers: practical/personal queries (banking, recipes, dog policy)
- Convergent thinking: 100 near-duplicate pairs (mostly same-day)
- Cross-platform migration: only 6 pairs above 0.88 cross-source
- Platform isolation: communities 87-100% single-source, 7.2% cross-source edges
- Community 6 is the only mixed-source community (61% CC, 37% ChatGPT)
- Temporal recurrence: revisitation patterns with 90+ day gaps
- Community temporal signatures: communities are temporally clustered by platform era
- Rich-club: phi rises monotonically (high-degree nodes cluster together)
- Intra-conversation drift: Claude Code slightly higher drift than ChatGPT

## Experiment 8: Exponential Embeddings

**Script:** exponential_embeddings.py --alpha 0.85
**Results:** terminal output (2026-04-10)

- 2,648 conversations with >= 3 non-short messages
- 513 end-to-start links (sim >= 0.80) vs 1,281 mean-to-mean links
- End-to-start links are 2.5x rarer (higher precision for continuation detection)
- Notable: "New Terrifying Concepts" END -> "Understanding Consciousness" START (1 year gap)
- Notable: "AGI Fast-Takeoff" END [annotated] -> Claude Code implementation START
- 79% of similar-mean pairs are divergent (conversations branch, not repeat)
- Drift distribution: mean=0.275, uniform across platforms

## Experiment 9: Refined Trail Detection (Surprise Scoring + End-to-Start)

**Script:** detect_trails.py --mode both
**Results:** terminal output (2026-04-10)

- Mean trails: 56 (with surprise scoring, threshold 0.82)
- End-to-start trails: 7 (much rarer, higher precision)
- Jaccard overlap: 0.023 (nearly orthogonal methods)
- Top surprise-scored trail (mean mode): RPDG -> 292d gap -> 193d gap (surprise=5.13)
- End-to-start Trail 2: "Reflecting on Suffering" -> "Uncle's Last Moments" -> "Philosophical Skepticism" (existential continuation chain)

## Experiment 10: Trail DAG (Branching Structure)

**Script:** detect_trails_dag.py
**Results:** experiments/results/trail_dag.json

- DAG: 2,648 nodes, 1,974 edges (1,727 revisitation, 247 continuation)
- Max out-degree: 3 (by design, max_links=3)
- Max in-degree: 103 (Claude Code convergence sink)
- Fan-out nodes reveal intellectual branching (e.g., "RNNs VS Transformers" spawns 3 threads)
- Key convergence node: "Distilling the ARC" (in=41), gravitational attractor in ChatGPT portion
- Annotated nodes are "structural joints": high in-degree AND out-degree
  - "Bayesian Perspective on AI": in=6, out=3
  - "Understanding Consciousness": in=6, out=3
  - "AGI Fast-Takeoff": in=1, out=3
- Longest paths (10 steps) all converge through "Distilling the ARC"

## Experiment 11: Exponential Embeddings

**Script:** exponential_embeddings.py --alpha 0.85
**Results:** terminal output (2026-04-10)

- 513 end-to-start links vs 1,281 mean-to-mean links (2.5x rarer)
- End-to-start links are nearly orthogonal to mean links (Jaccard=0.023)
- 79% of high-similarity conversation pairs are divergent (same topic, different endings)
- Notable continuation: "New Terrifying Concepts" END -> "Understanding Consciousness" START (1 year)
- Drift uniform across platforms: mean=0.275

## Experiment 12: Refined Trail Detection with Surprise Scoring

**Script:** detect_trails.py --mode both
**Results:** terminal output (2026-04-10)

- Surprise score: similarity * log(1 + gap_days)
- Mean trails: 56 (with surprise ranking), end-to-start: 7
- The two methods find nearly disjoint trails (Jaccard=0.023)
- Trail types: "revisitation" (topic return) vs "continuation" (picked up where left off)

## Experiment 13: Alpha Ablation Study

**Script:** inline
**Results:** experiments/results/alpha_ablation.json

- Alpha sweep: [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
- Clear precision-recall tradeoff:
  - alpha=0.50 (half-life 1 msg): 59 links, 2 trails (high precision, low recall)
  - alpha=0.85 (half-life 4.3 msgs): 212 links, 11 trails (sweet spot)
  - alpha=0.90 (half-life 6.6 msgs): 333 links, 24 trails (good balance)
  - alpha=0.99 (half-life 69 msgs): 563 links, 45 trails (degenerates to mean)
- End-mean divergence: 0.153 at alpha=0.50, 0.001 at alpha=0.99
- At alpha >= 0.95, exp embedding is effectively the mean (divergence < 0.015)
- Interesting regime: alpha=0.85-0.90 (half-life 4-7 messages)
- Key insight: "semantic half-life" gives interpretable, principled parameterization
  of how quickly a conversation forgets its beginning

## Literature Context: Exponential Embeddings

- **Not novel in principle, novel in application.**
- MEGA (ICLR 2023): EMA in transformer attention
- DemaFormer (EMNLP 2023): damped EMA for temporal language grounding (learnable alpha)
- weg2vec (Sci Reports 2020): exponential decay in temporal network event embeddings (closest)
- ETSformer (2022): exponential smoothing for time series
- Our contribution: post-hoc exponential weighting of frozen message embeddings to derive
  trajectory-aware conversation representations (start vs end), enabling
  continuation-based trail detection. The alpha parameter maps to a "semantic half-life."

## Experiment 14: DAG Alpha Sensitivity

**Script:** experiments/exp14_dag_alpha.py
**Results:** experiments/results/dag_alpha_sensitivity.json

- As alpha increases: revisitation links decrease (1,778 to 1,403), continuation links increase (64 to 821)
- Fan-out nodes increase monotonically (570 to 696): longer memory enables more branching
- Convergence nodes peak at alpha=0.85 (268), then plateau
- Longest path: 10 at alpha<=0.85, drops to 9 at alpha>=0.90
- Structural transition at alpha=0.85-0.90: below is revisitation-dominated, above is continuation-rich
- alpha=0.85 confirmed as principled choice: max convergence, balanced link types, longest paths intact

## Experiment 15: Revisitation vs Continuation Link Properties

**Script:** experiments/exp15_link_types.py
**Results:** experiments/results/link_type_analysis.json

- 9,920 revisitation links vs 868 continuation links (Jaccard=0.034, orthogonal)
- Continuation links 1.8x more likely to cross platforms (7.6% vs 4.3%)
- Continuation links originate from lower-drift conversations (0.265 vs 0.300, p<0.0001)
- Continuation links 1.5x more likely to cross communities (25.1% vs 17.0%)
- Interpretation: continuation captures "carried forward into new territory," revisitation captures "returned to same topic." Structurally distinct cognitive patterns.

## Experiment 16: Recurring Message-Level Ideas

**Script:** experiments/exp16_recurring_messages.py
**Results:** experiments/results/recurring_messages.json

- 45,794 user messages clustered into 200 semantic clusters (MiniBatchKMeans)
- 99 of 200 clusters span 2+ years (persistent ideas)
- 103 of 200 clusters are cross-source (ideas migrate across platforms)
- Cluster 71: 188 convs, cohesion=0.980, 100% Claude Code (core dev pattern)
- Annotated conversations: 27.6x higher cluster density (0.1076 vs 0.0039 clusters/msg)
- Annotated convs touch 43% of all idea clusters despite 1.6% of messages
- Interpretation: annotated conversations are semantically diverse (many idea domains per message), consistent with bridge/betweenness findings

## Experiment 17: Deep Exponential Embedding Analysis

**Script:** experiments/exp17_exp_embedding_deep.py
**Results:** experiments/results/exp_embedding_deep.json

### Analysis 1: Trajectory similarity
- Trajectory divergence (same topic, different path) is common (delta up to -0.42)
- Trajectory convergence (different topic, same path) is weak (max delta=+0.10)
- Conversations branch, they don't repeat the same journey

### Analysis 2: User vs assistant trajectory divergence
- Claude Code: mean divergence=0.260 (user and AI end up in different semantic places)
- ChatGPT: mean divergence=0.220 (user and AI stay more aligned)
- Highest divergence: long Claude Code sessions (assistant ranges across files, user stays focused)
- Lowest divergence: deep conceptual ChatGPT conversations (user and AI explore together)

### Analysis 3: Adaptive alpha (per-conversation optimal decay rate)
- Mean optimal alpha: 0.785, median: 0.800
- ChatGPT: alpha=0.762 (half-life ~2.4 msgs, faster topic evolution)
- Claude Code: alpha=0.827 (half-life ~4.3 msgs, slower, more focused)
- Distribution: roughly normal centered at 0.80, range 0.35-0.95
- Correlation with message count: r=0.039 (length doesn't predict decay rate)
- Key insight: adaptive alpha is a per-conversation MEASUREMENT, not a parameter. "This conversation had a semantic half-life of N messages."

## Experiment 18: Conversation Lens Family

**Script:** experiments/exp18_lens_family.py
**Results:** experiments/results/lens_family.json

- 8 lenses compared: mean, exponential, reverse_exp, surprise, gaussian, first_only, last_only, bookend
- Surprise lens nearly identical to mean (divergence=0.013). Surprises cancel out.
- Exponential and reverse_exp have symmetric divergence (0.060) but different trail counts (345 vs 477). Openings are more predictive than conclusions.
- First-only and last-only are maximally orthogonal (neighbor overlap=0.07).
- NN agreement matrix shows lenses form a hierarchy: mean/surprise (core), gaussian (moderate), exp/rev (directional), first/last (extreme).
- Annotated conversations most distinctive under last_only lens (1.33x ratio). Annotated conversations end in similar places.

## Experiment 19: Changepoint Detection

**Script:** experiments/exp19_changepoints.py
**Results:** experiments/results/changepoints.json

- 1,194 conversations with >= 20 messages analyzed
- Mean segments: 3.1, median: 3.0 (conversations naturally have ~3 phases)
- Max segments: 6 (trajectory method). Pairwise: up to 201.
- Weak correlation with length (r=0.129). Topic shifts are about content, not length.
- Claude Code: 3.4 segments (multi-task sessions). ChatGPT: 2.9.
- The 18,741-message session decomposes into 5 segments.
- Trajectory method is appropriately conservative vs pairwise/window.

## Open Questions
- Can adaptive alpha predict conversation "quality" or "importance"?
- Is the platform-dependent alpha a universal finding or specific to this user?
- Trail DAG visualization: how to render branching intellectual histories?
- Exponential embeddings as a standalone paper: semantic half-life as a new conversation-level feature
- Bush's missing primitives: trail sharing, cross-archive trails, typed link semantics
