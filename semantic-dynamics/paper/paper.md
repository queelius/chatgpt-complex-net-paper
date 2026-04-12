# Semantic Dynamics of AI Conversation Archives: Velocity, Curvature, and Episodes in Embedding Space

## Abstract

We analyze a 3.4-year, 3,097-conversation personal AI archive spanning ChatGPT, Claude, and Claude Code. By embedding individual messages and treating conversations as trajectories through embedding space, we discover structure invisible to standard semantic analysis. First, we show that the derivative of the embedding trajectory (velocity) and its second derivative (curvature) define representation spaces that are nearly orthogonal to semantic space: community detection in velocity space finds conversation *archetypes* (how thinking moves), not topics (what thinking is about), with NMI of 0.13-0.21 against semantic communities. Second, we introduce the *semantic half-life*: a per-conversation decay parameter that measures how quickly a conversation forgets its beginning. This parameter is platform-dependent (ChatGPT: 2.4 messages, Claude Code: 4.3 messages), reflecting the different cognitive tempos of conversational versus agentic AI interaction. Third, we show that conversations decompose into ~3-7 natural episodes via sliding-window changepoint detection on embedding trajectories, with a universal episode density of approximately one topic shift per 12-15 messages. These findings reframe conversation archives as dynamical systems: the same calculus of projection, differentiation, and segmentation that applies to physical trajectories reveals cognitive structure in sequential text. We release `embflow`, a Python package implementing the embedding flow calculus.

## 1. Introduction

When a person uses an AI assistant, the resulting conversation is typically stored as a flat transcript: a sequence of messages, filed under a title, retrievable by keyword search. This representation captures *what* was discussed but discards *how* the discussion evolved. A conversation that opens with a philosophical question, pivots to fiction writing, and closes with personal reflection is stored identically to one that stays on a single technical topic from start to finish.

We propose treating conversations as *trajectories* through embedding space. Each message maps to a point in a high-dimensional semantic space via a pre-trained embedding model. The ordered sequence of these points traces a curve. This curve has a direction (velocity), a rate of change of direction (curvature), and a measurable deviation from its starting point (drift). These dynamical properties constitute a representation of the conversation that is distinct from, and largely orthogonal to, its semantic content.

This reframing is not merely metaphorical. We show empirically that:

1. **Velocity and curvature define independent representation spaces.** Graphs constructed from velocity similarity find different communities than graphs constructed from semantic similarity (NMI = 0.13-0.21). Velocity communities correspond to *conversation archetypes* (exploratory, focused, high-drift, steady) rather than topics.

2. **Conversations have a measurable cognitive tempo.** An exponentially weighted embedding, parameterized by a decay rate alpha, captures the "semantic half-life" of a conversation: how quickly the influence of earlier messages decays. This parameter varies systematically by platform (ChatGPT conversations evolve faster than Claude Code sessions) and can be estimated per conversation from the data itself.

3. **Conversations decompose into natural episodes.** Sliding-window changepoint detection on the embedding trajectory identifies topic shifts with a consistent density of approximately one episode per 12-15 messages, independent of conversation length. This challenges the standard treatment of conversations as atomic units and suggests that the true episodic units of a conversation archive are segments, not sessions.

These findings emerge from a single user's archive of 3,097 conversations (2,496 ChatGPT, 472 Claude Code, 129 Anthropic Claude) spanning December 2022 to April 2026, comprising 149,623 embedded messages. The analysis uses per-message embeddings (OpenAI text-embedding-3-small, 256 dimensions) stored with content-hash provenance for incremental updates. All computations use `embflow`, a Python package we release that implements the embedding flow calculus: composable lens functions for weighted aggregation, trajectory computation via scan, and derivative operations (velocity, curvature, angular velocity, drift).

The remainder of this paper is organized as follows. Section 2 reviews related work on conversation analysis, embedding aggregation, and temporal text representation. Section 3 describes the corpus and embedding infrastructure. Section 4 introduces the embedding flow calculus. Sections 5-7 present the three main findings (orthogonal spaces, semantic half-life, episode detection). Section 8 discusses implications and limitations. Section 9 concludes.

## 2. Related Work

### 2.1 Complex Network Analysis of Conversation Archives

Prior work by [the authors] has applied complex network analysis to AI conversation archives. Paper 1 (Towell, 2025) constructed semantic similarity graphs from 1,908 ChatGPT conversations, finding small-world topology (sigma = 14.9), Heaps' law vocabulary consolidation (beta = 0.286), and stable community structure matching predictions from Complementary Learning Systems theory. Paper 2 (Towell, 2026) extended this to Claude Code sessions, discovering that the densification exponent gamma = 1.41 is a near-universal constant across platforms, and that delegation (parent-child session spawning) and semantic similarity define orthogonal network layers.

The present work departs from this line by shifting from static graph analysis to dynamical analysis. Rather than asking "which conversations are similar?", we ask "how do conversations move through embedding space?" This yields qualitatively different insights: velocity communities are not semantic communities, and the dynamical properties of a conversation (tempo, drift, episode structure) are independent of its topic.

### 2.2 Embedding Aggregation

The standard approach to representing a multi-sentence document is mean pooling of token or sentence embeddings. Weighted variants include TF-IDF weighting, attention-based pooling, and position-aware schemes. The MEGA architecture (Ma et al., 2023) integrates exponential moving averages into transformer attention, providing an inductive bias for sequential structure. DemaFormer (Nguyen et al., 2023) uses a learnable damped exponential moving average for temporal language grounding. ETSformer (Woo et al., 2022) applies exponential smoothing to time-series forecasting.

Our approach differs from these in two ways. First, we apply exponential weighting *post-hoc* to pre-computed embeddings, not as part of model training. This makes the method applicable to any frozen embedding model. Second, we treat the decay parameter (alpha) not as a hyperparameter to tune but as a *measurable property* of the data: the semantic half-life of a conversation. The finding that this parameter varies systematically by platform and can be estimated per conversation from the embedding trajectory is, to our knowledge, new.

### 2.3 Temporal Network Embeddings

The weg2vec method (Torricelli et al., 2020) applies exponential decay to temporal network events, weighting recent interactions more heavily. This is the closest prior work to our approach: both use exponential decay to encode recency in an embedding. The key difference is that weg2vec embeds *events* (interactions between network nodes), while we embed *messages within a conversation*. Our trajectory and derivative operations (velocity, curvature) have no analogue in weg2vec.

### 2.4 Changepoint Detection

Bayesian online changepoint detection (Adams & MacKay, 2007) and kernel-based methods (Harchaoui et al., 2009) detect distributional shifts in sequential data. Topic segmentation in text has been studied extensively (Hearst, 1997; Eisenstein & Barzilay, 2008). Our sliding-window method is conceptually simple compared to these approaches, but it operates in embedding space rather than on raw text features, and it benefits from the smoothing properties of the embedding model. The empirical finding that the local mean stabilizes at approximately 10 messages provides a principled, non-arbitrary window size.

## 3. Data and Embedding Infrastructure

### 3.1 Corpus

The corpus consists of 3,097 conversations from a single user across three AI platforms:

| Platform | Conversations | Messages (non-short) | Temporal range |
|----------|-------------|---------------------|---------------|
| ChatGPT (OpenAI) | 2,496 | ~104,000 | Dec 2022 - Apr 2026 |
| Claude Code | 472 | ~42,000 | Nov 2025 - Apr 2026 |
| Anthropic Claude | 129 | ~3,600 | Mar 2024 - Aug 2025 |
| **Total** | **3,097** | **~149,600** | **3.4 years** |

ChatGPT conversations are human-driven: the user poses questions, explores topics, and drives the direction. Claude Code sessions are agent-driven: the user provides high-level directives while the assistant performs multi-step technical work (reading files, writing code, running tests). This distinction is central to several findings.

### 3.2 Per-Message Embeddings

Each message is embedded individually using OpenAI's text-embedding-3-small model at 256 dimensions (Matryoshka truncation). Messages shorter than 20 characters are flagged as "short" and optionally excluded from analysis. Messages exceeding 8,192 tokens are truncated via tiktoken.

Embeddings are stored in a SQLite database with sqlite-vec for vector operations. Each embedding record includes a SHA-256 content hash, enabling incremental updates: only new or modified messages are re-embedded on subsequent runs. The total storage is 266 MB for 149,623 embeddings.

The choice of 256 dimensions reflects a precision-recall tradeoff: at this dimensionality, the embedding model retains ~96-97% of retrieval quality relative to the full 3,072 dimensions, while reducing storage by 12x. For graph construction (thresholded cosine similarity), the rank ordering of similarities is what matters, and 256 dimensions preserve this ordering well.

### 3.3 Conversation-Level Embeddings via Lenses

A conversation's messages form a sequence of embedding vectors $e_1, e_2, \ldots, e_n$. A *lens* is a weight function $w: \{1,\ldots,n\} \to \mathbb{R}^+$ that produces a conversation-level embedding via weighted average:

$$\text{emb}_w = \frac{\sum_k w(k) \cdot e_k}{\|\sum_k w(k) \cdot e_k\|}$$

Different lenses capture different aspects of the conversation. We use:

- **Uniform**: $w(k) = 1$. The mean embedding. What the conversation is "about."
- **Exponential**: $w(k) = \alpha^{n-1-k}$. Recency-weighted. Where the conversation "ended up."
- **Reverse exponential**: $w(k) = \alpha^{k-1}$. Where the conversation "started."
- **Surprise**: $w(k) = 1 - \cos(e_k, \bar{e}_{<k})$. Messages that shifted the topic.

Lenses compose via pointwise multiplication of weight vectors. The Uniform lens is the identity element. Role weighting (e.g., user messages weighted 3x over assistant messages) is itself a lens (FieldWeight) that composes with any other.

## 4. The Embedding Flow Calculus

### 4.1 Trajectory (Scan)

The *trajectory* of a conversation under a lens is the running projection at each position:

$$\text{traj}(j) = \text{normalize}\left(\sum_{k=0}^{j} w(k, j) \cdot e_k\right)$$

where $w(k, j)$ denotes the weight assigned to message $k$ when projecting at position $j$. For the exponential lens, $w(k, j) = \alpha^{j-k}$. The trajectory traces a curve through embedding space. Its final point equals the lens projection of the full sequence.

### 4.2 Velocity and Curvature

The *velocity* at step $j$ is the first difference of the trajectory:

$$v_j = \text{traj}(j+1) - \text{traj}(j)$$

The velocity vector encodes the *direction and magnitude of semantic change*. Two conversations with similar velocity vectors at a given point are "changing in the same way," regardless of what they are about.

The *curvature* is the second difference:

$$c_j = v_{j+1} - v_j$$

High curvature indicates a sharp turn in the trajectory: a topic shift, a pivot, a moment where the conversation redirects. Changepoints are high-curvature events.

The *angular velocity* is the cosine distance between consecutive trajectory points:

$$\omega_j = 1 - \cos(\text{traj}(j), \text{traj}(j+1))$$

This scalar measures the rate of directional change, independent of the magnitude of the velocity vector.

### 4.3 Semantic Half-Life

The exponential lens is parameterized by $\alpha \in (0, 1]$. The *semantic half-life* is:

$$h = \frac{\log 0.5}{\log \alpha}$$

This gives the number of messages after which the influence of a given message has halved. At $\alpha = 0.85$, $h \approx 4.3$ messages: by the time 4 more messages have been exchanged, the earlier message contributes half as much to the trajectory.

The half-life is interpretable as a measure of *cognitive tempo*: how quickly the conversation forgets its recent past. Fast-tempo conversations (low $\alpha$, short half-life) shift topics rapidly. Slow-tempo conversations (high $\alpha$, long half-life) develop ideas steadily.

### 4.4 Episode Detection

A conversation naturally decomposes into *episodes*: contiguous segments that are internally coherent but semantically distinct from their neighbors. We detect episode boundaries via sliding-window divergence:

At each position $j$, we compute the cosine distance between the mean embedding of the $w$ messages before $j$ and the $w$ messages after $j$:

$$d(j) = 1 - \cos\left(\bar{e}_{j-w:j},\; \bar{e}_{j:j+w}\right)$$

Peaks in $d(j)$ above a threshold of $\mu + \sigma$ (where $\mu$ and $\sigma$ are the mean and standard deviation of $d$ over the sequence) identify episode boundaries.

We find empirically that the local mean of embeddings stabilizes at approximately 10 messages (Section 5.3), providing a principled default for $w$ that is independent of conversation length.

## 5. Results

### 5.1 Orthogonal Representation Spaces

We constructed similarity graphs in four representation spaces: semantic (uniform lens projection), velocity (mean of velocity vectors), exponential velocity (recency-weighted velocity), and curvature (mean of curvature vectors). To ensure comparable graph density, we thresholded each space at its 95th percentile of pairwise similarity.

**Table 1.** Graph properties at matched density (p95 threshold).

| Space | Threshold | Edges | GC nodes | GC% | Modularity | Communities |
|-------|-----------|-------|----------|-----|------------|-------------|
| Semantic | 0.654 | 79,077 | 1,701 | 95.6% | 0.429 | 8 |
| Velocity (mean) | 0.222 | 79,077 | 1,779 | 100% | 0.480 | 4 |
| Velocity (exp) | 0.166 | 79,077 | 1,779 | 100% | 0.374 | 5 |
| Curvature | 0.204 | 79,077 | 1,777 | 99.9% | 0.474 | 5 |

The similarity distributions differ markedly across spaces. Semantic similarities are centered at 0.43 (most conversations share some topical overlap). Velocity similarities are centered near zero (0.056) with high variance: most conversations do not change in the same way, but when they do, the signal is strong.

**Community agreement is low across spaces.** Normalized Mutual Information between community assignments:

| Pair | NMI |
|------|-----|
| Semantic vs. Velocity (mean) | 0.189 |
| Semantic vs. Curvature | 0.154 |
| Velocity (mean) vs. Curvature | 0.207 |
| Velocity (mean) vs. Velocity (exp) | 0.131 |

An NMI of 0.2 indicates that knowing the semantic community of a conversation tells you almost nothing about its velocity community (and vice versa). These spaces capture genuinely independent dimensions of conversation structure.

**Velocity communities are conversation archetypes.** The four velocity communities correspond not to topics but to dynamical styles:

| Community | Size | Mean drift | Mean speed | Dominant source | Interpretation |
|-----------|------|-----------|-----------|----------------|----------------|
| 0 | 784 | 0.459 | 0.202 | 94% ChatGPT | Exploratory (high drift, moderate speed) |
| 1 | 534 | 0.412 | 0.204 | 81% ChatGPT | Focused (moderate drift) |
| 3 | 329 | 0.350 | 0.197 | 66% Claude Code | Technical (lower drift, lower speed) |
| 2 | 132 | 0.553 | 0.208 | 80% Claude Code | High-drift sessions (most dynamic) |

The archetype split is primarily by platform and drift. Community 2 (132 high-drift Claude Code sessions) represents sessions where the user ranged across many technical tasks, while Community 0 (784 exploratory ChatGPT conversations) represents the large pool of diverse intellectual explorations.

**Degree correlation is moderate.** Hub conversations in semantic space tend to also be hubs in velocity space (Spearman rho = 0.576), but with substantial disagreement in the tails. We identified "velocity hubs" (high velocity degree, low semantic degree) that represent conversations with common transition patterns but niche topics, and "semantic hubs" (high semantic degree, low velocity degree) representing topically central but dynamically unremarkable conversations.

### 5.2 Semantic Half-Life and Adaptive Alpha

We estimated the optimal alpha per conversation by fitting the exponential trajectory to predict the next message (maximize mean cosine similarity between trajectory[j] and message[j+1]).

**Table 2.** Adaptive alpha by platform.

| Platform | Mean alpha | Median alpha | Semantic half-life (messages) |
|----------|-----------|-------------|------------------------------|
| ChatGPT | 0.762 | 0.750 | 2.4 |
| Claude Code | 0.827 | 0.850 | 4.3 |
| All | 0.785 | 0.800 | 3.1 |

ChatGPT conversations have a shorter semantic half-life: they shift topics roughly every 2-3 messages. Claude Code sessions maintain context longer, shifting every 4-5 messages. This reflects the different interaction modalities: in ChatGPT, the user drives the topic and may explore tangents freely; in Claude Code, the user provides a goal and the assistant executes multi-step work that stays on-task longer.

The distribution of alpha is approximately normal, centered at 0.80, with range 0.35-0.95. Correlation with conversation length is essentially zero (r = 0.039): short conversations can be fast-shifting or slow, and long conversations can be steady or erratic. Alpha measures cognitive tempo, not duration.

The alpha ablation (Section 5.2.1) shows a precision-recall tradeoff for end-to-start continuation links:

| Alpha | Half-life | End-to-start links | End-to-start trails |
|-------|-----------|-------------------|-------------------|
| 0.50 | 1.0 | 59 | 2 |
| 0.85 | 4.3 | 212 | 11 |
| 0.90 | 6.6 | 333 | 24 |
| 0.99 | 69.0 | 563 | 45 |

At low alpha, the end embedding captures only the last message, finding very few but very precise continuation links. At high alpha, the end embedding approaches the mean, and continuation links degenerate to topic similarity. The interesting regime is alpha = 0.85-0.90, where the end embedding is meaningfully different from the mean (end-mean divergence > 0.03) and enough links exist to form trails.

### 5.3 Episode Detection

Sliding-window changepoint detection with window size $w = 10$ and sensitivity $\mu + 0.8\sigma$ decomposes conversations into episodes.

**The local mean stabilizes at 10 messages.** We measured the number of messages required for the running mean embedding to stabilize (change by less than 0.5% with each additional message). Across 701 conversations, the stabilization point is 10-11 messages (mean = 10.4, median = 11), with a tight distribution (range 6-14). This provides a principled, non-arbitrary default for the window size.

**Episode density is independent of conversation length.**

| Conversation length | Mean episodes | Episodes per message |
|--------------------|--------------|---------------------|
| 20-50 messages | 2.2 | 1 per 16 |
| 51-100 | 3.6 | 1 per 21 |
| 101-200 | 6.4 | 1 per 24 |
| 201-500 | 12.4 | 1 per 28 |
| 500+ | 40.1 | 1 per 15 |

The ratio of episodes to messages is approximately constant at 1 per 12-25 messages across all length scales. A 50-message conversation and a 500-message conversation have comparable episode density. This suggests that topic shifts are a property of the conversational process, not an artifact of session length.

**Qualitative validation.** Manual inspection of segmented conversations confirms that episode boundaries correspond to genuine cognitive shifts. A 188-message conversation on consciousness ("Understanding Consciousness: Philosophy & Science") decomposes into 5 episodes under default parameters: (1) philosophical inquiry about the nature of consciousness, (2) thought experiments involving philosophical zombies, (3) a role-play exercise asking the AI to simulate a p-zombie, (4) collaborative fiction writing based on the thought experiments, and (5) a return to personal beliefs and meta-narrative. Each episode represents a distinct cognitive mode: inquiry, hypothetical reasoning, embodied simulation, creative production, and reflection.

A 1,764-message Claude Code game development session decomposes into 43 episodes, each corresponding to a distinct work task: keyboard controls, tile organization, documentation, collision systems, generator mode, UI refinement, terrain generation, prefabs, and agent placement. These episodes align with the natural unit of software development work: a focused task with a beginning (directive from user), a middle (implementation by assistant), and an end (testing and iteration).

### 5.4 Lens Family: Multiple Views of the Same Conversation

We compared eight lenses on the full corpus (2,298 conversations with 5+ non-short messages). The nearest-neighbor overlap matrix reveals which lenses agree on "what is similar to this conversation":

| | Uniform | Exp | RevExp | Surprise | Gaussian | First | Last | Bookend |
|---|---|---|---|---|---|---|---|---|
| Uniform | 1.00 | 0.37 | 0.38 | 0.78 | 0.57 | 0.16 | 0.14 | 0.34 |
| Exp | | 1.00 | 0.21 | 0.35 | 0.25 | 0.11 | 0.24 | 0.26 |
| RevExp | | | 1.00 | 0.33 | 0.23 | 0.30 | 0.10 | 0.32 |
| First | | | | | | 1.00 | 0.07 | 0.24 |
| Last | | | | | | | 1.00 | 0.16 |

Two findings stand out. First, the Surprise lens agrees strongly with Uniform (overlap = 0.78): topic-shifting messages do not dominate the aggregate embedding because they point in different directions and cancel out. The Surprise lens is useful for identifying *which* conversations have shifts, not for producing a different embedding.

Second, First-only and Last-only lenses have near-zero neighbor overlap (0.07). The conversations most similar to where you started are almost completely disjoint from those most similar to where you ended. This quantifies what it means for a conversation to "go somewhere": the beginning and end of a conversation live in different neighborhoods of embedding space.

### 5.5 Continuation vs. Revisitation

Two types of inter-conversation links emerge from the embedding flow calculus:

- **Revisitation links** (mean-to-mean similarity): "I returned to this topic."
- **Continuation links** (end-to-start similarity): "I picked up where I left off."

These link types are nearly orthogonal (Jaccard = 0.034 overlap). Among all links:

| Property | Revisitation | Continuation |
|----------|-------------|-------------|
| Count | 9,920 | 868 |
| Cross-platform rate | 4.3% | 7.6% |
| Cross-community rate | 17.0% | 25.1% |
| Source conversation drift | 0.300 | 0.265 |

Continuation links are 1.8x more likely to cross platform boundaries than revisitation links. When you pick up where you left off, you're more likely to do it on a different platform (e.g., explore in ChatGPT, implement in Claude Code). Continuation links also cross community boundaries 1.5x more often: the end of one intellectual thread connects to the beginning of a different knowledge domain.

Continuation links originate from lower-drift conversations (0.265 vs. 0.300, p < 0.0001). Focused conversations produce coherent endings that are easy to continue from. Wandering conversations do not have a clear endpoint to pick up.

## 6. Discussion

### 6.1 Conversations as Dynamical Systems

The central finding of this work is that conversation archives have a dynamical structure that is invisible to static semantic analysis. Semantic graphs answer "what is this conversation about?" Velocity and curvature graphs answer "how does the conversation move?" and "where does it turn?" These questions have different answers: the NMI between semantic and velocity communities is 0.13-0.21, comparable to the correlation between orthogonal random partitions.

This suggests that a complete representation of a conversation archive requires multiple views, analogous to how a complete description of a physical system requires both position and momentum. The semantic embedding is position. The velocity embedding is momentum. Neither alone is sufficient.

### 6.2 Episodes, Not Conversations

The episode detection results challenge the standard practice of treating conversations as atomic units. A 1,764-message Claude Code session is not one cognitive unit; it is 43 distinct episodes that share a session container. Treating it as a single node in a semantic graph distorts the graph's structure by averaging over dozens of distinct topics.

This connects to Complementary Learning Systems theory (McClelland et al., 1995; Kumaran et al., 2016), which distinguishes between episodic memory (individual experiences) and semantic memory (consolidated knowledge). In our framing, episodes (segments) are the episodic units. Sessions (conversations) are an artifact of the platform. Trails (links between episodes across sessions) are the consolidation process.

### 6.3 Limitations

**Single user.** All findings derive from one person's archive. The episode density (~1 per 12-15 messages), the platform-dependent alpha, and the conversation archetypes may be specific to this user's interaction style. Replication across users is needed.

**Embedding model dependence.** The stabilization window of 10 messages and the similarity distributions are properties of text-embedding-3-small at 256 dimensions. Different embedding models or dimensionalities may yield different values.

**No ground truth for episodes.** Episode boundaries are evaluated qualitatively, not against a labeled dataset. The sliding-window method is simple but may miss gradual transitions that occur over many messages.

**N=1 is also N=3,097.** While the corpus comes from one user, it spans 3,097 conversations over 3.4 years across three platforms. The findings are robust within this archive; whether they generalize is an empirical question for future work.

## 7. Conclusion

We have shown that AI conversation archives possess a dynamical structure that complements their semantic structure. By treating conversations as trajectories through embedding space and applying a calculus of projection, differentiation, and segmentation, we discovered three independent dimensions of organization: topic (semantic space), transition pattern (velocity space), and structural shape (curvature space). We introduced the semantic half-life as an interpretable measure of conversational tempo and showed that conversations naturally decompose into episodes at a consistent density.

These findings suggest that personal AI archives are richer objects than they appear. The flat transcript is a lossy representation. The trajectory, its derivatives, and its episodes capture cognitive structure that may prove useful for retrieval, recommendation, and self-knowledge.

## References

Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv:0710.3742.

Eisenstein, J., & Barzilay, R. (2008). Bayesian unsupervised topic segmentation. EMNLP.

Harchaoui, Z., Moulines, E., & Bach, F. (2009). Kernel change-point analysis. NeurIPS.

Hearst, M. A. (1997). TextTiling: Segmenting text into multi-paragraph subtopic passages. Computational Linguistics.

Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary Learning Systems theory updated. Trends in Cognitive Sciences.

Ma, X., et al. (2023). Mega: Moving average equipped gated attention. ICLR.

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. Psychological Review.

Nguyen, T., et al. (2023). DemaFormer: Damped exponential moving average transformer with energy-based modeling for temporal language grounding. Findings of EMNLP.

Torricelli, M., Karsai, M., & Gauvin, L. (2020). weg2vec: Event embedding for temporal networks. Scientific Reports.

Towell, A. (2025). Cognitive MRI of AI conversation archives: Complex network analysis of a ChatGPT corpus. [comp-net-2025].

Towell, A. (2026). From conversation to delegation: Multi-layer network analysis of agentic AI sessions. [ISCS2026].

Woo, G., et al. (2022). ETSformer: Exponential smoothing transformers for time-series forecasting. arXiv:2202.01381.
