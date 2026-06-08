# The Operational Memex: From Cognitive Structure to Navigable Knowledge

**Working title**: "The Operational Memex: Cross-Platform Cognitive Architecture in Personal AI Conversation Archives"

**Thesis**: The structural laws discovered in single-platform AI conversation archives (small-world topology, Heaps' law consolidation, densification universality) generalize across platforms. Furthermore, user annotations (marginalia) correlate with structurally significant graph positions, suggesting that Bush's memex primitives (trails and marginalia) are not arbitrary design choices but reflect genuine cognitive architecture.

## Relation to Prior Work

- **Paper 1** (Cognitive MRI, comp-net-2025): Established semantic graph methodology on 1,908 ChatGPT conversations. Found small-world σ=14.9, Heaps β=0.286, γ=1.41 densification.
- **Paper 2** (Agentic, ISCS2026): Extended to Claude Code with delegation layer. Found γ=1.41 universality, orthogonal layers, Knowledge Archipelago vs Cognitive Web architectures.
- **This paper**: Cross-platform replication (ChatGPT + Claude + Gemini + Claude Code), marginalia as attention signal, the memex system as reproducible infrastructure.

## Key Differences from Prior Papers

- **Embedding model**: text-embedding-3-large (3072-dim) vs nomic-embed-text (768-dim). Enables ablation: are structural findings model-dependent?
- **Multi-source corpus**: ~3,500+ conversations across 4 platforms (excluding subagent/full-fidelity)
- **Marginalia layer**: First study of user annotation behavior in conversation archives
- **Operational framing**: Not just analysis; the system (memex) is itself a contribution

## Structure

### §1 Introduction
- Bush's 1945 vision: trails, marginalia, associative indexing
- Gap: prior work shows structure exists but doesn't operationalize it
- This paper: (1) cross-source replication, (2) marginalia as signal, (3) operational system
- Research questions:
  1. Do structural laws (small-world, Heaps, densification) replicate across AI platforms?
  2. Do user annotations correlate with structural significance in the semantic graph?
  3. Can conversation archives be made navigable through graph-native operations?

### §2 Background & Related Work
- Cognitive MRI framework (our prior work)
- Vannevar Bush and the memex (1945)
- Personal information management (PIM) and personal knowledge management (PKM)
- Complex networks in knowledge systems
- RAG and semantic retrieval

### §3 Data & Methods
- **§3.1 Corpus**: Multi-platform archive from a single user
  - Sources: ChatGPT, Anthropic Claude, Google Gemini, Claude Code (conversation-only)
  - Per-source statistics: count, temporal range, message counts
  - Exclusions: claude_code_full subagent sessions (machine-generated scaffolding)
- **§3.2 Embedding**: OpenAI text-embedding-3-large (3072-dim)
  - Conversation-level: concatenated role-prefixed messages
  - Storage: sqlite-vec in analysis database
- **§3.3 Graph construction**: Cosine similarity thresholding
  - Threshold selection via percolation analysis (sweep)
  - Edge weight = similarity score
- **§3.4 Network analysis**: Same pipeline as prior papers for comparability
  - Community detection (Louvain), small-world σ, degree distribution
  - Heaps' law vocabulary growth, densification exponent γ
- **§3.5 Marginalia analysis**
  - Notes from memex schema v4 (target_kind: message/conversation)
  - Structural properties of annotated nodes: degree, betweenness, community membership

### §4 The Memex System
- Architecture: SQLite + FTS5, convention-based importers/exporters
- MCP server for agent access (6 tools)
- Schema: conversations, messages, tags, enrichments, notes, provenance
- Multi-database configuration
- Reproducibility: `pip install py-memex`, full pipeline in this repo

### §5 Experiment 1: Cross-Source Replication
- **RQ1**: Do structural laws generalize?
- Per-source graphs at matched θ
- Metrics table: σ, modularity, γ, Heaps β, degree distribution α
- Key comparison: prior ChatGPT-only vs this paper's ChatGPT (different embedder)
- Cross-source combined graph: does mixing sources change topology?

### §6 Experiment 2: Marginalia as Attention Signal
- **RQ2**: Do annotated conversations occupy structurally significant positions?
- Compare annotated vs non-annotated nodes:
  - Degree centrality
  - Betweenness centrality
  - Community-bridging (inter-community edge fraction)
  - Concept density (if concept extraction is included)
- Statistical tests: Mann-Whitney U (non-parametric, accounts for skewed distributions)
- Null model: random annotation assignment, measure effect size

### §7 Experiment 3: Navigability / Trails (exploratory)
- **RQ3**: Can graph structure guide navigation?
- Trail suggestion: given seed conversation, rank neighbors by similarity
- Evaluation: user validates suggested trails (qualitative)
- Bush's vision actualized: named paths through the semantic graph
- (This section may be brief or become future work depending on what the data shows)

### §8 Discussion
- Do platforms produce different cognitive architectures?
- What does marginalia tell us about attention in knowledge archives?
- Embedding model sensitivity (comparison with prior nomic results)
- Limitations: N=1 user, API-dependent embeddings, self-annotation bias
- Future: trails as first-class graph layer, cross-domain federation

### §9 Conclusion
- Bush was right about marginalia: annotations mark structurally important nodes
- Structural laws are platform-invariant (or aren't: either finding is valuable)
- Open-source memex enables reproducible personal knowledge research

## Figures (planned)

- F1: Corpus overview (per-source timeline, conversation counts)
- F2: Threshold sweep / percolation plot
- F3: Per-source network visualizations (side by side)
- F4: Metrics comparison table (this paper vs prior papers)
- F5: Degree distribution (log-log) per source
- F6: Heaps' law plot per source
- F7: Densification plot per source
- F8: Marginalia analysis (annotated vs non-annotated node properties)
- F9: Community membership of annotated nodes

## Pipeline

```bash
# 1. Extract corpus from memex databases
python extract_corpus.py --output-dir data/corpus

# 2. Compute embeddings (requires OPENAI_API_KEY)
python compute_embeddings.py --corpus-dir data/corpus --db data/analysis.db

# 3. Threshold sweep (find good θ)
python build_graph.py --db data/analysis.db --sweep 0.80 0.95 0.01

# 4. Build graph and analyze
python build_graph.py --db data/analysis.db --threshold 0.90 --per-source

# 5. (Future) Marginalia analysis
# python analyze_marginalia.py --db data/analysis.db

# 6. (Future) Temporal analysis
# python temporal.py --db data/analysis.db
```
