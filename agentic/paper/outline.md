# Paper: Hierarchical Memory Structure in AI Conversation Archives

**Working title**: "From Episodes to Abstractions: Emergent Hierarchical Memory in 1,908 AI Conversations"

**Thesis**: AI conversation archives exhibit hierarchical memory structure analogous to human cognitive architecture. LLM-extracted semantic concepts from individual episodes naturally organize into a multi-level hierarchy (concepts → meta-concepts → themes → domains) that satisfies key predictions of Complementary Learning Systems theory and reproduces the small-world topology of human semantic memory networks (Steyvers & Tenenbaum, 2005).

## Structure

### 1. Introduction
- AI conversations as externalized cognitive artifacts
- Gap: treating conversation archives as flat lists vs structured knowledge
- CLS theory prediction: episodic memories should consolidate into semantic structure
- Research questions:
  1. Does hierarchical structure emerge from bottom-up concept extraction?
  2. Does the concept vocabulary grow sublinearly (consolidation signature)?
  3. Does the concept co-occurrence network exhibit small-world properties?
  4. How does the emergent hierarchy compare to geometric (embedding-based) clustering?

### 2. Related Work
- Complementary Learning Systems (McClelland et al., 1995; Kumaran et al., 2016)
- Semantic memory networks (Steyvers & Tenenbaum, 2005; Collins & Loftus, 1975)
- Prototype theory / categorization (Rosch, 1975; Rosch & Mervis, 1975)
- Knowledge graphs from text (automatic ontology construction)
- Network analysis of conversation data (our prior work)
- LLM-based concept extraction

### 3. Data and Methods
- **3.1 Dataset**: 1,908 ChatGPT conversations, Dec 2022 – Apr 2025
- **3.2 Concept extraction**: Claude Code (Sonnet) extracts 3-5 noun-phrase concepts per episode; 19 parallel agents; 6,275 raw concepts
- **3.3 Concept embedding**: nomic-embed-text (768-dim)
- **3.4 Hierarchical clustering**: Ward linkage on concept embeddings → 4-level hierarchy (500 meta-concepts → 50 themes → 8 domains)
- **3.5 Bipartite graph**: Many-to-many episode ↔ concept links
- **3.6 Co-occurrence network**: Meta-concepts linked if they co-occur in episodes
- **3.7 Comparison baseline**: Geometric hierarchy from episode embeddings (Experiment 1)

### 4. Results
- **4.1 Concept extraction**: 6,275 concepts, 3.5/episode, 95% singletons → meta-concept dedup
- **4.2 Emergent hierarchy**: 8 interpretable domains, 69% episodes span 2+ domains
- **4.3 Vocabulary growth**: Heaps' law β=0.286 (consolidation signature)
- **4.4 Small-world topology**: σ=4.8-5.5, matching Steyvers & Tenenbaum
- **4.5 Scale-free degree distribution**: α≈3.0
- **4.6 Continuous semantic space**: Silhouette monotonically increases with k
- **4.7 Complementarity with geometric hierarchy**: NMI=0.26

### 5. Discussion
- CLS signatures: sublinear vocabulary growth = consolidation
- Small-world match validates cognitive analogy
- Many-to-many structure captures cross-domain integration (unlike partitions)
- The hierarchy is imposed (useful simplification) not discovered (natural boundaries)
- Limitations: parallel extraction, single user, LLM sensitivity
- Implications for AI memory systems, RAG architectures, knowledge management

### 6. Conclusion

## Key Figures
1. Domain composition bar chart (Fig. 1)
2. Heaps' law vocabulary growth (Fig. 2)
3. Concept frequency + degree distributions (Fig. 3)
4. Domain co-occurrence heatmap (Fig. 4)
5. Silhouette scan (Fig. 5)

## Key Numbers
| Metric | Value |
|---|---|
| Episodes | 1,908 |
| Raw concepts | 6,275 |
| Meta-concepts | 500 |
| Themes | 50 |
| Domains | 8 |
| Small-world σ | 4.8-5.5 |
| Heaps' β | 0.286 |
| Episodes spanning 2+ domains | 69% |
| Semantic vs geometric NMI | 0.26 |
