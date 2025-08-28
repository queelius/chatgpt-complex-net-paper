# Ablation Study Design: User-AI Embedding Weight Ratios

## Overview

This ablation study investigates how different weighting ratios between user and AI messages affect the semantic network structure of ChatGPT conversations. We systematically vary the relative importance of user queries versus AI responses in the embedding generation process.

## Research Questions

1. How does the relative weighting of user vs AI messages affect network topology?
2. Is there an optimal ratio for capturing conversation semantics?
3. Do different ratios reveal different community structures or bridge patterns?
4. How does the degree distribution change with different weightings?

## Experimental Design

### Selected Weight Ratios (User:AI)

| Ratio | User Weight | AI Weight | Label | Motivation |
|-------|------------|-----------|--------|------------|
| 100:1 | 100.0 | 1.0 | `user100.0-ai1.0` | Near-pure user perspective (99% user, 1% AI trace) |
| 4:1 | 4.0 | 1.0 | `user4.0-ai1.0` | Strong user emphasis - heavily prioritizes human intent |
| 2:1 | 2.0 | 1.0 | `user2.0-ai1.0` | Paper's ratio - proven effective in original study |
| φ:1 | 1.618 | 1.0 | `user1.618-ai1.0` | Golden ratio - natural/aesthetic balance (user-favored) |
| 1:1 | 1.0 | 1.0 | `user1.0-ai1.0` | Perfect balance - baseline control condition |
| 1:φ | 1.0 | 1.618 | `user1.0-ai1.618` | Inverse golden ratio - natural balance (AI-favored) |
| 1:2 | 1.0 | 2.0 | `user1.0-ai2.0` | Inverse paper ratio - AI knowledge emphasis |
| 1:4 | 1.0 | 4.0 | `user1.0-ai4.0` | Strong AI emphasis - heavily prioritizes AI responses |
| 1:100 | 1.0 | 100.0 | `user1.0-ai100.0` | Near-pure AI perspective (1% user trace, 99% AI) |

### Design Justification

1. **Symmetric Design**: Four ratios on each side of the balanced condition (1:1) provide equal exploration of user-emphasized and AI-emphasized embeddings.

2. **Range Selection**: The 100:1 to 1:100 range captures near-pure perspectives (99% emphasis) while avoiding true zero weights that could cause computational issues.

3. **Strategic Spacing**: Including 100:1, 4:1, 2:1, and φ:1 provides coverage from extreme to moderate user emphasis, with symmetric coverage on the AI side (1:φ, 1:2, 1:4, 1:100).

4. **Golden Ratio Inclusion**: The golden ratio (φ ≈ 1.618) appears throughout nature and design as an optimal proportion. Testing whether this translates to semantic balance in conversation networks could reveal fundamental properties of human-AI interaction.

5. **Benchmark Comparisons**: Including the paper's 2:1 ratio allows direct comparison with published results, while its inverse (1:2) tests the opposite hypothesis.

## Implementation

### Directory Naming Convention

```
chatgpt-json-llm-user{user_weight}-ai{ai_weight}/
```

Examples:
- `chatgpt-json-llm-user100.0-ai1.0/`
- `chatgpt-json-llm-user4.0-ai1.0/`
- `chatgpt-json-llm-user1.618-ai1.0/`
- `chatgpt-json-llm-user1.0-ai1.0/`
- `chatgpt-json-llm-user1.0-ai4.0/`
- `chatgpt-json-llm-user1.0-ai100.0/`

### Generation Commands

```bash
# Generate all 9 embedding sets
for ratio in "100.0:1.0" "4.0:1.0" "2.0:1.0" "1.618:1.0" "1.0:1.0" "1.0:1.618" "1.0:2.0" "1.0:4.0" "1.0:100.0"; do
    IFS=':' read -r user_w ai_w <<< "$ratio"
    python cli.py node-embeddings \
        --input-dir ../dev/chatgpt-4-11-2025_json_no_embeddings \
        --method role-aggregate \
        --embedding-method llm \
        --user-weight $user_w \
        --assistant-weight $ai_w \
        --output-dir ../dev/chatgpt-json-llm-user${user_w}-ai${ai_w}
done
```

## Metrics for Comparison

### Network Topology Metrics
- **Degree Distribution**: Power law exponent, deviation from scale-free
- **Clustering Coefficient**: Local and global clustering patterns
- **Modularity**: Community detection strength
- **Giant Component Size**: Network connectivity

### Semantic Metrics
- **Bridge Conversations**: Number and strength of interdisciplinary connections
- **Community Coherence**: Semantic similarity within detected communities
- **Hub Identification**: Which conversations become central at different ratios

### Statistical Tests
- **KS Test**: Compare degree distributions across ratios
- **Modularity Significance**: Test community structure robustness
- **Correlation Analysis**: Relationship between ratio and network metrics

## Expected Outcomes

1. **Extreme User (100:1)**: Sparse network focused purely on user questions, revealing core human concerns
2. **Strong User (4:1, 2:1)**: Stronger clustering around user interests, clearer topic boundaries  
3. **Balanced (1:1)**: Most diverse network structure, balanced community detection
4. **Strong AI (1:2, 1:4)**: More uniform connectivity, potential knowledge graph structure
5. **Extreme AI (1:100)**: Dense network of AI knowledge connections, revealing LLM's internal knowledge structure
6. **Golden Ratios (φ:1, 1:φ)**: Potentially optimal balance between clustering and connectivity

## Analysis Pipeline

1. Generate embeddings for all 9 ratios
2. Create edge lists with consistent similarity threshold (0.9)
3. Export to GEXF format for Gephi visualization
4. Calculate network metrics for each ratio
5. Perform comparative statistical analysis
6. Generate visualization comparing key metrics across ratios

## Deliverables

1. Network visualizations for each ratio
2. Comparative metrics table
3. Statistical significance tests
4. Recommendation for optimal ratio based on specific use cases
5. Publication-ready figures showing ablation results