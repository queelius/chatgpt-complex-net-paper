# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Academic research repository analyzing ChatGPT conversations through complex network analysis. Contains both the LaTeX papers/presentations and Python implementation for network generation and analysis of 449 ChatGPT conversations as semantic networks.

## Project Structure

```
.
├── class_paper/                       # Original class paper version
│   ├── alex-chatgpt-complex-net.tex  # Main research paper
│   ├── alex-pres-complex-networks.tex # Beamer presentation
│   └── images/                        # 47+ network visualizations
├── comp-net-2025-paper/              # Conference submission version
│   ├── paper.tex                     # Updated paper for conference
│   └── images/                        # Enhanced visualizations
├── code/                              # Python implementation
│   ├── cli.py                        # Main CLI interface
│   ├── networks.py                   # Core network generation
│   ├── rec-conv.py                   # Recommendation system
│   ├── embedding/                    # Embedding generation modules
│   ├── graph/                        # Graph processing utilities
│   └── data/                         # Data processing utilities
└── dev/                              # Research data
    └── chatgpt-4-11-2025_json_no_embeddings/  # 1908 raw ChatGPT conversation logs (no embeddings)

```

## Common Development Commands

### LaTeX Document Building

```bash
# Build class paper version
cd class_paper
pdflatex alex-chatgpt-complex-net.tex
bibtex alex-chatgpt-complex-net        # If references change
pdflatex alex-chatgpt-complex-net.tex  # Second pass
pdflatex alex-chatgpt-complex-net.tex  # Final pass

# Build conference paper version  
cd comp-net-2025-paper
pdflatex paper.tex
bibtex paper                           # If references change
pdflatex paper.tex                     # Second pass
pdflatex paper.tex                     # Final pass

# Build presentations
pdflatex alex-pres-complex-networks.tex

# Convert SVG images to PDF
cd images && ./export-svg.sh
```

### Python Code Commands

```bash
# Install dependencies
cd code
pip install -r requirements.txt

# Generate node embeddings (creates new directories with embeddings)
python cli.py node-embeddings --input-dir ../dev/chatgpt-4-11-2025_json_no_embeddings \
    --method role-aggregate --embedding-method llm \
    --output-dir ../dev/chatgpt-4-11-2025_json_llm

# Generate edges (CPU)
python cli.py edges --input-dir ./embeddings_json --output-file edges.json

# Generate edges (GPU-accelerated)
python cli.py edges-gpu --input-dir ./embeddings_json --output-file edges.json

# Filter edges by similarity threshold
python cli.py cut-off --input-file edges.json --output-file filtered_edges.json --cutoff 0.7

# Export to Gephi format
python cli.py export --nodes-dir ./embeddings_json --edges-file filtered_edges.json \
    --format gexf --output-file graph.gexf

# Run recommendation system
python rec-conv.py --nodes-dir ./embeddings_json --csv nodes.csv --repl

# Validate JSON conversation files
python cli.py validate --input-dir ../dev/chatgpt-4-11-2025_json_simple

# Normalize embeddings
python cli.py normalize --input-dir ./embeddings_json

# Clean embeddings from JSON files
python cli.py clean --input-dir ./embeddings_json --embedding-type llm
```

## Architecture Overview

### Python Implementation (`code/`)

The codebase implements a pipeline for converting conversational data into analyzable networks:

1. **Embedding Generation** (`embedding/`):
   - `llm_embedding_model.py`: LLM-based embeddings via Ollama API
   - `tdidf_embedding_model.py`: Classical TF-IDF approach
   - Role-based aggregation (user vs assistant messages)
   - Automatic text chunking for long messages

2. **Graph Construction** (`graph/`):
   - `edge_utils.py`: Cosine similarity-based edge generation
   - `gpu-edge.py`: CUDA-accelerated edge computation
   - `normalization.py`: Vector normalization utilities
   - `export_utils.py`: Export to GEXF, GraphML, GML formats

3. **Analysis Tools**:
   - `networks.py`: Core network statistics and metrics
   - `rec-conv.py`: Interactive conversation recommendation
   - `cli.py`: Unified command-line interface

### LaTeX Documents

Two paper versions exist:
- `class_paper/`: Original academic paper with comprehensive analysis
- `comp-net-2025-paper/`: Conference-optimized version with ablation studies

Key findings documented:
- 15 distinct knowledge communities (0.75 modularity)
- Three bridge conversation types (evolutionary, integrative, pure)
- Non-standard degree distribution challenging scale-free assumptions

## Key Dependencies

### Python
- networkx (3.4.2) - Graph analysis
- scikit-learn (1.6.1) - TF-IDF embeddings
- numpy (2.2.5) - Numerical operations
- pandas (2.2.3) - Data manipulation
- requests - Ollama API communication
- tqdm - Progress visualization

### LaTeX Packages
- `svproc.cls` - Springer conference class
- `algorithm2e` - Algorithm presentation
- `tikz` - Network diagrams
- `graphicx` - Image inclusion
- `beamer` - Presentation framework

## External References

- Implementation repository: https://github.com/queelius/chatgpt-complex-net
- Dataset: `dev/chatgpt-4-11-2025_json_no_embeddings/` contains 1908 raw ChatGPT conversation exports
- Embeddings: Generated on-demand with different methods (LLM, TF-IDF), stored in separate directories
- Images use both vector (SVG/PDF) and raster (PNG) formats for compatibility