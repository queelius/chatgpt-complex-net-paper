# Ablation Study Guide: User-AI Embedding Weight Ratios

## Overview

This guide documents the complete ablation study pipeline for analyzing how different weighting ratios between user and AI messages affect the semantic network structure of ChatGPT conversations.

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
cd code
pip install -r requirements.txt

# Configure Ollama endpoint (edit .env file)
nano .env
# Set OLLAMA_HOST to your Ollama server address
# Set MODEL_NAME to your embedding model (e.g., nomic-embed-text)
```

### 2. Run Complete Study

```bash
# Generate all data (long-running, 1-3 days expected)
python generate_ablation_data.py \
  --input-dir ../dev/chatgpt-4-11-2025_json_no_embeddings \
  --output-base ../dev/ablation_study

# Monitor progress (in another terminal)
python monitor_progress.py

# Analyze results (can run multiple times)
python analyze_ablation_results.py \
  --data-dir ../dev/ablation_study \
  --plot-individual \
  --export-gexf
```

## Detailed Components

### Weight Ratios

The study examines 9 user:AI weight ratios:

| Ratio | User Weight | AI Weight | Purpose |
|-------|------------|-----------|---------|
| 100:1 | 100.0 | 1.0 | Near-pure user perspective |
| 4:1 | 4.0 | 1.0 | Strong user emphasis |
| 2:1 | 2.0 | 1.0 | Paper's baseline ratio |
| φ:1 | 1.618 | 1.0 | Golden ratio (user-favored) |
| 1:1 | 1.0 | 1.0 | Perfect balance |
| 1:φ | 1.0 | 1.618 | Inverse golden ratio |
| 1:2 | 1.0 | 2.0 | Inverse paper ratio |
| 1:4 | 1.0 | 4.0 | Strong AI emphasis |
| 1:100 | 1.0 | 100.0 | Near-pure AI perspective |

### Pipeline Architecture

```
Input Data (1908 conversations)
    ↓
[DATA GENERATION PHASE]
    ↓
1. Embedding Generation (per ratio)
   - LLM embeddings via Ollama API
   - Role-based aggregation
   - Weighted combination
    ↓
2. Edge Generation
   - Cosine similarity computation
   - GPU acceleration available
    ↓
3. Edge Filtering
   - Similarity threshold (0.9)
   - Multiple cutoffs supported
    ↓
[ANALYSIS PHASE]
    ↓
4. Network Statistics
   - Graph metrics computation
   - Community detection
   - Centrality measures
    ↓
5. Visualization
   - Degree distribution plots
   - Comparison metrics
   - LaTeX tables
    ↓
6. Export
   - GEXF for Gephi
   - CSV summaries
   - PDF plots
```

## Data Generation Script (`generate_ablation_data.py`)

### Features
- **Resilient**: Continues on failures, tracks all errors
- **Resumable**: Can restart from last completed ratio
- **Monitored**: Real-time progress tracking
- **Cached**: Skips existing valid data
- **Logged**: Comprehensive logging to file

### Usage Options

```bash
# Basic usage
python generate_ablation_data.py

# Resume from previous run
python generate_ablation_data.py --resume

# Use GPU for edge generation
python generate_ablation_data.py --use-gpu

# Process specific ratios only
python generate_ablation_data.py --ratios "2.0:1.0,1.0:1.0,1.0:2.0"

# Multiple similarity cutoffs
python generate_ablation_data.py --cutoffs "0.7,0.8,0.9"

# Skip embeddings if they exist
python generate_ablation_data.py --skip-embeddings

# Skip edge generation
python generate_ablation_data.py --skip-edges
```

### Output Structure

```
ablation_study/
├── logs/
│   └── generation_TIMESTAMP.log     # Detailed execution log
├── data/
│   ├── embeddings/
│   │   ├── chatgpt-json-llm-user100.0-ai1.0/
│   │   │   ├── *.json               # Embeddings per conversation
│   │   │   └── generation_metadata.json
│   │   └── ...
│   └── edges/
│       ├── edges_user100.0-ai1.0.json
│       ├── edges_user100.0-ai1.0_filtered_0.9.json
│       └── *.metadata.json          # Generation metadata
├── generation_progress.json          # Real-time progress
├── generation_summary.json           # Final summary
└── study_config.json                # Study configuration
```

## Progress Monitoring (`monitor_progress.py`)

### Real-time Monitoring

```bash
# Fancy terminal UI (recommended)
python monitor_progress.py

# Simple console output
python monitor_progress.py --simple

# Custom refresh interval
python monitor_progress.py --refresh 5

# Monitor specific progress file
python monitor_progress.py --progress-file /path/to/generation_progress.json
```

### Display Information
- Current status and progress percentage
- Completed and failed ratios
- Currently processing ratio
- Recent activity messages
- Elapsed time and ETA
- Color-coded status indicators

## Analysis Script (`analyze_ablation_results.py`)

### Features
- **Cached**: Saves computed statistics
- **Flexible**: Can re-analyze with different parameters
- **Publication-ready**: Creates high-quality plots and tables
- **Comparative**: Generates cross-ratio comparisons

### Usage Options

```bash
# Basic analysis
python analyze_ablation_results.py --data-dir ../dev/ablation_study

# Focus on paper's ratio
python analyze_ablation_results.py \
  --data-dir ../dev/ablation_study \
  --focus-ratio "user2.0-ai1.0" \
  --plot-individual

# Different similarity cutoff
python analyze_ablation_results.py \
  --data-dir ../dev/ablation_study \
  --cutoff 0.8

# Export for Gephi
python analyze_ablation_results.py \
  --data-dir ../dev/ablation_study \
  --export-gexf

# Force recalculation
python analyze_ablation_results.py \
  --data-dir ../dev/ablation_study \
  --recalculate
```

### Generated Outputs

```
analysis_TIMESTAMP/
├── statistics/
│   ├── stats_*.json                 # Per-ratio statistics
│   └── combined_statistics.json     # All ratios combined
├── plots/
│   ├── degree_distributions/        # Individual ratio plots
│   │   ├── degree_dist_*.png
│   │   └── degree_dist_*.pdf
│   └── comparisons/                 # Cross-ratio comparisons
│       ├── metrics_comparison_grid.png
│       ├── network_evolution.png
│       ├── metrics_heatmap.png
│       └── summary_table.tex
├── gexf/
│   └── network_*.gexf              # Gephi-compatible files
├── cache/                          # Cached statistics
└── summary_cutoff_*.csv           # Summary tables
```

## Network Statistics Computed

- **Basic**: nodes, edges, density
- **Components**: giant component size/fraction
- **Degree**: mean, std, max, min, distribution
- **Clustering**: average clustering, transitivity
- **Paths**: diameter, average shortest path (sampled)
- **Centrality**: degree, betweenness
- **Community**: modularity, number of communities
- **Assortativity**: degree assortativity coefficient

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Check OLLAMA_HOST in .env
   - Ensure Ollama server is running
   - Verify model is available: `curl http://localhost:11434/api/tags`

2. **Out of Memory**
   - Use GPU edge generation: `--use-gpu`
   - Process fewer ratios at once: `--ratios "2.0:1.0"`
   - Reduce batch size in .env

3. **Resume After Interruption**
   ```bash
   python generate_ablation_data.py --resume
   ```

4. **Corrupted Progress File**
   ```bash
   rm ../dev/ablation_study/generation_progress.json
   # Then restart normally
   ```

### Performance Tips

1. **Parallel Processing**: Run multiple instances with different ratios
   ```bash
   python generate_ablation_data.py --ratios "100.0:1.0,4.0:1.0,2.0:1.0"
   python generate_ablation_data.py --ratios "1.0:1.0,1.0:2.0,1.0:4.0"
   ```

2. **GPU Acceleration**: Enable for edge generation
   ```bash
   python generate_ablation_data.py --use-gpu
   ```

3. **Skip Existing**: Avoid regenerating completed data
   ```bash
   python generate_ablation_data.py --skip-embeddings --skip-edges
   ```

## Expected Timeline

For 1908 conversations with 9 ratios:

| Phase | Estimated Time | Notes |
|-------|---------------|-------|
| Embeddings | 24-48 hours | Depends on Ollama server speed |
| Edge Generation | 4-8 hours | GPU recommended |
| Edge Filtering | 1-2 hours | Per cutoff value |
| Analysis | 30-60 minutes | Can be repeated |

Total: 1-3 days for complete generation

## Publication Outputs

The analysis generates publication-ready materials:

1. **Degree Distribution Plots**: 4-panel analysis per ratio
2. **Comparison Grid**: 9-metric bar chart comparison
3. **Evolution Plots**: How metrics change across ratios
4. **Heatmap**: Normalized metrics visualization
5. **LaTeX Table**: Ready for paper inclusion
6. **GEXF Files**: For Gephi visualization

### Recreating Paper Figures

To recreate figures for the 2:1 ratio used in the paper:

```bash
python analyze_ablation_results.py \
  --data-dir ../dev/ablation_study \
  --focus-ratio "user2.0-ai1.0" \
  --plot-individual \
  --export-gexf

# Results will be in:
# analysis_*/plots/degree_distributions/degree_dist_user2.0-ai1.0.pdf
# analysis_*/gexf/network_user2.0-ai1.0_cutoff_0.9.gexf
```

## Configuration (.env)

Key settings to adjust:

```bash
# Ollama Configuration
OLLAMA_HOST=http://192.168.0.225:11434  # Your Ollama server
MODEL_NAME=nomic-embed-text             # Embedding model

# Processing
SIMILARITY_THRESHOLD=0.9                # Edge filtering threshold
USE_GPU=false                           # GPU acceleration
EMBEDDING_BATCH_SIZE=10                 # Batch size for API calls

# Performance
OLLAMA_TIMEOUT=120                      # API timeout (seconds)
MAX_RETRIES=3                           # Retry failed API calls
RETRY_DELAY=5                           # Delay between retries
```

## Code Structure

### Key Functions

**generate_ablation_data.py**:
- `generate_embeddings()`: Creates embeddings for a ratio
- `generate_edges()`: Computes similarity edges
- `filter_edges()`: Applies similarity threshold
- `update_progress()`: Updates progress tracking

**analyze_ablation_results.py**:
- `compute_network_statistics()`: Calculates graph metrics
- `plot_degree_distribution()`: Creates 4-panel plots
- `create_comparison_plots()`: Generates cross-ratio comparisons
- `export_to_gexf()`: Creates Gephi files

**monitor_progress.py**:
- `monitor_console()`: Simple text monitoring
- `monitor_curses()`: Enhanced terminal UI

## Further Analysis

After running the ablation study, consider:

1. **Statistical Tests**: Compare distributions across ratios
2. **Community Analysis**: Deep dive into detected communities
3. **Temporal Evolution**: How networks change over conversation time
4. **Bridge Analysis**: Identify key bridging conversations
5. **Visualization in Gephi**: Interactive exploration of networks

## Support

For issues or questions:
1. Check log files in `ablation_study/logs/`
2. Review generation_summary.json for data paths
3. Verify .env configuration
4. Ensure all dependencies installed: `pip install -r requirements.txt`