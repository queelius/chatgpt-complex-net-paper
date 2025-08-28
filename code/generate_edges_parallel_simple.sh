#!/bin/bash
# Simpler parallel edge generation without function exports

EMBEDDINGS_DIR="${1:-../dev/ablation_study/data/embeddings}"
OUTPUT_DIR="${2:-../dev/ablation_study/data/edges}"
JOBS="${JOBS:-$(nproc)}"

echo "=================================================="
echo "PARALLEL EDGE GENERATION (SIMPLE VERSION)"
echo "=================================================="
echo "Embeddings: $EMBEDDINGS_DIR"
echo "Output: $OUTPUT_DIR"
echo "Parallel jobs: $JOBS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create a simple worker script
cat > /tmp/process_edges.sh << 'EOF'
#!/bin/bash
EMB_DIR="$1"
OUTPUT_DIR="$2"

DIR_NAME=$(basename "$EMB_DIR")
USER_WEIGHT=$(echo "$DIR_NAME" | sed -n 's/.*user\([0-9.]*\)-ai.*/\1/p')
AI_WEIGHT=$(echo "$DIR_NAME" | sed -n 's/.*-ai\([0-9.]*\)/\1/p')

echo "[$(date +%H:%M:%S)] Processing $DIR_NAME"

EDGES_FILE="$OUTPUT_DIR/edges_${DIR_NAME}.json"
python cli.py edges --input-dir "$EMB_DIR" --output-file "$EDGES_FILE" 2>/dev/null

if [ $? -eq 0 ]; then
    EDGE_COUNT=$(python -c "import json; print(len(json.load(open('$EDGES_FILE'))))" 2>/dev/null || echo "unknown")
    echo "[$(date +%H:%M:%S)] ✓ $DIR_NAME: $EDGE_COUNT edges"
    
    # Filter at 0.9 threshold
    FILTERED_FILE="$OUTPUT_DIR/edges_${DIR_NAME}_filtered_0.9.json"
    python cli.py cut-off --input-file "$EDGES_FILE" --output-file "$FILTERED_FILE" --cutoff 0.9 2>/dev/null
    
    if [ $? -eq 0 ]; then
        FILTERED_COUNT=$(python -c "import json; print(len(json.load(open('$FILTERED_FILE'))))" 2>/dev/null || echo "unknown")
        echo "[$(date +%H:%M:%S)] ✓ $DIR_NAME: filtered to $FILTERED_COUNT edges"
    fi
else
    echo "[$(date +%H:%M:%S)] ✗ $DIR_NAME: FAILED"
fi
EOF

chmod +x /tmp/process_edges.sh

# Count directories
DIR_COUNT=$(ls -d "$EMBEDDINGS_DIR"/chatgpt-json-llm-user* 2>/dev/null | wc -l)
echo "Found $DIR_COUNT directories to process"

if [ "$DIR_COUNT" -eq 0 ]; then
    echo "ERROR: No embedding directories found!"
    echo "Looking in: $EMBEDDINGS_DIR"
    ls "$EMBEDDINGS_DIR" | head -5
    exit 1
fi

echo ""

# Force using xargs which is more reliable
echo "Using xargs with $JOBS parallel jobs"
echo ""
ls -d "$EMBEDDINGS_DIR"/chatgpt-json-llm-user* | \
    xargs -P $JOBS -I {} /tmp/process_edges.sh {} "$OUTPUT_DIR"

echo ""
echo "=================================================="
echo "EDGE GENERATION COMPLETE"
echo "=================================================="
echo "Edge files saved to: $OUTPUT_DIR"
echo ""

# Clean up
rm -f /tmp/process_edges.sh

echo "To generate all threshold variations, run:"
echo "  ./generate_all_thresholds.sh"