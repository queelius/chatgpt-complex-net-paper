#!/bin/bash
# Simple shell script to generate edges from embeddings with clear output

EMBEDDINGS_DIR="${1:-../dev/ablation_study/data/embeddings}"
OUTPUT_DIR="${2:-../dev/ablation_study/data/edges}"
METADATA_FILE="../dev/ablation_study/edge_generation_metadata.json"

echo "=================================================="
echo "EDGE GENERATION FOR ABLATION STUDY"
echo "=================================================="
echo "Embeddings: $EMBEDDINGS_DIR"
echo "Output: $OUTPUT_DIR"
echo "Metadata: $METADATA_FILE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start metadata JSON
echo "{" > "$METADATA_FILE"
echo "  \"generation_date\": \"$(date -Iseconds)\"," >> "$METADATA_FILE"
echo "  \"embeddings_source\": \"$EMBEDDINGS_DIR\"," >> "$METADATA_FILE"
echo "  \"edges_output\": \"$OUTPUT_DIR\"," >> "$METADATA_FILE"
echo "  \"threshold\": 0.9," >> "$METADATA_FILE"
echo "  \"ratios\": [" >> "$METADATA_FILE"

FIRST=true

# Process each embedding directory
for EMB_DIR in "$EMBEDDINGS_DIR"/chatgpt-json-llm-user*; do
    if [ ! -d "$EMB_DIR" ]; then
        continue
    fi
    
    DIR_NAME=$(basename "$EMB_DIR")
    
    # Extract user and AI weights from directory name
    USER_WEIGHT=$(echo "$DIR_NAME" | sed -n 's/.*user\([0-9.]*\)-ai.*/\1/p')
    AI_WEIGHT=$(echo "$DIR_NAME" | sed -n 's/.*-ai\([0-9.]*\)/\1/p')
    
    echo ""
    echo "=================================================="
    echo "Processing: $DIR_NAME"
    echo "User weight: $USER_WEIGHT, AI weight: $AI_WEIGHT"
    echo "=================================================="
    
    # Generate edges
    EDGES_FILE="$OUTPUT_DIR/edges_${DIR_NAME}.json"
    echo "Generating edges..."
    python cli.py edges \
        --input-dir "$EMB_DIR" \
        --output-file "$EDGES_FILE"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate edges for $DIR_NAME"
        continue
    fi
    
    # Count edges
    EDGE_COUNT=$(python -c "import json; print(len(json.load(open('$EDGES_FILE'))))" 2>/dev/null || echo "unknown")
    echo "Generated $EDGE_COUNT edges"
    
    # Filter edges with 0.9 threshold
    FILTERED_FILE="$OUTPUT_DIR/edges_${DIR_NAME}_filtered_0.9.json"
    echo "Filtering with threshold 0.9..."
    python cli.py cut-off \
        --input-file "$EDGES_FILE" \
        --output-file "$FILTERED_FILE" \
        --cutoff 0.9
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to filter edges for $DIR_NAME"
        continue
    fi
    
    # Count filtered edges
    FILTERED_COUNT=$(python -c "import json; print(len(json.load(open('$FILTERED_FILE'))))" 2>/dev/null || echo "unknown")
    echo "After filtering: $FILTERED_COUNT edges"
    
    # Calculate retention rate
    if [ "$EDGE_COUNT" != "unknown" ] && [ "$FILTERED_COUNT" != "unknown" ]; then
        RETENTION=$(python -c "print(f'{100*$FILTERED_COUNT/$EDGE_COUNT:.1f}%')")
        echo "Retention rate: $RETENTION"
    fi
    
    # Add to metadata
    if [ "$FIRST" = false ]; then
        echo "," >> "$METADATA_FILE"
    fi
    FIRST=false
    
    echo -n "    {" >> "$METADATA_FILE"
    echo -n "\"ratio\": \"${USER_WEIGHT}:${AI_WEIGHT}\", " >> "$METADATA_FILE"
    echo -n "\"user_weight\": $USER_WEIGHT, " >> "$METADATA_FILE"
    echo -n "\"ai_weight\": $AI_WEIGHT, " >> "$METADATA_FILE"
    echo -n "\"embeddings_dir\": \"$EMB_DIR\", " >> "$METADATA_FILE"
    echo -n "\"edges_file\": \"$EDGES_FILE\", " >> "$METADATA_FILE"
    echo -n "\"filtered_file\": \"$FILTERED_FILE\", " >> "$METADATA_FILE"
    echo -n "\"total_edges\": $EDGE_COUNT, " >> "$METADATA_FILE"
    echo -n "\"filtered_edges\": $FILTERED_COUNT" >> "$METADATA_FILE"
    echo -n "}" >> "$METADATA_FILE"
    
    echo "✓ Completed $DIR_NAME"
done

# Close metadata JSON
echo "" >> "$METADATA_FILE"
echo "  ]" >> "$METADATA_FILE"
echo "}" >> "$METADATA_FILE"

echo ""
echo "=================================================="
echo "EDGE GENERATION COMPLETE"
echo "=================================================="
echo "Edge files saved to: $OUTPUT_DIR"
echo "Metadata saved to: $METADATA_FILE"
echo ""
echo "Next steps:"
echo "1. Review metadata: cat $METADATA_FILE | python -m json.tool"
echo "2. Analyze results: python analyze_ablation_results.py --data-dir ../dev/ablation_study"
echo "3. Export for Gephi: python cli.py export --nodes-dir <emb_dir> --edges-file <filtered_edges> --format gexf --output-file graph.gexf"