#!/bin/bash
# Generate filtered edge lists for multiple thresholds - fixed version

EDGES_DIR="${1:-../dev/ablation_study/data/edges}"
OUTPUT_DIR="${2:-../dev/ablation_study/data/edges_filtered}"

# Thresholds to test
THRESHOLDS=(0.8 0.825 0.85 0.875 0.9 0.925 0.95)

echo "=================================================="
echo "THRESHOLD FILTERING FOR ABLATION STUDY"
echo "=================================================="
echo "Input edges: $EDGES_DIR"
echo "Output: $OUTPUT_DIR"
echo "Thresholds: ${THRESHOLDS[@]}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each complete edge file
for EDGE_FILE in "$EDGES_DIR"/edges_chatgpt-json-llm-user*.json; do
    # Skip already filtered files
    if [[ "$EDGE_FILE" == *"_filtered_"* ]]; then
        continue
    fi
    
    if [ ! -f "$EDGE_FILE" ]; then
        continue
    fi
    
    BASE_NAME=$(basename "$EDGE_FILE" .json)
    
    # Extract weights from filename
    USER_WEIGHT=$(echo "$BASE_NAME" | sed -n 's/.*user\([0-9.]*\)-ai.*/\1/p')
    AI_WEIGHT=$(echo "$BASE_NAME" | sed -n 's/.*-ai\([0-9.]*\)/\1/p')
    
    echo ""
    echo "Processing ratio ${USER_WEIGHT}:${AI_WEIGHT}"
    echo "=================================================="
    
    # Count total edges
    TOTAL_EDGES=$(python3 -c "import json; print(len(json.load(open('$EDGE_FILE'))))" 2>/dev/null || echo "0")
    echo "Total edges: $(printf "%'d" $TOTAL_EDGES)"
    
    # Generate filtered versions for each threshold
    for THRESHOLD in "${THRESHOLDS[@]}"; do
        FILTERED_FILE="$OUTPUT_DIR/${BASE_NAME}_t${THRESHOLD}.json"
        
        echo -n "  Filtering at threshold $THRESHOLD..."
        
        python3 cli.py cut-off \
            --input-file "$EDGE_FILE" \
            --output-file "$FILTERED_FILE" \
            --cutoff "$THRESHOLD" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            # Count filtered edges
            FILTERED_COUNT=$(python3 -c "import json; print(len(json.load(open('$FILTERED_FILE'))))" 2>/dev/null || echo "0")
            RETENTION=$(python3 -c "print(f'{100*$FILTERED_COUNT/$TOTAL_EDGES:.3f}')" 2>/dev/null || echo "0")
            
            printf " %'d edges (%.3f%% retention)\n" $FILTERED_COUNT $RETENTION
        else
            echo " ERROR"
        fi
    done
done

echo ""
echo "=================================================="
echo "THRESHOLD FILTERING COMPLETE"
echo "=================================================="
echo "Filtered edges saved to: $OUTPUT_DIR"
echo ""

# Quick summary
echo "Summary of filtered files:"
ls "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l | xargs printf "  Total filtered files created: %d\n"
echo ""
echo "Sample retention rates (user1.0-ai1.0 if exists):"
if [ -f "$EDGES_DIR/edges_chatgpt-json-llm-user1.0-ai1.0.json" ]; then
    for THRESHOLD in "${THRESHOLDS[@]}"; do
        FILE="$OUTPUT_DIR/edges_chatgpt-json-llm-user1.0-ai1.0_t${THRESHOLD}.json"
        if [ -f "$FILE" ]; then
            COUNT=$(python3 -c "import json; print(len(json.load(open('$FILE'))))" 2>/dev/null || echo "0")
            printf "  Threshold %.3f: %'d edges\n" $THRESHOLD $COUNT
        fi
    done
fi

echo ""
echo "Next: python analyze_ablation_results.py --data-dir ../dev/ablation_study"