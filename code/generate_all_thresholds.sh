#!/bin/bash
# Generate filtered edge lists for multiple thresholds from existing complete edge lists

EDGES_DIR="${1:-../dev/ablation_study/data/edges}"
OUTPUT_DIR="${2:-../dev/ablation_study/data/edges_filtered}"
METADATA_FILE="../dev/ablation_study/threshold_analysis_metadata.json"

# Thresholds to test
THRESHOLDS=(0.8 0.825 0.85 0.875 0.9 0.925 0.95)

echo "=================================================="
echo "THRESHOLD FILTERING FOR ABLATION STUDY"
echo "=================================================="
echo "Input edges: $EDGES_DIR"
echo "Output: $OUTPUT_DIR"
echo "Thresholds: ${THRESHOLDS[@]}"
echo ""

# Create output directory structure
mkdir -p "$OUTPUT_DIR"

# Start metadata JSON
echo "{" > "$METADATA_FILE"
echo "  \"generation_date\": \"$(date -Iseconds)\"," >> "$METADATA_FILE"
echo "  \"source_edges\": \"$EDGES_DIR\"," >> "$METADATA_FILE"
echo "  \"filtered_output\": \"$OUTPUT_DIR\"," >> "$METADATA_FILE"
echo "  \"thresholds\": [${THRESHOLDS[@]}]," >> "$METADATA_FILE"
echo "  \"analysis\": {" >> "$METADATA_FILE"

FIRST_RATIO=true

# Process each complete edge file
for EDGE_FILE in "$EDGES_DIR"/edges_chatgpt-json-llm-user*.json; do
    # Skip filtered files, only process complete edge lists
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
    echo "=================================================="
    echo "Processing ratio ${USER_WEIGHT}:${AI_WEIGHT}"
    echo "=================================================="
    
    # Count total edges
    TOTAL_EDGES=$(python -c "import json; print(len(json.load(open('$EDGE_FILE'))))" 2>/dev/null || echo "0")
    echo "Total edges: $TOTAL_EDGES"
    
    # Add ratio to metadata
    if [ "$FIRST_RATIO" = false ]; then
        echo "," >> "$METADATA_FILE"
    fi
    FIRST_RATIO=false
    
    echo "    \"ratio_${USER_WEIGHT}_${AI_WEIGHT}\": {" >> "$METADATA_FILE"
    echo "      \"user_weight\": $USER_WEIGHT," >> "$METADATA_FILE"
    echo "      \"ai_weight\": $AI_WEIGHT," >> "$METADATA_FILE"
    echo "      \"source_file\": \"$EDGE_FILE\"," >> "$METADATA_FILE"
    echo "      \"total_edges\": $TOTAL_EDGES," >> "$METADATA_FILE"
    echo "      \"threshold_analysis\": {" >> "$METADATA_FILE"
    
    FIRST_THRESHOLD=true
    
    # Generate filtered versions for each threshold
    for THRESHOLD in "${THRESHOLDS[@]}"; do
        FILTERED_FILE="$OUTPUT_DIR/${BASE_NAME}_t${THRESHOLD}.json"
        
        echo -n "  Filtering at threshold $THRESHOLD..."
        
        python cli.py cut-off \
            --input-file "$EDGE_FILE" \
            --output-file "$FILTERED_FILE" \
            --cutoff "$THRESHOLD" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            # Count filtered edges
            FILTERED_COUNT=$(python -c "import json; print(len(json.load(open('$FILTERED_FILE'))))" 2>/dev/null || echo "0")
            RETENTION=$(python -c "print(f'{100*$FILTERED_COUNT/$TOTAL_EDGES:.2f}')" 2>/dev/null || echo "0")
            
            echo " $FILTERED_COUNT edges (${RETENTION}% retention)"
            
            # Add to metadata
            if [ "$FIRST_THRESHOLD" = false ]; then
                echo "," >> "$METADATA_FILE"
            fi
            FIRST_THRESHOLD=false
            
            echo "        \"$THRESHOLD\": {" >> "$METADATA_FILE"
            echo "          \"file\": \"$FILTERED_FILE\"," >> "$METADATA_FILE"
            echo "          \"edge_count\": $FILTERED_COUNT," >> "$METADATA_FILE"
            echo "          \"retention_rate\": $RETENTION" >> "$METADATA_FILE"
            echo -n "        }" >> "$METADATA_FILE"
        else
            echo " ERROR"
        fi
    done
    
    echo "" >> "$METADATA_FILE"
    echo "      }" >> "$METADATA_FILE"
    echo -n "    }" >> "$METADATA_FILE"
done

# Close metadata JSON
echo "" >> "$METADATA_FILE"
echo "  }" >> "$METADATA_FILE"
echo "}" >> "$METADATA_FILE"

# Generate summary CSV for easy analysis
echo ""
echo "Generating summary CSV..."
SUMMARY_FILE="../dev/ablation_study/threshold_summary.csv"

python -c "
import json
import csv

with open('$METADATA_FILE', 'r') as f:
    metadata = json.load(f)

# Write CSV
with open('$SUMMARY_FILE', 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Header
    header = ['user_weight', 'ai_weight', 'total_edges']
    for t in metadata['thresholds']:
        header.extend([f't{t}_edges', f't{t}_retention'])
    writer.writerow(header)
    
    # Data rows
    for key, data in metadata['analysis'].items():
        if key.startswith('ratio_'):
            row = [data['user_weight'], data['ai_weight'], data['total_edges']]
            for t in metadata['thresholds']:
                t_str = str(t)
                if t_str in data['threshold_analysis']:
                    row.append(data['threshold_analysis'][t_str]['edge_count'])
                    row.append(data['threshold_analysis'][t_str]['retention_rate'])
                else:
                    row.extend(['', ''])
            writer.writerow(row)

print(f'Summary saved to: $SUMMARY_FILE')
"

echo ""
echo "=================================================="
echo "THRESHOLD FILTERING COMPLETE"
echo "=================================================="
echo "Filtered edges saved to: $OUTPUT_DIR"
echo "Metadata saved to: $METADATA_FILE"
echo "Summary CSV saved to: $SUMMARY_FILE"
echo ""
echo "Directory structure:"
echo "  $EDGES_DIR/"
echo "    └── edges_chatgpt-json-llm-userX-aiY.json  (complete edge lists)"
echo "  $OUTPUT_DIR/"
echo "    └── edges_chatgpt-json-llm-userX-aiY_t0.8.json  (filtered versions)"
echo "    └── edges_chatgpt-json-llm-userX-aiY_t0.825.json"
echo "    └── ... (for each threshold)"
echo ""
echo "Next: python analyze_ablation_results.py --data-dir ../dev/ablation_study"