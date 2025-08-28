#!/bin/bash

# Simple shell script to generate embeddings for different weight ratios
# Shows progress directly in terminal

# Configuration
INPUT_DIR="../dev/chatgpt-4-11-2025_json_no_embeddings"
OUTPUT_BASE="../dev/ablation_study/data/embeddings"

# Weight ratios to process
RATIOS=(
    "100.0:1.0"
    "4.0:1.0"
#    "2.0:1.0"
    "1.618:1.0"
#    "1.0:1.0"
    "1.0:1.618"
#    "1.0:2.0"
    "1.0:4.0"
    "1.0:100.0"
)

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Process each ratio
for ratio in "${RATIOS[@]}"; do
    IFS=':' read -r user_weight ai_weight <<< "$ratio"
    
    output_dir="$OUTPUT_BASE/chatgpt-json-llm-user${user_weight}-ai${ai_weight}"
    
    echo "=========================================="
    echo "Processing ratio: User $user_weight : AI $ai_weight"
    echo "Output: $output_dir"
    echo "=========================================="
    
    # Check if already exists with embeddings
    if [ -d "$output_dir" ]; then
        count=$(grep -l '"role_aggregate"' "$output_dir"/*.json 2>/dev/null | wc -l)
        if [ "$count" -gt "0" ]; then
            echo "Found $count files with embeddings, checking if complete..."
            total=$(ls "$output_dir"/*.json 2>/dev/null | wc -l)
            if [ "$count" -eq "$total" ] && [ "$total" -eq "1908" ]; then
                echo "✓ Already complete, skipping..."
                continue
            fi
            echo "Incomplete: $count/$total files have embeddings"
        fi
    fi
    
    # Create output directory and copy files
    mkdir -p "$output_dir"
    
    # Copy JSON files if not already there
    echo "Copying JSON files..."
    for file in "$INPUT_DIR"/*.json; do
        basename_file=$(basename "$file")
        if [ ! -f "$output_dir/$basename_file" ]; then
            cp "$file" "$output_dir/"
        fi
    done
    
    # Generate embeddings
    echo "Generating embeddings..."
    python cli.py node-embeddings \
        --input-dir "$output_dir" \
        --method role-aggregate \
        --embedding-method llm \
        --user-weight "$user_weight" \
        --assistant-weight "$ai_weight"
    
    # Check result
    if [ $? -eq 0 ]; then
        echo "✓ Successfully generated embeddings for ratio $ratio"
    else
        echo "✗ Failed to generate embeddings for ratio $ratio"
        echo "Continuing with next ratio..."
    fi
    
    echo ""
done

echo "=========================================="
echo "All ratios processed!"
echo "=========================================="

# Show summary
echo -e "\nSummary:"
for ratio in "${RATIOS[@]}"; do
    IFS=':' read -r user_weight ai_weight <<< "$ratio"
    output_dir="$OUTPUT_BASE/chatgpt-json-llm-user${user_weight}-ai${ai_weight}"
    if [ -d "$output_dir" ]; then
        count=$(grep -l '"role_aggregate"' "$output_dir"/*.json 2>/dev/null | wc -l)
        total=$(ls "$output_dir"/*.json 2>/dev/null | wc -l)
        echo "  Ratio $ratio: $count/$total files with embeddings"
    else
        echo "  Ratio $ratio: Not processed"
    fi
done
