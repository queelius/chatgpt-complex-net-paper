#!/bin/bash
# Debug script to see what's happening

EMBEDDINGS_DIR="../dev/ablation_study/data/embeddings"

echo "Current directory: $(pwd)"
echo "Embeddings dir: $EMBEDDINGS_DIR"
echo "Does it exist? $(test -d "$EMBEDDINGS_DIR" && echo YES || echo NO)"
echo ""

echo "Method 1 - for loop (like generate_edges.sh):"
count=0
for EMB_DIR in "$EMBEDDINGS_DIR"/chatgpt-json-llm-user*; do
    if [ -d "$EMB_DIR" ]; then
        echo "  Found: $(basename "$EMB_DIR")"
        ((count++))
    fi
done
echo "  Total: $count"
echo ""

echo "Method 2 - ls with glob:"
ls -d "$EMBEDDINGS_DIR"/chatgpt-json-llm-user* 2>/dev/null | while read dir; do
    echo "  Found: $(basename "$dir")"
done
echo "  Count: $(ls -d "$EMBEDDINGS_DIR"/chatgpt-json-llm-user* 2>/dev/null | wc -l)"
echo ""

echo "Method 3 - find:"
find "$EMBEDDINGS_DIR" -maxdepth 1 -type d -name "chatgpt-json-llm-user*" | while read dir; do
    echo "  Found: $(basename "$dir")"
done

echo ""
echo "Raw ls output:"
ls "$EMBEDDINGS_DIR" | grep "chatgpt-json" | head -5