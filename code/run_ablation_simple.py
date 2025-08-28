#!/usr/bin/env python3
"""
Simple ablation study runner - shows all output, no fancy logging
"""

import os
import sys
import subprocess
from pathlib import Path

# Weight ratios to test
WEIGHT_RATIOS = [
    (100.0, 1.0),   # Near-pure user perspective
    (4.0, 1.0),     # Strong user emphasis  
    (2.0, 1.0),     # Paper's ratio
    (1.618, 1.0),   # Golden ratio
    (1.0, 1.0),     # Perfect balance
    (1.0, 1.618),   # Inverse golden ratio
    (1.0, 2.0),     # Inverse paper ratio
    (1.0, 4.0),     # Strong AI emphasis
    (1.0, 100.0),   # Near-pure AI perspective
]

def run_ratio(input_dir, output_base, user_weight, ai_weight):
    """Run embedding generation for one ratio - shows all output"""
    
    output_dir = f"{output_base}/embeddings/user{user_weight}-ai{ai_weight}"
    
    # Create output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Copy input files
    print(f"\n{'='*60}")
    print(f"Setting up ratio user:{user_weight} ai:{ai_weight}")
    print(f"{'='*60}")
    
    cmd = f"cp {input_dir}/*.json {output_dir}/"
    print(f"Copying files: {cmd}")
    subprocess.run(cmd, shell=True, check=False)
    
    # Generate embeddings - DON'T CAPTURE OUTPUT, LET IT SHOW
    print(f"\nGenerating embeddings...")
    cmd = f"""python cli.py node-embeddings \
        --input-dir {output_dir} \
        --method role-aggregate \
        --embedding-method llm \
        --user-weight {user_weight} \
        --assistant-weight {ai_weight}"""
    
    print(f"Running: {cmd}\n")
    
    # Run without capturing - shows everything including progress bars
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"ERROR: Failed with code {result.returncode}")
        return False
    
    print(f"✓ Completed embeddings for ratio {user_weight}:{ai_weight}")
    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_ablation_simple.py <input_dir> <output_base>")
        print("Example: python run_ablation_simple.py ../dev/chatgpt-4-11-2025_json_no_embeddings ../dev/ablation_simple")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_base = sys.argv[2]
    
    print(f"Input: {input_dir}")
    print(f"Output: {output_base}")
    
    # Check input exists
    if not Path(input_dir).exists():
        print(f"ERROR: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    json_files = list(Path(input_dir).glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        print("ERROR: No JSON files found")
        sys.exit(1)
    
    # Process each ratio
    successful = []
    failed = []
    
    for user_w, ai_w in WEIGHT_RATIOS:
        success = run_ratio(input_dir, output_base, user_w, ai_w)
        if success:
            successful.append(f"{user_w}:{ai_w}")
        else:
            failed.append(f"{user_w}:{ai_w}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}/{len(WEIGHT_RATIOS)}")
    if failed:
        print(f"Failed: {failed}")
    
    print(f"\nData saved to: {output_base}")

if __name__ == "__main__":
    main()