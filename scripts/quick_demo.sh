#!/bin/bash
# Quick demo video generation script

echo "==================================="
echo "Diamond Segmentation Demo Generator"
echo "==================================="

# Set paths
DATA_PATH="${1:-data/raw}"
OUTPUT_DIR="${2:-demo_videos}"

echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run generation
python scripts/generate_demo_videos.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --fps 15 \
    --iterations 5

echo ""
echo "Demo generation complete!"
echo "Videos saved to: $OUTPUT_DIR"