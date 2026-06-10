#!/bin/bash

# Exit on error
set -e

# Use the directory of the script to find the relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

REPO_DIR="/tmp/pubmedqa_repo"
OUT_DIR="${SCRIPT_DIR}"

mkdir -p "$OUT_DIR"

echo "Cloning PubMedQA repository..."
git clone https://github.com/pubmedqa/pubmedqa.git "$REPO_DIR"

echo "Running conversion script..."
# Use the currently active Python environment
python3 "${SCRIPT_DIR}/convert_pubmedqa.py" \
  --input_dir="$REPO_DIR/data" \
  --output_dir="$OUT_DIR"

echo "Conversion complete! Output files are in $OUT_DIR"
