#!/bin/bash

# Exit on error
set -e

# Use the directory of the script to find the relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

OUT_DIR="${SCRIPT_DIR}"

mkdir -p "$OUT_DIR"

echo "Downloading Sudoku dataset from Kaggle..."
kaggle datasets download -d rohanrao/sudoku -p "$OUT_DIR" --unzip

echo "Running conversion script..."
# Use the currently active Python environment
python3 "${SCRIPT_DIR}/convert_sudoku.py" \
  --input_csv="$OUT_DIR/sudoku.csv" \
  --output_dir="$OUT_DIR"

echo "Conversion complete! Output files are in $OUT_DIR"
