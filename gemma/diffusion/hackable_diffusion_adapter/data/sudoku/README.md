# Sudoku Onboarding and Evaluation on HD Text Diffusion

This folder contains the dataset ingestion and preprocessing for the **Sudoku**
puzzle solving dataset.

Sudoku solving is a structured conditional generation task where the
model reads a space-separated 81-character puzzle representation and learns to:

1.  Preserve the unmasked "clue" digits provided in the initial puzzle context.
2.  Iteratively denoise and fill in the missing cells (marked by `0`).

## Directory Structure

*   **`prepare_sudoku_dataset.sh`**: Script that configures Kaggle credentials and fetches the dataset.
*   **`convert_sudoku.py`**: Preprocessing script to format CSV raw datasets
    into Bagz files.
*   **`sudoku_data.py`**: Dataset and transform configuration using standard
    Kauldron pipelines.

## Expected output

*   **`sudoku_train.bagz`**: Converted training records containing the source
    puzzles and reference solutions.
*   **`sudoku_eval.bagz`**: Converted evaluation records.
