# Kauldron Configurations for SFT Training

This directory contains the Kauldron configuration files defining the models,
datasets, training losses, optimizers, schedules, and evaluators for SFT
training.

The configurations are flat, self-contained files defining the end-to-end
setup (model, dataset, training losses, optimizer, schedule, and evaluators)
for each task.

## Configuration Files

*   **`sft_sudoku.py`** (Sudoku):
    *   Configures SFT training with LoRA on the sudoku dataset.

*   **`sft_sudoku_full.py`** (Sudoku):
    *   Configures full weight SFT training on the sudoku dataset.

*   **`sft_pubmedqa.py`** (PubMedQA):
    *   Configures Parameter-Efficient SFT training using LoRA on PubMedQA.
