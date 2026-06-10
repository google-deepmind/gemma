# Data Pipelines for Hackable Diffusion (HD) Text

This package contains all data loading, preprocessing, and dataset-specific
pipelines for the Hackable Diffusion text SFT project.

## Package Structure

*   `data.py`: Common data utilities, transforms (like `CanvasChunker` and
    `SequenceTargetShift`), and shared pipeline helpers.
*   `pubmedqa/`: Dataset pipeline for the PubMedQA medical Q&A task.
*   `sudoku/`: Dataset pipeline for the Sudoku puzzle solving task.

## Key Components

### Shared Transforms (`data.py`)

*   **`CanvasChunker`**: Splices a long response token sequence into multiple,
    fixed-size "canvas" chunks. This is required for the localized diffusion
    process, allowing the model to denoise specific parts of the text
    iteratively.
*   **`SequenceTargetShift`**: Shifts sequences to prepare targets for the
    encoder loss.

### Dataset-Specific Pipelines

Each subfolder contains:

1.  A data loader or custom `DataSource` (e.g., `JsonlDataSource` for
    PubMedQA, `Bagz` wrapper for Sudoku).
2.  A conversion script (`convert_*.py`) to preprocess raw data into the
    format expected by the pipeline.
3.  A `README.md` explaining the dataset structure.
