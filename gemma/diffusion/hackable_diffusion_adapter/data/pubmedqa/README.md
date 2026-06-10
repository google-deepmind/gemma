# PubMedQA Onboarding and Evaluation on HD DiffusionGemma

This folder contains the dataset ingestion and preprocessing for onboarding the
**PubMedQA** dataset.

PubMedQA is a medical question-answering dataset where the task is to read a
medical research abstract context and a question, and then:

1.  Output a short categorical answer: **"yes"**, **"no"**, or **"maybe"**.
2.  Generate a detailed paragraph-length explanation (**LONG_ANSWER**).

Our dataset tasks the model with generating both the categorical answer and the
detailed explanation.

## Directory Structure

*   **`prepare_pubmedqa_dataset.sh`**: Clones the dataset from the GitHub repo
    and calls the preprocessing script.
*   **`convert_pubmedqa.py`**: Script that formats the raw datasets with
    instruction-tuning turn tokens, and outputs them as line-delimited JSONL
    files.
*   **`pubmedqa_data.py`**: Implements `JsonlDataSource` (an
    in-memory JSONL loader compatible with the Grain `RandomAccessDataSource`
    Protocol) with slicing support to split training data.

## Expected Output

*   **Train split**: `ori_pqal.json` (1,000 expert-labeled examples) **minus**
    `test_set.json` (500 test IDs) → `pubmedqa_train.jsonl` (500 clean training
    examples).
*   **Test split**: `test_set.json` (500 official test examples) →
    `pubmedqa_test.jsonl`.
