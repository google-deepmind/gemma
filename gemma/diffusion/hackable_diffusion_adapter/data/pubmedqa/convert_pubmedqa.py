# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert PubMedQA JSON data to JSONL format for HD text diffusion SFT.

Reads raw PubMedQA JSON from source directory and writes JSONL files with
fields:
  - prompt: Formatted question + context
  - response: Short answer (yes/no/maybe)
  - long_response: Long-form answer
  - short_answer: Ground truth label (yes/no/maybe) for evaluation

Usage:
  python convert_pubmedqa.py --output_dir=./pubmedqa_jsonl
"""

import json
import os
from typing import Any

from absl import app
from absl import flags

_INPUT_DIR = flags.DEFINE_string(
    "input_dir",
    "gemma/diffusion/hackable_diffusion_adapter/data/pubmedqa",
    "Source directory containing PubMedQA JSON files.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "",
    "Output directory for JSONL files. Required.",
)
_MAX_CONTEXT_CHARS = flags.DEFINE_integer(
    "max_context_chars",
    4000,
    "Maximum total characters for concatenated context paragraphs.",
)


def _read_file(path: str) -> str:
  """Read a file using standard Python file I/O."""
  with open(path, "r", encoding="utf-8") as f:
    return f.read()


def _write_file(path: str, content: str) -> None:
  """Write a file using standard Python file I/O."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    f.write(content)


def _format_prompt(question: str, contexts: list[str]) -> str:
  """Format a PubMedQA example into a clean prompt.

  Concatenates all context paragraphs (truncating if needed) with the question.

  Args:
    question: The question string.
    contexts: A list of context paragraph strings.

  Returns:
    The formatted prompt string.
  """
  context_text = "\n".join(contexts)
  if len(context_text) > _MAX_CONTEXT_CHARS.value:
    context_text = context_text[: _MAX_CONTEXT_CHARS.value] + "..."

  return f"Context:\n{context_text}\n\nQuestion: {question}"


def _convert_json_to_examples(data: dict[str, Any]) -> list[dict[str, Any]]:
  """Convert PubMedQA nested JSON to flat list of examples."""
  examples = []
  for pubmed_id, entry in data.items():
    question = entry.get("QUESTION", "")
    contexts = entry.get("CONTEXTS", [])
    short_answer = entry.get("final_decision", "")
    long_answer = entry.get("LONG_ANSWER", "")

    prompt = _format_prompt(question, contexts)

    examples.append({
        "pubmed_id": str(pubmed_id),
        "prompt": prompt,
        "response": short_answer,
        "long_response": long_answer,
        "short_answer": short_answer,
    })
  return examples


def _write_jsonl(examples: list[dict[str, Any]], output_path: str) -> None:
  """Write examples as JSONL (one JSON object per line)."""
  lines = [json.dumps(ex, ensure_ascii=False) for ex in examples]
  content = "\n".join(lines) + "\n"
  _write_file(output_path, content)
  print(f"Wrote {len(examples)} examples to {output_path}")


def main(argv):
  del argv

  if not _OUTPUT_DIR.value:
    raise ValueError("--output_dir is required.")

  input_dir = _INPUT_DIR.value
  output_dir = _OUTPUT_DIR.value

  # --- Load full PQA-L dataset (1000 expert-labeled examples) ---
  pqal_path = os.path.join(input_dir, "ori_pqal.json")
  print(f"Reading PQA-L data from {pqal_path}...")
  pqal_data = json.loads(_read_file(pqal_path))

  # --- Load official test set ground truth (to get test IDs) ---
  test_gt_path = os.path.join(input_dir, "test_ground_truth.json")
  print(f"Reading test ground truth from {test_gt_path}...")
  test_gt_data = json.loads(_read_file(test_gt_path))
  test_ids = set(test_gt_data.keys())

  # --- Split full dataset into train and test ---
  # ori_pqal.json contains all 1000 PQA-L examples, including the 500 that
  # are officially designated as the test set. We split them using test_ids.
  train_data = {k: v for k, v in pqal_data.items() if k not in test_ids}
  test_data = {k: v for k, v in pqal_data.items() if k in test_ids}

  # --- Convert ---
  train_examples = _convert_json_to_examples(train_data)
  test_examples = _convert_json_to_examples(test_data)

  overlap = set(e["pubmed_id"] for e in train_examples) & set(
      e["pubmed_id"] for e in test_examples
  )
  if overlap:
    raise ValueError(
        f"Train/test overlap detected ({len(overlap)} IDs)! This should not"
        " happen after filtering."
    )

  print(f"Training examples: {len(train_examples)} (after removing test IDs)")
  print(f"Test examples: {len(test_examples)}")

  # --- Write JSONL ---
  _write_jsonl(train_examples, os.path.join(output_dir, "pubmedqa_train.jsonl"))
  _write_jsonl(test_examples, os.path.join(output_dir, "pubmedqa_test.jsonl"))

  print("Done!")


if __name__ == "__main__":
  app.run(main)
