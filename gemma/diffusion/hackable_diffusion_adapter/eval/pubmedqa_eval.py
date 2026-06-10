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

"""PubMedQA evaluation: answer extraction, accuracy, and BLEU metrics."""

from __future__ import annotations

import dataclasses
import re

from gemma.diffusion.hackable_diffusion_adapter.eval import base_metric
import sacrebleu


def extract_pubmedqa_answer(text: str) -> str | None:
  """Extract yes/no/maybe answer from generated text.

  Uses a single, reliable strategy: look for the explicit marker
  ``"The answer is: {answer}"`` which the structured system prompt
  instructs the model to produce.

  No fallback heuristics are used (e.g. checking the first word or
  scanning for any occurrence of yes/no/maybe) because they are too
  fragile and lead to false positives — for example "yesterday" would
  match "yes", or "no significant effect" would match "no".

  Args:
    text: Generated text that should contain a yes/no/maybe answer.

  Returns:
    Extracted answer string ('yes', 'no', or 'maybe'), or None if not found.
  """
  text_lower = text.strip().lower()

  # Explicit "The answer is: {answer}" marker.
  marker_match = re.search(
      r'the answer is[:\s]+\b(yes|no|maybe)\b', text_lower
  )
  if marker_match:
    return marker_match.group(1)

  return None


def score_pubmedqa(generated: str, ground_truth: str) -> bool:
  """Check whether the generated answer matches the ground truth.

  Args:
    generated: Full generated text.
    ground_truth: The ground-truth answer string ('yes', 'no', or 'maybe').

  Returns:
    True if the extracted answer matches.
  """
  gen_ans = extract_pubmedqa_answer(generated)
  gt_ans = ground_truth.strip().lower()
  if gen_ans is None:
    return False
  return gen_ans == gt_ans


@dataclasses.dataclass(kw_only=True, frozen=True)
class PubMedQAAccuracy(base_metric.BaseSimpleTextMetric):
  """Compute accuracy on PubMedQA by extracting yes/no/maybe from samples.

  This metric detokenizes generated token sequences, extracts the
  yes/no/maybe answer, and compares against the ground-truth short_answer.
  """

  def score_example(self, generated_text: str, ground_truth_text: str) -> bool:
    """Scores a single example."""
    return score_pubmedqa(generated_text, ground_truth_text)


def score_bleu(generated: str, ground_truth: str) -> float:
  """Compute smoothed sentence-level BLEU between generated and ground truth.

  Args:
    generated: The text produced by the model.
    ground_truth: The reference text.

  Returns:
    Sentence-level BLEU score in [0, 100].
  """
  generated = generated.strip()
  ground_truth = ground_truth.strip()

  if not generated or not ground_truth:
    return 0.0

  bleu = sacrebleu.corpus_bleu(
      [generated],
      [[ground_truth]],
      smooth_method='floor',
      use_effective_order=True,
  )
  return bleu.score


@dataclasses.dataclass(kw_only=True, frozen=True)
class BLEUScore(base_metric.BaseSimpleTextMetric):
  """Compute sentence-level BLEU between generated and ground-truth text.

  This metric detokenizes generated token sequences and computes
  smoothed sentence-level BLEU against the ground-truth tokens.
  """

  def score_example(self, generated_text: str, ground_truth_text: str) -> float:
    """Scores a single example."""
    return score_bleu(generated_text, ground_truth_text)
