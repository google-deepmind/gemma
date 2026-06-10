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

"""Sudoku evaluation utilities: scoring and accuracy metric."""

from __future__ import annotations

import dataclasses
import enum
import re

import flax.struct
from gemma.diffusion.hackable_diffusion_adapter.eval import base_metric
import jax.numpy as jnp
from kauldron import kd
import numpy as np

################################################################################
# MARK: Constants
################################################################################

_NUM_CELLS = 81


class ExtractionMode(enum.Enum):
  """Strategy for extracting the 81-digit Sudoku solution from model output.

  Attributes:
    SFT: Look for the ``"the answer is"`` delimiter (case-insensitive) and
      extract the first 81 digits after it.  Falls back to the last 81 digits if
      the delimiter is missing.
    THINKING: Always take the last 81 digits in the text.  This is robust when
      the model produces chain-of-thought reasoning before the answer.
  """

  SFT = "sft"
  THINKING = "thinking"


class HardnessCategory(enum.Enum):
  """Filter or stratification bin for evaluation by Sudoku hardness.

  Hardness is determined by the number of unmasked (given) clues in the puzzle.

  Attributes:
    ALL: All items in the dataset are evaluated.
    EASY: Puzzles with 40 or more unmasked clues (given cells >= 40).
    MEDIUM: Puzzles with at least 30 and fewer than 40 unmasked clues.
    HARD: Puzzles with fewer than 30 unmasked clues (given cells < 30).
  """

  ALL = "all"
  EASY = "easy"
  MEDIUM = "medium"
  HARD = "hard"


def matches_hardness(
    num_unmasked: int,
    category: HardnessCategory,
) -> bool:
  """Check if the number of unmasked clues matches a hardness category."""
  if category is HardnessCategory.ALL:
    return True
  elif category is HardnessCategory.EASY:
    return num_unmasked >= 40
  elif category is HardnessCategory.MEDIUM:
    return 30 <= num_unmasked < 40
  elif category is HardnessCategory.HARD:
    return num_unmasked < 30
  else:
    raise ValueError(f"Unknown hardness category: {category}")


################################################################################
# MARK: Extraction
################################################################################


def extract_sudoku_solution(
    text: str,
    mode: ExtractionMode = ExtractionMode.SFT,
) -> str | None:
  """Extract the 81-digit Sudoku solution from generated text.

  Args:
    text: The raw text produced by the model.
    mode: Extraction strategy — see ``ExtractionMode``.

  Returns:
    A string of exactly 81 digit characters, or ``None`` if extraction fails.
  """
  if mode is ExtractionMode.SFT:
    # Primary: extract after "the answer is" (case-insensitive).
    match = re.search(r"(?i)the\s+answer\s+is[:\s]*(.*)", text, re.DOTALL)
    if match:
      ans = match.group(1)
      ans_clean = "".join(c for c in ans if c.isdigit())
      if len(ans_clean) >= _NUM_CELLS:
        return ans_clean[:_NUM_CELLS]

    # Fallback: last 81 digits in the entire text.
    all_digits = "".join(c for c in text if c.isdigit())
    if len(all_digits) >= _NUM_CELLS:
      return all_digits[-_NUM_CELLS:]

    return None

  elif mode is ExtractionMode.THINKING:
    # Always take the last 81 digits.
    all_digits = "".join(c for c in text if c.isdigit())
    if len(all_digits) >= _NUM_CELLS:
      return all_digits[-_NUM_CELLS:]
    return None

  else:
    raise ValueError(f"Unknown extraction mode: {mode}")


################################################################################
# MARK: Scoring helpers
################################################################################


def _clean_sudoku_str(s: str) -> str:
  """Strip whitespace and turn tokens from a Sudoku string."""
  cleaned = "".join(s.split()).strip()
  cleaned = cleaned.replace("<turn|>", "").replace("<|turn>", "")
  return cleaned


def _unmasked_preserved(
    gen_digits: str,
    puzzle_text: str,
) -> bool:
  """Check that the model did not overwrite any unmasked (given) cell."""
  for j in range(_NUM_CELLS):
    if puzzle_text[j] != "0" and gen_digits[j] != puzzle_text[j]:
      return False
  return True


def exact_accuracy(
    generated: str,
    ground_truth: str,
    puzzle: str,
    mode: ExtractionMode = ExtractionMode.SFT,
) -> float:
  """Check whether the generated Sudoku solution matches the ground truth.

  Extracts the 81-digit solution from the generated text and compares it
  to the cleaned ground truth.

  Args:
    generated: The generated text.
    ground_truth: The ground-truth solution string.
    puzzle: The original puzzle string (unused, kept for API consistency with
      the other scoring helpers).
    mode: Extraction strategy — see ``ExtractionMode``.

  Returns:
    ``1.0`` if the generated solution matches the ground truth, ``0.0``
    otherwise.
  """
  del puzzle  # Unused — exact accuracy compares full grids.
  gen_clean = extract_sudoku_solution(generated, mode=mode)
  if gen_clean is None:
    return 0.0

  gt_clean = _clean_sudoku_str(ground_truth)

  return 1.0 if gen_clean == gt_clean else 0.0


def partial_accuracy(
    generated: str,
    ground_truth: str,
    puzzle: str,
    mode: ExtractionMode = ExtractionMode.SFT,
) -> float:
  """Number of correctly infilled masked digits divided by 81.

  Args:
    generated: The generated text.
    ground_truth: The ground-truth solution string (space-separated digits).
    puzzle: The original puzzle string (space-separated digits, ``0`` = blank).
    mode: Extraction strategy — see ``ExtractionMode``.

  Returns:
    A float in ``[0, 1]`` representing the number of correctly predicted
    masked cells divided by 81.  Returns ``0.0`` if the inputs are invalid or
    the solution could not be extracted.
  """
  gen_digits = extract_sudoku_solution(generated, mode=mode)
  gt_text = _clean_sudoku_str(ground_truth)
  puzzle_text = _clean_sudoku_str(puzzle)

  if len(puzzle_text) != _NUM_CELLS or len(gt_text) != _NUM_CELLS:
    return 0.0

  if gen_digits is None:
    return 0.0

  masked_indices = [j for j in range(_NUM_CELLS) if puzzle_text[j] == "0"]
  correct = sum(1 for j in masked_indices if gen_digits[j] == gt_text[j])
  return correct / _NUM_CELLS


def exact_mask_accuracy(
    generated: str,
    ground_truth: str,
    puzzle: str,
    mode: ExtractionMode = ExtractionMode.SFT,
) -> float:
  """Check that the model did not overwrite any unmasked (given) digit.

  Returns ``1.0`` if every non-zero cell in the puzzle is identical in the
  generated output.  Does **not** verify correctness of masked cells.

  Args:
    generated: The generated text.
    ground_truth: The ground-truth solution string (unused, kept for API
      consistency with the other scoring helpers).
    puzzle: The original puzzle string (space-separated digits, ``0`` = blank).
    mode: Extraction strategy — see ``ExtractionMode``.

  Returns:
    ``1.0`` if all unmasked cells are preserved, ``0.0`` otherwise.
  """
  del ground_truth  # Unused — only puzzle cells matter.
  gen_digits = extract_sudoku_solution(generated, mode=mode)
  if gen_digits is None:
    return 0.0

  puzzle_text = _clean_sudoku_str(puzzle)
  if len(puzzle_text) != _NUM_CELLS:
    return 0.0

  return 1.0 if _unmasked_preserved(gen_digits, puzzle_text) else 0.0


def partial_mask_accuracy(
    generated: str,
    ground_truth: str,
    puzzle: str,
    mode: ExtractionMode = ExtractionMode.SFT,
) -> float:
  """Fraction of unmasked (given) cells that were preserved by the model.

  Reports how many of the non-zero puzzle cells the model kept unchanged,
  divided by the total number of unmasked cells.  Does **not** check
  masked-cell correctness.

  Args:
    generated: The generated text.
    ground_truth: The ground-truth solution string (unused, kept for API
      consistency with the other scoring helpers).
    puzzle: The original puzzle string (space-separated digits, ``0`` = blank).
    mode: Extraction strategy — see ``ExtractionMode``.

  Returns:
    A float in ``[0, 1]``.  ``1.0`` means every unmasked cell was preserved.
    Returns ``0.0`` if extraction fails or the puzzle is invalid.
  """
  del ground_truth  # Unused — only puzzle cells matter.
  gen_digits = extract_sudoku_solution(generated, mode=mode)
  puzzle_text = _clean_sudoku_str(puzzle)

  if len(puzzle_text) != _NUM_CELLS:
    return 0.0

  if gen_digits is None:
    return 0.0

  unmasked_indices = [j for j in range(_NUM_CELLS) if puzzle_text[j] != "0"]
  if not unmasked_indices:
    return 1.0

  preserved = sum(
      1 for j in unmasked_indices if gen_digits[j] == puzzle_text[j]
  )
  return preserved / len(unmasked_indices)


################################################################################
# MARK: Metric
################################################################################

_ALL_SCORING_FNS = (
    exact_accuracy,
    partial_accuracy,
    exact_mask_accuracy,
    partial_mask_accuracy,
)
_NUM_MODES = len(_ALL_SCORING_FNS)

_HARDNESS_NAMES = ("", "_easy", "_medium", "_hard")
_HARDNESS_CATEGORIES = tuple(HardnessCategory)
_SCORING_NAMES = (
    "accuracy",
    "cell_accuracy",
    "exact_mask_accuracy",
    "partial_mask_accuracy",
)


@dataclasses.dataclass(kw_only=True, frozen=True)
class SudokuAllMetrics(base_metric.BaseTextMetric):
  """Compute all Sudoku accuracy modes and sample counts.

  Stores raw token arrays in the metric state and performs all detokenization
  and scoring on the host in ``compute()``.

  Reports 20 sub-metrics:
  ``{scoring_mode}``: EXACT, PARTIAL, EXACT_MASK, PARTIAL_MASK for ALL.
  ``{scoring_mode}_{hardness}``: for EASY, MEDIUM, HARD categories.
  ``count``, ``count_{hardness}``: sample counts per hardness category.
  """

  puzzle: kd.kontext.Key = kd.kontext.REQUIRED
  extraction_mode: ExtractionMode = ExtractionMode.SFT

  def __metric_names__(self) -> list[str]:
    names = []
    for cat_suffix in _HARDNESS_NAMES:
      for mode_name in _SCORING_NAMES:
        names.append(f"sudoku_{mode_name}{cat_suffix}")
    for cat_suffix in _HARDNESS_NAMES:
      names.append(f"sudoku_count{cat_suffix}")
    return names

  @flax.struct.dataclass
  class State(kd.metrics.AutoState["SudokuAllMetrics"]):
    """Collects raw tokens across batches for host-side scoring."""

    tokens: jnp.ndarray = kd.metrics.concat_field()
    ground_truth: jnp.ndarray = kd.metrics.concat_field()
    puzzle: jnp.ndarray = kd.metrics.concat_field()

    def compute(self) -> dict[str, float]:
      """Detokenize and score all accumulated examples on host."""
      final = self.finalize()
      tokens_np = np.asarray(final.tokens)
      gt_np = np.asarray(final.ground_truth)
      puzzle_np = np.asarray(final.puzzle)

      generated_texts = self.parent.decode_batch(tokens_np)
      gt_strs = self.parent.decode_batch(gt_np)
      puzzle_strs = self.parent.decode_batch(puzzle_np)

      extraction_mode = self.parent.extraction_mode

      num_examples = len(generated_texts)
      totals = np.zeros((4, _NUM_MODES), dtype=np.float64)
      counts = np.zeros((4, _NUM_MODES), dtype=np.float64)
      category_counts = np.zeros(4, dtype=np.float64)

      for i in range(num_examples):
        scores_i = [
            fn(
                generated_texts[i],
                gt_strs[i],
                puzzle_strs[i],
                extraction_mode,
            )
            for fn in _ALL_SCORING_FNS
        ]

        puzzle_clean = _clean_sudoku_str(puzzle_strs[i])
        num_unmasked_i = sum(1 for c in puzzle_clean if c != "0")

        for cat_idx, cat in enumerate(_HARDNESS_CATEGORIES):
          if matches_hardness(num_unmasked_i, cat):
            category_counts[cat_idx] += 1
            for mode_idx, score in enumerate(scores_i):
              totals[cat_idx, mode_idx] += score
              counts[cat_idx, mode_idx] += 1

      result: dict[str, float] = {}
      for cat_idx, cat_suffix in enumerate(_HARDNESS_NAMES):
        for mode_idx, mode_name in enumerate(_SCORING_NAMES):
          key = f"sudoku_{mode_name}{cat_suffix}"
          c = counts[cat_idx, mode_idx]
          result[key] = totals[cat_idx, mode_idx] / c if c > 0 else 0.0

      for cat_idx, cat_suffix in enumerate(_HARDNESS_NAMES):
        result[f"sudoku_count{cat_suffix}"] = float(category_counts[cat_idx])

      return result

  def get_state(
      self,
      *,
      tokens: jnp.ndarray,
      ground_truth: jnp.ndarray,
      puzzle: jnp.ndarray,
  ) -> "SudokuAllMetrics.State":
    """Store raw tokens — all scoring is deferred to ``compute()``."""
    return self.State(
        tokens=self._squeeze_3d(tokens),
        ground_truth=self._squeeze_3d(ground_truth),
        puzzle=self._squeeze_3d(puzzle),
    )
