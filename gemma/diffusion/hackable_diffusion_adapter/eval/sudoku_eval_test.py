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

"""Tests for Sudoku evaluation scoring helpers."""

from absl.testing import absltest
from absl.testing import parameterized
from gemma.diffusion.hackable_diffusion_adapter.eval import sudoku_eval
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Helpers: a complete valid puzzle / solution pair for tests.
# ---------------------------------------------------------------------------

# 81-digit ground truth (all digits 1-9, space-separated).
_GT_81 = (
    "5 3 4 6 7 8 9 1 2 "
    "6 7 2 1 9 5 3 4 8 "
    "1 9 8 3 4 2 5 6 7 "
    "8 5 9 7 6 1 4 2 3 "
    "4 2 6 8 5 3 7 9 1 "
    "7 1 3 9 2 4 8 5 6 "
    "9 6 1 5 3 7 2 8 4 "
    "2 8 7 4 1 9 6 3 5 "
    "3 4 5 2 8 6 1 7 9"
)
# Cleaned 81-char version (no spaces).
_GT_CLEAN = (
    "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
)

# Puzzle with some zeros (masked cells).  Non-zero cells match _GT_CLEAN.
_PUZZLE_81 = (
    "5 3 0 0 7 0 0 0 0 "
    "6 0 0 1 9 5 0 0 0 "
    "0 9 8 0 0 0 0 6 0 "
    "8 0 0 0 6 0 0 0 3 "
    "4 0 0 8 0 3 0 0 1 "
    "7 0 0 0 2 0 0 0 6 "
    "0 6 0 0 0 0 2 8 0 "
    "0 0 0 4 1 9 0 0 5 "
    "0 0 0 0 8 0 0 7 9"
)
_PUZZLE_CLEAN = (
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
)

_NUM_MASKED = _PUZZLE_CLEAN.count("0")
_NUM_UNMASKED = 81 - _NUM_MASKED


# ---------------------------------------------------------------------------
# extract_sudoku_solution
# ---------------------------------------------------------------------------


class ExtractSudokuSolutionTest(parameterized.TestCase):
  """Tests for ``extract_sudoku_solution``."""

  def test_sft_mode_with_delimiter(self):
    text = "Some preamble. The answer is: " + _GT_CLEAN
    result = sudoku_eval.extract_sudoku_solution(
        text, mode=sudoku_eval.ExtractionMode.SFT
    )
    self.assertEqual(result, _GT_CLEAN)

  def test_sft_mode_case_insensitive(self):
    text = "THE ANSWER IS " + _GT_CLEAN
    result = sudoku_eval.extract_sudoku_solution(
        text, mode=sudoku_eval.ExtractionMode.SFT
    )
    self.assertEqual(result, _GT_CLEAN)

  def test_sft_mode_with_spaces_in_answer(self):
    spaced = " ".join(_GT_CLEAN)
    text = "The answer is: " + spaced
    result = sudoku_eval.extract_sudoku_solution(
        text, mode=sudoku_eval.ExtractionMode.SFT
    )
    self.assertEqual(result, _GT_CLEAN)

  def test_sft_mode_fallback_no_delimiter(self):
    text = "Here is the grid: " + _GT_CLEAN
    result = sudoku_eval.extract_sudoku_solution(
        text, mode=sudoku_eval.ExtractionMode.SFT
    )
    self.assertEqual(result, _GT_CLEAN)

  def test_sft_mode_too_few_digits(self):
    text = "The answer is: 123"
    result = sudoku_eval.extract_sudoku_solution(
        text, mode=sudoku_eval.ExtractionMode.SFT
    )
    self.assertIsNone(result)

  def test_thinking_mode_last_81(self):
    text = "Step 1: cell is 3. Step 2: cell is 5. " + _GT_CLEAN
    result = sudoku_eval.extract_sudoku_solution(
        text, mode=sudoku_eval.ExtractionMode.THINKING
    )
    self.assertEqual(result, _GT_CLEAN)

  def test_thinking_mode_too_few_digits(self):
    text = "only 12345"
    result = sudoku_eval.extract_sudoku_solution(
        text, mode=sudoku_eval.ExtractionMode.THINKING
    )
    self.assertIsNone(result)

  def test_thinking_mode_ignores_delimiter(self):
    """THINKING mode should NOT use the delimiter — just last 81 digits."""
    text = "99 The answer is: " + _GT_CLEAN
    result = sudoku_eval.extract_sudoku_solution(
        text, mode=sudoku_eval.ExtractionMode.THINKING
    )
    # THINKING grabs the last 81 digits, which is still _GT_CLEAN.
    self.assertEqual(result, _GT_CLEAN)


# ---------------------------------------------------------------------------
# exact_accuracy
# ---------------------------------------------------------------------------


class ExactAccuracyTest(absltest.TestCase):
  """Tests for ``exact_accuracy``."""

  def test_correct(self):
    self.assertEqual(
        sudoku_eval.exact_accuracy(_GT_CLEAN, _GT_81, ""), 1.0
    )

  def test_incorrect(self):
    wrong = "1" + _GT_CLEAN[1:]
    self.assertEqual(sudoku_eval.exact_accuracy(wrong, _GT_81, ""), 0.0)

  def test_extraction_fails(self):
    self.assertEqual(
        sudoku_eval.exact_accuracy("no digits", _GT_81, ""), 0.0
    )


# ---------------------------------------------------------------------------
# partial_accuracy
# ---------------------------------------------------------------------------


class PartialAccuracyTest(absltest.TestCase):
  """Tests for ``partial_accuracy``."""

  def test_perfect_solution(self):
    score = sudoku_eval.partial_accuracy(_GT_CLEAN, _GT_81, _PUZZLE_81)
    self.assertAlmostEqual(score, _NUM_MASKED / 81)

  def test_all_wrong_masked_cells(self):
    gen = list(_GT_CLEAN)
    for j in range(81):
      if _PUZZLE_CLEAN[j] == "0":
        gen[j] = "1" if _GT_CLEAN[j] != "1" else "2"
    gen_str = "".join(gen)
    score = sudoku_eval.partial_accuracy(gen_str, _GT_81, _PUZZLE_81)
    self.assertAlmostEqual(score, 0.0)

  def test_extraction_fails(self):
    self.assertAlmostEqual(
        sudoku_eval.partial_accuracy("no digits", _GT_81, _PUZZLE_81), 0.0
    )

  def test_invalid_puzzle_length(self):
    self.assertAlmostEqual(
        sudoku_eval.partial_accuracy(_GT_CLEAN, _GT_81, "123"), 0.0
    )

  def test_divides_by_81_not_num_masked(self):
    """Score = num_correct_masked / 81, NOT / num_masked."""
    gen = list(_GT_CLEAN)
    masked_indices = [j for j in range(81) if _PUZZLE_CLEAN[j] == "0"]
    # Keep only the first masked cell correct; corrupt the rest.
    for idx, j in enumerate(masked_indices):
      if idx > 0:
        gen[j] = "1" if _GT_CLEAN[j] != "1" else "2"
    gen_str = "".join(gen)
    score = sudoku_eval.partial_accuracy(gen_str, _GT_81, _PUZZLE_81)
    self.assertAlmostEqual(score, 1.0 / 81)


# ---------------------------------------------------------------------------
# exact_mask_accuracy
# ---------------------------------------------------------------------------


class ExactMaskAccuracyTest(absltest.TestCase):
  """Tests for ``exact_mask_accuracy``.

  This metric checks ONLY that unmasked (given) cells are preserved.
  It does NOT care about masked-cell correctness.
  """

  def test_all_unmasked_preserved(self):
    """Perfect solution preserves all unmasked cells → 1.0."""
    score = sudoku_eval.exact_mask_accuracy(_GT_CLEAN, _GT_81, _PUZZLE_81)
    self.assertAlmostEqual(score, 1.0)

  def test_unmasked_overwritten(self):
    """Overwriting an unmasked cell → 0.0."""
    gen = list(_GT_CLEAN)
    gen[0] = "9"  # Puzzle[0] = '5' (unmasked), now '9'.
    gen_str = "".join(gen)
    score = sudoku_eval.exact_mask_accuracy(gen_str, _GT_81, _PUZZLE_81)
    self.assertAlmostEqual(score, 0.0)

  def test_masked_wrong_but_unmasked_ok(self):
    """Wrong masked cells but unmasked preserved → 1.0."""
    gen = list(_GT_CLEAN)
    masked_idx = next(j for j in range(81) if _PUZZLE_CLEAN[j] == "0")
    gen[masked_idx] = "1" if _GT_CLEAN[masked_idx] != "1" else "2"
    gen_str = "".join(gen)
    score = sudoku_eval.exact_mask_accuracy(gen_str, _GT_81, _PUZZLE_81)
    self.assertAlmostEqual(score, 1.0)

  def test_extraction_fails(self):
    score = sudoku_eval.exact_mask_accuracy("no digits", _GT_81, _PUZZLE_81)
    self.assertAlmostEqual(score, 0.0)

  def test_invalid_puzzle(self):
    score = sudoku_eval.exact_mask_accuracy(_GT_CLEAN, _GT_81, "short")
    self.assertAlmostEqual(score, 0.0)


# ---------------------------------------------------------------------------
# partial_mask_accuracy
# ---------------------------------------------------------------------------


class PartialMaskAccuracyTest(absltest.TestCase):
  """Tests for ``partial_mask_accuracy``.

  This metric reports the fraction of unmasked (given) cells that are
  preserved.  It does NOT care about masked-cell correctness.
  """

  def test_all_unmasked_preserved(self):
    """Perfect solution → all unmasked cells preserved → 1.0."""
    score = sudoku_eval.partial_mask_accuracy(_GT_CLEAN, _GT_81, _PUZZLE_81)
    self.assertAlmostEqual(score, 1.0)

  def test_one_unmasked_overwritten(self):
    """Overwriting one unmasked cell → (_NUM_UNMASKED - 1) / _NUM_UNMASKED."""
    gen = list(_GT_CLEAN)
    gen[0] = "9"  # Puzzle[0] = '5' (unmasked), now '9'.
    gen_str = "".join(gen)
    score = sudoku_eval.partial_mask_accuracy(gen_str, _GT_81, _PUZZLE_81)
    expected = (_NUM_UNMASKED - 1) / _NUM_UNMASKED
    self.assertAlmostEqual(score, expected)

  def test_all_unmasked_overwritten(self):
    """Overwriting every unmasked cell → 0.0."""
    gen = list(_GT_CLEAN)
    for j in range(81):
      if _PUZZLE_CLEAN[j] != "0":
        gen[j] = "1" if _PUZZLE_CLEAN[j] != "1" else "2"
    gen_str = "".join(gen)
    score = sudoku_eval.partial_mask_accuracy(gen_str, _GT_81, _PUZZLE_81)
    self.assertAlmostEqual(score, 0.0)

  def test_masked_wrong_but_unmasked_ok(self):
    """Wrong masked cells but unmasked preserved → 1.0."""
    gen = list(_GT_CLEAN)
    masked_idx = next(j for j in range(81) if _PUZZLE_CLEAN[j] == "0")
    gen[masked_idx] = "1" if _GT_CLEAN[masked_idx] != "1" else "2"
    gen_str = "".join(gen)
    score = sudoku_eval.partial_mask_accuracy(gen_str, _GT_81, _PUZZLE_81)
    self.assertAlmostEqual(score, 1.0)

  def test_extraction_fails(self):
    score = sudoku_eval.partial_mask_accuracy("no digits", _GT_81, _PUZZLE_81)
    self.assertAlmostEqual(score, 0.0)


# ---------------------------------------------------------------------------
# _clean_sudoku_str
# ---------------------------------------------------------------------------


class CleanSudokuStrTest(absltest.TestCase):
  """Tests for ``_clean_sudoku_str``."""

  def test_strips_spaces(self):
    self.assertEqual(sudoku_eval._clean_sudoku_str("1 2 3"), "123")

  def test_strips_turn_tokens(self):
    self.assertEqual(
        sudoku_eval._clean_sudoku_str("<|turn>123<turn|>"), "123"
    )


# ---------------------------------------------------------------------------
# Metric classes integration tests
# ---------------------------------------------------------------------------


class SudokuAllMetricsTest(absltest.TestCase):
  """Tests for ``SudokuAllMetrics``."""

  def _make_mock_tokenizer(self):
    """Returns a mock tokenizer that maps token IDs to known strings."""

    class MockTokenizer:
      def decode(self, ids):
        val = ids[0]
        if val == 10 or val == 100:
          return _GT_CLEAN
        elif val == 20:
          # Partially correct solution: corrupt first 5 masked cells.
          gen = list(_GT_CLEAN)
          masked_indices = [j for j in range(81) if _PUZZLE_CLEAN[j] == "0"]
          for j in masked_indices[:5]:
            gen[j] = "1" if _GT_CLEAN[j] != "1" else "2"
          return "".join(gen)
        elif val == 30:
          return _GT_81
        elif val == 40:
          return _PUZZLE_81
        return ""

    return MockTokenizer()

  def test_perfect_solution_scores(self):
    """SudokuAllMetrics should match the scoring helpers for a perfect solution."""
    metric = sudoku_eval.SudokuAllMetrics(
        tokens="samples",
        ground_truth="batch.solution_tokens",
        puzzle="batch.puzzle_tokens",
    )
    metric.__dict__["tokenizer"] = self._make_mock_tokenizer()

    tokens = jnp.array([[10]], dtype=jnp.int32)
    gt_tokens = jnp.array([[30]], dtype=jnp.int32)
    puzzle_tokens = jnp.array([[40]], dtype=jnp.int32)

    state = metric.get_state(
        tokens=tokens, ground_truth=gt_tokens, puzzle=puzzle_tokens
    )
    result = state.compute()

    # Compute expected values from the scoring helpers directly.
    expected_exact = sudoku_eval.exact_accuracy(_GT_CLEAN, _GT_81, _PUZZLE_81)
    expected_partial = sudoku_eval.partial_accuracy(
        _GT_CLEAN, _GT_81, _PUZZLE_81
    )
    expected_exact_mask = sudoku_eval.exact_mask_accuracy(
        _GT_CLEAN, _GT_81, _PUZZLE_81
    )
    expected_partial_mask = sudoku_eval.partial_mask_accuracy(
        _GT_CLEAN, _GT_81, _PUZZLE_81
    )

    self.assertAlmostEqual(result["sudoku_accuracy"], expected_exact)
    self.assertAlmostEqual(result["sudoku_cell_accuracy"], expected_partial)
    self.assertAlmostEqual(
        result["sudoku_exact_mask_accuracy"], expected_exact_mask
    )
    self.assertAlmostEqual(
        result["sudoku_partial_mask_accuracy"], expected_partial_mask
    )

  def test_partial_solution_scores(self):
    """SudokuAllMetrics should match the scoring helpers for a partial solution."""
    metric = sudoku_eval.SudokuAllMetrics(
        tokens="samples",
        ground_truth="batch.solution_tokens",
        puzzle="batch.puzzle_tokens",
    )
    mock_tok = self._make_mock_tokenizer()
    metric.__dict__["tokenizer"] = mock_tok

    tokens = jnp.array([[20]], dtype=jnp.int32)
    gt_tokens = jnp.array([[30]], dtype=jnp.int32)
    puzzle_tokens = jnp.array([[40]], dtype=jnp.int32)

    state = metric.get_state(
        tokens=tokens, ground_truth=gt_tokens, puzzle=puzzle_tokens
    )
    result = state.compute()

    # Decode what the mock tokenizer would produce for token 20.
    generated_text = mock_tok.decode([20])
    expected_exact = sudoku_eval.exact_accuracy(
        generated_text, _GT_81, _PUZZLE_81
    )
    expected_partial = sudoku_eval.partial_accuracy(
        generated_text, _GT_81, _PUZZLE_81
    )
    expected_exact_mask = sudoku_eval.exact_mask_accuracy(
        generated_text, _GT_81, _PUZZLE_81
    )
    expected_partial_mask = sudoku_eval.partial_mask_accuracy(
        generated_text, _GT_81, _PUZZLE_81
    )

    self.assertAlmostEqual(result["sudoku_accuracy"], expected_exact)
    self.assertAlmostEqual(result["sudoku_cell_accuracy"], expected_partial)
    self.assertAlmostEqual(
        result["sudoku_exact_mask_accuracy"], expected_exact_mask
    )
    self.assertAlmostEqual(
        result["sudoku_partial_mask_accuracy"], expected_partial_mask
    )

  def test_merge_averages_correctly(self):
    """Merging two single-sample states should average the scores."""
    metric = sudoku_eval.SudokuAllMetrics(
        tokens="samples",
        ground_truth="batch.solution_tokens",
        puzzle="batch.puzzle_tokens",
    )
    metric.__dict__["tokenizer"] = self._make_mock_tokenizer()

    gt_tokens = jnp.array([[30]], dtype=jnp.int32)
    puzzle_tokens = jnp.array([[40]], dtype=jnp.int32)

    state_perfect = metric.get_state(
        tokens=jnp.array([[10]], dtype=jnp.int32),
        ground_truth=gt_tokens,
        puzzle=puzzle_tokens,
    )
    state_partial = metric.get_state(
        tokens=jnp.array([[20]], dtype=jnp.int32),
        ground_truth=gt_tokens,
        puzzle=puzzle_tokens,
    )

    merged = state_perfect.merge(state_partial)
    res = merged.compute()

    res_perfect = state_perfect.compute()
    res_partial = state_partial.compute()

    for key in (
        "sudoku_accuracy",
        "sudoku_cell_accuracy",
        "sudoku_exact_mask_accuracy",
        "sudoku_partial_mask_accuracy",
    ):
      expected = (res_perfect[key] + res_partial[key]) / 2
      self.assertAlmostEqual(res[key], expected, msg=f"Mismatch for {key}")

  def test_count_reported(self):
    """The count sub-metric should reflect the number of samples."""
    metric = sudoku_eval.SudokuAllMetrics(
        tokens="samples",
        ground_truth="batch.solution_tokens",
        puzzle="batch.puzzle_tokens",
    )
    metric.__dict__["tokenizer"] = self._make_mock_tokenizer()

    state = metric.get_state(
        tokens=jnp.array([[10]], dtype=jnp.int32),
        ground_truth=jnp.array([[30]], dtype=jnp.int32),
        puzzle=jnp.array([[40]], dtype=jnp.int32),
    )
    result = state.compute()
    self.assertAlmostEqual(result["sudoku_count"], 1.0)


class MatchesHardnessTest(parameterized.TestCase):
  """Tests for ``matches_hardness``."""

  @parameterized.parameters(
      (45, sudoku_eval.HardnessCategory.ALL, True),
      (35, sudoku_eval.HardnessCategory.ALL, True),
      (25, sudoku_eval.HardnessCategory.ALL, True),
      (40, sudoku_eval.HardnessCategory.EASY, True),
      (45, sudoku_eval.HardnessCategory.EASY, True),
      (39, sudoku_eval.HardnessCategory.EASY, False),
      (35, sudoku_eval.HardnessCategory.MEDIUM, True),
      (30, sudoku_eval.HardnessCategory.MEDIUM, True),
      (29, sudoku_eval.HardnessCategory.MEDIUM, False),
      (40, sudoku_eval.HardnessCategory.MEDIUM, False),
      (29, sudoku_eval.HardnessCategory.HARD, True),
      (25, sudoku_eval.HardnessCategory.HARD, True),
      (30, sudoku_eval.HardnessCategory.HARD, False),
  )
  def test_matches_hardness(self, num_unmasked, category, expected):
    self.assertEqual(
        sudoku_eval.matches_hardness(num_unmasked, category),
        expected,
    )

  def test_invalid_category(self):
    with self.assertRaises(ValueError):
      sudoku_eval.matches_hardness(35, "invalid_category")


if __name__ == "__main__":
  absltest.main()
