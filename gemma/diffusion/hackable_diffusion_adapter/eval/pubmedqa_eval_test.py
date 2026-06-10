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

"""Unit tests for PubMedQA evaluation utilities."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from gemma.diffusion.hackable_diffusion_adapter.eval import pubmedqa_eval
import jax.numpy as jnp


class ExtractPubMedQAAnswerTest(parameterized.TestCase):

  @parameterized.parameters(
      # Explicit "The answer is:" marker variants.
      ("The answer is: yes", "yes"),
      ("The answer is: no", "no"),
      ("The answer is: maybe", "maybe"),
      ("The answer is:yes", "yes"),  # No space after colon.
      ("The answer is yes", "yes"),  # Without colon.
      (
          "Based on the evidence, the study shows improvement."
          + " The answer is: yes",
          "yes",
      ),
      (
          "The results are inconclusive. The answer is: maybe",
          "maybe",
      ),
  )
  def test_extraction(self, text, expected):
    self.assertEqual(pubmedqa_eval.extract_pubmedqa_answer(text), expected)

  @parameterized.parameters(
      "",
      "The study was inconclusive.",
      "42",
      # No fallback heuristics — bare answers without the marker return None.
      "yes",
      "Yes",
      "no",
      "maybe",
      "No, the study does not support this.",
      "Maybe, further studies are needed.",
      # Previously these would incorrectly match via heuristics.
      "Based on the results, no significant effect was found.",
      "The study showed novel approaches to treatment.",
      "Yesterday was a good day.",  # "yes" substring in "yesterday".
  )
  def test_extraction_returns_none(self, text):
    self.assertIsNone(pubmedqa_eval.extract_pubmedqa_answer(text))


class ScorePubMedQATest(parameterized.TestCase):

  @parameterized.parameters(
      ("The answer is: yes", "yes", True),
      ("The answer is: no", "no", True),
      ("The answer is: maybe", "maybe", True),
      ("The answer is: yes", "no", False),
      ("The answer is: no", "maybe", False),
      # Without marker — no match even if the answer word appears.
      ("yes", "yes", False),
      ("no", "no", False),
      ("The study is inconclusive.", "yes", False),
      ("Based on the results, no significant effect was found.", "no", False),
  )
  def test_scoring(self, generated, ground_truth, expected):
    self.assertEqual(
        pubmedqa_eval.score_pubmedqa(generated, ground_truth), expected
    )


class MockTokenizer:

  def decode(self, tokens: list[int]) -> str:
    # 1 maps to "The answer is: yes", 2 to "yes",
    # 3 to "The answer is: no", 4 to "no"
    mapping = {
        1: "The answer is: yes",
        2: "yes",
        3: "The answer is: no",
        4: "no",
    }
    return " ".join(mapping.get(t, "") for t in tokens if t > 0).strip()


class PubMedQAAccuracyTest(parameterized.TestCase):

  def test_pubmedqa_accuracy(self):

    @dataclasses.dataclass(kw_only=True, frozen=True)
    class TestPubMedQAAccuracy(pubmedqa_eval.PubMedQAAccuracy):

      @property
      def tokenizer(self) -> MockTokenizer:
        return MockTokenizer()

    metric = TestPubMedQAAccuracy(
        tokens="samples",
        ground_truth="batch.short_answer_tokens",
    )

    tokens = jnp.array([
        [1, -1],  # "The answer is: yes"
        [3, -1],  # "The answer is: no"
    ])
    ground_truth = jnp.array([
        [2, -1],  # "yes"
        [2, -1],  # "yes"
    ])

    state = metric.get_state(tokens=tokens, ground_truth=ground_truth)
    accuracy = state.compute()

    self.assertEqual(accuracy, 0.5)


class ScoreBleuTest(parameterized.TestCase):
  """Tests for score_bleu."""

  def test_perfect_match(self):
    """Identical strings should yield 100."""
    score = pubmedqa_eval.score_bleu("Bonjour le monde", "Bonjour le monde")
    self.assertAlmostEqual(score, 100.0, places=1)

  def test_wrong_text(self):
    """Completely wrong text should yield low BLEU."""
    score = pubmedqa_eval.score_bleu("This is wrong", "Bonjour le monde")
    self.assertLess(score, 10.0)

  def test_empty_generated(self):
    score = pubmedqa_eval.score_bleu("", "Bonjour")
    self.assertEqual(score, 0.0)

  def test_empty_ground_truth(self):
    score = pubmedqa_eval.score_bleu("Bonjour", "")
    self.assertEqual(score, 0.0)

  def test_both_empty(self):
    score = pubmedqa_eval.score_bleu("", "")
    self.assertEqual(score, 0.0)

  def test_partial_match(self):
    gt = "Le chat dort sur le tapis rouge dans la grande maison blanche"
    partial = "Le chat mange sur le tapis rouge dans la petite maison blanche"
    score = pubmedqa_eval.score_bleu(partial, gt)
    self.assertGreater(score, 0.0)
    self.assertLess(score, 100.0)

  def test_whitespace_handling(self):
    score = pubmedqa_eval.score_bleu("  Bonjour  ", "  Bonjour  ")
    self.assertAlmostEqual(score, 100.0, places=1)


class BLEUScoreMetricTest(absltest.TestCase):

  def test_tokenizer_path_default(self):
    metric = pubmedqa_eval.BLEUScore(
        tokens="samples",
        ground_truth="batch.response_tokens",
    )
    self.assertIn("gemma4", metric.tokenizer_path)

  def test_bleu_round_trip(self):

    class MockBLEUTokenizer:

      def decode(self, tokens: list[int]) -> str:
        mapping = {1: "Bonjour le monde", 2: "Bonjour le monde"}
        return " ".join(mapping.get(t, "") for t in tokens if t > 0).strip()

    @dataclasses.dataclass(kw_only=True, frozen=True)
    class TestBLEUScore(pubmedqa_eval.BLEUScore):

      @property
      def tokenizer(self) -> MockBLEUTokenizer:
        return MockBLEUTokenizer()

    metric = TestBLEUScore(
        tokens="samples",
        ground_truth="batch.response_tokens",
    )
    tokens = jnp.array([[1, -1]])
    ground_truth = jnp.array([[2, -1]])

    state = metric.get_state(tokens=tokens, ground_truth=ground_truth)
    score = state.compute()

    self.assertAlmostEqual(score, 100.0, places=1)


if __name__ == "__main__":
  absltest.main()
