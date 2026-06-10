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

"""Unit tests for base evaluation metrics."""

import dataclasses

from absl.testing import absltest
from gemma.diffusion.hackable_diffusion_adapter.eval import base_metric
import jax.numpy as jnp


class MockTokenizer:

  def decode(self, tokens: list[int]) -> str:
    # Decode by converting positive tokens to strings, skipping pad/eos (-1)
    return " ".join(str(t) for t in tokens if t >= 0)


@dataclasses.dataclass(kw_only=True, frozen=True)
class DummySimpleMetric(base_metric.BaseSimpleTextMetric):

  # Override tokenizer to avoid loading heavy SentencePiece model in test.
  @property
  def tokenizer(self) -> MockTokenizer:
    return MockTokenizer()

  def score_example(self, generated_text: str, ground_truth_text: str) -> float:
    # Score is 1.0 if generated matches ground truth, else 0.0
    return 1.0 if generated_text == ground_truth_text else 0.0


class BaseMetricTest(absltest.TestCase):

  def test_simple_text_metric_2d(self):
    """Test BaseSimpleTextMetric with 2D token inputs."""
    metric = DummySimpleMetric(tokens="samples", ground_truth="batch.response")

    # Batches of shape [B, L]
    tokens = jnp.array([
        [1, 2, 3, -1],  # "1 2 3"
        [4, 5, -1, -1],  # "4 5"
    ])
    ground_truth = jnp.array([
        [1, 2, 3, -1],  # "1 2 3" (Match)
        [4, 6, -1, -1],  # "4 6"   (Mismatch)
    ])

    state = metric.get_state(tokens=tokens, ground_truth=ground_truth)
    computed = state.compute()

    # Match gives 1.0, mismatch gives 0.0 -> average is 0.5
    self.assertEqual(computed, 0.5)

  def test_simple_text_metric_3d_squeezing(self):
    """Test that BaseSimpleTextMetric squeezes 3D [B, L, 1] inputs to 2D."""
    metric = DummySimpleMetric(tokens="samples", ground_truth="batch.response")

    # Batches of shape [B, L, 1]
    tokens = jnp.array([
        [[1], [2], [-1]],  # "1 2"
        [[3], [4], [-1]],  # "3 4"
    ])
    ground_truth = jnp.array([
        [[1], [2], [-1]],  # "1 2" (Match)
        [[3], [4], [-1]],  # "3 4" (Match)
    ])

    state = metric.get_state(tokens=tokens, ground_truth=ground_truth)
    computed = state.compute()

    # Both match -> average is 1.0
    self.assertEqual(computed, 1.0)


if __name__ == "__main__":
  absltest.main()
