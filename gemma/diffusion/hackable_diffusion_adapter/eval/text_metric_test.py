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

"""Tests for text evaluation metrics."""

from unittest import mock

from absl.testing import absltest
from gemma.diffusion.hackable_diffusion_adapter.eval import text_metric
import jax.numpy as jnp


class DetokenizePromptAndResponseTest(absltest.TestCase):

  @mock.patch("seqio.vocabularies.SentencePieceVocabulary")
  def test_detokenize_prompt_and_response(self, mock_vocab):
    mock_instance = mock_vocab.return_value
    mock_instance.decode.side_effect = lambda t: f"d_{t}"

    metric = text_metric.DetokenizePromptAndResponse(
        prompt="batch.prompt",
        response="batch.response",
        tokenizer_path="fake_path",
        num_texts=2,
        separator=" | ",
    )
    prompt = jnp.array([[[1]], [[2]], [[3]]])
    response = jnp.array([[[4]], [[5]], [[6]]])
    state = metric.get_state(prompt=prompt, response=response)
    state = state.finalize()

    self.assertEqual(state.prompt.shape, (2, 1))
    self.assertEqual(state.response.shape, (2, 1))

    res = state.compute()
    self.assertEqual(res, ["d_[1] | d_[4]", "d_[2] | d_[5]"])

  @mock.patch("seqio.vocabularies.SentencePieceVocabulary")
  def test_default_tokenizer_path(self, mock_vocab):
    """Default constructor should work without explicit tokenizer_path."""
    mock_instance = mock_vocab.return_value
    mock_instance.decode.side_effect = lambda t: f"d_{t}"

    metric = text_metric.DetokenizePromptAndResponse(
        prompt="batch.prompt",
        response="batch.response",
        num_texts=3,
        separator=" ## ",
    )
    # Verify default tokenizer path is used.
    self.assertIn("gemma4", metric.tokenizer_path)
    self.assertEqual(metric.num_texts, 3)
    self.assertEqual(metric.separator, " ## ")


if __name__ == "__main__":
  absltest.main()
