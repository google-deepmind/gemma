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

from typing import Any

from gemma.gm.nn.gemma4 import _gemma4 as gemma4_models
from gemma.gm.nn.gemma4 import _transformer as gt
from gemma.gm.vision import _token_utils
import jax
import jax.numpy as jnp
import pytest


BATCH_SIZE = 4
SEQ_LEN = 16


def _get_output(
    model: gt.Transformer, **kwargs
) -> tuple[gt.Output, Any]:

  def init_fn(**kwargs):
    out, params = model.init_with_output(jax.random.key(0), **kwargs)
    return out, params['params']

  return jax.eval_shape(init_fn, **kwargs)


@pytest.mark.parametrize(
    'model_cls',
    [
        gemma4_models.Gemma4_E2B,
        gemma4_models.Gemma4_E4B,
        gemma4_models.Gemma4_31B,
    ],
)
def test_transformer(model_cls: type[gt.Transformer]):
  model = model_cls()  # pylint: disable=missing-kwoa  # pytype: disable=missing-parameter
  tokens = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
  out, _ = _get_output(model, tokens=tokens)
  assert out.logits.shape == (BATCH_SIZE, SEQ_LEN, model.config.num_embed)


def test_text_only():
  model = gemma4_models.Gemma4_31B(text_only=True)  # pylint: disable=missing-kwoa  # pytype: disable=missing-parameter
  tokens = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
  out, params = _get_output(model, tokens=tokens)
  assert 'vision_encoder' not in params
  assert out.logits.shape == (BATCH_SIZE, SEQ_LEN, model.config.num_embed)


def test_compute_valid_audio_soft_token_counts():
  # Test with dummy audio lengths
  # audio_lengths has shape [B, num_clips]
  # For 16000 Hz:
  # 2.0s = 32000 samples.
  # mel frames = (32000 - 321) // 160 + 1 = 31679 // 160 + 1 = 197 + 1 = 198
  # mel frames.
  # Downsampling 1:
  # t = (198 + 2 - 3) // 2 + 1 = 197 // 2 + 1 = 98 + 1 = 99.
  # Downsampling 2:
  # t = (99 + 2 - 3) // 2 + 1 = 98 // 2 + 1 = 49 + 1 = 50.
  # So for 32000 samples, it should return 50 soft tokens.

  # For 7.5s = 120000 samples.
  # mel frames = (120000 - 321) // 160 + 1 = 119679 // 160 + 1 = 747 + 1 =
  # 748 mel frames.
  # Downsampling 1: (748 + 2 - 3) // 2 + 1 = 373 + 1 = 374.
  # Downsampling 2: (374 + 2 - 3) // 2 + 1 = 186 + 1 = 187.
  # So 187 soft tokens.

  audio_lengths = jnp.array([[32000, 120000]], dtype=jnp.int32)
  counts = gt.compute_valid_audio_soft_token_counts(audio_lengths, 750)
  assert jnp.array_equal(counts, jnp.array([[50, 187]], dtype=jnp.int32))


def test_mask_padded_audio_tokens():
  # Placeholder token ID is usually _token_utils.AUDIO_SOFT_TOKEN_PLACEHOLDER
  placeholder = _token_utils.AUDIO_SOFT_TOKEN_PLACEHOLDER

  # tokens has shape [B, L]
  # B=1, L=13
  # We have 2 audio clips, max 4 tokens each (so self.audio_seq_length=4)
  # Clip 0 has valid count 2 (e.g. audio_lengths corresponds to 2 soft tokens)
  # Clip 1 has valid count 3

  tokens = jnp.array(
      [[
          1,
          2,
          placeholder,
          placeholder,
          placeholder,
          placeholder,
          3,
          4,
          placeholder,
          placeholder,
          placeholder,
          placeholder,
          5,
      ]],
      dtype=jnp.int32,
  )

  # inputs_mask initially all True except PAD (none here)
  inputs_mask = jnp.ones((1, 13), dtype=jnp.bool_)

  # audio_lengths corresponding to [2, 3] soft tokens.
  audio_lengths = jnp.array([[961, 1601]], dtype=jnp.int32)

  valid_counts = gt.compute_valid_audio_soft_token_counts(
      audio_lengths, audio_seq_length=4
  )
  new_inputs_mask = gt.mask_padded_audio_tokens(
      tokens, inputs_mask, valid_counts, audio_seq_length=4
  )

  # Expected mask:
  # Clip 0 (indices 2..5): first 2 True, last 2 False -> [T, T, F, F]
  # Clip 1 (indices 8..11): first 3 True, last 1 False -> [T, T, T, F]
  # Others: True
  # Expected: [T, T,  T, T, F, F,  T, T,  T, T, T, F,  T]
  expected_mask = jnp.array(
      [[
          True,
          True,
          True,
          True,
          False,
          False,
          True,
          True,
          True,
          True,
          True,
          False,
          True,
      ]],
      dtype=jnp.bool_,
  )

  assert jnp.array_equal(new_inputs_mask, expected_mask)
