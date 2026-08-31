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

"""Tests for Gemma 4 transformer modules."""

from gemma.gm.nn.gemma4 import _modules
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _capture_attention_probs(
    attn_type: _modules.AttentionType,
    *,
    disable_sliding_window: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  seq_len = 4
  sliding_window_size = 2

  attention = _modules.Attention(
      num_heads=1,
      num_kv_heads=1,
      features=4,
      key_size=2,
      attn_type=attn_type,
      sliding_window_size=sliding_window_size,
      qk_norm_with_scale=False,
      rope_proportion=1.0,
  )

  x = jnp.arange(
      1,
      seq_len * 4 + 1,
      dtype=jnp.float32,
  ).reshape(1, seq_len, 4)

  positions = jnp.arange(
      seq_len,
      dtype=jnp.int32,
  )[None, :]

  attention_mask = jnp.ones(
      (1, seq_len, seq_len),
      dtype=jnp.bool_,
  )

  # Each row includes a token outside the local window so that applying
  # the sliding-window mask produces a result different from this base mask.
  sliding_attention_mask = jnp.array(
      [[
          [1, 0, 0, 1],
          [0, 1, 0, 1],
          [1, 0, 1, 0],
          [1, 0, 0, 1],
      ]],
      dtype=jnp.bool_,
  )

  variables = attention.init(
      jax.random.key(0),
      x,
      positions,
      None,
      attention_mask,
      sliding_attention_mask=sliding_attention_mask,
      disable_sliding_window=disable_sliding_window,
  )

  (_, _), state = attention.apply(
      variables,
      x,
      positions,
      None,
      attention_mask,
      sliding_attention_mask=sliding_attention_mask,
      disable_sliding_window=disable_sliding_window,
      capture_intermediates=True,
      mutable=['intermediates'],
  )

  probs = state['intermediates']['attention_weights']['__call__'][0]

  if attn_type == _modules.AttentionType.LOCAL_SLIDING:
    expected_mask = sliding_attention_mask

    if not disable_sliding_window:
      window_mask = _modules._create_sliding_mask(
          positions,
          sliding_window_size=sliding_window_size,
      )
      expected_mask = sliding_attention_mask & window_mask
  else:
    expected_mask = attention_mask

  return np.asarray(probs), np.asarray(expected_mask)


@pytest.mark.parametrize(
    'attn_type',
    [
        _modules.AttentionType.GLOBAL,
        _modules.AttentionType.LOCAL_SLIDING,
    ],
)
def test_attention_owns_mask_selection(
    attn_type: _modules.AttentionType,
):
  probs, expected_mask = _capture_attention_probs(attn_type)

  # Masked logits receive K_MASK and therefore have zero probability.
  observed_mask = probs[:, :, 0, :] > 0.0

  np.testing.assert_array_equal(
      observed_mask,
      expected_mask,
  )


def test_attention_can_disable_sliding_window():
  _, normal_mask = _capture_attention_probs(
      _modules.AttentionType.LOCAL_SLIDING,
  )

  probs, disabled_mask = _capture_attention_probs(
      _modules.AttentionType.LOCAL_SLIDING,
      disable_sliding_window=True,
  )

  observed_mask = probs[:, :, 0, :] > 0.0

  # The fixture must distinguish S from S & W, otherwise this test could pass
  # without proving that the window was actually disabled.
  assert not np.array_equal(normal_mask, disabled_mask)

  np.testing.assert_array_equal(
      observed_mask,
      disabled_mask,
  )
