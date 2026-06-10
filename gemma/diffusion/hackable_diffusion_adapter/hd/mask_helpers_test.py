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

from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest


class BuildPositionsFromMaskTest(absltest.TestCase):
  """Tests for build_positions_from_mask."""

  def test_no_padding(self):
    """All-real tokens produce a simple arange."""
    mask = jnp.array([[True, True, True, True]])
    positions = mask_helpers.build_positions_from_mask(mask)
    np.testing.assert_array_equal(positions, [[0, 1, 2, 3]])

  def test_right_padding(self):
    """Pad tokens repeat the last real position."""
    mask = jnp.array([[True, True, True, False, False]])
    positions = mask_helpers.build_positions_from_mask(mask)
    np.testing.assert_array_equal(positions, [[0, 1, 2, 2, 2]])

  def test_all_padding(self):
    """All-pad input produces all zeros."""
    mask = jnp.array([[False, False, False]])
    positions = mask_helpers.build_positions_from_mask(mask)
    np.testing.assert_array_equal(positions, [[0, 0, 0]])

  def test_single_token(self):
    mask = jnp.array([[True]])
    positions = mask_helpers.build_positions_from_mask(mask)
    np.testing.assert_array_equal(positions, [[0]])

  def test_batched(self):
    """Different padding lengths within a batch."""
    mask = jnp.array([
        [True, True, True, True, False],
        [True, True, False, False, False],
    ])
    positions = mask_helpers.build_positions_from_mask(mask)
    np.testing.assert_array_equal(
        positions,
        [
            [0, 1, 2, 3, 3],
            [0, 1, 1, 1, 1],
        ],
    )


class MakeCausalPrefillMaskTest(absltest.TestCase):
  """Tests for make_causal_prefill_mask."""

  def test_no_padding_no_cache_pad(self):
    """cache_length == seq_len, no right-padding on either axis."""
    mask = jnp.array([[True, True, True]])
    result = mask_helpers.make_causal_prefill_mask(mask, cache_length=3)
    expected = jnp.array(
        [[[True, False, False], [True, True, False], [True, True, True]]]
    )
    np.testing.assert_array_equal(result, expected)

  def test_right_padded_tokens(self):
    """Pad tokens in the mask should zero out their columns."""
    mask = jnp.array([[True, True, False]])
    result = mask_helpers.make_causal_prefill_mask(mask, cache_length=3)
    expected = jnp.array(
        [[[True, False, False], [True, True, False], [True, True, False]]]
    )
    np.testing.assert_array_equal(result, expected)

  def test_cache_padding(self):
    """cache_length > seq_len adds False columns on the right."""
    mask = jnp.array([[True, True]])
    result = mask_helpers.make_causal_prefill_mask(mask, cache_length=5)
    self.assertEqual(result.shape, (1, 2, 5))
    expected = jnp.array([[
        [True, False, False, False, False],
        [True, True, False, False, False],
    ]])
    np.testing.assert_array_equal(result, expected)

  def test_combined_token_and_cache_padding(self):
    """Both token-level pad and cache_length > seq_len."""
    mask = jnp.array([[True, False]])
    result = mask_helpers.make_causal_prefill_mask(mask, cache_length=4)
    self.assertEqual(result.shape, (1, 2, 4))
    expected = jnp.array(
        [[[True, False, False, False], [True, False, False, False]]]
    )
    np.testing.assert_array_equal(result, expected)

  def test_cache_length_equals_seq_len(self):
    """No extra padding when cache_length == seq_len."""
    mask = jnp.array([[True, True, True]])
    result = mask_helpers.make_causal_prefill_mask(mask, cache_length=3)
    self.assertEqual(result.shape, (1, 3, 3))

  def test_batched(self):
    """Different padding within a batch."""
    mask = jnp.array([
        [True, True, True],
        [True, False, False],
    ])
    result = mask_helpers.make_causal_prefill_mask(mask, cache_length=5)
    self.assertEqual(result.shape, (2, 3, 5))

    np.testing.assert_array_equal(
        result[0],
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
        ],
    )

    np.testing.assert_array_equal(
        result[1],
        [
            [True, False, False, False, False],
            [True, False, False, False, False],
            [True, False, False, False, False],
        ],
    )

  def test_output_dtype_is_bool(self):
    mask = jnp.array([[True, True]])
    result = mask_helpers.make_causal_prefill_mask(mask, cache_length=3)
    self.assertEqual(result.dtype, jnp.bool_)


class SetCacheEndIndexTest(absltest.TestCase):
  """Tests for set_cache_end_index."""

  def _make_fake_cache(self, batch_size, cache_length, num_layers=2):
    cache = {}
    for i in range(num_layers):
      cache[f'layer_{i}'] = {
          'k': jnp.zeros((batch_size, cache_length, 2, 4)),
          'v': jnp.zeros((batch_size, cache_length, 2, 4)),
          'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
          'positions': jnp.zeros((batch_size, cache_length), dtype=jnp.int32),
      }
    return cache

  def test_sets_end_index_all_layers(self):
    cache = self._make_fake_cache(batch_size=2, cache_length=8)
    end_index = jnp.array([3, 5], dtype=jnp.int32)
    new_cache = mask_helpers.set_cache_end_index(cache, end_index)
    for layer_name in ['layer_0', 'layer_1']:
      np.testing.assert_array_equal(new_cache[layer_name]['end_index'], [3, 5])

  def test_preserves_other_keys(self):
    cache = self._make_fake_cache(batch_size=1, cache_length=4)
    cache['layer_0']['k'] = jnp.ones_like(cache['layer_0']['k'])
    end_index = jnp.array([2], dtype=jnp.int32)
    new_cache = mask_helpers.set_cache_end_index(cache, end_index)
    np.testing.assert_array_equal(
        new_cache['layer_0']['k'], jnp.ones_like(cache['layer_0']['k'])
    )

  def test_scalar_broadcast(self):
    cache = self._make_fake_cache(batch_size=3, cache_length=8)
    end_index = jnp.array(7, dtype=jnp.int32)
    new_cache = mask_helpers.set_cache_end_index(cache, end_index)
    for layer_name in cache:
      np.testing.assert_array_equal(
          new_cache[layer_name]['end_index'], [7, 7, 7]
      )

  def test_does_not_mutate_original(self):
    cache = self._make_fake_cache(batch_size=1, cache_length=4)
    end_index = jnp.array([5], dtype=jnp.int32)
    _ = mask_helpers.set_cache_end_index(cache, end_index)
    np.testing.assert_array_equal(cache['layer_0']['end_index'], [0])


class MakeFullAttentionMaskTest(absltest.TestCase):
  """Tests for make_full_attention_mask."""

  def test_basic(self):
    input_mask = jnp.array([[True, True, False]])
    result = mask_helpers.make_full_attention_mask(input_mask, cache_length=5)
    expected = jnp.array([[True, True, False, True, True]])
    np.testing.assert_array_equal(result, expected)

  def test_no_padding(self):
    input_mask = jnp.array([[True, True, True]])
    result = mask_helpers.make_full_attention_mask(input_mask, cache_length=5)
    expected = jnp.array([[True, True, True, True, True]])
    np.testing.assert_array_equal(result, expected)


class CreateDecoderAttentionMaskTest(absltest.TestCase):
  """Tests for create_decoder_attention_mask."""

  def test_shape(self):
    batch_size = 2
    prompt_len = 3
    total_canvas_len = 4
    canvas_size = 2
    num_queries = 4
    prompt_mask = jnp.ones((batch_size, prompt_len), dtype=jnp.bool_)
    canvas_mask = jnp.ones((batch_size, total_canvas_len), dtype=jnp.bool_)
    selected_canvas_idx = jnp.array([0, 1])
    result = mask_helpers.create_decoder_attention_mask(
        prompt_mask=prompt_mask,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        prompt_len=prompt_len,
        total_canvas_len=total_canvas_len,
        canvas_size=canvas_size,
        num_queries=num_queries,
    )
    self.assertEqual(
        result.shape, (batch_size, num_queries, prompt_len + total_canvas_len)
    )


class MakeCausalAttentionMaskRightPadTest(absltest.TestCase):
  """Tests for make_causal_attention_mask_right_pad."""

  def test_no_cache(self):
    """Without cache, returns a simple lower-triangular mask."""
    result = mask_helpers.make_causal_attention_mask_right_pad(
        batch_size=1,
        canvas_length=3,
        cache_length=None,
        num_valid_cache_tokens=None,
    )
    expected = jnp.array([
        [[True, False, False], [True, True, False], [True, True, True]],
    ])
    np.testing.assert_array_equal(result, expected)

  def test_with_cache(self):
    """With cache, mask reveals valid entries + causal triangle at write pos."""
    result = mask_helpers.make_causal_attention_mask_right_pad(
        batch_size=1,
        canvas_length=2,
        cache_length=5,
        num_valid_cache_tokens=jnp.array([3]),
    )
    self.assertEqual(result.shape, (1, 2, 5))


if __name__ == '__main__':
  absltest.main()
