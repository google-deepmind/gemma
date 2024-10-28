# Copyright 2024 DeepMind Technologies Limited.
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

"""Tests for transformer modules."""

import logging

from absl.testing import absltest
from absl.testing import parameterized
from gemma import modules
import jax
import jax.numpy as jnp
import numpy as np


_ATTN_TYPE = modules.AttentionType.GLOBAL


class EmbedderTest(absltest.TestCase):

  def test_encodes(self):
    vocab_size = 10
    embed_dim = 4
    embedder = modules.Embedder(vocab_size=vocab_size, embed_dim=embed_dim)
    output = embedder.apply(
        {'params': {'input_embedding': jnp.ones((vocab_size, embed_dim))}},
        [2, 3],
        method=modules.Embedder.encode,
    )
    expected = [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]
    np.testing.assert_array_equal(output, jnp.array(expected))

  def test_decodes(self):
    vocab_size = 5
    embed_dim = 2
    embedder = modules.Embedder(vocab_size=vocab_size, embed_dim=embed_dim)
    output = embedder.apply(
        {'params': {'input_embedding': jnp.ones((vocab_size, embed_dim))}},
        jnp.array([1, 2]),
        method=modules.Embedder.decode,
    )
    expected = [3.0, 3.0, 3.0, 3.0, 3.0]
    np.testing.assert_array_equal(output, jnp.array(expected))


class SlidingWindowTest(absltest.TestCase):

  def test_create_sliding_mask_decode_none_rotated_cache_pos(self):
    cache_len = 4
    end_index = 1
    segment_pos = jnp.array([[1]])

    sliding_mask = modules._create_sliding_mask(
        segment_pos, end_index, cache_len, sliding_window_size=1
    )
    np.testing.assert_array_equal(
        sliding_mask,
        [[[False, True, False, False]]],
    )

    sliding_mask = modules._create_sliding_mask(
        segment_pos, end_index, cache_len, sliding_window_size=2
    )
    np.testing.assert_array_equal(
        sliding_mask,
        [[[True, True, True, False]]],
    )

    sliding_mask = modules._create_sliding_mask(
        segment_pos, end_index, cache_len, sliding_window_size=3
    )
    np.testing.assert_array_equal(
        sliding_mask,
        [[[True, True, True, True]]],
    )

  def test_create_sliding_mask_decode_rotated_cache_pos(self):
    cache_len = 4
    end_index = 5
    segment_pos = jnp.array([[5]])

    sliding_mask = modules._create_sliding_mask(
        segment_pos, end_index, cache_len, sliding_window_size=1
    )
    np.testing.assert_array_equal(
        sliding_mask,
        # cache_positions = [
        #   4,      5,     2,     3,
        # ]
        [[[False, True, False, False]]],
    )

    sliding_mask = modules._create_sliding_mask(
        segment_pos, end_index, cache_len, sliding_window_size=2
    )
    np.testing.assert_array_equal(
        sliding_mask,
        [[[True, True, False, False]]],
    )

    sliding_mask = modules._create_sliding_mask(
        segment_pos, end_index, cache_len, sliding_window_size=3
    )
    np.testing.assert_array_equal(
        sliding_mask,
        [[[True, True, False, True]]],
    )

  def test_create_sliding_mask_prefill_rotated_cache_pos(self):
    cache_len = 4
    end_index = 5
    segment_pos = jnp.array([[5, 6]])

    sliding_mask = modules._create_sliding_mask(
        segment_pos, end_index, cache_len, sliding_window_size=1
    )
    np.testing.assert_array_equal(
        sliding_mask,
        # cache_positions = [
        #   4,      5,     6,     3,
        # ]
        [[[False, True, False, False],
          [False, False, True, False],]],
    )

    sliding_mask = modules._create_sliding_mask(
        segment_pos, end_index, cache_len, sliding_window_size=2
    )
    np.testing.assert_array_equal(
        sliding_mask,
        [[[True, True, True, False],
          [False, True, True, False],]],
    )

    sliding_mask = modules._create_sliding_mask(
        segment_pos, end_index, cache_len, sliding_window_size=3
    )
    np.testing.assert_array_equal(
        sliding_mask,
        [[[True, True, True, True],
          [True, True, True, False],]],
    )


class AttentionTest(absltest.TestCase):

  def _get_attn_output(
      self,
      num_heads: int,
      head_dim: int,
      features: int,
      query_pre_attn_scalar: float | None = None,
      num_kv_heads: int | None = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    segment_pos = 0
    cache_size = 3
    batch_size = 2
    attn_mask = jnp.ones((batch_size, 1, cache_size))
    if query_pre_attn_scalar is None:
      query_pre_attn_scalar = head_dim**-0.5
    if num_kv_heads is None:
      # num_kv_heads is only different from num_heads in the GQA case.
      num_kv_heads = num_heads
    attn = modules.Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        features=features,
        head_dim=head_dim,
        attn_type=_ATTN_TYPE,
        query_pre_attn_scalar=query_pre_attn_scalar,
    )
    cache = modules.Attention.init_cache(
        cache_size=cache_size,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        dtype=jnp.float32,
    )
    x = jnp.ones((batch_size, 1, features))
    params = attn.init(
        jax.random.PRNGKey(0),
        x,
        jnp.array([[segment_pos]]),
        cache,
        attn_mask,
    )
    cache, output = attn.apply(
        params, x, jnp.array([[segment_pos]]), cache, attn_mask
    )
    return cache, output

  def test_attention(self):
    cache, output = self._get_attn_output(
        num_heads=2,
        head_dim=4,
        features=8,
    )
    expected_cache_shape = (2, 3, 2, 4)
    expected_output_shape = (2, 1, 8)
    self.assertEqual(cache['k'].shape, expected_cache_shape)
    self.assertEqual(output.shape, expected_output_shape)

  def test_attention_with_gqa(self):
    cache, output = self._get_attn_output(
        num_heads=8,
        head_dim=2,
        features=10,
        num_kv_heads=4,
    )
    expected_cache_shape = (2, 3, 4, 2)
    expected_output_shape = (2, 1, 10)
    self.assertEqual(cache['k'].shape, expected_cache_shape)
    self.assertEqual(output.shape, expected_output_shape)

  def test_sliding_window(self):
    num_heads = 2
    head_dim = 4
    features = 8
    segment_pos = 0
    cache_size = 3
    batch_size = 2
    query_pre_attn_scalar = head_dim**-0.5
    attn_mask = jnp.ones((batch_size, 1, cache_size))
    cache = modules.Attention.init_cache(
        cache_size=cache_size,
        num_heads=num_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        dtype=jnp.float32,
    )
    x = jnp.ones((batch_size, 1, features))
    attn = modules.Attention(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        features=features,
        head_dim=head_dim,
        attn_type=_ATTN_TYPE,
        query_pre_attn_scalar=query_pre_attn_scalar,
    )
    params = attn.init(
        jax.random.PRNGKey(0),
        x,
        jnp.array([[segment_pos]]),
        cache,
        attn_mask,
    )
    _, output = attn.apply(
        params, x, jnp.array([[segment_pos]]), cache, attn_mask
    )
    sliding_attn = modules.Attention(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        features=features,
        head_dim=head_dim,
        attn_type=modules.AttentionType.LOCAL_SLIDING,
        sliding_window_size=2,
        query_pre_attn_scalar=query_pre_attn_scalar,
    )
    _, sliding_output = sliding_attn.apply(
        params, x, jnp.array([[segment_pos]]), cache, attn_mask
    )

    self.assertFalse((output == sliding_output).all())

  def test_query_pre_attn_scalar_modifies_output(self):
    num_heads = 2
    head_dim = 4
    features = 8
    query_pre_attn_scalar_by_embed_dim_div_num_heads: float = (
        features // num_heads
    )
    query_pre_attn_scalar_by_head_dim: float = head_dim**-0.5
    _, output_by_head_dim = self._get_attn_output(
        num_heads,
        head_dim,
        features,
        query_pre_attn_scalar=query_pre_attn_scalar_by_head_dim,
    )
    _, output_by_embed_dim_div_num_heads = self._get_attn_output(
        num_heads,
        head_dim,
        features,
        query_pre_attn_scalar=query_pre_attn_scalar_by_embed_dim_div_num_heads,
    )
    expected_output_by_head_dim = [
        [[
            1.1596170e-04,
            3.0531217e-05,
            4.5884139e-05,
            -3.3920849e-05,
            -5.5468496e-05,
            8.6856808e-06,
            -1.5840206e-04,
            1.0944265e-04,
        ]],
        [[
            1.1596170e-04,
            3.0531217e-05,
            4.5884139e-05,
            -3.3920849e-05,
            -5.5468496e-05,
            8.6856808e-06,
            -1.5840206e-04,
            1.0944265e-04,
        ]],
    ]
    np.testing.assert_array_almost_equal(
        output_by_head_dim, expected_output_by_head_dim
    )
    expected_output_by_embed_dim_div_num_heads = [
        [[
            1.15790164e-04,
            3.05866670e-05,
            4.57668611e-05,
            -3.40082588e-05,
            -5.54954640e-05,
            8.75260412e-06,
            -1.58223527e-04,
            1.09341796e-04,
        ]],
        [[
            1.15790164e-04,
            3.05866670e-05,
            4.57668611e-05,
            -3.40082588e-05,
            -5.54954640e-05,
            8.75260412e-06,
            -1.58223527e-04,
            1.09341796e-04,
        ]],
    ]
    np.testing.assert_array_almost_equal(
        output_by_embed_dim_div_num_heads,
        expected_output_by_embed_dim_div_num_heads,
    )


class FeedForwardTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          transpose_gating_einsum=False,
      ),
      dict(
          transpose_gating_einsum=True,
      ),
  )
  def test_ffw(self, transpose_gating_einsum: bool):
    features = 2
    hidden_dim = 3
    batch_size = 2
    inputs = jnp.arange(1, batch_size + 1)[:, None, None]
    inputs = jnp.repeat(inputs, features, axis=-1)
    ffw = modules.FeedForward(
        features=features,
        hidden_dim=hidden_dim,
        transpose_gating_einsum=transpose_gating_einsum,
    )

    params = {'linear': jnp.ones((hidden_dim, features))}

    # Different checkpoints have params saved in different order
    if transpose_gating_einsum:
      params['gating_einsum'] = jnp.ones((batch_size, hidden_dim, features))
    else:
      params['gating_einsum'] = jnp.ones((batch_size, features, hidden_dim))

    outputs = ffw.apply({'params': params}, inputs)

    expected_val = [11.72758674, 47.99916]
    expected_shape = (2, 1, 2)
    np.testing.assert_array_almost_equal(outputs[:, 0, 0], expected_val)
    self.assertEqual(outputs.shape, expected_shape)

  @parameterized.parameters(
      dict(
          transpose_gating_einsum=False,
          expected_grad=[-1.916515e-04, -5.391428e-05, -2.923766e-04],
      ),
      dict(
          transpose_gating_einsum=True,
          expected_grad=[1.574128e-05, -1.301362e-04, -1.037612e-04],
      ),
  )
  def test_ffw_grad(self, transpose_gating_einsum: bool,
                    expected_grad: list[float]):
    features = 2
    hidden_dim = 3
    batch_size = 2
    inputs = jnp.arange(1, batch_size + 1)[:, None, None]
    inputs = jnp.repeat(inputs, features, axis=-1)
    ffw = modules.FeedForward(
        features=features,
        hidden_dim=hidden_dim,
        transpose_gating_einsum=transpose_gating_einsum,
    )
    loss = lambda params, inputs: jnp.square(
        ffw.apply(params, inputs) - jnp.ones((batch_size, 1, features))
    ).mean()

    params = ffw.init(jax.random.PRNGKey(0), inputs)

    grad_loss = jax.grad(loss)
    grad = grad_loss(params, inputs)
    np.testing.assert_array_almost_equal(
        grad['params']['linear'][:, 0], expected_grad
    )


class BlockTest(absltest.TestCase):

  def test_block(self):
    num_heads = 2
    embed_dim = 4
    head_dim = 6
    cache_size = 3
    batch_size = 2
    inputs = jnp.ones((batch_size, 1, embed_dim))
    cache = modules.Attention.init_cache(
        cache_size=cache_size,
        num_heads=num_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        dtype=jnp.float32,
    )
    attn_mask = jnp.ones((batch_size, 1, cache_size))
    block = modules.Block(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        embed_dim=embed_dim,
        head_dim=head_dim,
        hidden_dim=1,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        attn_type=_ATTN_TYPE,
        query_pre_attn_scalar=head_dim**-0.5,
        transpose_gating_einsum=False,
    )
    params = block.init(
        jax.random.PRNGKey(0), inputs, jnp.array([[0]]), cache, attn_mask
    )

    new_cache, outputs = block.apply(
        params, inputs, jnp.array([[0]]), cache, attn_mask
    )

    expected_cache_shape = (2, 3, 2, 6)
    expected_output_shape = (2, 1, 4)
    self.assertEqual(new_cache['k'].shape, expected_cache_shape)
    self.assertEqual(outputs.shape, expected_output_shape)

  def test_post_attention_norm_modifies_output(self):
    num_heads = 1
    embed_dim = 1
    head_dim = 2
    hidden_dim = 1
    cache_size = 1
    batch_size = 1
    query_pre_attn_scalar = head_dim**-0.5
    inputs = jnp.ones((batch_size, 1, embed_dim))
    cache = modules.Attention.init_cache(
        cache_size=cache_size,
        num_heads=num_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        dtype=jnp.float32,
    )
    attn_mask = jnp.ones((batch_size, 1, cache_size))
    normed_block = modules.Block(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        embed_dim=embed_dim,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        use_post_attn_norm=True,
        use_post_ffw_norm=False,
        attn_type=_ATTN_TYPE,
        query_pre_attn_scalar=query_pre_attn_scalar,
        transpose_gating_einsum=False,
    )
    unnormed_block = modules.Block(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        embed_dim=embed_dim,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        attn_type=_ATTN_TYPE,
        query_pre_attn_scalar=query_pre_attn_scalar,
        transpose_gating_einsum=False,
    )

    all_outputs = []
    for block in (normed_block, unnormed_block):
      params = block.init(
          jax.random.PRNGKey(0), inputs, jnp.array([[0]]), cache, attn_mask
      )

      _, outputs = block.apply(
          params, inputs, jnp.array([[0]]), cache, attn_mask
      )
      all_outputs.append(outputs)

    normed_output, unnormed_output = all_outputs  # pylint: disable=unbalanced-tuple-unpacking
    logging.info('normed_output: %s', normed_output)
    logging.info('unnormed_output: %s', unnormed_output)

    # Normed and unnormed outputs should not be equal.
    with np.testing.assert_raises(AssertionError):
      np.testing.assert_array_almost_equal(normed_output, unnormed_output)

  def test_post_ffw_norm_modifies_output(self):
    num_heads = 1
    embed_dim = 1
    head_dim = 2
    hidden_dim = 1
    cache_size = 1
    batch_size = 1
    query_pre_attn_scalar = head_dim**-0.5
    inputs = jnp.ones((batch_size, 1, embed_dim))
    cache = modules.Attention.init_cache(
        cache_size=cache_size,
        num_heads=num_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        dtype=jnp.float32,
    )
    attn_mask = jnp.ones((batch_size, 1, cache_size))
    normed_block = modules.Block(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        embed_dim=embed_dim,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        use_post_attn_norm=False,
        use_post_ffw_norm=True,
        attn_type=_ATTN_TYPE,
        query_pre_attn_scalar=query_pre_attn_scalar,
        transpose_gating_einsum=False,
    )
    unnormed_block = modules.Block(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        embed_dim=embed_dim,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        attn_type=_ATTN_TYPE,
        query_pre_attn_scalar=query_pre_attn_scalar,
        transpose_gating_einsum=False,
    )

    all_outputs = []
    for block in (normed_block, unnormed_block):

      params = block.init(
          jax.random.PRNGKey(0), inputs, jnp.array([[0]]), cache, attn_mask
      )

      # Replace mlp block params with 1s as ffw will initialize with
      # 0s which will not properly test normalization.
      for param in ['gating_einsum', 'linear']:
        params['params']['mlp'][param] = jnp.ones_like(
            params['params']['mlp'][param]
        )

      _, outputs = block.apply(
          params, inputs, jnp.array([[0]]), cache, attn_mask
      )
      all_outputs.append(outputs)

    normed_output, unnormed_output = all_outputs  # pylint: disable=unbalanced-tuple-unpacking
    logging.info('normed_output: %s', normed_output)
    logging.info('unnormed_output: %s', unnormed_output)

    # Normed and unnormed outputs should not be equal.
    with np.testing.assert_raises(AssertionError):
      np.testing.assert_array_almost_equal(normed_output, unnormed_output)


if __name__ == '__main__':
  absltest.main()
