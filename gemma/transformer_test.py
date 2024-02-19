# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for the Gemma transformer."""

from absl.testing import absltest
from absl.testing import parameterized
from gemma import transformer as transformer_lib
import jax
import jax.numpy as jnp
import numpy as np


class TransformerTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          num_layers=3,
          num_embed=4,
          embed_dim=2,
          num_heads=2,
          num_kv_heads=2,
          hidden_dim=4,
          head_dim=4,
          cache_size=2,
          batch_size=1,
          expected_outputs_shape=(1, 1, 4),
          expected_cache_shape=(1, 2, 2, 4),
      ),
      dict(
          num_layers=3,
          num_embed=4,
          embed_dim=2,
          num_heads=2,
          num_kv_heads=1,
          hidden_dim=4,
          head_dim=4,
          cache_size=2,
          batch_size=1,
          expected_outputs_shape=(1, 1, 4),
          expected_cache_shape=(1, 2, 2, 4),
      ),
  )
  def test_transformer(
      self,
      num_layers,
      num_embed,
      embed_dim,
      num_heads,
      num_kv_heads,
      hidden_dim,
      head_dim,
      cache_size,
      batch_size,
      expected_outputs_shape,
      expected_cache_shape,
  ):

    config = transformer_lib.TransformerConfig(
        num_layers=num_layers,
        num_embed=num_embed,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        max_cache_length=cache_size,
    )
    cache = transformer_lib.init_cache(
        config, batch_size, dtype=jnp.float32
    )
    attention_mask = jnp.ones((batch_size, 1, cache_size), dtype=jnp.bool)
    transformer = transformer_lib.Transformer(config=config)
    params = transformer.init(
        jax.random.PRNGKey(0),
        jnp.array([[1]]),
        jnp.array([[1]]),
        cache,
        attention_mask,
    )

    outputs, cache = transformer.apply(
        params, jnp.array([[1]]), jnp.array([[1]]), cache, attention_mask
    )

    self.assertEqual(outputs.shape, expected_outputs_shape)
    self.assertEqual(cache['layer_0']['v'].shape, expected_cache_shape)

  @parameterized.parameters([
      dict(
          config=transformer_lib.TransformerConfig(
              num_layers=2,
              num_embed=0,  # unused
              embed_dim=0,  # unused
              hidden_dim=0,  # unused
              num_heads=3,
              head_dim=4,
              num_kv_heads=3,
              max_cache_length=2,
          ),
          keys=['layer_0', 'layer_1'],
          k_shape=(1, 2, 3, 4),
          v_shape=(1, 2, 3, 4),
      )
  ])
  def test_creates_cache(self, config, keys, k_shape, v_shape):
    cache = transformer_lib.init_cache(config, 1)
    self.assertEqual(list(cache.keys()), keys)
    self.assertEqual(cache['layer_0']['k'].shape, k_shape)
    self.assertEqual(cache['layer_0']['v'].shape, v_shape)

  @parameterized.parameters([
      dict(
          batch_size=1,
          seq_size=4,
          config=transformer_lib.TransformerConfig(
              num_layers=2,
              num_embed=4,  # unused
              embed_dim=2,
              hidden_dim=12,  # unused
              num_heads=3,
              head_dim=4,
              num_kv_heads=3,
              max_cache_length=6,
          ),
      )
  ])
  def test_forward_no_cache(
      self,
      batch_size: int,
      seq_size: int,
      config: transformer_lib.TransformerConfig,
  ):

    token_input = jnp.ones((batch_size, seq_size), dtype=jnp.int32)
    empty_cache = transformer_lib.init_cache(
        config, batch_size, dtype=jnp.float32
    )
    transformer = transformer_lib.Transformer(config=config)
    attention_mask = jnp.ones(
        (batch_size, seq_size, config.max_cache_length), dtype=jnp.bool
    )
    positions = transformer_lib.build_positions_from_mask(token_input != 0)
    params = transformer.init(
        jax.random.PRNGKey(0),
        token_input,
        positions,
        empty_cache,
        attention_mask,
    )

    output_cache, _ = transformer.apply(
        params, token_input, positions, empty_cache, attention_mask
    )

    attention_mask = jnp.ones((batch_size, seq_size, seq_size), dtype=jnp.bool)
    output_none, cache_none = transformer.apply(
        params, token_input, positions, None, attention_mask
    )

    self.assertIsNone(cache_none)
    np.testing.assert_array_almost_equal(output_cache, output_none, 1e-5)


if __name__ == '__main__':
  absltest.main()
