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

jax.config.update('jax_numpy_rank_promotion', 'raise')


class TransformerTest(parameterized.TestCase):

  @parameterized.parameters(
      # Prime number to ease shape tracing
      dict(
          num_layers=3,
          num_embed=17,
          embed_dim=2,
          num_heads=2,
          num_kv_heads=2,
          hidden_dim=11,
          head_dim=8,
          cache_size=29,
          batch_size=7,
          sequence_length=17,
          expected_outputs_shape=(7, 17, 17),
          expected_cache_shape=(7, 29, 2, 8),
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
          sequence_length=1,
          expected_outputs_shape=(1, 1, 4),
          expected_cache_shape=(1, 2, 1, 4),
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
      sequence_length,
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
        logit_softcapping=None,
        attn_query_splits=None,
        attention_type=transformer_lib.AttentionType.GLOBAL,
        use_post_attn_norm=False,
    )
    cache = config.init_cache(batch_size, dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, 1, cache_size), dtype=jnp.bool)
    transformer = transformer_lib.Transformer(config=config)
    params = transformer.init(
        jax.random.PRNGKey(0),
        last_tokens=jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
        positions=jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
        cache=cache,
        attention_mask=attention_mask,
    )

    outputs, cache = transformer.apply(
        params,
        jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
        jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
        cache,
        attention_mask,
    )

    self.assertEqual(outputs.shape, expected_outputs_shape)
    self.assertEqual(cache['layer_0']['v'].shape, expected_cache_shape)

  def test_logit_softcapping(
      self,
  ):
    cache_size = 2
    batch_size = 1
    sequence_length = 1
    soft_cap_val = 0.001

    attention_mask = jnp.ones((batch_size, 1, cache_size), dtype=jnp.bool)

    params = dict(
        num_layers=3,
        num_embed=4,
        embed_dim=2,
        num_heads=2,
        num_kv_heads=1,
        hidden_dim=4,
        head_dim=4,
        max_cache_length=cache_size,
        attn_query_splits=None,
        attention_type=transformer_lib.AttentionType.GLOBAL,
        use_post_attn_norm=False,
    )
    config_soft_cap = transformer_lib.TransformerConfig(
        **(params | {'logit_softcapping': soft_cap_val})
    )
    config_no_soft_cap = transformer_lib.TransformerConfig(
        **(params | {'logit_softcapping': None})
    )

    all_outputs = []
    for config in [config_soft_cap, config_no_soft_cap]:
      cache = config.init_cache(batch_size, dtype=jnp.float32)
      transformer = transformer_lib.Transformer(config=config)

      params = transformer.init(
          jax.random.PRNGKey(0),
          last_tokens=jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
          positions=jnp.array([[1]]),
          cache=cache,
          attention_mask=attention_mask,
      )

      outputs, _ = transformer.apply(
          params, jnp.array([[1]]), jnp.array([[1]]), cache, attention_mask
      )
      all_outputs.append(outputs)

    soft_cap_outputs, no_soft_cap_outputs = all_outputs  # pylint: disable=unbalanced-tuple-unpacking

    # Ensure that values aren't equal coming out of computation
    self.assertFalse((soft_cap_outputs == no_soft_cap_outputs).all())

    # Run soft capping manually
    manual_soft_cap_logits = no_soft_cap_outputs / soft_cap_val
    manual_soft_cap_logits = jnp.tanh(manual_soft_cap_logits) * soft_cap_val

    np.testing.assert_array_almost_equal(
        manual_soft_cap_logits, soft_cap_outputs, 1e-5
    )

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
              logit_softcapping=None,
              attn_query_splits=None,
              attention_type=transformer_lib.AttentionType.GLOBAL,
              use_post_attn_norm=False,
          ),
          keys=['layer_0', 'layer_1'],
          k_shape=(1, 2, 3, 4),
          v_shape=(1, 2, 3, 4),
      )
  ])
  def test_creates_cache(self, config, keys, k_shape, v_shape):
    cache = config.init_cache(1)
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
              logit_softcapping=None,
              attn_query_splits=None,
              attention_type=transformer_lib.AttentionType.GLOBAL,
              use_post_attn_norm=False,
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
    empty_cache = config.init_cache(batch_size, dtype=jnp.float32)
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
