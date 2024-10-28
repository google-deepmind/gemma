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

"""Tests for the Gemma transformer."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from gemma import modules
from gemma import transformer as transformer_lib
import jax
import jax.numpy as jnp
import numpy as np


class TransformerConfigTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(ctor=transformer_lib.TransformerConfig.gemma_2b),
      dict(ctor=transformer_lib.TransformerConfig.gemma_7b),
      dict(ctor=transformer_lib.TransformerConfig.gemma2_9b),
      dict(ctor=transformer_lib.TransformerConfig.gemma2_27b),
  )
  def test_known_architectures(self, ctor):
    transformer_config = ctor(cache_size=16)
    self.assertIsInstance(transformer_config, transformer_lib.TransformerConfig)


class TransformerTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          num_layers=transformer_lib._NUM_LAYERS_GEMMA_2B,
          expected_call='gemma_2b',
      ),
      dict(
          num_layers=transformer_lib._NUM_LAYERS_GEMMA_7B,
          expected_call='gemma_7b',
      ),
      dict(
          num_layers=transformer_lib._NUM_LAYERS_GEMMA2_2B,
          expected_call='gemma2_2b',
      ),
      dict(
          num_layers=transformer_lib._NUM_LAYERS_GEMMA2_9B,
          expected_call='gemma2_9b',
      ),
      dict(
          num_layers=transformer_lib._NUM_LAYERS_GEMMA2_27B,
          expected_call='gemma2_27b',
      ),
  )
  def test_params_load(self, num_layers, expected_call):
    mock_transformer = {
        'transformer': {f'layer_{i}': None for i in range(num_layers)}
    }

    with mock.patch.object(
        transformer_lib.TransformerConfig, expected_call
    ) as mock_method:
      transformer_lib.TransformerConfig.from_params(mock_transformer)
      mock_method.assert_called_once()

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
        final_logit_softcap=None,
        attention_types=[modules.AttentionType.GLOBAL] * num_layers,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )
    cache = config.init_cache(batch_size, dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, 1, cache_size), dtype=jnp.bool)
    with jax.numpy_rank_promotion('raise'):
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

  @parameterized.parameters(
      ('final_logit_softcap',),
      ('attn_logits_soft_cap',),
  )
  def test_logit_softcap(
      self,
      soft_cap_arg,
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
        attention_types=[modules.AttentionType.GLOBAL] * 3,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )

    no_soft_cap_args = {
        'final_logit_softcap': None,
        'attn_logits_soft_cap': None,
    }

    soft_cap_args = no_soft_cap_args.copy()
    soft_cap_args[soft_cap_arg] = soft_cap_val

    config_soft_cap = transformer_lib.TransformerConfig(
        **(params | soft_cap_args)
    )
    config_no_soft_cap = transformer_lib.TransformerConfig(
        **(params | no_soft_cap_args)
    )

    all_outputs = []
    for config in [config_soft_cap, config_no_soft_cap]:
      with jax.numpy_rank_promotion('raise'):
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
              final_logit_softcap=None,
              attention_types=[modules.AttentionType.GLOBAL] * 2,
              use_post_attn_norm=False,
              use_post_ffw_norm=False,
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
              final_logit_softcap=None,
              attention_types=[modules.AttentionType.GLOBAL] * 2,
              use_post_attn_norm=False,
              use_post_ffw_norm=False,
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
    with jax.numpy_rank_promotion('raise'):
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

      attention_mask = jnp.ones(
          (batch_size, seq_size, seq_size), dtype=jnp.bool
      )
      output_none, cache_none = transformer.apply(
          params, token_input, positions, None, attention_mask
      )

      self.assertIsNone(cache_none)
      np.testing.assert_array_almost_equal(output_cache, output_none, 1e-5)

  def test_attention_types(
      self,
  ):
    config = transformer_lib.TransformerConfig(
        num_layers=2,
        num_embed=4,
        embed_dim=2,
        hidden_dim=12,
        num_heads=3,
        head_dim=4,
        num_kv_heads=3,
        max_cache_length=6,
        final_logit_softcap=None,
        attention_types=[modules.AttentionType.GLOBAL] * 2,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )

    cache = config.init_cache(batch_size=1, dtype=jnp.float32)
    self.assertTrue(cache)

  @parameterized.named_parameters([
      (
          'by_head_dim',
          transformer_lib.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
          0.5,
      ),
      (
          'by_embed_dim_div_num_heads',
          transformer_lib.QueryPreAttentionNormalisation.BY_EMBED_DIM_DIV_NUM_HEADS,
          6,
      ),
      (
          'unset',
          None,
          0.5,
      ),
  ])
  def test_query_pre_attn_scalar(
      self,
      query_pre_attn_norm,
      expected_scalar,
  ):
    num_layers = 2
    config = transformer_lib.TransformerConfig(
        num_layers=num_layers,
        num_embed=4,
        embed_dim=48,
        hidden_dim=12,
        num_heads=8,
        head_dim=4,
        num_kv_heads=3,
        final_logit_softcap=None,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        attention_types=[modules.AttentionType.GLOBAL] * num_layers,
        query_pre_attn_norm=query_pre_attn_norm,
    )
    self.assertEqual(config.query_pre_attn_scalar(), expected_scalar)


class TransformerUtilsTest(parameterized.TestCase):

  def test_make_causal_attn_mask(self):
    input_mask = jnp.array(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    causal_attn_mask = transformer_lib.make_causal_attn_mask(input_mask)

    expected_mask_shape = tuple(list(input_mask.shape) + [input_mask.shape[-1]])
    self.assertEqual(causal_attn_mask.shape, expected_mask_shape)
    self.assertEqual(causal_attn_mask.dtype, jnp.bool)

    # This reduces the attention mask, to a mask of which tokens are ever (once
    # or more) attended to. It should be the same as the input mask, if
    # attention mask is correct.
    token_ever_attended_mask = jnp.sum(
        jnp.astype(causal_attn_mask, jnp.int32), axis=1, dtype=jnp.bool
    )
    np.testing.assert_array_equal(input_mask, token_ever_attended_mask)

    # Iterate over sequences in batch.
    for i in range(causal_attn_mask.shape[0]):

      last_number_of_tokens_attended = 0
      # Iterate over tokens in sequence.
      for j in range(causal_attn_mask.shape[1]):
        if not input_mask[i, j]:
          break
        number_of_tokens_attended = jnp.sum(
            jnp.astype(causal_attn_mask[i, j, :], jnp.int32)
        )
        # Each token in the sequence pays attention to one more token than the
        # previous token in the sequence.
        self.assertEqual(
            number_of_tokens_attended, last_number_of_tokens_attended + 1
        )
        last_number_of_tokens_attended = number_of_tokens_attended

  def test_make_causal_attn_mask_fails_with_bad_input_mask_shape(self):
    bad_input_mask = jnp.array([[[True]]])
    with self.assertRaises(ValueError):
      transformer_lib.make_causal_attn_mask(bad_input_mask)

  def test_make_causal_with_prefix_attn_mask(self):
    input_mask = jnp.array(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    prefix_mask = jnp.array(
        [[True, True, False, False, False], [True, True, False, False, False]]
    )
    causal_with_prefix_attn_mask = (
        transformer_lib.make_causal_with_prefix_attn_mask(
            input_mask, prefix_mask
        )
    )

    expected_mask_shape = tuple(list(input_mask.shape) + [input_mask.shape[-1]])
    self.assertEqual(causal_with_prefix_attn_mask.shape, expected_mask_shape)
    self.assertEqual(causal_with_prefix_attn_mask.dtype, jnp.bool)

    # This reduces the attention mask, to a mask of which tokens are ever (once
    # or more) attended to. It should be the same as the input mask, if
    # attention mask is correct.
    token_ever_attended_mask = jnp.sum(
        jnp.astype(causal_with_prefix_attn_mask, jnp.int32),
        axis=1,
        dtype=jnp.bool,
    )
    np.testing.assert_array_equal(input_mask, token_ever_attended_mask)

    # This reduces the attention mask, to a mask of which tokens are *always*
    # attended to. It should be the same as the prefix mask, if attention mask
    # is correct.
    token_always_attended_mask = jnp.prod(
        jnp.astype(causal_with_prefix_attn_mask, jnp.int32),
        axis=1,
        dtype=jnp.bool,
    )
    np.testing.assert_array_equal(prefix_mask, token_always_attended_mask)

    # Iterate over sequences in batch.
    for i in range(causal_with_prefix_attn_mask.shape[0]):

      last_number_of_tokens_attended = 0
      # Iterate over tokens in sequence.
      for j in range(causal_with_prefix_attn_mask.shape[1]):
        if not input_mask[i, j]:
          break
        number_of_tokens_attended = jnp.sum(
            jnp.astype(causal_with_prefix_attn_mask[i, j, :], jnp.int32)
        )

        if prefix_mask[i, j]:
          # Each token in the prefix part of the sequence pays attention to all
          # the tokens in the prefix part of the sequence.
          self.assertEqual(
              number_of_tokens_attended, jnp.sum(prefix_mask[i, :])
          )
        else:
          # Each token in the non-prefix part of the sequence pays attention to
          # one more token than the previous token in the sequence.
          self.assertEqual(
              number_of_tokens_attended, last_number_of_tokens_attended + 1
          )

        last_number_of_tokens_attended = number_of_tokens_attended

  def test_make_causal_with_prefix_attn_mask_fails_with_bad_input_mask_shape(
      self,
  ):
    bad_input_mask = jnp.array([[[True]]])
    prefix_mask = jnp.array([[True], [True]])
    with self.assertRaises(ValueError):
      transformer_lib.make_causal_with_prefix_attn_mask(
          bad_input_mask, prefix_mask
      )

  def test_make_causal_with_prefix_attn_mask_fails_with_bad_prefix_mask_shape(
      self,
  ):
    input_mask = jnp.array([[True], [True]])
    bad_prefix_mask = jnp.array([[[True]]])
    with self.assertRaises(ValueError):
      transformer_lib.make_causal_with_prefix_attn_mask(
          input_mask, bad_prefix_mask
      )


if __name__ == '__main__':
  absltest.main()
