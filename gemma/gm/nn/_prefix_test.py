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

from gemma import gm
from gemma.gm.nn._prefix import PrefixTuning
from gemma.gm.nn._transformer import ModelInfo
import jax
import jax.numpy as jnp
import numpy as np


class TinyGemma(gm.nn.Transformer):
  """A small dummy Gemma3-like model for fast testing."""

  config: gm.nn.config.TransformerConfig = gm.nn.config.TransformerConfig(
      final_logit_softcap=None,
      num_embed=128,
      embed_dim=32,
      hidden_dim=64,
      num_heads=2,
      head_dim=16,
      num_kv_heads=1,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      use_qk_norm=True,
      attention_types=tuple([
          gm.nn.AttentionType.LOCAL_SLIDING,
          gm.nn.AttentionType.GLOBAL,
      ]),
      query_pre_attn_norm=gm.nn.config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
      attn_logits_soft_cap=None,
      sliding_window_size=32,
      transpose_gating_einsum=True,
  )
  INFO = ModelInfo(tokenizer_version=3)


def test_prefix_tuning_global_layers_only():
  model = TinyGemma()
  prefix_model = PrefixTuning.from_model(
      model, prefix_length=2, global_layers_only=True
  )

  batch_size = 1
  seq_len = 4
  tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
  rng = jax.random.key(0)

  # Use a single initialization to avoid redundancy
  variables = prefix_model.init(rng, tokens)
  params = variables['params']

  # Verify that some prefix parameters are created
  has_prefix = any(k.startswith('prefix_k_') for k in params)
  assert has_prefix

  # Run forward pass capturing output cache and intermediates
  out, state = prefix_model.apply(
      variables,
      tokens,
      capture_intermediates=True,
      mutable=['intermediates'],
  )

  assert out.cache is not None

  # Verify cache shape for layer 0
  layer_0_cache = out.cache['layer_0']
  assert layer_0_cache['k'].shape[1] == 6  # prefix (2) + seq (4) = 6

  captured = state['intermediates']

  # Find attention weights for layer 0
  attention_weights = captured['layer_0']['attn']['attention_weights'][
      '__call__'
  ][0]

  prefix_weights = attention_weights[0, :, :, :2]
  np.testing.assert_allclose(prefix_weights, 0.0, atol=1e-6)
