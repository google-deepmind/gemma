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

import dataclasses
import math
from typing import Any

from gemma import gm
from gemma.gm.nn.gemma4 import _gemma4_270m
from gemma.gm.nn.gemma4 import _transformer as gt
from gemma.gm.nn.gemma4.audio import _modules as gemma4_audio
from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
import jax
import jax.numpy as jnp


def _get_output(model: gt.Transformer, **kwargs) -> tuple[gt.Output, Any]:

  def init_fn(**kwargs):
    out, params = model.init_with_output(jax.random.key(0), **kwargs)
    return out, params['params']

  return jax.eval_shape(init_fn, **kwargs)


def test_gemma4_270m_preset_and_multimodal_forward():
  assert gm.nn.Gemma4_270M is _gemma4_270m.Gemma4_270M
  base_config = gm.nn.Gemma4_270M.config
  assert (base_config.num_layers, base_config.embed_dim) == (24, 512)
  assert base_config.per_layer_input_dim == 512
  assert base_config.use_bidirectional_attention is None
  assert base_config.vision_encoder is not None
  assert base_config.vision_encoder.use_clipped_linears
  assert base_config.audio_encoder == gemma4_audio.ConformerConfig(
      use_key_scale=False
  )

  out, full_params = _get_output(
      gm.nn.Gemma4_270M(), tokens=jnp.ones((1, 4), dtype=jnp.int32)
  )
  assert out.logits.shape == (1, 4, base_config.num_embed)
  assert 'vision_encoder' in full_params
  assert 'audio_encoder' in full_params
  embedder_params = full_params['embedder']
  assert embedder_params['per_layer_embeddings'].shape == (1024, 10, 512)
  assert embedder_params['mm_input_embedding_extra'].shape == (128, 768)

  config = dataclasses.replace(
      base_config,
      num_embed=64,
      attention_types=base_config.attention_types[:2],
      vision_encoder=dataclasses.replace(
          base_config.vision_encoder, num_layers=1, output_length=1
      ),
      audio_encoder=None,
  )
  assert config.vision_encoder is not None
  patches, positions_xy, _ = gemma_vision.patchify_and_pad(
      [jnp.full((32, 32, 3), 0.5, dtype=jnp.float32)],
      patch_size=config.vision_encoder.patch_size,
      max_soft_tokens=config.vision_encoder.num_mm_tokens_per_image,
      pooling_kernel_size=config.vision_encoder.pooling_kernel_size,
  )
  vision_input = gt.PreprocessedVisionInput(
      patches=patches, positions_xy=positions_xy, soft_token_counts=(1,)
  )
  tokens = jnp.array(
      [[2, gemma_vision.TOKEN_PLACEHOLDER, 258882, 64]],
      dtype=jnp.int32,
  )
  model = gm.nn.Gemma4_270M(config=config, text_only=False)
  out, variables = model.init_with_output(
      jax.random.key(0),
      tokens=tokens,
      images=vision_input,
  )
  assert out.logits.shape == (1, 4, 64)
  assert 'vision_encoder' in variables['params']
  assert variables['params']['embedder']['per_layer_embeddings'].shape == (
      1024,
      10,
      512,
  )


def test_gemma4_270m_eoi_scaling_and_pq_ple_lookup():
  embedder = _gemma4_270m._Tiny270MEmbedder(
      vocab_size=262144,
      embed_dim=16,
      vision_proj_dim=16,
      per_layer_input_dim=8,
      num_layers=2,
  )
  tokens = jnp.array([[513, 258882, 262144]], dtype=jnp.int32)
  zero_hidden = jnp.zeros((1, 3, 16), dtype=jnp.float32)
  encode_ple = embedder.encode_per_layer_input
  variables = embedder.init(jax.random.key(0), tokens, method=embedder.encode)
  ple_vars = embedder.init(
      jax.random.key(1), zero_hidden, tokens, method=encode_ple
  )
  ple_table = jnp.zeros((1024, 10, 8), dtype=jnp.float32)
  ple_table = ple_table.at[1, 0, :].set(1.0).at[513, 5, :].set(2.0)
  updated_vars = {
      'params': {
          **ple_vars['params'],
          **variables['params'],
          'per_layer_embeddings': ple_table,
          'mm_input_embedding_extra': jnp.ones((128, 16), dtype=jnp.float32),
          'mm_input_projection': {'w': jnp.eye(16, 16, dtype=jnp.float32)},
      }
  }
  encoded: Any = embedder.apply(updated_vars, tokens, method=embedder.encode)
  expected_mm = jnp.full((16,), math.sqrt(16.0))
  assert jnp.allclose(encoded[0, 1], expected_mm, rtol=1e-5, atol=1e-5)
  assert jnp.allclose(encoded[0, 2], expected_mm, rtol=1e-5, atol=1e-5)
  ple_out: Any = embedder.apply(
      updated_vars, zero_hidden, tokens, method=encode_ple
  )
  assert jnp.allclose(
      ple_out[0, 0, 0], jnp.full((8,), 6.0), rtol=1e-5, atol=1e-5
  )
  assert jnp.allclose(
      ple_out[0, 1:], jnp.zeros((2, 2, 8)), rtol=1e-5, atol=1e-5
  )


def test_gemma4_270m_audio_extra_lookup_and_ple_zeroing():
  embedder = _gemma4_270m._Tiny270MEmbedder(
      vocab_size=262144,
      embed_dim=16,
      vision_proj_dim=16,
      audio_proj_dim=16,
      per_layer_input_dim=8,
      num_layers=2,
  )
  tokens = jnp.array([[513, 258883, 262272]], dtype=jnp.int32)
  zero_hidden = jnp.zeros((1, 3, 16), dtype=jnp.float32)
  encode_ple = embedder.encode_per_layer_input
  variables = embedder.init(jax.random.key(0), tokens, method=embedder.encode)
  ple_vars = embedder.init(
      jax.random.key(1), zero_hidden, tokens, method=encode_ple
  )
  audio_extra = jnp.zeros((128, 16), dtype=jnp.float32).at[0, :].set(3.0)
  updated_vars = {
      'params': {
          **ple_vars['params'],
          **variables['params'],
          'per_layer_embeddings': jnp.zeros((1024, 10, 8), dtype=jnp.float32),
          'audio_input_embedding_extra': audio_extra,
          'audio_input_projection': {'w': jnp.eye(16, 16, dtype=jnp.float32)},
      }
  }
  encoded: Any = embedder.apply(updated_vars, tokens, method=embedder.encode)
  assert jnp.allclose(encoded[0, 1], jnp.full((16,), 3.0), rtol=1e-5, atol=1e-5)
  assert jnp.allclose(encoded[0, 2], jnp.full((16,), 3.0), rtol=1e-5, atol=1e-5)
  ple_out: Any = embedder.apply(
      updated_vars, zero_hidden, tokens, method=encode_ple
  )
  assert jnp.allclose(
      ple_out[0, 1:], jnp.zeros((2, 2, 8)), rtol=1e-5, atol=1e-5
  )
