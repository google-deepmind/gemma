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

"""Tests for Gemma4 vision transformer."""

from gemma.gm.nn.gemma4.vision import _transformer
import jax
import jax.numpy as jnp


def test_vision_block_output_shape():

  batch_size = 3
  seq_len = 16
  d_model = 8

  block = _transformer.VisionBlock(
      d_model=d_model,
      ffw_hidden=d_model * 4,
      num_heads=2,
      num_kv_heads=1,
      key_size=4,
  )

  x = jax.random.uniform(jax.random.PRNGKey(42), [batch_size, seq_len, d_model])
  input_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
  attn_mask = input_mask[:, :, None] * input_mask[:, None, :]
  positions = jnp.ones((batch_size, seq_len, 2), dtype=jnp.int32)
  params = block.init(jax.random.PRNGKey(42), x, attn_mask, positions)['params']
  output = block.apply({'params': params}, x, attn_mask, positions)[0]

  assert output.shape == (batch_size, seq_len, d_model)


def test_vision_transformer_output_shape():

  batch_size = 3
  seq_len = 16
  d_model = 8

  block = _transformer.VisionTransformer(
      d_model=d_model,
      ffw_hidden=d_model * 4,
      num_heads=2,
      num_layers=1,
  )

  x = jax.random.uniform(jax.random.PRNGKey(42), [batch_size, seq_len, d_model])
  input_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
  positions = jnp.ones((batch_size, seq_len, 2), dtype=jnp.int32)
  params = block.init(jax.random.PRNGKey(42), x, input_mask, positions)[
      'params'
  ]
  output = block.apply({'params': params}, x, input_mask, positions)

  assert output.shape == (batch_size, seq_len, d_model)
