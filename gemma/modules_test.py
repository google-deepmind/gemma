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
"""Tests for transformer modules."""

from absl.testing import absltest
from absl.testing import parameterized
from gemma import modules
import jax
import jax.numpy as jnp
import numpy as np


class EmbedderTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          vocab_size=10,
          embed_dim=4,
          inputs=[2, 3],
          expected=[[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
      ),
  )
  def test_encodes(self, vocab_size, embed_dim, inputs, expected):
    embedder = modules.Embedder(vocab_size=vocab_size, embed_dim=embed_dim)
    output = embedder.apply(
        {'params': {'input_embedding': jnp.ones((vocab_size, embed_dim))}},
        inputs,
        method=modules.Embedder.encode,
    )
    np.testing.assert_array_equal(output, jnp.array(expected))

  @parameterized.parameters(
      dict(
          vocab_size=5,
          embed_dim=2,
          inputs=[[1, 2]],
          expected=[[3.0, 3.0, 3.0, 3.0, 3.0]],
      ),
  )
  def test_decodes(self, vocab_size, embed_dim, inputs, expected):
    embedder = modules.Embedder(vocab_size=vocab_size, embed_dim=embed_dim)
    output = embedder.apply(
        {'params': {'input_embedding': jnp.ones((vocab_size, embed_dim))}},
        jnp.array(inputs),
        method=modules.Embedder.decode,
    )
    np.testing.assert_array_equal(output, jnp.array(expected))


class AttentionTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          num_heads=2,
          head_dim=4,
          features=8,
          segment_pos=0,
          cache_size=2,
          batch_size=2,
          expected_cache_shape=(2, 2, 2, 4),
          expected_output_shape=(2, 1, 8),
      ),
  )
  def test_attention(
      self,
      num_heads,
      head_dim,
      features,
      segment_pos,
      cache_size,
      batch_size,
      expected_cache_shape,
      expected_output_shape,
  ):
    attn_mask = jnp.ones((batch_size, 1, num_heads))
    attn = modules.Attention(num_heads, num_heads, features, head_dim)
    cache = modules.init_layer_cache(
        cache_size, num_heads, head_dim, batch_size, dtype=jnp.float32
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

    self.assertEqual(cache['k'].shape, expected_cache_shape)
    self.assertEqual(output.shape, expected_output_shape)


class FeedForwardTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          features=2,
          hidden_dim=3,
          batch_size=2,
          expected_val=[11.72758674, 47.99916],
          expected_shape=(2, 1, 2),
      ),
  )
  def test_ffw(
      self, features, hidden_dim, batch_size, expected_val, expected_shape
  ):
    inputs = jnp.arange(1, batch_size+1)[:, None, None]
    inputs = jnp.repeat(inputs, features, axis=-1)
    ffw = modules.FeedForward(features=features, hidden_dim=hidden_dim)
    params = {
        'gating_einsum': jnp.ones((2, features, hidden_dim)),
        'linear': jnp.ones((hidden_dim, features)),
    }

    outputs = ffw.apply({'params': params}, inputs)

    np.testing.assert_array_almost_equal(outputs[:, 0, 0], expected_val)
    self.assertEqual(outputs.shape, expected_shape)


class BlockTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          num_heads=2,
          embed_dim=4,
          head_dim=6,
          cache_size=3,
          batch_size=2,
          expected_cache_shape=(2, 3, 2, 6),
          expected_output_shape=(2, 1, 4),
      ),
  )
  def test_block(
      self,
      num_heads,
      embed_dim,
      head_dim,
      cache_size,
      batch_size,
      expected_cache_shape,
      expected_output_shape,
  ):
    inputs = jnp.ones((batch_size, 1, embed_dim))
    cache = modules.init_layer_cache(
        cache_size, num_heads, head_dim, batch_size, dtype=jnp.float32
    )
    attn_mask = jnp.ones((batch_size, 1, cache_size))
    block = modules.Block(num_heads, num_heads, embed_dim, head_dim, 1)
    params = block.init(
        jax.random.PRNGKey(0), inputs, jnp.array([[0]]), cache, attn_mask
    )

    new_cache, outputs = block.apply(
        params, inputs, jnp.array([[0]]), cache, attn_mask
    )

    self.assertEqual(new_cache['k'].shape, expected_cache_shape)
    self.assertEqual(outputs.shape, expected_output_shape)


if __name__ == '__main__':
  absltest.main()
