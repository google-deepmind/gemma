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
"""Transformer sub-modules."""

import enum
from flax import linen as nn
from gemma import layers
from gemma import positional_embeddings
import jax
import jax.numpy as jnp

K_MASK = -2.3819763e38  # Set to a large negative number.
LayerCache = dict[str, jax.Array]


class AttentionType(enum.Enum):
  GLOBAL = 1
  LOCAL_SLIDING = 2


class Embedder(nn.Module):
  """Embedder module."""

  vocab_size: int
  embed_dim: int

  def setup(self):
    self.input_embedding_table = self.param(
        'input_embedding',
        nn.initializers.normal(),
        (self.vocab_size, self.embed_dim),
    )

  def encode(self, x: jax.Array) -> jax.Array:
    x = self.input_embedding_table[(x,)]
    x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
    return x

  def decode(self, x: jax.Array) -> jax.Array:
    return jnp.dot(x, self.input_embedding_table.T)


class Attention(nn.Module):
  """Attention module."""

  num_heads: int
  num_kv_heads: int
  features: int
  head_dim: int
  attn_type: AttentionType
  query_pre_attn_scalar: float
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None

  @property
  def use_qkv_einsum(self):
    return self.num_kv_heads == self.num_heads

  @property
  def use_gqa(self):
    return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1

  def setup(self):
    self.attn_vec_einsum = layers.Einsum(
        shape=(self.num_heads, self.head_dim, self.features),
    )

    if self.use_qkv_einsum:
      self.qkv_einsum = layers.Einsum(
          shape=(3, self.num_heads, self.features, self.head_dim),
      )
    else:
      self.q_einsum = layers.Einsum(
          shape=(self.num_heads, self.features, self.head_dim),
      )
      self.kv_einsum = layers.Einsum(
          shape=(2, self.num_kv_heads, self.features, self.head_dim),
      )

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array]:
    seq_len = x.shape[1]

    if self.use_qkv_einsum:
      query_proj, key_proj, value_proj = self.qkv_einsum('BTD,SNDH->SBTNH', x)
    else:
      query_proj = self.q_einsum('BTD,NDH->BTNH', x)
      key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

    query_proj = positional_embeddings.apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
    )
    query_scaled = query_proj * self.query_pre_attn_scalar
    key_proj = positional_embeddings.apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
    )

    # Cache is left aligned.
    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )

    if self.use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = query_scaled.shape
      query_scaled = query_scaled.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_scaled, key_proj)
      b, t, k, g, s = logits.shape
      logits = logits.reshape((b, t, k * g, s))
    else:
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

    if self.attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / self.attn_logits_soft_cap)
      logits = logits * self.attn_logits_soft_cap

    if self.attn_type == AttentionType.LOCAL_SLIDING:
      if self.sliding_window_size is None:
        raise ValueError(
            'Sliding_window_size must be set if Local Sliding attention type'
        )

      all_ones = jnp.ones_like(attn_mask)
      sliding_mask = jnp.triu(
          all_ones, -1 * self.sliding_window_size + 1
      ) * jnp.tril(all_ones, self.sliding_window_size - 1)
      attn_mask = sliding_mask * attn_mask

    padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)
    if self.use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = probs.shape
      probs = probs.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs, value_proj)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
      encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)

    if cache is not None:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, attn_output

  @classmethod
  def init_cache(
      cls,
      cache_size: int,
      num_heads: int,
      head_dim: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> LayerCache:
    del cls  # not used
    return {
        'v': jnp.zeros(
            (batch_size, cache_size, num_heads, head_dim), dtype=dtype
        ),
        'k': jnp.zeros(
            (batch_size, cache_size, num_heads, head_dim), dtype=dtype
        ),
        'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
    }


class FeedForward(nn.Module):
  """Feed forward module."""

  features: int
  hidden_dim: int
  transpose_gating_einsum: bool

  @nn.compact
  def __call__(self, x):
    # Some versions use an alternate parameter ordering that
    # transposes hidden_dim and features.
    if self.transpose_gating_einsum:
      w_gating = self.param(
          'gating_einsum',
          nn.initializers.zeros_init(),
          ((2, self.hidden_dim, self.features)),
      )
      w_gating = w_gating.transpose((0, 2, 1))
    else:
      w_gating = self.param(
          'gating_einsum',
          nn.initializers.zeros_init(),
          ((2, self.features, self.hidden_dim)),
      )
    ff_gate = jnp.dot(x, w_gating[0])
    gate_value = nn.gelu(ff_gate)

    # Up projection
    ff1 = jnp.dot(x, w_gating[1])
    activations = gate_value * ff1

    # Down projection
    w_linear = self.param(
        'linear',
        nn.initializers.zeros_init(),
        (self.hidden_dim, self.features),
    )
    outputs = jnp.dot(activations, w_linear)

    return outputs


class Block(nn.Module):
  """Transformer block."""

  num_heads: int
  num_kv_heads: int
  embed_dim: int
  head_dim: int
  hidden_dim: int
  use_post_attn_norm: bool
  use_post_ffw_norm: bool
  attn_type: AttentionType
  query_pre_attn_scalar: float
  transpose_gating_einsum: bool
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None

  def setup(self):
    self.pre_attention_norm = layers.RMSNorm()
    self.attn = Attention(
        num_heads=self.num_heads,
        features=self.embed_dim,
        head_dim=self.head_dim,
        num_kv_heads=self.num_kv_heads,
        attn_type=self.attn_type,
        query_pre_attn_scalar=self.query_pre_attn_scalar,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
    )
    self.post_attention_norm = None
    if self.use_post_attn_norm:
      self.post_attention_norm = layers.RMSNorm()

    self.pre_ffw_norm = layers.RMSNorm()
    self.mlp = FeedForward(
        features=self.embed_dim,
        hidden_dim=self.hidden_dim,
        transpose_gating_einsum=self.transpose_gating_einsum,
    )
    self.post_ffw_norm = None
    if self.use_post_ffw_norm:
      self.post_ffw_norm = layers.RMSNorm()

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array]:
    inputs_normalized = self.pre_attention_norm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
    )
    if self.post_attention_norm is not None:
      attn_output = self.post_attention_norm(attn_output)
    attn_output += x
    outputs = self.pre_ffw_norm(attn_output)
    outputs = self.mlp(outputs)
    if self.post_ffw_norm is not None:
      outputs = self.post_ffw_norm(outputs)
    outputs += attn_output
    return cache, outputs
