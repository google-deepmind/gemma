# Copyright 2025 DeepMind Technologies Limited.
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

"""Transformer sub-modules."""

import enum
from flax import linen as nn
from gemma.gm.math import _positional_embeddings
from gemma.gm.nn import _layers
import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Bool, Int  # pylint: disable=g-multiple-import,g-importing-member

K_MASK = -2.3819763e38  # Set to a large negative number.
DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0

# A dictionary with the following array shapes as keys:
# v: [batch_size, cache_size, num_heads, head_dim]
# k: [batch_size, cache_size, num_heads, head_dim]
# positions: [batch_size, cache_size]
# end_index: [batch_size]
LayerCache = dict[str, jax.Array]


def create_sliding_mask(
    positions: Int['B L'],
    *,
    cache_positions: Int['B cache_len'] | None = None,
    sliding_window_size: int,
) -> Bool['B L cache_len']:
  """Create the sliding mask for local sliding attention."""
  if cache_positions is None:
    cache_positions = positions

  cache_positions = cache_positions[..., None, :]  # B 1 cache_len
  positions = positions[..., :, None]  # B L 1
  sliding_mask = cache_positions > positions - sliding_window_size
  sliding_mask *= cache_positions < positions + sliding_window_size
  return sliding_mask


class AttentionType(enum.Enum):
  GLOBAL = 1
  LOCAL_SLIDING = 2


class Embedder(nn.Module):
  """Embedder module."""

  vocab_size: int
  embed_dim: int

  vision_proj_dim: int | None = None

  def setup(self):
    # Embedding matrix of shape [vocab_size, embed_dim]
    self.input_embedding_table = self.param(
        'input_embedding',
        nn.initializers.normal(),
        (self.vocab_size, self.embed_dim),
    )

    # For the multi-modal models, the encoder has additional parameters:
    # * `mm_soft_embedding_norm` and `mm_input_projection`: Those weights
    #   serve to project the soft tokens from the image encoder into the
    #   embedding space of the text encoder. Those tokens are then merged with
    #   the text tokens inside `Transformer._include_vision_embeddings`.
    if self.vision_proj_dim:
      self.mm_soft_embedding_norm = _layers.RMSNorm()
      self.mm_input_projection = _layers.Einsum(
          (self.vision_proj_dim, self.embed_dim)
      )

  def encode(self, x: jax.Array) -> jax.Array:
    """Encodes the input tokens.

    Args:
      x: Input tokens of shape [seq_len] or [batch_size, seq_len], where
        each token is an integer in [0, vocab_size).

    Returns:
      Encoded tokens of shape [seq_len, embed_dim] or [batch_size, seq_len,
      embed_dim].
    """
    x = self.input_embedding_table[(x,)]
    x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
    return x

  def decode(self, x: jax.Array) -> jax.Array:
    """Decodes the input vectors.

    Args:
      x: Array of shape [seq_len, embed_dim] or [batch_size, seq_len,
        embed_dim].

    Returns:
      Array of shape [seq_len, vocab_size] or [batch_size, seq_len, vocab_size].
    """
    return jnp.dot(x, self.input_embedding_table.T)

  def encode_vision(self, x: jax.Array) -> jax.Array:
    """Projects siglip embeddings to the embedding space of the text encoder."""
    x = self.mm_soft_embedding_norm(x)
    x = self.mm_input_projection('...tm,md->...td', x)
    return x


class Attention(nn.Module):
  """Attention module."""

  num_heads: int
  num_kv_heads: int
  features: int
  head_dim: int
  attn_type: AttentionType
  query_pre_attn_scalar: float
  rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
  rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  use_qk_norm: bool = False

  @property
  def use_qkv_einsum(self):
    return self.num_kv_heads == self.num_heads

  @property
  def use_gqa(self):
    return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1

  def setup(self):
    self.attn_vec_einsum = _layers.Einsum(
        shape=(self.num_heads, self.head_dim, self.features),
    )

    if self.use_qkv_einsum:
      self.qkv_einsum = _layers.Einsum(
          shape=(3, self.num_heads, self.features, self.head_dim),
      )
    else:
      self.q_einsum = _layers.Einsum(
          shape=(self.num_heads, self.features, self.head_dim),
      )
      self.kv_einsum = _layers.Einsum(
          shape=(2, self.num_kv_heads, self.features, self.head_dim),
      )
    if self.use_qk_norm:
      self._query_norm = _layers.RMSNorm()
      self._key_norm = _layers.RMSNorm()

    self.attention_weights = kd.nn.Identity()

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array]:
    """Applies multi-head attention to the inputs.

    Args:
      x: Input sequence of shape [batch_size, seq_len, embed_dim].
      segment_pos: Input absolute positions of shape [batch_size, seq_len].
      cache: KV cache or None.
      attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

    Returns:
      cache: Updated attention KV cache.
      outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
    """
    if self.use_qkv_einsum:
      # [batch_size, seq_len, num_heads, head_dim]
      query_proj, key_proj, value_proj = self.qkv_einsum('BTD,SNDH->SBTNH', x)
    else:
      query_proj = self.q_einsum('BTD,NDH->BTNH', x)
      key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

    if self.use_qk_norm:
      query_proj = self._query_norm(query_proj)
      key_proj = self._key_norm(key_proj)

    query_proj = _positional_embeddings.apply_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )
    query_scaled = query_proj * self.query_pre_attn_scalar

    key_proj = _positional_embeddings.apply_rope(
        key_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )

    # Cache is left aligned.
    # Save the KV values to the cache.
    if cache is not None:
      end_index = cache['end_index'][0]
      cache_size = cache['v'].shape[1]
      update_index = end_index % cache_size
      slice_indices = (0, update_index, 0, 0)

      # [batch_size, cache_size, num_heads, head_dim]
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )

      # [batch_size, cache_size, num_heads, head_dim]
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'],
          key_proj,
          slice_indices,
      )

      # [batch_size, cache_size]
      cache_positions = jax.lax.dynamic_update_slice(
          cache['positions'],
          segment_pos,
          slice_indices[:2],
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
      # [batch_size, seq_len, num_heads, cache_size]
      # If cache is None, then cache_size = seq_len.
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

    if self.attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / self.attn_logits_soft_cap)
      logits = logits * self.attn_logits_soft_cap

    if self.attn_type == AttentionType.LOCAL_SLIDING:
      if self.sliding_window_size is None:
        raise ValueError(
            'Sliding_window_size must be set if Local Sliding attention type'
        )
      sliding_mask = create_sliding_mask(
          segment_pos,
          cache_positions=cache_positions if cache else None,  # pylint: disable=undefined-variable
          sliding_window_size=self.sliding_window_size,
      )
      # [batch_size, seq_len, cache_size]
      attn_mask *= sliding_mask

    # [batch_size, seq_len, num_heads, cache_size]
    padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)

    # Multi-head attention matrices.
    # [batch_size, seq_len, num_heads, cache_size]
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)
    probs = self.attention_weights(probs)

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
      # [batch_size, seq_len, num_heads, head_dim]
      encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)

    # [batch_size, seq_len, features]
    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)

    if cache is not None:
      seq_len = x.shape[1]
      new_cache = {
          # [batch_size, cache_size, num_heads, head_dim]
          'v': value_proj,
          # [batch_size, cache_size, num_heads, head_dim]
          'k': key_proj,
          # TODO(epot): end_index & positions could be shared across layers.
          # [batch_size]
          'end_index': cache['end_index'] + seq_len,
          # [batch_size, cache_size]
          'positions': cache_positions,  # pylint: disable=undefined-variable
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
        # Save the positions for the sliding window attention.
        'positions': jnp.zeros((batch_size, cache_size), dtype=jnp.int32),
    }


class FeedForward(nn.Module):
  """Feed forward module."""

  features: int  # features = embed_dim
  hidden_dim: int
  transpose_gating_einsum: bool

  @nn.compact
  def __call__(self, x):
    """Applies the feed forward module.

    Args:
      x: Input sequence of shape [batch_size, seq_len, features].

    Returns:
      Output sequence of shape [batch_size, seq_len, features].
    """
    # Some versions use an alternate parameter ordering that
    # transposes hidden_dim and features.
    if self.transpose_gating_einsum:
      eq = '...F,NHF->...NH'
      gating = _layers.Einsum(
          shape=(2, self.hidden_dim, self.features),
          weight_name='gating_einsum',
      )
    else:
      eq = '...F,NFH->...NH'
      gating = _layers.Einsum(
          shape=(2, self.features, self.hidden_dim),
          weight_name='gating_einsum',
      )

    # Use the same scope for backwards compatibility with existing checkpoints
    # created before using `_layers.Einsum` here.
    nn.share_scope(self, gating)

    # [batch_size, seq_len, 2, hidden_dim]
    gate = gating(eq, x)
    # [batch_size, seq_len, hidden_dim]
    activations = nn.gelu(gate[..., 0, :]) * gate[..., 1, :]

    # Project back from hidden_dim to features.
    linear = _layers.Einsum(
        shape=(self.hidden_dim, self.features),
        weight_name='linear',
    )
    nn.share_scope(self, linear)

    # [batch_size, seq_len, features]
    outputs = linear('...H,HF->...F', activations)

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
  rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
  rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  use_qk_norm: bool = False

  def setup(self):
    self.pre_attention_norm = _layers.RMSNorm()

    self.attn = Attention(
        num_heads=self.num_heads,
        features=self.embed_dim,
        head_dim=self.head_dim,
        num_kv_heads=self.num_kv_heads,
        attn_type=self.attn_type,
        query_pre_attn_scalar=self.query_pre_attn_scalar,
        rope_base_frequency=self.rope_base_frequency,
        rope_scale_factor=self.rope_scale_factor,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
        use_qk_norm=self.use_qk_norm,
    )

    self.post_attention_norm = None
    if self.use_post_attn_norm:
      self.post_attention_norm = _layers.RMSNorm()

    self.pre_ffw_norm = _layers.RMSNorm()

    self.mlp = FeedForward(
        features=self.embed_dim,
        hidden_dim=self.hidden_dim,
        transpose_gating_einsum=self.transpose_gating_einsum,
    )

    self.post_ffw_norm = None
    if self.use_post_ffw_norm:
      self.post_ffw_norm = _layers.RMSNorm()

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array]:
    """Applies the block to the inputs.

    Args:
      x: Input sequence of shape [batch_size, seq_len, embed_dim].
      segment_pos: Input absolute positions of shape [batch_size, seq_len].
      cache: KV cache or None.
      attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

    Returns:
      cache: Updated attention KV cache.
      outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
    """
    inputs_normalized = self.pre_attention_norm(x)

    # attn_output.shape = [batch_size, seq_len, embed_dim]
    # cache["k"].shape = [batch_size, cache_size, num_heads, head_dim]
    # cache["v"].shape = [batch_size, cache_size, num_heads, head_dim]
    # cache["end_index"].shape = [batch_size]
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
