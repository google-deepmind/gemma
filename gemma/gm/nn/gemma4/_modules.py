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

"""Transformer sub-modules."""

import enum
from flax import linen as nn
from gemma.gm.math import _positional_embeddings
from gemma.gm.nn.gemma4 import _layers
import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron.ktyping import Bool, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member

K_MASK = -2.3819763e38  # Set to a large negative number.
DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0

# A dictionary with the following array shapes as keys:
# v: [batch_size, cache_size, num_heads, key_size]
# k: [batch_size, cache_size, num_heads, key_size]
# positions: [batch_size, cache_size]
# end_index: [batch_size]
LayerCache = dict[str, jax.Array]


def _create_sliding_mask(
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
  num_layers: int = 0
  per_layer_input_dim: int = 0

  vision_proj_dim: int | None = None

  audio_proj_dim: int | None = None

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
    #   the text tokens inside `Transformer._merge_mm_embeddings`.
    # * `audio_input_projection` and `audio_soft_embedding_norm`: Analogous
    #   weights for projecting audio encoder outputs into the text embedding
    #   space. These tokens are merged via `Transformer._encode_audio`.
    if self.vision_proj_dim:
      self.mm_input_projection = _layers.Einsum(
          (self.vision_proj_dim, self.embed_dim)
      )
      self.mm_pre_projection_norm = _layers.RMSNorm(with_scale=False)

    if self.audio_proj_dim:
      self.audio_input_projection = _layers.Einsum(
          (self.audio_proj_dim, self.embed_dim)
      )
      self.audio_soft_embedding_norm = _layers.RMSNorm(with_scale=False)

    if self.per_layer_input_dim:
      self.per_layer_input_embedding_table = self.param(
          'per_layer_embeddings',
          nn.initializers.normal(),
          (self.vocab_size, self.num_layers, self.per_layer_input_dim),
      )
      self.per_layer_model_projection = _layers.Einsum(
          (self.embed_dim, self.num_layers, self.per_layer_input_dim),
          w_scale=(float(self.embed_dim) ** -0.5),
      )
      self.per_layer_projection_norm = _layers.RMSNorm()

  def encode(self, x: jax.Array) -> jax.Array:
    """Encodes the input tokens.

    Args:
      x: Input tokens of shape [seq_len] or [batch_size, seq_len], where each
        token is an integer in [0, vocab_size).

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

  @typechecked
  def encode_logits(self, x: Float['*B L V']) -> Float['*B L D']:
    """Encodes the input logits.

    Converts the logits to probabilities and uses that as a weighted sum of the
    embeddings.

    Args:
      x: Logits of shape [batch_size, seq_len, vocab_size].

    Returns:
      Encoded logits of shape [batch_size, seq_len, embed_dim].
    """
    probs = jax.nn.softmax(x.astype(jnp.float32), axis=-1).astype(x.dtype)
    x = jnp.einsum('...v,ve->...e', probs, self.input_embedding_table)
    x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
    return x

  def encode_vision(self, x: jax.Array) -> jax.Array:
    """Projects vision embeddings to the embedding space of the text encoder."""
    x = self.mm_pre_projection_norm(x)
    x = self.mm_input_projection('...tm,md->...td', x)
    return x

  def encode_audio(self, x: jax.Array) -> jax.Array:
    """Projects audio embeddings to the embedding space of the text encoder."""
    x = self.audio_input_projection('...tm,md->...td', x)
    x = self.audio_soft_embedding_norm(x)
    return x

  def encode_per_layer_input(self, x: jax.Array, t: jax.Array) -> jax.Array:
    """Encodes the input tokens.

    Args:
      x: Input shape [seq_len, embed_dim] or [batch_size, seq_len, embed_dim].
      t: Input tokens of shape [seq_len] or [batch_size, seq_len], where each
        token is an integer in [0, vocab_size).

    Returns:
      Encoded input of shape [seq_len, num_layers, per_layer_input_dim] or
      [batch_size, seq_len, num_layers, per_layer_input_dim].
    """
    # Replace tokens outside of the text vocab with zeros.
    t = jnp.where(
        jnp.logical_and(t >= 0, t < self.vocab_size), t, jnp.zeros_like(t)
    )
    x = self.per_layer_model_projection('...td,dnp->...tnp', x)
    x = self.per_layer_projection_norm(x)
    y = self.per_layer_input_embedding_table[(t,)]
    y *= jnp.sqrt(self.per_layer_input_dim).astype(y.dtype)
    return (x + y) * jax.lax.rsqrt(2.0).astype(x.dtype)


class Attention(nn.Module):
  """Attention module."""

  num_heads: int
  num_kv_heads: int
  features: int
  key_size: int
  attn_type: AttentionType
  rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
  rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
  rope_proportion: float | None = None
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  qk_norm_with_scale: bool = True
  k_eq_v: bool = False

  @property
  def use_gqa(self):
    return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1

  def setup(self):
    self.attn_vec_einsum = _layers.Einsum(
        shape=(self.num_heads, self.key_size, self.features),
    )
    self.q_einsum = _layers.Einsum(
        shape=(self.num_heads, self.features, self.key_size),
    )
    if self.k_eq_v:
      self.k_einsum = _layers.Einsum(
          shape=(self.num_kv_heads, self.features, self.key_size)
      )
    else:
      self.kv_einsum = _layers.Einsum(
          shape=(2, self.num_kv_heads, self.features, self.key_size),
      )
    self.query_norm = _layers.RMSNorm(with_scale=self.qk_norm_with_scale)
    self.key_norm = _layers.RMSNorm(with_scale=self.qk_norm_with_scale)
    self.value_norm = _layers.RMSNorm(with_scale=False)

    self.attention_weights = kd.nn.Identity()

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
      kv_shared_cache: LayerCache | None = None,
  ) -> tuple[LayerCache | None, jax.Array]:
    """Applies multi-head attention to the inputs.

    Args:
      x: Input sequence of shape [batch_size, seq_len, embed_dim].
      segment_pos: Input absolute positions of shape [batch_size, seq_len].
      cache: KV cache or None.
      attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].
      kv_shared_cache: Cache for shared KV layers.

    Returns:
      cache: Updated attention KV cache.
      outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
    """
    query_proj = self.q_einsum('BTD,NDH->BTNH', x)
    query_proj = self.query_norm(query_proj)
    query_proj = _positional_embeddings.apply_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
        rope_proportion=self.rope_proportion,
    )

    # TODO(imayank): move the key_proj and value_proj to kv_shared_cache=None
    # case after checkpoints remove the kv_einsum from the shared layers.
    if self.k_eq_v:
      output = self.k_einsum('BSD,KDH->BSKH', x)
      key_proj, value_proj = output, output
    else:
      key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)
    key_proj = self.key_norm(key_proj)
    value_proj = self.value_norm(value_proj)

    if kv_shared_cache is not None:
      key_proj = kv_shared_cache['k']
      value_proj = kv_shared_cache['v']
    else:
      key_proj = _positional_embeddings.apply_rope(
          key_proj,
          segment_pos,
          base_frequency=self.rope_base_frequency,
          scale_factor=self.rope_scale_factor,
          rope_proportion=self.rope_proportion,
      )

    # Cache is left aligned.
    # Save the KV values to the cache.
    if kv_shared_cache is not None:
      cache_positions = kv_shared_cache.get('positions')
    elif cache is not None:
      end_index = cache['end_index']
      cache_size = cache['v'].shape[1]
      seq_len = x.shape[1]
      # [batch_size, seq_len]
      indices = (end_index[:, None] + jnp.arange(seq_len)[None, :]) % cache_size
      batch_indices = jnp.arange(x.shape[0])[:, None]

      # [batch_size, cache_size, num_heads, key_size]
      value_proj = cache['v'].at[batch_indices, indices].set(value_proj)

      # [batch_size, cache_size, num_heads, key_size]
      key_proj = cache['k'].at[batch_indices, indices].set(key_proj)

      # [batch_size, cache_size]
      cache_positions = (
          cache['positions'].at[batch_indices, indices].set(segment_pos)
      )
    else:
      cache_positions = None

    if self.use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = query_proj.shape
      query_proj = query_proj.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_proj, key_proj)
      b, t, k, g, s = logits.shape
      logits = logits.reshape((b, t, k * g, s))
    else:
      # [batch_size, seq_len, num_heads, cache_size]
      # If cache is None, then cache_size = seq_len.
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_proj, key_proj)

    if self.attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / self.attn_logits_soft_cap)
      logits = logits * self.attn_logits_soft_cap

    if self.attn_type == AttentionType.LOCAL_SLIDING:
      if self.sliding_window_size is None:
        raise ValueError(
            'Sliding_window_size must be set if Local Sliding attention type'
        )
      sliding_mask = _create_sliding_mask(
          segment_pos,
          cache_positions=cache_positions,
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
      # [batch_size, seq_len, num_heads, key_size]
      encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)

    # [batch_size, seq_len, features]
    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)

    # Always cache the layer-sharing KV.
    # This also includes the context KV if cache is not None.
    # i.e. cache_size can be == seq_len or == cache_len if cache is not None.
    new_cache = {
        # [batch_size, cache_size, num_heads, key_size]
        'v': value_proj,
        # [batch_size, cache_size, num_heads, key_size]
        'k': key_proj,
    }
    # Remaining keys for context KV.
    if cache is not None:
      seq_len = x.shape[1]
      # [batch_size]
      new_cache['end_index'] = cache['end_index'] + seq_len
      assert (
          cache_positions is not None
      ), 'cache_positions should not be None when cache is not None'
      # [batch_size, cache_size]
      new_cache['positions'] = cache_positions

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
    eq = '...F,NHF->...NH'
    gating = _layers.Einsum(
        shape=(2, self.hidden_dim, self.features),
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
  rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
  rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  qk_norm_with_scale: bool = True
  num_global_kv_heads: int | None = None
  global_key_size: int | None = None
  k_eq_v_global: bool = False
  global_rope_proportion: float | None = None
  local_rope_proportion: float | None = None
  per_layer_input_dim: int = 0
  # MoE parameters (only used when enable_moe=True).
  enable_moe: bool = False
  num_experts: int = 0
  expert_dim: int = 0
  top_k_experts: int = 0

  def setup(self):
    self.pre_attention_norm = _layers.RMSNorm()

    self.skip_scale = self.param('skip_scale', nn.initializers.ones, (1,))

    # Local attention parameters.
    self.effective_num_kv_heads = self.num_kv_heads
    self.key_size = self.head_dim
    self.k_eq_v = False
    rope_proportion = self.local_rope_proportion

    # Global attention parameters.
    if self.attn_type == AttentionType.GLOBAL:
      if self.num_global_kv_heads is not None:
        self.effective_num_kv_heads = self.num_global_kv_heads
      if self.global_key_size is not None:
        self.key_size = self.global_key_size
      self.k_eq_v = self.k_eq_v_global
      rope_proportion = self.global_rope_proportion

    self.attn = Attention(
        num_heads=self.num_heads,
        features=self.embed_dim,
        key_size=self.key_size,
        num_kv_heads=self.effective_num_kv_heads,
        attn_type=self.attn_type,
        rope_base_frequency=self.rope_base_frequency,
        rope_scale_factor=self.rope_scale_factor,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
        qk_norm_with_scale=self.qk_norm_with_scale,
        rope_proportion=rope_proportion,
        k_eq_v=self.k_eq_v,
    )

    self.post_attention_norm = None
    if self.use_post_attn_norm:
      self.post_attention_norm = _layers.RMSNorm()

    if self.enable_moe:
      self._setup_moe()
    else:
      self._setup_dense()

    if self.per_layer_input_dim:
      self.post_per_layer_input_norm = _layers.RMSNorm()
      self.per_layer_input_gate = _layers.Einsum(
          shape=(self.embed_dim, self.per_layer_input_dim),
      )
      self.per_layer_projection = _layers.Einsum(
          shape=(self.per_layer_input_dim, self.embed_dim),
      )

  def _setup_dense(self):
    """Setup for standard (non-MoE) FFW."""
    self.pre_ffw_norm = _layers.RMSNorm()

    self.mlp = FeedForward(
        features=self.embed_dim,
        hidden_dim=self.hidden_dim,
    )

    self.post_ffw_norm = None
    if self.use_post_ffw_norm:
      self.post_ffw_norm = _layers.RMSNorm()

  def _setup_moe(self):
    """Setup for Mixture-of-Experts FFW."""
    from gemma.gm.nn.gemma4 import _moe  # pylint: disable=g-import-not-at-top

    # Dense shared branch: pre_ffw2_norm -> mlp2 -> post_ffw2_norm
    self.pre_ffw2_norm = _layers.RMSNorm()
    self.mlp2 = FeedForward(
        features=self.embed_dim,
        hidden_dim=self.hidden_dim,
    )
    self.post_ffw2_norm = None
    if self.use_post_ffw_norm:
      self.post_ffw2_norm = _layers.RMSNorm()

    # MoE branch: pre_ffw_norm -> mlp(moe) -> post_ffw1_norm
    self.pre_ffw_norm = _layers.RMSNorm()
    self.mlp = _moe.MoERagged(
        features=self.embed_dim,
        hidden_dim=self.expert_dim,
        num_experts=self.num_experts,
        num_experts_per_datapoint=self.top_k_experts,
    )
    self.post_ffw1_norm = None
    if self.use_post_ffw_norm:
      self.post_ffw1_norm = _layers.RMSNorm()

    # Post-FFW norm applied after combining both branches
    self.post_ffw_norm = None
    if self.use_post_ffw_norm:
      self.post_ffw_norm = _layers.RMSNorm()

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
      per_layer_input: jax.Array | None = None,
      kv_shared_cache: LayerCache | None = None,
  ) -> tuple[LayerCache | None, jax.Array]:
    """Applies the block to the inputs.

    Args:
      x: Input sequence of shape [batch_size, seq_len, embed_dim].
      segment_pos: Input absolute positions of shape [batch_size, seq_len].
      cache: KV cache or None.
      attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].
      per_layer_input: Per-layer input of shape [batch_size, seq_len,
        per_layer_input_dim].
      kv_shared_cache: Cache for shared KV layers.

    Returns:
      cache: Updated attention KV cache.
      outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
    """
    # 1. Attention
    inputs_normalized = self.pre_attention_norm(x)

    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
        kv_shared_cache,
    )

    if self.post_attention_norm is not None:
      attn_output = self.post_attention_norm(attn_output)

    attn_output += x

    # 2. Feed-forward
    if self.enable_moe:
      outputs = self._forward_moe(attn_output)
    else:
      outputs = self._forward_dense(attn_output)

    outputs += attn_output

    # 3. Per-layer input
    if self.per_layer_input_dim:
      gating_input = outputs
      per_layer_inputs_mapped = self.per_layer_input_gate(
          '...D,DP->...P', gating_input
      )
      per_layer_inputs_mapped = (
          nn.gelu(per_layer_inputs_mapped) * per_layer_input
      )
      per_layer_inputs_mapped = self.per_layer_projection(
          '...P,PD->...D', per_layer_inputs_mapped
      )
      per_layer_inputs_mapped = self.post_per_layer_input_norm(
          per_layer_inputs_mapped
      )
      outputs += per_layer_inputs_mapped

    # 4. Scale
    outputs = outputs * self.skip_scale

    return cache, outputs

  def _forward_dense(self, attn_output: jax.Array) -> jax.Array:
    """Standard FFW forward pass."""
    outputs = self.pre_ffw_norm(attn_output)
    outputs = self.mlp(outputs)
    if self.post_ffw_norm is not None:
      outputs = self.post_ffw_norm(outputs)
    return outputs

  def _forward_moe(self, attn_output: jax.Array) -> jax.Array:
    """MoE FFW forward pass with dense shared + MoE branches."""
    # Dense shared branch (mlp2 in checkpoint)
    dense_out = self.pre_ffw2_norm(attn_output)
    dense_out = self.mlp2(dense_out)
    if self.post_ffw2_norm is not None:
      dense_out = self.post_ffw2_norm(dense_out)

    # MoE branch (mlp in checkpoint)
    moe_in = self.pre_ffw_norm(attn_output)
    moe_out = self.mlp(moe_in)
    if self.post_ffw1_norm is not None:
      moe_out = self.post_ffw1_norm(moe_out)

    # Combine: dense + MoE, then post_ffw_norm
    outputs = dense_out + moe_out
    if self.post_ffw_norm is not None:
      outputs = self.post_ffw_norm(outputs)

    return outputs
