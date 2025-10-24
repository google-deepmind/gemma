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
from typing import List

from flax import linen as nn
from gemma.gm.math import _positional_embeddings
from gemma.gm.nn import _modules
from gemma.gm.nn.gemma3n import _layers
import jax
import jax.numpy as jnp


K_MASK = -2.3819763e38  # Set to a large negative number.
DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0

# A dictionary with the following array shapes as keys:
# v: [batch_size, cache_size, num_heads, head_dim]
# k: [batch_size, cache_size, num_heads, head_dim]
# positions: [batch_size, cache_size]
# end_index: [batch_size]
LayerCache = dict[str, jax.Array]
# Gemma 3n uses layer-shared KV caching (vertical, for current tokens) in
# addition to context KV caching (horizontal, for previous tokens in sampling).
# The data for both is shared and lives on the same LayerCache data structure.
#
# If context caching is disabled (the `cache` parameter is None):
#   - Only layer-sharing is performed.
#   - `cache_size` corresponds to `seq_len`.
#   - `positions` and `end_index` are not used.
# If context caching is enabled (the `cache` parameter is not None):
#   - `cache_size` corresponds to `cache_len`.
#   - `positions` stores token positions for sliding window attention.


def _gaussian_topk(
    inputs: jax.Array, target_sparsity: float | None
) -> jax.Array:
  """Fast topk with gaussian assumption."""
  if target_sparsity is None:
    return inputs
  std_multiplier = jax.scipy.stats.norm.ppf(
      jnp.array(target_sparsity, dtype=jnp.float32)
  ).astype(inputs.dtype)
  # apply Gaussian topk to the last dimension of inputs
  inputs_mean = jnp.mean(inputs, axis=-1, keepdims=True)
  inputs_std = jnp.std(inputs, axis=-1, keepdims=True)
  cutoff_x = inputs_mean + inputs_std * std_multiplier
  return jax.nn.relu(inputs - cutoff_x)


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

    # For models with per-layer inputs, the encoder has additional parameters:
    # * `per_layer_input_projection` and `per_layer_input_embedding_table`:
    #   The first serves to project the main input (derived either from the soft
    #   tokens from the image encoder or the output of the main emebdding table)
    #   into the per_layer_input space and the latter provides distinct per
    #   layer inputs for each text token. These are then merged to provide a
    #   single per_layer_input_dim input for each layer.
    if self.per_layer_input_dim:
      self.per_layer_input_embedding_table = self.param(
          'per_layer_input_embedding',
          nn.initializers.normal(),
          (self.vocab_size, self.num_layers, self.per_layer_input_dim),
      )
      self.per_layer_input_projection = _layers.Einsum(
          (self.embed_dim, self.num_layers, self.per_layer_input_dim),
          w_scale=(float(self.embed_dim) ** -0.5),
      )
      self.per_layer_projection_norm = _layers.RMSNorm(
          scale_plus_one=False,
          guard_against_excess_precision=True,
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

  def encode_per_layer_input(self, x: jax.Array, t: jax.Array) -> jax.Array:
    """Encodes the input tokens.

    Args:
      x: Input shape [seq_len, embed_dim] or [batch_size, seq_len, embed_dim].
      t: Input tokens of shape [seq_len] or [batch_size, seq_len], where
        each token is an integer in [0, vocab_size).

    Returns:
      Encoded input of shape [seq_len, num_layers, per_layer_input_dim] or
      [batch_size, seq_len, num_layers, per_layer_input_dim].
    """
    # Replace tokens outside of the text vocab with zeros.
    t = jnp.where(
        jnp.logical_and(t >= 0, t < self.vocab_size),
        t,
        jnp.zeros_like(t)
    )
    x = self.per_layer_input_projection('...td,dnp->...tnp', x)
    x = self.per_layer_projection_norm(x)
    y = self.per_layer_input_embedding_table[(t,)]
    y *= jnp.sqrt(self.per_layer_input_dim).astype(y.dtype)
    return (x + y) * jax.lax.rsqrt(2.0).astype(x.dtype)


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
  qk_norm_with_scale: bool = True
  use_value_norm: bool = False
  scale_plus_one: bool = True
  guard_against_excess_precision: bool = False

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
      self.query_norm = _layers.RMSNorm(
          with_scale=self.qk_norm_with_scale,
          scale_plus_one=self.scale_plus_one,
          guard_against_excess_precision=self.guard_against_excess_precision,
      )
      self.key_norm = _layers.RMSNorm(
          with_scale=self.qk_norm_with_scale,
          scale_plus_one=self.scale_plus_one,
          guard_against_excess_precision=self.guard_against_excess_precision,
      )

    if self.use_value_norm:
      self.value_norm = _layers.RMSNorm(
          with_scale=False,
          guard_against_excess_precision=self.guard_against_excess_precision,
      )

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
    if self.use_qkv_einsum:
      # [batch_size, seq_len, num_heads, head_dim]
      query_proj, key_proj, value_proj = self.qkv_einsum('BTD,SNDH->SBTNH', x)
      if kv_shared_cache is not None:
        # This cache includes layer-sharing KVs (vertical) and,
        # if context caching is enabled, context KVs (horizontal).
        # [batch_size, cache_size, num_heads, head_dim].
        key_proj = kv_shared_cache['k']
        value_proj = kv_shared_cache['v']
    else:
      query_proj = self.q_einsum('BTD,NDH->BTNH', x)
      if kv_shared_cache is not None:
        # This cache includes layer-sharing KVs (vertical) and,
        # if context caching is enabled, context KVs (horizontal).
        # [batch_size, cache_size, num_heads, head_dim].
        key_proj = kv_shared_cache['k']
        value_proj = kv_shared_cache['v']
      else:
        key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

    if self.use_qk_norm:
      query_proj = self.query_norm(query_proj)
      if kv_shared_cache is None:
        key_proj = self.key_norm(key_proj)

    if self.use_value_norm:
      if kv_shared_cache is None:
        value_proj = self.value_norm(value_proj)

    query_proj = _positional_embeddings.apply_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )
    query_scaled = query_proj * self.query_pre_attn_scalar

    if kv_shared_cache is None:
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

      if kv_shared_cache is None:
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
      sliding_mask = _modules.create_sliding_mask(
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
    probs = jax.nn.softmax(
        (
            _layers.reduce_precision(padded_logits)
            if self.guard_against_excess_precision
            else padded_logits
        ),
        axis=-1,
    ).astype(key_proj.dtype)

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

    # Always cache the layer-sharing KV.
    # This also includes the context KV if cache is not None.
    # i.e. cache_size can be == seq_len or == cache_len if cache is not None.
    new_cache = {
        # [batch_size, cache_size, num_heads, head_dim]
        'v': value_proj,
        # [batch_size, cache_size, num_heads, head_dim]
        'k': key_proj,
    }
    # Remaining keys for context KV.
    if cache is not None:
      seq_len = x.shape[1]
      # [batch_size]
      new_cache['end_index'] = cache['end_index'] + seq_len
      # [batch_size, cache_size]
      new_cache['positions'] = cache_positions  # pylint: disable=undefined-variable

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
  activation_sparsity: float | None = None

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
    if self.activation_sparsity is not None and self.activation_sparsity > 0:
      sparse_gate = _gaussian_topk(gate[..., 0, :], self.activation_sparsity)
      activations = nn.gelu(sparse_gate) * gate[..., 1, :]
    else:
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


class PerLayerMapping(nn.Module):
  """Per-layer input module."""
  embed_dim: int
  per_layer_input_dim: int = 0

  def setup(self):
    self.post_per_layer_input_norm = _layers.RMSNorm(
        scale_plus_one=False,
        guard_against_excess_precision=True,
    )
    self.per_layer_input_gate = _layers.Einsum(
        shape=(self.embed_dim, self.per_layer_input_dim),
    )
    self.per_layer_projection = _layers.Einsum(
        shape=(self.per_layer_input_dim, self.embed_dim),
    )

  def __call__(
      self,
      x: jax.Array,
      per_layer_input: jax.Array,
  ) -> jax.Array:
    """Map per-layer inputs to the embedding space.

    Args:
      x: Input sequence of shape [batch_size, seq_len, embed_dim].
      per_layer_input: Per-layer input of shape [batch_size, seq_len,
        per_layer_input_dim].

    Returns:
      Output sequence of shape [batch_size, seq_len, embed_dim].
    """

    x = self.per_layer_input_gate('...D,DP->...P', x)
    x = nn.gelu(x) * per_layer_input
    x = self.per_layer_projection('...P,PD->...D', x)
    x = self.post_per_layer_input_norm(x)
    return x


class AltUp(nn.Module):
  """AltUp module."""

  d_model: int
  num_altup_inputs: int = 4
  active_idx: int = 0
  num_modalities: int = 4
  coef_clip: float | None = None
  router_norm_layer: nn.Module | None = None

  def setup(self):
    self.correction_coefs = self.param(
        'correction_coefs',
        nn.initializers.normal(stddev=1e-2),
        (self.num_modalities, self.num_altup_inputs))
    self.prediction_coefs = self.param(
        'prediction_coefs',
        nn.initializers.normal(stddev=1e-4),
        (self.num_modalities, self.num_altup_inputs, self.num_altup_inputs))
    self.correct_output_scale = self.param(
        'correct_output_scale',
        nn.initializers.ones,
        (self.d_model,))
    self.modality_router = _layers.Einsum(
        shape=(self.d_model, self.num_modalities))

  def compute_router_modalities(self, x: jax.Array) -> jax.Array:
    x = self.router_norm_layer(x) if self.router_norm_layer is not None else x
    x *= self.d_model ** -1.0
    eq = '...F,FD->...D'
    modalities = jnp.tanh(
        self.modality_router(eq, x).astype(jnp.float32)).astype(jnp.float32)
    return modalities

  def predict(self, x: List[jax.Array]) -> List[jax.Array]:
    modalities = self.compute_router_modalities(x[self.active_idx])
    outputs = [jnp.asarray(0.0)] * self.num_altup_inputs
    pred_coefs = self.prediction_coefs.astype(jnp.float32)
    if self.coef_clip is not None:
      pred_coefs = jnp.clip(pred_coefs, -self.coef_clip, self.coef_clip)
    all_coefs = jnp.einsum('...p,pij->...ij', modalities, pred_coefs)
    for i in range(self.num_altup_inputs):
      out = 0.0
      for j in range(self.num_altup_inputs):
        coef = jnp.expand_dims(all_coefs[..., i, j], axis=-1)
        out += coef * x[j]
      outputs[i] = x[i] + out
    outputs[self.active_idx] = outputs[self.active_idx].astype(
        x[self.active_idx].dtype)
    return outputs

  def correct(
      self,
      predictions: List[jax.Array],
      activated: jax.Array,
  ) -> List[jax.Array]:
    modalities = self.compute_router_modalities(activated)
    innovation = activated - predictions[self.active_idx]
    corrected = [jnp.asarray(0.0)] * self.num_altup_inputs
    correct_coefs = self.correction_coefs.astype(jnp.float32)
    if self.coef_clip is not None:
      correct_coefs = jnp.clip(correct_coefs, -self.coef_clip, self.coef_clip)
    all_coefs = jnp.einsum('...p,pi->...i', modalities, correct_coefs) + 1
    all_coefs = all_coefs.astype(jnp.float32)

    for i in range(self.num_altup_inputs):
      coef = jnp.expand_dims(all_coefs[..., i], axis=-1)
      coef = _layers.reduce_precision(coef)
      corrected[i] = (
          predictions[i] + coef * innovation).astype(activated.dtype)
    return corrected

  def scale_corrected_output(self, corrected: jax.Array) -> jax.Array:
    return corrected * self.correct_output_scale

  def __call__(
      self,
      inputs: List[jax.Array],
      activated: jax.Array,
  ) -> List[jax.Array]:
    predicted = self.predict(inputs)
    activated = self.scale_corrected_output(activated)
    return self.correct(predicted, activated)


class Laurel(nn.Module):
  """Laurel module."""

  d_model: int
  rank: int = 64

  def setup(self):
    self.linear_left = _layers.Einsum(
        (self.d_model, self.rank),
        'w',
    )
    self.linear_right = _layers.Einsum(
        (self.rank, self.d_model),
        'w',
    )

  def __call__(
      self,
      x: jax.Array) -> jax.Array:
    x = self.linear_left('btd,dr->btr', x)
    x = self.linear_right('btr,rd->btd', x)
    return x


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
  qk_norm_with_scale: bool = True
  use_value_norm: bool = False
  use_altup: bool = False
  num_altup_inputs: int = 4
  altup_coef_clip: float | None = None
  activation_sparsity: float | None = None
  use_laurel: bool = False
  laurel_rank: int = 64
  per_layer_input_dim: int = 0
  kv_cache_sharing_pattern: int | None = None
  scale_plus_one: bool = True
  guard_against_excess_precision: bool = False

  def setup(self):
    self.pre_attention_norm = _layers.RMSNorm(
        scale_plus_one=self.scale_plus_one,
        guard_against_excess_precision=self.guard_against_excess_precision,
    )

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
        qk_norm_with_scale=self.qk_norm_with_scale,
        use_value_norm=self.use_value_norm,
        scale_plus_one=self.scale_plus_one,
        guard_against_excess_precision=self.guard_against_excess_precision,
    )

    self.post_attention_norm = None
    if self.use_post_attn_norm:
      self.post_attention_norm = _layers.RMSNorm(
          scale_plus_one=self.scale_plus_one,
          guard_against_excess_precision=self.guard_against_excess_precision,
      )

    self.pre_ffw_norm = _layers.RMSNorm(
        scale_plus_one=self.scale_plus_one,
        guard_against_excess_precision=self.guard_against_excess_precision,
    )

    self.mlp = FeedForward(
        features=self.embed_dim,
        hidden_dim=self.hidden_dim,
        transpose_gating_einsum=self.transpose_gating_einsum,
        activation_sparsity=self.activation_sparsity,
    )

    self.post_ffw_norm = None
    if self.use_post_ffw_norm:
      self.post_ffw_norm = _layers.RMSNorm(
          scale_plus_one=self.scale_plus_one,
          guard_against_excess_precision=self.guard_against_excess_precision,
      )

    if self.use_altup:
      self.altup = AltUp(
          d_model=self.embed_dim,
          num_altup_inputs=self.num_altup_inputs,
          coef_clip=self.altup_coef_clip,
          router_norm_layer=_layers.RMSNorm(
              scale_plus_one=self.scale_plus_one,
              guard_against_excess_precision=self.guard_against_excess_precision,
          ),
      )

    if self.use_laurel:
      self.laurel = Laurel(d_model=self.embed_dim, rank=self.laurel_rank)
      self.post_laurel_norm = _layers.RMSNorm(
          scale_plus_one=self.scale_plus_one,
          guard_against_excess_precision=self.guard_against_excess_precision,
      )

    if self.per_layer_input_dim:
      self.per_layer_mapping = PerLayerMapping(
          embed_dim=self.embed_dim,
          per_layer_input_dim=self.per_layer_input_dim,
      )

  def __call__(
      self,
      x: jax.Array | List[jax.Array],
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
      per_layer_input: jax.Array | None = None,
      kv_shared_cache: LayerCache | None = None,
  ) -> tuple[LayerCache | None, jax.Array | List[jax.Array]]:
    """Applies the block to the inputs.

    Args:
      x: Input sequence of shape [batch_size, seq_len, embed_dim] or a list of
        input sequences.
      segment_pos: Input absolute positions of shape [batch_size, seq_len].
      cache: KV cache or None.
      attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].
      per_layer_input: Per-layer input of shape [batch_size, seq_len,
        per_layer_input_dim].
      kv_shared_cache: Cache for shared KV layers.

    Returns:
      cache: Updated attention KV cache.
      outputs: Output sequence of shape [batch_size, seq_len, embed_dim] or a
        list of output sequences.
    """
    predictions = [0.0]
    if self.use_altup:
      predictions = self.altup.predict(x)
      x = predictions[self.altup.active_idx]
    inputs_normalized = self.pre_attention_norm(x)

    # attn_output.shape = [batch_size, seq_len, embed_dim]
    # cache["k"].shape = [batch_size, cache_size, num_heads, head_dim]
    # cache["v"].shape = [batch_size, cache_size, num_heads, head_dim]
    # cache["end_index"].shape = [batch_size]
    # cache["positions"].shape = [batch_size, cache_size]
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
        kv_shared_cache,
    )

    laurel_out_normed = 0.0
    if self.use_laurel:
      laurel_x = self.laurel(inputs_normalized)
      laurel_x_normed = self.post_laurel_norm(laurel_x)
      laurel_out_normed = (inputs_normalized + laurel_x_normed)

    if self.post_attention_norm is not None:
      attn_output = self.post_attention_norm(attn_output)

    attn_output += x

    if self.use_laurel:
      attn_output = (attn_output + laurel_out_normed) * jax.lax.rsqrt(2.0)

    outputs = self.pre_ffw_norm(attn_output)

    outputs = self.mlp(outputs)

    if self.post_ffw_norm is not None:
      outputs = self.post_ffw_norm(outputs)

    outputs += attn_output

    if self.use_altup:
      outputs = self.altup.correct(predictions, outputs)

    if self.per_layer_input_dim:
      gating_input = (
          outputs
          if not self.use_altup
          else self.altup.scale_corrected_output(outputs[0])
      )
      per_layer_inputs_mapped = self.per_layer_mapping(
          gating_input, per_layer_input
      )
      if self.use_altup:  # outputs[0] is altup output, outputs[1:] inc PLI.
        for i in range(1, len(outputs)):
          outputs[i] += per_layer_inputs_mapped
      else:
        outputs += per_layer_inputs_mapped

    return cache, outputs
