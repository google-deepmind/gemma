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

"""Gemma transformer with cross attention."""

import dataclasses
from typing import Any, Iterable, Sequence

import flax
from flax import linen as nn
from gemma.gm import math as positional_embeddings
from gemma.gm import nn as layers
from gemma.gm.nn import config as gemma_config
from gemma.gm.utils import _dtype_params
import jax
import jax.numpy as jnp
from kauldron.typing import Array, typechecked  # pylint: disable=g-multiple-import,g-importing-member

K_MASK = -2.3819763e38  # Set to a large negative number.
DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0

LayerCache = dict[str, jax.Array]
Cache = dict[str, LayerCache]

Embedder = layers.Embedder
AttentionType = layers.AttentionType
RMSNorm = layers.RMSNorm
Einsum = layers.Einsum
FeedForward = layers.FeedForward

QueryPreAttentionNormalisation = gemma_config.QueryPreAttentionNormalisation


def get_sliding_mask(
    attn_mask: jax.Array,
    sliding_window_size: int | None,
    bidirectional: bool = False,
) -> jax.Array:
  """Computes sliding attention mask.

  Args:
    attn_mask: base attention mask. If bidirectional is False, then it should be
      causal mask.
    sliding_window_size: context size in ONE direction from the current token.
    bidirectional: if True, sliding_window is extended into both sides from the
      current token.

  Returns:
    Updated attention mask.
  """
  if sliding_window_size is None or attn_mask.shape[-1] <= sliding_window_size:
    return attn_mask
  else:
    mask = jnp.triu(attn_mask, 1 - sliding_window_size)
    if bidirectional:
      # If attention is bidirectional, mask is not causal, so set elements
      # above shifted positive diagonal to zero to limit the attention to the
      # sliding_window_size.
      mask = jnp.tril(mask, sliding_window_size - 1)
    return mask


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
  enable_cross_attention: bool = False
  bidirectional: bool = False
  kv_features: int | None = None

  @property
  def use_qkv_einsum(self):
    return (
        not self.enable_cross_attention and self.num_kv_heads == self.num_heads
    )

  @property
  def use_gqa(self):
    return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1

  def setup(self):
    if self.enable_cross_attention and self.attn_type != AttentionType.GLOBAL:
      raise NotImplementedError(
          'Cross-attention supports only global attention,'
          f' {self.attn_type} was passed instead.'
      )

    if (
        self.attn_type == AttentionType.LOCAL_SLIDING
        and self.sliding_window_size is None
    ):
      raise ValueError(
          'sliding_window_size must be set if local sliding attention is used.'
      )

    self.attn_vec_einsum = Einsum(
        shape=(self.num_heads, self.head_dim, self.features),
    )

    if self.use_qkv_einsum:   # Only for self-attention.
      self.qkv_einsum = Einsum(
          shape=(3, self.num_heads, self.features, self.head_dim),
      )
    else:
      self.q_einsum = Einsum(
          shape=(self.num_heads, self.features, self.head_dim),
      )
      kv_features = self.kv_features or self.features
      self.kv_einsum = Einsum(
          shape=(2, self.num_kv_heads, kv_features, self.head_dim),
      )

  def _maybe_key_rope(
      self,
      key_proj: jax.Array,
      segment_pos_kv: jax.Array | None,
  ):
    """Applies rope positional embeddings to the key."""

    # Apply rope embeddings if not cross attention.
    if segment_pos_kv is not None and not self.enable_cross_attention:
      key_proj = positional_embeddings.apply_rope(
          key_proj,
          segment_pos_kv,
          base_frequency=self.rope_base_frequency,
          scale_factor=self.rope_scale_factor,
      )

    return key_proj

  def _maybe_query_rope(
      self,
      query_proj: jax.Array,
      segment_pos_q: jax.Array | None,
  ):
    """Applies rope positional embeddings to the query."""

    # Apply rope embeddings if not cross attention.
    if segment_pos_q is not None and not self.enable_cross_attention:
      query_proj = positional_embeddings.apply_rope(
          query_proj,
          segment_pos_q,
          base_frequency=self.rope_base_frequency,
          scale_factor=self.rope_scale_factor,
      )

    query_scaled = query_proj * self.query_pre_attn_scalar
    return query_scaled

  @typechecked
  def __call__(
      self,
      x_q: Array['B T D'],
      x_kv: Array['B S D2'] | None,
      segment_pos_q: Array['B T'],
      segment_pos_kv: Array['B S'] | None,
      cache: LayerCache | None,
      attn_mask: Array['#B T S'],
  ) -> tuple[LayerCache | None, Array['B T D']]:
    """Applies multi-head attention to the inputs.

    Args:
      x_q: Query input of shape [batch_size, q_len, embed_dim].
      x_kv: Key/Value input of shape [batch_size, kv_len, embed_dim]. For self
        attention, x_kv=x_q and q_len=kv_len.
      segment_pos_q: Input absolute positions of shape [batch_size, q_len].
      segment_pos_kv: Input absolute positions of shape [batch_size, kv_len].
      cache: KV cache or None.
      attn_mask: Attention mask of shape [batch_size, q_len, kv_len].

    Returns:
      cache: Updated attention KV cache.
      outputs: Output sequence of shape [batch_size, q_len, embed_dim].
    """
    if self.use_qkv_einsum:
      # [batch_size, q_len, num_heads, head_dim]
      query_proj, key_proj, value_proj = self.qkv_einsum('BTD,SNDH->SBTNH', x_q)
      segment_pos_kv = segment_pos_q

      key_proj = self._maybe_key_rope(key_proj, segment_pos_kv)
    else:
      query_proj = self.q_einsum('BTD,NDH->BTNH', x_q)
      if x_kv is None:
        x_kv = x_q
        segment_pos_kv = segment_pos_q
      if cache is None or (
          not self.enable_cross_attention or
          # We use the dimension of the cache to distinguish between the initial
          # cache and the non-initial cache.
          cache['is_initial_cache'].ndim == 1
      ):
        key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x_kv)
        key_proj = self._maybe_key_rope(key_proj, segment_pos_kv)
      else:
        key_proj, value_proj = cache['k'], cache['v']

    query_scaled = self._maybe_query_rope(query_proj, segment_pos_q)

    # Combine current KV with the past for self attention.
    if cache is not None and not self.enable_cross_attention:
      end_index = cache['end_index'][0]
      cache_size = cache['v'].shape[1]
      slice_indices = (0, end_index % cache_size, 0, 0)

      # [batch_size, cache_size, num_heads, head_dim]
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'], value_proj, slice_indices,
      )

      # [batch_size, cache_size, num_heads, head_dim]
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
      # [batch_size, q_len, num_heads, kv_len]
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

    if self.attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / self.attn_logits_soft_cap)
      logits = logits * self.attn_logits_soft_cap

    # Apply sliding window.
    if self.attn_type == AttentionType.LOCAL_SLIDING:
      sliding_window_size = self.sliding_window_size
    else:
      sliding_window_size = None
    attn_mask = get_sliding_mask(
        attn_mask, sliding_window_size, self.bidirectional
    )
    # [batch_size, q_len, num_heads, kv_len]
    padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)

    # Multi-head attention matrices.
    # [batch_size, q_len, num_heads, kv_len]
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
      # [batch_size, q_len, num_heads, head_dim]
      encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)

    # [batch_size, q_len, features]
    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)

    # Mask out the output with no attended token.
    no_attended_tokens = jnp.all(attn_mask == 0, axis=-1, keepdims=True)
    attn_output = jnp.where(
        no_attended_tokens, jnp.zeros_like(attn_output), attn_output
    )

    # Update cache.
    if cache is None:
      new_cache = None
    else:
      new_cache = {
          # [batch_size, kv_len, num_heads, head_dim]
          'v': value_proj,
          # [batch_size, kv_len, num_heads, head_dim]
          'k': key_proj,
          # Cache updated thus not initial cache anymore.
          'is_initial_cache': jnp.zeros((1, 1)),
      }

      if self.enable_cross_attention:
        # Cross-attention cache is not updated.
        q_len = 0
      else:
        q_len = x_q.shape[1]

      new_cache.update({
          # [batch_size]
          'end_index': cache['end_index'] + q_len,
      })
    return new_cache, attn_output  # pytype: disable=bad-return-type

  @classmethod
  def init_cache(
      cls,
      seq_len: int,
      num_heads: int,
      head_dim: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> LayerCache:
    del cls  # not used
    return {
        'v': jnp.zeros((batch_size, seq_len, num_heads, head_dim), dtype=dtype),
        'k': jnp.zeros((batch_size, seq_len, num_heads, head_dim), dtype=dtype),
        'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
        # A flag used to indicate whether the cache is the initial cache.
        # For initial cache, we need to update the cross-attention cache.
        # Not used for self-attention cache.
        # ndim=1 => initial cache
        # ndim=2 => not initial cache
        'is_initial_cache': jnp.zeros((1,)),
    }


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
  enable_cross_attention: bool = False
  bidirectional: bool = False
  kv_embed_dim: int | None = None

  def setup(self):
    self.pre_attention_norm = RMSNorm()

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
        enable_cross_attention=False,
        bidirectional=self.bidirectional,
    )

    self.post_attention_norm = None
    if self.use_post_attn_norm:
      self.post_attention_norm = RMSNorm()

    if self.enable_cross_attention:
      self.pre_cross_attention_norm = RMSNorm()
      self.cross_attn = Attention(
          num_heads=self.num_heads,
          features=self.embed_dim,
          kv_features=self.kv_embed_dim,
          head_dim=self.head_dim,
          num_kv_heads=self.num_kv_heads,
          # Cross-attention only supports global attention.
          attn_type=AttentionType.GLOBAL,
          sliding_window_size=None,
          query_pre_attn_scalar=self.query_pre_attn_scalar,
          rope_base_frequency=self.rope_base_frequency,
          rope_scale_factor=self.rope_scale_factor,
          attn_logits_soft_cap=self.attn_logits_soft_cap,
          enable_cross_attention=True,
          bidirectional=True,
      )

      self.post_cross_attention_norm = None
      if self.use_post_attn_norm:
        self.post_cross_attention_norm = RMSNorm()

    self.pre_ffw_norm = RMSNorm()

    self.mlp = FeedForward(
        features=self.embed_dim,
        hidden_dim=self.hidden_dim,
        transpose_gating_einsum=self.transpose_gating_einsum,
    )

    self.post_ffw_norm = None
    if self.use_post_ffw_norm:
      self.post_ffw_norm = RMSNorm()

  @typechecked
  def __call__(
      self,
      x: Array['B T D'],
      segment_pos: Array['B T'],
      self_attn_cache: LayerCache | None,
      self_attn_mask: Array['#B T _'],
      cross_attn_kv: Array['B S D2'] | None = None,
      cross_attn_mask: Array['B #T S'] | None = None,
      cross_attn_cache: LayerCache | None = None,
  ) -> tuple[LayerCache | None, LayerCache | None, Array['B T D']]:
    """Applies the block to the inputs.

    Args:
      x: Query input of shape [batch_size, q_len, embed_dim].
      segment_pos: Input absolute positions of shape [batch_size, q_len].
      self_attn_cache: KV cache or None.
      self_attn_mask: Attention mask of shape [batch_size, q_len, q_len].
      cross_attn_kv: Key/value input of shape [batch_size, kv_len, embed_dim].
      cross_attn_mask: Cross-attention mask of shape [batch_size, q_len,
        kv_len].
      cross_attn_cache: Cross-attention KV cache or None.

    Returns:
      self_attn_cache: Updated self-attention KV cache.
      cross_attn_cache: Updated cross-attention KV cache.
      outputs: Output sequence of shape [batch_size, q_len, embed_dim].
    """
    inputs_normalized = self.pre_attention_norm(x)

    # Self attention.
    # attn_output.shape = [batch_size, q_len, embed_dim]
    # cache["k"].shape = [batch_size, kv_len, num_heads, head_dim]
    # cache["v"].shape = [batch_size, kv_len, num_heads, head_dim]
    # cache["end_index"].shape = [batch_size]
    self_attn_cache, attn_output = self.attn(
        x_q=inputs_normalized,
        x_kv=None,
        segment_pos_q=segment_pos,
        segment_pos_kv=None,
        cache=self_attn_cache,
        attn_mask=self_attn_mask,
    )

    if self.post_attention_norm is not None:
      attn_output = self.post_attention_norm(attn_output)

    attn_output += x

    # Cross attention.
    if self.enable_cross_attention:
      inputs_normalized = self.pre_cross_attention_norm(attn_output)
      cross_attn_cache, cross_attn_output = self.cross_attn(
          x_q=inputs_normalized,
          x_kv=cross_attn_kv,
          segment_pos_q=segment_pos,
          segment_pos_kv=None,
          cache=cross_attn_cache,
          attn_mask=cross_attn_mask,
      )
      if self.post_cross_attention_norm is not None:
        cross_attn_output = self.post_cross_attention_norm(cross_attn_output)
      attn_output += cross_attn_output

    # Feed forward.
    outputs = self.pre_ffw_norm(attn_output)

    outputs = self.mlp(outputs)

    if self.post_ffw_norm is not None:
      outputs = self.post_ffw_norm(outputs)
    outputs += attn_output

    return self_attn_cache, cross_attn_cache, outputs


@flax.struct.dataclass
class TransformerOutput:
  """Transformer output."""

  activations: Sequence[jax.Array]
  cache: Cache | None


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
  """Configuration for the gemma transformer."""

  num_layers: int
  num_embed: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  final_logit_softcap: float | None
  use_post_attn_norm: bool
  use_post_ffw_norm: bool
  attention_types: Iterable[AttentionType]
  query_pre_attn_norm: QueryPreAttentionNormalisation = (
      QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
  )
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  transpose_gating_einsum: bool = False
  local_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
  global_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
  local_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
  global_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
  mm_extra_vocab_size: int = 0
  enable_cross_attention: bool = False
  bidirectional: bool = False
  kv_embed_dim: int | None = None

  def query_pre_attn_scalar(self) -> float:
    """Returns the scalar to multiply the query by before attention."""
    match self.query_pre_attn_norm:
      case QueryPreAttentionNormalisation.BY_EMBED_DIM_DIV_NUM_HEADS:
        return self.embed_dim // self.num_heads
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS:  # pylint: disable=line-too-long
        return (self.embed_dim // self.num_heads) ** -0.5
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM | _:
        return self.head_dim**-0.5

  def init_cache(
      self,
      batch_size: int,
      prefill_length: int,
      generation_length: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> Cache:
    """Initializes a new Transformer cache."""
    cache = {
        # Decoder self-attention cache.
        f'layer_{i}': Attention.init_cache(
            generation_length,
            self.num_kv_heads,
            self.head_dim,
            batch_size,
            dtype,
        )
        for i in range(self.num_layers)
    }
    # The cross attention cache is initialized with None, which will be
    # updated in the forward pass.
    if self.enable_cross_attention:
      cache.update({
          # Decoder cross-attention cache.
          f'cross_layer_{i}': Attention.init_cache(
              prefill_length,
              self.num_kv_heads,
              self.head_dim,
              batch_size,
              dtype,
          )
          for i in range(self.num_layers)
      })
    return cache

  def make(self, name: str = 'transformer', **kwargs: Any) -> 'Transformer':
    """Make Transformer class from the configuration."""
    return Transformer(self, name=name, **kwargs)


class Transformer(nn.Module):
  """Gemma transformer."""

  config: TransformerConfig
  dtype: jnp.dtype = jnp.bfloat16

  def setup(self):
    self.embedder = Embedder(
        vocab_size=self.config.num_embed,
        embed_dim=self.config.embed_dim
    )

    self.blocks = [
        Block(
            name=f'layer_{i}',
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            embed_dim=self.config.embed_dim,
            kv_embed_dim=self.config.kv_embed_dim,
            head_dim=self.config.head_dim,
            hidden_dim=self.config.hidden_dim,
            sliding_window_size=self.config.sliding_window_size,
            use_post_attn_norm=self.config.use_post_attn_norm,
            use_post_ffw_norm=self.config.use_post_ffw_norm,
            attn_logits_soft_cap=self.config.attn_logits_soft_cap,
            attn_type=attn_type,
            query_pre_attn_scalar=self.config.query_pre_attn_scalar(),
            transpose_gating_einsum=self.config.transpose_gating_einsum,
            rope_base_frequency=self.config.local_base_frequency
            if attn_type == AttentionType.LOCAL_SLIDING
            else self.config.global_base_frequency,
            rope_scale_factor=self.config.local_scale_factor
            if attn_type == AttentionType.LOCAL_SLIDING
            else self.config.global_scale_factor,
            enable_cross_attention=self.config.enable_cross_attention,
            bidirectional=self.config.bidirectional,
        )
        for i, attn_type in zip(
            range(self.config.num_layers), self.config.attention_types
        )
    ]
    self.final_norm = RMSNorm()

  def decode(self, activations: jax.Array) -> jax.Array:
    logits = self.embedder.decode(activations)
    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap
    return logits

  @typechecked
  def __call__(
      self,
      tokens: Array['B T'],
      positions: Array['B T'],
      self_attn_mask: Array['#B T _'],
      cache: Cache | None = None,
      cross_attn_kv: Array['B S D2'] | None = None,
      cross_attn_mask: Array['B #T S'] | None = None,
  ) -> TransformerOutput:
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache. When cross_attn_kv is provided, the transformer will run in
    cross-attention mode.

    Args:
      tokens: Input sequence of tokens of shape [batch_size, q_len].
      positions: Input absolute positions of shape [batch_size, q_len].
      self_attn_mask: Self-attention mask of shape [batch_size,
        q_len, q_len].
      cache: Joint self-attention and cross-attention cache.
      cross_attn_kv: Key/value input of shape [batch_size, kv_len, d_model].
      cross_attn_mask: Cross attention mask of shape [batch_size, q_len,
        kv_len].

    Returns:
      output: output activations of shape [batch_size, q_len, d_model], and
        potentially updated cache.
    """
    with _dtype_params.initialize_param_with_dtype(
        self.dtype,
        exclude=[
            # Skip the LoRA params
            'lora',
        ],
    ):
      x = self.embedder.encode(tokens)

      activations = []
      for i, block in enumerate(self.blocks):
        activations.append(x)

        layer_name = f'layer_{i}'
        self_attn_layer_cache, cross_attn_layer_cache = None, None
        if cache:
          self_attn_layer_cache = cache[layer_name]
          if self.config.enable_cross_attention:
            cross_attn_layer_cache = cache[f'cross_{layer_name}']

        self_attn_layer_cache, cross_attn_layer_cache, x = block(
            x,
            segment_pos=positions,
            self_attn_cache=self_attn_layer_cache,
            self_attn_mask=self_attn_mask,
            cross_attn_kv=cross_attn_kv,
            cross_attn_mask=cross_attn_mask,
            cross_attn_cache=cross_attn_layer_cache,
        )

        # Update cache.
        if cache:
          cache[layer_name] = self_attn_layer_cache  # pytype: disable=container-type-mismatch
          if self.config.enable_cross_attention:
            cache[f'cross_{layer_name}'] = cross_attn_layer_cache  # pytype: disable=container-type-mismatch

      x = self.final_norm(x)
      activations.append(x)

      return TransformerOutput(
          activations=activations,
          cache=cache,
      )
