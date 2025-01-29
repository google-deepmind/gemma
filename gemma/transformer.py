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

"""Gemma transformer."""

import dataclasses
import enum
from typing import Iterable

from flax import linen as nn
from gemma import layers
from gemma import modules
from gemma import params as params_lib
import jax
import jax.numpy as jnp

Cache = dict[str, modules.LayerCache]


class QueryPreAttentionNormalisation(enum.Enum):
  """Initialization strategy."""

  # Whether to scale the query by 1/sqrt(head_dim)
  BY_ONE_OVER_SQRT_HEAD_DIM = enum.auto()

  # Whether to scale the query by `embed_dim // num_heads`
  BY_EMBED_DIM_DIV_NUM_HEADS = enum.auto()

  # Whether to scale the query by `1/sqrt(embed_dim // num_heads)`
  BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS = enum.auto()


_NUM_LAYERS_GEMMA_2B = 18
_NUM_LAYERS_GEMMA_7B = 28
_NUM_LAYERS_GEMMA2_2B = 26
_NUM_LAYERS_GEMMA2_9B = 42
_NUM_LAYERS_GEMMA2_27B = 46


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
  attention_types: Iterable[modules.AttentionType]
  max_cache_length: int | None = 1024
  query_pre_attn_norm: QueryPreAttentionNormalisation = (
      QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
  )
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  transpose_gating_einsum: bool = False
  use_qk_norm: bool = False

  def query_pre_attn_scalar(self) -> float:
    """Returns the scalar to multiply the query by before attention."""
    match self.query_pre_attn_norm:
      case QueryPreAttentionNormalisation.BY_EMBED_DIM_DIV_NUM_HEADS:
        return self.embed_dim // self.num_heads
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS:  # pylint: disable=line-too-long
        return (self.embed_dim // self.num_heads)**-0.5
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM | _:
        return self.head_dim**-0.5

  @classmethod
  def from_params(
      cls, params: params_lib.Params, cache_size: int | None = 1024
  ) -> 'TransformerConfig':
    """Creates a TransformerConfig from loaded parameters.

    Args:
      params: Model parameters
      cache_size: Number of tokens to cache

    Returns:
      TransformerConfig.
    """
    layer_names = [
        name for name in params['transformer'].keys() if 'layer' in name
    ]
    layer_names = [name.replace('layer_', '') for name in layer_names]
    num_layers = max([int(layer) for layer in layer_names]) + 1

    if num_layers == _NUM_LAYERS_GEMMA_2B:
      return cls.gemma_2b(cache_size)
    if num_layers == _NUM_LAYERS_GEMMA_7B:
      return cls.gemma_7b(cache_size)
    if num_layers == _NUM_LAYERS_GEMMA2_2B:
      return cls.gemma2_2b(cache_size)
    if num_layers == _NUM_LAYERS_GEMMA2_9B:
      return cls.gemma2_9b(cache_size)
    if num_layers == _NUM_LAYERS_GEMMA2_27B:
      return cls.gemma2_27b(cache_size)

    raise ValueError(
        'Params are not a Gemma 2b, 7b, or Gemma 2 2b, 9b, or 27b variant.'
    )

  @classmethod
  def gemma_2b(cls, cache_size: int | None):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA_2B,
        num_embed=256128,
        embed_dim=2048,
        hidden_dim=16384,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        final_logit_softcap=None,
        attention_types=(modules.AttentionType.GLOBAL,) * _NUM_LAYERS_GEMMA_2B,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        max_cache_length=cache_size,
    )

  @classmethod
  def gemma_7b(cls, cache_size: int | None):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA_7B,
        num_embed=256128,
        embed_dim=3072,
        hidden_dim=24576,
        num_heads=16,
        head_dim=256,
        num_kv_heads=16,
        final_logit_softcap=None,
        attention_types=(modules.AttentionType.GLOBAL,) * _NUM_LAYERS_GEMMA_7B,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        max_cache_length=cache_size,
    )

  @classmethod
  def gemma2_2b(cls, cache_size: int | None):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA2_2B,
        num_embed=256128,
        embed_dim=2304,
        hidden_dim=9216,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        final_logit_softcap=30.0,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(_NUM_LAYERS_GEMMA2_2B / 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        max_cache_length=cache_size,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  @classmethod
  def gemma2_9b(cls, cache_size: int | None):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA2_9B,
        num_embed=256128,
        embed_dim=3584,
        hidden_dim=14336,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        final_logit_softcap=30.0,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(_NUM_LAYERS_GEMMA2_9B / 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        max_cache_length=cache_size,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
        transpose_gating_einsum=True,
    )

  @classmethod
  def gemma2_27b(cls, cache_size: int | None):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA2_27B,
        num_embed=256128,
        embed_dim=4608,
        hidden_dim=36864,
        num_heads=32,
        head_dim=128,
        num_kv_heads=16,
        final_logit_softcap=30.0,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(_NUM_LAYERS_GEMMA2_27B / 2),
        max_cache_length=cache_size,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
        transpose_gating_einsum=True,
    )

  def init_cache(
      self,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
      *,
      cache_length: int | None = None,
  ) -> Cache:
    """Initializes a new Transformer cache."""
    cache_length = cache_length or self.max_cache_length
    if cache_length is None:
      raise ValueError(
          'Missing `cache_length=` kwarg when calling `init_cache()`.'
      )
    cache = {
        f'layer_{i}': modules.Attention.init_cache(
            cache_length,
            self.num_kv_heads,
            self.head_dim,
            batch_size,
            dtype,
        )
        for i in range(self.num_layers)
    }
    return cache


class Transformer(nn.Module):
  """Gemma transformer."""

  config: TransformerConfig

  def setup(self):
    self.embedder = modules.Embedder(
        vocab_size=self.config.num_embed,
        embed_dim=self.config.embed_dim,
    )

    self.blocks = [
        modules.Block(
            name=f'layer_{i}',
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            embed_dim=self.config.embed_dim,
            head_dim=self.config.head_dim,
            hidden_dim=self.config.hidden_dim,
            sliding_window_size=self.config.sliding_window_size,
            use_post_attn_norm=self.config.use_post_attn_norm,
            use_post_ffw_norm=self.config.use_post_ffw_norm,
            attn_logits_soft_cap=self.config.attn_logits_soft_cap,
            attn_type=attn_type,
            query_pre_attn_scalar=self.config.query_pre_attn_scalar(),
            transpose_gating_einsum=self.config.transpose_gating_einsum,
            use_qk_norm=self.config.use_qk_norm,
        )
        for i, attn_type in zip(
            range(self.config.num_layers), self.config.attention_types
        )
    ]
    self.final_norm = layers.RMSNorm()

  def __call__(
      self,
      last_tokens: jax.Array,  # [B, L]
      positions: jax.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jax.Array,  # [B, L, L']
  ) -> tuple[jax.Array, Cache | None]:
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      last_tokens: input sequence of tokens.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    x = self.embedder.encode(last_tokens)
    for i, block in enumerate(self.blocks):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = block(
          x,
          positions,
          layer_cache,
          attention_mask,
      )
      if cache is not None:
        cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)
    logits = self.embedder.decode(x)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return logits, cache  # pytype: disable=bad-return-type


def make_causal_attn_mask(
    input_mask: jax.Array,  # Shape [B, L]
) -> jax.Array:
  """Makes a causal attention mask.

  I.e., as in middle diagram of Figure 3 in https://arxiv.org/pdf/1910.10683.

  Args:
    input_mask: Input mask for the input. True for non-padded tokens only, else
      False.

  Returns:
    Attention mask of shape [B, L, L] (where B=batch dim and L=sequence dim).
  """
  if len(input_mask.shape) != 2:
    raise ValueError(
        f'Input mask must be 2D (shape [B, L]), but got {input_mask.shape}.'
    )
  seq_len = input_mask.shape[-1]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
  attn_mask = input_mask[..., None, :]
  attn_mask *= causal_mask[None, ...]
  return attn_mask


def make_causal_with_prefix_attn_mask(
    input_mask: jax.Array,  # Shape [B, L]
    prefix_mask: jax.Array,  # Shape [B, L]
) -> jax.Array:
  """Makes a causal with prefix attention mask.

  I.e., as in the right diagram of Figure 3 in https://arxiv.org/pdf/1910.10683.

  Args:
    input_mask: Input mask for the input. True for non-padded tokens only, else
      False.
    prefix_mask: Input mask for the prefix. True for prefix tokens only, else
      False.

  Returns:
    Attention mask of shape [B, L, L] (where B=batch dim and L=sequence dim).
  """
  if len(input_mask.shape) != 2:
    raise ValueError(
        f'Input mask must be 2D (shape [B, L]), but got {input_mask.shape}.'
    )
  if len(prefix_mask.shape) != 2:
    raise ValueError(
        f'Prefix mask must be 2D (shape [B, L]), but got {prefix_mask.shape}.'
    )
  seq_len = input_mask.shape[-1]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
  prefix_mask = jnp.tile(jnp.expand_dims(prefix_mask, axis=1), [1, seq_len, 1])
  causal_or_prefix_mask = jnp.logical_or(causal_mask, prefix_mask)
  attn_mask = input_mask[..., None, :]
  attn_mask *= causal_or_prefix_mask
  return attn_mask


def build_positions_from_mask(input_mask: jax.Array) -> jax.Array:
  """Computes the `positions` from the `input_mask`.

  Args:
    input_mask: The tokens `input_mask`, True for non-padded tokens only.

  Returns:
    The indices to use for RoPE and absolute position encodings for the given
    input mask.
  """
  positions = jnp.cumsum(input_mask, axis=-1)
  # Subtract one for all positions from the first valid one as they are
  # 0-indexed
  return positions - (positions >= 1)
