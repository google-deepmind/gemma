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
"""Gemma transformer."""

import dataclasses
from typing import Iterable

from flax import linen as nn
from gemma import layers
from gemma import modules
from gemma import params as params_lib
import jax
import jax.numpy as jnp

Cache = dict[str, modules.LayerCache]


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
  max_cache_length: int = 1024
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None

  @classmethod
  def from_path(cls, path: str, cache_size: int = 1024) -> 'TransformerConfig':
    """Creates a TransformerConfig from loaded parameters."""
    metadata = params_lib.load_metadata(path)
    params = params_lib.load_params(path)

    try:
      model = metadata['somewhere in orbax checkpoint']

      if model in ('gemma-2-27-pt', 'gemma-2-27-it'):
        return cls.gemma_27b(cache_size)
      elif model in ('gemma-2-9-pt', 'gemma-2-9-it'):
        return cls.gemma_9b(cache_size)
    except KeyError:
      # V1 model that does not include model metadata.
      # Fall back to previous method
      return cls.from_params(params, cache_size)

    raise ValueError('Verify checkpoint path is a Gemma checkpoint')

  @classmethod
  def from_params(
      cls, params: params_lib.Params, cache_size: int = 1024
  ) -> 'TransformerConfig':
    """Creates a TransformerConfig from loaded parameters.

    Use for V1 models only.

    Args:
      params: Model parameters
      cache_size: Number of tokens to cache

    Returns:
      TransformerConfig.
    """
    use_qkv_einsum = 'qkv_einsum' in params['transformer']['layer_0']['attn']
    if use_qkv_einsum:
      return cls.gemma_7b((cache_size))
    elif not use_qkv_einsum:  # And something else
      return cls.gemma_2b((cache_size))
    else:
      raise ValueError(
          'Params are not a Gemma 2b, or 7b variant. These may be a different'
          ' Gemma Architecture. Use from_path function to load params.'
      )

  @classmethod
  def gemma_2b(cls, cache_size: int):
    num_layers = 18
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=2048,
        hidden_dim=16384,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        final_logit_softcap=None,
        attention_types=(modules.AttentionType.GLOBAL,) * num_layers,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        max_cache_length=cache_size,
    )

  @classmethod
  def gemma_7b(cls, cache_size: int):
    num_layers = 28
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=3072,
        hidden_dim=24576,
        num_heads=16,
        head_dim=256,
        num_kv_heads=16,
        final_logit_softcap=None,
        attention_types=(modules.AttentionType.GLOBAL,) * 28,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        max_cache_length=cache_size,
    )

  @classmethod
  def gemma_27b(cls, cache_size: int):
    num_layers = 46
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=4608,
        hidden_dim=72728,
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
        * int(num_layers / 2),
        max_cache_length=cache_size,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  @classmethod
  def gemma_9b(cls, cache_size: int):
    num_layers = 42
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=3584,
        hidden_dim=28672,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        final_logit_softcap=30.0,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(num_layers / 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        max_cache_length=cache_size,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  def init_cache(
      self,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> Cache:
    """Initializes a new Transformer cache."""
    cache = {
        f'layer_{i}': modules.Attention.init_cache(
            self.max_cache_length,
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
    input_mask: jax.Array,
) -> jax.Array:
  """Attention mask in batch mode.

  Args:
    input_mask: Input mask for the input. True for non-padded tokens only, else
      False.

  Returns:
    Attention mask.
  """
  seq_len = input_mask.shape[-1]
  attn_mask = input_mask[..., None, :]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  # Prefixes can be attended by all tokens
  attn_mask *= causal_mask[None, ...]
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
