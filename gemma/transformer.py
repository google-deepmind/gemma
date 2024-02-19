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

from flax import linen as nn
from gemma import layers
from gemma import modules
from gemma import params as params_lib
import jax
import jax.numpy as jnp

Cache = dict[str, modules.LayerCache]


def make_causal_attn_mask(
    input_mask: jax.Array,
) -> jax.Array:
  """Attention mask in batch mode.

  Args:
    input_mask: Input mask for the input

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
  max_cache_length: int = 1024

  @classmethod
  def from_params(
      cls, params: params_lib.Params, cache_size: int = 1024
  ) -> 'TransformerConfig':
    """Creates a TransformerConfig from loaded parameters."""
    num_layers = (
        max([
            int(k.split('_')[1])
            for k in params['transformer'].keys()
            if 'layer_' in k
        ])
        + 1
    )

    hidden_dim, embed_dim = (
        params['transformer']['layer_0']['mlp']['linear'].shape
    )

    num_heads, head_dim, _ = (
        params['transformer']['layer_0']['attn']['attn_vec_einsum']['w'].shape
    )

    use_qkv_einsum = 'qkv_einsum' in params['transformer']['layer_0']['attn']
    if use_qkv_einsum:
      num_kv_heads = num_heads
    else:
      num_kv_heads = params['transformer']['layer_0']['attn']['kv_einsum'][
          'w'
      ].shape[1]

    num_embed = params['transformer']['embedder']['input_embedding'].shape[0]

    return cls(
        num_layers=num_layers,
        num_embed=num_embed,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        max_cache_length=cache_size,
    )


def init_cache(
    config: TransformerConfig,
    batch_size: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Cache:
  """Initializes a new Transformer cache."""
  cache = {
      f'layer_{i}': modules.init_layer_cache(
          config.max_cache_length, config.num_heads, config.head_dim, batch_size, dtype
      )
      for i in range(config.num_layers)
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
        )
        for i in range(self.config.num_layers)
    ]
    self.final_norm = layers.RMSNorm()

  def __call__(
      self,
      last_tokens: jax.Array,     # [B,L]
      positions: jax.Array,       # [B, L]
      cache: Cache | None,        # (sequence length L')
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

    return logits, cache  # pytype: disable=bad-return-type
