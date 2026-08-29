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

"""Prefix tuning module for Gemma KV cache injection."""

import dataclasses
from flax import linen as nn
from gemma.gm.nn import _config
from gemma.gm.nn import _modules
from gemma.gm.nn import _transformer
import jax
import jax.numpy as jnp

# Very large negative position used to mask a token from the sliding window
# in local attention layers. By setting the position to a value far outside
# the typical range, tokens associated with this position are effectively
# excluded from the local attention window, preventing them from being attended to.
_MASKED_TOKEN_POSITION = -1000000


class PrefixTuning(_transformer.Transformer):
  """Wrapper around Gemma model to apply prefix tuning via KV cache injection.

  This class extends a Gemma Transformer to inject learnable prefixes into the
  KV cache, allowing the model to condition on these prefixes.

  Attributes:
    prefix_length: The length of the prefix to inject.
    global_layers_only: Whether to apply prefixes only to global layers.
      Note: If False, prefixes are applied to all layers. When using local
      attention layers, ensure `prefix_length` is within the sliding window size
      to allow tokens to attend to the prefix.
  """

  _: dataclasses.KW_ONLY
  prefix_length: int
  global_layers_only: bool = True

  @classmethod
  def from_model(
      cls, model: _transformer.Transformer, prefix_length: int, **kwargs
  ):
    """Creates a prefix-tuned model using the configuration from an existing model."""
    return cls(
        config=model.config,
        prefix_length=prefix_length,
        dtype=model.dtype,
        return_last_only=model.return_last_only,
        # Tunnel through standard fields bound to the original model
        tokens=model.tokens,
        images=model.images,
        positions=model.positions,
        attention_mask=model.attention_mask,
        **kwargs,
    )

  @nn.compact
  def __call__(
      self,
      tokens: jax.Array,
      *,
      images: jax.Array | None = None,
      positions: jax.Array | None = None,
      cache: _config.Cache | None = None,
      attention_mask: jax.Array | None = None,
      return_last_only: bool | None = None,
      return_hidden_states: bool | None = None,
  ) -> _transformer.Output:
    """Applies prefix tuning by injecting KV cache and adjusting the attention mask.

    This method injects learnable prefix parameters into the KV cache for
    global attention layers (and optionally local layers if `global_layers_only`
    is set to False). It modifies the cache to include these prefixes and
    adjusts the attention mask to allow all tokens to attend to
    the prefix tokens.

    Args:
      tokens: Input token IDs.
      images: Input images.
      positions: Input positions. If not provided, they are inferred from the
        input sequence length.
      cache: An optional cache structure for incremental decoding. If not
        provided, a new cache is initialized.
      attention_mask: An optional attention mask. If not provided, a causal
        mask is created, allowing attention to the prefix.
      return_last_only: If true, only return the logits for the last token.
      return_hidden_states: If true, return all hidden states.

    Returns:
      An Output object containing logits and optionally hidden states.
    """

    config = self.config

    is_1d = tokens.ndim == 1
    if is_1d:
      tokens = jnp.expand_dims(tokens, axis=0)

    batch_size = tokens.shape[0]
    seq_len = tokens.shape[1]

    # 1. Define the prefix parameters
    prefix_params = {}
    for i, attn_type in enumerate(config.attention_types):
      layer_name = f'layer_{i}'

      should_apply = (
          not self.global_layers_only
          or attn_type == _modules.AttentionType.GLOBAL
      )

      if should_apply:
        # Learnable prefixes
        prefix_k = self.param(
            f'prefix_k_{i}',
            nn.initializers.xavier_uniform(),
            (self.prefix_length, config.num_kv_heads, config.head_dim),
            self.dtype,
        )
        prefix_v = self.param(
            f'prefix_v_{i}',
            nn.initializers.xavier_uniform(),
            (self.prefix_length, config.num_kv_heads, config.head_dim),
            self.dtype,
        )
        prefix_k_expanded = jnp.broadcast_to(
            prefix_k,
            (
                batch_size,
                self.prefix_length,
                config.num_kv_heads,
                config.head_dim,
            ),
        )
        prefix_v_expanded = jnp.broadcast_to(
            prefix_v,
            (
                batch_size,
                self.prefix_length,
                config.num_kv_heads,
                config.head_dim,
            ),
        )
      else:
        # Fixed zeros
        prefix_k_expanded = jnp.zeros(
            (
                batch_size,
                self.prefix_length,
                config.num_kv_heads,
                config.head_dim,
            ),
            dtype=self.dtype,
        )
        prefix_v_expanded = jnp.zeros(
            (
                batch_size,
                self.prefix_length,
                config.num_kv_heads,
                config.head_dim,
            ),
            dtype=self.dtype,
        )

      prefix_params[layer_name] = {
          'k': prefix_k_expanded,
          'v': prefix_v_expanded,
      }

    # 2. Prepare the cache if not provided
    if cache is None:
      cache_length = self.prefix_length + seq_len
      cache = self.config.init_cache(
          batch_size=batch_size, dtype=self.dtype, cache_length=cache_length
      )
    else:
      cache_length = cache['layer_0']['k'].shape[1]
      if cache_length < seq_len + self.prefix_length:
        new_cache_length = seq_len + self.prefix_length
        new_cache = self.config.init_cache(
            batch_size=batch_size,
            dtype=self.dtype,
            cache_length=new_cache_length,
        )
        for l_name, layer_data in cache.items():
          for k_name, val in layer_data.items():
            if k_name in ('k', 'v'):
              new_cache[l_name][k_name] = jax.lax.dynamic_update_slice(
                  new_cache[l_name][k_name], val, (0, 0, 0, 0)
              )
            elif k_name == 'positions':
              new_cache[l_name][k_name] = jax.lax.dynamic_update_slice(
                  new_cache[l_name][k_name], val, (0, 0)
              )
            elif k_name == 'end_index':
              new_cache[l_name][k_name] = val

        cache = new_cache

    if cache is not None:
      def _inject_prefix_to_cache(c):
        for i, attn_type in enumerate(config.attention_types):
          layer_name = f'layer_{i}'
          should_apply = (
              not self.global_layers_only
              or attn_type == _modules.AttentionType.GLOBAL
          )

          # Shift end_index
          c[layer_name]['end_index'] = (
              c[layer_name]['end_index'] + self.prefix_length
          )

          # Inject K and V using dynamic_update_slice
          c[layer_name]['k'] = jax.lax.dynamic_update_slice(
              c[layer_name]['k'], prefix_params[layer_name]['k'], (0, 0, 0, 0)
          )
          c[layer_name]['v'] = jax.lax.dynamic_update_slice(
              c[layer_name]['v'], prefix_params[layer_name]['v'], (0, 0, 0, 0)
          )

          # Set positions
          if should_apply:
            positions_val = jnp.broadcast_to(
                jnp.arange(self.prefix_length), (batch_size, self.prefix_length)
            )
          else:
            positions_val = jnp.broadcast_to(
                jnp.full(
                    (self.prefix_length,),
                    _MASKED_TOKEN_POSITION,
                    dtype=jnp.int32,
                ),
                (batch_size, self.prefix_length),
            )

          c[layer_name]['positions'] = jax.lax.dynamic_update_slice(
              c[layer_name]['positions'], positions_val, (0, 0)
          )
        return c

      cache = jax.lax.cond(
          jnp.all(cache['layer_0']['end_index'] == 0),
          _inject_prefix_to_cache,
          lambda c: c,
          cache,
      )

    # 3. Prepare Attention Mask
    if attention_mask is None:
      # Create default causal mask + prefix
      prefix_mask = jnp.ones(
          (batch_size, seq_len, self.prefix_length), dtype=jnp.bool_
      )
      causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
      causal_mask = jnp.broadcast_to(
          causal_mask, (batch_size, seq_len, seq_len)
      )
      attention_mask = jnp.concatenate([prefix_mask, causal_mask], axis=-1)
    else:
      prefix_mask = jnp.ones(
          (batch_size, seq_len, self.prefix_length), dtype=jnp.bool_
      )
      attention_mask = jnp.concatenate([prefix_mask, attention_mask], axis=-1)

    if cache is not None:
      cache_len = cache['layer_0']['k'].shape[1]
      # Truncate attention mask to match the cache length
      attention_mask = attention_mask[..., :cache_len]

    # 4. Call the base class
    out = super().__call__(
        tokens=tokens,
        images=images,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
        return_last_only=return_last_only,
        return_hidden_states=return_hidden_states,
    )

    if is_1d:
      logits = jnp.squeeze(out.logits, axis=0)
      hidden_states = (
          jnp.squeeze(out.hidden_states, axis=0)
          if out.hidden_states is not None
          else None
      )
      out = out.replace(logits=logits, hidden_states=hidden_states)

    return out
