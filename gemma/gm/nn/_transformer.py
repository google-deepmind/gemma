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

"""Model."""

from __future__ import annotations

import functools

import flax
from flax import linen as nn
from gemma import transformer
from gemma.gm.utils import _jax_utils
import jax.numpy as jnp
from kauldron import kontext
from kauldron.typing import Float, Int  # pylint: disable=g-multiple-import,g-importing-member

_PADDING_ID = 0


@flax.struct.dataclass
class Output:
  """Output of the Gemma model.

  Attributes:
    logits: Predicted logits of the model.
    cache: Updated cache if the input cache is not None, None elsewhere.
  """

  # When `return_last_only`, `logits` is `*B V`
  logits: Float['*B L V'] | Float['*B V']
  cache: transformer.Cache | None


class Transformer(transformer.Transformer):
  """Base transformer class.

  Attributes:
    return_last_only: If `True`, only compute and return the last token.
      Otherwise, return all logits. Default to `False`
  """

  return_last_only: bool | None = None

  # Keys to specify in the config which inputs to pass to the `__call__`
  # function (e.g. `tokens='batch.tokens'`).
  tokens: kontext.Key = kontext.REQUIRED
  positions: kontext.Key | None = None
  cache: kontext.Key | None = None
  attention_mask: kontext.Key | None = None

  def __post_init__(self):
    # TODO(epot): Config should not have `max_cache_length` parameter as
    # this is a sampling argument independent of the model architecture.
    # Also rather than inheriting from Transformer, could try unify the API
    # in a single class.
    if self.config.max_cache_length is not None:
      raise ValueError(
          'The config `max_cache_length` should be None. Got:'
          f' {self.config.max_cache_length}. Instead, the cache size is set'
          ' directly in the sampler.'
      )
    super().__post_init__()

  # Calling `model.apply` on Colab makes the Kernel crash unless it is jitted.
  @functools.partial(
      nn.jit,
      static_argnames=(
          'self',
          'return_last_only',
      ),
  )
  # The function accepts/returns aribtrary batch shape, but inside the
  # function, the batch dimension is flattened to a single dimension.
  @_jax_utils.flatten_unflatten_batch_dim()
  # TODO(epot): Restore the `@typechecked` annotation. Currently disable it
  # because during sampling, `attention_mask` is `[*B 1 1024]`, which is
  # not `*B L L`
  # @typechecked
  def __call__(  # pytype: disable=signature-mismatch
      self,
      tokens: Int['*B L'],
      *,
      positions: Int['*B L'] | None = None,
      cache: transformer.Cache | None = None,
      attention_mask: Int['*B L L'] | None = None,
      return_last_only: bool | None = None,
  ) -> Output:  # Output['*B']
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      tokens: input sequence of tokens.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.
      return_last_only: If `True`, only compute and return the logits of the
        last input token in sequence. Useful for decoding where we don't need to
        compute logits for the whole sequence, but only for the last token.
        Otherwise, return all logits. Default to `False`.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    # TODO(epot): Add `default=False` to `nn.merge_param`
    if return_last_only is None and self.return_last_only is None:
      return_last_only = False
    else:
      return_last_only = nn.merge_param(
          'return_last_only', return_last_only, self.return_last_only
      )

    inputs_mask = jnp.array(tokens != _PADDING_ID, dtype=jnp.int32)
    if positions is None:
      positions = transformer.build_positions_from_mask(inputs_mask)
    if attention_mask is None:
      attention_mask = transformer.make_causal_attn_mask(inputs_mask)

    x = self.embedder.encode(tokens)

    old_cache = cache or {}
    new_cache = {}
    for i, block in enumerate(self.blocks):
      layer_name = f'layer_{i}'
      layer_cache, x = block(
          x,
          positions,
          old_cache.get(layer_name),
          attention_mask,
      )
      new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)

    if return_last_only:
      last_input_token_idx = jnp.sum(inputs_mask, axis=-1) - 1
      # TODO(epot): Use `jnp.take_along_axis`
      x = x[jnp.arange(len(x)), last_input_token_idx, ...]

    logits = self.embedder.decode(x)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return Output(
        logits=logits,
        cache=None if cache is None else new_cache,
    )


class Gemma2_2B(Transformer):  # pylint: disable=invalid-name
  """Gemma2 transformer architecture."""

  config: transformer.TransformerConfig = (
      transformer.TransformerConfig.gemma2_2b(cache_size=None)
  )


class Gemma2_9B(Transformer):  # pylint: disable=invalid-name
  """Gemma2 transformer architecture."""

  config: transformer.TransformerConfig = (
      transformer.TransformerConfig.gemma2_9b(cache_size=None)
  )


class Gemma2_27B(Transformer):  # pylint: disable=invalid-name
  """Gemma2 transformer architecture."""

  config: transformer.TransformerConfig = (
      transformer.TransformerConfig.gemma2_27b(cache_size=None)
  )
