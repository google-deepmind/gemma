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

"""Prefill stage for the sampler."""

from __future__ import annotations

import dataclasses

from gemma.gm.data import _functional
from gemma.gm.nn import _config
from gemma.gm.nn import _transformer_like
from gemma.gm.text import _sampler_loop
from gemma.gm.text import _turn_utils
from gemma.gm.typing import _common
from gemma.gm.utils import _cache_helper
from gemma.gm.utils import _types
import jax
from jax import numpy as jnp
from kauldron import kd
from kauldron.typing import Bool, Int, PRNGKey, UInt8  # pylint: disable=g-multiple-import,g-importing-member

_PADDING_ID = 0


# TODO(epot): Could try to unify with `_types.Input` and allow `Transformer`
# to directly take an `Input` object.
@dataclasses.dataclass(frozen=True, kw_only=True)
class PrefillInput:
  """Input for the prefill phase."""

  tokens: Int['B L']
  images: UInt8['B N H W C'] | None
  positions: Int['B L']
  attention_mask: Bool['B L cache_length']
  cache: _cache_helper.Cache


def prefill(
    *,
    model: _transformer_like.TransformerLike,
    params: _common.Params,
    input: _types.Input,  # pylint: disable=redefined-builtin
    last_state: _sampler_loop.SamplingState | None,
    # Pad sizes for the cache.
    cache_length: int,
    max_out_length: int,
    pad_length: None | int | tuple[int, ...] = None,
    rng: PRNGKey,
    sharding: kd.sharding.ShardingTree | None,
) -> _sampler_loop.SamplingState:
  """Pre-fill the KV cache and initial model input.

  Note: The initial state contains the last token, so the sampling loop
  will re-start from the last prompt token.

  Args:
    model: The transformer model.
    params: The model parameters.
    input: The input tokens.
    last_state: The last state of the sampling loop (for multi-turn).
    cache_length: The maximum length of the sequence.
    max_out_length: The maximum length of the output.
    pad_length: The pad length for the prompt.
    rng: The random number generator.
    sharding: The sharding tree.

  Returns:
    The initial state for the sampling loop.
  """

  if isinstance(pad_length, int):
    pad_length = (pad_length,)

  # Wrap the last state to simplify the logic of adding the previous turns to
  # the prefill.
  prev_turns = _turn_utils.PrevTurns(last_state=last_state)
  del last_state

  # TODO(epot): Have a version that is fully jittable.

  full_cache = _get_or_init_cache(
      inputs=input,
      prev_turns=prev_turns,
      model=model,
      params=params,
      cache_length=cache_length,
      sharding=sharding,
  )

  prefill_input = _make_prefill_input(
      input=input,
      cache=full_cache,
      prev_turns=prev_turns,
      pad_lengths=pad_length,
  )

  # Call the model to fill up the cache.
  out = model.apply(
      {'params': params},
      tokens=prefill_input.tokens,
      images=prefill_input.images,
      # Slice the cache to the prompt length, to avoid shape missmatch error.
      cache=prefill_input.cache.cache,
      positions=prefill_input.positions,
      attention_mask=prefill_input.attention_mask,
      return_last_only=True,
  )

  # TODO(epot): Could check whether the cache is full.

  # Write the new cache back to the full cache.
  cache = _merge_cache(
      full_cache=full_cache,
      prefill_cache=out.cache,
  )
  del out

  # Set the end index (which indicates the last kv cache index used).
  # -1 because `_sample_loop` will re-start from the last prompt token.
  # Note this is smaller than `init_cache_length` as we remove the padding.
  #
  # Example: For input tokens batch like:
  #
  # [
  #     [p0, p1, p2, 0, 0]
  #     [p0, p1, 0, 0, 0]
  #     [p0, p1, p2, p3, p4]
  # ]
  #
  # During prefill, the cache is filled up:
  #
  # [
  #     [kv0, kv1, kv2, 0, 0, 0, ...]
  #     [kv0, kv1, 0, 0, 0, 0, ...]
  #     [kv0, kv1, kv2, kv3, kv4, 0, ...]
  # ]
  #
  # `length_with_mm == 5` but `used_cache_length == 4`, so the first sampling
  # step will overwrite the last cache[4] values with the last prompt token:
  #
  # [
  #     [kv0, kv1, kv2, 0, kv2, 0, ...]
  #     [kv0, kv1, 0, 0, kv1, 0, ...]
  #     [kv0, kv1, kv2, kv3, kv4, 0, ...]
  # ]
  #
  # As you can see, the last token for padded sequence appears twice in the KV
  # cache. However in practice, the attention mask ensures only one of those
  # values is attended to.
  # The attention for the first sampling step is:
  #
  # [
  #     [1, 1, 1, 0, 0, 0, ...]
  #     [1, 1, 0, 0, 0, 0, ...]
  #     [1, 1, 1, 1, 1, 0, ...]
  # ]
  #
  # So in practice, this means that the query of first sampling step will
  # attend to the last prompt token computed during prefill, rather than
  # itself (for padded sequences). But the two values should be identical
  # (minus Jax precision errors ^^).
  # For the second sampling step, the cache and attention mask will be:
  #
  # [
  #     [kv0, kv1, kv2, 0, kv2, kv3, ...]
  #     [kv0, kv1, 0, 0, kv1, kv2, ...]
  #     [kv0, kv1, kv2, kv3, kv4, kv5, ...]
  # ]
  # [
  #     [1, 1, 1, 0, 0, 1, ...]
  #     [1, 1, 0, 0, 0, 1, ...]
  #     [1, 1, 1, 1, 1, 1, ...]
  # ]
  #
  # So the KV value computed during the first sampling step is never attended
  # to for padded sequences.
  #
  # A cleaner implementation could be to have a per-batch cache index, to
  # remove padding. But I leave this to my future self (or to future Gemini).

  new_used_cache_length = (
      prev_turns.used_cache_length + input.length_with_mm - 1
  )
  cache = cache.set_end_index(new_used_cache_length)

  # TODO(epot): The first token was predicted, so could use this, but would
  # require to duplicate the logic of `_sample_step`, so leave this for later
  # The `_sample_loop` will re-start from the last prompt token, so use `-1`
  # as the first token is re-computed.
  return _make_init_state(
      input=input,
      max_out_length=max_out_length,
      new_used_cache_length=new_used_cache_length,
      prev_turns=prev_turns,
      cache=cache,
      rng=rng,
  )


def _make_init_state(
    *,
    input: _types.Input,  # pylint: disable=redefined-builtin
    max_out_length: int,
    new_used_cache_length: int,
    prev_turns: _turn_utils.PrevTurns,
    cache: _cache_helper.Cache,
    rng: PRNGKey,
) -> _sampler_loop.SamplingState:
  """Initial state for the sampling loop."""

  # The new last token position is shifted by the prompt length (after MM).
  last_token_pos = input.last_token_pos + prev_turns.last_token_pos

  # Pre-compute the full attention mask for the last step.
  full_attention_mask = _make_full_attention_mask(
      input=input,
      prev_turns=prev_turns,
      cache_length=cache.total_cache_length,
  )

  return _sampler_loop.SamplingState(
      step=jnp.asarray(0),
      done=jnp.zeros((input.batch_size,), dtype=jnp.bool_),
      # Last token for autoregressive sampling.
      last_token=input.last_token,
      last_token_pos=last_token_pos,
      # In theory, those values only need to be `B max_new_tokens`, however,
      # to avoid re-compilation when prompt length and max_new_tokens changes,
      # we set this to the fixed maximum static size.
      predicted_tokens=jnp.zeros(
          (input.batch_size, max_out_length), dtype=jnp.int32
      ),
      # predicted_logits=jnp.zeros(
      #     (batch_size, self.max_out_length, out.logits.shape[-1]),
      #     dtype=jnp.float32,
      # ),
      cache=cache.cache,
      rng=rng,
      full_attention_mask=full_attention_mask,
      init_cache_length=jnp.asarray(new_used_cache_length),
  )


def _get_or_init_cache(
    *,
    inputs: _types.Input,
    prev_turns: _turn_utils.PrevTurns,
    model: _transformer_like.TransformerLike,
    params: _common.Params,
    cache_length: int,
    sharding: kd.sharding.ShardingTree | None,
) -> _cache_helper.Cache:
  """Initialize or reuse the cache."""

  if not prev_turns:
    cache = model.init_cache(
        batch_size=inputs.batch_size,
        dtype=_dtype(params),
        cache_length=cache_length,
        sharding=sharding,
    )
  else:
    # TODO(epot): Should check shape is compatible with `cache_length`.
    cache = prev_turns.cache

  # Wrap cache to help resizing.
  cache = _cache_helper.Cache(cache)
  return cache


def _make_prefill_input(
    input: _types.Input,  # pylint: disable=redefined-builtin
    cache: _cache_helper.Cache,
    prev_turns: _turn_utils.PrevTurns,
    pad_lengths: tuple[int, ...],
) -> PrefillInput:
  """Make the transformer inputs for the prefill stage."""
  # Supports:
  # * Multi-turn
  # * Multi-modal
  # * Add padding for static shapes

  # Pad the input, to avoid re-compilation and make the shapes static.
  # Pad token length is the smallest pad bucket that fits the input.
  token_length_padded = _pad_to_bucket(input.length_with_mm, pad_lengths)
  input = input.pad(length_with_mm=token_length_padded)

  # Cache length is equal to the pad token length + the previous turns.
  prefill_cache_length = prev_turns.used_cache_length + token_length_padded
  # Pad the cache length, to avoid unecessary re-compilations.
  prefill_cache_length = _pad_to_bucket(prefill_cache_length, pad_lengths)
  cache = cache[:, :prefill_cache_length]

  return PrefillInput(
      tokens=input.text,
      images=input.images,
      # For multi-turn, the positions should be shifted to take into account
      # the previous turns.
      positions=input.positions + prev_turns.last_token_pos[..., None],
      attention_mask=prev_turns.make_prefill_attention_mask(
          next_turn_attention_mask=input.attention_mask,
          prefill_cache_length=prefill_cache_length,
      ),
      cache=cache,
  )


def _pad_to_bucket(length: int, pad_lengths: tuple[int, ...] | None) -> int:
  """Get the smallest pad length (or the original length if too large)."""
  if pad_lengths is None:
    return length

  for pad_length in pad_lengths:
    if length <= pad_length:
      return pad_length
  return length


def _merge_cache(
    *,
    full_cache: _cache_helper.Cache,
    prefill_cache: _config.Cache,
) -> _cache_helper.Cache:
  prefill_cache = _cache_helper.Cache(prefill_cache)

  return full_cache.at[:, : prefill_cache.total_cache_length].set_kv(
      prefill_cache
  )


def _make_full_attention_mask(
    *,
    input: _types.Input,  # pylint: disable=redefined-builtin
    prev_turns: _turn_utils.PrevTurns,
    cache_length: int,
):
  """Pre-compute the full attention mask for the full `cache_length`.

  During sampling, the attention mask is masked for the current step.

  For an input tokens like:

  ```python
  [

      [p0, p1, p2, 0, 0]
      [p0, p1, 0, 0, 0]
      [p0, p1, p2, p3, p4]
  ]
  ```

  The attention mask will be:

  ```python
  [
      [1, 1, 1, 0, 0, 1, 1, 1, ...]
      [1, 1, 0, 0, 0, 1, 1, 1, ...]
      [1, 1, 1, 1, 1, 1, 1, 1, ...]
  ]
  ```

  Args:
    input: The input tokens.
    prev_turns: The previous turns.
    cache_length: The maximum length of the sequence.

  Returns:
    The full attention mask.
  """
  # Mask out the padding tokens.
  full_attention_mask = input.tokens_with_mm != _PADDING_ID

  # Compute the full attention mask across turns.
  if prev_turns:
    full_attention_mask = jnp.concatenate(
        [prev_turns.prev_attention_mask, full_attention_mask], axis=-1
    )

  # Pad the mask to the full `cache_length` for static shape.
  full_attention_mask = _functional.pad(
      full_attention_mask,
      max_length=cache_length,
      fill_value=True,
  )
  return full_attention_mask


def _dtype(params: _common.Params) -> jnp.dtype:
  return jax.tree.leaves(params)[0].dtype
