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
from kauldron.ktyping import Bool, Int, PRNGKey, UInt8  # pylint: disable=g-multiple-import,g-importing-member

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
    vision_input=None,
    audio=None,
    audio_lengths=None,
    audio_soft_token_counts=None,
    kv_cache_mode: _cache_helper.KVCacheMode = (
        _cache_helper.KVCacheMode.LEGACY
    ),
    kv_prefill_mode: _cache_helper.KVPrefillMode = (
        _cache_helper.KVPrefillMode.LEGACY_SCRATCH
    ),
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
    vision_input: PreprocessedVisionInput or None.
    audio: Audio input data or None.
    audio_lengths: Lengths of audio inputs or None.
    audio_soft_token_counts: Soft token counts for audio or None.
    kv_cache_mode: KV cache allocation/storage policy.
    kv_prefill_mode: KV cache prefill strategy.

  Returns:
    The initial state for the sampling loop.
  """

  if isinstance(pad_length, int):
    pad_length = (pad_length,)

  if kv_prefill_mode != _cache_helper.KVPrefillMode.LEGACY_SCRATCH:
    raise NotImplementedError(
        f'Unsupported kv_prefill_mode={kv_prefill_mode!r}. Only '
        'LEGACY_SCRATCH is implemented.'
    )

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
      kv_cache_mode=kv_cache_mode,
  )

  # LOCAL_WINDOW correctness: the persistent local cache is a ring buffer of
  # `sliding_window_size` slots, but prefill must compute attention over the
  # full prompt without mod-wrap (otherwise an early prompt token's KV gets
  # overwritten before later tokens attend to it). We therefore allocate a
  # full-size SCRATCH cache for local layers during prefill, run prefill
  # normally (legacy semantics inside Attention.__call__ because the scratch
  # carries no `logical_index` metadata), then compact each row's last valid
  # local-window tokens back into the persistent ring buffer.
  using_local_window = (
      kv_cache_mode == _cache_helper.KVCacheMode.LOCAL_WINDOW
      and any(_cache_helper.is_local_window_layer(d)
              for d in full_cache.cache.values())
  )

  if using_local_window:
    prefill_input = _make_prefill_input_local_window(
        input=input,
        cache=full_cache,
        prev_turns=prev_turns,
        pad_lengths=pad_length,
        model=model,
        vision_input=vision_input,
    )
  else:
    prefill_input = _make_prefill_input(
        input=input,
        cache=full_cache,
        prev_turns=prev_turns,
        pad_lengths=pad_length,
        vision_input=vision_input,
    )

  images_for_model = (
      vision_input if vision_input is not None else prefill_input.images
  )
  has_multimodal = images_for_model is not None or audio is not None
  is_first_turn = not prev_turns

  kwargs = {
      'tokens': prefill_input.tokens,
      'images': images_for_model,
      'cache': prefill_input.cache.cache,
      'positions': (
          None
          if (has_multimodal and is_first_turn)
          else prefill_input.positions
      ),
      'attention_mask': (
          None
          if (has_multimodal and is_first_turn)
          else prefill_input.attention_mask
      ),
      'return_last_only': True,
  }
  if audio is not None:
    kwargs.update({
        'audio': audio,
        'audio_lengths': audio_lengths,
        'audio_soft_token_counts': audio_soft_token_counts,
    })
  out = model.apply({'params': params}, **kwargs)

  # TODO(epot): Could check whether the cache is full.

  # Write the new cache back to the full cache.
  if using_local_window:
    cache = _merge_cache_local_window(
        full_cache=full_cache,
        prefill_cache=out.cache,
        logical_valid_mask=prefill_input.attention_mask[:, -1, :],
        model=model,
    )
  else:
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
      cache_length=cache_length,
      rng=rng,
  )


def _make_init_state(
    *,
    input: _types.Input,  # pylint: disable=redefined-builtin
    max_out_length: int,
    new_used_cache_length: int,
    prev_turns: _turn_utils.PrevTurns,
    cache: _cache_helper.Cache,
    cache_length: int,
    rng: PRNGKey,
) -> _sampler_loop.SamplingState:
  """Initial state for the sampling loop."""

  # The new last token position is shifted by the prompt length (after MM).
  last_token_pos = input.last_token_pos + prev_turns.last_token_pos

  # Pre-compute the full attention mask for the last step.
  full_attention_mask = _make_full_attention_mask(
      input=input,
      prev_turns=prev_turns,
      cache_length=cache_length,
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
    kv_cache_mode: _cache_helper.KVCacheMode = (
        _cache_helper.KVCacheMode.LEGACY
    ),
) -> _cache_helper.Cache:
  """Initialize or reuse the cache."""

  if not prev_turns:
    init_kwargs = {
        'batch_size': inputs.batch_size,
        'dtype': _dtype(params),
        'cache_length': cache_length,
        'sharding': sharding,
    }
    # Only forward kv_cache_mode if the model accepts it (Gemma 4 path).
    # Older transformer .init_cache signatures will reject the kwarg.
    if kv_cache_mode != _cache_helper.KVCacheMode.LEGACY:
      init_kwargs['kv_cache_mode'] = kv_cache_mode
    cache = model.init_cache(**init_kwargs)
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
    vision_input=None,  # pylint: disable=unused-argument
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

  prefill_cache_length = prev_turns.used_cache_length + token_length_padded
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


def _make_prefill_input_local_window(
    input: _types.Input,  # pylint: disable=redefined-builtin
    cache: _cache_helper.Cache,
    prev_turns: _turn_utils.PrevTurns,
    pad_lengths: tuple[int, ...],
    model: _transformer_like.TransformerLike,
    vision_input=None,  # pylint: disable=unused-argument
) -> PrefillInput:
  """Make the transformer inputs for the prefill stage in LOCAL_WINDOW mode.

  Local-sliding layers in the persistent cache are window-sized ring buffers,
  but prefill needs a contiguous, non-wrapping cache so that the prompt's
  attention math is correct when prompt_length > sliding_window_size. We
  allocate fresh full-size scratch K/V for local layers; global layers are
  sliced from the persistent cache as in the LEGACY path. Metadata fields
  (`logical_index`, `valid`) are NOT included in the scratch. That is
  deliberate, so `Attention.__call__` takes the legacy code path during
  prefill.
  """
  token_length_padded = _pad_to_bucket(input.length_with_mm, pad_lengths)
  input = input.pad(length_with_mm=token_length_padded)

  prefill_cache_length = prev_turns.used_cache_length + token_length_padded
  prefill_cache_length = _pad_to_bucket(prefill_cache_length, pad_lengths)

  scratch: _config.Cache = {}
  config = model.config
  for i, _ in enumerate(config.attention_types):
    layer_name = f'layer_{i}'
    persistent_layer = cache.cache[layer_name]
    if _cache_helper.is_local_window_layer(persistent_layer):
      # Fresh full-size scratch for this local layer (legacy shape, no
      # metadata, so Attention.__call__ takes the legacy code path). Previous
      # local-window turns are re-expanded into their logical scratch slots
      # before the new prompt is appended.
      scratch[layer_name] = _make_local_window_prefill_scratch_layer(
          persistent_layer=persistent_layer,
          prefill_cache_length=prefill_cache_length,
      )
    else:
      # Global / legacy: slice the persistent cache prefix as in the
      # LEGACY path. Slicing returns views; downstream merge writes back.
      scratch[layer_name] = {
          'k': persistent_layer['k'][:, :prefill_cache_length],
          'v': persistent_layer['v'][:, :prefill_cache_length],
          'positions': persistent_layer['positions'][:, :prefill_cache_length],
          'end_index': persistent_layer['end_index'],
      }

  return PrefillInput(
      tokens=input.text,
      images=input.images,
      positions=input.positions + prev_turns.last_token_pos[..., None],
      attention_mask=prev_turns.make_prefill_attention_mask(
          next_turn_attention_mask=input.attention_mask,
          prefill_cache_length=prefill_cache_length,
      ),
      cache=_cache_helper.Cache(scratch),
  )


def _make_local_window_prefill_scratch_layer(
    *,
    persistent_layer: dict,
    prefill_cache_length: int,
) -> dict:
  """Re-expand a local-window layer into a full logical prefill scratch."""
  b, _, num_kv_heads, head_dim = persistent_layer['k'].shape
  scratch = {
      'k': jnp.zeros(
          (b, prefill_cache_length, num_kv_heads, head_dim),
          dtype=persistent_layer['k'].dtype,
      ),
      'v': jnp.zeros(
          (b, prefill_cache_length, num_kv_heads, head_dim),
          dtype=persistent_layer['v'].dtype,
      ),
      'positions': jnp.full(
          (b, prefill_cache_length), -(10**9), dtype=jnp.int32
      ),
      'end_index': persistent_layer['end_index'],
  }

  logical_index = persistent_layer['logical_index']
  valid = persistent_layer['valid'] & (logical_index >= 0)
  valid = valid & (logical_index < prefill_cache_length)
  scatter_index = jnp.where(valid, logical_index, prefill_cache_length)
  batch_indices = jnp.arange(b)[:, None]

  scratch['k'] = scratch['k'].at[batch_indices, scatter_index].set(
      persistent_layer['k'], mode='drop'
  )
  scratch['v'] = scratch['v'].at[batch_indices, scatter_index].set(
      persistent_layer['v'], mode='drop'
  )
  scratch['positions'] = scratch['positions'].at[
      batch_indices, scatter_index
  ].set(persistent_layer['positions'], mode='drop')
  return scratch


def _merge_cache_local_window(
    *,
    full_cache: _cache_helper.Cache,
    prefill_cache: _config.Cache,
    logical_valid_mask: jax.Array,
    model: _transformer_like.TransformerLike,
) -> _cache_helper.Cache:
  """Merge a LOCAL_WINDOW prefill back into the persistent cache.

  - Global layers: copy the prefill prefix into the persistent prefix (the
    LEGACY merge path).
  - Local layers: compact the last W valid logical slots for each batch row
    into the persistent W-slot ring buffer, placing each token at
    `absolute_position % W`. Also writes the new `logical_index` and `valid`
    metadata.
  """
  config = model.config
  full_dict = full_cache.cache
  new_dict: _config.Cache = {}

  for i, _ in enumerate(config.attention_types):
    layer_name = f'layer_{i}'
    persistent_layer = full_dict[layer_name]
    prefill_layer = prefill_cache[layer_name]

    if _cache_helper.is_local_window_layer(persistent_layer):
      new_dict[layer_name] = _compact_local_window_layer(
          persistent_layer=persistent_layer,
          prefill_layer=prefill_layer,
          logical_valid_mask=logical_valid_mask,
      )
    else:
      # Global: standard prefix write (matches existing _set_cache).
      p_len = prefill_layer['k'].shape[1]
      new_dict[layer_name] = {
          'k': persistent_layer['k'].at[:, :p_len].set(prefill_layer['k']),
          'v': persistent_layer['v'].at[:, :p_len].set(prefill_layer['v']),
          'positions': persistent_layer['positions'].at[:, :p_len].set(
              prefill_layer['positions']
          ),
          'end_index': prefill_layer['end_index'],
      }

  return _cache_helper.Cache(new_dict)


def _compact_local_window_layer(
    *,
    persistent_layer: dict,
    prefill_layer: dict,
    logical_valid_mask: jax.Array,
):
  """Compact a full-size prefill scratch into a window-sized ring buffer.

  After prefill, `prefill_layer` is shape [B, P, num_kv_heads, head_dim] with
  P = prefill_cache_length (the prompt bucket length, may be < or > W).
  `persistent_layer` is shape [B, W, ...] with W = sliding_window_size.

  We keep the last W valid logical slots per batch row, then place each token
  at slot (absolute_position % W). Logical indices are stored only as metadata
  for mapping the sampler's full logical attention mask. This distinction is
  important for padded batches: a short row should retain its live local
  context even when another row forced a much longer padded logical timeline.
  """
  W = persistent_layer['k'].shape[1]      # static
  P = prefill_layer['k'].shape[1]         # static
  B = persistent_layer['k'].shape[0]      # static

  logical_valid_mask = logical_valid_mask[:, :P]
  logical_indices = jnp.arange(P, dtype=jnp.int32)[None, :]

  # Select the last W valid logical slots for each batch row. The rank-based
  # selection avoids treating padding slots as local-cache residents.
  valid_rank = jnp.cumsum(logical_valid_mask.astype(jnp.int32), axis=-1)
  num_valid = valid_rank[:, -1]
  first_kept_rank = jnp.maximum(0, num_valid - W)
  keep = logical_valid_mask & (valid_rank > first_kept_rank[:, None])
  selected_rank = jnp.clip(valid_rank - first_kept_rank[:, None] - 1, 0, W - 1)

  batch_indices_p = jnp.arange(B)[:, None]
  selected_logical = jnp.full((B, W), -1, dtype=jnp.int32)
  selected_logical = selected_logical.at[
      batch_indices_p, selected_rank
  ].max(jnp.where(keep, logical_indices, -1))
  selected_valid = selected_logical >= 0
  safe_selected_logical = jnp.clip(selected_logical, 0, P - 1)

  selected_k = jnp.take_along_axis(
      prefill_layer['k'], safe_selected_logical[:, :, None, None], axis=1
  )
  selected_v = jnp.take_along_axis(
      prefill_layer['v'], safe_selected_logical[:, :, None, None], axis=1
  )
  selected_pos = jnp.take_along_axis(
      prefill_layer['positions'], safe_selected_logical, axis=1
  ).astype(jnp.int32)

  dst_phys = selected_pos % W
  slots = jnp.arange(W, dtype=jnp.int32)
  matches = (dst_phys[:, :, None] == slots[None, None, :])
  matches = matches & selected_valid[:, :, None]
  source_for_slot = jnp.argmax(matches, axis=1)
  slot_valid = jnp.any(matches, axis=1)

  out_k = jnp.take_along_axis(
      selected_k, source_for_slot[:, :, None, None], axis=1
  )
  out_v = jnp.take_along_axis(
      selected_v, source_for_slot[:, :, None, None], axis=1
  )
  out_pos = jnp.take_along_axis(selected_pos, source_for_slot, axis=1)
  out_logical = jnp.take_along_axis(
      selected_logical, source_for_slot, axis=1
  )

  return {
      'k': jnp.where(slot_valid[:, :, None, None], out_k,
                     jnp.zeros_like(out_k)),
      'v': jnp.where(slot_valid[:, :, None, None], out_v,
                     jnp.zeros_like(out_v)),
      'positions': jnp.where(slot_valid, out_pos, -(10**9)),
      'logical_index': jnp.where(slot_valid, out_logical, -1),
      'valid': slot_valid,
      'end_index': prefill_layer['end_index'],
  }


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
