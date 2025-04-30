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

"""Sampler implementation."""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import functools

import einops
from etils import epy
import flax
from gemma.gm.data import _functional
from gemma.gm.nn import _config
from gemma.gm.nn import _transformer
from gemma.gm.text import _sampling
from gemma.gm.text import _tokenizer
from gemma.gm.typing import _common
from gemma.gm.utils import _attention_mask
from gemma.gm.vision import _token_utils
import jax
import jax.numpy as jnp
from kauldron.typing import Bool, Int, PRNGKey, UInt8, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member

_PADDING_ID = 0

# TODO(epot): Remove hardcoded value
# \n\n + 256 + \n\n + <end_of_image>
# (Not 260 because `<start_of_image>` is already present in the input tokens)
_NUM_EXTRA_TOKENS_PER_IMAGE = 259


@flax.struct.dataclass(kw_only=True)
class SamplingState:
  """Internal sampling state.

  Attributes:
    step: Number of steps decoding steps taken so far (between [0,
      max_new_tokens]).
    done: For each sequence in the batch, `True` if the sequence is done (i.e
      has predicted a `<eos>` token).
    last_token: Model input for the next sampling step.
    last_token_pos: The RoPE position of the last token in the input. Used to
      compute the positional embedding, so includes MM tokens, but ignore all
      padding.
    predicted_tokens: Fixed-size buffer for accumulating the output tokens.
    full_attention_mask: Pre-computed attention mask for the full sequence.
    cache: Attention KV cache.
    rng: Seed to use for sampling.
    init_cache_length: Length of the cache length in the pre-fill phase. Include
      the prompt, the MM tokens, and the previous turns.
    full_attention_mask: Pre-computed attention mask for the full sequence.
  """

  step: Int['']
  done: Bool['B']
  last_token: Int['B']
  last_token_pos: Int['B']
  predicted_tokens: Int['B max_out_length']
  # TODO(epot): Better way to extract logits. This takes a lot of memory.
  # TODO(epot): Only keep the top-k logits instead? But sorting might increase
  # computation.
  # predicted_logits: Float['B max_out_length V']
  cache: _config.Cache
  rng: PRNGKey

  # Static values (i.e. do not changes between steps)
  # Are converted to array to avoid re-compilation when `init_cache_length`
  # changes.
  init_cache_length: Int['']
  full_attention_mask: Bool['B cache_length']

  @property
  def used_cache_length(self) -> Int['']:
    """Length of the cache currently used."""
    return self.init_cache_length + self.step


@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplerCall:
  """Single sampling call.

  This class only has static hashable attributes, so it can be passed to
  `jax.jit`.
  """

  model: _transformer.Transformer
  end_tokens: tuple[int, ...]
  forbidden_tokens: tuple[int, ...] | None
  sampling: _sampling.SamplingMethod
  cache_length: int
  max_out_length: int
  special_tokens: type[_tokenizer.SpecialTokens]

  # Sampling is fully jit-compatible, however, not sure it's worth doing it
  # as every-time the prompt length changes, it trigger a recompilation.
  # For optimization, prompts could be padded to a fixed size, though this
  # would slow down the pre-filling stage.
  # @functools.partial(
  #     jax.jit,
  #     static_argnames=('self', 'init_cache_length'),
  # )
  @typechecked
  def sample(
      self,
      *,
      params: _common.Params,
      tokens: Int['B L'],
      images: UInt8['B N H W C'] | None,
      cache: _config.Cache,
      last_state: SamplingState | None,
      max_new_tokens: Int[''],
      init_cache_length: int,
      rng: PRNGKey,
      stream: bool = False,
  ) -> SamplingState | Iterator[SamplingState]:
    """Sample the prompt."""

    # Pre-fill the KV cache and initial model input.
    init_state = self._init_state(
        params=params,
        tokens=tokens,
        images=images,
        init_cache_length=init_cache_length,
        cache=cache,
        last_state=last_state,
        rng=rng,
    )

    # Sample autoregressively.
    sample_fn = self._stream_sample_loop if stream else self._sample_loop
    state = sample_fn(
        params=params,
        state=init_state,
        max_new_tokens=max_new_tokens,
    )

    return state

  @typechecked
  def _init_state(
      self,
      *,
      params: _common.Params,
      cache: _config.Cache,
      tokens: Int['B L'],
      images: UInt8['B N H W C'] | None,
      last_state: SamplingState | None,
      # init_cache_length cannot be auto-computed from the tokens and images
      # shape, as it also includes the previous turns.
      init_cache_length: int,
      rng: PRNGKey,
  ) -> SamplingState:
    """Prefills the KV cache."""
    batch_size, _ = tokens.shape

    if last_state is None:
      positions_offset = None
      attention_mask = None
    else:
      positions_offset = last_state.last_token_pos
      attention_mask = _make_multi_turn_attention_mask(
          tokens=tokens,
          last_state=last_state,
      )

    out = self.model.apply(
        {'params': params},
        tokens=tokens,
        images=images,
        # Slice the cache to the prompt length, to avoid shape missmatch error.
        cache=_slice_cache(cache, length=init_cache_length),
        positions_offset=positions_offset,
        attention_mask=attention_mask,
        return_last_only=True,
    )

    # Merge the computed kv values from the prompt back into the old cache.
    cache = _merge_cache(
        old_cache=cache,
        new_cache=out.cache,
        length=init_cache_length,
    )

    # The new last token position is shifted by the prompt length (after MM).
    last_token_pos = _get_last_token_pos_after_mm(
        tokens,
        has_images=images is not None,
        special_tokens=self.special_tokens,
    )
    if positions_offset:
      last_token_pos += positions_offset
    last_token = _get_last_token(tokens)

    # Pre-compute the full attention mask for the last step.
    full_attention_mask = _make_full_attention_mask(
        tokens=tokens,
        cache_length=self.cache_length,
        max_num_images=images.shape[1] if images is not None else 0,
        special_tokens=self.special_tokens,
    )

    # TODO(epot): The first token was predicted, so could use this, but would
    # require to duplicate the logic of `_sample_step`, so leave this for later
    # /!\ !!! If doing this, remove the `-1` in `_merge_cache` !!!

    return SamplingState(
        step=jnp.asarray(0),
        done=jnp.zeros((batch_size,), dtype=jnp.bool_),
        last_token=last_token,
        last_token_pos=last_token_pos,
        # In theory, those values only need to be `B max_new_tokens`, however,
        # to avoid re-compilation when prompt length and max_new_tokens changes,
        # we set this to the fixed maximum static size.
        predicted_tokens=jnp.zeros(
            (batch_size, self.max_out_length), dtype=jnp.int32
        ),
        # predicted_logits=jnp.zeros(
        #     (batch_size, self.max_out_length, out.logits.shape[-1]),
        #     dtype=jnp.float32,
        # ),
        cache=cache,
        rng=rng,
        full_attention_mask=full_attention_mask,
        init_cache_length=jnp.asarray(init_cache_length),
    )

  @functools.partial(jax.jit, static_argnames=('self',))
  def _sample_loop(
      self,
      *,
      params: _common.Params,
      state: SamplingState,
      max_new_tokens: Int[''],
  ) -> SamplingState:
    """Internal sampling function (to be jitted)."""

    # ******** Sampling Loop. ********

    step_fn = functools.partial(self._sample_step, params=params)

    def cond_fn(state: SamplingState):
      # Keep going while we have not reached the maximum number of tokens, and
      # at least one of the samples is not done.
      return (state.step < max_new_tokens) & ~jnp.all(state.done)

    state = jax.lax.while_loop(cond_fn, step_fn, state)

    # ******** Post-processing after the loop. ********

    # Mask out tokens predicted after the end tokens.
    # TODO(epot): Could integrate this directly in the `_sample_step` function.
    predicted_tokens = _mask_tokens_after_end_tokens(
        state.predicted_tokens,
        end_tokens=self.end_tokens,
    )
    state = dataclasses.replace(state, predicted_tokens=predicted_tokens)
    return state

  def _stream_sample_loop(
      self,
      *,
      params: _common.Params,
      state: SamplingState,
      max_new_tokens: Int[''],
  ) -> Iterator[SamplingState]:
    """Streaming sampling function."""
    # Sample autoregressively.
    for _ in range(max_new_tokens):
      state = self._sample_step(
          params=params,
          state=state,
      )
      yield state
      if state.done[0].tolist():
        break

  @functools.partial(jax.jit, static_argnames=('self',))
  @typechecked
  def _sample_step(
      self,
      state: SamplingState,
      *,
      params: _common.Params,
  ) -> SamplingState:
    """Single sampling step."""

    # Select the slice of the attention mask for the current step.
    # For step == 2, init_cache_length == 5:
    # In:  [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, ..., 1, 1, 1]
    # Out: [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, ..., 0, 0, 0]
    step_mask = jnp.arange(self.cache_length) < state.used_cache_length
    attention_mask = state.full_attention_mask * step_mask
    attention_mask = einops.rearrange(attention_mask, 'B L -> B 1 L')

    out = self.model.apply(
        {'params': params},
        tokens=state.last_token[..., None],
        cache=state.cache,
        positions=state.last_token_pos[..., None],
        attention_mask=attention_mask,
    )

    logits = out.logits
    # Logit is `B L V` with `L=1`, so collapse the L dimension.
    logits = einops.rearrange(logits, 'B 1 V -> B V')
    if self.forbidden_tokens:  # Eventually filter out the forbidden tokens.
      logits = logits.at[:, self.forbidden_tokens].set(-jnp.inf)

    # Sample next token.
    next_rng, curr_rng = jax.random.split(state.rng)
    next_token = self.sampling.get_next_tokens(logits, rng=curr_rng)
    check_type(next_token, Int['B'])

    # Update the buffers to save the outputs.
    predicted_tokens = state.predicted_tokens.at[:, state.step].set(next_token)
    # predicted_logits = state.predicted_logits.at[:, state.step].set(logits)

    # Check whether we have reached an end token.
    done = state.done | jnp.isin(next_token, jnp.asarray(self.end_tokens))

    return SamplingState(
        step=state.step + 1,
        done=done,
        last_token=next_token,
        # Only update the position if we are not done. The last predicted token
        # is still incremented, so we use previous `state.done`.
        last_token_pos=state.last_token_pos + ~state.done,
        predicted_tokens=predicted_tokens,
        # predicted_logits=predicted_logits,
        cache=out.cache,
        rng=next_rng,
        init_cache_length=state.init_cache_length,
        full_attention_mask=state.full_attention_mask,
    )


def _make_full_attention_mask(
    *,
    tokens: Int['B L'],
    cache_length: int,
    max_num_images: int,
    special_tokens: type[_tokenizer.SpecialTokens],
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
      [1, 1, 1, 1, 0, 1, 1, 1, ...]
  ]
  ```

  Args:
    tokens: The input tokens.
    cache_length: The maximum length of the sequence.
    max_num_images: The maximun number of images in one prompt.
    special_tokens: The special tokens.

  Returns:
    The full attention mask.
  """
  # Mask out the padding tokens.
  full_attention_mask = tokens != _PADDING_ID

  # Insert `1` for the extra inserted MM tokens.
  if max_num_images:
    _token_utils.insert_sequence(
        tokens=full_attention_mask,
        at=special_tokens.START_OF_IMAGE,
        sequence=[True] * (_NUM_EXTRA_TOKENS_PER_IMAGE + 1),
        max_num_images=max_num_images,
    )

  # Pad the mask to the full `cache_length` for static shape.
  full_attention_mask = _functional.pad(
      full_attention_mask,
      max_length=cache_length,
      fill_value=True,
  )
  return full_attention_mask


@typechecked
def _get_last_token_pos_before_mm(tokens: Int['B L']) -> Int['B']:
  input_mask = jnp.array(tokens != _PADDING_ID, dtype=jnp.int32)
  return jnp.sum(input_mask, axis=-1) - 1


@typechecked
def _get_last_token_pos_after_mm(
    tokens: Int['B L'],
    *,
    has_images: bool,
    special_tokens: type[_tokenizer.SpecialTokens],
) -> Int['B']:
  """Returns the position of the last token in the input."""
  last_token_pos = _get_last_token_pos_before_mm(tokens)
  # Each prompt can have a different number of images. Count the number of
  # `<start_of_image>` and shift the position correspondingly.
  if has_images:
    num_images = jnp.count_nonzero(
        tokens == special_tokens.START_OF_IMAGE, axis=-1
    )
    last_token_pos += num_images * _NUM_EXTRA_TOKENS_PER_IMAGE
  return last_token_pos


@typechecked
def _get_last_token(tokens: Int['B L']) -> Int['B']:
  # Get the last non-padding token of the prompt.
  last_token_pos = _get_last_token_pos_before_mm(tokens)
  x = jnp.take_along_axis(tokens, last_token_pos[:, None], axis=-1)
  x = jnp.squeeze(x, axis=-1)
  return x


def _slice_cache(cache, *, length: int):
  new_cache = {}
  for k, layer_data in cache.items():
    new_data = dict(layer_data)
    new_data['k'] = layer_data['k'][:, :length, :, :]
    new_data['v'] = layer_data['v'][:, :length, :, :]
    new_cache[k] = new_data
  return new_cache


def _merge_cache(
    *,
    old_cache: _config.Cache,
    new_cache: _config.Cache,
    length: int,
):
  """Merges a new cache into an existing cache, updating 'k' and 'v' arrays.

  Args:
      old_cache: The original/existing cache dictionary.
      new_cache: The dictionary containing the new 'k' and 'v' arrays to be
        merged.
      length: The new length up to which the old cache should be updated

  Returns:
      The updated (merged) cache dictionary.
  """
  updated_cache = jax.tree.map(lambda x: x, new_cache)  # Deep-copy

  for k, (old_data, new_data) in epy.zip_dict(old_cache, new_cache):
    # The `_sample_loop` will re-start from the last prompt token, so use `-1`
    # as the first token is re-computed.
    updated_cache[k]['end_index'] = new_data['end_index'] - 1
    updated_cache[k]['k'] = (
        old_data['k'].at[:, :length, :, :].set(new_data['k'])
    )
    updated_cache[k]['v'] = (
        old_data['v'].at[:, :length, :, :].set(new_data['v'])
    )
  return updated_cache


@typechecked
def _make_multi_turn_attention_mask(
    *,
    tokens: Int['B L'],
    last_state: SamplingState,
) -> Bool['B L used_cache_length']:
  """Make the attention mask for the next prompt."""
  # Make the next prompt attention mask.
  next_prompt_att_mask = _attention_mask.make_causal_bidirectional_attention_mask(
      causal_mask=tokens != _PADDING_ID,
      # TODO(epot): Add bidirectional mask when images.
  )

  # Make the attention mask for the KV cache.
  cache_att_mask = _make_cache_mask(
      state=last_state,
      next_prompt_length=next_prompt_att_mask.shape[-1],  # L
  )

  return jnp.concat([cache_att_mask, next_prompt_att_mask], axis=-1)


@typechecked
def _make_cache_mask(
    *,
    state: SamplingState,
    next_prompt_length: int,
) -> Bool['B {next_prompt_length} used_cache_length']:
  """Slice the previous attention mask from the KV cache."""
  # TODO(epot): The `: state.used_cache_length` is likely not jit-compatible !!!
  cache_att_mask = state.full_attention_mask[:, : state.used_cache_length]
  cache_att_mask = cache_att_mask[:, None, :]  # b 1 used_cache_length
  cache_att_mask = jnp.broadcast_to(
      cache_att_mask,
      (
          cache_att_mask.shape[0],  # b
          next_prompt_length,  # L
          cache_att_mask.shape[2],  # used_cache_length
      ),
  )
  return cache_att_mask


@typechecked
def _mask_tokens_after_end_tokens(
    tokens: Int['B L'],
    *,
    end_tokens: tuple[int, ...],
) -> Int['B L']:
  """Mask token IDs after the EOS token with the padding ID."""
  end_tokens_mask = jnp.isin(tokens, jnp.asarray(end_tokens))
  end_tokens_mask = jnp.cumsum(end_tokens_mask, axis=-1) - end_tokens_mask == 0
  return tokens * end_tokens_mask
