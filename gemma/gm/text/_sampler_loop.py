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

"""Sampler implementation."""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import functools

import einops
import flax
from gemma.gm.nn import _config
from gemma.gm.nn import _transformer_like
from gemma.gm.text import _sampling
from gemma.gm.text import _tokenizer
from gemma.gm.typing import _common
from gemma.gm.utils import _cache_helper
import jax
import jax.numpy as jnp
from kauldron.typing import Bool, Int, PRNGKey, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member


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
  # TODO(epot): Uses `_cache_helper.Cache` everywhere !!!
  cache: _config.Cache
  rng: PRNGKey

  # Static values (i.e. do not changes between steps)
  # Are converted to array to avoid re-compilation when `init_cache_length`
  # changes.
  # TODO(epot): Remove `init_cache_length` and only use `used_cache_length` ?
  init_cache_length: Int['']
  full_attention_mask: Bool['B cache_length']

  @property
  def used_cache_length(self) -> Int['']:
    """Length of the cache currently used."""
    return self.init_cache_length + self.step

  @property
  def attention_mask_for_step(self) -> Bool['B cache_length']:
    """Attention mask for the current step."""
    # Select the slice of the attention mask for the current step.
    # For step == 2, init_cache_length == 5:
    # In:  [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, ..., 1, 1, 1]
    # Out: [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, ..., 0, 0, 0]

    cache_length = self.full_attention_mask.shape[-1]

    # +1 because the current step can be self-attended too.
    step_mask = jnp.arange(cache_length) < self.used_cache_length + 1
    attention_mask = self.full_attention_mask * step_mask
    return attention_mask


# TODO(epot): Refactor into simple function, rather than class.
# * autoregressive_sample()
# * autoregressive_stream_sample()
@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplerLoop:
  """Single sampling call.

  This class only has static hashable attributes, so it can be passed to
  `jax.jit`.
  """

  model: _transformer_like.TransformerLike
  end_tokens: tuple[int, ...]
  forbidden_tokens: tuple[int, ...] | None
  sampling: _sampling.SamplingMethod
  cache_length: int
  special_tokens: type[_tokenizer.SpecialTokens]

  # @functools.partial(
  #     jax.jit,
  #     static_argnames=('self', 'init_cache_length'),
  # )
  @typechecked
  def sample(
      self,
      *,
      params: _common.Params,
      init_state: SamplingState,
      max_new_tokens: Int[''],
      stream: bool = False,
  ) -> SamplingState | Iterator[SamplingState]:
    """Sample the prompt."""

    # Sample autoregressively.
    sample_fn = self._stream_sample_loop if stream else self._sample_loop
    state = sample_fn(
        params=params,
        state=init_state,
        max_new_tokens=max_new_tokens,
    )

    return state

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

      cache = _cache_helper.Cache(state.cache)
      return (
          # We predicted too many tokens.
          (state.step < max_new_tokens)
          # All batch have yield the `<end_of_turn>` token.
          & ~jnp.all(state.done)
          # End if the cache is full.
          & ~cache.is_full
      )

    state = jax.lax.while_loop(cond_fn, step_fn, state)

    # ******** Post-processing after the loop. ********

    # Mask out tokens predicted after the end tokens.
    # TODO(epot): Could integrate this directly in the `_sample_step` function.
    predicted_tokens = _mask_tokens_after_end_tokens(
        state.predicted_tokens,
        end_tokens=self.end_tokens,
    )

    # In multi-turn, the new full attention mask will be concatenated with the
    # one from previous turns, so need to mask out tokens predicted after the
    # end tokens.
    full_attention_mask = _mask_full_attention_mask_prefix_for_next_turn(
        full_attention_mask=state.full_attention_mask,
        predicted_tokens=predicted_tokens,
        init_cache_length=state.init_cache_length,
    )

    state = dataclasses.replace(
        state,
        predicted_tokens=predicted_tokens,
        full_attention_mask=full_attention_mask,
    )
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
      # Exit if the cache is full.
      cache = _cache_helper.Cache(state.cache)
      if state.done[0].tolist() or cache.is_full:
        break

      state = self._sample_step(
          params=params,
          state=state,
      )
      yield state

  @functools.partial(jax.jit, static_argnames=('self',))
  @typechecked
  def _sample_step(
      self,
      state: SamplingState,
      *,
      params: _common.Params,
  ) -> SamplingState:
    """Single sampling step."""

    out = self.model.apply(
        {'params': params},
        tokens=state.last_token[..., None],
        cache=state.cache,
        positions=state.last_token_pos[..., None],
        attention_mask=state.attention_mask_for_step[:, None, :],  # B 1 L
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


def _mask_full_attention_mask_prefix_for_next_turn(
    *,
    full_attention_mask: Bool['B cache_length'],
    predicted_tokens: Int['B L'],
    init_cache_length: Int[''],
) -> Bool['B cache_length']:
  """Mask the full attention mask for the next turn."""
  num_predicted_tokens = jnp.sum(predicted_tokens != 0, axis=-1)

  cache_length = full_attention_mask.shape[-1]
  length_pred = init_cache_length + 1 + num_predicted_tokens
  mask = jnp.arange(cache_length)[None, ...] < length_pred[..., None]
  full_attention_mask = full_attention_mask * mask
  return full_attention_mask
