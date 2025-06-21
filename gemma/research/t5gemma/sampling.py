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

"""Sampler for text."""

import dataclasses
import functools
import logging
import random as py_random
from typing import Any, Mapping, Sequence

import einops
from etils import epy
import flax
from gemma.gm import data
from gemma.gm import text
from gemma.research.t5gemma import t5gemma
import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Array, Bool, Float, Int, PRNGKey, PRNGKeyLike, typechecked, check_type  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np


_PADDING_ID = 0

Params = Mapping[str, Any]
Cache = t5gemma.Cache
T5Gemma = t5gemma.T5Gemma

Tokenizer = text.Tokenizer
SpecialTokens = text.SpecialTokens
SamplingMethod = text.SamplingMethod
Greedy = text.Greedy
RandomSampling = text.RandomSampling


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
      compute the positional embedding.
    predicted_tokens: Fixed-size buffer for accumulating the output tokens.
    predicted_logits: Logits for each output token.
    cache: Attention KV cache.
    rng: Seed to use for sampling.
    encoder_mask: Encoder mask for the input tokens.
  """

  step: Int['']
  done: Bool['B']
  last_token: Int['B']
  last_token_pos: Int['B']
  predicted_tokens: Int['B max_output_length']
  predicted_logits: Float['B max_out_length']
  cache: Cache
  rng: PRNGKey
  encoder_mask: Bool['B L']


@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplerOutput:
  """Output of the sampler when `return_state=True`.

  Attributes:
    text: Sampled text.
    state: State for extra information, and which can be used in the next turn.
  """

  text: str | list[str]
  state: SamplingState

  @property
  def tokens(self) -> Int['B L'] | Int['L']:
    """Predicted tokens."""
    return self._maybe_unbatch(self.state.predicted_tokens)

  def _maybe_unbatch(self, x: Array['B *d']) -> Float['*d']:
    if isinstance(self.text, str):
      (x,) = x
    return x


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sampler:
  # pylint: disable=g-docstring-quotes
  """Sampler.

  This is a low-level API. For most use cases, prefer `gm.text.ChatSampler`
  instead.

  ```python
  sampler = Sampler(
      model=model,
      params=params,
  )

  output = sampler.sample(prompt)
  ```

  This sampler:

  * Is stateless (state has to be manually forwarded between calls)
  * User has to manually format the prompt using `<start_of_turn>`,...
  * The BOS (beginning of sequence) token is automatically added.

  Attributes:
    model: Gemma transformer model.
    params: Model parameters.
    tokenizer: Tokenizer.
    sampling: Sampling method to use. Default to greedy sampling.
    forbidden_tokens: List of tokens that are forbidden to be generated. If
      providing `str`, it should map to a single token id in the vocab.
    max_input_length: Maximum number of input tokens, used for prefill cache.
    max_output_length: Maximum number of output tokens to generate, used for
      generation cache.
  """
  # pylint: enable=g-docstring-quotes

  model: T5Gemma
  params: Params
  tokenizer: Tokenizer
  sampling: SamplingMethod = dataclasses.field(
      default_factory=Greedy
  )
  forbidden_tokens: Sequence[str | int] | None = None
  max_input_length: int = 1024
  max_output_length: int = 1024

  def sample(
      self,
      prompt: str | Sequence[str],
      *,
      max_new_tokens: int | None = None,
      rng: PRNGKeyLike | None = None,
      return_state: bool = False,
      sampling: SamplingMethod | None = None,
      sharding: kd.sharding.ShardingTree | None = None,
  ) -> str | list[str] | SamplerOutput:
    # pylint: disable=g-docstring-quotes
    '''Samples a string from the model.

    Example:

    ```python
    prompt = """<start_of_turn>user
    Tell me an unknown interesting biology fact about the brain.
    <end_of_turn>
    <start_of_turn>model
    """
    sampler.sample(prompt)
    ```

    Args:
      prompt: Prompt to sample from. Can be a single string or a list of
        strings.
      max_new_tokens: Maximum number of new tokens to generate. The transformer
        will process `input_length + max_new_tokens`.
      rng: Seed to use for the sampling method. If `None`, a random seed is
        used. Can be a seed `int` or a `jax.random.PRNGKey` object.
      return_state: If `True`, returns `SamplerOutput` object with additional
        values of the output (logits, cache,...).
      sampling: Sampling method to use. If given, will override the sampling
        method provided in `__init__` (default: greedy).
      sharding: If provided, shard the tokens according to the specified
        sharding. Users are responsible for ensuring the tokenized prompt is
        compatible with the sharding. For example, if
        `sharding=kd.sharding.FIRST_DIM`, the number of prompts must be
        divisible by the number of devices.

    Returns:
      The sampled output.
    '''
    sampling = sampling or self.sampling

    # Normalize the seed.
    rng = _normalize_rng(rng)

    # Normalize inputs to always be batched.
    tokens = self._encode_prompts(prompt)
    tokens = kd.sharding.with_sharding_constraint(tokens, sharding)

    # Normalize the maximum generation length.
    max_new_tokens = max_new_tokens or self.max_output_length
    if max_new_tokens > self.max_output_length:
      logging.warning(
          'Max new tokens %s is longer than max output length'
          ' %s. Fallback to max output length.',
          max_new_tokens,
          self.max_output_length,
      )
      max_new_tokens = min(max_new_tokens, self.max_output_length)

    # Initialize the cache.
    cache = self.model.config.init_cache(
        batch_size=len(tokens),
        prefill_length=self.max_input_length,
        generation_length=self.max_output_length,
        dtype=self._dtype,
    )
    cache = kd.sharding.with_sharding_constraint(cache, sharding)

    sampler = SamplerCall(
        # Static attributes. Changing those will trigger a recompilation.
        model=self.model,
        end_tokens=(
            self.tokenizer.special_tokens.EOS,
            self.tokenizer.special_tokens.END_OF_TURN,
        ),
        forbidden_tokens=self._normalized_forbidden_tokens,
        sampling=sampling,
        max_output_length=self.max_output_length,
        special_tokens=self.tokenizer.special_tokens,
    )

    state = sampler.sample(
        # Dynamic attributes. If the shape changes, will trigger a
        # recompilation.
        params=self.params,
        tokens=tokens,
        cache=cache,
        max_new_tokens=max_new_tokens,
        rng=rng,
    )

    # Decode the output state.
    return self._decode_state(  # pytype: disable=bad-return-type
        state,
        predicted_tokens=state.predicted_tokens,
        is_single_prompt=isinstance(prompt, str),
        return_state=return_state,
    )

  def _encode_prompts(
      self,
      prompt: str | Sequence[str],
  ) -> Float['B L']:
    """Encode the prompts."""
    prompt = _normalize_prompt(prompt)
    tokens = [self.tokenizer.encode(p) for p in prompt]

    max_prompt_len = max(len(t) for t in tokens)
    if max_prompt_len > self.max_input_length:
      raise ValueError(
          f'Max prompt length {max_prompt_len} is longer than max input length'
          f' {self.max_input_length}'
      )

    # Batch tokens together
    tokens = data.pad(tokens, max_length=self.max_input_length)
    return jnp.asarray(tokens)

  def _decode_state(
      self,
      state: SamplingState,
      predicted_tokens: Int['B L'],
      *,
      is_single_prompt: bool,
      return_state: bool,
  ) -> str | list[str] | SamplerOutput:
    """Decode the output state."""
    # Decode the logits.
    predicted_texts = [self.tokenizer.decode(t) for t in predicted_tokens]

    # Unbatch the single prompts.
    if is_single_prompt:
      (predicted_texts,) = predicted_texts

    # Returns either text or detailed output.
    if return_state:
      return SamplerOutput(
          text=predicted_texts,
          state=state,
      )
    else:
      return predicted_texts  # pytype: disable=bad-return-type

  @functools.cached_property
  def _normalized_forbidden_tokens(self) -> tuple[int, ...] | None:
    if self.forbidden_tokens is None:
      forbidden_tokens = ()
    else:
      forbidden_tokens = tuple(
          _normalize_token(self.tokenizer, t) for t in self.forbidden_tokens
      )
    forbidden_tokens += self.tokenizer.FORBIDDEN_TOKENS
    return forbidden_tokens

  @functools.cached_property
  def _dtype(self) -> jnp.dtype:
    return jax.tree.leaves(self.params)[0].dtype


@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplerCall:
  """Single sampling call.

  This class only has static hashable attributes, so it can be passed to
  `jax.jit`.
  """

  model: T5Gemma
  end_tokens: tuple[int, ...]
  forbidden_tokens: tuple[int, ...] | None
  sampling: SamplingMethod
  max_output_length: int
  special_tokens: type[SpecialTokens]

  @functools.partial(
      jax.jit,
      static_argnames=('self', 'max_new_tokens'),
  )
  @typechecked
  def sample(
      self,
      *,
      params: Params,
      tokens: Int['B L'],
      cache: Cache,
      max_new_tokens: int,
      rng: PRNGKey,
  ) -> SamplingState:
    """Sample the prompt."""

    # Pre-fill the KV cache and initial model input.
    init_state = self._init_state(
        params=params,
        tokens=tokens,
        cache=cache,
        rng=rng,
    )

    # Sample autoregressively.
    state = self._sample_loop(
        params=params,
        state=init_state,
        max_new_tokens=max_new_tokens,
    )

    return state

  @typechecked
  def _init_state(
      self,
      *,
      params: Params,
      cache: Cache,
      tokens: Int['B L'],
      rng: PRNGKey,
  ) -> SamplingState:
    """Prefills the KV cache."""
    batch_size, _ = tokens.shape
    start_token = jnp.zeros_like(tokens[:, 0]) + self.special_tokens.BOS

    out = self.model.apply(
        {'params': params},
        input_tokens=tokens,
        target_tokens=start_token[..., None],
        cache=cache,
    )

    # Merge the computed kv values from the prompt back into the old cache.
    cache = _merge_initial_cache(old_cache=cache, new_cache=out.cache)

    # T5Gemma starts from position 0 for the first BOS token.
    last_token_pos = jnp.zeros((batch_size,), dtype=jnp.int32)
    last_token = start_token

    return SamplingState(
        step=jnp.asarray(0),
        done=jnp.zeros((batch_size,), dtype=jnp.bool_),
        last_token=last_token,
        last_token_pos=last_token_pos,
        predicted_tokens=jnp.zeros(
            (batch_size, self.max_output_length), dtype=jnp.int32
        ),
        predicted_logits=jnp.zeros(
            (batch_size, self.max_output_length), dtype=jnp.float32,
        ),
        cache=cache,
        rng=rng,
        encoder_mask=tokens != _PADDING_ID,
    )

  @functools.partial(jax.jit, static_argnames=('self', 'max_new_tokens'))
  def _sample_loop(
      self,
      *,
      params: Params,
      state: SamplingState,
      max_new_tokens: int,
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
    predicted_tokens = _mask_tokens_after_end_tokens(
        state.predicted_tokens,
        end_tokens=self.end_tokens,
    )
    state = dataclasses.replace(state, predicted_tokens=predicted_tokens)
    return state

  @functools.partial(jax.jit, static_argnames=('self',))
  @typechecked
  def _sample_step(
      self,
      state: SamplingState,
      *,
      params: Params,
  ) -> SamplingState:
    """Single sampling step."""

    # Self attention mask: attending to all past tokens.
    # If step = 2, max_output_length = 5
    # step_mask = [1, 1, 1, 0, 0]
    step_mask = jnp.arange(self.max_output_length) <= state.step
    self_attn_mask = einops.rearrange(step_mask, 'L -> 1 1 L')
    cross_attn_mask = einops.rearrange(state.encoder_mask, 'B L -> B 1 L')

    out = self.model.apply(
        {'params': params},
        target_tokens=state.last_token[..., None],
        cache=state.cache,
        positions=state.last_token_pos[..., None],
        self_attn_mask=self_attn_mask,
        cross_attn_mask=cross_attn_mask,
        method=self.model.decode_one_step,
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
    next_logits = jnp.take_along_axis(logits, next_token[..., None], axis=-1)
    next_logits = jnp.squeeze(next_logits, axis=-1)
    check_type(next_logits, Float['B'])

    # Update the buffers to save the outputs.
    predicted_tokens = state.predicted_tokens.at[:, state.step].set(next_token)
    predicted_logits = state.predicted_logits.at[:, state.step].set(next_logits)

    # Check whether we have reached an end token.
    done = state.done | jnp.isin(
        next_token, jnp.asarray(self.end_tokens))

    return SamplingState(
        step=state.step + 1,
        done=done,
        last_token=next_token,
        # Only update the position if we are not done. The last predicted token
        # is still incremented, so we use previous `state.done`.
        last_token_pos=state.last_token_pos + ~state.done,
        predicted_tokens=predicted_tokens,
        predicted_logits=predicted_logits,
        cache=out.cache,
        rng=next_rng,
        encoder_mask=state.encoder_mask,
    )


def _normalize_prompt(prompt: str | Sequence[str]) -> list[str]:
  """Normalize the inputs."""
  if _is_str_array(prompt):  # Supports batched input array
    assert isinstance(prompt, np.ndarray)
    prompt = prompt.tolist()

  return [prompt] if isinstance(prompt, str) else list(prompt)


def _normalize_token(tokenizer, token: str | int) -> int:
  """Normalize the token."""
  if isinstance(token, int):
    return token
  token_id = tokenizer.encode(token)
  if len(token_id) != 1:
    raise ValueError(
        'Invalid forbidden token: {token!r}. Forbidden tokens must map to'
        ' single token ids in the vocab.'
    )
  (token_id,) = token_id
  return token_id


def _normalize_rng(seed_or_rng: PRNGKeyLike | None) -> PRNGKey:
  if seed_or_rng is None:
    seed_or_rng = py_random.randint(0, 1000000000)
  if not isinstance(seed_or_rng, jax.Array):
    seed_or_rng = jax.random.key(seed_or_rng)
  return seed_or_rng


def _is_str_array(x) -> bool:
  if not isinstance(x, np.ndarray):
    return False
  return np.dtype(x.dtype).type in {np.object_, np.str_}


def _merge_initial_cache(
    *,
    old_cache: Cache,
    new_cache: Cache,
):
  """Merges a new cache into an existing cache, updating 'k' and 'v' arrays."""
  updated_cache = jax.tree.map(lambda x: x, new_cache)  # Deep-copy

  for k, (old_data, new_data) in epy.zip_dict(old_cache, new_cache):
    del old_data
    updated_cache[k] = new_data
    # Update end_index for self-attention layers.
    if not k.startswith('cross_'):
      updated_cache[k]['end_index'] = new_data['end_index'] - 1
  return updated_cache


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


@dataclasses.dataclass(frozen=True, kw_only=True)
class TopkSampling(SamplingMethod):
  """Top-k sampling."""

  temperature: float = 1.0
  k: int = 1

  @typechecked
  def get_next_tokens(self, logits: Float['*B V'], rng: PRNGKey) -> Int['*B']:
    batch_size = logits.shape[0]
    topk_values, topk_indices = jax.lax.top_k(logits, self.k)
    sampled_topk_indices = jax.random.categorical(
        rng, topk_values / self.temperature, axis=-1
    )
    batch_indices = jnp.arange(batch_size)
    return topk_indices[batch_indices, sampled_topk_indices]
