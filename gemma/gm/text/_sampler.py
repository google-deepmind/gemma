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

"""Sampler for text."""

from collections.abc import Sequence
import dataclasses
import functools
import random as py_random
import typing
from typing import Literal

from etils import enp
from gemma import params as params_lib
from gemma.gm.data import _functional
from gemma.gm.nn import _transformer
from gemma.gm.text import _sampler_call
from gemma.gm.text import _sampling
from gemma.gm.text import _tokenizer
from gemma.gm.vision import _token_utils
import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Array, Float, Int, PRNGKey, PRNGKeyLike, UInt8  # pylint: disable=g-multiple-import,g-importing-member

import numpy as np


# TODO(epot):
# * Supports sampling with token_ids (`list[int]` / jnp) rather than `str`
#   so the same data pipeline can be used for both training and sampling.
# * Mode which queue the prompts and compute them asynchronously ?
# * Mode which yields tokens as they get predicted ?


@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplerOutput:
  """Output of the sampler when `return_state=True`.

  Attributes:
    text: Sampled text.
    state: State for extra information, and which can be used in the next turn.
  """

  text: str | list[str]
  state: _sampler_call.SamplingState

  @property
  def tokens(self) -> Int['B L'] | Int['L']:
    """Predicted tokens."""
    return self._maybe_unbatch(self.state.predicted_tokens)

  # @property
  # def logits(self) -> Float['B L V'] | Float['L V']:
  #   """Logits of the predicted tokens."""
  #   return self._maybe_unbatch(self.state.predicted_logits)

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
    cache_length: Cache length to use. This is the maximum number of tokens the
      conversation can have (prompts, answers, images for all turns). Setting
      this to a fixed value avoids re-compilation between turns.
    max_out_length: Length of the output buffer for a single turn. Static value
      used to avoid trigering a jit recompilation. Shouldn't be changed unless
      you have a task where the model generates really long outputs.
  """
  # pylint: enable=g-docstring-quotes

  model: _transformer.Transformer
  params: params_lib.Params
  tokenizer: _tokenizer.Tokenizer = None  # pytype: disable=annotation-type-mismatch
  sampling: _sampling.SamplingMethod = dataclasses.field(
      default_factory=_sampling.Greedy
  )
  forbidden_tokens: Sequence[str | int] | None = None
  # TODO(epot): Support and test rolling cache.
  cache_length: int = 4096
  max_out_length: int = 2048

  def __post_init__(self):
    # If not provided, initialize the tokenizer.
    if self.tokenizer is None:
      if not self.model.INFO.tokenizer_version:
        raise ValueError(
            'The model does not specify a tokenizer to use. '
            'Please explicitly set the tokenizer argument.'
        )
      object.__setattr__(
          self,
          'tokenizer',
          _tokenizer.Tokenizer.from_version(self.model.INFO.tokenizer_version),
      )

    # pylint: disable=protected-access]
    if (
        self.model.INFO.tokenizer_version
        and self.tokenizer.VERSION
        and self.model.INFO.tokenizer_version != self.tokenizer.VERSION
    ):
      # pylint: enable=protected-access]
      raise ValueError(
          'Incompatible model and tokenizer: '
          f'Got {type(self.model).__name__} and {type(self.tokenizer).__name__}'
      )

  # Unbatched version (`str -> str`)
  @typing.overload
  def sample(
      self,
      prompt: str,
      *,
      images: UInt8['N? H W C'] | None = None,
      max_new_tokens: int | None = ...,
      sampling: _sampling.SamplingMethod = ...,
      rng: PRNGKeyLike | None = ...,
      return_state: Literal[False] = ...,
      last_state: _sampler_call.SamplingState | None = ...,
      sharding: kd.sharding.ShardingTree | None = None,
  ) -> str:
    ...

  # Batched version (`list[str] -> list[str]`)
  @typing.overload
  def sample(
      self,
      prompt: Sequence[str],
      *,
      images: Sequence[UInt8['N H W C']] | None = None,
      max_new_tokens: int | None = ...,
      sampling: _sampling.SamplingMethod = ...,
      rng: PRNGKeyLike | None = ...,
      return_state: Literal[False] = ...,
      last_state: _sampler_call.SamplingState | None = ...,
      sharding: kd.sharding.ShardingTree | None = None,
  ) -> list[str]:
    ...

  # `return_logits=True` returns detailed output (`... -> SamplerOutput`).
  # Supports both batched (`list[str]`) and unbatched (`str`) inputs.
  @typing.overload
  def sample(
      self,
      prompt: str | Sequence[str],
      *,
      images: UInt8['B? N? H W C'] | None = None,
      max_new_tokens: int | None = ...,
      sampling: _sampling.SamplingMethod = ...,
      rng: PRNGKeyLike | None = ...,
      return_state: Literal[True] = ...,
      last_state: _sampler_call.SamplingState | None = ...,
      sharding: kd.sharding.ShardingTree | None = None,
  ) -> SamplerOutput:
    ...

  def sample(
      self,
      prompt,
      *,
      images=None,
      max_new_tokens=None,
      sampling=None,
      rng=None,
      return_state=False,
      last_state=None,
      sharding=None,
  ):
    # pylint: disable=g-docstring-quotes
    '''Samples a string from the model.

    Example:

    ```python
    prompt = """<start_of_turn>user
    I'm hesitating between those two options:

    Option 1: <start_of_image>
    Option 2: <start_of_image>

    Any thoughts ?
    <end_of_turn>
    <start_of_turn>model
    """
    sampler.sample(prompt, images=images))
    ```

    Args:
      prompt: Prompt to sample from. Can be a single string or a list of
        strings.
      images: Images for the prompt. The position where the image should be
        inserted in the prompt is determined by the `<start_of_image>` token in
        the prompt.
      max_new_tokens: Maximum number of new tokens to generate. The transformer
        will process `input_length + max_new_tokens`.
      sampling: Sampling method to use. If given, will override the default
        sampling method.
      rng: Seed to use for the sampling method. If `None`, a random seed is
        used. Can be a seed `int` or a `jax.random.PRNGKey` object.
      return_state: If `True`, returns `SamplerOutput` object with additional
        values of the output (logits, cache,...).
      last_state: When `return_state=True`, the state can be propagated across
        calls to the sampler, for multi-turn conversations. Use
        `gm.text.ChatSampler` for a simpler API which handles the state for you.
      sharding: If provided, shard the tokens according to the
        specified sharding. Users are responsible for ensuring the tokenized
        prompt is compatible with the sharding. For example, if
        `sharding=kd.sharding.FIRST_DIM`, the number of prompts must be
        divisible by the number of devices.
    Returns:
      The sampled output.
    '''

    # pylint: enable=g-docstring-quotes
    sampling = sampling or self.sampling

    # Normalize the seed.
    rng = _normalize_rng(rng)

    # Normalize inputs to always be batched.
    tokens, is_single_prompt = self._encode_prompts(
        prompt,
        add_bos=last_state is None,  # Only add BOS for the first turn.
    )
    tokens = kd.sharding.with_sharding_constraint(
        tokens, sharding
    )
    images = _normalize_images(images, is_single_prompt=is_single_prompt)

    # Cache size in the pre-fill phase.
    # Note that it includes the previous turns and MM tokens.
    init_cache_length = _get_max_total_len(
        tokens=tokens,
        images=images,
        num_tokens_per_image=self.model.vision_encoder.num_mm_tokens_per_image
        if self.model.vision_encoder
        else 0,
    )
    if last_state is not None:
      init_cache_length += int(last_state.used_cache_length)
    if init_cache_length > self.cache_length:
      raise ValueError(
          'Cache buffer filled up. With the new input, it uses:'
          f' {init_cache_length}/{self.cache_length} tokens.'
      )

    # Compute the maximum number of new tokens we can generate before filling
    # up the cache.
    # +1 as the last token is not predicted, so if we generate a single
    # token (max_new_tokens==1), we only need to forward on the
    # `init_cache_length` (i.e. no extra length needed)
    remaining_cache_length = self.cache_length - init_cache_length + 1
    max_new_tokens = max_new_tokens or self.max_out_length
    # Make sure we do not fill up the cache.
    # TODO(epot): Should raise an error if that ends up being the case after
    # sampling. We cannot know in advance as maybe the model only predict a
    # single token.
    max_new_tokens = min(max_new_tokens, remaining_cache_length)

    if last_state is None:
      cache = self.model.init_cache(
          batch_size=len(tokens),
          dtype=self._dtype,
          cache_length=self.cache_length,
      )
    else:
      # TODO(epot): Should check shape is compatible with `cache_length`.
      cache = last_state.cache

    sampler = _sampler_call.SamplerCall(
        # Static attributes. Changing those will trigger a recompilation.
        model=self.model,
        end_tokens=(
            self.tokenizer.special_tokens.EOS,
            self.tokenizer.special_tokens.END_OF_TURN,
        ),
        forbidden_tokens=self._normalized_forbidden_tokens,
        sampling=sampling,
        cache_length=self.cache_length,
        max_out_length=self.max_out_length,
        special_tokens=self.tokenizer.special_tokens,
    )
    state = sampler.sample(
        # Dynamic attributes. If the shape changes, will trigger a
        # recompilation.
        params=self.params,
        tokens=tokens,
        images=images,
        cache=cache,
        last_state=last_state,
        max_new_tokens=jnp.asarray(max_new_tokens),
        init_cache_length=init_cache_length,
        rng=rng,
    )

    # TODO(epot): Check that the text ends with an exit token (i.e. the
    # cache buffer hasn't been filled up).

    # Decode the logits.
    predicted_texts = [self.tokenizer.decode(t) for t in state.predicted_tokens]

    # # Unbatch the single prompts.
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

  def _encode_prompts(
      self,
      prompt: str | Sequence[str],
      *,
      add_bos: bool,
  ) -> tuple[Float['B L'], bool]:
    """Encode the prompts."""
    prompt, is_single_prompt = _normalize_prompt(prompt)
    tokens = [self.tokenizer.encode(p, add_bos=add_bos) for p in prompt]

    max_prompt_len = max(len(t) for t in tokens)

    # Batch tokens together
    tokens = _functional.pad(tokens, max_length=max_prompt_len)
    tokens = jnp.asarray(tokens)
    return tokens, is_single_prompt

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


def _get_max_total_len(
    *,
    tokens: Float['B L'],
    images: UInt8['B N H W C'] | None = None,
    num_tokens_per_image: int,
) -> int:
  """Compute the maximum length of the output."""
  if images is None:
    max_num_images = 0
  else:
    _, max_num_images, _, _, _ = images.shape
  inserted_mm_tokens = _token_utils.get_num_mm_tokens(
      max_num_images=max_num_images,
      num_tokens_per_image=num_tokens_per_image,
  )
  return tokens.shape[-1] + inserted_mm_tokens


def _normalize_prompt(prompt: str | Sequence[str]) -> tuple[list[str], bool]:
  """Normalize the inputs."""
  if _is_str_array(prompt):  # Supports batched input array
    assert isinstance(prompt, np.ndarray)
    prompt = prompt.tolist()

  if isinstance(prompt, str):
    is_single_prompt = True
    prompt = [prompt]
  else:
    is_single_prompt = False
    prompt = list(prompt)

  return prompt, is_single_prompt


def _normalize_images(
    images: Sequence[UInt8['N? H W C']] | UInt8['N? H W C'] | None = None,
    *,
    is_single_prompt: bool,
) -> UInt8['B N H W C'] | None:
  """Add optional `B` and `N` dimensions if needed."""
  if images is None:
    return None

  # TODO(epot): This assume all images have the same shape.
  # TODO(epot): Pad / resize images !!!
  if not enp.is_array(images):
    images = jnp.asarray(images)

  # TODO(epot): Supports sequences of images, rather than array. Need then
  # to resize and batch the images.
  if is_single_prompt:
    if len(images.shape) == 3:  # Add the `N` optional dimension   # pytype: disable=attribute-error
      images = images[None, ...]
    images = images[None, ...]  # Add the `B` dimension
  else:
    if len(images.shape) == 4:  # Add the `N` optional dimension   # pytype: disable=attribute-error
      images = images[:, None, ...]
  return images


def _normalize_token(tokenizer, token: str | int) -> int:
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
