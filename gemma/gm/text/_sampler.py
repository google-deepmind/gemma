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

from collections.abc import Iterator, Sequence
import dataclasses
import functools
import random as py_random
import typing
from typing import Literal

from etils import enp
from gemma.gm.data import _functional
from gemma.gm.nn import _transformer_like
from gemma.gm.text import _prefill
from gemma.gm.text import _sampler_loop
from gemma.gm.text import _sampling
from gemma.gm.text import _tokenizer
from gemma.gm.typing import _common
from gemma.gm.utils import _types
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
  state: _sampler_loop.SamplingState

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
    stop_tokens: List of tokens that will stop generation if generated. If
      providing `str`, it should map to a single token id in the vocab.
    cache_length: Cache length to use. This is the maximum number of tokens the
      conversation can have (prompts, answers, images for all turns). Setting
      this to a fixed value avoids re-compilation between turns.
    max_out_length: Length of the output buffer for a single turn. Static value
      used to avoid trigering a jit recompilation. Shouldn't be changed unless
      you have a task where the model generates really long outputs.
    pad_length: If provided, pad the prompt to this length. This ensure the
      prompt is always the same length, to avoid jit re-compilation.
  """
  # pylint: enable=g-docstring-quotes

  model: _transformer_like.TransformerLike
  params: _common.Params
  tokenizer: _tokenizer.Tokenizer = None  # pytype: disable=annotation-type-mismatch
  sampling: _sampling.SamplingMethod = dataclasses.field(
      default_factory=_sampling.Greedy
  )
  forbidden_tokens: Sequence[str | int] | None = None
  stop_tokens: Sequence[str | int] | None = None
  # TODO(epot): Support and test rolling cache.
  cache_length: int = 4096
  max_out_length: int = 2048
  pad_length: None | int | tuple[int, ...] = (256, 512, 1024)

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
      images: UInt8['N? H W C'] | None = ...,
      max_new_tokens: int | None = ...,
      stream: Literal[False] = ...,
      sampling: _sampling.SamplingMethod = ...,
      rng: PRNGKeyLike | None = ...,
      return_state: Literal[False] = ...,
      last_state: _sampler_loop.SamplingState | None = ...,
      sharding: kd.sharding.ShardingTree | None = ...,
  ) -> str:
    ...

  # Batched version (`list[str] -> list[str]`)
  @typing.overload
  def sample(
      self,
      prompt: Sequence[str],
      *,
      images: Sequence[UInt8['N H W C']] | None = ...,
      max_new_tokens: int | None = ...,
      stream: Literal[False] = ...,
      sampling: _sampling.SamplingMethod = ...,
      rng: PRNGKeyLike | None = ...,
      return_state: Literal[False] = ...,
      last_state: _sampler_loop.SamplingState | None = ...,
      sharding: kd.sharding.ShardingTree | None = ...,
  ) -> list[str]:
    ...

  # `return_state=True` returns detailed output (`... -> SamplerOutput`).
  # Supports both batched (`list[str]`) and unbatched (`str`) inputs.
  @typing.overload
  def sample(
      self,
      prompt: str | Sequence[str],
      *,
      images: UInt8['B? N? H W C'] | None = ...,
      max_new_tokens: int | None = ...,
      stream: Literal[False] = ...,
      sampling: _sampling.SamplingMethod = ...,
      rng: PRNGKeyLike | None = ...,
      return_state: Literal[True] = ...,
      last_state: _sampler_loop.SamplingState | None = ...,
      sharding: kd.sharding.ShardingTree | None = ...,
  ) -> SamplerOutput:
    ...

  # TODO(epot): Re-activate this. Currently pytype is confused when adding
  # this, so disabling it. It's ok, as it's mostly for Colab use.
  # # Streaming version (`stream=True`), yields tokens as they get predicted.
  # @typing.overload
  # def sample(
  #     self,
  #     prompt: str | Sequence[str],
  #     *,
  #     images: UInt8['B? N? H W C'] | None = ...,
  #     max_new_tokens: int = ...,
  #     stream: Literal[True] = ...,
  #     sampling: _sampling.SamplingMethod = ...,
  #     rng: PRNGKeyLike | None = ...,
  #     return_state: bool = ...,
  #     last_state: _sampler_loop.SamplingState | None = ...,
  #     sharding: kd.sharding.ShardingTree | None = ...,
  # ) -> Iterator[str | SamplerOutput]:
  #   ...

  def sample(
      self,
      prompt,
      *,
      images=None,
      max_new_tokens=None,
      stream=False,
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
      stream: If `True`, yields tokens as they get predicted.
      sampling: Sampling method to use. If given, will override the sampling
        method provided in `__init__` (default: greedy).
      rng: Seed to use for the sampling method. If `None`, a random seed is
        used. Can be a seed `int` or a `jax.random.PRNGKey` object.
      return_state: If `True`, returns `SamplerOutput` object with additional
        values of the output (logits, cache,...).
      last_state: When `return_state=True`, the state can be propagated across
        calls to the sampler, for multi-turn conversations. Use
        `gm.text.ChatSampler` for a simpler API which handles the state for you.
      sharding: If provided, shard the tokens according to the specified
        sharding. Users are responsible for ensuring the tokenized prompt is
        compatible with the sharding. For example, if
        `sharding=kd.sharding.FIRST_DIM`, the number of prompts must be
        divisible by the number of devices.

    Returns:
      The sampled output.
    '''

    # pylint: enable=g-docstring-quotes
    sampling = sampling or self.sampling

    # Normalize the seed.
    rng = _normalize_rng(rng)

    has_batch_dim = _get_has_batch_dim(prompt)
    if stream and has_batch_dim:
      raise ValueError(
          'Streaming is not supported for batched prompts. Let us know if you'
          ' need this feature.'
      )

    # Normalize the text, images. Tokenize, shard,...
    inputs = self._get_inputs(
        prompt=prompt,
        images=images,
        add_bos=last_state is None,  # Only add BOS for the first turn.
        has_batch_dim=has_batch_dim,
        sharding=sharding,
    )

    # Prefill the cache.
    init_state = _prefill.prefill(
        model=self.model,
        params=self.params,
        input=inputs,
        last_state=last_state,
        cache_length=self.cache_length,
        pad_length=self.pad_length,
        rng=rng,
        sharding=sharding,
        # Here we use the static `max_out_length`, as it is used to initialize
        # the output buffer. However in the sampling loop, users can choose
        # to only decode a subset by setting a smaller `max_new_tokens`.
        max_out_length=self.max_out_length,
    )

    # Max out length is static, while max_new_tokens is dynamic.
    # This allow to change the max out length without recompiling.
    if max_new_tokens and max_new_tokens > self.max_out_length:
      raise ValueError(
          'max_new_tokens should be smaller or equal to max_out_length. Got:'
          f' {max_new_tokens} / {self.max_out_length}'
      )
    max_new_tokens = max_new_tokens or self.max_out_length
    max_new_tokens = jnp.asarray(max_new_tokens)

    # TODO(epot): Donate the `init_state`, `last_state`

    sampler = _sampler_loop.SamplerLoop(
        # Static attributes. Changing those will trigger a recompilation.
        model=self.model,
        end_tokens=(
            self.tokenizer.special_tokens.EOS,
            self.tokenizer.special_tokens.END_OF_TURN,
            *self._normalized_stop_tokens,
        ),
        forbidden_tokens=self._normalized_forbidden_tokens,
        sampling=sampling,
        cache_length=self.cache_length,
        special_tokens=self.tokenizer.special_tokens,
    )

    # TODO(epot): Use `jnp.cond` to detect when the cache is full (or use
    # rolling-cache). Also do add a check that the cache wasn't filled up
    # after the sampling.
    state = sampler.sample(
        # Dynamic attributes. If the shape changes, will trigger a
        # recompilation.
        params=self.params,
        init_state=init_state,
        max_new_tokens=max_new_tokens,
        stream=stream,
    )

    if stream:
      return self._stream_decode_state(  # pytype: disable=bad-return-type
          state,
          return_state=return_state,
      )
    else:
      return self._decode_state(  # pytype: disable=bad-return-type
          state,
          predicted_tokens=state.predicted_tokens,
          has_batch_dim=has_batch_dim,
          return_state=return_state,
      )

  def _get_inputs(
      self,
      *,
      prompt,
      images,
      add_bos,
      has_batch_dim,
      sharding,
  ) -> _types.Input:
    """Normalize the inputs."""
    # Normalize inputs to always be batched.
    tokens = self._tokenize_prompts(
        prompt,
        add_bos=add_bos,  # Only add BOS for the first turn.
    )
    if sharding is not None:
      tokens = kd.sharding.device_put(tokens, sharding)
    # TODO(epot): Reshape images to avoid jax.jit recompilation.
    images = _normalize_images(images, has_batch_dim=has_batch_dim)
    if images is not None and sharding is not None:
      images = kd.sharding.device_put(images, sharding)

    return _types.Input(
        text=tokens,
        images=images,
        config=self.model.config.input_config,
    )

  def _tokenize_prompts(
      self,
      prompt: str | Sequence[str],
      *,
      add_bos: bool,
      pad_length: int | None = None,
  ) -> Float['B L']:
    """Encode the prompts."""
    prompt = _normalize_prompt(prompt)
    tokens = [self.tokenizer.encode(p, add_bos=add_bos) for p in prompt]

    # Notice that if pad_length exceeds the maximum length of the prompts,
    # an error will be raised by the `.pad` function below.
    max_prompt_len = pad_length or max(len(t) for t in tokens)
    # In multi-host, each host read different data, so sync to the max length
    # across all hosts.
    max_prompt_len = _max_across_hosts(max_prompt_len)

    # Batch tokens together
    tokens = _functional.pad(tokens, max_length=max_prompt_len)
    tokens = jnp.asarray(tokens)
    return tokens

  def _decode_state(
      self,
      state: _sampler_loop.SamplingState,
      predicted_tokens: Int['B L'],
      *,
      has_batch_dim: bool,
      return_state: bool,
  ) -> str | list[str] | SamplerOutput:
    """Decode the output state."""
    # TODO(epot): Check that the text ends with an exit token (i.e. the
    # cache buffer hasn't been filled up).

    # In multi-host, each host only has a slice of the data. We need to
    # replicate the data, so each host can decode texts from all other hosts.
    if jax.process_count() > 1:
      predicted_tokens = kd.sharding.with_sharding_constraint(
          predicted_tokens,
          kd.sharding.REPLICATED,
      )

    # Decode the logits.
    predicted_texts = [self.tokenizer.decode(t) for t in predicted_tokens]

    # # Unbatch the single prompts.
    if not has_batch_dim:
      (predicted_texts,) = predicted_texts

    # Returns either text or detailed output.
    if return_state:
      return SamplerOutput(
          text=predicted_texts,
          state=state,
      )
    else:
      return predicted_texts  # pytype: disable=bad-return-type

  def _stream_decode_state(
      self,
      state_iter: Iterator[_sampler_loop.SamplingState],
      *,
      return_state: bool,
  ):
    for i, state in enumerate(state_iter):
      yield self._decode_state(
          state,
          predicted_tokens=state.predicted_tokens[..., i],
          has_batch_dim=False,
          return_state=return_state,
      )

  @functools.cached_property
  def _normalized_forbidden_tokens(self) -> tuple[int, ...] | None:
    forbidden_tokens = self._normalize_tokens(self.forbidden_tokens)
    forbidden_tokens += self.tokenizer.FORBIDDEN_TOKENS
    return forbidden_tokens

  @functools.cached_property
  def _normalized_stop_tokens(self) -> tuple[int, ...]:
    return self._normalize_tokens(self.stop_tokens)

  def _normalize_tokens(
      self, tokens: Sequence[str | int] | None
  ) -> tuple[int, ...]:
    if tokens is None:
      return ()
    else:
      return tuple(_normalize_token(self.tokenizer, t) for t in tokens)


def _get_has_batch_dim(prompt: str | Sequence[str]) -> bool:
  """Returns whether the prompt batched or not."""
  if isinstance(prompt, str):
    return False
  elif _is_str_array(prompt):  # Scalar str array.
    assert isinstance(prompt, np.ndarray)
    return bool(prompt.ndim)  # pylint: disable=g-explicit-bool-comparison
  else:
    return True


def _normalize_prompt(prompt: str | Sequence[str]) -> list[str]:
  """Normalize the inputs."""
  if _is_str_array(prompt):  # Supports batched input array
    assert isinstance(prompt, np.ndarray)
    prompt = prompt.tolist()

  if isinstance(prompt, str):
    prompt = [prompt]
  else:
    prompt = list(prompt)

  return prompt


def _normalize_images(
    images: Sequence[UInt8['N? H W C']] | UInt8['N? H W C'] | None = None,
    *,
    has_batch_dim: bool,
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
  if not has_batch_dim:
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
        'Invalid token: {token!r}. `stop_token`s and `forbidden_token`s must'
        ' map to single token ids in the vocab.'
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


def _max_across_hosts(x: int) -> int:
  """Returns the maximum value across all hosts."""
  if jax.process_count() == 1:
    return x
  x = jnp.asarray([x] * jax.local_device_count())
  x = _max_across_hosts_pmap(x)
  return x[0]


@functools.partial(jax.pmap, axis_name='i')
def _max_across_hosts_pmap(x: jax.Array) -> jax.Array:
  return jax.lax.pmax(x, 'i')
