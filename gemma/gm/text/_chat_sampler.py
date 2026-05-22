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

"""Chat sampler."""

from collections.abc import Iterator, Sequence
import dataclasses
import functools
import warnings

import dialog
from etils import epy
# from gemma.gm.data import _functional
from gemma.gm.nn import _transformer_like
from gemma.gm.text import _gemma4_sampler
from gemma.gm.text import _sampler
from gemma.gm.text import _sampler_loop
from gemma.gm.text import _sampling
from gemma.gm.text import _template
from gemma.gm.text import _tokenizer
from gemma.gm.typing import _common
# from gemma.gm.vision import _token_utils
from kauldron import kd
from kauldron.ktyping import UInt8  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.typing import PRNGKeyLike  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from PIL import Image


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class ChatSampler:
  """Chat sampler.

  A unified chat sampler that works with all Gemma model versions (2, 3, 3n,
  4). Automatically selects the correct underlying sampler and prompt format
  based on the model's tokenizer version.

  ```python
  sampler = ChatSampler(
      model=model,
      params=params,
      multi_turn=True,
  )

  output0 = sampler.chat('Write a poem about cats.')
  output1 = sampler.chat('And about dogs.')
  output2 = sampler.chat('Which one do you prefer?')
  ```

  For Gemma 4 models with multimodal inputs:

  ```python
  sampler = ChatSampler(
      model=model,
      params=params,
      multi_turn=True,
  )

  out0 = sampler.chat('Describe this image <|image|>.', images=[img1])
  out1 = sampler.chat('What about this one <|image|>?', images=[img2])
  out2 = sampler.chat('Summarize your observations.')
  ```

  This sampler:

  * Is stateful (the KV-cache state is automatically handled)
  * Automatically formats the prompt with turn tags, adds the BOS
    (beginning of sequence) token. And filters the end-of-turn tokens from
    the output.
  * For Gemma 4 models: supports per-turn images (variable aspect ratio)
    and audio via the `images` and `audio` arguments.

  Attributes:
    model: Gemma transformer model.
    params: Model parameters.
    multi_turn: If `True`, reuse the previous turns as context.
    print_stream: If `True`, will print the sampled output as it is generated.
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
      used to avoid triggering a jit recompilation. Shouldn't be changed unless
      you have a task where the model generates really long outputs.
    pad_length: Pad lengths for static shapes (Gemma 4 only).
    patch_size: Patch size for vision encoder (Gemma 4 only).
    max_soft_tokens: Maximum soft tokens per image (Gemma 4 only).
    pooling_kernel_size: Pooling kernel size (Gemma 4 only).
    audio_sample_rate: Audio sample rate in Hz (Gemma 4 only).
    audio_seq_length: Maximum audio sequence length (Gemma 4 only).
    last_state: Last state of the sampler, automatically handled by the sampler,
      but exposed for power users to access the logits, cache, ... or initialize
      the sampler.
    turns: Track the conversation.
  """

  # TODO(epot): Custom repr to avoid displaying the full weights.

  model: _transformer_like.TransformerLike
  params: _common.Params = dataclasses.field(repr=False)
  multi_turn: bool = False
  print_stream: bool | dialog.Stream = False
  tokenizer: _tokenizer.Tokenizer = None  # pytype: disable=annotation-type-mismatch
  sampling: _sampling.SamplingMethod = dataclasses.field(
      default_factory=_sampling.Greedy
  )
  forbidden_tokens: Sequence[str | int] | None = None
  stop_tokens: Sequence[str | int] | None = None
  # TODO(epot): Support and test rolling cache.
  # TODO(epot): Add a property to show how much of the cache is used.
  cache_length: int | None = 4096
  max_out_length: int = 2048

  # Gemma 4-specific fields (ignored for non-Gemma4 models).
  pad_length: None | int | tuple[int, ...] = (256, 512, 1024)
  patch_size: int = 16
  max_soft_tokens: int = 1120
  pooling_kernel_size: int = 3
  audio_sample_rate: int = 16000
  audio_seq_length: int = 750

  # Internal variables, but exposed for power users.

  # Last state of the sampler.
  last_state: _sampler_loop.SamplingState = dataclasses.field(  # pytype: disable=annotation-type-mismatch
      default=None, repr=False
  )
  turns: list[_template.Turn] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    if self.turns:
      raise ValueError(
          'Currently initializing the sampler with previous conversation is not'
          ' supported.'
      )
    # No state by default.
    object.__setattr__(self, 'last_state', None)

    # Set the tokenizer if not provided.
    if self.tokenizer is None:
      object.__setattr__(self, 'tokenizer', self._inner_sampler.tokenizer)

  @functools.cached_property
  def _is_gemma4(self) -> bool:
    """Returns True if the model is a Gemma 4 model."""
    return getattr(self.model, 'INFO', None) is not None and (
        self.model.INFO.tokenizer_version == 4
    )

  @functools.cached_property
  def _inner_sampler(self) -> _sampler.Sampler | _gemma4_sampler.Gemma4Sampler:
    """Returns the underlying sampler, auto-detecting the model type."""
    if self._is_gemma4:
      return _gemma4_sampler.Gemma4Sampler(
          model=self.model,
          params=self.params,
          tokenizer=self.tokenizer,
          sampling=self.sampling,
          forbidden_tokens=self.forbidden_tokens,
          stop_tokens=self.stop_tokens,
          cache_length=self.cache_length,
          max_out_length=self.max_out_length,
          pad_length=self.pad_length,
          patch_size=self.patch_size,
          max_soft_tokens=self.max_soft_tokens,
          pooling_kernel_size=self.pooling_kernel_size,
          audio_sample_rate=self.audio_sample_rate,
          audio_seq_length=self.audio_seq_length,
      )
    else:
      return _sampler.Sampler(
          model=self.model,
          params=self.params,
          tokenizer=self.tokenizer,
          sampling=self.sampling,
          forbidden_tokens=self.forbidden_tokens,
          stop_tokens=self.stop_tokens,
          cache_length=self.cache_length,
          max_out_length=self.max_out_length,
      )

  # Keep backwards-compatible property name for non-Gemma4 users.
  @property
  def sampler(self) -> _sampler.Sampler:
    """Returns the underlying sampler (for backwards compatibility)."""
    inner = self._inner_sampler
    if isinstance(inner, _sampler.Sampler):
      return inner
    raise AttributeError(
        'The `sampler` property is not available for Gemma 4 models. '
        'Use `gemma4_sampler` instead.'
    )

  @property
  def gemma4_sampler(self) -> _gemma4_sampler.Gemma4Sampler:
    """Returns the underlying Gemma4Sampler (Gemma 4 models only)."""
    inner = self._inner_sampler
    if isinstance(inner, _gemma4_sampler.Gemma4Sampler):
      return inner
    raise AttributeError(
        'The `gemma4_sampler` property is not available for non-Gemma4 models. '
        'Use `sampler` instead.'
    )

  def _sample(
      self,
      prompt_text: str,
      *,
      images,
      audio,
      audio_lengths,
      sampling,
      max_new_tokens,
      rng,
      last_state,
      stream,
      sharding,
  ):
    """Dispatches to the correct underlying sampler."""
    if self._is_gemma4:
      return self.gemma4_sampler.sample(
          prompt_text,
          images=images,
          audio=audio,
          audio_lengths=audio_lengths,
          sampling=sampling,
          max_new_tokens=max_new_tokens,
          rng=rng,
          return_state=True,
          last_state=last_state,
          sharding=sharding,
      )
    else:
      return self.sampler.sample(  # pytype: disable=wrong-arg-types
          prompt_text,
          images=images,
          sampling=sampling,
          max_new_tokens=max_new_tokens,
          rng=rng,
          return_state=True,
          last_state=last_state,
          stream=bool(stream),
      )

  def chat(
      self,
      prompt: str | dialog.Conversation,
      *,
      images: list[np.ndarray | Image.Image] | UInt8['N? H W C'] | None = None,
      audio: list[np.ndarray] | None = None,
      audio_lengths: list[int] | None = None,
      sampling: _sampling.SamplingMethod | None = None,
      rng: PRNGKeyLike | None = None,
      max_new_tokens: int | None = None,
      multi_turn: bool | None = None,
      print_stream: bool | dialog.Stream | None = None,
      is_legacy_tool_answer: bool = False,
      sharding: kd.sharding.ShardingTree | None = None,
  ) -> str:
    """Samples a string from the model.

    The API always expects new gemma format tokens (``<|image|>``,
    ``<|audio|>``, etc.). The ``dialog`` library automatically
    converts to the correct format for the underlying model.

    Example:

    ```python
    # Text-only (all Gemma versions):
    output = sampler.chat('Write a poem about cats.')

    # With images (Gemma 4 or Gemma 3):
    output = sampler.chat(
        'Describe this image <|image|>.',
        images=[image1],
    )

    # With audio (Gemma 4):
    output = sampler.chat(
        'Transcribe this audio <|audio|>.',
        audio=[audio_array],
    )
    ```

    Args:
      prompt: Prompt to sample from. Can be a single string or a
        `dialog.Conversation` object.
      images: Images for the prompt. For Gemma 4: list of raw numpy arrays or
        PIL Images (variable aspect ratio). For Gemma 2/3: a batched uint8
        array.
      audio: List of audio arrays (Gemma 4 only).
      audio_lengths: List of audio lengths (Gemma 4 only).
      sampling: Sampling method to use. If given, will override the default
        sampling method.
      rng: Seed to use for the sampling method. If `None`, a random seed is
        used. Can be a seed `int` or a `jax.random.PRNGKey` object.
      max_new_tokens: If given, will stop sampling after this many tokens. Used
        for quicker iterations when debugging. By default, sample until the
        end-of-turn token is found, or until the `max_out_length` buffer is
        filled.
      multi_turn: If `True`, reuse the previous turns as context. Overrides the
        `multi_turn` attribute.
      print_stream: If `True`, will print the sampled output as it is generated.
        Overrides the `print_stream` attribute.
      is_legacy_tool_answer: When `True`, indicates that the model has emitted
        `<eos>` rather than `<|tool_response>`, thus this needs to be corrected.
        (this is an internal variable that should never be explicitly set).
      sharding: Sharding tree (Gemma 4 only).

    Returns:
      The sampled output.
    """
    if multi_turn is None:
      multi_turn = self.multi_turn
    stream = self.initialize_stream(print_stream)

    if not multi_turn:
      # Non-multi-turn, erase the previous conversations.
      object.__setattr__(self, 'last_state', None)
      object.__setattr__(self, 'turns', [])

    # --- Unified prompt formatting via `dialog` library ---
    if isinstance(prompt, str):
      if _has_legacy_gemma3_format(prompt):
        if self._is_gemma4:
          raise ValueError(
              'Detected deprecated Gemma 3 format tokens (e.g. '
              '<start_of_image>) in the prompt, but the model '
              'is Gemma 4 which uses <|image|>, <|audio|>, etc. '
              'Please use Gemma 4 format tokens instead.'
          )
        else:
          warnings.warn(
              'Detected deprecated Gemma 3 format tokens (e.g.'
              ' <start_of_image>) in the prompt, but the api expects Gemma 4'
              ' format tokens. The legacy format is deprecated and will be'
              ' removed in a future release.',
              DeprecationWarning,
              stacklevel=2,
          )
          prompt = dialog.Conversation(dialog.User(prompt))
      else:
        if not self._is_gemma4:
          prompt = prompt.replace('<|image|>', '<start_of_image>')
          if prompt.find('<|audio|>') != -1:
            raise ValueError(
                'Audio input is not supported for non-Gemma4 models.'
            )
        prompt = dialog.Conversation(dialog.User(prompt))
    elif not isinstance(prompt, dialog.Conversation):
      raise TypeError(f'Unsupported prompt type: {type(prompt)}')

    last_state = self.last_state
    if is_legacy_tool_answer:
      # This means the previous model turn ended with an EOS token rather than
      # the expected `<|tool_response>`
      last_state = _remove_eos_token(last_state, tokenizer=self.tokenizer)

    prompt_text = prompt.as_text(
        format=self.tokenizer.FORMAT,
        add_tool_response_tag_after_call=not is_legacy_tool_answer,
    )

    # --- Dispatch to the correct sampler ---
    out = self._sample(
        prompt_text,
        images=images,
        audio=audio,
        audio_lengths=audio_lengths,
        sampling=sampling,
        max_new_tokens=max_new_tokens,
        rng=rng,
        last_state=last_state,
        stream=stream,
        sharding=sharding,
    )

    # In streaming mode, the output is an iterator, yielding tokens one at a
    # time.
    if stream:
      out = _print_stream(out, stream=stream)
    assert isinstance(out, _sampler.SamplerOutput)  # For pytype.
    assert isinstance(out.text, str)  # For pytype.

    # TODO(epot): Remove the <turn|> end-of-turn token.

    # Save the raw turns text (unformatted).
    # Only save the user turn after the sampling has successfully finished.
    self.turns.append(_template.Prompt(prompt_text))
    self.turns.append(_template.Response(out.text))
    object.__setattr__(self, 'last_state', out.state)
    return out.text  # pytype: disable=bad-return-type

  def initialize_stream(
      self,
      stream: dialog.Stream | bool | None,
  ) -> dialog.Stream | None:
    """Initializes a stream for the sampler."""
    if stream is None:
      stream = self.print_stream

    if stream is False:  # pylint: disable=g-bool-id-comparison
      return None
    elif stream is True:  # pylint: disable=g-bool-id-comparison
      stream = dialog.Stream()
      if epy.is_notebook():
        dialog.Model(stream).show()
      return stream
    elif isinstance(stream, dialog.Stream):
      return stream
    else:
      raise ValueError(f'Unexpected stream type: {type(stream)}')

  @property
  def conversation(self) -> dialog.Conversation:
    """Returns the conversation."""
    return dialog.Conversation(''.join(t.text for t in self.turns))


# Legacy Gemma 3 tokens to detect in user prompts.
_LEGACY_GEMMA3_TOKENS = ('<start_of_image>',)


def _has_legacy_gemma3_format(prompt: str) -> bool:
  """Returns True if the prompt contains legacy Gemma 3 format tokens."""
  return any(token in prompt for token in _LEGACY_GEMMA3_TOKENS)


def _remove_eos_token(
    state: _sampler_loop.SamplingState,
    tokenizer: _tokenizer.Tokenizer,
) -> _sampler_loop.SamplingState:
  """Removes the EOS token from the sampling state."""
  cache_info = state.cache_info.set_end_index(state.cache_info.end_index - 1)
  return dataclasses.replace(
      state,
      step=state.step - 1,
      # done is True and last_token is EOS => False
      # Otherwise, keep the same.
      done=state.done ^ (state.last_token == tokenizer.special_tokens.EOS),
      last_token_pos=state.last_token_pos - 1,
      cache=cache_info.cache,
  )


def _print_stream(
    out: Iterator[_sampler.SamplerOutput],
    *,
    stream: dialog.Stream,
) -> _sampler.SamplerOutput:
  """Prints the streaming output."""
  text_tokens = []

  for state in out:
    print_(stream, state.text)

    text_tokens.append(state.text)
    if (
        state.text == '<end_of_turn>' or state.text == '<turn|>'
    ):  # Last token is not printed.
      continue
  out = dataclasses.replace(state, text=''.join(text_tokens))  # pylint: disable=undefined-variable,undefined-loop-variable
  return out


def print_(
    stream: dialog.Stream,
    text: str,
) -> None:
  if epy.is_notebook():
    stream.add(text)
  else:
    print(text, end='', flush=True)
