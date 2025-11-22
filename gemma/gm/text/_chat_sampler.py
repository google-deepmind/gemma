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

"""Chat sampler."""

from collections.abc import Iterator, Sequence
import dataclasses
import functools

# from gemma.gm.data import _functional
from gemma.gm.nn import _transformer_like
from gemma.gm.text import _sampler
from gemma.gm.text import _sampler_loop
from gemma.gm.text import _sampling
from gemma.gm.text import _template
from gemma.gm.text import _tokenizer
from gemma.gm.typing import _common
# from gemma.gm.vision import _token_utils
from kauldron.typing import PRNGKeyLike, UInt8  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class ChatSampler:
  """Chat sampler.

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

  This sampler:

  * Is statefull (the KV-cache state is automatically handled)
  * Automatically format the prompt with `<start_of_turn>` and
    `<end_of_turn>` tokens, adds the BOS (beginning of sequence) token. And
    filter the `<end_of_turn>` tokens from the output.

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
      used to avoid trigering a jit recompilation. Shouldn't be changed unless
      you have a task where the model generates really long outputs.
    last_state: Last state of the sampler, automatically handled by the sampler,
      but exposed for power users to access the logits, cache, ... or initialize
      the sampler.
    turns: The current conversation.
  """
  # TODO(epot): Custom repr to avoid displaying the full weights.

  model: _transformer_like.TransformerLike
  params: _common.Params = dataclasses.field(repr=False)
  multi_turn: bool = False
  print_stream: bool = False
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
      object.__setattr__(self, 'tokenizer', self.sampler.tokenizer)

  @functools.cached_property
  def sampler(self) -> _sampler.Sampler:
    """Returns the underlying sampler."""
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

  def chat(
      self,
      prompt: str,
      *,
      images: UInt8['N? H W C'] | None = None,
      sampling: _sampling.SamplingMethod | None = None,
      rng: PRNGKeyLike | None = None,
      max_new_tokens: int | None = None,
      multi_turn: bool | None = None,
      print_stream: bool | None = None,
  ) -> str:
    # pylint: disable=g-docstring-quotes
    '''Samples a string from the model.

    Example:

    ```python
    prompt = """I'm hesitating between those two options:

    Option 1: <start_of_image>
    Option 2: <start_of_image>

    Any thoughts ?"""
    sampler.sample(prompt, images=[image1, image2]))
    ```

    Args:
      prompt: Prompt to sample from. Can be a single string or a list of
        strings.
      images: Images for the prompt. The position where the image should be
        inserted in the prompt is determined by the `<start_of_image>` token in
        the prompt.
      sampling: Sampling method to use. If given, will override the default
        sampling method.
      rng: Seed to use for the sampling method. If `None`, a random seed is
        used. Can be a seed `int` or a `jax.random.PRNGKey` object.
      max_new_tokens: If given, will stop sampling after this many tokens. Used
        for quicker iterations when debugging. By default, sample until the
        `<end_of_turn>` token is found, or until the `max_out_length` buffer is
        filled.
      multi_turn: If `True`, reuse the previous turns as context. Overrites the
        `multi_turn` attribute.
      print_stream: If `True`, will print the sampled output as it is generated.
        Overrites the `multi_turn` attribute.

    Returns:
      The sampled output.
    '''
    if multi_turn is None:
      multi_turn = self.multi_turn
    if print_stream is None:
      print_stream = self.print_stream

    # Save the prompt (before formatting!).
    unformatted_prompt = prompt

    # Format the prompt.
    prompt = _template.PROMPT.format(prompt)

    if not multi_turn:
      # Non-multi-turn, erase the previous conversations.
      # Clear state memory ? (in theory, should be auto-cleared by Jax), and
      # might interfere if user manually use the cache.
      object.__setattr__(self, 'last_state', None)
      object.__setattr__(self, 'turns', [])

    out = self.sampler.sample(  # pytype: disable=wrong-arg-types
        prompt,
        images=images,
        sampling=sampling,
        max_new_tokens=max_new_tokens,
        rng=rng,
        return_state=True,
        last_state=self.last_state,
        stream=print_stream,
    )

    # In streaming mode, the output is an iterator, yielding tokens one at a
    # time.
    if print_stream:
      out = _print_stream(out)
    assert isinstance(out, _sampler.SamplerOutput)  # For pytype.
    assert isinstance(out.text, str)  # For pytype.
    model_output = out.text.removesuffix('<end_of_turn>')  # pytype: disable=attribute-error

    # Save the turns (after un-formatting).
    # Only save the user turn after the sampling has successfully finished.
    self.turns.append(_template.UserTurn(unformatted_prompt))
    self.turns.append(_template.ModelTurn(model_output))
    object.__setattr__(self, 'last_state', out.state)
    return model_output


def _print_stream(
    out: Iterator[_sampler.SamplerOutput],
) -> _sampler.SamplerOutput:
  """Prints the streaming output."""
  text_tokens = []
  for state in out:
    text_tokens.append(state.text)
    if state.text == '<end_of_turn>':  # Last token is not printed.
      continue
    print(state.text, end='', flush=True)
  out = dataclasses.replace(state, text=''.join(text_tokens))  # pylint: disable=undefined-variable,undefined-loop-variable
  return out
