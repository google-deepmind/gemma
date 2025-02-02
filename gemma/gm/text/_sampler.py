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
import typing

from gemma import params as params_lib
from gemma.gm.nn import _transformer
from gemma.gm.text import _sampler_impl
from gemma.gm.text import _tokenizer


# TODO(epot):
# * Supports sampling with token_ids (`list[int]` / jnp) rather than `str`
#   so the same data pipeline can be used for both training and sampling.
# * Mode which queue the prompts and compute them asynchronously ?
# * Mode which yields tokens as they get predicted ?


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sampler:
  """Sampler.

  ```python
  sampler = Sampler(
      model=model,
      params=params,
      tokenizer=tokenizer,
  )

  output = sampler.sample(prompt)
  ```

  Attributes:
    model: Gemma transformer model.
    params: Model parameters.
    tokenizer: Tokenizer.
    cache_length: Size of the attention cache.
  """

  model: _transformer.Transformer
  params: params_lib.Params
  tokenizer: _tokenizer.Tokenizer

  # TODO(epot): Add a `max_length` argument to the `sample()` method.
  @typing.overload
  def sample(
      self,
      prompt: str,
      *,
      max_new_tokens: int = ...,
  ) -> str:
    ...

  @typing.overload
  def sample(
      self,
      prompt: Sequence[str],
      *,
      max_new_tokens: int = ...,
  ) -> list[str]:
    ...

  def sample(
      self,
      prompt,
      *,
      max_new_tokens=200,
  ):
    """Samples a string from the model.

    Args:
      prompt: Prompt to sample from. Can be a single string or a list of
        strings.
      max_new_tokens: Maximum number of new tokens to generate. The transformer
        will process `input_length + max_new_tokens`.

    Returns:
      The sampled output.
    """
    if isinstance(prompt, str):
      is_single_prompt = True
      prompt = [prompt]
    else:
      is_single_prompt = False

    output = self._sampler(prompt, total_generation_steps=max_new_tokens)
    output = output.text

    if is_single_prompt:
      (output,) = output
      return output
    else:
      return output

  @functools.cached_property
  def _sampler(self) -> _sampler_impl.Sampler:
    return _sampler_impl.Sampler(
        transformer=self.model,
        tokenizer=self.tokenizer,
        params=self.params,
        # TODO(epot): This should be dynamically infered from
        # `max_length=` in `def sample()` ? No need to allocate extra memory
        # when the sequence is smaller.
        cache_length=1024,
    )
