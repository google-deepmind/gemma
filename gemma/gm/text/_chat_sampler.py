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

"""Chat sampler."""

from collections.abc import Sequence
import dataclasses
import functools

from gemma import params as params_lib
from gemma.gm.nn import _transformer
from gemma.gm.text import _sampler
from gemma.gm.text import _sampler_call
from gemma.gm.text import _sampling
from gemma.gm.text import _template
from gemma.gm.text import _tokenizer
from kauldron.typing import PRNGKeyLike, UInt8  # pylint: disable=g-multiple-import,g-importing-member

@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class ChatSampler:
    """Chat sampler.
    
    This sampler:
    * Is stateful (the KV-cache state is automatically handled)
    * Automatically formats the prompt with `<start_of_turn>` and `<end_of_turn>` tokens.
    * Filters the `<end_of_turn>` tokens from the output.
    
    Attributes:
      model: Gemma transformer model.
      params: Model parameters.
      multi_turn: If `True`, reuse the previous turns as context.
      tokenizer: Tokenizer.
      sampling: Sampling method to use. Default to greedy sampling.
      forbidden_tokens: List of tokens that are forbidden to be generated.
      cache_length: Cache length to use. This is the maximum number of tokens.
      max_out_length: Length of the output buffer for a single turn.
      last_state: Last state of the sampler.
      turns: The current conversation.
    """

    model: _transformer.Transformer
    params: params_lib.Params = dataclasses.field(repr=False)
    multi_turn: bool = False
    tokenizer: _tokenizer.Tokenizer = None  # pytype: disable=annotation-type-mismatch
    sampling: _sampling.SamplingMethod = dataclasses.field(default_factory=_sampling.Greedy)
    forbidden_tokens: Sequence[str | int] | None = None
    cache_length: int | None = 4096
    max_out_length: int = 2048
    last_state: _sampler_call.SamplingState = dataclasses.field(default=None, repr=False)
    turns: list[_template.Turn] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.turns:
            raise ValueError(
                'Currently initializing the sampler with previous conversation is not supported.'
            )
        object.__setattr__(self, 'last_state', None)
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
            cache_length=self.cache_length,
        )

    def resize_cache(self, new_cache_length: int):
        """Resize the cache length dynamically."""
        object.__setattr__(self, 'cache_length', new_cache_length)
        # Reinitialization
        object.__setattr__(self, 'sampler', self.sampler)

    def chat(
        self,
        prompt: str,
        *,
        images: UInt8['N? H W C'] | None = None,
        sampling: _sampling.SamplingMethod | None = None,
        rng: PRNGKeyLike | None = None,
        max_new_tokens: int | None = None,
        multi_turn: bool | None = None,
    ) -> str:
        """Samples a string from the model."""
        if multi_turn is None:
            multi_turn = self.multi_turn

        # Save the unformatted prompt
        unformatted_prompt = prompt

        # Format the prompt
        prompt = _template.PROMPT.format(prompt)

        if not multi_turn:
            # Reset state and conversation history if not multi-turn
            object.__setattr__(self, 'last_state', None)
            object.__setattr__(self, 'turns', [])

        if self.last_state is not None and images is not None:
            raise NotImplementedError(
                'Multi-turn with images on the second turn is not supported yet.'
            )

        out = self.sampler.sample(
            prompt,
            images=images,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            rng=rng,
            return_state=True,
            last_state=self.last_state,
        )

        assert isinstance(out.text, str)  # For pytype.
        model_output = out.text.removesuffix('<end_of_turn>')

        # Save conversation turns
        self.turns.append(_template.UserTurn(unformatted_prompt))
        self.turns.append(_template.ModelTurn(model_output))
        object.__setattr__(self, 'last_state', out.state)
        return model_output
