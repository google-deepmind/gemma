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

"""Tool use sampler."""

from __future__ import annotations

import dataclasses

from etils import epy
from gemma.gm.text import _chat_sampler
from gemma.gm.text import _sampling
from gemma.gm.tools import _manager as _manager_lib
from gemma.gm.tools import _tools
from kauldron.typing import PRNGKeyLike, UInt8  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class ToolSampler(_chat_sampler.ChatSampler):
  """Sampler with tool support.

  Example:

  ```python
  sampler = gm.text.ToolSampler(
      model=model,
      params=params,
      tools=[gm.tools.Calculator()],
  )
  sampler.chat('What is 25 times 4 plus 10?')
  ```

  Attributes:
    tools: List of tools to use.
    manager_cls: Allow to customize how the system prompt and tools are handled.
  """

  tools: list[_tools.Tool] = dataclasses.field(default_factory=list)
  manager_cls: type[_manager_lib.ToolManagerBase] = (
      _manager_lib.OneShotToolManager
  )

  # Manager class is mutable (as states can be stateful), so reset for every new
  # conversation.
  _manager: _manager_lib.ToolManagerBase = dataclasses.field(
      init=False, repr=False
  )

  def __post_init__(self):
    super().__post_init__()
    # Normalize `Tool` to `list[Tool]`.
    if isinstance(self.tools, _tools.Tool):
      object.__setattr__(self, 'tools', [self.tools])

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
    """Sampler which supports tool use.

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
    """
    if print_stream is None:
      print_stream = self.print_stream

    if not self.turns or not multi_turn:  # First turn
      # Initialize the manager.
      manager = self.manager_cls(tools=list(self.tools))
      object.__setattr__(self, '_manager', manager)

      # Add the system prompt.
      prompt = self._manager.system_prompt + prompt

    model_output = super().chat(
        prompt,
        images=images,
        sampling=sampling,
        rng=rng,
        max_new_tokens=max_new_tokens,
        multi_turn=multi_turn,
        print_stream=print_stream,
    )

    # Detect if the model requested a tool call, and execute it.
    tool_answer = self._manager.maybe_execute_tool(model_output)

    # If model requested a tool call, execute it and fetch the response
    # back to the model.
    if tool_answer:
      if print_stream:
        print(flush=True)  # New line
        _plot_separator()
        print(tool_answer.text, flush=True)
        _plot_separator()
      # TODO(epot): Should use `_template.ToolTurn` or similar.
      return self.chat(
          tool_answer.text,
          images=tool_answer.images,
          multi_turn=True,
          sampling=sampling,
          rng=rng,
          print_stream=print_stream,
      )
    else:
      return model_output


def _plot_separator() -> None:
  if epy.is_notebook():
    import IPython.display  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    IPython.display.display(IPython.display.HTML('<hr>'))
  else:
    print('------------------------------------------------------------------')
