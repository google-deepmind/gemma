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

"""Tool use sampler."""

from __future__ import annotations

import dataclasses

import dialog
from gemma.gm.text import _chat_sampler
from gemma.gm.text import _sampling
from gemma.gm.tools import _manager as _manager_lib
from kauldron.typing import PRNGKeyLike, UInt8  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class ToolSampler(_chat_sampler.ChatSampler):
  """Sampler with tool support.

  Example:

  ```python
  sampler = gm.text.ToolSampler(
      model=model,
      params=params,
      tool_handler=fastmcp.Client(server),
  )
  sampler.chat('Do you see an issue with my ~/.bashrc ?')
  ```

  Attributes:
    tool_handler: Allow to customize how the system prompt and tools are
      handled.
  """

  tool_handler: _manager_lib.ToolHandlerBase

  def chat(
      self,
      prompt: str | dialog.Conversation,
      *,
      images: UInt8['N? H W C'] | None = None,
      sampling: _sampling.SamplingMethod | None = None,
      rng: PRNGKeyLike | None = None,
      max_new_tokens: int | None = None,
      multi_turn: bool | None = None,
      print_stream: bool | dialog.Stream | None = None,
      is_legacy_tool_answer: bool = False,
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
      is_legacy_tool_answer: When `True`, indicates that the model has emitted
        `<eos>` rather than `<|tool_response>`, thus this needs to be corrected.
        (this is an internal variable that should never be explictly set).

    Returns:
      The sampled output.
    """
    if multi_turn is None:
      multi_turn = self.multi_turn

    stream = self.initialize_stream(print_stream)

    with self.tool_handler:
      if isinstance(prompt, str):
        prompt = dialog.Conversation(dialog.User(prompt))
      if not self.conversation or not multi_turn:  # First turn
        # Add the system prompt.
        prompt = self.tool_handler.add_system_prompt(prompt)

      model_output = super().chat(
          prompt,
          images=images,
          sampling=sampling,
          rng=rng,
          max_new_tokens=max_new_tokens,
          multi_turn=multi_turn,
          print_stream=stream,
          is_legacy_tool_answer=is_legacy_tool_answer,
      )

      # Detect if the model requested a tool call, and execute it.
      tool_answer = self.tool_handler.maybe_execute_tool(model_output)

      # If model requested a tool call, execute it and fetch the response
      # back to the model.
      if tool_answer:
        # The nano models (2B and 4B), sometimes emit `<eos>` instead of
        # `<|tool_response>`. We need to:
        # * Mutate the cache to remove the last `<eos>` token.
        # * Ensure that `<|tool_response>` is provided to the next model turn.
        if not model_output.endswith(dialog.Tags.TOOL_RESPONSE.open):
          is_legacy_tool_answer = True
        else:
          is_legacy_tool_answer = False

        tool_answer = dialog.Conversation(dialog.Model(tool_answer))
        if stream:
          text = tool_answer.as_text(
              add_tool_response_tag_after_call=not is_legacy_tool_answer
          )
          _chat_sampler.print_(stream, text)

        # TODO(epot): Should use `_template.ToolTurn` or similar.
        return self.chat(
            tool_answer,
            # images=tool_answer.images,
            multi_turn=True,
            sampling=sampling,
            rng=rng,
            print_stream=stream,
            is_legacy_tool_answer=is_legacy_tool_answer,
        )
      else:
        return model_output
