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

"""Base class to orchestrate tools."""

from __future__ import annotations

import abc
from collections.abc import Iterator
import contextlib
import dataclasses
import functools
from typing import Any

import dialog
from etils import epy

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
  import mcp
  # pylint: enable=g-import-not-at-top # pytype: enable=import-error


@dataclasses.dataclass(frozen=True, kw_only=True)
class ToolHandlerBase(epy.ContextManager, abc.ABC):
  """Base class to orchestrate tools."""

  # Used if ToolHandlerBase should be used as a context manager.
  # See `McpToolHandler` for an example.
  is_active: bool = dataclasses.field(default=False, repr=False, init=False)

  def __post_init__(self):
    object.__setattr__(self, 'is_active', False)

  def add_system_prompt(
      self, prompt: dialog.Conversation
  ) -> dialog.Conversation:
    """Returns the system prompt."""
    if isinstance(prompt[0], dialog.System):
      system_prompt = prompt[0]
      prompt = prompt[1:]
    else:
      system_prompt = dialog.System()
    system_prompt += [dialog.Tool(t) for t in self.tools]
    return system_prompt + prompt

  def maybe_execute_tool(
      self, model_output: str
  ) -> list[dialog.ToolResponse] | None:
    """Parses the model output answer and call the associated tool if needed."""

    if not model_output.endswith((
        dialog.Tags.TOOL_RESPONSE.open,
        # There's a bug in Gemma nano 4 (2B and 4B only) which predict `<eos>`
        # rather than `<|tool_response>`. Thus we still need to support those.
        dialog.Tags.TOOL_CALL.close,
    )):
      return None
    model_output = model_output.removesuffix(dialog.Tags.TOOL_RESPONSE.open)
    model_output = (
        f'{dialog.Tags.TURN.open}model\n{model_output}{dialog.Tags.TURN.close}'
    )
    conv = dialog.ConversationStr(model_output).as_conversation()
    (turn,) = conv
    tool_responses = []
    for chunk in turn:
      if isinstance(chunk, dialog.ToolCall):
        data = chunk.data.full_json
        response = self.call_tool(**data)
        response.name = chunk.data.name
        response = _normalize_response(response)
        tool_responses.append(dialog.ToolResponse(response))
    return tool_responses

  @functools.cached_property
  def tools(self) -> list[mcp.types.Tool]:
    """Returns the tools."""
    with self:
      return self._tools()

  def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
    """Calls the tool."""
    with self:
      return self._call_tool(name, arguments)

  def _tools(self):
    """Returns the tools."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
    """Calls the tool."""
    raise NotImplementedError()

  def __contextmanager__(self) -> Iterator[None]:
    """Context manager to activate the tool handler."""
    if self.is_active:  # Do not recurse if already active.
      yield
    else:
      try:
        object.__setattr__(self, 'is_active', True)
        with self._activate():
          yield
      finally:
        object.__setattr__(self, 'is_active', False)

  @contextlib.contextmanager
  def _activate(self) -> Iterator[None]:
    """Activate the tool handler."""
    del self
    yield


def _normalize_response(
    response: mcp.types.CallToolResult,
) -> mcp.types.CallToolResult:
  """Normalizes the response."""
  if response.structuredContent is not None:
    return response

  # Assume the response is a single text block.
  assert len(response.content) == 1
  assert response.content[0].type == 'text'

  content = response.content[0].text
  if response.isError:
    structured_content = {'error': content}
  else:
    structured_content = {'result': content}
  response.structuredContent = structured_content
  return response
