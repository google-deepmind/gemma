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

"""Base class to orchestrate tools."""

from __future__ import annotations

import abc
import dataclasses
import functools
import json
from typing import Any

from etils import epy
from gemma.gm.tools import _tools


@dataclasses.dataclass(frozen=True, kw_only=True)
class ToolManagerBase(abc.ABC):
  """Base class to orchestrate tools."""

  tools: list[_tools.Tool]

  @property
  @abc.abstractmethod
  def system_prompt(self) -> str:
    """Returns the preprompt for the tool manager."""
    raise NotImplementedError()

  @abc.abstractmethod
  def maybe_execute_tool(self, model_output: str) -> _tools.ToolOutput | None:
    """Parses the model output answer and call the associated tool if needed."""
    raise NotImplementedError()

  @functools.cached_property
  def name_to_tool(self) -> dict[str, _tools.Tool]:
    return {tool.name: tool for tool in self.tools}

  def update_tools(self, tools: list[_tools.Tool]) -> None:
    """Replace the current tools by the new ones."""
    object.__setattr__(self, 'tools', tools)
    # Reset the cached_property.
    if 'name_to_tool' in self.__dict__:
      object.__delattr__(self, 'name_to_tool')


_SYSTEM_PROMPT = """\
You are a helpful assistant that can use tools.
If you need to use a tool, output it in the specified format.
IMPORTANT: Do NOT write `[Tool result: ...]` !!! Instead end your turn and
let the user talk.


Available tools:

{}

Now, answer the following:

"""

_TOOL_CALL_TEMPLATE = """\
**{tool_name}**:

Description: {description}
Call format: {call_format}
Example:
{example}
"""


@dataclasses.dataclass(frozen=True, kw_only=True)
class OneShotToolManager(ToolManagerBase):
  """Tool manager that instructs the model through one-shot prompting."""

  @functools.cached_property
  def system_prompt(self) -> str:
    """Returns the preprompt for the tool manager."""

    lines = epy.Lines()
    for tool in self.tools:
      lines += format_tool_instructions(tool)

    return _SYSTEM_PROMPT.format(lines.join())

  def maybe_execute_tool(self, model_output: str) -> _tools.ToolOutput | None:
    """Executes the tool if the model output is a tool call."""
    tool_kwargs = _parse_tool_call(model_output)
    if not tool_kwargs:
      return None

    tool_name = tool_kwargs.pop('tool_name', None)
    # If the model output contained JSON but it wasn't a tool call, ignore it.
    if not isinstance(tool_name, str) or not tool_name:
      return None
    if tool_name not in self.name_to_tool:
      return _tools.ToolOutput(
          text=f'Unknown (or unregistered) tool: {tool_name}.'
      )
    tool = self.name_to_tool[tool_name]
    tool_result = tool.call(**tool_kwargs)

    # Normalize `str` to `ToolOutput`.
    if not isinstance(tool_result, _tools.ToolOutput):
      tool_result = _tools.ToolOutput(text=tool_result)

    # Tools can also interact with the manager (e.g. to register, update,...
    # tools)
    if tool_result.update_tools:
      self.update_tools(tool_result.update_tools(self.tools))

    text = _format_tool_result(result=tool_result.text)
    tool_result = dataclasses.replace(tool_result, text=text)
    return tool_result


def _parse_tool_call(model_output: str) -> dict[str, Any] | None:
  """Parses a tool call dict from the model output.

  The model may emit arbitrary JSON (e.g. when asked to output structured data).
  We should only treat JSON as a tool call if it looks like a tool call (i.e.
  it is a JSON object containing a `tool_name` field).
  """
  # Avoid regex-based extraction: it is brittle (greedy brace matching).
  decoder = json.JSONDecoder()
  for i, ch in enumerate(model_output):
    if ch != '{':
      continue
    try:
      obj, _ = decoder.raw_decode(model_output, i)
    except json.JSONDecodeError:
      continue
    # Only consider JSON objects that look like a tool call.
    if isinstance(obj, dict) and isinstance(obj.get('tool_name'), str):
      return obj
  return None


def _format_tool_example(
    *,
    example: _tools.Example,
    tool: _tools.Tool,
) -> str:
  """Formats a tool example."""

  lines = epy.Lines()
  lines += f'User: {example.query}'
  lines += 'Assistant:'
  if example.thought:
    lines += f'Thought: {example.thought}'
  lines += _format_tool_call(tool_kwargs=example.tool_kwargs, tool=tool)
  lines += _format_tool_result(result=example.result)
  lines += example.answer
  return lines.join()


def format_tool_instructions(tool: _tools.Tool) -> str:
  return _TOOL_CALL_TEMPLATE.format(
      tool_name=tool.name,
      description=tool.DESCRIPTION,
      call_format=_format_tool_call(
          tool_kwargs=tool.EXAMPLE.tool_kwargs_doc, tool=tool
      ),
      example=_format_tool_example(example=tool.EXAMPLE, tool=tool),
  )


def _format_tool_call(
    *,
    tool_kwargs: dict[str, str],
    tool: _tools.Tool,
) -> str:
  return json.dumps({'tool_name': tool.name, **tool_kwargs})


def _format_tool_result(result: str) -> str:
  return f'[Tool result: {result}]'
