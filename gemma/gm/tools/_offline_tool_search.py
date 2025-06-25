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

"""Tool search."""

from __future__ import annotations

import dataclasses

from etils import epy
from gemma.gm.tools import _manager
from gemma.gm.tools import _tools


@dataclasses.dataclass(frozen=True, kw_only=True)
class OfflineToolSearch(_tools.Tool):
  """Tool which can search for other tools.

  This tool does not have internet access. The available list of tools
  has to be explicitly provided at initialization.

  Usage:

  ```python
  tool = gm.tools.OfflineToolSearch(
      tools=[
          gm.tools.Calculator(),
          gm.tools.FileExplorer(),
          ...,
      ],
  )
  ```
  """

  tools: list[_tools.Tool]

  DESCRIPTION = (
      'Search and register new tools. Once a tool has been searched, it can be'
      ' used directly afterwards.'
  )
  EXAMPLE = _tools.Example(
      query='Is it sunny in Paris?',
      thought='I need a weather tool to answer this question.',
      tool_kwargs={'query': 'weather'},
      tool_kwargs_doc={'query': '<SEARCH TAGS>'},
      result='Found 1 tool(s) with the following specs: [...]',
      answer='Now I can call the weather tool, to answer the user question.',
  )

  def call(self, query: str) -> str | _tools.ToolOutput:  # pytype: disable=signature-mismatch
    """Search for a tool."""
    found_tools = [
        tool for tool in self.tools if _match_keywords(query, tool.KEYWORDS)
    ]

    if not found_tools:
      return 'No tools found. Maybe try a different query ?'

    tool_instructions = epy.Lines()

    for tool in found_tools:
      tool_instructions += _manager.format_tool_instructions(tool)

    tool_instructions = tool_instructions.join()

    return _tools.ToolOutput(
        text=(
            f'Found {len(found_tools)} tool(s) with the following specs:\n'
            f'{tool_instructions}'
        ),
        update_tools=lambda tools: tools + found_tools,
    )


def _match_keywords(query: str, tags: tuple[str, ...]) -> bool:
  """Check if the query matches the tags."""
  # Simple search heuristic:
  words = query.split()
  for word in words:
    for tag in tags:
      if tag in word:
        return True
  return False
