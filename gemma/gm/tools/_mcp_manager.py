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

"""MCP tool handler."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
from typing import Any, Iterator, Optional

from etils import epy
from gemma.gm.tools import _manager

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
  import fastmcp
  import mcp
  # pylint: enable=g-import-not-at-top # pytype: enable=import-error

type FastMcpClientLike = (fastmcp.Client | fastmcp.FastMCP | str | Any)


@dataclasses.dataclass(frozen=True)
class McpToolHandler(_manager.ToolHandlerBase, epy.ContextManager):
  """Mcp tool handler."""

  mcp_client: FastMcpClientLike

  def _tools(self) -> list[mcp.types.Tool]:
    """Returns the list of tools."""
    return self._run(self.client.list_tools())

  def _call_tool(
      self,
      name: str,
      arguments: Optional[dict[str, Any]] = None,
  ) -> mcp.types.CallToolResult:
    return self._run(self.client.call_tool_mcp(name, arguments=arguments))

  @functools.cached_property
  def client(self) -> fastmcp.Client:
    """Returns the MCP client."""
    mcp_client = self.mcp_client
    if not isinstance(mcp_client, fastmcp.Client):
      mcp_client = fastmcp.Client(mcp_client)
    return mcp_client

  @contextlib.contextmanager
  def _activate(self) -> Iterator[None]:
    """Activate the tool handler."""
    try:
      self._run(self.client.__aenter__())
      yield
    finally:
      try:
        self._run(self.client.__aexit__(None, None, None))
      finally:
        self._loop.close()
        object.__delattr__(self, '_loop')  # Reset the cached_property

  @functools.cached_property
  def _loop(self) -> asyncio.AbstractEventLoop:
    """Loop."""
    return asyncio.new_event_loop()

  def _run(self, future):
    return self._loop.run_until_complete(future)
