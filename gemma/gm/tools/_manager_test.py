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

"""Tests for tool manager base class."""

from collections.abc import Iterator
import contextlib
import dataclasses
from typing import Any
from unittest import mock

from gemma.gm.tools import _manager
import mcp.types
import pytest


# Concrete implementation for testing the abstract base class.
@dataclasses.dataclass(frozen=True, kw_only=True)
class _FakeToolHandler(_manager.ToolHandlerBase):
  """A minimal concrete ToolHandlerBase for testing."""

  fake_tools: tuple[mcp.types.Tool, ...] = ()

  def _tools(self):
    return list(self.fake_tools)

  def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
    return mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type='text', text=f'result:{name}')],
    )


class TestToolHandlerBase:
  """Tests for ToolHandlerBase abstract class."""

  def test_is_active_false_by_default(self):
    handler = _FakeToolHandler()
    assert handler.is_active is False

  def test_context_manager_activates(self):
    handler = _FakeToolHandler()
    with handler:
      assert handler.is_active is True
    assert handler.is_active is False

  def test_nested_context_manager_does_not_raise(self):
    """Entering context when already active should not raise."""
    handler = _FakeToolHandler()
    with handler:
      assert handler.is_active is True
      with handler:  # Nested — should not fail.
        pass  # No crash is the key guarantee.

  def test_tools_property(self):
    tool = mcp.types.Tool(
        name='test_tool',
        description='A test tool',
        inputSchema={'type': 'object', 'properties': {}},
    )
    handler = _FakeToolHandler(fake_tools=(tool,))
    tools = handler.tools
    assert len(tools) == 1
    assert tools[0].name == 'test_tool'

  def test_call_tool(self):
    handler = _FakeToolHandler()
    result = handler.call_tool(name='my_func', arguments={'a': 1})
    assert isinstance(result, mcp.types.CallToolResult)
    assert result.content[0].text == 'result:my_func'


class TestNormalizeResponse:
  """Tests for _normalize_response helper."""

  def test_text_response_gets_structured_content(self):
    """A plain text response should be wrapped in structuredContent."""
    response = mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type='text', text='hello')],
    )
    normalized = _manager._normalize_response(response)
    assert normalized.structuredContent == {'result': 'hello'}

  def test_error_response_gets_error_key(self):
    """An error response should use 'error' key in structuredContent."""
    response = mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type='text', text='something failed')],
        isError=True,
    )
    normalized = _manager._normalize_response(response)
    assert normalized.structuredContent == {'error': 'something failed'}

  def test_already_structured_response_unchanged(self):
    """If structuredContent already set, it should be returned as-is."""
    response = mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type='text', text='ignored')],
        structuredContent={'custom': 'data'},
    )
    normalized = _manager._normalize_response(response)
    assert normalized.structuredContent == {'custom': 'data'}
