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

"""Tests for _manager._normalize_response."""

import pytest
from gemma.gm.tools import _manager
from unittest import mock


def _make_text_content(text: str):
  content = mock.MagicMock()
  content.type = 'text'
  content.text = text
  return content


def _make_image_content():
  content = mock.MagicMock()
  content.type = 'image'
  return content


def _make_tool_result(
    *,
    content,
    is_error: bool = False,
    structured_content=None,
):
  result = mock.MagicMock()
  result.content = list(content)
  result.isError = is_error
  result.structuredContent = structured_content
  return result


def test_normalize_response_passthrough_when_structured_content_present():
  """If structuredContent is already set, return the response unchanged."""
  result = _make_tool_result(
      content=[_make_text_content('hello')],
      structured_content={'result': 'already set'},
  )
  out = _manager._normalize_response(result)
  assert out is result


def test_normalize_response_success_single_text_block():
  """Single text block with no error should map to {'result': <text>}."""
  result = _make_tool_result(content=[_make_text_content('hello world')])
  out = _manager._normalize_response(result)
  assert out.structuredContent == {'result': 'hello world'}


def test_normalize_response_error_single_text_block():
  """Single text block with isError=True should map to {'error': <text>}."""
  result = _make_tool_result(
      content=[_make_text_content('something went wrong')], is_error=True
  )
  out = _manager._normalize_response(result)
  assert out.structuredContent == {'error': 'something went wrong'}


def test_normalize_response_raises_for_multiple_content_blocks():
  """ValueError (not AssertionError) is raised when content has >1 blocks.

  Bare `assert` statements are silently removed under `python -O`, making
  this guard disappear in optimized production deployments.  The fix
  replaces them with explicit ValueError so the check is always active and
  the error message identifies what was returned.
  """
  result = _make_tool_result(
      content=[_make_text_content('block1'), _make_text_content('block2')]
  )
  with pytest.raises(ValueError, match='Expected a single content block'):
    _manager._normalize_response(result)


def test_normalize_response_raises_for_non_text_content_type():
  """ValueError is raised when the single content block is not type 'text'.

  MCP tools may return image or blob content; this path is unsupported and
  must raise a clear, actionable error rather than an opaque AssertionError.
  """
  result = _make_tool_result(content=[_make_image_content()])
  with pytest.raises(ValueError, match="Expected a text content block"):
    _manager._normalize_response(result)
