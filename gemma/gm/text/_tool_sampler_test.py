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

"""Tests for ToolSampler max_tool_depth limit."""

from unittest import mock

from gemma.gm.text import _tool_sampler
from gemma.gm.tools import _manager as _manager_lib
import pytest


class AlwaysCallToolHandler(_manager_lib.ToolHandlerBase):
  """A fake tool handler that always returns a tool answer.

  This simulates the bug scenario where the model keeps generating
  tool calls indefinitely (e.g., due to hallucination).
  """

  def _tools(self):
    return []

  def _call_tool(self, name, arguments):
    return mock.MagicMock()

  def add_system_prompt(self, prompt):
    return prompt

  def maybe_execute_tool(self, model_output):
    """Always returns a fake tool answer to force recursion."""
    return [mock.MagicMock()]


class NeverCallToolHandler(_manager_lib.ToolHandlerBase):
  """A fake tool handler that never triggers tool calls."""

  def _tools(self):
    return []

  def _call_tool(self, name, arguments):
    return mock.MagicMock()

  def add_system_prompt(self, prompt):
    return prompt

  def maybe_execute_tool(self, model_output):
    """Never returns a tool answer — model responds normally."""
    return None


def _make_tool_sampler(tool_handler, max_tool_depth=10):
  """Helper to create a ToolSampler with mocked model components."""
  sampler = mock.MagicMock(spec=_tool_sampler.ToolSampler)
  sampler.max_tool_depth = max_tool_depth
  sampler.tool_handler = tool_handler
  sampler.multi_turn = False
  sampler.conversation = None

  # Make chat() call the real implementation
  sampler.chat = lambda *args, **kwargs: _tool_sampler.ToolSampler.chat(
      sampler, *args, **kwargs
  )
  return sampler


def test_max_tool_depth_default():
  """Verify the default max_tool_depth is 10."""
  # Check the dataclass field default directly
  fields = {f.name: f for f in _tool_sampler.ToolSampler.__dataclass_fields__.values()}
  assert fields['max_tool_depth'].default == 10


def test_max_tool_depth_raises():
  """Verify RecursionError is raised when depth exceeds max_tool_depth."""
  handler = AlwaysCallToolHandler()

  # We can't easily instantiate ToolSampler (needs real model), so we test
  # the depth check logic directly by calling chat with _current_depth
  # already at the limit.
  sampler = mock.MagicMock(spec=_tool_sampler.ToolSampler)
  sampler.max_tool_depth = 3

  with pytest.raises(RecursionError, match='Tool call depth exceeded maximum of 3'):
    # Call with _current_depth already exceeding the limit
    _tool_sampler.ToolSampler.chat(
        sampler,
        'test prompt',
        _current_depth=4,
    )


def test_max_tool_depth_exact_boundary():
  """Verify that depth exactly equal to max_tool_depth is still allowed."""
  sampler = mock.MagicMock(spec=_tool_sampler.ToolSampler)
  sampler.max_tool_depth = 5
  sampler.multi_turn = False

  # At depth == max_tool_depth, should NOT raise (it's the boundary)
  # It will fail later due to mocking, but should get past the depth check
  try:
    _tool_sampler.ToolSampler.chat(
        sampler,
        'test prompt',
        _current_depth=5,
    )
  except RecursionError:
    pytest.fail('Should not raise RecursionError when _current_depth == max_tool_depth')
  except Exception:
    # Other errors from mocking are expected — the depth check passed
    pass


def test_max_tool_depth_exceeded_boundary():
  """Verify that depth one more than max_tool_depth raises."""
  sampler = mock.MagicMock(spec=_tool_sampler.ToolSampler)
  sampler.max_tool_depth = 5

  with pytest.raises(RecursionError, match='Tool call depth exceeded maximum of 5'):
    _tool_sampler.ToolSampler.chat(
        sampler,
        'test prompt',
        _current_depth=6,
    )


def test_zero_depth_blocks_recursion():
  """Verify max_tool_depth=0 allows initial call but blocks tool recursion."""
  sampler = mock.MagicMock(spec=_tool_sampler.ToolSampler)
  sampler.max_tool_depth = 0

  # Depth 0 should be allowed (the initial user call)
  try:
    _tool_sampler.ToolSampler.chat(
        sampler,
        'test prompt',
        _current_depth=0,
    )
  except RecursionError:
    pytest.fail('Should not raise RecursionError at depth 0 even with max_tool_depth=0')
  except Exception:
    pass

  # Depth 1 should be blocked (first tool recursion)
  with pytest.raises(RecursionError, match='Tool call depth exceeded maximum of 0'):
    _tool_sampler.ToolSampler.chat(
        sampler,
        'test prompt',
        _current_depth=1,
    )


def test_error_message_is_actionable():
  """Verify the error message tells the user what to do."""
  sampler = mock.MagicMock(spec=_tool_sampler.ToolSampler)
  sampler.max_tool_depth = 3

  with pytest.raises(RecursionError, match='Increase `max_tool_depth`'):
    _tool_sampler.ToolSampler.chat(
        sampler,
        'test prompt',
        _current_depth=4,
    )
