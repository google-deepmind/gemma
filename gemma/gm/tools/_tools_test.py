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

"""Tests for the tools module."""

import math

from gemma.gm.tools import _calculator
from gemma.gm.tools import _manager
from gemma.gm.tools import _tools
import pytest


# ============================================================
# Tool base class tests
# ============================================================


class _DummyTool(_tools.Tool):
  """Minimal tool for testing base class behavior."""

  DESCRIPTION = 'A dummy tool for testing.'
  EXAMPLE = _tools.Example(
      query='test query',
      thought='test thought',
      tool_kwargs={'arg1': 'value1'},
      tool_kwargs_doc={'arg1': '<ARG1>'},
      result='test result',
      answer='test answer',
  )

  def call(self, arg1: str, arg2: str = 'default') -> str:
    return f'{arg1}-{arg2}'


def test_tool_name():
  tool = _DummyTool()
  assert tool.name == '_dummytool'


def test_tool_argnames():
  tool = _DummyTool()
  assert tool.tool_argnames == ('arg1', 'arg2')


def test_tool_call():
  tool = _DummyTool()
  result = tool.call(arg1='hello', arg2='world')
  assert result == 'hello-world'


# ============================================================
# Calculator tests
# ============================================================


def test_calculator_basic_arithmetic():
  calc = _calculator.Calculator()
  assert calc.call(expression='2 + 3') == 5
  assert calc.call(expression='10 - 4') == 6
  assert calc.call(expression='3 * 7') == 21
  assert calc.call(expression='15 / 3') == 5.0


def test_calculator_compound_expression():
  calc = _calculator.Calculator()
  assert calc.call(expression='25 * 4 + 10') == 110


def test_calculator_math_functions():
  calc = _calculator.Calculator()
  assert calc.call(expression='sqrt(16)') == 4.0
  assert calc.call(expression='floor(3.7)') == 3
  assert calc.call(expression='ceil(3.2)') == 4


def test_calculator_trig_functions():
  calc = _calculator.Calculator()
  result = calc.call(expression='sin(0)')
  assert result == pytest.approx(0.0, abs=1e-10)
  result = calc.call(expression='cos(0)')
  assert result == pytest.approx(1.0, abs=1e-10)


def test_calculator_name():
  calc = _calculator.Calculator()
  assert calc.name == 'calculator'


def test_calculator_argnames():
  calc = _calculator.Calculator()
  assert calc.tool_argnames == ('expression',)


# ============================================================
# Tool manager: _parse_tool_call tests
# ============================================================


def test_parse_tool_call_valid_json():
  model_output = '{"tool_name": "calculator", "expression": "2 + 3"}'
  result = _manager._parse_tool_call(model_output)
  assert result == {'tool_name': 'calculator', 'expression': '2 + 3'}


def test_parse_tool_call_json_in_text():
  model_output = (
      'I need to calculate this.'
      ' {"tool_name": "calculator", "expression": "5 * 5"}'
      ' Let me check.'
  )
  result = _manager._parse_tool_call(model_output)
  assert result == {'tool_name': 'calculator', 'expression': '5 * 5'}


def test_parse_tool_call_no_json():
  model_output = 'This is just regular text with no tool call.'
  result = _manager._parse_tool_call(model_output)
  assert result is None


def test_parse_tool_call_invalid_json():
  model_output = '{invalid json content}'
  result = _manager._parse_tool_call(model_output)
  assert result is None


# ============================================================
# Tool manager: format helpers
# ============================================================


def test_format_tool_result():
  result = _manager._format_tool_result('42')
  assert result == '[Tool result: 42]'


# ============================================================
# OneShotToolManager tests
# ============================================================


def test_tool_manager_execute_calculator():
  calc = _calculator.Calculator()
  manager = _manager.OneShotToolManager(tools=[calc])
  model_output = '{"tool_name": "calculator", "expression": "10 + 20"}'
  result = manager.maybe_execute_tool(model_output)
  assert result is not None
  assert '30' in result.text


def test_tool_manager_no_tool_call():
  calc = _calculator.Calculator()
  manager = _manager.OneShotToolManager(tools=[calc])
  result = manager.maybe_execute_tool('Just a normal response.')
  assert result is None


def test_tool_manager_unknown_tool():
  calc = _calculator.Calculator()
  manager = _manager.OneShotToolManager(tools=[calc])
  model_output = '{"tool_name": "unknown_tool", "arg": "value"}'
  result = manager.maybe_execute_tool(model_output)
  assert result is not None
  assert 'Unknown' in result.text


def test_tool_manager_system_prompt_contains_tool():
  calc = _calculator.Calculator()
  manager = _manager.OneShotToolManager(tools=[calc])
  assert 'calculator' in manager.system_prompt
  assert 'mathematical' in manager.system_prompt.lower() or (
      'calculation' in manager.system_prompt.lower()
  )


def test_tool_manager_name_to_tool():
  calc = _calculator.Calculator()
  manager = _manager.OneShotToolManager(tools=[calc])
  assert 'calculator' in manager.name_to_tool
  assert manager.name_to_tool['calculator'] is calc
