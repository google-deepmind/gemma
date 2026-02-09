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

"""Tests for Calculator tool."""

import math
import sys
from unittest.mock import MagicMock
from contextlib import contextmanager

@contextmanager
def _mock_lazy_imports(*args, **kwargs):
  yield

# Mock dependencies to allow running tests in isolated environments.
# These can be removed in a fully configured environment.
mock_kauldron = MagicMock()
sys.modules["kauldron"] = mock_kauldron
sys.modules["kauldron.typing"] = mock_kauldron
mock_etils = MagicMock()
mock_etils.epy.lazy_api_imports = _mock_lazy_imports
sys.modules["etils"] = mock_etils
sys.modules["etils.epy"] = mock_etils.epy
sys.modules["etils.etree"] = MagicMock()
sys.modules["etils.etree.jax"] = MagicMock()
sys.modules["jax"] = MagicMock()
sys.modules["jax.numpy"] = MagicMock()
sys.modules["flax"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["einops"] = MagicMock()
sys.modules["sentencepiece"] = MagicMock()
sys.modules["treescope"] = MagicMock()
sys.modules["orbax"] = MagicMock()
sys.modules["orbax.checkpoint"] = MagicMock()
sys.modules["absl"] = MagicMock()
sys.modules["absl.logging"] = MagicMock()

import pytest
from gemma.gm.tools import _calculator

@pytest.fixture
def calculator():
  return _calculator.Calculator()

def test_basic_arithmetic(calculator):
  assert calculator.call("1 + 2") == "3"
  assert calculator.call("10 - 5") == "5"
  assert calculator.call("4 * 3") == "12"
  assert calculator.call("10 / 2") == "5"
  assert calculator.call("2 ** 3") == "8"
  assert calculator.call("10 % 3") == "1"
  assert calculator.call("10 // 3") == "3"

def test_complex_expressions(calculator):
  assert calculator.call("2 * (3 + 4) - 5") == "9"
  assert calculator.call("10 / (2 + 3) * 4") == "8"

def test_math_functions(calculator):
  assert calculator.call("sqrt(16)") == "4"
  assert calculator.call("sin(0)") == "0"
  assert calculator.call("cos(0)") == "1"
  assert calculator.call("log(1)") == "0"
  assert calculator.call("exp(0)") == "1"
  assert calculator.call("ceil(4.2)") == "5"
  assert calculator.call("floor(4.8)") == "4"

def test_math_constants(calculator):
  assert calculator.call("pi") == f"{math.pi:.10f}".rstrip('0').rstrip('.')
  assert calculator.call("e") == f"{math.e:.10f}".rstrip('0').rstrip('.')
  assert calculator.call("pi * 2") == f"{math.pi * 2:.10f}".rstrip('0').rstrip('.')

def test_float_formatting(calculator):
  assert calculator.call("1 / 3") == "0.3333333333"
  assert calculator.call("0.1 + 0.2") == "0.3"

def test_security_boundaries(calculator):
  # Forbidden operations
  assert "Error" in calculator.call("__import__('os').system('ls')")
  assert "Error" in calculator.call("eval('1+1')")
  assert "Error" in calculator.call("exec('print(1)')")
  # Forbidden node types (Attribute access)
  assert "Unsupported function call type: Attribute" in calculator.call("math.sqrt(16)")
  # Forbidden function calls
  assert "Unsupported function: abs" in calculator.call("abs(-5)")
  # Name access to functions
  assert "Name access to function: sqrt" in calculator.call("sqrt")

def test_mathematical_edge_cases(calculator):
  # Deep nesting
  nesting = "1+" * 100 + "1"
  assert calculator.call(nesting) == "101"
  # Large numbers (standardized to scientific notation)
  assert calculator.call("10**100") == "1.0000000000e+100"
  # Division by zero
  assert "Error" in calculator.call("1 / 0")
  # Complex results
  assert "Error" in calculator.call("sqrt(-1)")
  # Stacked unary operators
  assert calculator.call("--5") == "5"
  assert calculator.call("-+-5") == "5"
  # Whitespace (verified fix for 'unexpected indent' bug)
  assert calculator.call("  1  +   2   ") == "3"

def test_invalid_syntax(calculator):
  assert "Error" in calculator.call("1 +")
  assert "Error" in calculator.call("(")
