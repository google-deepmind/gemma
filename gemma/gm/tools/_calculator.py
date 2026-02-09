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

"""Calculator tool."""

from __future__ import annotations

import ast
import math

from gemma.gm.tools import _tools

_OPS = {
    'sqrt': math.sqrt,
    'log': math.log,
    'exp': math.exp,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'asin': math.asin,
    'acos': math.acos,
    'atan': math.atan,
    'atan2': math.atan2,
    'ceil': math.ceil,
    'floor': math.floor,
    'pi': math.pi,
    'e': math.e,
}


class _SafeEvaluator(ast.NodeVisitor):
  """Safely evaluates mathematical expressions using AST."""

  def __init__(self, ops: dict[str, object]):
    self._ops = ops

  def evaluate(self, expression: str) -> float:
    """Evaluates the mathematical expression."""
    node = ast.parse(expression.strip(), mode='eval')
    return float(self.visit(node.body))

  def visit_BinOp(self, node: ast.BinOp) -> float:
    left = self.visit(node.left)
    right = self.visit(node.right)
    if isinstance(node.op, ast.Add):
      return left + right
    if isinstance(node.op, ast.Sub):
      return left - right
    if isinstance(node.op, ast.Mult):
      return left * right
    if isinstance(node.op, ast.Div):
      return left / right
    if isinstance(node.op, ast.Pow):
      return left**right
    if isinstance(node.op, ast.Mod):
      return left % right
    if isinstance(node.op, ast.FloorDiv):
      return left // right
    raise ValueError(f'Unsupported binary operator: {type(node.op).__name__}')

  def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
    operand = self.visit(node.operand)
    if isinstance(node.op, ast.USub):
      return -operand
    if isinstance(node.op, ast.UAdd):
      return +operand
    raise ValueError(f'Unsupported unary operator: {type(node.op).__name__}')

  def visit_Call(self, node: ast.Call) -> float:
    if not isinstance(node.func, ast.Name):
      raise ValueError(
          f'Unsupported function call type: {type(node.func).__name__}')
    if node.func.id not in self._ops:
      raise ValueError(f'Unsupported function: {node.func.id}')
    args = [self.visit(arg) for arg in node.args]
    return self._ops[node.func.id](*args)

  def visit_Constant(self, node: ast.Constant) -> float:
    if not isinstance(node.value, (int, float)):
      raise ValueError(f'Unsupported constant type: {type(node.value).__name__}')
    return float(node.value)

  def visit_Name(self, node: ast.Name) -> float:
    if node.id not in self._ops:
      raise ValueError(f'Unsupported name access: {node.id}')
    value = self._ops[node.id]
    if callable(value):
      raise ValueError(f'Name access to function: {node.id}')
    return float(value)

  def generic_visit(self, node: ast.AST):
    raise ValueError(f'Unsupported expression node: {type(node).__name__}')


class Calculator(_tools.Tool):
  """Simple calculator to demonstrate tool use."""

  DESCRIPTION = 'Perform mathematical calculations.'
  EXAMPLE = _tools.Example(
      query='What is 25 times 4 plus 10?',
      thought='I need to calculate 25 * 4 + 10.',
      tool_kwargs={'expression': '25 * 4 + 10'},
      tool_kwargs_doc={'expression': '<MATH_EXPRESSION>'},
      result='110',
      answer='The result is 110.',
  )
  KEYWORDS = ('math', 'calculator', 'operation')

  def call(self, expression: str) -> str:  # pytype: disable=signature-mismatch
    """Calculates the expression."""
    evaluator = _SafeEvaluator(_OPS)
    try:
      result = evaluator.evaluate(expression)
    except Exception as e:
      return f'Error evaluating expression "{expression}": {e}'

    # Format result to string, handling precision and scientific notation.
    if isinstance(result, (int, float)):
      # Use scientific notation for very large or very small numbers.
      if abs(result) >= 1e10 or (0 < abs(result) < 1e-10):
        return f'{float(result):.10e}'

      if isinstance(result, float):
        if result.is_integer():
          return str(int(result))
        # Standardize on 10 decimal places.
        return f'{result:.10f}'.rstrip('0').rstrip('.')

    return str(result)
