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
}


class _SafeEvaluator(ast.NodeVisitor):
  """Safely evaluates mathematical expressions using AST."""

  def __init__(self, ops: dict[str, object]):
    self._ops = ops
    self._allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Name,
        ast.Call,
    )

  def visit(self, node: ast.AST) -> object:
    """Visits a node and validates it's safe."""
    if not isinstance(node, self._allowed_nodes):
      raise ValueError(f'Unsafe operation: {type(node).__name__}')
    return super().visit(node)

  def visit_Expression(self, node: ast.Expression) -> object:
    """Evaluates the expression."""
    return self.visit(node.body)

  def visit_Constant(self, node: ast.Constant) -> object:
    """Returns constant values."""
    if isinstance(node.value, (int, float, complex)):
      return node.value
    raise ValueError(f'Unsupported constant type: {type(node.value).__name__}')

  def visit_Name(self, node: ast.Name) -> object:
    """Resolves names to allowed operations."""
    if node.id in self._ops:
      return self._ops[node.id]
    raise ValueError(f'Unknown operation: {node.id}')

  def visit_BinOp(self, node: ast.BinOp) -> object:
    """Evaluates binary operations."""
    left = self.visit(node.left)
    right = self.visit(node.right)
    op = node.op

    if isinstance(op, ast.Add):
      return left + right
    elif isinstance(op, ast.Sub):
      return left - right
    elif isinstance(op, ast.Mult):
      return left * right
    elif isinstance(op, ast.Div):
      return left / right
    elif isinstance(op, ast.FloorDiv):
      return left // right
    elif isinstance(op, ast.Mod):
      return left % right
    elif isinstance(op, ast.Pow):
      return left ** right
    else:
      raise ValueError(f'Unsupported binary operation: {type(op).__name__}')

  def visit_UnaryOp(self, node: ast.UnaryOp) -> object:
    """Evaluates unary operations."""
    operand = self.visit(node.operand)
    op = node.op

    if isinstance(op, ast.UAdd):
      return +operand
    elif isinstance(op, ast.USub):
      return -operand
    else:
      raise ValueError(f'Unsupported unary operation: {type(op).__name__}')

  def visit_Call(self, node: ast.Call) -> object:
    """Evaluates function calls."""
    if not isinstance(node.func, ast.Name):
      raise ValueError('Only simple function calls are supported')
    func = self.visit(node.func)
    args = [self.visit(arg) for arg in node.args]
    return func(*args)


def _safe_eval(expression: str, ops: dict[str, object]) -> object:
  """Safely evaluates a mathematical expression."""
  try:
    tree = ast.parse(expression, mode='eval')
    evaluator = _SafeEvaluator(ops)
    result = evaluator.visit(tree)
    return result
  except SyntaxError as e:
    raise ValueError(f'Invalid expression syntax: {e}') from e
  except Exception as e:
    raise ValueError(f'Error evaluating expression: {e}') from e


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
    result = _safe_eval(expression, _OPS)
    # Convert result to string, handling floats appropriately
    if isinstance(result, float):
      # Format floats to avoid scientific notation for reasonable values
      if (abs(result) < 1e10 and abs(result) > 1e-10) or result == 0.0:
        return str(result)
      else:
        return f'{result:.10e}'
    return str(result)
