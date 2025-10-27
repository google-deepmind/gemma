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

"""Calculator tool."""

from __future__ import annotations

import math

from gemma.gm.tools import _tools


class Calculator(_tools.Tool):
  """Secure calculator to demonstrate tool use.

  This calculator safely evaluates mathematical expressions using a restricted
  eval() environment, preventing arbitrary code execution vulnerabilities.
  """

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

  def __init__(self):
    """Initialize the secure calculator."""
    super().__init__()
    # Create a safe environment with only math functions
    self._safe_globals = {
        '__builtins__': {},  # Block all builtins
        # Basic math functions
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'exp': math.exp,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'ceil': math.ceil,
        'floor': math.floor,
        'fabs': math.fabs,
        'factorial': math.factorial,
        'gcd': math.gcd,
        'degrees': math.degrees,
        'radians': math.radians,
        'abs': abs,
        'min': min,
        'max': max,
        'round': round,
        'sum': sum,
        # Math constants
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'inf': math.inf,
        'nan': math.nan,
    }

  def call(self, expression: str) -> str:  # pytype: disable=signature-mismatch
    """Safely calculates the mathematical expression.

    This method uses eval() with a restricted environment containing only
    safe mathematical functions, preventing arbitrary code execution.

    Args:
        expression: The mathematical expression to evaluate.

    Returns:
        The result as a string, or an error message if evaluation fails.
    """
    try:
      # Use eval() with restricted globals - only math functions allowed
      result = eval(expression, self._safe_globals, {})  # pylint: disable=eval-used

      # Format the result appropriately
      if isinstance(result, complex):
        return str(result)
      elif isinstance(result, float):
        # Handle special float values
        if math.isnan(result):
          return 'nan'
        elif math.isinf(result):
          return 'inf' if result > 0 else '-inf'
        else:
          # Format with reasonable precision
          if result == int(result):
            return str(int(result))
          else:
            return f"{result:.10g}"
      else:
        return str(result)

    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
      return f"Error: {e}"
    except Exception as e:
      return f"Unexpected error: {e}"

