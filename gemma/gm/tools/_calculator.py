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
    # TODO(epot): Uses lark parser instead.
    return eval(expression, _OPS)  # pylint: disable=eval-used
