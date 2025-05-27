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

"""Conversation templates."""

import dataclasses
import textwrap

# from kauldron.typing import UInt8  # pylint: disable=g-multiple-import,g-importing-member

# Note: The template end by `\n` !
PROMPT = """\
<start_of_turn>user
{}<end_of_turn>
<start_of_turn>model
"""
ANSWER = '{}<end_of_turn>'


@dataclasses.dataclass(frozen=True)
class Turn:
  """Base class for a turn."""

  text: str

  def __repr__(self):
    # Prettier display for multi-line strings.
    if '\n' in self.text:
      text = textwrap.indent(self.text, prefix='    ')
      text = f'"""\n{text}\n"""'
    else:
      text = repr(self.text)

    return f'<{type(self).__name__}({text})>'


@dataclasses.dataclass(frozen=True, repr=False)
class ModelTurn(Turn):
  """Model turn."""


@dataclasses.dataclass(frozen=True, repr=False)
class UserTurn(Turn):
  """User turn."""

  # images: UInt8["N? H W C"] | None = None
