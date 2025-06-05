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

"""Tools."""

from __future__ import annotations

import abc
import dataclasses
import functools
import inspect
from typing import ClassVar


class Tool(abc.ABC):
  """Tool."""

  DESCRIPTION: ClassVar[str]
  EXAMPLE: ClassVar[Example]

  @abc.abstractmethod
  def call(self, **kwargs) -> str:
    """Calls the tool."""
    raise NotImplementedError()

  @functools.cached_property
  def name(self) -> str:
    """Returns the name of the tool."""
    return self.__class__.__name__.lower()

  @functools.cached_property
  def tool_argnames(self) -> tuple[str, ...]:
    """Returns the name of the tool."""
    sig = inspect.signature(self.call)
    return tuple(sig.parameters)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Example:
  """Tool call."""

  query: str
  thought: str | None = None
  tool_kwargs: dict[str, str]
  tool_kwargs_doc: dict[str, str]
  result: str
  answer: str
