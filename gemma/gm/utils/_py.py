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

"""Python utils."""

import dataclasses


class FrozenDataclass:
  """Mixin to make a dataclass immutable and hashable.

  Python raises a `cannot inherit frozen dataclass from a non-frozen one` when
  trying to make a dataclass frozen. This mixin is a workaround to make a
  dataclass frozen and hashable.
  """

  # TODO(epot): Should also add `__setattr__` and `__del__` guard check, and
  # decorate the `__init__` only block mutation once initialization is done.

  def __eq__(self, other):
    if other.__class__ is self.__class__:
      return _get_comparable_fields(self) == _get_comparable_fields(other)
    return NotImplemented

  def __hash__(self):
    return hash(_get_comparable_fields(self))


def _get_comparable_fields(obj):
  """Get the fields that are comparable."""
  return tuple(
      getattr(obj, f.name) for f in dataclasses.fields(obj) if f.compare  # pytype: disable=wrong-arg-types
  )
