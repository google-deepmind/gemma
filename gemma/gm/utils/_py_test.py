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

"""Tests for Python utilities."""

import dataclasses

from gemma.gm.utils import _py


class TestFrozenDataclass:
  """Tests for FrozenDataclass mixin."""

  def test_equal_instances(self):
    """Two instances with same fields should be equal."""

    @dataclasses.dataclass(eq=False)
    class Cfg(_py.FrozenDataclass):
      x: int = 1
      y: str = 'hello'

    a = Cfg(x=1, y='hello')
    b = Cfg(x=1, y='hello')
    assert a == b

  def test_not_equal_instances(self):
    """Two instances with different fields should not be equal."""

    @dataclasses.dataclass(eq=False)
    class Cfg(_py.FrozenDataclass):
      x: int = 1

    a = Cfg(x=1)
    b = Cfg(x=2)
    assert a != b

  def test_hash_equal_for_equal_instances(self):
    """Equal instances must have the same hash."""

    @dataclasses.dataclass(eq=False)
    class Cfg(_py.FrozenDataclass):
      x: int = 1
      y: str = 'hello'

    a = Cfg(x=1, y='hello')
    b = Cfg(x=1, y='hello')
    assert hash(a) == hash(b)

  def test_hash_differs_for_different_instances(self):
    """Different instances should (usually) have different hashes."""

    @dataclasses.dataclass(eq=False)
    class Cfg(_py.FrozenDataclass):
      x: int = 1

    a = Cfg(x=1)
    b = Cfg(x=2)
    # Not guaranteed, but very likely for distinct ints.
    assert hash(a) != hash(b)

  def test_usable_in_set(self):
    """FrozenDataclass instances should be usable as set elements."""

    @dataclasses.dataclass(eq=False)
    class Cfg(_py.FrozenDataclass):
      x: int = 1

    a = Cfg(x=1)
    b = Cfg(x=1)
    c = Cfg(x=2)
    s = {a, b, c}
    assert len(s) == 2

  def test_usable_as_dict_key(self):
    """FrozenDataclass instances should be usable as dict keys."""

    @dataclasses.dataclass(eq=False)
    class Cfg(_py.FrozenDataclass):
      x: int = 1

    key = Cfg(x=42)
    d = {key: 'value'}
    assert d[Cfg(x=42)] == 'value'

  def test_compare_false_is_excluded(self):
    """Fields with compare=False should not affect equality or hash."""

    @dataclasses.dataclass(eq=False)
    class Cfg(_py.FrozenDataclass):
      x: int = 1
      meta: str = dataclasses.field(default='a', compare=False)

    a = Cfg(x=1, meta='a')
    b = Cfg(x=1, meta='b')
    assert a == b
    assert hash(a) == hash(b)

  def test_not_equal_to_different_class(self):
    """Instances of different classes should not be equal."""

    @dataclasses.dataclass(eq=False)
    class CfgA(_py.FrozenDataclass):
      x: int = 1

    @dataclasses.dataclass(eq=False)
    class CfgB(_py.FrozenDataclass):
      x: int = 1

    assert CfgA(x=1) != CfgB(x=1)
