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

"""Tests for data tasks helpers."""

from gemma.gm.data import _tasks


class TestDecodeBytes:
  """Tests for _decode_bytes helper."""

  def test_bytes_decoded_to_str(self):
    assert _tasks._decode_bytes(b'hello world') == 'hello world'

  def test_str_passthrough(self):
    assert _tasks._decode_bytes('already a string') == 'already a string'

  def test_utf8_bytes(self):
    text = 'café'
    assert _tasks._decode_bytes(text.encode('utf-8')) == text

  def test_empty_bytes(self):
    assert _tasks._decode_bytes(b'') == ''

  def test_empty_string(self):
    assert _tasks._decode_bytes('') == ''

  def test_non_string_passthrough(self):
    """Non-bytes, non-string values should pass through unchanged."""
    assert _tasks._decode_bytes(42) == 42
    assert _tasks._decode_bytes(None) is None
