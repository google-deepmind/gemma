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

from gemma.gm.data import _tasks

def test_decode_bytes():
  # Valid UTF-8
  assert _tasks._decode_bytes(b"hello") == "hello"
  
  # Non-bytes input
  assert _tasks._decode_bytes("already_str") == "already_str"
  assert _tasks._decode_bytes(123) == 123
  
  # Invalid UTF-8 (should not crash)
  # b"\xff" is invalid UTF-8. 
  # errors="replace" should return the replacement character.
  assert _tasks._decode_bytes(b"\xff") == "\ufffd"
