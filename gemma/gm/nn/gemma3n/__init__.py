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

"""Gemma models."""

# pylint: disable=g-importing-member,g-import-not-at-top

from etils import epy as _epy


with _epy.lazy_api_imports(globals()):
  # Gemma 3n
  from gemma.gm.nn.gemma3n._gemma3n import Gemma3n_E2B
  from gemma.gm.nn.gemma3n._gemma3n import Gemma3n_E4B
