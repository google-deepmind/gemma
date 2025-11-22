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

"""gemma API."""

# A new PyPI release will be pushed every time `__version__` is increased.
# When changing this, also update the CHANGELOG.md.
__version__ = '3.3.0'


def __getattr__(name: str):  # pylint: disable=invalid-name
  """Catches `import gemma as gm` errors."""
  del name
  raise AttributeError(
      'Please use "from gemma import gm", NOT "import gemma as gm".'
  )
