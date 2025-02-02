# Copyright 2024 DeepMind Technologies Limited.
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

"""Kauldron API for Gemma."""

from etils import epy as _epy

# pylint: disable=g-import-not-at-top

with _epy.lazy_api_imports(globals()):
  # API match the `kd` namespace.
  from gemma.gm import ckpts
  from gemma.gm import data
  from gemma.gm import evals
  from gemma.gm import losses
  from gemma.gm import nn
  from gemma.gm import text
  from gemma.gm import sharding
