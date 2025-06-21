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

"""T5Gemma API."""

from etils import epy as _epy

# pylint: disable=g-import-not-at-top,g-importing-member

with _epy.lazy_api_imports(globals()):
  # Configs
  from gemma.research.t5gemma.config import CKPTType
  from gemma.research.t5gemma.config import PretrainType
  from gemma.research.t5gemma.config import T5GemmaPreset

  # Sampling
  from gemma.research.t5gemma.sampling import Sampler
  from gemma.research.t5gemma.sampling import Greedy
  from gemma.research.t5gemma.sampling import RandomSampling
  from gemma.research.t5gemma.sampling import TopkSampling

  # Model
  from gemma.research.t5gemma.t5gemma import T5Gemma

