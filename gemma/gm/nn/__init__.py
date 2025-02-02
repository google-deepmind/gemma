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

"""Gemma models."""

# pylint: disable=g-importing-member,g-import-not-at-top

from etils import epy as _epy


with _epy.lazy_api_imports(globals()):
  # Gemma models
  from gemma.gm.nn._transformer import Gemma2_2B
  from gemma.gm.nn._transformer import Gemma2_9B
  from gemma.gm.nn._transformer import Gemma2_27B

  from gemma.gm.nn._lora import LoRAWrapper
  from gemma.gm.nn._policy import AnchoredPolicy
  from gemma.gm.nn._transformer import Transformer

  # Model outputs
  from gemma.gm.nn._transformer import Output
  from gemma.gm.nn._policy import AnchoredPolicyOutput
