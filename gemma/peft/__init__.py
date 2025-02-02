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

"""PEFT utils for `flax.linen`."""

# pylint: disable=g-importing-member

from gemma.peft._interceptors import Interceptor
from gemma.peft._interceptors import ModuleInterceptor
from gemma.peft._lora import LoRADense
from gemma.peft._lora import LoRADenseAdapter
from gemma.peft._lora import LoRAEinsum
from gemma.peft._lora import LoRAEinsumAdapter
from gemma.peft._tree_utils import fuse_params
from gemma.peft._tree_utils import merge_params
from gemma.peft._tree_utils import split_params
from gemma.peft._tree_utils import unfuse_params
