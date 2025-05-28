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

"""Utils for LoRA checkpoint managment."""

from __future__ import annotations

from typing import Any

from gemma.gm.typing import _common
import jax


def convert_to_qat_checkpoint(
    params: _common.Params, *, checkpoint_kernel_key='w'
) -> _common.Params:
  """map regular checkpoint to QAT checkpoint."""

  def is_leaf(data: Any) -> bool:
    if isinstance(data, (jax.Array, jax.ShapeDtypeStruct)):
      return True
    if checkpoint_kernel_key in data:
      return True
    if 'gating_einsum' in data or 'linear' in data:
      return True
    return False

  def quantize_leaf(data: Any, key: str) -> Any:
    new_kernel = data[key]
    new_data = dict(data)
    del new_data[key]
    name = f'_SimulateQuantizedEinsum_{key}'
    if key == checkpoint_kernel_key:
      name = '_SimulateQuantizedEinsum_0'
    new_data[name] = {'kernel': new_kernel}
    return new_data

  def convert_leaf(data: Any) -> Any:
    if isinstance(data, (jax.Array, jax.ShapeDtypeStruct)):
      return data
    if checkpoint_kernel_key in data:
      data = quantize_leaf(data, checkpoint_kernel_key)
    # This hack is required because the FeedForward layer call two different
    # Einsum with using `nn.share_scope`, so the two wrappers need a different
    # name. Weights are not stored under any kernel key and are isntead under
    # the following names.
    if 'gating_einsum' in data or 'linear' in data:
      data = quantize_leaf(data, 'gating_einsum')
      data = quantize_leaf(data, 'linear')
    return data

  params = jax.tree.map(convert_leaf, params, is_leaf=is_leaf)
  return params
