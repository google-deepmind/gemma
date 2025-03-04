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

"""params pre-processing."""

import enum
from typing import Any, TypeVar
from etils import epy
from flax.typing import Array  # pylint: disable=g-importing-member
import jax
from jax import numpy as jnp

PyTree = TypeVar('PyTree')
_NON_ZERO_DIVISION_EPS = 1e-6


class QuantizationMethod(epy.StrEnum):
  """Quantization methods.

  Attributes:
    NONE: No quantization.
    INT4: 4 bits per-channel.
    Q4_0: 4 bits per-block.
    Q4_0_TRANSPOSE: 4 bits per-block (transpose first MLP layer).
    SFP8: 8 bits floating points.
  """

  NONE = enum.auto()
  INT4 = enum.auto()
  Q4_0 = enum.auto()
  Q4_0_TRANSPOSE = enum.auto()
  SFP8 = enum.auto()


class QuantizationGranularity(epy.StrEnum):
  """Granularity of the quantization.

  Attributes:
    PER_TENSOR: scale the entire tensor.
    PER_CHANNEL: scale each channel independently.
    PER_BLOCK: scale each block independently.

  NOTE: a block is defined a contiguous sub sequence of a tensor, e.g. in
    128x128 array would comprise 4x32 blocks of size 32.
  """

  PER_TENSOR = enum.auto()
  PER_CHANNEL = enum.auto()
  PER_BLOCK = enum.auto()


def quantize(
    params: PyTree,
    *,
    method: QuantizationMethod | str,
) -> PyTree:
  """Quantizes the given params.

  In ths API, we convert the elements of params in order to actually get
  quantized values. It is currently limited to INT$ per-channel weight
  quantization.

  Args:
    params: The params to quantize.
    method: The quantization method to use.

  Returns:
    The quantized params.
  """
  method = QuantizationMethod(method)
  match method:
    case QuantizationMethod.NONE:
      return params
    case QuantizationMethod.INT4:
      quantization_func = uniform_quantize
    case _:
      raise ValueError(f'Quantization method {method} is not yet supported.')

  def is_leaf(data: Any) -> bool:
    if isinstance(data, jax.Array):
      return True
    if 'kernel' in data:
      return True
    return False

  def convert_leaf(data: Any) -> Any:
    if 'kernel' in data:
      new_kernel, scales = quantization_func(
          data['kernel'],
          bitwidth=4,
          granularity=QuantizationGranularity.PER_CHANNEL,
      )
      data['kernel'] = new_kernel
      data['scale'] = scales
    return data

  return jax.tree.map(convert_leaf, params, is_leaf=is_leaf)


def uniform_quantize(
    x: Array,
    *,
    bitwidth: int,
    granularity: QuantizationGranularity,
) -> tuple[Array, Array]:
  """Applies uniform quantization to the given array.

  We return the actually quantized array and the scale used (for de-quantization
  as a division, i.e. DQ(x) = x / scale)..

  Args:
    x: The array to quantize.
    bitwidth: The bitwidth of the quantization.
    granularity: The granularity of the quantization.

  Returns:
    The quantized array and the scale used.
  """
  assert bitwidth == 4  # only int4 quantization is supported
  upper_bound = 2 ** (bitwidth - 1) - 1
  lower_bound = -(2 ** (bitwidth - 1))
  match granularity:
    case QuantizationGranularity.PER_TENSOR:
      max_ = jnp.max(jnp.abs(x))
      scales = upper_bound * jnp.squeeze(jnp.array([1.0 / max_]))
    case QuantizationGranularity.PER_CHANNEL:
      max_ = reduce_max_all_but_one_axis(jnp.abs(x), -1)
      scales = upper_bound * jnp.array(1.0 / max_)
    case _:
      raise ValueError(f'Unknown granularity: {granularity}')
  # apply scale
  x = x * scales
  # clip and round
  x = jnp.clip(jnp.round(x), lower_bound, upper_bound)
  return x.astype(jnp.int4), jnp.maximum(scales, _NON_ZERO_DIVISION_EPS)


def reduce_max_all_but_one_axis(
    x: Array, axis: int = 0, keepdims: bool = True
) -> Array:
  """Reduce the max of an array over all dimensions except one.

  This is useful to extract the max w.r.t. to the batch dimension for example.

  Args:
    x: The array to reduce
    axis: The axis not to reduce over
    keepdims: Whether to keep the reduced dimension

  Returns:
    The reduced array
  """
  if axis < 0:
    axis += x.ndim
  dims = list(range(x.ndim))
  dims.pop(axis)
  return jnp.max(x, axis=tuple(dims), keepdims=keepdims)
