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

"""params pre-processing."""

from collections.abc import Sequence
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
  INT8 = enum.auto()
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
    checkpoint_kernel_key: str = 'w',
    in_place_keys: bool = False,
) -> PyTree:
  """Quantizes the given params.

  In ths API, we convert the elements of params in order to actually get
  quantized values. It is currently limited to INT$ per-channel weight
  quantization.

  Args:
    params: The params to quantize.
    method: The quantization method to use.
    checkpoint_kernel_key: The key of the kernel in the checkpoint (in
      pre-trained checkpoitns that is 'w').
    in_place_keys: Whether to quantize the keys in place.

  Returns:
    The quantized params.
  """
  method = QuantizationMethod(method)
  match method:
    case QuantizationMethod.NONE:
      return params
    case QuantizationMethod.INT4:
      quantization_func = uniform_quantize
      bitwidth = 4
      dtype = jnp.int4
    case QuantizationMethod.INT8:
      quantization_func = uniform_quantize
      bitwidth = 8
      dtype = jnp.int8
    case _:
      raise ValueError(f'Quantization method {method} is not yet supported.')

  def is_leaf(data: Any) -> bool:
    if isinstance(data, jax.Array):
      return True
    if checkpoint_kernel_key in data:
      return True
    if 'gating_einsum' in data or 'linear' in data:
      return True
    return False

  try:
    # get models dims to identify layer based on weight shape
    head_dim, d_model, _ = params['layer_0']['attn']['q_einsum'][
        checkpoint_kernel_key
    ].shape
  except KeyError:
    head_dim, d_model = -1, -1

  def quantize_leaf(data: Any, key: str) -> Any:
    new_kernel, scales = quantization_func(
        data[key],
        bitwidth=bitwidth,
        granularity=QuantizationGranularity.PER_CHANNEL,
        dtype=dtype,
        axis_to_reduce=_get_axis_to_reduce_from_weight_shape(
            data[key].shape, head_dim=head_dim, d_model=d_model
        ),
    )
    new_data = dict(data)
    del new_data[key]
    if in_place_keys:
      new_data['kernel'] = new_kernel
      new_data['scale'] = scales
      return new_data
    name = f'_IntEinsum_{key}'
    if key == checkpoint_kernel_key:
      name = '_IntEinsum_0'
    new_data[name] = {'kernel': new_kernel, 'scale': scales}
    return new_data

  def convert_leaf(data: Any) -> Any:
    if isinstance(data, jax.Array):
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

  quantized_params = jax.tree.map(convert_leaf, dict(params), is_leaf=is_leaf)
  return quantized_params


def uniform_quantize(
    x: Array,
    *,
    bitwidth: int,
    granularity: QuantizationGranularity,
    dtype: jnp.dtype = jnp.int4,
    axis_to_reduce: int | None = None,
) -> tuple[Array, Array]:
  """Applies uniform quantization to the given array.

  We return the actually quantized array and the scale used (for de-quantization
  as a division, i.e. DQ(x) = x / scale)..

  Args:
    x: The array to quantize.
    bitwidth: The bitwidth of the quantization.
    granularity: The granularity of the quantization.
    dtype: The dtype to use for the quantization.
    axis_to_reduce: The axis to reduce over.

  Returns:
    The quantized array and the scale used.
  """
  upper_bound = 2 ** (bitwidth - 1) - 1
  lower_bound = -(2 ** (bitwidth - 1))
  match granularity:
    case QuantizationGranularity.PER_TENSOR:
      max_ = jnp.max(jnp.abs(x))
      scales = upper_bound * jnp.squeeze(jnp.array([1.0 / max_]))
    case QuantizationGranularity.PER_CHANNEL:
      if axis_to_reduce is None:
        max_ = reduce_max_all_but_one_axis(jnp.abs(x), axis=-1)
      else:
        max_ = jnp.max(jnp.abs(x), keepdims=True, axis=axis_to_reduce)
      scales = upper_bound * jnp.array(1.0 / max_)
    case _:
      raise ValueError(f'Unknown granularity: {granularity}')
  # apply scale
  x = x * scales
  # clip and round
  x = jnp.clip(jnp.round(x), lower_bound, upper_bound)
  return x.astype(dtype), jnp.maximum(scales, _NON_ZERO_DIVISION_EPS)


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


def _replace_intermediate_keys(data, old_key, new_key):
  """Replaces intermediate keys in a nested dictionary.

  We use this to go from the simulated quantized nested params to the int4
  inference nested params.

  Args:
      data: The nested dictionary.
      old_key: The key to replace.
      new_key: The new key.

  Returns:
      The modified dictionary.
  """
  if isinstance(data, dict):
    return {
        k.replace(old_key, new_key): _replace_intermediate_keys(
            v, old_key, new_key
        )
        for k, v in data.items()
    }
  else:
    return data


def _get_axis_to_reduce_from_weight_shape(
    shape: Sequence[int], *, head_dim: int, d_model: int
) -> Sequence[int] | None:
  """Returns the axis to reduce over."""
  if head_dim == -1 or d_model == -1:  # no model dims available
    return None
  if len(shape) == 2:
    return (0,)
  if len(shape) == 3:
    if shape[0] == head_dim and shape[1] == d_model:  # query einsum
      return (1,)
    if shape[0] == head_dim and shape[2] == d_model:  # att out einsum
      return (0, 1)
    if shape[2] == d_model:  # gate einsum
      return (2,)
    raise ValueError(f'Unsupported weight shape: {shape}')
  if len(shape) == 4:
    return (2,)
  raise ValueError(f'Unsupported weight shape: {shape}')
