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

"""Flax linen Quantization modules."""

from collections.abc import Callable, Sequence
import dataclasses
import einops
from flax import linen as nn
from flax.linen import dtypes as flax_dtypes
from flax.typing import Array  # pylint: disable=g-importing-member
from gemma.peft import _quantization_utils
import jax
import jax.numpy as jnp


# (0.5 - 1/32) 2**-22 is the smallest non-zero number in sfp8
_SFP8_ZERO_THRESHOLD = 1.1175882588358173e-07
# 0.625 * 2**-22 is the smallest non-zero number in sfp8
_SFP8_LOWEST_NON_ZERO_BOUND = 1.4901161193847656e-07
_SFP8_UPPER_BOUND = 1.875
_QUANTIZATION_BLOCK_SIZE = 32  # for Q4_0 quantization
_NON_ZERO_DIVISION_EPS = 1e-6


def simulate_quantize(
    x: Array,
    method: _quantization_utils.QuantizationMethod | str,
    axis_to_reduce: int | None = None,
) -> Array:
  """Quantizes the given array.

  In this API, we do not actually quantize tensors as the output is not stored
  using less bits but rather simulate quantization to enable quantization aware
  training.

  NOTE: you can use this implementation to evaluate a checkpoint as if it was
  quantized.

  Args:
    x: The array to simulate_quantize.
    method: The quantization method to use.
    axis_to_reduce: The axis to reduce the array over.

  Returns:
    The simulate_quantized array.
  """
  method = _quantization_utils.QuantizationMethod(method)
  match method:
    case _quantization_utils.QuantizationMethod.NONE:
      return x
    case _quantization_utils.QuantizationMethod.INT4:
      return _simulate_uniform_quantization(
          x,
          bitwidth=4,
          granularity=_quantization_utils.QuantizationGranularity.PER_CHANNEL,
          axis_to_reduce=axis_to_reduce,
      )
    case _quantization_utils.QuantizationMethod.INT8:
      return _simulate_uniform_quantization(
          x,
          bitwidth=8,
          granularity=_quantization_utils.QuantizationGranularity.PER_CHANNEL,
      )
    case _quantization_utils.QuantizationMethod.Q4_0:
      return _simulate_uniform_quantization(
          x,
          bitwidth=4,
          granularity=_quantization_utils.QuantizationGranularity.PER_BLOCK,
      )
    case _quantization_utils.QuantizationMethod.Q4_0_TRANSPOSE:
      return _simulate_uniform_quantization(
          x,
          bitwidth=4,
          granularity=_quantization_utils.QuantizationGranularity.PER_BLOCK,
          transpose=True,
      )
    case _quantization_utils.QuantizationMethod.SFP8:
      return _simulate_sfp8_quantization(x)
    case _:
      raise ValueError(f'Unknown quantization method: {method}')


class SimulateQuantizedDense(nn.Module):
  """Wrapper around `nn.Dense` which adds a Quantized adapter."""

  _: dataclasses.KW_ONLY

  wrapped: nn.Dense
  method: _quantization_utils.QuantizationMethod = (
      _quantization_utils.QuantizationMethod.NONE
  )

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': {'wrapped': params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    kernel = self.param(  # pytype: disable=wrong-keyword-args
        'kernel',
        self.wrapped.kernel_init,
        (inputs.shape[-1], self.wrapped.features),
        dtype=self.wrapped.dtype,
    )
    w = simulate_quantize(
        kernel,
        self.method,
        axis_to_reduce=None,
    )
    y = inputs @ w
    if self.wrapped.use_bias:
      b = self.param(  # pytype: disable=wrong-keyword-args
          'bias',
          self.wrapped.bias_init,
          (self.wrapped.features,),
          dtype=self.wrapped.dtype,
      )
      y += b
    return y


class SimulateQuantizedEinsum(nn.Module):
  """Wrapper around `nn.Einsum` which adds a Quantized adapter."""

  _: dataclasses.KW_ONLY

  wrapped: nn.Einsum
  method: _quantization_utils.QuantizationMethod = (
      _quantization_utils.QuantizationMethod.NONE
  )

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': {'wrapped': params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  def process_einsum_str(self, einsum_str: str) -> str:
    """Processes the einsum string."""

    einsum_str = einsum_str.replace(' ', '')
    if '->' not in einsum_str:
      raise ValueError(
          '`einsum_str` equation must be explicit and include "->".'
      )
    if einsum_str.count(',') != 1:
      raise ValueError(
          '`einsum_str` equation must have exactly two operands and '
          'therefore, exactly one comma character, instead of '
          f'{einsum_str.count(",")}'
      )
    return einsum_str

  @nn.compact
  def __call__(self, inputs: Array, einsum_str: str | None = None) -> Array:
    einsum_str = nn.merge_param(
        'einsum_str', self.wrapped.einsum_str, einsum_str
    )
    einsum_str = self.process_einsum_str(einsum_str)

    kernel = self.param(
        'kernel',
        self.wrapped.kernel_init,
        self.wrapped.shape,
        self.wrapped.param_dtype,
    )

    inputs, kernel, _ = flax_dtypes.promote_dtype(
        inputs, kernel, None, dtype=self.wrapped.dtype
    )

    kernel = simulate_quantize(
        kernel,
        self.method,
        axis_to_reduce=get_axis_to_reduce_from_einsum_str(
            einsum_str=self.wrapped.name
        ),
    )

    y = jnp.einsum(einsum_str, inputs, kernel, precision=self.wrapped.precision)
    if self.wrapped.use_bias:
      bias_shape, _ = self.wrapped._get_bias_shape(einsum_str, inputs, kernel)
      bias = self.param(
          'bias', self.wrapped.bias_init, bias_shape, self.wrapped.param_dtype
      )
      y += bias
    return y


class IntDense(nn.Module):
  """Wrapper around `nn.Dense` which adds a Quantized adapter."""

  _: dataclasses.KW_ONLY

  wrapped: nn.Dense
  dtype: jnp.dtype = jnp.int4

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': {'wrapped': params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    kernel = self.param(  # pytype: disable=wrong-keyword-args
        'kernel',
        nn.initializers.ones_init(),
        (inputs.shape[-1], self.wrapped.features),
        self.dtype,
    )
    scale = self.param(
        'scale',
        nn.initializers.ones_init(),
        (1, self.wrapped.features),
        self.wrapped.dtype,
    )
    w = kernel.astype(self.wrapped.dtype) / scale
    y = inputs @ w
    if self.wrapped.use_bias:
      b = self.param(  # pytype: disable=wrong-keyword-args
          'bias',
          self.wrapped.bias_init,
          (self.wrapped.features,),
          dtype=self.wrapped.dtype,
      )
      y += b
    return y


class IntEinsum(nn.Module):
  """Wrapper around `nn.Einsum` which adds a Quantized adapter."""

  _: dataclasses.KW_ONLY

  wrapped: nn.Einsum
  dtype: jnp.dtype = jnp.int4

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': {'wrapped': params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  def process_einsum_str(self, einsum_str: str) -> str:
    """Processes the einsum string."""

    einsum_str = einsum_str.replace(' ', '')
    if '->' not in einsum_str:
      raise ValueError(
          '`einsum_str` equation must be explicit and include "->".'
      )
    if einsum_str.count(',') != 1:
      raise ValueError(
          '`einsum_str` equation must have exactly two operands and '
          'therefore, exactly one comma character, instead of '
          f'{einsum_str.count(",")}'
      )
    return einsum_str

  @nn.compact
  def __call__(self, inputs: Array, einsum_str: str | None = None) -> Array:
    einsum_str = nn.merge_param(
        'einsum_str', self.wrapped.einsum_str, einsum_str
    )
    einsum_str = self.process_einsum_str(einsum_str)

    kernel = self.param(
        'kernel',
        nn.initializers.ones_init(),
        self.wrapped.shape,
        self.dtype,
    ).astype(self.wrapped.dtype)
    scale = self.param(
        'scale',
        nn.initializers.ones_init(),
        tuple(jnp.ones(len(self.wrapped.shape) - 1, dtype=jnp.int32).tolist())
        + (self.wrapped.shape[-1],),
        self.wrapped.dtype,
    )

    inputs, kernel, _ = flax_dtypes.promote_dtype(
        inputs, kernel, None, dtype=self.wrapped.dtype
    )

    kernel = kernel / scale

    y = jnp.einsum(einsum_str, inputs, kernel, precision=self.wrapped.precision)
    if self.wrapped.use_bias:
      bias_shape, _ = self.wrapped._get_bias_shape(einsum_str, inputs, kernel)
      bias = self.param(
          'bias', self.wrapped.bias_init, bias_shape, self.wrapped.param_dtype
      )
      y += bias
    return y


def _straight_through_estimator(
    func: Callable[..., Array],
) -> Callable[..., Array]:
  """Decorator to make a straight through estimator w.r.t the first argument.

  Args:
    func: The function to make a straight through estimator for

  Returns:
    The straight through estimator
  """

  def wrapper(x: Array, *args, **kwargs):
    zero = x - jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(func(x, *args, **kwargs))
    return zero + y

  return wrapper


def _pack(
    x: Array, *, transpose: bool = True
) -> tuple[Array, jax.ShapeDtypeStruct]:
  """Reshape the input array into shape (-1, block_size)."""
  if transpose:
    x = einops.rearrange(x, '... a b -> ... b a')
  orig_shape, orig_dtype = x.shape, x.dtype
  orig_shape_dtype = jax.ShapeDtypeStruct(shape=orig_shape, dtype=orig_dtype)
  assert orig_shape[-1] % _QUANTIZATION_BLOCK_SIZE == 0
  n_blocks = x.size // _QUANTIZATION_BLOCK_SIZE
  blocks = x.reshape((n_blocks, _QUANTIZATION_BLOCK_SIZE)).astype(jnp.float32)
  return blocks, orig_shape_dtype


def _unpack(
    x: Array,
    *,
    transpose: bool,
    orig_shape_dtype: jax.ShapeDtypeStruct,
    cast_to_orig_dtype: bool = True,
) -> Array:
  """Reshape back the input array to its original shape."""
  res = x.reshape(orig_shape_dtype.shape)
  if cast_to_orig_dtype:
    res = res.astype(orig_shape_dtype.dtype)
  if transpose:
    res = einops.rearrange(res, '... b a -> ... a b')
  return res


def _q4_0(
    x: Array, *, upper_bound: int, bitwidth: int, transpose: bool = False
) -> Array:
  """Approximation of x using Q4_0."""

  def _simulate_quantize_block(arr: Array) -> Array:
    """Quantizes a single block of the input array."""
    assert arr.ndim == 1

    amax_idx = jnp.argmax(jnp.abs(arr))
    max_val = arr[amax_idx]

    d = -max_val / upper_bound
    safe_d = jnp.where(d == 0.0, 1.0, d)
    safe_inv_d = jnp.where(d == 0.0, 0.0, 1.0 / safe_d)

    scale = d.astype(jnp.float16)
    qs = arr * safe_inv_d
    qs = jnp.clip(jnp.trunc(qs + upper_bound + 0.5), 0.0, 2**bitwidth - 1)
    d = scale.astype(jnp.float32)
    return d * (qs - upper_bound)

  packed_x, orig_shape_dtype = _pack(x, transpose=transpose)
  vmaped_approx = jax.vmap(_simulate_quantize_block, in_axes=0, out_axes=0)
  packed_approx_x = vmaped_approx(packed_x)
  approx_x = _unpack(
      packed_approx_x, transpose=transpose, orig_shape_dtype=orig_shape_dtype
  )
  return approx_x


@_straight_through_estimator
def _simulate_uniform_quantization(
    x: Array,
    *,
    bitwidth: int,
    granularity: _quantization_utils.QuantizationGranularity,
    transpose: bool = False,
    axis_to_reduce: int | None = None,
) -> Array:
  """Applies uniform quantization to the given array.

  Quantization here is defined as x_hat = round(x * scale) / scale. We wrap this
  op in a straight through estimator for optimization compatibility.

  Args:
    x: The array to simulate_quantize.
    bitwidth: The bitwidth of the quantization.
    granularity: The granularity of the quantization.
    transpose: Whether to transpose the array before quantization.
    axis_to_reduce: The axis to reduce the max over.

  Returns:
    The simulate_quantized array.
  """
  # get bitwidth bounds
  assert bitwidth >= 2  # binary quantization is not supported
  lower_bound = -(2 ** (bitwidth - 1))
  upper_bound = 2 ** (bitwidth - 1) - 1
  # compute scales
  match granularity:
    case _quantization_utils.QuantizationGranularity.PER_TENSOR:
      max_ = jnp.max(jnp.abs(x))
      scales = upper_bound * jnp.squeeze(jnp.array([1.0 / max_]))
    case _quantization_utils.QuantizationGranularity.PER_CHANNEL:
      if axis_to_reduce is None:
        max_ = _quantization_utils.reduce_max_all_but_one_axis(
            jnp.abs(x), axis=-1
        )
      else:
        max_ = jnp.max(jnp.abs(x), keepdims=True, axis=axis_to_reduce)
      scales = upper_bound * jnp.array(1.0 / max_)
    case _quantization_utils.QuantizationGranularity.PER_BLOCK:
      return _q4_0(
          x, upper_bound=upper_bound + 1, bitwidth=bitwidth, transpose=transpose
      )
    case _:
      raise ValueError(f'Unknown granularity: {granularity}')

  # apply scale
  x = x * scales
  # clip and round
  x = jnp.clip(jnp.round(x), lower_bound, upper_bound)
  # unscale
  x = x / jnp.maximum(scales, _NON_ZERO_DIVISION_EPS)
  return x


def _floor_to_closest_power_of_two(x: Array, eps: float = 1e-6) -> Array:
  """Floor x to the closest power of two.

  We compute the log2 of x and floor it to the closest integer. Then we
  re-compute the value of x from the floored log2. To avoid logs of zero, we
  extract the maximum between abs(x) and eps. Furthermore, we handle negative
  values by extracting the sign ahead of computing the log.

  Args:
    x: The array to floor
    eps: A small value to threshold "zero" (choose carefully to avoid
      underflows)

  Returns:
    The floored array
  """
  x_sign = jnp.sign(x)
  return x_sign * jnp.exp2(jnp.floor(jnp.log2(jnp.maximum(jnp.abs(x), eps))))


@_straight_through_estimator
def _simulate_sfp8_quantization(x: Array) -> Array:
  """Applies SFP8 quantization to the given array.

  Args:
    x: The array to simulate_quantize.

  Returns:
    The simulate_quantized array.
  """
  sfp8_scale = _SFP8_UPPER_BOUND / jnp.maximum(
      _quantization_utils.reduce_max_all_but_one_axis(jnp.abs(x), -1),
      _NON_ZERO_DIVISION_EPS,
  )
  x = x * sfp8_scale

  x_sign = jnp.sign(x)
  x = jnp.abs(x)
  zero_mask = x < _SFP8_ZERO_THRESHOLD
  lower_bound_mask = x < _SFP8_LOWEST_NON_ZERO_BOUND
  lower_bound = jnp.maximum(
      _floor_to_closest_power_of_two(x), _SFP8_LOWEST_NON_ZERO_BOUND
  )
  switched_step_mask = jnp.where(lower_bound <= 1.0 / 256.0, 2.0, 1.0)
  step_size = switched_step_mask * lower_bound / 8.0
  x = jnp.round((x - lower_bound) / step_size) * step_size + lower_bound
  x = jnp.where(lower_bound_mask, _SFP8_LOWEST_NON_ZERO_BOUND, x)
  x = jnp.where(zero_mask, 0, x)
  return x_sign * x / sfp8_scale


def get_axis_to_reduce_from_einsum_str(
    einsum_str: str,
) -> Sequence[int] | None:
  """Returns the axis to reduce over."""
  match einsum_str:
    case 'BTD,NDH->BTNH':
      return (1,)
    case 'BSD,CKDH->CBSKH':
      return (2,)
    case 'BTNH,NHD->BTD':
      return (0, 1)
    case '...F,NHF->...NH':
      return (2,)
    case '...H,HF->...F':
      return (0,)
    case _:
      return None
