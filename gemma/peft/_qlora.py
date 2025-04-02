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

"""Flax linen QLoRA modules for quantized weight, low-rank adaptation."""

from collections.abc import Sequence
import dataclasses

from flax import linen as nn
from flax.linen import dtypes as flax_dtypes
from flax.typing import Array  # pylint: disable=g-importing-member
from gemma.peft import _einsum_utils
from gemma.peft import _quantization
from gemma.peft import _quantization_utils
import jax
import jax.numpy as jnp


class QLoRADenseAdapter(nn.Module):
  """QLoRA module for Dense layers.

  This module implements the LoRA adapter part of QLoRA (x @ A @ B).
  Works in conjunction with quantized weights.
  """

  _: dataclasses.KW_ONLY

  rank: int
  features: int  # Output dimension.

  dtype: jnp.dtype = jnp.float_
  a_init: nn.initializers.Initializer = nn.initializers.kaiming_uniform()
  b_init: nn.initializers.Initializer = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    # Use standard parameter names 'a' and 'b' that match the original LoRA implementation
    a = self.param(  # pytype: disable=wrong-keyword-args
        'a', self.a_init, (inputs.shape[-1], self.rank), dtype=self.dtype
    )
    b = self.param(  # pytype: disable=wrong-keyword-args
        'b', self.b_init, (self.rank, self.features), dtype=self.dtype
    )
    return inputs @ a @ b


class QLoRADense(nn.Module):
  """QLoRA wrapper around quantized Dense layers.

  This module combines a quantized Dense layer with LoRA adapters.
  """

  _: dataclasses.KW_ONLY

  rank: int
  wrapped: nn.Dense
  quant_method: _quantization_utils.QuantizationMethod = (
      _quantization_utils.QuantizationMethod.INT4
  )

  dtype: jnp.dtype = jnp.float_
  a_init: nn.initializers.Initializer = nn.initializers.kaiming_uniform()
  b_init: nn.initializers.Initializer = nn.initializers.zeros_init()

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': {'wrapped': params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    # Quantize the Dense kernel
    kernel = self.param(  # pytype: disable=wrong-keyword-args
        'kernel',
        self.wrapped.kernel_init,
        (inputs.shape[-1], self.wrapped.features),
        dtype=self.wrapped.dtype,
    )
    w = _quantization.simulate_quantize(
        kernel,
        self.quant_method,
        axis_to_reduce=None,
    )
    # Compute the quantized forward pass
    y = inputs @ w
    if self.wrapped.use_bias:
      b = self.param(  # pytype: disable=wrong-keyword-args
          'bias',
          self.wrapped.bias_init,
          (self.wrapped.features,),
          dtype=self.wrapped.dtype,
      )
      y += b

    # Add the LoRA adaptation
    # Use a consistent name for proper RNG key handling
    adapter_name = 'lora_dense'
    adapter = QLoRADenseAdapter(
        name=adapter_name,
        rank=self.rank,
        features=self.wrapped.features,
        dtype=self.dtype,
        a_init=self.a_init,
        b_init=self.b_init,
    )
    return y + adapter(inputs)


class QLoRAEinsumAdapter(nn.Module):
  """QLoRA Einsum adapter module.

  This module implements the LoRA adapter part of QLoRA for Einsum operations.
  Works in conjunction with quantized weights.
  """

  _: dataclasses.KW_ONLY

  rank: int
  einsum_str: str
  shape: Sequence[int]

  dtype: jnp.dtype = jnp.float_
  a_init: nn.initializers.Initializer = nn.initializers.kaiming_uniform()
  b_init: nn.initializers.Initializer = nn.initializers.zeros_init()

  def setup(self):
    # Get the einsum decomposition given the original einsum op.
    # e.g. `BTNH,NHD->BTD` becomes `BTNH,NHr,rD->BTD`
    out = _einsum_utils.get_lora_einsum_str_and_shapes(
        einsum_str=self.einsum_str,
        weights_shape=self.shape,
        rank=self.rank,
    )
    (lora_einsum_str, a_shape, b_shape) = out

    self._lora_einsum_str = lora_einsum_str
    
    # Use standard 'a' and 'b' param names expected by split_params
    # This matches the approach in the original LoRAEinsumAdapter
    self._a = self.param(
        'a', 
        self.a_init, 
        a_shape, 
        dtype=self.dtype
    )
    self._b = self.param(
        'b', 
        self.b_init, 
        b_shape, 
        dtype=self.dtype
    )

  def __call__(self, inputs: Array) -> Array:
    return jnp.einsum(self._lora_einsum_str, inputs, self._a, self._b)


class QLoRAEinsum(nn.Module):
  """QLoRA wrapper around quantized Einsum layers.

  This module combines a quantized Einsum layer with LoRA adapters.
  """

  _: dataclasses.KW_ONLY

  rank: int
  wrapped: nn.Einsum
  quant_method: _quantization_utils.QuantizationMethod = (
      _quantization_utils.QuantizationMethod.INT4
  )

  dtype: jnp.dtype = jnp.float_
  a_init: nn.initializers.Initializer = nn.initializers.kaiming_uniform()
  b_init: nn.initializers.Initializer = nn.initializers.zeros_init()

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

    # Quantize the kernel
    kernel = _quantization.simulate_quantize(
        kernel,
        self.quant_method,
        axis_to_reduce=_quantization.get_axis_to_reduce_from_einsum_str(
            einsum_str=einsum_str
        ),
    )

    # Compute the quantized forward pass
    y = jnp.einsum(einsum_str, inputs, kernel, precision=self.wrapped.precision)
    if self.wrapped.use_bias:
      bias_shape, _ = self.wrapped._get_bias_shape(einsum_str, inputs, kernel)
      bias = self.param(
          'bias', self.wrapped.bias_init, bias_shape, self.wrapped.param_dtype
      )
      y += bias

    # Add the LoRA adaptation
    # Use a consistent name for proper RNG key handling
    adapter_name = 'lora_einsum'
    adapter = QLoRAEinsumAdapter(
        name=adapter_name,
        rank=self.rank,
        einsum_str=einsum_str,
        shape=self.wrapped.shape,
        dtype=self.dtype,
        a_init=self.a_init,
        b_init=self.b_init,
    )
    return y + adapter(inputs)