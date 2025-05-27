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

"""Quantization Aware Training (QAT) wrapper around Gemma models."""

import dataclasses
import functools
from typing import Any

from flax import linen as nn
from flax.linen import dtypes as flax_dtypes
from gemma import peft
from gemma.gm.nn import _layers
from gemma.peft import _quantization
import jax
from jax import numpy as jnp
from kauldron import kontext


class QuantizationAwareWrapper(nn.Module):
  """Wrapper around a Gemma model to enable quantization aware training.

  The model wrapped will have all it's `nn.Dense`, `nn.Einsum`,... layers
  replaced by their quantization aware training versions. See `gemma.peft`
  documentation for more details.

  Attributes:
    method: The quantization method to use.
    model: The model to wrap.
  """

  _: dataclasses.KW_ONLY

  method: peft.QuantizationMethod = peft.QuantizationMethod.NONE
  model: nn.Module

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': model_params}}` rather than
    # `{'params': {'model': model_params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.model)

  @nn.compact
  def __call__(self, *args, **kwargs):
    """Calls the model."""
    replace_module_fn = functools.partial(
        _replace_by_simulated_quantization, method=self.method
    )
    with peft.ModuleInterceptor(replace_module_fn):
      return self.model(*args, **kwargs)

  def __kontext_keys__(self) -> dict[str, str]:
    """Kauldron keys when calling `kontext.get_from_keys_obj`."""
    # Forward the keys from the wrapped model.
    # This allow to define the config as:
    # gm.nn.QuantizationAwareWrapper(
    #   model=MyModel(
    #     input='batch.input',  # propagated to wrapper
    #   ),
    # )
    return kontext.get_keypaths(self.model)

  def __getattr__(self, name: str) -> Any:
    # Forward attribute accesses to the wrapped model.
    return getattr(self.model, name)


class IntWrapper(nn.Module):
  """Wrapper around a Gemma model to enable int4 inference.

  The model wrapped will have all it's `nn.Dense`, `nn.Einsum`,... layers
  replaced by their int4 versions. See `gemma.peft` documentation for more
  details.

  Attributes:
    model: The model to wrap.
  """

  _: dataclasses.KW_ONLY

  model: nn.Module
  dtype: jnp.dtype = jnp.int4

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': model_params}}` rather than
    # `{'params': {'model': model_params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.model)

  @nn.compact
  def __call__(self, *args, **kwargs):
    """Calls the model."""
    replace_module_fn = functools.partial(_replace_by_int, dtype=self.dtype)
    with peft.ModuleInterceptor(replace_module_fn):
      return self.model(*args, **kwargs)

  def __kontext_keys__(self) -> dict[str, str]:
    """Kauldron keys when calling `kontext.get_from_keys_obj`."""
    # Forward the keys from the wrapped model.
    # This allow to define the config as:
    # gm.nn.QuantizationAwaretrainingWrapper(
    #   model=MyModel(
    #     input='batch.input',  # keys propagated to the `IntWrapper`
    #   ),
    # )
    return kontext.get_keypaths(self.model)

  def __getattr__(self, name: str) -> Any:
    # Forward attribute accesses to the wrapped model.
    return getattr(self.model, name)


def _replace_by_simulated_quantization(
    module: nn.Module, *, method: peft.QuantizationMethod
):
  match module:
    case nn.Dense():
      return peft.SimulateQuantizedDense(wrapped=module, method=method)
    case nn.Einsum():
      return peft.SimulateQuantizedEinsum(wrapped=module, method=method)
    case _layers.Einsum():
      # This hack is required because the FeedForward layer call two different
      # Einsum with using `nn.share_scope`, so the two wrappers need a different
      # name.
      # This seems to be a bug in flax interceptor.
      return _SimulateQuantizedEinsum(
          name=module.name + 'qat',
          method=method,
          shape=module.shape,
          weight_name=module.weight_name,
          wrapped=module,
      )
    case _:
      return module


def _replace_by_int(module: nn.Module, dtype: jnp.dtype):
  match module:
    case nn.Dense():
      return peft.IntDense(wrapped=module, dtype=dtype)
    case nn.Einsum():
      return peft.IntEinsum(wrapped=module, dtype=dtype)
    case _layers.Einsum():
      # This hack is required because the FeedForward layer call two different
      # Einsum with using `nn.share_scope`, so the two wrappers need a different
      # name.
      # This seems to be a bug in flax interceptor.
      if module.weight_name != 'w':
        name = f'_IntEinsum_{module.weight_name}'
      else:
        name = None
      return _IntEinsum(name=name, shape=module.shape, dtype=dtype)
    case _:
      return module


class _SimulateQuantizedEinsum(nn.Module):
  """LoRA wrapper around a Gemma Einsum."""

  _: dataclasses.KW_ONLY
  shape: tuple[int, ...]
  weight_name: str
  method: peft.QuantizationMethod
  kernel_init: nn.initializers.Initializer = nn.initializers.normal()
  wrapped: _layers.Einsum

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
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    eqn = self.process_einsum_str(eqn)
    kernel = self.param(self.weight_name, self.kernel_init, self.shape)
    x, kernel, _ = flax_dtypes.promote_dtype(x, kernel, None, dtype=x.dtype)

    kernel = peft.simulate_quantize(kernel, self.method)

    y = jnp.einsum(eqn, x, kernel)
    return y


class _IntEinsum(nn.Module):
  """LoRA wrapper around a Gemma Einsum."""

  _: dataclasses.KW_ONLY
  shape: tuple[int, ...]
  dtype: jnp.dtype

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

  def deduce_scale_shape(self, eqn: str) -> tuple[int, ...]:
    """Deduces the scale shape from the kernel shape."""
    axis_to_reduce = _quantization.get_axis_to_reduce_from_einsum_str(eqn)
    new_shape = list(self.shape)
    if axis_to_reduce is None:
      return tuple([1] * (len(self.shape) - 1)) + (self.shape[-1],)
    for axis in axis_to_reduce:
      new_shape[axis] = 1
    return tuple(new_shape)

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    eqn = self.process_einsum_str(eqn)
    kernel = self.param(
        'kernel',
        nn.initializers.ones_init(),
        self.shape,
        self.dtype,
    ).astype(x.dtype)
    scale = self.param(
        'scale',
        nn.initializers.ones_init(),
        self.deduce_scale_shape(eqn),
        x.dtype,
    )
    x, kernel, _ = flax_dtypes.promote_dtype(x, kernel, None, dtype=x.dtype)
    kernel = kernel / scale
    y = jnp.einsum(eqn, x, kernel)
    return y
