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

"""QLoRA (Quantized Low-Rank Adaptation) wrapper around Gemma models."""

import dataclasses
import functools
from typing import Any

from flax import linen as nn
from flax.linen import dtypes as flax_dtypes
from gemma import layers
from gemma import peft
from gemma.peft import _quantization
import jax
from jax import numpy as jnp
from kauldron import kontext


class QLoRA(nn.Module):
  """Wrapper around a Gemma model to enable QLoRA fine-tuning.

  The model wrapped will have all its `nn.Dense`, `nn.Einsum`, etc. layers
  replaced by their QLoRA versions, which use quantized weights with LoRA adapters.
  See `gemma.peft` documentation for more details.

  Attributes:
    rank: The rank of the LoRA decomposition.
    model: The model to wrap.
    quant_method: The quantization method to use for the weights.
    dtype: The dtype to use for the LoRA weights.
  """

  _: dataclasses.KW_ONLY

  rank: int
  model: nn.Module
  quant_method: peft.QuantizationMethod = peft.QuantizationMethod.INT4
  dtype: jnp.dtype = jnp.bfloat16

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': model_params}}` rather than
    # `{'params': {'model': model_params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.model)

  @nn.compact
  def __call__(self, *args, **kwargs):
    """Calls the model with QLoRA applied to its weights."""
    replace_module_fn = functools.partial(
        _replace_by_qlora,
        rank=self.rank,
        quant_method=self.quant_method,
        dtype=self.dtype,
    )
    with peft.ModuleInterceptor(replace_module_fn):
      return self.model(*args, **kwargs)

  def __kontext_keys__(self) -> dict[str, str]:
    """Kauldron keys when calling `kontext.get_from_keys_obj`."""
    # Forward the keys from the wrapped model.
    # This allow to define the config as:
    # gm.nn.QLoRA(
    #   model=MyModel(
    #     input='batch.input',  # keys propagated to QLoRA
    #   ),
    # )
    return kontext.get_keypaths(self.model)

  def __getattr__(self, name: str) -> Any:
    # Forward attribute accesses to the wrapped model.
    return getattr(self.model, name)


def _replace_by_qlora(
    module: nn.Module,
    *,
    rank: int,
    quant_method: peft.QuantizationMethod,
    dtype: jnp.dtype,
) -> nn.Module:
  """Replaces compatible modules by their QLoRA version."""
  match module:
    case nn.Dense():
      return peft.QLoRADense(
          rank=rank, 
          wrapped=module, 
          quant_method=quant_method,
          dtype=dtype
      )
    case nn.Einsum():
      return peft.QLoRAEinsum(
          rank=rank, 
          wrapped=module, 
          quant_method=quant_method,
          dtype=dtype
      )
    case layers.Einsum():
      # This hack is required because the FeedForward layer calls two different
      # Einsum with using `nn.share_scope`, so the two wrappers need a different
      # name. This seems to be a bug in flax interceptor.
      if module.weight_name != 'w':
        name = f'_QLoRAEinsum_{module.weight_name}'
      else:
        name = None
      return _QLoRAEinsum(
          name=name,
          rank=rank,
          quant_method=quant_method,
          dtype=dtype,
          shape=module.shape,
          weight_name=module.weight_name,
          wrapped=module,
      )
    case _:
      return module


class _QLoRAEinsum(nn.Module):
  """QLoRA wrapper around a Gemma Einsum layer."""

  _: dataclasses.KW_ONLY
  
  rank: int
  quant_method: peft.QuantizationMethod
  dtype: jnp.dtype
  shape: tuple[int, ...]
  weight_name: str
  wrapped: layers.Einsum

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
    kernel = self.param(
        self.weight_name,
        self.wrapped.initializer if hasattr(self.wrapped, 'initializer') else nn.initializers.normal(),
        self.shape,
    )
    x, kernel, _ = flax_dtypes.promote_dtype(x, kernel, None, dtype=x.dtype)

    # Quantize the kernel
    kernel = _quantization.simulate_quantize(
        kernel,
        self.quant_method,
        axis_to_reduce=_quantization.get_axis_to_reduce_from_einsum_str(eqn),
    )

    # Compute the quantized forward pass
    y = jnp.einsum(eqn, x, kernel)

    # Add the LoRA adaptation
    # Use a unique name for each adapter to avoid name collision errors
    adapter_name = f'lora_{self.weight_name}'
    adapter = peft.QLoRAEinsumAdapter(
        name=adapter_name,
        rank=self.rank,
        einsum_str=eqn,
        shape=self.shape,
        dtype=self.dtype,
    )
    return y + adapter(x)