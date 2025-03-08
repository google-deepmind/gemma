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

"""QAT wrapper around Gemma models."""

import dataclasses
import functools
from typing import Any

from flax import linen as nn
from gemma import layers
from gemma import peft
import jax
from kauldron import kontext


class QATWrapper(nn.Module):
  """Wrapper around a Gemma model to enable quantization aware training.

  The model wrapped will have all it's `nn.Dense`, `nn.Einsum`,... layers
  replaced by their QAT versions. See `gemma.peft` documentation for more
  details.

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
    # gm.nn.QATWrapper(
    #   model=MyModel(
    #     input='batch.input',  # keys propagated to the `QATWrapper`
    #   ),
    # )
    return kontext.get_keypaths(self.model)

  def __getattr__(self, name: str) -> Any:
    # Forward attribute accesses to the wrapped model.
    return getattr(self.model, name)


class Int4Wrapper(nn.Module):
  """Wrapper around a Gemma model to enable int4 inference.

  The model wrapped will have all it's `nn.Dense`, `nn.Einsum`,... layers
  replaced by their int4 versions. See `gemma.peft` documentation for more
  details.

  Attributes:
    model: The model to wrap.
  """

  _: dataclasses.KW_ONLY

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
    replace_module_fn = functools.partial(_replace_by_int4)
    with peft.ModuleInterceptor(replace_module_fn):
      return self.model(*args, **kwargs)

  def __kontext_keys__(self) -> dict[str, str]:
    """Kauldron keys when calling `kontext.get_from_keys_obj`."""
    # Forward the keys from the wrapped model.
    # This allow to define the config as:
    # gm.nn.QATWrapper(
    #   model=MyModel(
    #     input='batch.input',  # keys propagated to the `QATWrapper`
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
    case layers.Einsum():
      # This hack is required because the FeedForward layer call two different
      # Einsum with using `nn.share_scope`, so the two wrappers need a different
      # name.
      # This seems to be a bug in flax interceptor.
      if module.weight_name != 'w':
        name = f'_SimulateQuantizedEinsum_{module.weight_name}'
      else:
        name = None
      return _SimulateQuantizedEinsum(name=name, method=method, wrapped=module)
    case _:
      return module


def _replace_by_int4(module: nn.Module):
  match module:
    case nn.Dense():
      return peft.Int4Dense(wrapped=module)
    case nn.Einsum():
      return peft.Int4Einsum(wrapped=module)
    case layers.Einsum():
      # This hack is required because the FeedForward layer call two different
      # Einsum with using `nn.share_scope`, so the two wrappers need a different
      # name.
      # This seems to be a bug in flax interceptor.
      if module.weight_name != 'w':
        name = f'_Int4Einsum_{module.weight_name}'
      else:
        name = None
      return _Int4Einsum(name=name, wrapped=module)
    case _:
      return module


class _SimulateQuantizedEinsum(nn.Module):
  """LoRA wrapper around a Gemma Einsum."""

  _: dataclasses.KW_ONLY
  wrapped: layers.Einsum
  method: peft.QuantizationMethod = peft.QuantizationMethod.NONE

  # Do not use `nn.share_scope` here as the `wrapped` module inside
  # `FeedForward` already uses `nn.share_scope`, so the two Einsum used in
  # the `FeedForward` would colide.
  # TODO(epot): Remove this hack by updating the checkpoint loader to re-map
  # the params structure.

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    # Warning: Calling multiple times with different `einsum_str` will
    # fail as the decomposition would not be the same.
    q_module = peft.SimulateQuantizedEinsumAdapter(
        einsum_str=eqn,
        method=self.method,
        name='QAT',
        shape=self.wrapped.shape,
    )
    return q_module(x)


class _Int4Einsum(nn.Module):
  """LoRA wrapper around a Gemma Einsum."""

  _: dataclasses.KW_ONLY
  wrapped: layers.Einsum

  # Do not use `nn.share_scope` here as the `wrapped` module inside
  # `FeedForward` already uses `nn.share_scope`, so the two Einsum used in
  # the `FeedForward` would colide.
  # TODO(epot): Remove this hack by updating the checkpoint loader to re-map
  # the params structure.

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    # Warning: Calling multiple times with different `einsum_str` will
    # fail as the decomposition would not be the same.
    q_module = peft.Int4EinsumAdapter(
        name='Int4', einsum_str=eqn, shape=self.wrapped.shape, dtype=x.dtype
    )
    return q_module(x)
