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

"""LoRA wrapper around Gemma models."""

import dataclasses
import functools
from typing import Any

from flax import linen as nn
from gemma import layers
from gemma import peft
import jax
import jax.numpy as jnp
from kauldron import kontext
import numpy as np


class LoRAWrapper(nn.Module):
  """Wrapper around a Gemma model to enable LoRA.

  The model wrapped will have all it's `nn.Dense`, `nn.Einsum`,... layers
  replaced by their LoRA versions. See `gemma.peft` documentation for more
  details.

  Attributes:
    rank: The rank of the LoRA decomposition.
    model: The model to wrap.
    dtype: The dtype to use for the LoRA weights.
  """

  _: dataclasses.KW_ONLY

  rank: int
  model: nn.Module
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
    """Calls the model."""
    replace_module_fn = functools.partial(
        _replace_by_lora,
        rank=self.rank,
        dtype=self.dtype,
    )
    with peft.ModuleInterceptor(replace_module_fn):
      return self.model(*args, **kwargs)

  def __kontext_keys__(self) -> dict[str, str]:
    """Kauldron keys when calling `kontext.get_from_keys_obj`."""
    # Forward the keys from the wrapped model.
    # This allow to define the config as:
    # gm.nn.LoRAWrapper(
    #   model=MyModel(
    #     input='batch.input',  # keys propagated to the `LoRAWrapper`
    #   ),
    # )
    return kontext.get_keypaths(self.model)

  def __getattr__(self, name: str) -> Any:
    # Forward attribute accesses to the wrapped model.
    return getattr(self.model, name)


def _replace_by_lora(
    module: nn.Module,
    *,
    rank: int,
    dtype: np.dtype,
) -> nn.Module:
  """Replaces compatible modules by their LoRA version."""
  # TODO(epot): Replace by generic LoRA wrapper ?
  match module:
    case nn.Dense():
      return peft.LoRADense(rank=rank, dtype=dtype, wrapped=module)
    case nn.Einsum():
      return peft.LoRAEinsum(rank=rank, dtype=dtype, wrapped=module)
    case layers.Einsum():
      # This hack is required because the FeedForward layer call two different
      # Einsum with using `nn.share_scope`, so the two wrappers need a different
      # name.
      # This seems to be a bug in flax interceptor.
      if module.weight_name != 'w':
        name = f'_LoRAEinsum_{module.weight_name}'
      else:
        name = None
      return _LoRAEinsum(name=name, rank=rank, dtype=dtype, wrapped=module)
    case _:
      return module


class _LoRAEinsum(nn.Module):
  """LoRA wrapper around a Gemma Einsum."""

  _: dataclasses.KW_ONLY
  rank: int
  dtype: np.dtype
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
    adapter = peft.LoRAEinsumAdapter(
        name='lora',
        rank=self.rank,
        dtype=self.dtype,
        einsum_str=eqn,
        shape=self.wrapped.shape,
    )
    return self.wrapped(eqn, x) + adapter(x)
