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

"""Utils to load pre-trained weights."""

import dataclasses
import operator
import sys
import typing
from typing import TypeVar

from etils import epath
from gemma import params as params_lib
import jax
import jax.numpy as jnp
from kauldron import kd
from orbax import checkpoint as ocp

if typing.TYPE_CHECKING:
  # Likely overkill, but avoid resolving the lazy-import on importing this file.
  _StateT = TypeVar('_StateT', bound=kd.train.TrainState)
else:
  _StateT = TypeVar('_StateT')


# TODO(epot): Should add some ModelInfo ? Like:
# class ModelInfo:
#   version: str = '2.0'
#   size: ModelSize = '2B'
#   variant: Variant | str = 'instruction_tuned'
# Then could be used to load `CheckpointPath.from_info()`,
# `Transformer.from_info()`,...


# TODO(epot): Should be part of core Kauldron
@dataclasses.dataclass(frozen=True, kw_only=True)
class LoadCheckpoint(kd.ckpts.AbstractPartialLoader):
  """Loads weights from a Gemma checkpoint.

  Note: The checpoint only contains the Gemma transformer weights, not the
  step, optimizer state,... Use `kd.ckpts.PartialKauldronLoader` to load
  the state from a Kauldron checkpoint.

  Attributes:
    path: The path to the orbax checkpoint.
  """

  path: epath.PathLike

  def transform(self, state: _StateT) -> _StateT:  # pytype: disable=signature-mismatch
    new_params = load_params(self.path, params=state.params)
    return dataclasses.replace(state, params=new_params)


def load_params(
    path: epath.PathLike,
    *,
    params: params_lib.Params | None = None,
    donate: bool = True,
) -> params_lib.Params:
  """Restore the params from a checkpoint.

  Args:
    path: The path to the orbax checkpoint.
    params: The state matching the checkpoint structure. Is used to restore the
      params with the correct sharding.
    donate: If `True` and `params` is provided, the memory from params will be
      released.

  Returns:
    The restored params.
  """
  ckpt = ocp.StandardCheckpointer()

  if donate and params is not None:
    params = release_memory(params)

  restore_fn = ckpt.restore

  # Apply transformations to the weights before/after restoring:
  # Note: Decorators are applied in reverse order (i.e. the outermost
  # decorator is called first but wrapped last here).

  # * Normalize the params to match the checkpoint layout.
  # TODO(epot): Should update the checkpoints to the new layout.
  if _is_legacy_layout(ckpt.metadata(path)):
    restore_fn = _unformat_format_params(restore_fn)

  params = restore_fn(path, params)

  # By default, Orbax restore as numpy rather than jax.
  params = jax.tree.map(jnp.asarray, params)
  return params


def _unformat_format_params(fn):
  """Decorator which unformat and format the params."""

  def decorator(path, params):
    if params is not None:
      params = _unreformat_params(params)
    params = fn(path, params)
    params = _reformat_params(params)
    return params

  return decorator


def _unreformat_params(params: params_lib.Params) -> params_lib.Params:
  params = params_lib.flatten_and_remap_params(params)
  params = {f'transformer/{k}': v for k, v in params.items()}
  return params


def _reformat_params(params: params_lib.Params) -> params_lib.Params:
  params = params_lib.param_remapper(params)
  params = params_lib.nest_params(params)
  params = params['transformer']
  return params


def release_memory(x):
  """Deletes and releases the memory of a Jax array."""
  return jax.tree.map(_release_memory, x)


def _release_memory(x):
  """Deletes and releases the memory of a Jax array."""
  if isinstance(x, jax.Array):
    x.delete()
  return x


def _is_legacy_layout(params: params_lib.Params) -> bool:
  """Returns True is the structure is the legacy one."""
  return all(k.startswith('transformer/') for k in params.keys())
