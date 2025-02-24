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

from __future__ import annotations

import dataclasses
import operator
import sys
import typing
from typing import TypeVar

from etils import epath
from gemma import params as params_lib
import jax
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
@dataclasses.dataclass(frozen=True)
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
    text_only: bool = False,
    sharding: kd.sharding.ShardingTree | None = None,
) -> params_lib.Params:
  """Restore the params from a checkpoint.

  Args:
    path: The path to the orbax checkpoint.
    params: The state matching the checkpoint structure. Is used to restore the
      params with the correct sharding.
    donate: If `True` and `params` is provided, the memory from params will be
      released.
    text_only: If `True`, only the text params are restored, and the multimodal
      params are ignored.
    sharding: If provided, the params will be restored with this sharding. This
      is mutually exclusive with `params`.

  Returns:
    The restored params.
  """
  if sharding is not None and params is not None:
    raise ValueError('`sharding` and `params` are mutually exclusive.')

  ckpt = ocp.StandardCheckpointer()

  metadata, path = _get_metadata_and_path(ckpt, path)

  # TODO(epot): Split the logic into `is_legacy` and not legacy.
  # Would be simpler.
  is_legacy = _is_legacy_layout(metadata)
  is_kauldron = _is_kauldron_layout(metadata)

  if donate and params is not None:
    params = release_memory(params)

  if params is None:
    # If the params are not provided, we create a dummy tree matching the
    # checkpoint structure, so orbax restore as bfloat16 jax.Array, rather than
    # numpy arrays.
    params = jax.tree.map(_as_shape_dtype_struct, metadata)
    if is_kauldron:
      params = params['params']
    if sharding is not None:
      params = kd.sharding.with_sharding_constraint(params, sharding)
    if is_legacy:
      params = _reformat_params(params)

  restore_fn = ckpt.restore

  # Apply transformations to the weights before/after restoring:
  # Note: Decorators are applied in reverse order (i.e. the outermost
  # decorator is called first but wrapped last here).

  # * Normalize the params to match the checkpoint layout.
  # TODO(epot): Should update the checkpoints to the new layout.
  if is_legacy:
    restore_fn = _unformat_format_params(restore_fn)

  if is_kauldron:
    restore_fn = _add_remove_kauldron_params(restore_fn, metadata)

  params = restore_fn(path, params)

  return params


def _unformat_format_params(fn):
  """Decorator which unformat and format the params."""

  def decorator(path, params):
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


def _add_remove_kauldron_params(fn, metadata):
  """Decorator for compatibility with Kauldron checkpoints."""

  def decorator(path, params):
    # Add Kauldron params
    # TODO(epot): Should use orbax partial loading instead when open-sourced.
    kauldron_params = _as_shape_dtype_struct(dict(metadata))
    kauldron_params['params'] = params

    kauldron_params = fn(path, kauldron_params)

    # Remove Kauldron params
    params = kauldron_params.pop('params')
    release_memory(kauldron_params)
    return params

  return decorator


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
  return all(
      k.startswith(('transformer/', 'SigLiPFromPatches_0/'))
      for k in params.keys()
  )


def _is_kauldron_layout(params: params_lib.Params) -> bool:
  """Returns True is the structure is the Kauldron one."""
  return set(params) == {'collections', 'opt_state', 'params', 'step'}


def _get_metadata_and_path(
    ckpt: ocp.StandardCheckpointer,
    path: epath.PathLike,
):
  """Returns the metadata of the checkpoint."""
  path = epath.Path(path)
  try:
    metadata = ckpt.metadata(path)
  except FileNotFoundError:
    # Kauldron checkpoints structure is different, so the params are contained
    # in a sub-directory
    if path.joinpath('_CHECKPOINT_METADATA').exists():
      path = path / 'default'
      metadata = ckpt.metadata(path)
    else:
      raise
  return metadata, path


def _as_shape_dtype_struct(tree):
  """Converts orbax ArrayMetadata to a jax.ShapeDtypeStruct."""
  return jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(
          dtype=x.dtype,
          shape=x.shape,
          # Set sharding so orbax restore the weights as `jax.Array` rather than
          # numpy.
          sharding=kd.sharding.REPLICATED,
      ),
      tree,
  )
