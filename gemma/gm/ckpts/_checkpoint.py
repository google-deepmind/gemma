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

"""Checkpoint loading logic."""

from __future__ import annotations

import dataclasses
import enum
import functools
import typing
from typing import Any, TypeVar

from etils import epath
import flax
from gemma import params as params_lib
from gemma.gm.ckpts import _quantization
import jax
from kauldron import kd
import numpy as np
from orbax import checkpoint as ocp

if typing.TYPE_CHECKING:
  # Likely overkill, but avoid resolving the lazy-import on importing this file.
  _StateT = TypeVar('_StateT', bound=kd.train.TrainState)
else:
  _StateT = TypeVar('_StateT')


# TODO(epot): Should be part of core Kauldron
@dataclasses.dataclass(frozen=True)
class LoadCheckpoint(kd.ckpts.AbstractPartialLoader):
  """Loads weights from a Gemma checkpoint.

  Note: The checpoint only contains the Gemma transformer weights, not the
  step, optimizer state,... Use `kd.ckpts.PartialKauldronLoader` to load
  the state from a Kauldron checkpoint.

  Attributes:
    path: The path to the orbax checkpoint.
    quantize: If `True`, the params will be mapped to enable quantization aware
      training.
  """

  path: epath.PathLike
  quantize: bool = False

  def transform(self, state: _StateT) -> _StateT:  # pytype: disable=signature-mismatch
    new_params = load_params(
        self.path, params=state.params, quantize=self.quantize
    )
    return dataclasses.replace(state, params=new_params)


class _CheckpointType(enum.StrEnum):
  """Structure of the checkpoint.

  Attributes:
    NESTED: This is the final structure matching the Flax `model.init()`
      structure, stored as nested dict (`{'layer_0': {'attn': ...}}`).
    FLAT: Internal checkpoint structure where params are stored as flat dict
      (e.g. `'{'transformer/layer_0/attn/_key_norm'' ...}`). The structure is
      very messy, but that's unfortunately how the official Gemma checkpoints
      where released.
    KAULDRON: Kauldron `kd.train.Trainer` checkpoint. Those checkpoints contains
      the optimizer state, step,... in addition to the params.
  """

  NESTED = enum.auto()
  FLAT = enum.auto()
  KAULDRON = enum.auto()


@flax.struct.dataclass
class _CheckpointTree:
  """Util class to convert checkpoint structures."""

  tree: params_lib.Params

  @classmethod
  def shape_dtype_struct_like(cls, tree: params_lib.Params) -> _CheckpointTree:
    """Returns a tree matching the input tree, but with `jax.ShapeDtypeStruct`."""
    tree = jax.tree.map(_as_shape_dtype_struct, tree)
    return _CheckpointTree(tree=tree)

  @functools.cached_property
  def type(self) -> _CheckpointType:
    """Structures of the checkpoint."""
    if _is_flat_layout(self.tree):
      return _CheckpointType.FLAT
    elif _is_kauldron_layout(self.tree):
      return _CheckpointType.KAULDRON
    else:
      return _CheckpointType.NESTED

  @functools.cached_property
  def nested_tree(self) -> params_lib.Params:
    """Returns the tree matching the NESTED checkpoint structure."""
    if self.type == _CheckpointType.FLAT:
      return _flat_to_nested(self.tree)
    elif self.type == _CheckpointType.KAULDRON:
      return self.tree['params']
    elif self.type == _CheckpointType.NESTED:
      return self.tree
    else:
      raise ValueError(f'Unsupported checkpoint structure: {self.type}')

  def as_nested(self, *, remove_mm: bool = False) -> _CheckpointTree:
    """Returns a copy of the tree matching the NESTED checkpoint structure."""
    tree = self.nested_tree
    if remove_mm:
      tree = _remove_mm_params(tree)
    return _CheckpointTree(tree=tree)

  def make_tree_for_params(self, params: _CheckpointTree) -> params_lib.Params:
    """Returns the tree matching the checkpoint structure."""
    metadata = _wrap_skip(self)

    # 1. Create the NESTED tree of the params
    # If the checkpoint has MM params, but those should not be restored,
    # should add skip mm params so the structure matches.
    ckpt_params = params.tree
    if self.has_mm_params and not params.has_mm_params:
      ckpt_params = _add_skip_mm_params(ckpt_params, metadata)

    # 2. Reformat the nested tree to match the checkpoint structure.
    if self.type == _CheckpointType.NESTED:
      target_params = ckpt_params  # No need to reformat
    elif self.type == _CheckpointType.FLAT:
      # Unflatten the params structure
      target_params = _nested_to_flat(ckpt_params)
    elif self.type == _CheckpointType.KAULDRON:
      target_params = _copy(metadata.tree)
      target_params['params'] = ckpt_params
    else:
      raise ValueError(f'Unsupported checkpoint structure: {self.type}')

    return target_params

  @functools.cached_property
  def has_mm_params(self) -> bool:
    return 'vision_encoder' in self.nested_tree


def load_params(
    path: epath.PathLike,
    *,
    params: params_lib.Params | None = None,
    donate: bool = True,
    text_only: bool = False,
    sharding: kd.sharding.ShardingTree | None = None,
    quantize: bool = False,
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
    quantize: If `True`, the params will be mapped to enable quantization aware
      training.

  Returns:
    The restored params.
  """
  if sharding is not None and params is not None:
    raise ValueError('`sharding` and `params` are mutually exclusive.')

  ckpt = ocp.StandardCheckpointer()

  metadata, path = _get_metadata_and_path(ckpt, path)

  metadata = _CheckpointTree.shape_dtype_struct_like(tree=metadata)

  # Eventually clear up the memory.
  if donate and params is not None:
    params = release_memory(params)

  # Params contain the output structure.
  if params is None:
    # If the params are not provided, we create a dummy tree matching the
    # checkpoint structure, so orbax restore as bfloat16 jax.Array, rather than
    # numpy arrays.
    # Params are always restored as NESTED
    params = metadata.as_nested(remove_mm=text_only and metadata.has_mm_params)
    if sharding is not None:
      params = kd.sharding.with_sharding_constraint(params, sharding)
  else:
    # If params explicitly provided, use that
    params = _CheckpointTree(tree=params)
    if params.type != _CheckpointType.NESTED:
      raise ValueError(
          'The input params provided to `load_params()` should be the raw'
          " model params matching the Flax `model.init()['params']` structure."
          f' Got: {_CheckpointType.NESTED}'
      )
    if text_only and params.has_mm_params:
      raise ValueError(
          'The input params provided to `load_params()` has multimodal params,'
          ' but `text_only` is `True`.'
      )

  if params.has_mm_params and not metadata.has_mm_params:
    raise ValueError(
        'The input params provided to `load_params()` has MM params, but the'
        ' checkpoint does not. This is not supported.'
    )

  # Restore the params
  # To supports different checkpoint structures, the original params have to
  # be remapped into the checkpoint structure.
  output_with_skip = metadata.make_tree_for_params(params)
  # TODO(epot): Support CMSMeta

  restore_fn = functools.partial(ckpt.restore, path)
  output = _partial_restore(restore_fn, output_with_skip, quantize=quantize)

  # Then after restoring, the params are remapped back to the final structure.
  output = _CheckpointTree(tree=output)
  output = output.as_nested(
      remove_mm=metadata.has_mm_params and not params.has_mm_params
  )

  # HACK: Manually cast the MM embedder params to f32, otherwise, image
  # produce wrong output on old GPUs (T4, V100)
  tree = output.tree
  if output.has_mm_params:
    tree['embedder']['mm_input_projection'] = jax.tree.map(
        lambda x: x.astype(np.float32),
        output.tree['embedder']['mm_input_projection'],
    )
    tree['embedder']['mm_soft_embedding_norm'] = jax.tree.map(
        lambda x: x.astype(np.float32),
        output.tree['embedder']['mm_soft_embedding_norm'],
    )
  return tree


# ======================== Structure reformat utils ========================


def _flat_to_nested(params: params_lib.Params) -> params_lib.Params:
  """Reformat the params from FLAT to NESTED."""
  params = _copy(params)
  # Split the params for the MM and the transformer.
  transformer_params = {
      k: v for k, v in params.items() if k.startswith('transformer/')
  }
  transformer_params = _flat_to_nested_single(
      transformer_params, name='transformer'
  )

  mm_params = {
      k: v for k, v in params.items() if k.startswith('SigLiPFromPatches_0/')
  }
  if mm_params:
    mm_params = _flat_to_nested_single(mm_params, name='SigLiPFromPatches_0')
    # TODO(epot): More conversions needed.
    transformer_params['vision_encoder'] = mm_params  # pytype: disable=unsupported-operands
  return transformer_params


def _nested_to_flat(params: params_lib.Params) -> params_lib.Params:
  """Reformat the params from NESTED to FLAT."""
  params = _copy(params)  # Copy to allow mutating the tree.

  mm_params = params.pop('vision_encoder', {})
  if mm_params:
    mm_params = _nested_to_flat_single(mm_params, name='SigLiPFromPatches_0')

  transformer_params = _nested_to_flat_single(params, name='transformer')

  # TODO(epot): Reshape the MM params too.
  return transformer_params | mm_params


def _nested_to_flat_single(
    params: params_lib.Params, *, name: str
) -> params_lib.Params:
  params = params_lib.flatten_and_remap_params(params)
  params = {f'{name}/{k}': v for k, v in params.items()}
  return params


def _flat_to_nested_single(
    params: params_lib.Params, *, name: str
) -> params_lib.Params:
  params = params_lib.param_remapper(params)
  params = params_lib.nest_params(params)
  params = params[name]
  return params


def _remove_mm_params(params):
  """Remove the MM params."""
  # Copy to allow mutating the tree.
  params = _copy(params)

  # TODO(epot): Once orbax supports partial restore, we would not need to
  # load those extra params in the first place.

  del params['vision_encoder']
  del params['embedder']['mm_input_projection']
  del params['embedder']['mm_soft_embedding_norm']
  return params


def _add_skip_mm_params(
    params: params_lib.Params, metadata: _CheckpointTree
) -> params_lib.Params:
  """Add skip MM params to restore."""
  params = _copy(params)
  params_with_mm = metadata.nested_tree

  # Params should not be restored in the first place.
  params['vision_encoder'] = params_with_mm['vision_encoder']
  for k in (
      'mm_input_projection',
      'mm_soft_embedding_norm',
  ):
    params['embedder'][k] = params_with_mm['embedder'][k]

  return params


def _is_flat_layout(params: params_lib.Params) -> bool:
  """Returns True is the structure is the legacy one."""
  return all(
      k.startswith(('transformer/', 'SigLiPFromPatches_0/'))
      for k in params.keys()
  )


def _is_kauldron_layout(params: params_lib.Params) -> bool:
  """Returns True is the structure is the Kauldron one."""
  return set(params) == {'collections', 'opt_state', 'params', 'step'}


# ======================== Skip utils ========================


@dataclasses.dataclass(frozen=True)
class _Skip:
  """Skip object to skip the restore of a param."""

  val: Any


def _wrap_skip(tree):
  """Wrap the params in a `Skip` object."""
  # Currently has no effect but when orbax will support partial restore,
  # this will skip the restore of those params.
  return jax.tree.map(_Skip, tree)


def _unwrap_skip(tree):
  return jax.tree.map(lambda x: x.val if isinstance(x, _Skip) else x, tree)


def _partial_restore(restore_fn, tree_with_skip, quantize: bool = False):
  """Restore the params with partial restore."""
  # TODO(epot): Implement once orbax supports partial restore.

  tree = _unwrap_skip(tree_with_skip)
  if quantize:
    tree = _quantization.convert_to_qat_checkpoint(tree)
  tree = restore_fn(tree)
  _release_skip(tree, tree_with_skip)
  return tree


def _release_skip(tree, tree_with_skip) -> None:
  """Release the memory of the skipped params."""
  jax.tree.map(
      lambda x, y: x.delete() if isinstance(y, _Skip) else None,
      tree,
      tree_with_skip,
  )


# ======================== Other utils ========================


def release_memory(x):
  """Deletes and releases the memory of a Jax array."""
  return jax.tree.map(_release_memory, x)


def _release_memory(x):
  """Deletes and releases the memory of a Jax array."""
  if isinstance(x, jax.Array):
    x.delete()
  return x


def _copy(tree):
  return jax.tree.map(lambda x: x, tree)


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
  metadata = dict(metadata)  # Normalize metadata
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
