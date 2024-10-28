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

"""Utils for loading Gemma params."""

import functools
from typing import Any, Mapping, Optional

import flax
import jax
import jax.numpy as jnp
import orbax.checkpoint

Params = Mapping[str, Any]


def load_and_format_params(path: str) -> Params:
  """Loads parameters and formats them for compatibility."""
  params = load_params(path)
  param_state = jax.tree_util.tree_map(jnp.array, params)
  remapped_params = param_remapper(param_state)
  nested_params = nest_params(remapped_params)
  return nested_params


def load_metadata(path: str) -> Optional[Any]:
  """Loads metadata from a checkpoint path."""
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  metadata = checkpointer.metadata(path)
  return metadata


@functools.cache
def load_params(path: str) -> Params:
  """Loads parameters from a checkpoint path."""
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  params = checkpointer.restore(path)
  return params


def format_and_save_params(
    params: Params,
    path: str,
) -> None:
  """Formats and saves a parameter checkpoint to the path."""
  params = flatten_and_remap_params(params)
  save_params(params, path)


def save_params(params: Params, path: str) -> None:
  """Saves the given parameters to the given path."""
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  checkpointer.save(path, params)


def param_remapper(orig_params: Params) -> Params:
  """Remaps params to new module layout.

  This is needed here because the model definition  does not have a separate
  `mlp` module.

  Args:
    orig_params: original dict of parameters in Gemma format.

  Returns:
    dict of params with different names.
  """
  new_params = {}
  for k, v in orig_params.items():
    if 'mlp/' in k:
      layer_name, param = k.rsplit('/', maxsplit=1)
      if layer_name not in new_params:
        new_params[layer_name] = {}
      if 'w' in v:
        new_params[layer_name][param] = v['w']
    else:
      new_params[k] = v
  return new_params


def nest_params(params: Params) -> Params:
  """Nests params as a dict of dicts rather than a flat dict."""
  nested_params = {}
  for path, param in params.items():
    *path, leaf = path.split('/')
    subdict = nested_params
    for key in path:
      subdict = subdict.setdefault(key, {})
    subdict[leaf] = param
  return nested_params


def flatten_and_remap_params(params: Params) -> Params:
  """Flattens and remaps params from new to old module layout.

  Inverse of gemma.params.param_remapper(...) followed by
  gemma.params.nest_params(...).

  Args:
    params: Parameters in new Gemma format (deeply nested pytree)

  Returns:
    semi-flat dict of params with parameter names remapped to old format.
  """
  # Fully flatten the nested param dict
  params = flax.traverse_util.flatten_dict(params, sep='/')

  # Rename the paths in the flattened dict:
  # 1st, we add the 'w' for MLP layers, undoing the remapping from
  # `gemma.params.param_remapper(...)`:
  #  '../layer_?/mlp/linear' -> '../layer_/mlp/linear/w'
  #  '../layer_?/mlp/gating_einsum -> '../layer_/mlp/gating_einsum/w'
  # 2nd, separate the last component of the path with a `&` instead of a `/`,
  # because we need to unflatten one level closest to the leafs:
  def remap_name(n: str):
    if n.endswith('/mlp/linear') or n.endswith('/mlp/gating_einsum'):
      n += '/w'

    left, right = n.rsplit('/', maxsplit=1)
    return left + '&' + right

  params = {remap_name(k): v for k, v in params.items()}

  # Unflatten the leaf-level params again.
  return flax.traverse_util.unflatten_dict(params, sep='&')
