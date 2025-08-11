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

"""Reformatting Gemma params."""

from __future__ import annotations

import re

import flax
from gemma.gm.typing._common import Params  # pylint: disable=g-importing-member


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


def unstack_params(params: Params) -> Params:
  """Unstack the params stacked by attention pattern to a flattened out format."""
  new_params = {}

  attention_pattern_len = _get_attention_pattern_len(params)

  for k, v in params.items():
    if 'stacked_layers/attention_type_' in k:
      prefix, pattern_index, suffix = _parse_stacked_layers_key(k)
      if prefix is None or pattern_index is None or suffix is None:
        raise ValueError(f'Invalid stacked layers key: {k}')
      for parameter_name, parameter_value in v.items():
        # parameter_name is typically of the form 'w' or 'bias' or 'scale'
        for i, parameter_slice in enumerate(parameter_value):
          layer_number = i * attention_pattern_len + pattern_index
          layer_key = f'{prefix}layer_{layer_number}{suffix}'
          new_params.setdefault(layer_key, {})[parameter_name] = parameter_slice
    else:
      new_params[k] = v
  return new_params


def _parse_stacked_layers_key(input_string):
  """Parses a checkpoint key to extract the attention pattern index.

  Args:
    input_string: The checkpoint key.

  Returns:
    A tuple containing the prefix, index, and suffix, or (None, None, None) if
    no match is found.
  """
  match = re.search(
      r'(.*?)stacked_layers/attention_type_(\d+)(.*)', input_string
  )
  if match:
    prefix = match.group(1)
    index = int(match.group(2))
    suffix = match.group(3)
    return prefix, index, suffix
  return None, None, None


def _get_attention_pattern_len(params: Params) -> int:
  """Returns the size of the attention pattern."""
  attention_pattern_len = 0
  for k, _ in params.items():
    if 'stacked_layers/attention_type_' in k:
      _, index, _ = _parse_stacked_layers_key(k)
      if index is not None:
        attention_pattern_len = max(index + 1, attention_pattern_len)
  return attention_pattern_len
