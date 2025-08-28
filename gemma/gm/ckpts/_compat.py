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

import collections
import re
from typing import Any, Optional

import flax
from gemma.gm.typing._common import Params  # pylint: disable=g-importing-member
import jax
import jax.numpy as jnp


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


def get_attention_pattern_len(params: Params) -> int:
  """Returns the size of the attention pattern."""
  attention_pattern_len = 0
  for k, _ in params.items():
    if 'stacked_layers/attention_type_' in k:
      _, index, _ = _parse_stacked_layers_key(k)
      if index is not None:
        attention_pattern_len = max(index + 1, attention_pattern_len)
  return attention_pattern_len


def unstack_params(params: Params) -> Params:
  """Unstack the params stacked by attention pattern to a flattened out format."""
  new_params = collections.defaultdict(dict)

  attention_pattern_len = get_attention_pattern_len(params)

  flat_key_format = '{prefix}layer_{layer_number}{suffix}'

  for k, v in params.items():
    if 'stacked_layers/attention_type_' in k:
      prefix, pattern_index, suffix = _parse_stacked_layers_key(k)
      if prefix is None or pattern_index is None or suffix is None:
        raise ValueError(f'Invalid stacked layers key: {k}')
      for parameter_name, parameter_value in v.items():
        # parameter_name is typically of the form 'w' or 'bias' or 'scale'
        if isinstance(parameter_value, jax.ShapeDtypeStruct):
          # Metadata only (jax.ShapeDtypeStruct)
          num_slices, *slice_shape = parameter_value.shape
          for slice_index in range(num_slices):
            layer_number = slice_index * attention_pattern_len + pattern_index
            layer_key = flat_key_format.format(
                prefix=prefix,
                layer_number=layer_number,
                suffix=suffix,
            )
            new_params[layer_key][parameter_name] = (
                jax.ShapeDtypeStruct(
                    shape=slice_shape,
                    dtype=parameter_value.dtype,
                )
            )
        else:
          # Actual weight tensors are sliced
          for i, parameter_slice in enumerate(parameter_value):
            layer_number = i * attention_pattern_len + pattern_index
            layer_key = flat_key_format.format(
                prefix=prefix,
                layer_number=layer_number,
                suffix=suffix,
            )
            new_params.setdefault(layer_key, {})[
                parameter_name
            ] = parameter_slice
    else:
      new_params[k] = v
  return dict(new_params)


def stack_params(params: Params, attn_pattern_len: int) -> Params:
  """Stack the params from a flattened format to a stacked format."""
  new_params: dict[str, Any] = {}

  # An intermediate collector to group parameter slices before stacking them.
  # e.g., {('.../stacked_layers/attention_type_0/...', 'w'):
  #        {0: slice0, 1: slice1}}
  stacked_layers_collector: dict[tuple[str, str], dict[int, Any]] = {}

  for key, value in params.items():
    parsed_key = _parse_flattened_layers_key(key)

    if parsed_key:
      prefix, layer_number, suffix = parsed_key

      pattern_index = layer_number % attn_pattern_len
      slice_index = layer_number // attn_pattern_len

      target_key = (
          f'{prefix}stacked_layers/attention_type_{pattern_index}{suffix}'
      )

      # value is a dict of parameter names to tensors (e.g., {'w': tensor})
      for param_name, param_value in value.items():
        collector_key = (target_key, param_name)
        stacked_layers_collector.setdefault(collector_key, {})[
            slice_index
        ] = param_value
    else:
      new_params[key] = value

  for (target_key, param_name), slices_dict in stacked_layers_collector.items():
    if not slices_dict:
      continue

    sorted_slices = [v for _, v in sorted(slices_dict.items())]

    if isinstance(sorted_slices[0], jax.ShapeDtypeStruct):
      num_slices = len(sorted_slices)
      old_shape_dtype = sorted_slices[0]
      new_shape = (num_slices, *old_shape_dtype.shape)
      stacked_value = old_shape_dtype.update(shape=new_shape)
    else:
      stacked_value = jnp.stack(sorted_slices)

    new_params.setdefault(target_key, {})[param_name] = stacked_value

  return new_params


def _parse_flattened_layers_key(key: str) -> Optional[tuple[str, int, str]]:
  """Parses a flattened layer key using regex.

  For example, 'transformer/layer_15/attn/attn_vec_einsum' -> ('transformer/',
  15, '/attn/attn_vec_einsum'). Returns None if the key does not match the
  pattern.

  Args:
    key: The flattened layer key to parse.

  Returns:
    A tuple containing the prefix, layer number, and suffix, or None if the key
    does not match the pattern.
  """
  # Regex to capture the prefix, layer number, and suffix.
  match = re.match(r'(.*?)layer_(\d+)(.*)', key)
  if match:
    prefix, layer_number_str, suffix = match.groups()
    return prefix, int(layer_number_str), suffix
  return None


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
