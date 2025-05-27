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

"""Params manipulation utils."""

from __future__ import annotations

from typing import Any, NamedTuple

_ParamsDict = dict[str, Any]


class SplittedParams(NamedTuple):
  original: _ParamsDict
  lora: _ParamsDict


def split_params(params: _ParamsDict) -> SplittedParams:
  """Split a nested tree into 2 trees, one with and without 'lora' branches.

  Example:

  ```python
  params = {
      'dense': {
          'kernel': w,
          'bias': b,
          'lora': {
              'a': a,
              'b': b,
          },
      },
      'other': other,
  }


  original, lora = peft.split_params(params)

  assert original == {
      'dense': {
          'kernel': w,
          'bias': b,
      },
      'other': other,
  }
  assert lora == {
      'dense': {
          'lora': {
              'a': a,
              'b': b,
          },
      },
  }
  ```

  Args:
    params: A nested dictionary representing the input tree containing 'lora'
      branches.

  Returns:
    A named tuple: `(original, lora)`
  """
  original_tree = {}
  lora_tree = {}

  def _split_recursive(input_subtree, original_subtree, lora_subtree):
    for key, value in input_subtree.items():
      if isinstance(value, dict):
        if key == 'lora':
          lora_subtree[key] = value
        else:
          original_subtree[key] = {}
          lora_subtree[key] = {}
          _split_recursive(value, original_subtree[key], lora_subtree[key])
      elif key != 'lora':
        original_subtree[key] = value

  _split_recursive(params, original_tree, lora_tree)

  # Remove empty dicts in lora_tree
  def _remove_empty_dicts(tree):
    if not isinstance(tree, dict):
      return tree

    new_tree = {}
    for key, value in tree.items():
      if isinstance(value, dict):
        sub_tree = _remove_empty_dicts(value)
        if sub_tree:  # Only add if subtree is not empty
          new_tree[key] = sub_tree
      else:
        new_tree[key] = value
    return new_tree

  lora_tree = _remove_empty_dicts(lora_tree)

  return SplittedParams(original_tree, lora_tree)


def merge_params(original: _ParamsDict, lora: _ParamsDict) -> _ParamsDict:
  """Inverse of `split_params`.

  Args:
    original: The original tree without the 'lora' branches.
    lora: The tree containing the 'lora' branches.

  Returns:
    The merged tree.
  """

  def _merge_recursive(original_subtree, lora_subtree):
    new_tree = {}

    for key, value in original_subtree.items():
      if isinstance(value, dict) and key in lora_subtree:
        new_tree[key] = _merge_recursive(value, lora_subtree[key])
      else:
        new_tree[key] = value

    # Add the branches not present in the original tree
    for k in sorted(set(lora_subtree) - set(original_subtree)):
      new_tree[k] = lora_subtree[k]

    return new_tree

  return _merge_recursive(original, lora)


def fuse_params():
  raise NotImplementedError()


def unfuse_params():
  raise NotImplementedError()
