# Copyright 2026 DeepMind Technologies Limited.
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
  peft: _ParamsDict


def split_params(params: _ParamsDict) -> SplittedParams:
  """Split a nested tree into 2 trees, one with and without 'peft' branches.

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


  original, peft = peft.split_params(params)

  assert original == {
      'dense': {
          'kernel': w,
          'bias': b,
      },
      'other': other,
  }
  assert peft == {
      'dense': {
          'lora': {
              'a': a,
              'b': b,
          },
      },
  }
  ```

  Args:
    params: A nested dictionary representing the input tree containing `peft
      branches (e.g. 'lora', 'prefix').

  Returns:
    A named tuple: `(original, peft)`
  """
  node_name_prefixes = ('lora', 'prefix')
  original_tree = {}
  peft_tree = {}

  def _split_recursive(input_subtree, original_subtree, peft_subtree):
    for key, value in input_subtree.items():
      if isinstance(value, dict):
        if key.startswith(node_name_prefixes):
          peft_subtree[key] = value
        else:
          original_subtree[key] = {}
          peft_subtree[key] = {}
          _split_recursive(value, original_subtree[key], peft_subtree[key])
      elif key.startswith(node_name_prefixes):
        peft_subtree[key] = value
      else:
        original_subtree[key] = value

  _split_recursive(params, original_tree, peft_tree)

  # Remove empty dicts in peft_tree
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

  peft_tree = _remove_empty_dicts(peft_tree)

  return SplittedParams(original_tree, peft_tree)


def merge_params(original: _ParamsDict, peft: _ParamsDict) -> _ParamsDict:
  """Inverse of `split_params`.

  Args:
    original: The original tree without the 'peft' branches.
    peft: The tree containing the 'peft' branches.

  Returns:
    The merged tree.
  """

  def _merge_recursive(original_subtree, peft_subtree):
    new_tree = {}

    for key, value in original_subtree.items():
      if isinstance(value, dict) and key in peft_subtree:
        new_tree[key] = _merge_recursive(value, peft_subtree[key])
      else:
        new_tree[key] = value

    # Add the branches not present in the original tree
    for k in sorted(set(peft_subtree) - set(original_subtree)):
      new_tree[k] = peft_subtree[k]

    return new_tree

  return _merge_recursive(original, peft)


def fuse_params():
  raise NotImplementedError()


def unfuse_params():
  raise NotImplementedError()
