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
  """Splits a nested dictionary into two, one with and one without 'lora' keys.

  This function recursively traverses the input dictionary and separates
  key-value pairs. If a key is 'lora', it and its corresponding value are
  placed into the 'lora' dictionary. All other keys are placed into the
  'original' dictionary. The structure of the dictionaries is preserved.

  Args:
    params: A nested dictionary representing the input tree containing 'lora'
      branches.

  Returns:
    A named tuple `(original, lora)` containing the two split dictionaries.
  """
  original_tree = {}
  lora_tree = {}

  for key, value in params.items():
    if key == "lora":
      lora_tree[key] = value
    elif isinstance(value, dict):
      original_sub, lora_sub = split_params(value)
      if original_sub:
        original_tree[key] = original_sub
      if lora_sub:
        lora_tree[key] = lora_sub
    else:
      original_tree[key] = value

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
