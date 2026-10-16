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

import itertools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

_ParamsDict = dict[str, Any]
_WEIGHT_KEYS = ('kernel', 'w')


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


def fuse_params(params: _ParamsDict) -> _ParamsDict:
  """Fuse LoRA adapters into the base weight tensors.

  For every subtree that contains both a base weight (`kernel` or `w`) and a
  `lora` branch with `a` / `b`, this replaces the weight with
  `weight + lora_delta(a, b)`.

  The `lora` adapters are left in the tree so `unfuse_params` can reverse the
  operation. After fusing, callers that want LoRA-free inference can drop the
  adapters with `split_params` and load the fused weights into a non-LoRA model.

  This mirrors the usual LoRA merge semantics:

  * Dense: `kernel += a @ b`
  * Einsum / DenseGeneral (no batch dims): contract `a` and `b` over the rank
    axis, then permute if needed so the result matches the weight layout.

  LoRA DenseGeneral layers with non-empty `batch_dims` are not supported
  because those adapters are not shared with a single base kernel.

  Args:
    params: Nested parameter tree, typically containing LoRA adapters.

  Returns:
    A new parameter tree with LoRA deltas folded into the base weights.
  """
  return _map_lora_fusion(params, sign=1)


def unfuse_params(params: _ParamsDict) -> _ParamsDict:
  """Inverse of `fuse_params`.

  Subtracts the LoRA delta from each fused base weight. Requires the `lora`
  adapters (`a`, `b`) to still be present in the tree (as left by
  `fuse_params`).

  Args:
    params: Nested parameter tree previously returned by `fuse_params`.

  Returns:
    A new parameter tree with LoRA deltas removed from the base weights.
  """
  return _map_lora_fusion(params, sign=-1)


def _map_lora_fusion(params: _ParamsDict, *, sign: int) -> _ParamsDict:
  """Recursively apply `weight += sign * lora_delta` for every LoRA node."""

  def _recurse(node: Any) -> Any:
    if not isinstance(node, dict):
      return node

    new_node = {key: _recurse(value) for key, value in node.items()}
    lora = new_node.get('lora')
    if not isinstance(lora, dict) or 'a' not in lora or 'b' not in lora:
      return new_node

    weight_key = _find_weight_key(new_node)
    if weight_key is None:
      return new_node

    delta = _lora_weight_delta(lora['a'], lora['b'], new_node[weight_key])
    new_node[weight_key] = new_node[weight_key] + sign * delta
    return new_node

  return _recurse(params)


def _find_weight_key(node: _ParamsDict) -> str | None:
  for key in _WEIGHT_KEYS:
    value = node.get(key)
    if _is_array(value):
      return key
  return None


def _is_array(value: Any) -> bool:
  return isinstance(value, (jax.Array, jnp.ndarray)) or (
      hasattr(value, 'shape')
      and hasattr(value, 'dtype')
      and hasattr(value, '__array__')
  )


def _lora_weight_delta(a: Any, b: Any, weight: Any) -> jax.Array:
  """Materialize the LoRA update with the same shape as `weight`."""
  a = jnp.asarray(a)
  b = jnp.asarray(b)
  weight = jnp.asarray(weight)

  if a.shape[-1] != b.shape[0]:
    raise ValueError(
        'Unsupported LoRA layout for fusion: expected `b` to start with the '
        f'rank dimension matching `a.shape[-1]` (got a.shape={a.shape}, '
        f'b.shape={b.shape}). LoRA DenseGeneral with non-empty batch_dims is '
        'not supported by fuse_params/unfuse_params.'
    )

  delta = jnp.tensordot(a, b, axes=([-1], [0]))
  if delta.shape == weight.shape:
    return delta

  if sorted(delta.shape) != sorted(weight.shape):
    raise ValueError(
        'LoRA delta shape is incompatible with the base weight: '
        f'delta.shape={delta.shape}, weight.shape={weight.shape}, '
        f'a.shape={a.shape}, b.shape={b.shape}.'
    )

  # Einsum LoRA factors are built as (reduced_dims + rank) / (rank + out_dims)
  # in weight-letter order, which can differ from the original weight layout.
  for perm in itertools.permutations(range(delta.ndim)):
    if tuple(delta.shape[i] for i in perm) == weight.shape:
      return jnp.transpose(delta, perm)

  raise ValueError(
      'Could not permute LoRA delta to match the base weight layout: '
      f'delta.shape={delta.shape}, weight.shape={weight.shape}.'
  )
