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

"""Jax utils."""

from __future__ import annotations

import functools
import inspect
import math
import types
import typing
from typing import Any, Callable, TypeAlias, TypeVar

import jax
import jaxtyping

_FnT = TypeVar('_FnT', bound=Callable[..., Any])


def flatten_unflatten_batch_dim() -> Callable[[_FnT], _FnT]:
  """Decorator which flattens/unflattens the batch dimension of the args/output.

  Example:

  ```python
  @flatten_unflatten_batch_dim()
  def fn(x: Float['*B L L'], y: Float['*B']):
    # Inside the function, the batch dimension is flattened.
    assert len(x.shape) == 3
    assert len(y.shape) == 1
    return x
  ```

  Returns:
    A decorator which flattens/unflattens the batch dimension of the args
      and outputs.
  """

  def decorator(fn: _FnT) -> _FnT:
    argname_to_non_batch_dim_size = None

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
      nonlocal argname_to_non_batch_dim_size
      if argname_to_non_batch_dim_size is None:
        argname_to_non_batch_dim_size = _get_argname_to_non_batch_dim_size(fn)

      sig = inspect.signature(fn)
      bound_args = sig.bind(*args, **kwargs)

      # Flatten the `*B` dimension
      batch_shape = _get_batch_shape(
          bound_args.arguments, argname_to_non_batch_dim_size
      )

      batch_size = math.prod(batch_shape)

      def _flatten_batch_dim(x):
        if isinstance(x, jax.Array):
          return x.reshape((batch_size,) + x.shape[len(batch_shape) :])
        else:
          return x

      bound_args.arguments = jax.tree.map(
          _flatten_batch_dim,
          bound_args.arguments,
      )

      output = fn(*bound_args.args, **bound_args.kwargs)

      # Unflatten the `*B` dimension
      output = jax.tree.map(
          lambda x: x.reshape([*batch_shape, *x.shape[1:]]),
          output,
      )
      return output

    return decorated

  return decorator


def _get_batch_shape(
    args: dict[str, Any],
    argname_to_non_batch_dim_size: dict[str, int],
) -> tuple[int, ...]:
  """Infer the batch shape from the args."""
  batch_shapes = set()
  # Collect all the batch shapes
  for k, v in args.items():
    if not isinstance(v, jax.Array):
      continue
    if k not in argname_to_non_batch_dim_size:
      continue
    batch_shapes.add(v.shape[: -argname_to_non_batch_dim_size[k] or None])
  if len(batch_shapes) != 1:
    raise ValueError(
        f'Could not infer batch shape. Got conflicting values: {batch_shapes}.'
    )
  return batch_shapes.pop()


def _get_argname_to_non_batch_dim_size(
    fn: Callable[..., Any],
) -> dict[str, int]:
  """Returns mapping from argument name to the number of non-batch dims.

  This allow to implement a function as if there were a single batch dimension,
  but the function actually supports arbitrary batch shapes.

  Example:

  ```python
  def fn(
      x: kd.typing.Float['*B N L'],  # 2 non-batch dims
      y: kd.typing.Int['*B'] | None,  # 0 non-batch dims
      z: bool,  # Not an array
  ) -> int:
    pass


  _get_argname_to_non_batch_dim_size(fn) == {
      'x': 2,
      'y': 0,
  }
  ```

  Args:
    fn: The function to inspect.

  Returns:
    A mapping from argument name to the number of non-batch dims.
  """
  hints = typing.get_type_hints(fn)

  argname_to_non_batch_dim_size = {
      k: _get_non_batch_dim_size(ann) for k, ann in hints.items()
  }
  return {
      k: v
      for k, v in argname_to_non_batch_dim_size.items()
      if v is not None and k != 'return'  # Ignore `->` returned type
  }


def _get_non_batch_dim_size(ann: TypeAlias) -> int | None:
  """Returns the number of non-batch dims from the annotation.

  Example:

  * `Float['*B N L']` -> 2
  * `Float['*B N']` -> 1
  * `Float['*B']` -> 0
  * `Float['*B'] | None` -> 0
  * `bool` -> None

  Args:
    ann: The annotation to inspect.

  Returns:
    The number of non-batch dims, or `None` if the annotation is not an array.
  """
  origin = typing.get_origin(ann)
  if origin in (types.UnionType, typing.Union):
    # Recurse into union types: `X | Y`
    non_batch_dim_size = {
        _get_non_batch_dim_size(a) for a in typing.get_args(ann)
    }
    non_batch_dim_size -= {None}  # Only keep Array annotations
    if len(non_batch_dim_size) > 1:
      raise ValueError(
          f'Could not infer non-batch size for `{ann}`. Got conflicting'
          ' values: {non_batch_dim_size}.'
      )
    elif len(non_batch_dim_size) == 1:
      return next(iter(non_batch_dim_size))
    else:
      return None
  elif _is_jaxtyping(ann):  # Leaf
    return _non_batch_dim_from_jaxtyping(ann)  # pytype: disable=wrong-arg-types
  else:
    return None


def _is_jaxtyping(ann: TypeAlias) -> bool:
  """Returns `True` is the annotation is a `jaxtyping.Array['...']`."""
  return (
      isinstance(ann, type)
      and jaxtyping._array_types.AbstractArray in ann.__bases__  # pylint: disable=protected-access
  )


def _non_batch_dim_from_jaxtyping(
    ann: jaxtyping._array_types.AbstractArray,
) -> int:
  """Returns the number of non-batch dim.

  Example:

  * `Array['*b']` -> 0
  * `Array['*b n']` -> 1
  * `Array['*b n l']` -> 2

  Args:
    ann: The `jaxtyping.Array['...']` annotation

  Returns:
    The number of non-batch dim.
  """
  # Could do a more fancy parsing, but should be good enough for this use case.
  return len(ann.dim_str.split()) - 1
