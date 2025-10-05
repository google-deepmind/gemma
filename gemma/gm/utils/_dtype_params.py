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

"""Training property."""

from collections.abc import Iterator
import contextlib
import dataclasses
import functools

from etils import edc
from etils.epy import _internal
from flax import linen as nn
import jax

_DType = jax.typing.DTypeLike


@dataclasses.dataclass(frozen=True, kw_only=True)
class _DTypeState:
  dtype: _DType | None
  exclude: list[str] | None


_dtypes_stack = edc.ContextStack[_DTypeState]()


@contextlib.contextmanager
def initialize_param_with_dtype(
    dtype: _DType | None,
    *,
    exclude: list[str] | None = None,
) -> Iterator[None]:
  """Set the params dtype to the given value.

  Inside the contextmanager, `self.param()` will use the given dtype.
  If nested, only the last contextmanager will be used.

  Args:
    dtype: The dtype to use.
    exclude: List of module paths to exclude from the dtype conversion.

  Yields:
    None
  """
  try:
    state = _DTypeState(dtype=dtype, exclude=exclude)
    _dtypes_stack.append(state)
    yield
  finally:
    _dtypes_stack.pop()


@functools.cache
def _mock_flax_module_param() -> None:
  """Mock `nn.Module.params` method to convert the params to dtype."""
  param = _internal.unwrap_on_reload(nn.Module.param)  # pylint: disable=protected-access

  @_internal.wraps_with_reload(param)
  def decorated(
      self: nn.Module,
      name: str,
      init_fn,  # : Callable[..., Any],
      shape: tuple[int, ...],
      dtype: _DType | None = None,
      **kwargs,
  ):
    if _should_replace_dtype(module=self, stack=_dtypes_stack):
      del dtype  # The dtype is overwritten by the contextmanager
      state = _dtypes_stack.stack[-1]
      return param(self, name, init_fn, shape, **kwargs, dtype=state.dtype)
    else:
      return param(self, name, init_fn, shape, dtype, **kwargs)

  nn.Module.param = decorated


def _should_replace_dtype(
    *,
    module: nn.Module,
    stack: edc.ContextStack[_DTypeState],
) -> bool:
  """Whether or not the dtype should be replaced."""
  if not module.is_initializing() or not stack:
    return False
  last_state = stack[-1]
  # If `None` is provided, use the default dtype
  if last_state.dtype is None:
    return False

  # Eventually filter out some modules
  if last_state.exclude is not None:
    path = '.'.join(module.scope.path)
    # Hack so matching `xxx` do not match `.xxx_yyy.`
    path = f'.{path}.'
    for p in last_state.exclude:
      if f'.{p}.' in path:
        return False

  return True


_mock_flax_module_param()
