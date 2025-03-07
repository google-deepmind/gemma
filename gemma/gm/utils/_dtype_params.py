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


@edc.dataclass
@dataclasses.dataclass
class _Context:
  """Context for the dtype stack."""

  dtypes: edc.ContextVar[list[_DType | None]] = dataclasses.field(
      default_factory=list
  )


_context = _Context()


@contextlib.contextmanager
def initialize_param_with_dtype(dtype: _DType | None) -> Iterator[None]:
  """Set the params dtype to the given value.

  Inside the contextmanager, `self.param()` will use the given dtype.

  Args:
    dtype: The dtype to use.

  Yields:
    None
  """
  try:
    _context.dtypes.append(dtype)
    yield
  finally:
    _context.dtypes.pop()


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
    # TODO(epot): Check this do not break LoRA
    if (
        self.is_initializing()
        and _context.dtypes
        # LoRA modules provide the dtype as kwargs
        and 'dtype' not in kwargs
        # If `None` is provided, use the default dtype
        and _context.dtypes[-1] is not None
    ):
      del dtype  # The dtype is overwritten by the contextmanager
      return param(
          self, name, init_fn, shape, **kwargs, dtype=_context.dtypes[-1]
      )
    else:
      return param(self, name, init_fn, shape, dtype, **kwargs)

  nn.Module.param = decorated


_mock_flax_module_param()
