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

"""Interceptor utils."""

from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
from typing import Any, Callable, Iterable

from etils import epy
from flax import linen as nn


class Interceptor(epy.ContextManager, abc.ABC):
  """Base class for interceptors.

  Subclasses can be used as context managers like:

  ```python
  with MyInterceptor():
    y = nn.Dense(10)(x)
  ```
  """

  def __contextmanager__(self) -> Iterable[Any]:
    with nn.intercept_methods(self.interceptor):
      yield

  @abc.abstractmethod
  def interceptor(
      self,
      next_fun,
      args,
      kwargs,
      context: nn.module.InterceptorContext,
  ):
    """Returns the names of the methods to intercept."""
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class ModuleInterceptor(Interceptor):
  """Interceptor that capture all modules to eventually replaces them.

  For each modules, this interceptor call the `replace_module_fn` function
  which returns the module to use instead.

  Example:

  ```python
  def _replace_dense_by_lora(module):
    if isinstance(module, nn.Dense):
      return peft.LoRADense(rank=3, wrapped=module)
    else:
      return module

  # Within the context, the dense layers are replaced by their LoRA version.
  with ModuleInterceptor(_replace_dense_by_lora):
    y = model(x)
  ```
  """

  replace_module_fn: Callable[[nn.Module], nn.Module]

  _: dataclasses.KW_ONLY

  _modules_to_replace: dict[int, _CachedReplace] = dataclasses.field(
      default_factory=dict, init=False
  )
  # TODO(epot): Should be `ContextVar` to supports threads & async.
  _id_to_not_recurse_to_count: dict[int, int] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(int), init=False
  )

  def interceptor(
      self,
      next_fun,
      args,
      kwargs,
      context: nn.module.InterceptorContext,
  ):
    origin_id = id(context.module)

    # Avoid infinite recursion for modules already replaced.
    # Otherwise, wrapped modules would be replaced again when the wrapper is
    # called.
    if self._id_to_not_recurse_to_count[origin_id] > 0:
      return next_fun(*args, **kwargs)

    # The first time we see this module, we replace it.
    if origin_id not in self._modules_to_replace:
      self._modules_to_replace[origin_id] = _CachedReplace(
          original_module=context.module,
          replaced_module=self.replace_module_fn(context.module),
      )

    # Use the replaced module.
    module = self._modules_to_replace[origin_id].replaced_module
    replaced_id = id(module)

    # The module was not replaced, no need to do anything.
    if replaced_id == origin_id:
      return next_fun(*args, **kwargs)

    # To avoid infinite recursion, do not apply the interceptor to
    # nested calls.
    with self._do_not_recurse_in({id(module), id(context.module)}):
      return getattr(module, context.method_name)(*args, **kwargs)

  @contextlib.contextmanager
  def _do_not_recurse_in(self, ids: Iterable[int]):
    for id_ in ids:
      self._id_to_not_recurse_to_count[id_] += 1
    try:
      yield
    finally:
      for id_ in ids:
        self._id_to_not_recurse_to_count[id_] -= 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CachedReplace:
  # Keep a reference alived on the original module, so the
  # `id(original_module)` do not get reused.
  original_module: nn.Module
  replaced_module: nn.Module
