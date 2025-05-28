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

from typing import Any

from flax import linen as nn
from gemma import peft
import jax
import jax.numpy as jnp


class WrapperModule(nn.Module):
  wrapped: nn.Dense
  share_scope: bool = True

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': {'wrapped': params}}`).
    if self.share_scope and self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    # Create an extra param.
    self.param('extra_param', lambda _: jnp.zeros(()))

    # Wrapped the output, using the features as key to ensure we captured the
    # correct module.
    return {f'wrapped_{self.wrapped.features}': self.wrapped(*args, **kwargs)}


class MyModule(nn.Module):

  @nn.compact
  def __call__(self, x):
    # Call a same model twice
    model = nn.Dense(1)
    y0 = nn.Dropout(0.5, deterministic=False)(x)
    y1 = model(x)
    y2 = model(x)

    # Call another model
    y3 = nn.Dense(2)(x)
    return {
        'y0': y0,
        'y1': y1,
        'y2': y2,
        'y3': y3,
    }


def test_module():
  denses_features_replaces = []

  def _replace_module(module):
    if isinstance(module, nn.Dense):
      denses_features_replaces.append(module.features)
      return WrapperModule(wrapped=module)
    else:
      return module

  model = MyModule()
  with peft.ModuleInterceptor(_replace_module):
    out, params = model.init_with_output(jax.random.key(0), jnp.zeros((3,)))

  # We only care about the structure, not the values.
  out = jax.tree.map(lambda x: None, out)
  params = jax.tree.map(lambda x: None, params)

  assert denses_features_replaces == [1, 2]
  assert out == {
      'y0': None,
      'y1': {'wrapped_1': None},
      'y2': {'wrapped_1': None},
      'y3': {'wrapped_2': None},
  }
  assert params == {
      'params': {
          'Dense_0': {
              'extra_param': None,
              'bias': None,
              'kernel': None,
          },
          'Dense_1': {
              'extra_param': None,
              'bias': None,
              'kernel': None,
          },
      },
  }


def test_module_non_share_scope():
  denses_features_replaces = []

  def _replace_module(module):
    if isinstance(module, nn.Dense):
      denses_features_replaces.append(module.features)
      return WrapperModule(wrapped=module, share_scope=False)
    else:
      return module

  model = MyModule()
  with peft.ModuleInterceptor(_replace_module):
    out, params = model.init_with_output(jax.random.key(0), jnp.zeros((3,)))

  # We only care about the structure, not the values.
  out = jax.tree.map(lambda x: None, out)
  params = jax.tree.map(lambda x: None, params)

  assert denses_features_replaces == [1, 2]
  assert out == {
      'y0': None,
      'y1': {'wrapped_1': None},
      'y2': {'wrapped_1': None},
      'y3': {'wrapped_2': None},
  }
  # TODO(epot): Is it possible to have the `Dense_0` to be nested inside the
  # `WrapperModule_0` (By changing the scope or copying the module) ? Is it
  # desirable ?
  assert params == {
      'params': {
          'Dense_0': {
              'WrapperModule_0': {'extra_param': None},
              'bias': None,
              'kernel': None,
          },
          'Dense_1': {
              'WrapperModule_0': {'extra_param': None},
              'bias': None,
              'kernel': None,
          },
      },
  }


# TODO(epot): Test a nested replace (module replaced also has sub-modules which
# should be replaced)

# TODO(epot): Test nested `ModuleInterceptor()` (e.g. try to wrap twice the
# `nn.Dense()`)
