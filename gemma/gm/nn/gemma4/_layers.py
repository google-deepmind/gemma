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

"""Base layers."""

from flax import linen as nn
import jax
import jax.numpy as jnp


class Einsum(nn.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""

  shape: tuple[int, ...]
  weight_name: str = 'w'
  initializer: nn.initializers.Initializer = nn.initializers.normal()
  dtype: jnp.dtype | None = None
  w_scale: float | None = None

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    w = self.param(
        self.weight_name,
        self.initializer,
        self.shape,
        self.dtype if self.dtype is not None else None,
    )
    # Workaround for behavior with nn.share_scope in parent modules:
    # self.param might return a dict {'w': tensor} instead of the bare tensor.
    # The key appears to be the default class weight_name 'w'.
    if isinstance(w, dict):
      if 'w' in w:
        w = w['w']
    if self.w_scale is not None:
      w *= self.w_scale
    return jnp.einsum(eqn, x, w)


class ClippedEinsum(nn.Module):
  """Einsum with input and output activation clamping."""

  shape: tuple[int, ...]
  weight_name: str = 'w'
  initializer: nn.initializers.Initializer = nn.initializers.normal()
  dtype: jnp.dtype | None = None
  w_scale: float | None = None

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    w = self.param(
        self.weight_name,
        self.initializer,
        self.shape,
        self.dtype if self.dtype is not None else None,
    )
    if isinstance(w, dict):
      if 'w' in w:
        w = w['w']
    if self.w_scale is not None:
      w *= self.w_scale

    inf = float('inf')
    clip_input_min = self.param(
        'clip_input_min', lambda key, shape, dtype=None: jnp.array(-inf), ()
    )
    clip_input_max = self.param(
        'clip_input_max', lambda key, shape, dtype=None: jnp.array(inf), ()
    )
    clip_output_min = self.param(
        'clip_output_min', lambda key, shape, dtype=None: jnp.array(-inf), ()
    )
    clip_output_max = self.param(
        'clip_output_max', lambda key, shape, dtype=None: jnp.array(inf), ()
    )

    x = jnp.clip(x, clip_input_min, clip_input_max)
    x = jnp.einsum(eqn, x, w)
    x = jnp.clip(x, clip_output_min, clip_output_max)
    return x


class RMSNorm(nn.Module):
  """RMSNorm layer."""

  with_scale: bool = True

  @nn.compact
  def __call__(self, x):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

    # Jax.lax.rsqrt is used because it returns different floats than
    # jnp.reciprocal(jnp.sqrt(var + 1e-06))
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

    if self.with_scale:
      scale = self.param(
          'scale',
          nn.initializers.ones,
          (x.shape[-1]),
      )
      # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale
      # is a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
      # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
      scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
      normed_inputs = normed_inputs * scale
    return normed_inputs
