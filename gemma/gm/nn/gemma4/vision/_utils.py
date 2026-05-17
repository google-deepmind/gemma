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

"""Utility functions for Gemma4 vision models."""

import jax
import jax.numpy as jnp
from kauldron.ktyping import Bool, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@typechecked
def avg_pool_by_positions(
    x: Float['B L D'],
    *,
    positions_xy: Int['B L 2'],
    length: int,
) -> tuple[Float['B l D'], Bool['B l']]:
  """2D spatial pooling according to patch positions.

  Pools the input tokens by averaging patches within a k×k grid, where
  k is determined by the ratio between input and output lengths.

  Args:
    x: Input embeddings [batch, seq_len, dim].
    positions_xy: Patch positions as (x, y) coordinates [batch, seq_len, 2].
    length: Target output sequence length.

  Returns:
    Tuple of:
      - Pooled embeddings [batch, length, dim]
      - Mask indicating valid (non-padded) tokens [batch, length]
  """
  k = int((x.shape[1] // length) ** 0.5)
  assert k * k * length == x.shape[1], f'Cannot pool {x.shape=} to {length=}'

  max_x = positions_xy[..., 0].max(axis=-1, keepdims=True) + 1
  kernel_idxs = jnp.floor_divide(positions_xy, k)
  # kernel_idxs is a `B, L` array which indexes which of the `l` output tokens
  # each of the`L` input tokens contributes to, corresponding to a kxk grid.
  flat_kernel_idx = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
  # a `B, L, l` array. Each of the L input tokens makes a weighted contribution
  # to each of the `l` output tokens.
  weights = jax.nn.one_hot(flat_kernel_idx, length) / k**2
  output = jnp.einsum('bLl,bLd->bld', weights, x)
  mask = jnp.logical_not((weights == 0).all(axis=1))
  return output, mask
