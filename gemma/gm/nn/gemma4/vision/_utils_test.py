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

"""Tests for vision utility functions."""

from gemma.gm.nn.gemma4.vision import _utils
import jax
import jax.numpy as jnp


def test_avg_pool_by_positions():
  """Test spatial pooling by positions."""
  batch, seq_len, dim = 2, 16, 8
  length = 4

  x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, dim))
  # Create 4x4 grid positions (x, y)
  positions_xy = jnp.array([
      [[i % length, i // length] for i in range(seq_len)] for _ in range(batch)
  ])

  pooled, mask = _utils.avg_pool_by_positions(
      x, positions_xy=positions_xy, length=length
  )

  # Check shapes
  assert pooled.shape == (batch, length, dim)
  assert mask.shape == (batch, length)

  # Check that mask is all True (no padding)
  assert jnp.all(mask)


def test_avg_pool_by_positions_variable_aspect():
  """Test spatial pooling with variable aspect ratio."""
  batch, seq_len, dim = 2, 24, 8
  length = 6

  x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, dim))
  # Create 6x4 grid positions
  positions_xy = jnp.array([
      [[i % length, i // length] for i in range(seq_len)] for _ in range(batch)
  ])

  pooled, mask = _utils.avg_pool_by_positions(
      x, positions_xy=positions_xy, length=length
  )

  assert pooled.shape == (batch, length, dim)
  assert mask.shape == (batch, length)
