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

"""Tests for Gemma4 vision utility functions."""

from gemma.gm.nn.gemma4.vision import _images
import jax.numpy as jnp
import numpy as np


def test_factorized_posemb():
  batch_size, seq_len, dim = 2, 4, 6
  pos_emb_size = 5
  posemb = jnp.arange(pos_emb_size * 2 * dim, dtype=jnp.float32).reshape(
      (pos_emb_size, 2, dim)
  )
  positions_xy = jnp.array([
      [[0, 0], [1, 1], [2, 2], [3, 3]],
      [[0, 4], [6, 0], [-1, -1], [-10, -10]],
  ])

  # Manual calculation for the first element [0, 0]
  # x=0, y=0
  pe_x_00 = posemb[0, 0, :]
  pe_y_00 = posemb[0, 1, :]
  expected_00 = pe_x_00 + pe_y_00

  # Manual calculation for the second element [1, 1]
  # x=1, y=1
  pe_x_11 = posemb[1, 0, :]
  pe_y_11 = posemb[1, 1, :]
  expected_11 = pe_x_11 + pe_y_11

  result = _images.factorized_posemb(posemb, positions_xy)

  assert result.shape == (batch_size, seq_len, dim)
  np.testing.assert_allclose(result[0, 0, :], expected_00)
  np.testing.assert_allclose(result[0, 1, :], expected_11)
  # Check for zeroed padding values - second item, third element [-1, -1]
  assert jnp.sum(result[1, 2, :]) == 0
  # Check NaN for OOB positive value - second item, second element [6, 0]
  assert jnp.all(jnp.isnan(result[1, 1, :]))
  # Check NaN for OOB negative value - second item, last element [-10, -10]
  assert jnp.all(jnp.isnan(result[1, -1, :]))


def test_patchify():
  images = jnp.arange(2 * 32 * 32 * 3, dtype=jnp.float32).reshape(
      (2, 32, 32, 3)
  )
  patch_size = 16
  patches, positions_xy = _images.patchify(images, patch_size)

  assert patches.shape == (2, 4, 16 * 16 * 3)
  assert positions_xy.shape == (2, 4, 2)

  # Expected positions: (x, y)
  # (0,0), (1,0), (0,1), (1,1)
  expected_positions = jnp.array([
      [[0, 0], [1, 0], [0, 1], [1, 1]],
      [[0, 0], [1, 0], [0, 1], [1, 1]],
  ])
  np.testing.assert_array_equal(positions_xy, expected_positions)

  # Check patch content for the first patch [0, 0]
  expected_patch_00 = images[0, :16, :16, :].reshape(-1)
  np.testing.assert_array_equal(patches[0, 0, :], expected_patch_00)

  # Check patch content for the second patch [1, 0]
  expected_patch_10 = images[0, :16, 16:, :].reshape(-1)
  np.testing.assert_array_equal(patches[0, 1, :], expected_patch_10)
