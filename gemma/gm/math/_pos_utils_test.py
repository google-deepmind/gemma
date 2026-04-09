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

"""Tests for position utilities."""

from gemma.gm.math import _pos_utils
import jax.numpy as jnp
import numpy as np


class TestBuildPositionsFromMask:
  """Tests for build_positions_from_mask."""

  def test_all_valid_tokens(self):
    """All tokens valid => positions should be [0, 1, 2, ...]."""
    mask = jnp.array([True, True, True, True])
    positions = _pos_utils.build_positions_from_mask(mask)
    np.testing.assert_array_equal(positions, [0, 1, 2, 3])

  def test_leading_padding(self):
    """Padding at the start: valid tokens should still be 0-indexed."""
    mask = jnp.array([False, False, True, True, True])
    positions = _pos_utils.build_positions_from_mask(mask)
    # First valid token gets position 0, then 1, 2.
    expected = [0, 0, 0, 1, 2]
    np.testing.assert_array_equal(positions, expected)

  def test_all_padding(self):
    """All padded => cumsum is 0 everywhere, positions stay 0."""
    mask = jnp.array([False, False, False])
    positions = _pos_utils.build_positions_from_mask(mask)
    np.testing.assert_array_equal(positions, [0, 0, 0])

  def test_single_token(self):
    """Single valid token should get position 0."""
    mask = jnp.array([True])
    positions = _pos_utils.build_positions_from_mask(mask)
    np.testing.assert_array_equal(positions, [0])

  def test_batched_input(self):
    """Should work with batched (2D) input masks."""
    mask = jnp.array([
        [True, True, True, False],
        [False, True, True, True],
    ])
    positions = _pos_utils.build_positions_from_mask(mask)
    expected = jnp.array([
        [0, 1, 2, 2],
        [0, 0, 1, 2],
    ])
    np.testing.assert_array_equal(positions, expected)

  def test_output_shape(self):
    """Output shape should match input shape."""
    mask = jnp.array([True, True, False])
    positions = _pos_utils.build_positions_from_mask(mask)
    assert positions.shape == (3,)
