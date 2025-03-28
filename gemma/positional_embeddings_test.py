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

"""Tests for the positional embeddings utilities."""

from absl.testing import absltest
from absl.testing import parameterized
from gemma import positional_embeddings
import jax.numpy as jnp
import numpy as np


class PositionalEmbeddingsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          input_embedding_shape=(2, 1, 2, 4),
          positions=[[1], [0]],
          max_wavelength=100,
          expected=[
              [[
                  [-0.30116868, 0.89517075, 1.3817732, 1.0948375],
                  [-0.30116868, 0.89517075, 1.3817732, 1.0948375],
              ]],
              [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
          ],
      )
  )
  def test_rope_positional_embeddings(
      self, input_embedding_shape, positions, max_wavelength, expected
  ):
    outputs = positional_embeddings.apply_rope(
        jnp.ones(input_embedding_shape),
        jnp.array(positions),
        base_frequency=max_wavelength,
    )
    np.testing.assert_array_almost_equal(outputs, jnp.array(expected))


if __name__ == '__main__':
  absltest.main()
