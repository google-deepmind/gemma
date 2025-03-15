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

"""Tests for transformer layers."""

from absl.testing import absltest
from absl.testing import parameterized
from gemma import layers
import jax
import jax.numpy as jnp
import numpy as np


class EinsumTest(parameterized.TestCase):

    @parameterized.parameters(
        ((1, 4), (3, 2, 4, 3), 'TD,SNDH->STNH', (3, 1, 2, 3)),
        ((1, 2, 4), (2, 4, 8), 'ANH,NHD->AD', (1, 8)),
    )
    def test_einsum(self, inputs_shape, params_shape, eqn, expected_shape):
        einsum = layers.Einsum(params_shape)
        output = einsum.apply(
            {'params': {'w': jnp.ones(params_shape)}},
            eqn,
            jnp.ones(inputs_shape),
        )
        self.assertEqual(output.shape, expected_shape, f"Expected {expected_shape}, got {output.shape}")

    @parameterized.parameters(([0.1, 0.2], [0.6324429, 1.2648858]))
    def test_rmsnorm(self, x, expected):
        x = jnp.array([x])
        rmsnorm = layers.RMSNorm()
        params = rmsnorm.init(jax.random.PRNGKey(0), x)
        output = rmsnorm.apply(params, x)
        np.testing.assert_allclose(output, jnp.array([expected]), rtol=1e-6)


if __name__ == '__main__':
    absltest.main()
