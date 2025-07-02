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

"""Tests for transformer layers."""

from flax import linen as nn
from gemma.gm.nn.gemma3n import _layers
import jax
import jax.numpy as jnp
import numpy as np
import pytest

_ZERO_INIT = nn.initializers.zeros_init()
_ONES_INIT = nn.initializers.ones_init()


@pytest.mark.parametrize(
    'inputs_shape, params_shape, eqn, expected_shape',
    [
        ((1, 4), (3, 2, 4, 3), 'TD,SNDH->STNH', (3, 1, 2, 3)),
        ((1, 2, 4), (2, 4, 8), 'ANH,NHD->AD', (1, 8)),
    ],
)
def test_einsum(inputs_shape, params_shape, eqn, expected_shape):
  einsum = _layers.Einsum(params_shape)
  output = einsum.apply(
      {'params': {'w': jnp.ones(params_shape)}},
      eqn,
      jnp.ones(inputs_shape),
  )
  assert output.shape == expected_shape


@pytest.mark.parametrize(
    'x, expected',
    [([0.1, 0.2], [0.6324429, 1.2648858])],
)
def test_rmsnorm(x, expected):
  x = jnp.array([x])
  rmsnorm = _layers.RMSNorm()
  params = rmsnorm.init(jax.random.PRNGKey(0), x)
  output = rmsnorm.apply(params, x)
  np.testing.assert_array_equal(output, jnp.array([expected]))


@pytest.mark.parametrize(
    'x, expected,with_scale,scale_init',
    [
        # This is the default case.
        ([0.1, 0.2], [0.6324429, 1.2648858], True, _ZERO_INIT),
        # In this case, the output is simply scaled by (1 + scale).
        ([0.1, 0.2], [1.2648858, 2.5297716], True, _ONES_INIT),
        # When with_scale is False, the output is not scaled.
        ([0.1, 0.2], [0.6324429, 1.2648858], False, _ZERO_INIT),
        ([0.1, 0.2], [0.6324429, 1.2648858], False, _ONES_INIT),
    ],
)
def test_rmsnorm_with_scale(x, expected, with_scale, scale_init):
  x = jnp.array([x])
  rmsnorm = _layers.RMSNorm(with_scale=with_scale, scale_init=scale_init)
  params = rmsnorm.init(jax.random.PRNGKey(0), x)
  output = rmsnorm.apply(params, x)
  np.testing.assert_array_equal(output, jnp.array([expected]))
