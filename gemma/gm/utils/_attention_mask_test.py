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

from gemma.gm.utils import _attention_mask
import jax.numpy as jnp
import numpy as np


def test_make_causal_bidirectional_attention_mask():

  causal_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  bidirectional_mask = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

  causal_mask = jnp.asarray(causal_mask, dtype=bool)[None, ...]
  bidirectional_mask = jnp.asarray(bidirectional_mask, dtype=bool)[None, ...]

  expected_attention_mask = [
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
  ]
  expected_attention_mask = jnp.asarray(expected_attention_mask, dtype=bool)[
      None, ...
  ]

  out = _attention_mask.make_causal_bidirectional_attention_mask(
      causal_mask=causal_mask,
      bidirectional_mask=bidirectional_mask,
  )

  np.testing.assert_array_equal(out, expected_attention_mask)
