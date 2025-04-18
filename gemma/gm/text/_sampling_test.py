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

from gemma import gm
import jax.numpy
import numpy as np  # pylint: disable=unused-import


def test_greedy_sampling():
  sampling = gm.text.Greedy()
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 3
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens = sampling.get_next_tokens(logits, rng)
  assert tokens.shape == (batch_size,)


def test_random_sampling():
  sampling = gm.text.RandomSampling()
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 3
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens = sampling.get_next_tokens(logits, rng)
  assert tokens.shape == (batch_size,)
