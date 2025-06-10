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

from gemma import gm
import jax.numpy
import numpy as np


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


def test_topk_sampling():
  sampling = gm.text.TopkSampling(k=3)
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 5
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens = sampling.get_next_tokens(logits, rng)
  assert tokens.shape == (batch_size,)


def test_topp_sampling():
  sampling = gm.text.TopPSampling(p=0.9)
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 5
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens = sampling.get_next_tokens(logits, rng)
  assert tokens.shape == (batch_size,)


def test_topp_sampling_with_skewed_logits():
  sampling = gm.text.TopPSampling(p=0.6)
  rng = jax.random.PRNGKey(0)

  neg_inf = jax.numpy.finfo(jax.numpy.float32).min
  logits = jax.numpy.array([
      [1.0, neg_inf, neg_inf, neg_inf, neg_inf],
  ])
  tokens = sampling.get_next_tokens(logits, rng)
  assert np.allclose(tokens, [0])


def test_top1_sampling_matches_greedy_sampling():
  greedy = gm.text.Greedy()
  top1_sampling = gm.text.TopkSampling(k=1)
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 5
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens_greedy = greedy.get_next_tokens(logits, rng)
  tokens_top1 = top1_sampling.get_next_tokens(logits, rng)
  np.testing.assert_array_equal(tokens_greedy, tokens_top1)

