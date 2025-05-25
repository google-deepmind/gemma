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
import jax
import jax.numpy as jnp
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


def test_nucleus_sampling():
  sampling = gm.text.NucleusSampling(p=0.9)
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 5
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens = sampling.get_next_tokens(logits, rng)
  assert tokens.shape == (batch_size,)


def test_nucleus_sampling_p_1_includes_all_tokens():
  # With p=1.0, nucleus sampling should consider all tokens
  sampling = gm.text.NucleusSampling(p=1.0)
  random_sampling = gm.text.RandomSampling()
  rng = jax.random.PRNGKey(42)
  batch_size = 2
  vocab_size = 10
  
  # Create logits where all tokens have equal probability
  logits = jnp.ones((batch_size, vocab_size))
  
  # Generate many samples to verify all tokens can be selected
  num_samples = 500
  tokens_nucleus = []
  tokens_random = []
  
  for i in range(num_samples):
    rng_i = jax.random.fold_in(rng, i)
    nucleus_result = sampling.get_next_tokens(logits, rng_i)
    random_result = random_sampling.get_next_tokens(logits, rng_i)
    tokens_nucleus.extend(nucleus_result.tolist())
    tokens_random.extend(random_result.tolist())
  
  # Both should cover a similar range of tokens
  unique_nucleus = set(tokens_nucleus)
  unique_random = set(tokens_random)
  
  # With equal probabilities and p=1.0, nucleus should behave similarly to random
  assert len(unique_nucleus) >= vocab_size // 2  # Should see most tokens


def test_nucleus_sampling_small_p_limits_tokens():
  # With very small p, nucleus sampling should be very selective
  sampling = gm.text.NucleusSampling(p=0.1)
  rng = jax.random.PRNGKey(42)
  batch_size = 2
  vocab_size = 10
  
  # Create skewed logits where one token is much more likely
  logits = jnp.array([[10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      [10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
  
  # Generate many samples
  num_samples = 50
  tokens = []
  
  for i in range(num_samples):
    rng_i = jax.random.fold_in(rng, i)
    result = sampling.get_next_tokens(logits, rng_i)
    tokens.extend(result.tolist())
  
  # With small p and skewed distribution, should mostly sample the most likely token
  unique_tokens = set(tokens)
  most_frequent_token = max(unique_tokens, key=lambda x: tokens.count(x))
  assert most_frequent_token == 0  # The most likely token (index 0)

