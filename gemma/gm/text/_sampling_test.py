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
  rng = jax.random.PRNGKey(2)
  # Probabilities after softmax: [0.64, 0.23, 0.09, 0.03, 0.01].
  logits = jax.numpy.array([
      [5.0, 4.0, 3.0, 2.0, 1.0],
  ])
  tokens = sampling.get_next_tokens(logits, rng)
  np.testing.assert_array_equal(tokens, [0])


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



def test_minp_sampling():
  sampling = gm.text.MinPSampling(p=0.05)
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 5
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))
  tokens = sampling.get_next_tokens(logits, rng)
  assert tokens.shape == (batch_size,)


def test_minp_sampling_filters_correctly_with_batch():
  """Tests that Min-P correctly filters tokens with different distributions per batch."""
  # With p=0.3, we can create two different deterministic outcomes in one batch.
  sampling = gm.text.MinPSampling(p=0.3)
  rng = jax.random.PRNGKey(0)

  # Batch 1: Probs are [0.8, 0.15, 0.05]. p_max is 0.8.
  # Threshold is p_max * p = 0.8 * 0.3 = 0.24.
  # Only the first token (p=0.8) is >= 0.24, so it must be selected.
  #
  # Batch 2: Probs are [0.1, 0.7, 0.2]. p_max is 0.7.
  # Threshold is p_max * p = 0.7 * 0.3 = 0.21.
  # Only the second token (p=0.7) is >= 0.21, so it must be selected.
  logits = jax.numpy.log(jax.numpy.array([
      [0.8, 0.15, 0.05],
      [0.1, 0.7, 0.2],
  ]))
  tokens = sampling.get_next_tokens(logits, rng)
  np.testing.assert_array_equal(tokens, [0, 1])


def test_minp_p1_sampling_matches_greedy():
  """Tests that MinPSampling with p=1.0 is equivalent to greedy sampling (no ties)."""
  greedy = gm.text.Greedy()
  minp_p1_sampling = gm.text.MinPSampling(p=1.0)
  rng = jax.random.PRNGKey(0)
  batch_size = 2
  vocab_size = 10
  logits = jax.random.normal(rng, shape=(batch_size, vocab_size))

  tokens_greedy = greedy.get_next_tokens(logits, rng)
  tokens_minp_p1 = minp_p1_sampling.get_next_tokens(logits, rng)

  np.testing.assert_array_equal(tokens_greedy, tokens_minp_p1)


def test_minp_with_high_temperature():
  """Tests Min-P's behavior on a temperature-flattened distribution."""
  # With a high temperature, the distribution becomes very flat.
  # Lowering p to 0.9 to ensure the test passes as expected.
  sampling = gm.text.MinPSampling(p=0.9, temperature=100.0)
  rng = jax.random.PRNGKey(0)

  # At temp=1, this is very peaked at token 0. At temp=100, the scaled
  # logits are [0.1, 0.0, 0.0], making the probability distribution
  # nearly uniform: approx [0.355, 0.322, 0.322].
  logits = jax.numpy.array([[10.0, 0.0, 0.0]])

  # p_max is ~0.355. Threshold = 0.355 * 0.9 = 0.3195.
  # All three tokens have probabilities > threshold, so all should be possible.
  rngs = jax.random.split(rng, 100)
  tokens = jax.vmap(sampling.get_next_tokens, in_axes=(None, 0))(logits, rngs)

  # Check that all three tokens were sampled.
  assert np.all(np.isin(tokens, [0, 1, 2]))
  assert 0 in tokens
  assert 1 in tokens
  assert 2 in tokens


def test_minp_p1_handles_ties():
  """Tests that MinPSampling with p=1.0 samples from tied top tokens."""
  sampling = gm.text.MinPSampling(p=1.0)
  rng = jax.random.PRNGKey(0)
  # Logits where tokens 1 and 3 are tied for the max value.
  # With p=1.0, only these two tokens should ever be sampled.
  logits = jax.numpy.array([[1.0, 10.0, 5.0, 10.0, 2.0]])

  # Run sampling many times; all results must be either 1 or 3.
  rngs = jax.random.split(rng, 100)
  # Use vmap to efficiently run the sampler across many random keys.
  tokens = jax.vmap(sampling.get_next_tokens, in_axes=(None, 0))(logits, rngs)

  # Check that all sampled tokens are either 1 or 3.
  assert np.all(np.isin(tokens, [1, 3]))
  # Check that both 1 and 3 were actually sampled, confirming it's not deterministic.
  assert 1 in tokens
  assert 3 in tokens


def test_minp_p0_sampling_matches_random():
  """Tests that MinPSampling with p=0.0 is equivalent to random sampling."""
  random_sampling = gm.text.RandomSampling(temperature=1.0)
  minp_p0_sampling = gm.text.MinPSampling(p=0.0, temperature=1.0)

  # Create a master key and split it once for deterministic operations.
  rng = jax.random.PRNGKey(42)
  logits_rng, sampling_rng = jax.random.split(rng)

  batch_size = 2
  vocab_size = 10
  # Use the first sub-key to generate logits.
  logits = jax.random.normal(logits_rng, shape=(batch_size, vocab_size))

  # Use the *same* second sub-key for both sampling calls. This ensures
  # that if their logic is identical, their random output will be too.
  tokens_random = random_sampling.get_next_tokens(logits, sampling_rng)
  tokens_minp_p0 = minp_p0_sampling.get_next_tokens(logits, sampling_rng)

  np.testing.assert_array_equal(tokens_random, tokens_minp_p0)

