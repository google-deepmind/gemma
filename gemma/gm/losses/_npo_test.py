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

"""Tests for NPO loss."""

from gemma.gm.losses import _npo
import jax
import jax.numpy as jnp
import numpy as np


def test_npo_get_logprobs_for_target():
  """Test that _get_logprobs_for_target computes correct masked logprobs."""
  # logits: [B=1, L=3, V=4]
  logits = jnp.array([[
      [2.0, 0.0, 0.0, 0.0],
      [0.0, 3.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
  ]])

  targets = jnp.array([[0, 1, 2]])
  sequence_mask = jnp.array([[1, 1, 0]], dtype=jnp.bool_)

  logprobs = _npo._get_logprobs_for_target(
      logits=logits,
      targets=targets,
      sequence_mask=sequence_mask,
  )

  assert logprobs.shape == (1,)

  log_softmax_0 = jax.nn.log_softmax(logits[0, 0])
  log_softmax_1 = jax.nn.log_softmax(logits[0, 1])
  expected = log_softmax_0[0] + log_softmax_1[1]
  np.testing.assert_allclose(logprobs[0], expected, atol=1e-5)


def test_npo_loss_output_shape():
  """NPO loss should return shape [B, 1]."""
  batch_size = 2
  seq_len = 4
  vocab_size = 8

  tokens = jax.random.randint(
      jax.random.PRNGKey(0), (batch_size, seq_len, 1), 0, vocab_size
  )
  sequence_mask = jnp.ones((batch_size, seq_len, 1), dtype=jnp.bool_)
  policy_logits = jax.random.normal(
      jax.random.PRNGKey(1), (batch_size, seq_len, vocab_size)
  )
  anchor_logits = jax.random.normal(
      jax.random.PRNGKey(2), (batch_size, seq_len, vocab_size)
  )

  loss_fn = _npo.NpoLoss(
      tokens='tokens',
      sequence_mask='mask',
      policy_logits='policy',
      anchor_logits='anchor',
  )
  loss = loss_fn.get_values(
      tokens=tokens,
      sequence_mask=sequence_mask,
      policy_logits=policy_logits,
      anchor_logits=anchor_logits,
  )
  assert loss.shape == (batch_size, 1)


def test_npo_loss_penalizes_high_policy_prob():
  """NPO loss should be higher when policy assigns more prob than anchor."""
  seq_len = 2
  vocab_size = 4

  # Undesired tokens.
  tokens = jnp.array([[[0], [0]]])
  sequence_mask = jnp.ones((1, seq_len, 1), dtype=jnp.bool_)

  # Anchor: uniform.
  anchor_logits = jnp.zeros((1, seq_len, vocab_size))

  # Policy A: assigns high probability to undesired tokens.
  policy_high = jnp.zeros((1, seq_len, vocab_size))
  policy_high = policy_high.at[0, :, 0].set(5.0)

  # Policy B: assigns low probability to undesired tokens.
  policy_low = jnp.zeros((1, seq_len, vocab_size))
  policy_low = policy_low.at[0, :, 0].set(-5.0)

  loss_fn = _npo.NpoLoss(
      tau=1.0,
      tokens='tokens',
      sequence_mask='mask',
      policy_logits='policy',
      anchor_logits='anchor',
  )

  loss_high = loss_fn.get_values(
      tokens=tokens,
      sequence_mask=sequence_mask,
      policy_logits=policy_high,
      anchor_logits=anchor_logits,
  )
  loss_low = loss_fn.get_values(
      tokens=tokens,
      sequence_mask=sequence_mask,
      policy_logits=policy_low,
      anchor_logits=anchor_logits,
  )

  # NPO penalizes policy > anchor, so loss should be higher for policy_high.
  assert loss_high[0, 0] > loss_low[0, 0]


def test_npo_loss_zero_when_policy_matches_anchor():
  """When policy matches anchor, delta is 0 and loss equals -log_sigmoid(0)."""
  tokens = jnp.array([[[0], [1]]])
  sequence_mask = jnp.ones((1, 2, 1), dtype=jnp.bool_)
  logits = jnp.zeros((1, 2, 4))

  loss_fn = _npo.NpoLoss(
      tau=1.0,
      tokens='tokens',
      sequence_mask='mask',
      policy_logits='policy',
      anchor_logits='anchor',
  )
  loss = loss_fn.get_values(
      tokens=tokens,
      sequence_mask=sequence_mask,
      policy_logits=logits,
      anchor_logits=logits,
  )

  # When policy == anchor, delta = 0, so loss = -log_sigmoid(0) = log(2).
  expected = -jax.nn.log_sigmoid(jnp.array(0.0))
  np.testing.assert_allclose(loss[0, 0], expected, atol=1e-5)
