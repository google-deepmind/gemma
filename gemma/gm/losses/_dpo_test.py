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

"""Tests for DPO loss."""

from gemma.gm.losses import _dpo
import jax
import jax.numpy as jnp
import numpy as np


def _make_simple_logits(target_token, vocab_size, high_logit=5.0):
  """Make logits where `target_token` has a high logit value."""
  logits = jnp.zeros(vocab_size)
  logits = logits.at[target_token].set(high_logit)
  return logits


def test_get_logprobs_for_target():
  """Test that _get_logprobs_for_target computes correct masked logprobs."""
  vocab_size = 4

  # logits: [B=1, N=2, L=3, V=4]
  logits = jnp.array([[
      [[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
      [[0.0, 0.0, 0.0, 2.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
  ]])

  targets = jnp.array([[[0, 1, 2], [3, 0, 0]]])
  sequence_mask = jnp.array([[[1, 1, 0], [1, 1, 0]]], dtype=jnp.bool_)

  logprobs = _dpo._get_logprobs_for_target(
      logits=logits,
      targets=targets,
      sequence_mask=sequence_mask,
  )

  assert logprobs.shape == (1, 2)

  log_softmax_0 = jax.nn.log_softmax(logits[0, 0, 0])
  log_softmax_1 = jax.nn.log_softmax(logits[0, 0, 1])
  expected_0 = log_softmax_0[0] + log_softmax_1[1]
  np.testing.assert_allclose(logprobs[0, 0], expected_0, atol=1e-5)


def test_get_logprobs_masked_positions_ignored():
  """Masked positions should not contribute to the logprob sum."""
  logits = jnp.ones((1, 1, 3, 4)) * 10.0
  targets = jnp.array([[[0, 1, 2]]])
  mask_all = jnp.array([[[1, 1, 1]]], dtype=jnp.bool_)
  mask_first_only = jnp.array([[[1, 0, 0]]], dtype=jnp.bool_)

  logprobs_all = _dpo._get_logprobs_for_target(
      logits=logits, targets=targets, sequence_mask=mask_all
  )
  logprobs_first = _dpo._get_logprobs_for_target(
      logits=logits, targets=targets, sequence_mask=mask_first_only
  )

  assert logprobs_all[0, 0] != logprobs_first[0, 0]


def test_dpo_loss_output_shape():
  """DPO loss should return shape [B, 1]."""
  batch_size = 2
  seq_len = 4
  vocab_size = 8

  tokens = jax.random.randint(
      jax.random.PRNGKey(0), (batch_size, 2, seq_len), 0, vocab_size
  )
  sequence_mask = jnp.ones((batch_size, 2, seq_len), dtype=jnp.bool_)
  policy_logits = jax.random.normal(
      jax.random.PRNGKey(1), (batch_size, 2, seq_len, vocab_size)
  )
  anchor_logits = jax.random.normal(
      jax.random.PRNGKey(2), (batch_size, 2, seq_len, vocab_size)
  )

  loss_fn = _dpo.DpoLoss(
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


def test_dpo_loss_prefers_chosen():
  """Loss should be lower when policy increases chosen response probability."""
  seq_len = 2
  vocab_size = 4

  # Preferred response targets token 0, dispreferred targets token 1.
  tokens = jnp.array([[[0, 0], [1, 1]]])
  sequence_mask = jnp.ones((1, 2, seq_len), dtype=jnp.bool_)

  # Anchor: uniform logits for both.
  anchor_logits = jnp.zeros((1, 2, seq_len, vocab_size))

  # Policy A: strongly prefers chosen (token 0 high for response 0).
  policy_a = jnp.zeros((1, 2, seq_len, vocab_size))
  policy_a = policy_a.at[0, 0, :, 0].set(5.0)
  policy_a = policy_a.at[0, 1, :, 1].set(-5.0)

  # Policy B: strongly prefers dispreferred (opposite).
  policy_b = jnp.zeros((1, 2, seq_len, vocab_size))
  policy_b = policy_b.at[0, 0, :, 0].set(-5.0)
  policy_b = policy_b.at[0, 1, :, 1].set(5.0)

  loss_fn = _dpo.DpoLoss(
      tau=0.1,
      tokens='tokens',
      sequence_mask='mask',
      policy_logits='policy',
      anchor_logits='anchor',
  )

  loss_a = loss_fn.get_values(
      tokens=tokens,
      sequence_mask=sequence_mask,
      policy_logits=policy_a,
      anchor_logits=anchor_logits,
  )
  loss_b = loss_fn.get_values(
      tokens=tokens,
      sequence_mask=sequence_mask,
      policy_logits=policy_b,
      anchor_logits=anchor_logits,
  )

  assert loss_a[0, 0] < loss_b[0, 0]


def test_dpo_loss_label_smoothing():
  """With label_smoothing=0.5, loss should be symmetric."""
  tokens = jnp.array([[[0, 0], [1, 1]]])
  sequence_mask = jnp.ones((1, 2, 2), dtype=jnp.bool_)
  policy_logits = jax.random.normal(
      jax.random.PRNGKey(0), (1, 2, 2, 4)
  )
  anchor_logits = jnp.zeros((1, 2, 2, 4))

  loss_fn_smooth = _dpo.DpoLoss(
      tau=0.1,
      label_smoothing=0.5,
      tokens='tokens',
      sequence_mask='mask',
      policy_logits='policy',
      anchor_logits='anchor',
  )

  loss = loss_fn_smooth.get_values(
      tokens=tokens,
      sequence_mask=sequence_mask,
      policy_logits=policy_logits,
      anchor_logits=anchor_logits,
  )

  # With label_smoothing=0.5:
  # loss = -(log_sigmoid(delta) * 0.5 + log_sigmoid(-delta) * 0.5)
  # This is symmetric in delta, so swapping preferred/dispreferred
  # should give the same loss.
  tokens_swapped = jnp.array([[[1, 1], [0, 0]]])
  policy_swapped = policy_logits[:, ::-1, :, :]

  loss_swapped = loss_fn_smooth.get_values(
      tokens=tokens_swapped,
      sequence_mask=sequence_mask,
      policy_logits=policy_swapped,
      anchor_logits=anchor_logits[:, ::-1, :, :],
  )

  np.testing.assert_allclose(loss[0, 0], loss_swapped[0, 0], atol=1e-5)
