# Copyright 2026 DeepMind Technologies Limited.
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


def _make_one_hot_logits(targets, vocab_size):
  """Creates logits that are high for the target tokens."""
  one_hot = jax.nn.one_hot(targets, vocab_size)
  # Scale so softmax is strongly peaked at the target.
  return one_hot * 10.0 - 5.0


class TestGetLogprobsForTarget:
  """Tests for _get_logprobs_for_target helper."""

  def test_shape(self):
    batch, n, seq_len, vocab = 2, 2, 4, 8
    logits = jnp.ones((batch, n, seq_len, vocab))
    targets = jnp.zeros((batch, n, seq_len), dtype=jnp.int32)
    mask = jnp.ones((batch, n, seq_len), dtype=jnp.bool_)

    result = _dpo._get_logprobs_for_target(
        logits=logits, targets=targets, sequence_mask=mask,
    )
    assert result.shape == (batch, n)

  def test_masked_tokens_ignored(self):
    """Masked positions should not contribute to log-probs."""
    vocab = 4
    # B=1, N=1, L=3
    logits = jnp.zeros((1, 1, 3, vocab))
    targets = jnp.array([[[0, 1, 2]]], dtype=jnp.int32)

    mask_all = jnp.ones((1, 1, 3), dtype=jnp.bool_)
    mask_partial = jnp.array([[[True, False, False]]])

    result_all = _dpo._get_logprobs_for_target(
        logits=logits, targets=targets, sequence_mask=mask_all,
    )
    result_partial = _dpo._get_logprobs_for_target(
        logits=logits, targets=targets, sequence_mask=mask_partial,
    )
    # Partial mask should give a less-negative (higher) value since fewer
    # tokens contribute.
    assert float(result_partial[0, 0]) > float(result_all[0, 0])

  def test_perfect_logits_give_near_zero_logprob(self):
    """When logits strongly favor the target, log-prob should be near 0."""
    vocab = 4
    targets = jnp.array([[[0, 1]]], dtype=jnp.int32)  # B=1, N=1, L=2
    logits = _make_one_hot_logits(targets, vocab)
    mask = jnp.ones((1, 1, 2), dtype=jnp.bool_)

    result = _dpo._get_logprobs_for_target(
        logits=logits, targets=targets, sequence_mask=mask,
    )
    # Sum of log-probs should be close to 0 (each token ~ log(1) = 0).
    np.testing.assert_allclose(float(result[0, 0]), 0.0, atol=0.02)


class TestDpoLoss:
  """Tests for DpoLoss.get_values."""

  def test_output_shape(self):
    batch, n, seq_len, vocab = 2, 2, 4, 8
    tokens = jnp.zeros((batch, n, seq_len), dtype=jnp.int32)
    mask = jnp.ones((batch, n, seq_len), dtype=jnp.bool_)
    logits = jnp.ones((batch, n, seq_len, vocab))

    loss = _dpo.DpoLoss()
    result = loss.get_values(
        tokens=tokens,
        sequence_mask=mask,
        policy_logits=logits,
        anchor_logits=logits,
    )
    assert result.shape == (batch, 1)

  def test_zero_when_policy_equals_anchor(self):
    """When policy == anchor, diff_logprob is 0 for both chosen/rejected.

    po_delta = 0, so loss = -(log_sigmoid(0)*(1-ls) + log_sigmoid(0)*ls)
                           = -log_sigmoid(0) = -log(0.5) = log(2).
    """
    batch, n, seq_len, vocab = 1, 2, 3, 4
    rng = jax.random.PRNGKey(42)
    logits = jax.random.normal(rng, (batch, n, seq_len, vocab))
    tokens = jnp.zeros((batch, n, seq_len), dtype=jnp.int32)
    mask = jnp.ones((batch, n, seq_len), dtype=jnp.bool_)

    loss = _dpo.DpoLoss(tau=0.1, label_smoothing=0.0)
    result = loss.get_values(
        tokens=tokens,
        sequence_mask=mask,
        policy_logits=logits,
        anchor_logits=logits,
    )
    # po_delta = 0 => loss = -log_sigmoid(0) = log(2)
    np.testing.assert_allclose(
        float(result[0, 0]), np.log(2), atol=1e-5
    )

  def test_loss_is_non_negative(self):
    """DPO loss should always be non-negative."""
    batch, n, seq_len, vocab = 3, 2, 5, 8
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    policy_logits = jax.random.normal(k1, (batch, n, seq_len, vocab))
    anchor_logits = jax.random.normal(k2, (batch, n, seq_len, vocab))
    tokens = jax.random.randint(k1, (batch, n, seq_len), 0, vocab)
    mask = jnp.ones((batch, n, seq_len), dtype=jnp.bool_)

    loss = _dpo.DpoLoss(tau=0.1, label_smoothing=0.0)
    result = loss.get_values(
        tokens=tokens,
        sequence_mask=mask,
        policy_logits=policy_logits,
        anchor_logits=anchor_logits,
    )
    assert jnp.all(result >= 0.0)

  def test_label_smoothing_effect(self):
    """Label smoothing should change the loss value."""
    batch, n, seq_len, vocab = 1, 2, 3, 4
    rng = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(rng)
    policy_logits = jax.random.normal(k1, (batch, n, seq_len, vocab))
    anchor_logits = jax.random.normal(k2, (batch, n, seq_len, vocab))
    tokens = jnp.zeros((batch, n, seq_len), dtype=jnp.int32)
    mask = jnp.ones((batch, n, seq_len), dtype=jnp.bool_)

    loss_no_smooth = _dpo.DpoLoss(tau=0.1, label_smoothing=0.0)
    loss_smooth = _dpo.DpoLoss(tau=0.1, label_smoothing=0.5)

    result_no_smooth = loss_no_smooth.get_values(
        tokens=tokens,
        sequence_mask=mask,
        policy_logits=policy_logits,
        anchor_logits=anchor_logits,
    )
    result_smooth = loss_smooth.get_values(
        tokens=tokens,
        sequence_mask=mask,
        policy_logits=policy_logits,
        anchor_logits=anchor_logits,
    )
    # With label_smoothing=0.5, loss = -log_sigmoid(x)*0.5 - log_sigmoid(-x)*0.5
    # = -0.5*(log_sigmoid(x) + log_sigmoid(-x))
    # which differs from label_smoothing=0.0 unless po_delta=0.
    assert not jnp.allclose(result_no_smooth, result_smooth)
