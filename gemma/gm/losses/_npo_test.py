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

"""Tests for NPO loss."""

from gemma.gm.losses import _npo
import jax
import jax.numpy as jnp
import numpy as np


class TestGetLogprobsForTarget:
  """Tests for _get_logprobs_for_target helper."""

  def test_shape(self):
    batch, seq_len, vocab = 2, 4, 8
    logits = jnp.ones((batch, seq_len, vocab))
    targets = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    mask = jnp.ones((batch, seq_len), dtype=jnp.bool_)

    result = _npo._get_logprobs_for_target(
        logits=logits, targets=targets, sequence_mask=mask,
    )
    assert result.shape == (batch,)

  def test_logprobs_are_negative(self):
    """Log-probabilities should always be <= 0."""
    batch, seq_len, vocab = 3, 5, 8
    rng = jax.random.PRNGKey(0)
    logits = jax.random.normal(rng, (batch, seq_len, vocab))
    targets = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    mask = jnp.ones((batch, seq_len), dtype=jnp.bool_)

    result = _npo._get_logprobs_for_target(
        logits=logits, targets=targets, sequence_mask=mask,
    )
    assert jnp.all(result <= 0.0)

  def test_masked_tokens_ignored(self):
    """Masked positions should not contribute to log-probs."""
    vocab = 4
    logits = jnp.zeros((1, 3, vocab))
    targets = jnp.array([[0, 1, 2]], dtype=jnp.int32)

    mask_all = jnp.ones((1, 3), dtype=jnp.bool_)
    mask_first_only = jnp.array([[True, False, False]])

    result_all = _npo._get_logprobs_for_target(
        logits=logits, targets=targets, sequence_mask=mask_all,
    )
    result_partial = _npo._get_logprobs_for_target(
        logits=logits, targets=targets, sequence_mask=mask_first_only,
    )
    # Fewer tokens summed => less-negative value.
    assert float(result_partial[0]) > float(result_all[0])


class TestNpoLoss:
  """Tests for NpoLoss.get_values."""

  def test_output_shape(self):
    batch, seq_len, vocab = 2, 4, 8
    tokens = jnp.zeros((batch, seq_len, 1), dtype=jnp.int32)
    mask = jnp.ones((batch, seq_len, 1), dtype=jnp.bool_)
    logits = jnp.ones((batch, seq_len, vocab))

    loss = _npo.NpoLoss()
    result = loss.get_values(
        tokens=tokens,
        sequence_mask=mask,
        policy_logits=logits,
        anchor_logits=logits,
    )
    assert result.shape == (batch, 1)

  def test_zero_delta_when_policy_equals_anchor(self):
    """When policy == anchor, po_delta = 0.

    NPO loss = -log_sigmoid(0) = -log(0.5) = log(2).
    """
    batch, seq_len, vocab = 1, 3, 4
    rng = jax.random.PRNGKey(42)
    logits = jax.random.normal(rng, (batch, seq_len, vocab))
    tokens = jnp.zeros((batch, seq_len, 1), dtype=jnp.int32)
    mask = jnp.ones((batch, seq_len, 1), dtype=jnp.bool_)

    loss = _npo.NpoLoss(tau=1.0)
    result = loss.get_values(
        tokens=tokens,
        sequence_mask=mask,
        policy_logits=logits,
        anchor_logits=logits,
    )
    np.testing.assert_allclose(
        float(result[0, 0]), np.log(2), atol=1e-5
    )

  def test_loss_is_non_negative(self):
    """NPO loss should always be non-negative."""
    batch, seq_len, vocab = 3, 5, 8
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    policy_logits = jax.random.normal(k1, (batch, seq_len, vocab))
    anchor_logits = jax.random.normal(k2, (batch, seq_len, vocab))
    tokens = jax.random.randint(k1, (batch, seq_len, 1), 0, vocab)
    mask = jnp.ones((batch, seq_len, 1), dtype=jnp.bool_)

    loss = _npo.NpoLoss(tau=1.0)
    result = loss.get_values(
        tokens=tokens,
        sequence_mask=mask,
        policy_logits=policy_logits,
        anchor_logits=anchor_logits,
    )
    assert jnp.all(result >= 0.0)

  def test_tau_scales_loss(self):
    """Different tau values should produce different losses."""
    batch, seq_len, vocab = 1, 3, 4
    rng = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(rng)
    policy_logits = jax.random.normal(k1, (batch, seq_len, vocab))
    anchor_logits = jax.random.normal(k2, (batch, seq_len, vocab))
    tokens = jnp.zeros((batch, seq_len, 1), dtype=jnp.int32)
    mask = jnp.ones((batch, seq_len, 1), dtype=jnp.bool_)

    loss_tau1 = _npo.NpoLoss(tau=1.0)
    loss_tau5 = _npo.NpoLoss(tau=5.0)

    result_tau1 = loss_tau1.get_values(
        tokens=tokens,
        sequence_mask=mask,
        policy_logits=policy_logits,
        anchor_logits=anchor_logits,
    )
    result_tau5 = loss_tau5.get_values(
        tokens=tokens,
        sequence_mask=mask,
        policy_logits=policy_logits,
        anchor_logits=anchor_logits,
    )
    assert not jnp.allclose(result_tau1, result_tau5)
