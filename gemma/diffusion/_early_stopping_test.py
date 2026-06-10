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

from absl.testing import absltest
from absl.testing import parameterized
from gemma.diffusion import _early_stopping
import jax
import jax.numpy as jnp
import numpy as np


def _logits_with_argmax(tokens, vocab_size=32):
  """Creates logits whose argmax along the last axis equals `tokens`.

  Sets the logit at each token index to 10.0, all others to 0.0.

  Args:
    tokens: Integer token indices.
    vocab_size: Size of the vocabulary dimension.

  Returns:
    Logits array of shape `tokens.shape + (vocab_size,)`.
  """
  one_hot = jax.nn.one_hot(tokens, vocab_size)
  return one_hot * 10.0


class NoEarlyStopTest(absltest.TestCase):

  def test_never_stops(self):
    """NoEarlyStop.should_stop always returns False for every batch element."""
    stopper = _early_stopping.NoEarlyStop()
    canvas = jnp.array([[1, 2, 3], [4, 5, 6]])
    logits = _logits_with_argmax(canvas)
    result = stopper.should_stop(
        step=jnp.int32(5),
        canvas=canvas,
        previous_canvas=canvas,
        logits=logits,
    )
    np.testing.assert_array_equal(result, [False, False])


class TokenStabilityEarlyStopTest(parameterized.TestCase):

  def test_stops_when_argmax_matches_previous(self):
    """Returns True per batch element when argmax(logits) == previous_canvas."""
    stopper = _early_stopping.TokenStabilityEarlyStop()
    previous_canvas = jnp.array([[2, 5, 7]])
    logits = _logits_with_argmax(previous_canvas)
    result = stopper.should_stop(
        step=jnp.int32(1),
        canvas=jnp.array([[99, 99, 99]]),  # canvas is ignored
        previous_canvas=previous_canvas,
        logits=logits,
    )
    np.testing.assert_array_equal(result, [True])

  def test_does_not_stop_when_argmax_differs(self):
    """Returns False per batch element when argmax(logits) != previous_canvas."""
    stopper = _early_stopping.TokenStabilityEarlyStop()
    previous_canvas = jnp.array([[1, 2, 3]])
    logits = _logits_with_argmax(jnp.array([[4, 5, 6]]))
    result = stopper.should_stop(
        step=jnp.int32(1),
        canvas=previous_canvas,
        previous_canvas=previous_canvas,
        logits=logits,
    )
    np.testing.assert_array_equal(result, [False])

  def test_partial_mismatch_does_not_stop(self):
    """Returns False when even a single argmax token differs in a sequence."""
    stopper = _early_stopping.TokenStabilityEarlyStop()
    previous_canvas = jnp.array([[1, 2, 3]])
    logits = _logits_with_argmax(jnp.array([[1, 2, 7]]))
    result = stopper.should_stop(
        step=jnp.int32(1),
        canvas=previous_canvas,
        previous_canvas=previous_canvas,
        logits=logits,
    )
    np.testing.assert_array_equal(result, [False])

  def test_per_batch_independence(self):
    """Each batch element terminates independently."""
    stopper = _early_stopping.TokenStabilityEarlyStop()
    previous_canvas = jnp.array([[1, 2], [3, 4]])
    # Batch 0: argmax matches, Batch 1: argmax differs.
    current_logits = _logits_with_argmax(jnp.array([[1, 2], [3, 9]]))
    result = stopper.should_stop(
        step=jnp.int32(1),
        canvas=previous_canvas,
        previous_canvas=previous_canvas,
        logits=current_logits,
    )
    np.testing.assert_array_equal(result, [True, False])


class EntropyEarlyStopTest(parameterized.TestCase):

  def test_stops_below_threshold(self):
    """Returns True when entropy is at or below the threshold."""
    logits = jnp.array([[[10.0, -10.0, -10.0, -10.0]]])
    stopper = _early_stopping.EntropyEarlyStop(entropy_threshold=1.0)
    result = stopper.should_stop(
        step=jnp.int32(0),
        canvas=jnp.array([[0]]),
        previous_canvas=jnp.array([[0]]),
        logits=logits,
    )
    np.testing.assert_array_equal(result, [True])

  def test_does_not_stop_above_threshold(self):
    """Returns False when entropy is above the threshold."""
    logits = jnp.zeros((1, 3, 32), dtype=jnp.float32)
    stopper = _early_stopping.EntropyEarlyStop(entropy_threshold=0.01)
    result = stopper.should_stop(
        step=jnp.int32(0),
        canvas=jnp.array([[0, 0, 0]]),
        previous_canvas=jnp.array([[0, 0, 0]]),
        logits=logits,
    )
    np.testing.assert_array_equal(result, [False])

  def test_exact_threshold_stops(self):
    """Returns True when entropy equals the threshold (uses <=)."""
    logits = jnp.array([[[10.0, -10.0, -10.0, -10.0]]])
    log_probs = jax.nn.log_softmax(logits)
    probs = jnp.exp(log_probs)
    log_probs_safe = jnp.where(probs == 0, 0.0, log_probs)
    exact_entropy = float(-jnp.sum(log_probs_safe * probs, axis=-1).mean())

    stopper = _early_stopping.EntropyEarlyStop(entropy_threshold=exact_entropy)
    result = stopper.should_stop(
        step=jnp.int32(0),
        canvas=jnp.array([[0]]),
        previous_canvas=jnp.array([[0]]),
        logits=logits,
    )
    np.testing.assert_array_equal(result, [True])

  def test_handles_zero_probability(self):
    """Logits with -inf entries (probs==0) should not produce NaN."""
    logits = jnp.array([[[0.0, -jnp.inf, -jnp.inf, 0.0]]])
    stopper = _early_stopping.EntropyEarlyStop(entropy_threshold=10.0)
    result = stopper.should_stop(
        step=jnp.int32(0),
        canvas=jnp.array([[0]]),
        previous_canvas=jnp.array([[0]]),
        logits=logits,
    )
    np.testing.assert_array_equal(result, [True])

  def test_per_batch_independence(self):
    """Each batch element can have different entropy."""
    # Batch 0: peaked (low entropy), Batch 1: uniform (high entropy).
    logits = jnp.array([
        [[10.0, -10.0, -10.0, -10.0]],
        [[0.0, 0.0, 0.0, 0.0]],
    ])
    stopper = _early_stopping.EntropyEarlyStop(entropy_threshold=0.5)
    result = stopper.should_stop(
        step=jnp.int32(0),
        canvas=jnp.array([[0], [0]]),
        previous_canvas=jnp.array([[0], [0]]),
        logits=logits,
    )
    np.testing.assert_array_equal(result, [True, False])


class ChainedEarlyStopTest(absltest.TestCase):

  def test_requires_all_to_agree(self):
    """ChainedEarlyStop only stops when all sub-stoppers agree per element."""
    previous_canvas = jnp.array([[0, 1, 2]])
    logits = _logits_with_argmax(jnp.array([[5, 6, 7]]))

    chained = _early_stopping.ChainedEarlyStop(
        early_stop_fns=[
            _early_stopping.TokenStabilityEarlyStop(),
            _early_stopping.EntropyEarlyStop(entropy_threshold=10.0),
        ]
    )
    result = chained.should_stop(
        step=jnp.int32(1),
        canvas=previous_canvas,
        previous_canvas=previous_canvas,
        logits=logits,
    )
    np.testing.assert_array_equal(result, [False])

  def test_stops_when_all_agree(self):
    """ChainedEarlyStop stops when all sub-stoppers return True."""
    previous_canvas = jnp.array([[0, 1, 2]])
    logits = _logits_with_argmax(previous_canvas)

    chained = _early_stopping.ChainedEarlyStop(
        early_stop_fns=[
            _early_stopping.TokenStabilityEarlyStop(),
            _early_stopping.EntropyEarlyStop(entropy_threshold=10.0),
        ]
    )
    result = chained.should_stop(
        step=jnp.int32(1),
        canvas=previous_canvas,
        previous_canvas=previous_canvas,
        logits=logits,
    )
    np.testing.assert_array_equal(result, [True])

  def test_empty_raises(self):
    """ChainedEarlyStop raises ValueError if early_stop_fns is empty."""
    with self.assertRaises(ValueError):
      _early_stopping.ChainedEarlyStop(early_stop_fns=[])


if __name__ == '__main__':
  absltest.main()
