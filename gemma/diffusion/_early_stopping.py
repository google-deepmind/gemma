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

"""Early stopping strategies for diffusion sampling."""

import dataclasses
from typing import Protocol
from typing import Sequence

import jax
import jax.numpy as jnp
from kauldron.ktyping import Bool, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class EarlyStopFn(Protocol):
  """Determines whether denoising should terminate early.

  Implementations receive the current and previous canvas tokens, the current
  logits, and the step index. They return a per-batch bool indicating whether
  each sequence in the batch should stop.
  """

  def should_stop(
      self,
      *,
      step: Int[''],
      canvas: Int['*B L'],
      previous_canvas: Int['*B L'],
      logits: Float['*B L V'],
  ) -> Bool['*B']:
    """Returns True for each batch element that should stop."""
    ...


@dataclasses.dataclass(frozen=True)
class NoEarlyStop(EarlyStopFn):
  """Default: never stop early. Equivalent to the original loop behavior."""

  @typechecked
  def should_stop(
      self,
      *,
      step: Int[''],
      canvas: Int['*B L'],
      previous_canvas: Int['*B L'],
      logits: Float['*B L V'],
  ) -> Bool['*B']:
    del step, previous_canvas, logits
    batch_size = canvas.shape[0]
    return jnp.zeros(batch_size, dtype=jnp.bool_)


@dataclasses.dataclass(frozen=True)
class TokenStabilityEarlyStop(EarlyStopFn):
  """Stop denoising when most-likely tokens stabilize across consecutive steps.

  Compares the argmax of the current logits with the previous canvas tokens.
  When the most confident predictions match the previous output, the denoiser
  has converged and further iterations are unlikely to change the output.

  Returns a per-batch boolean: True for each batch element whose most-likely
  tokens are identical to the previous canvas.
  """

  @typechecked
  def should_stop(
      self,
      *,
      step: Int[''],
      canvas: Int['*B L'],
      previous_canvas: Int['*B L'],
      logits: Float['*B L V'],
  ) -> Bool['*B']:
    del step, canvas
    most_likely_tokens = jnp.argmax(logits, axis=-1)
    return jnp.all(most_likely_tokens == previous_canvas, axis=-1)


@dataclasses.dataclass(frozen=True)
class EntropyEarlyStop(EarlyStopFn):
  """Stop denoising when the entropy of the logits is below a threshold.

  When the entropy is low, the denoiser has become very confident in its
  predictions, and further iterations are unlikely to yield significant
  improvements.

  Returns a per-batch boolean: True for each batch element whose mean
  per-token entropy is at or below the threshold.
  """

  entropy_threshold: float = 0.005

  @typechecked
  def should_stop(
      self,
      *,
      step: Int[''],
      canvas: Int['*B L'],
      previous_canvas: Int['*B L'],
      logits: Float['*B L V'],
  ) -> Bool['*B']:
    del step, canvas, previous_canvas
    log_probs = jax.nn.log_softmax(logits)
    probs = jnp.exp(log_probs)
    # Guard against log(0) producing NaN in the entropy sum.
    log_probs = jnp.where(probs == 0, 0.0, log_probs)
    entropy_per_token = -jnp.sum(log_probs * probs, axis=-1)
    # Mean over the sequence (token) dimension, keeping batch dimension.
    entropy = jnp.mean(entropy_per_token, axis=-1)
    return entropy <= self.entropy_threshold


@dataclasses.dataclass(frozen=True)
class ChainedEarlyStop(EarlyStopFn):
  """Stop denoising if all of the provided early stopping functions agree.

  Returns a per-batch boolean: True for each batch element where every
  sub-stopper returns True (logical AND across stoppers).
  """

  early_stop_fns: Sequence['EarlyStopFn']

  def __post_init__(self):
    object.__setattr__(self, 'early_stop_fns', tuple(self.early_stop_fns))
    if not self.early_stop_fns:
      raise ValueError(
          'ChainedEarlyStop requires at least one EarlyStopFn, use NoEarlyStop'
          ' for the default behavior.'
      )

  @typechecked
  def should_stop(
      self,
      *,
      step: Int[''],
      canvas: Int['*B L'],
      previous_canvas: Int['*B L'],
      logits: Float['*B L V'],
  ) -> Bool['*B']:
    results = jnp.stack([
        fn.should_stop(
            step=step,
            canvas=canvas,
            previous_canvas=previous_canvas,
            logits=logits,
        )
        for fn in self.early_stop_fns
    ])
    # AND across stoppers (axis=0), keeping per-batch dimension.
    return jnp.all(results, axis=0)
