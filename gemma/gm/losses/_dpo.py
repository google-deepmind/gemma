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

"""DPO loss."""

import dataclasses

import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron import kontext
from kauldron.typing import Bool, Dim, Float, Int, Schedule, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(frozen=True, kw_only=True)
class DpoLoss(kd.losses.Loss):
  """DPO loss.

  Attributes:
    tau: The temperature of the loss.
    label_smoothing: The label smoothing to apply to the loss.
    tokens: The key to the tokens to predict.
    sequence_mask: The key to the sequence mask.
    policy_logits: The key to the policy logits.
    anchor_logits: The key to the anchor logits.
  """

  tau: float | Schedule = 0.1
  label_smoothing: float | Schedule = 0.0

  # Keys defined in the config to specify which inputs to
  # pass to the `get_values` function.
  tokens: kontext.Key = kontext.REQUIRED
  sequence_mask: kontext.Key = kontext.REQUIRED
  policy_logits: kontext.Key = kontext.REQUIRED
  anchor_logits: kontext.Key = kontext.REQUIRED

  @typechecked
  def get_values(
      self,
      *,
      tokens: Int['*B N L'],
      sequence_mask: Bool['*B N L'],
      policy_logits: Float['*B N L V'],
      anchor_logits: Float['*B N L V'],
  ) -> Float['*B 1']:
    """Computes the DPO loss."""
    # TODO(epot): Supports schedules for tau and label_smoothing !!!

    # DPO loss has positive and negative rewards, so N == 2.

    policy_logprobs = _get_logprobs_for_target(
        logits=policy_logits,
        targets=tokens,
        sequence_mask=sequence_mask,
    )
    anchor_logprobs = _get_logprobs_for_target(
        logits=anchor_logits,
        targets=tokens,
        sequence_mask=sequence_mask,
    )

    # Float['*B N']
    diff_logprob = policy_logprobs - anchor_logprobs

    # Float['*B']
    po_delta = (diff_logprob[..., 0] - diff_logprob[..., 1]) * self.tau

    # Compute the per-example loss. Note that the averaging across examples
    # is done by the base class.

    dpo_loss = -(
        jax.nn.log_sigmoid(po_delta) * (1 - self.label_smoothing)
        + jax.nn.log_sigmoid(-po_delta) * self.label_smoothing
    )

    return dpo_loss[..., None]  # Float['*B 1'] for Loss compatibility.


@typechecked
def _get_logprobs_for_target(
    *,
    logits: Float['*B N L V'],
    targets: Int['*B N L'],
    sequence_mask: Bool['*B N L'],
) -> Float['*B N']:
  """Computes the per token xent given logits."""
  # We perform softmax in float32 to improve stability.
  logits = logits.astype(jnp.float32)

  log_probs = jax.nn.log_softmax(logits)
  log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
  log_probs = jnp.squeeze(log_probs, axis=-1)
  log_probs = jnp.sum(log_probs * sequence_mask, axis=-1)
  return log_probs
