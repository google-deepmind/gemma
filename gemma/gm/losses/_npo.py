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

"""NPO loss."""

import dataclasses

import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron import kontext
from kauldron.typing import Bool, Float, Int, Schedule, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(frozen=True, kw_only=True)
class NpoLoss(kd.losses.Loss):
  """NPO loss.

  Attributes:
    tau: The temperature of the loss.
    label_smoothing: The label smoothing to apply to the loss.
    tokens: The key to the tokens to predict.
    sequence_mask: The key to the sequence mask.
    policy_logits: The key to the policy logits.
    anchor_logits: The key to the anchor logits.
  """

  tau: float | Schedule = 1.0

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
      tokens: Int['*B L 1'],
      sequence_mask: Bool['*B L 1'],
      policy_logits: Float['*B L V'],
      anchor_logits: Float['*B L V'],
  ) -> Float['*B 1']:
    """Computes the NPO loss."""

    sequence_mask = jnp.squeeze(sequence_mask, axis=-1)
    targets = jnp.squeeze(tokens, axis=-1)

    policy_logprobs = _get_logprobs_for_target(
        logits=policy_logits,
        targets=targets,
        sequence_mask=sequence_mask,
    )
    anchor_logprobs = _get_logprobs_for_target(
        logits=anchor_logits,
        targets=targets,
        sequence_mask=sequence_mask,
    )

    # Float['*B']
    po_delta = (policy_logprobs - anchor_logprobs) * self.tau

    # Compute the per-example loss. Note that the averaging across examples
    # is done by the base class.

    npo_loss = -jax.nn.log_sigmoid(-po_delta)

    return npo_loss[..., None]  # Float['*B 1'] for Loss compatibility.


@typechecked
def _get_logprobs_for_target(
    *,
    logits: Float['*B L V'],
    targets: Int['*B L'],
    sequence_mask: Bool['*B L'],
) -> Float['*B']:
  """Computes the per token xent given logits."""
  # We perform softmax in float32 to improve stability.
  logits = logits.astype(jnp.float32)

  log_probs = jax.nn.log_softmax(logits)
  log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
  log_probs = jnp.squeeze(log_probs, axis=-1)
  log_probs = jnp.sum(log_probs * sequence_mask, axis=-1)
  return log_probs
