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

"""Policy wrapper for DPO-like losses."""

from typing import Any, Optional

import flax
from flax import linen as nn
import jax
from kauldron import kontext


@flax.struct.dataclass
class AnchoredPolicyOutput:
  """Output of the `gm.nn.AnchoredPolicy`."""

  policy: Any
  anchor: Any


class AnchoredPolicy(nn.Module):
  """Wrapper around a model to compute policy and anchor outputs.

  This wrapper takes an input and pass it through two models:
  - `policy`: Model trained.
  - `anchor`: Frozen model. If not provided, is set to a copy of `policy`.

  To initialize the model, use `gm.ckpts.AnchoredPolicyLoader`.
  """

  policy: nn.Module
  anchor: Optional[nn.Module] = None

  @nn.compact
  def __call__(self, *args, **kwargs) -> AnchoredPolicyOutput:
    # If no anchor is provided, we use a copy of the policy.
    if self.anchor is None:
      anchor = self.policy.copy(name='anchor')
    else:
      anchor = self.anchor

    # Pass the inputs to both policy and anchor models.
    policy_preds = self.policy(*args, **kwargs)

    anchor_preds = anchor(*args, **kwargs)
    anchor_preds = jax.lax.stop_gradient(anchor_preds)  # Anchor is frozen.

    return AnchoredPolicyOutput(
        policy=policy_preds,
        anchor=anchor_preds,
    )

  def __kontext_keys__(self) -> dict[str, str]:
    """Kauldron keys when calling `kontext.get_from_keys_obj`."""
    # Forward the keys from the policy model.
    # This allow to define the config as:
    # gm.nn.AnchoredPolicy(
    #   policy=MyModel(
    #     input='batch.input',  # keys propagated to the `AnchoredPolicy`
    #   ),
    # )
    return kontext.get_keypaths(self.policy)
