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

"""Checkpoint loader for pair-wise models."""

import dataclasses

from gemma.gm.ckpts import _checkpoint
import jax
import jax.numpy as jnp
from kauldron import kd


@dataclasses.dataclass(frozen=True, kw_only=True)
class AnchoredPolicyLoader(kd.ckpts.AbstractPartialLoader):
  """Loader for `gm.nn.AnchoredPolicy` models.

  Loaded load policy and anchor separately by providing
  sub-transforms.

  This assume the sub-loaders only overwrite the `state.params` without
  modifying the rest of the state.
  """

  policy: kd.ckpts.AbstractPartialLoader
  anchor: kd.ckpts.AbstractPartialLoader | None = None

  def transform(self, state: kd.train.TrainState) -> kd.train.TrainState:
    if set(state.params.keys()) != {'policy', 'anchor'}:
      raise ValueError(
          'AnchoredPolicyLoader is meant to be used with'
          ' `model=gm.nn.AnchoredPolicy`.'
      )

    # Load the policy params.
    policy_state = dataclasses.replace(state, params=state.params['policy'])
    policy_state = self.policy.transform(policy_state)

    # Load the anchor params.
    if self.anchor is None:
      # If `anchor` is not provided, load a copy the policy params.
      _checkpoint.release_memory(state.params['anchor'])
      anchor_params = jax.tree.map(jnp.copy, policy_state.params)
      anchor_state = dataclasses.replace(
          policy_state,
          params=anchor_params,
      )
    else:
      anchor_state = dataclasses.replace(state, params=state.params['anchor'])
      anchor_state = self.anchor.transform(anchor_state)

    # Merge the two states back together.
    state = dataclasses.replace(
        state,
        params={
            'policy': policy_state.params,
            'anchor': anchor_state.params,
        },
    )
    return state
