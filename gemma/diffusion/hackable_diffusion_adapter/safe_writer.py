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

"""Safe metric writer that avoids NCCL crashes on multi-GPU.

Overrides ``write_param_overview`` to count parameters without
calling ``jax.device_get()`` on sharded arrays.
"""

import dataclasses
from kauldron.train import metric_writer


# --- Patch B: Safe _convert_leaf using single-shard reads ---
# Never call jax.device_get() on sharded arrays.  Always read from a
# single local shard to avoid triggering NCCL AllGather.
import jax
import numpy as np
import kauldron.train.auxiliaries as aux
def _convert_leaf(leaf):
  if isinstance(leaf, jax.Array):
    shard_data = leaf.addressable_shards[0].data
    return np.asarray(shard_data)
  return leaf
aux._convert_leaf = _convert_leaf

# --- Patch C: Safe _compute_metric with synchronization ---
# Force jax.block_until_ready after metric computation to prevent
# concurrent NCCL operations that corrupt the communicator.

_orig_compute = aux.AuxiliariesState.compute
def _safe_compute(self, *args, **kwargs):
  result = _orig_compute(self, *args, **kwargs)
  jax.block_until_ready(jax.tree_util.tree_leaves(result))
  return result
aux.AuxiliariesState.compute = _safe_compute


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class SafeMetricWriter(metric_writer.KDMetricWriter):
  """Metric writer that avoids NCCL crashes on multi-GPU.

  Overrides ``write_param_overview`` to count parameters without
  calling ``jax.device_get()`` on sharded arrays.
  """

  def write_param_overview(self, step: int, params) -> None:
    num_parameters = metric_writer.parameter_overview.count_parameters(params)
    self.write_summaries(step=step, values={'num_params': num_parameters})
