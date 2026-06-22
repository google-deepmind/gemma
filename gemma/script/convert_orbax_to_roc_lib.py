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

"""Library for converting Gemma orbax checkpoints to ROC format."""

from typing import Any

from gemma.gm.ckpts import _compat


def convert_full_model_to_flat(
    nested_model_dict: dict[str, Any],
) -> dict[str, Any]:
  """Converts the full nested tree to the flat checkpoint structure.

  The output structure matches what JetEngine and ROC checkpoints expect: a
  semi-flat dict whose keys are slash-separated module paths prefixed with
  `transformer/` (text encoder) or `SigLiPFromPatches_0/` (vision encoder),
  and whose values are dicts mapping leaf parameter names (`'w'`, `'scale'`,
  `'bias'`, ...) to arrays.

  This uses the gemma library's `_compat.flatten_and_remap_params`, which is
  the proven inverse of `_compat.param_remapper` + `_compat.nest_params` used
  by `gm.ckpts.load_params`. Using the library function directly (rather than
  reimplementing it with structural heuristics) guarantees correctness for any
  Gemma model architecture, including Gemma 4 MoE layers which contain both
  submodule dicts (`gating_einsum`, `linear`, ...) and standalone array
  parameters (`per_expert_scale`, `router_scale`) at the same level.

  Args:
    nested_model_dict: The nested params tree returned by `gm.ckpts.load_params`
      (matching the Flax `model.init()['params']` structure).

  Returns:
    A semi-flat dict of params with the structure expected by JetEngine.
  """
  # Split the params into the text transformer and (optional) vision encoder
  # sub-trees, since each is stored under a different top-level prefix in the
  # flat layout (`transformer/` vs. `SigLiPFromPatches_0/`).
  nested_model_dict = dict(nested_model_dict)
  vision_params = nested_model_dict.pop('vision_encoder', None)

  flat = {
      f'transformer/{k}': v
      for k, v in _compat.flatten_and_remap_params(nested_model_dict).items()
  }
  if vision_params:
    flat.update({
        f'SigLiPFromPatches_0/{k}': v
        for k, v in _compat.flatten_and_remap_params(vision_params).items()
    })
  return flat
