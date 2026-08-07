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

"""Checkpointers for DiffusionGemma SFT logic.

This module contains the GemmaDiffusionCheckpointLoader, which is used to load
pretrained DiffusionGemma weights into the SFT model. It also contains the
GemmaCheckpointFormatter, which is used to save the params into the
DiffusionGemma-format and lora format.
"""

import copy
import dataclasses
from typing import Any

from absl import logging
from etils import epath
import flax
from gemma import peft
from gemma.diffusion.hackable_diffusion_adapter.hd import lora
import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron.utils import config_util
import orbax.checkpoint as ocp

from gemma.diffusion.hackable_diffusion_adapter import checkpointed_evaluator as _checkpointed_evaluator  #pylint: disable=line-too-long
CheckpointedEvaluator = _checkpointed_evaluator.CheckpointedEvaluator


# Override path finalization for GCS checkpoints.
import orbax.checkpoint._src.path.step as _step_lib
_step_lib.is_path_finalized = lambda path: True


################################################################################
# MARK: Gemma diffusion checkpointer.
################################################################################


def _remap_and_match_params(
    model_flat: dict[str, Any],
    ckpt_flat: dict[str, Any],
    lora_init_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Remaps checkpoint keys and merges them into the model param dict.

  This is the core logic of checkpoint-to-model parameter matching:
    1. Dynamically strip trailing '/w' from checkpoint paths when the stripped
       path matches a model key but the unstripped path does not (handles the
       difference between ``FeedForward`` with ``nn.share_scope`` and
       ``MoERagged``'s ``_Weight``).
    2. Replace model values with matching checkpoint values.
    3. Restore any LoRA init values.
    4. Validate that no non-LoRA model keys are missing from the checkpoint.

  Args:
    model_flat: Flattened model param dict ``{path: value}``.  Values are either
      ``ShapeDtypeStruct`` specs or arrays.
    ckpt_flat: Flattened checkpoint param dict ``{path: value}``.
    lora_init_values: Optional dict of LoRA param paths to their initialized
      values.  These are preserved as-is (not loaded from the checkpoint).

  Returns:
    The ``model_flat`` dict with values replaced by checkpoint values where
    paths match, and LoRA values restored.

  Raises:
    KeyError: If any non-LoRA model keys have no matching checkpoint key.
  """
  if lora_init_values is None:
    lora_init_values = {}

  # Remap checkpoint paths to match model format
  # Some Flax modules (e.g. FeedForward with nn.share_scope) hoist the
  # param name into the parent scope, so the model path is e.g.
  # 'layer_0/mlp/gating_einsum' while the checkpoint has
  # 'layer_0/mlp/gating_einsum/w'.  Other modules
  #  keep the '/w' in both model and checkpoint.
  #
  # We dynamically strip the
  # trailing '/w' only when:
  #   1. The stripped path matches a model key, AND
  #   2. The unstripped path does NOT match a model key.
  remapped_ckpt = {}
  for ckpt_path, value in ckpt_flat.items():
    if ckpt_path.endswith('/w'):
      stripped = ckpt_path.rsplit('/w', 1)[0]
      if stripped in model_flat and ckpt_path not in model_flat:
        remapped_ckpt[stripped] = value
        continue
    remapped_ckpt[ckpt_path] = value

  # Replace model values with checkpoint values (where paths match)
  loaded_count = 0
  for path in model_flat:
    if path in remapped_ckpt:
      model_flat[path] = remapped_ckpt[path]
      loaded_count += 1

  # Restore preserved LoRA init values (actual arrays, not specs)
  for k, v in lora_init_values.items():
    model_flat[k] = v

  # Log checkpoint-only keys (not present in the model)
  ckpt_only = set(remapped_ckpt) - set(model_flat)
  if ckpt_only:
    logging.warning(
        'Discarding %d checkpoint-only key(s) not present in the model: %s',
        len(ckpt_only),
        sorted(ckpt_only),
    )

  # Separate LoRA-only keys (not expected in the checkpoint)
  model_only = set(model_flat) - set(remapped_ckpt)
  lora_keys = {k for k in model_only if '/lora/' in k}
  non_lora_model_only = model_only - lora_keys

  if lora_keys:
    logging.info(
        'Keeping %d LoRA key(s) with their initialized values.', len(lora_keys)
    )

  # Throw an exception for non-LoRA model_only keys
  if non_lora_model_only:
    raise KeyError(
        f'Found {len(non_lora_model_only)} model-only key(s) '
        f'(excluding LoRA): {sorted(non_lora_model_only)}'
    )

  logging.info(
      'Checkpoint loading complete: %d params loaded, '
      '%d checkpoint-only (discarded).',
      loaded_count,
      len(ckpt_only),
  )

  return model_flat


def _convert_to_element_spec_with_sharding(tree):
  """Converts a param tree to ShapeDtypeStruct specs preserving sharding."""
  return jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(
          dtype=x.dtype,
          shape=x.shape,
          sharding=x.sharding,
      ),
      tree,
  )


def cheaply_load_params(params_from_state, checkpoint_path: epath.PathLike):
  """Loads params from a checkpoint into a model with LoRA layers.

  Args:
    params_from_state: Existing model parameter tree or spec.
    checkpoint_path: Path to the checkpoint directory.

  Returns:
    A merged parameter dictionary matching the model's expected structure.
  """

  # Keep only the spec and free device memory for the arrays.
  existing = params_from_state
  model_param_spec = _convert_to_element_spec_with_sharding(existing)

  # Preserve LoRA param values before releasing memory — these won't be
  # in the checkpoint and must keep their init values.
  _existing_flat_arrays = flax.traverse_util.flatten_dict(existing, sep='/')
  _lora_init_values = {
      k: v for k, v in _existing_flat_arrays.items() if '/lora/' in k
  }

  for k, v in _existing_flat_arrays.items():
    if k not in _lora_init_values and isinstance(v, jax.Array):
      v.delete()

  # Load raw Gemma params (auto-detects checkpoint format).
  # Load into CPU memory to avoid OOM on device.
  with jax.default_device(jax.devices('cpu')[0]):

    import numpy as np

    def _make_empty_cpu_array(spec):
      return np.empty(spec.shape, spec.dtype)

    ckpt = ocp.PyTreeCheckpointer()
    metadata = ckpt.metadata(checkpoint_path)
    lparams_empty = jax.tree.map(
        _make_empty_cpu_array, metadata.item_metadata.tree
    )
    gemma_params = ckpt.restore(
        checkpoint_path, item=lparams_empty
    )

  # Flatten both trees to leaf-level paths
  existing_flat = flax.traverse_util.flatten_dict(model_param_spec, sep='/')
  ckpt_flat = flax.traverse_util.flatten_dict(gemma_params, sep='/')

  existing_flat = _remap_and_match_params(
      existing_flat, ckpt_flat, _lora_init_values
  )

  # Unflatten back to the model's original nested structure
  merged = flax.traverse_util.unflatten_dict(existing_flat, sep='/')

  # Convert dtype and move the params back on device
  merged = jax.tree.map(
      lambda x, y: jax.device_put(x.astype(y.dtype), device=y.sharding),
      merged,
      model_param_spec,
  )
  return merged


@dataclasses.dataclass(frozen=True)
class GemmaDiffusionCheckpointLoader(kd.ckpts.InitTransform):
  """Loads pretrained Gemma Diffusion weights into the SFT model.

  Uses a flatten-and-match strategy to guarantee the output tree has the
  **exact same structure** as model.init() (as required by Kauldron's
  InitTransform contract).  Only leaf values are replaced; the tree
  structure is never modified.

  Workflow:
    1. Flatten the model's params to ``{path: leaf_array}``
    2. Flatten the checkpoint's params to ``{path: leaf_array}``
    3. For each model path that has a matching checkpoint path, replace the
       value with the pretrained weight
    4. Assert checkpoint-only paths match an expected allowlist
    5. Raise if any model-only paths exist (missing checkpoint coverage)
    6. Unflatten back to the model's original nested structure

  Attributes:
    path: Path to the Orbax checkpoint directory containing Gemma Diffusion
      weights.
    gemma_param_path: Sequence of keys from the state.params root to the
      gemma_model params. Adjust if the Flax module nesting differs.
  """

  path: epath.PathLike
  gemma_param_path: str = 'gemma_network.gemma_model'

  def transform(self, state):
    # Retrieve the model's existing (randomly initialized) params at the
    # gemma_param_path.
    existing = kd.kontext.get_by_path(state.params, self.gemma_param_path)
    merged = cheaply_load_params(
        params_from_state=existing, checkpoint_path=self.path
    )
    params = copy.copy(state.params)  # Shallow copy in case of FrozenDict.
    kd.kontext.set_by_path(params, self.gemma_param_path, merged)
    return dataclasses.replace(state, params=params)


###############################################################################
# MARK: Gemma diffusion checkpointing formatting.
################################################################################


def save_params_into_original_and_lora_params(
    params: dict[str, Any],
    step_nr: int,
    workdir: epath.Path,
    write_original_params: bool = True,
    write_lora_params: bool = True,
    write_fused_lora_params: bool = True,
):
  """Saves the params into the gemma-diffusion format and lora format.

  This function saves the params into three different formats:
  1. Original params: The params without the lora adapters.
  2. Lora params: The lora adapters.
  3. Fused lora params: The params with the lora adapters fused into the
  original params.

  Args:
    params: The params to save.
    step_nr: The step number to save.
    workdir: The workdir to save the params to.
    write_original_params: Whether to save the original params.
    write_lora_params: Whether to save the lora params.
    write_fused_lora_params: Whether to save the fused lora params.
  """
  # Split into original and lora
  splitted_params = peft.split_params(params)

  # Checkpointer to save the params.
  checkpointer = ocp.PyTreeCheckpointer()

  if write_original_params:
    original_params_dir = workdir / f'original_params_{step_nr}'
    if original_params_dir.exists():
      logging.info('Skipping save: %s already exists.', original_params_dir)
    else:
      checkpointer.save(original_params_dir, splitted_params.original)

  # If we have lora -> save them too.
  if len(splitted_params.lora) > 0:
    if write_lora_params:
      lora_params_dir = workdir / f'lora_params_{step_nr}'
      if lora_params_dir.exists():
        logging.info('Skipping save: %s already exists.', lora_params_dir)
      else:
        checkpointer.save(lora_params_dir, splitted_params.lora)

    if write_fused_lora_params:
      # Fuse the lora params and save them.
      fused_lora_params = lora.fuse_lora_params(params)

      fused_lora_params_dir = workdir / f'fused_lora_params_{step_nr}'
      if fused_lora_params_dir.exists():
        logging.info('Skipping save: %s already exists.', fused_lora_params_dir)
      else:
        checkpointer.save(fused_lora_params_dir, fused_lora_params)


class GemmaCheckpointFormatter(CheckpointedEvaluator):
  """Saves Gemma params (and LoRA) in the gemma-diffusion format.

  Overrides ``evaluate`` directly so all work happens outside JIT —
  no tracers, no callbacks needed for file I/O.
  """

  workdir: epath.PathLike = config_util.ROOT_CFG_REF.workdir
  write_original_params: bool = True
  write_lora_params: bool = True
  write_fused_lora_params: bool = True
  gemma_param_path: str = 'gemma_network.gemma_model'

  checkpointer: kd.ckpts.BaseCheckpointer = kd.ckpts.NoopCheckpointer()

  # No model outputs → no losses/metrics/summaries.
  losses: dict[str, kd.losses.Loss] = dataclasses.field(default_factory=dict)
  metrics: dict[str, kd.metrics.Metric] = dataclasses.field(
      default_factory=dict
  )
  summaries: dict[str, kd.metrics.Metric] = dataclasses.field(
      default_factory=dict
  )

  # pytype: disable=bad-return-type
  def evaluate(
      self, *, state: kd.train.TrainState, step: int
  ) -> kd.train.AuxiliariesState:
    """Save checkpoint — runs outside JIT, no tracing issues."""
    gemma_params = kd.kontext.get_by_path(state.params, self.gemma_param_path)
    save_params_into_original_and_lora_params(
        params=gemma_params,
        step_nr=int(state.step),
        workdir=epath.Path(self.workdir) / 'gemma_like_params',
        write_original_params=self.write_original_params,
        write_lora_params=self.write_lora_params,
        write_fused_lora_params=self.write_fused_lora_params,
    )
    return {}

  def __hash__(self) -> int:
    """Make Evaluator hashable, so its methods can be jitted."""
    return id(self)
