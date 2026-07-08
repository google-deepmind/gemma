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

"""Evaluator that can be checkpointed.

Vendored from kauldron.contrib.evals.checkpointed_evaluator with patches for
checkpoint stability.
"""

from __future__ import annotations

import dataclasses
import typing

from etils import epath
import flax
import jax
from jax.experimental import checkify
from kauldron import checkpoints
from kauldron.evals import evaluators
from kauldron.train import auxiliaries
from kauldron.train import train_step
from kauldron.train import trainer_lib
from kauldron.utils import utils
from kauldron.utils.sharding_utils import sharding as sharding_lib  # pylint: disable=g-importing-member

# Patches for checkpointing stability

# At checkpoint steps, JAX may still be executing the training graph
# asynchronously.  Orbax's transfer_arrays_to_host triggers ncclAllGather
# to fetch sharded arrays, colliding with in-flight NCCL ops and causing
# "invalid argument".  Additionally, replica-parallel serialization and
# pinned host transfers cause CUDA stream deadlocks on NVLink.
#
# Fix: wrap transfer_arrays_to_host to (1) barrier-sync the GPU first,
# and (2) force single-replica, non-pinned host transfers.
try:
  import orbax.checkpoint._src.serialization.replica_slices as _rs
  _orig_transfer = _rs.transfer_arrays_to_host
  def _safe_transfer(
      arrays, replica_id, use_replica_parallel=False, *args, **kwargs
  ):
    kwargs["enable_pinned_host_transfer"] = False
    return _orig_transfer(arrays, replica_id, False, *args, **kwargs)
  _rs.transfer_arrays_to_host = _safe_transfer
except Exception:
  pass

# Orbax path finalization for GCS checkpoints
try:
  import orbax.checkpoint as ocp
  ocp._src.path.step.is_path_finalized = lambda *args, **kwargs: True
except Exception:
  pass


@flax.struct.dataclass
class EvalState(checkpoints.items.StandardCheckpointItem):
  """State of the CheckpointedEvaluator."""

  merged_aux: auxiliaries.AuxiliariesState
  step_nr: int


class _EvalCheckpointerState(typing.NamedTuple):
  """State of the CheckpointedEvaluator."""

  eval_state: EvalState
  ds_iter: typing.Iterator[typing.Any]
  # `eval_state` is saved as the default name
  DEFAULT_ITEM = "eval_state"


# this is necessary because NamedTuple does not support double inheritance.
class EvalCheckpointerState(
    _EvalCheckpointerState, checkpoints.items.TopLevelCheckpointItem
):
  """State of the CheckpointedEvaluator."""

  pass


class CheckpointedEvaluator(evaluators.Evaluator):
  """An evaluator that can save and restore its progress."""

  checkpointer: checkpoints.checkpointer.BaseCheckpointer

  def __post_init__(self):
    super().__post_init__()
    if isinstance(self.base_cfg, trainer_lib.Trainer):
      # check if self.checkpointer exists, has attr workdir, and if the workdir
      # is None, then set it to the eval workdir.
      if hasattr(self, "checkpointer") and hasattr(
          self.checkpointer, "workdir"
      ):
        workdir = epath.Path(self.base_cfg.workdir)
        # check the workdir for the checkpointer has not been changed.
        if self.checkpointer.workdir == workdir:
          eval_workdir = workdir / "evals" / self.name
          new_checkpointer = dataclasses.replace(
              self.checkpointer, workdir=eval_workdir  # pyrefly: ignore[unexpected-keyword]
          )
          object.__setattr__(self, "checkpointer", new_checkpointer)
    if self.cache:
      raise TypeError("CheckpointedEvaluator does not support caching, yet.")

  def _get_init_aux_state(
      self, state: train_step.TrainState
  ) -> auxiliaries.AuxiliariesState:
    """Get the initial aux state from the first batch."""
    batch = next(iter(self.ds))
    batch = sharding_lib.device_put(batch, self.base_cfg.sharding.batch)
    step_nr_jax = sharding_lib.device_put(0, sharding_lib.REPLICATED)
    return self.step(step_nr=step_nr_jax, state=state, batch=batch).finalize()

  def evaluate(
      self, state: train_step.TrainState, step: int
  ) -> auxiliaries.AuxiliariesState:
    """Run one full evaluation with checkpointing."""
    self._assert_root_cfg_resolved()
    if self.discard_opt:
      state = state.replace(opt_state=None)
    state = self.init_transform.transform(state)

    # MARK: Load state
    step_nr = 0
    aux_state = self._get_init_aux_state(state)
    ds_iter = iter(self.ds)

    initial_eval_state = EvalCheckpointerState(
        eval_state=EvalState(merged_aux=aux_state, step_nr=step_nr),
        ds_iter=ds_iter,
    )

    (eval_state, ds_iter) = self.checkpointer.restore(
        initial_eval_state,
        noop_if_missing=True,
    )

    latest_eval_step = eval_state.step_nr
    merged_aux = eval_state.merged_aux

    if latest_eval_step == 0:
      merged_aux = None

    # steps are 1-indexed.
    try:
      total_steps = len(self.ds) + 1
    except TypeError:  # Unknown length.
      total_steps = None

    # MARK: Run evaluation.
    for step_nr, batch in utils.enum_iter(
        ds_iter,
        init_step=latest_eval_step + 1,
        total_steps=total_steps,
        desc=self.name,
    ):
      if self.num_batches is not None:
        if step_nr > self.num_batches:
          break
      step_nr_jax = sharding_lib.device_put(step_nr, sharding_lib.REPLICATED)
      batch = sharding_lib.device_put(batch, self.base_cfg.sharding.batch)
      aux_state = self.step(
          step_nr=step_nr_jax,
          state=state,
          batch=batch,
      )

      with jax.transfer_guard("allow"):
        if aux_state.error is not None:
          checkify.check_error(aux_state.error)
        # Merge/accumulate all states
        merged_aux = merged_aux | aux_state

        # Save checkpoint.
        if self.checkpointer.should_save(step_nr):
          state_to_save = EvalCheckpointerState(
              eval_state=EvalState(
                  merged_aux=merged_aux.finalize(),
                  step_nr=step_nr,
              ),
              ds_iter=ds_iter,
          )
          self.checkpointer.save(
              state_to_save,
              step=step_nr,
          )

    if merged_aux is None:
      raise ValueError(
          f"Dataset for eval {self.name!r} did not yield any elements."
      )

    self.writer.write_step_metrics(
        step=step,
        aux=merged_aux,
        schedules={},
        log_summaries=True,
    )

    # Wait for the last checkpoint to be saved before completing the evaluation.
    self.checkpointer.wait_until_finished()
    return merged_aux




