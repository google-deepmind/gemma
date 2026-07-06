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

r"""Evaluation binary for Hackable Diffusion adapter.

Runs AR sampling evaluation on a single checkpoint and reports task-specific
metrics (accuracy, BLEU, etc.) plus text sample summaries.
"""

from __future__ import annotations

from absl import app
from absl import flags
from absl import logging
from gemma.diffusion.hackable_diffusion_adapter.eval import ar_eval
import jax
from kauldron import kd
import tensorflow as tf

# pylint: disable=g-import-not-at-top
with kd.konfig.imports():
  from kauldron import kd as kd_cfg  # pylint: disable=reimported
  from gemma.diffusion.hackable_diffusion_adapter.eval import sudoku_eval as sudoku_eval_cfg
  from gemma.diffusion.hackable_diffusion_adapter.eval import pubmedqa_eval as pubmedqa_eval_cfg
# pylint: enable=g-import-not-at-top

################################################################################
# MARK: Flags
################################################################################

_CONFIG = kd.konfig.DEFINE_config_file(
    "cfg",
    None,
    "Training configuration.",
    lock_config=False,
)

_TASK = flags.DEFINE_enum(
    "task",
    None,
    ["sudoku", "pubmedqa"],
    "Task to evaluate.  Determines which metrics are reported.",
)

_EVAL_NAMES = flags.DEFINE_list(
    "eval_names",
    None,
    "Evaluation(s) to run.  If not set, runs all evaluators.",
)

_STEP = flags.DEFINE_integer(
    "step",
    None,
    "The checkpoint step to evaluate.  If not set, restores the latest"
    " checkpoint.",
)

################################################################################
# MARK: Metrics
################################################################################


def _denoising_metrics():
  """Return metrics common to every task (denoising step counts)."""
  return {
      "processed_denoising_steps": kd_cfg.metrics.SingleDimension(
          tensor="processed_denoising_steps", index=None
      ),
      "processed_num_canvases": kd_cfg.metrics.SingleDimension(
          tensor="processed_num_canvases", index=None
      ),
      "average_denoising_steps_per_canvas": kd_cfg.metrics.SingleDimension(
          tensor="average_denoising_steps_per_canvas", index=None
      ),
  }


def _sudoku_metrics():
  """Return task-specific metrics for the Sudoku config."""
  return {
      "sudoku": sudoku_eval_cfg.SudokuAllMetrics(
          tokens="samples",
          ground_truth="batch.solution_tokens",
          puzzle="batch.puzzle_tokens",
          extraction_mode=sudoku_eval_cfg.ExtractionMode.SFT,
      ),
  }


def _pubmedqa_metrics():
  """Return task-specific metrics for the PubMedQA config."""
  return {
      "pubmedqa_accuracy": pubmedqa_eval_cfg.PubMedQAAccuracy(
          tokens="samples",
          ground_truth="batch.short_answer_tokens",
      ),
      "pubmedqa_bleu": pubmedqa_eval_cfg.BLEUScore(
          tokens="samples",
          ground_truth="batch.long_answer_tokens",
      ),
  }


_TASK_METRICS = {
    "sudoku": _sudoku_metrics,
    "pubmedqa": _pubmedqa_metrics,
}


################################################################################
# MARK: Eval injection
################################################################################


def _inject_ar_evals(cfg, task):
  """Build AR evaluators with task-specific metrics and add them to ``cfg``.

  Args:
    cfg: The (unresolved) Kauldron trainer config.
    task: Task name (``"sudoku"`` or ``"pubmedqa"``).
  """
  eval_metrics = _denoising_metrics()
  if task_metrics_fn := _TASK_METRICS.get(task):
    eval_metrics.update(task_metrics_fn())

  all_evals = ar_eval.make_ar_evals(
      cfg,
      gemma_network_ref=cfg.ref.model.gemma_network,
      corruption_process_ref=cfg.ref.aux.corruption_process,
      canvas_size_ref=cfg.ref.aux.canvas_size,
      metrics=eval_metrics,
      max_num_canvases=cfg.ref.aux.num_canvases,
      use_early_stopping=False,
  )
  early_stopping_evals = ar_eval.make_ar_evals(
      cfg,
      gemma_network_ref=cfg.ref.model.gemma_network,
      corruption_process_ref=cfg.ref.aux.corruption_process,
      canvas_size_ref=cfg.ref.aux.canvas_size,
      metrics=eval_metrics,
      max_num_canvases=cfg.ref.aux.num_canvases,
      use_early_stopping=True,
  )
  all_evals.update(early_stopping_evals)
  cfg.evals.update(all_evals)


################################################################################
# MARK: Main
################################################################################


def main(_):
  """Run single-checkpoint evaluation."""
  # Early JAX initialization to secure GPU contexts and prevent NCCL corruption.
  jax.devices()

  # Hide GPUs from TensorFlow to avoid memory conflicts with JAX.
  tf.config.set_visible_devices([], "GPU")

  cfg = _CONFIG.value
  task = _TASK.value
  eval_names = _EVAL_NAMES.value
  step = _STEP.value
  _inject_ar_evals(cfg, task)

  if hasattr(cfg, "init_transform"):
    cfg.init_transform = None  # pyrefly: ignore[missing-attribute]

  trainer: kd.train.Trainer = kd.konfig.resolve(cfg)  # pyrefly: ignore[bad-assignment]

  logging.info("Initializing state...")
  state = trainer.init_state()

  if step is None:
    logging.info("Restoring latest checkpoint...")
    state = trainer.checkpointer.restore(state)
    step = int(state.step)
  else:
    logging.info("Restoring checkpoint at step %d...", step)
    state = trainer.checkpointer.restore(state, step=step)

  available_evals = [
      name for name in trainer.evals.keys() if name != "gemma_checkpointer"
  ]
  if not eval_names:
    eval_names = available_evals

  logging.info("Running evaluation(s) %s for step %d...", eval_names, step)
  for name in eval_names:
    if name not in trainer.evals:
      raise ValueError(
          f"Evaluator {name!r} not found. Available: {available_evals}"
      )
    evaluator = trainer.evals[name]
    evaluator.evaluate(state, step=step)

  logging.info("All evaluations done!")


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
