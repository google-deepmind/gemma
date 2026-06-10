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

"""DiffusionGemma SFT config — Sudoku dataset.

Flat configuration that contains all setup (model, optimizer,
checkpointer, dataset pipeline, and evaluation).
"""

from gemma.diffusion.hackable_diffusion_adapter.eval import ar_eval
from kauldron import konfig

import functools
import jax
from flax import linen as nn
from gemma.gm.nn.gemma4 import _modules

# Gradient checkpointing.
# nn.remat requires only positional args so we use a wrapper.
_orig_block = _modules.Block
_orig_call = _orig_block.__call__

@functools.partial(
    nn.remat,
    policy=jax.checkpoint_policies.nothing_saveable,
    static_argnums=7,
)
def rematted_call_fn(
    self,
    x,
    segment_pos,
    cache,
    attn_mask,
    per_layer_input,
    kv_shared_cache,
    skip_sliding_mask,
):
  return _orig_call(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
      per_layer_input=per_layer_input,
      kv_shared_cache=kv_shared_cache,
      skip_sliding_mask=skip_sliding_mask,
  )

def new_call(
    self,
    x,
    segment_pos,
    cache,
    attn_mask,
    per_layer_input=None,
    kv_shared_cache=None,
    skip_sliding_mask=False,
):
  return rematted_call_fn(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
      per_layer_input,
      kv_shared_cache,
      skip_sliding_mask,
  )

_modules.Block.__call__ = new_call

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma.diffusion import _models
  from gemma.diffusion import _paths  # pytlint: disable=unused-import
  from gemma.diffusion.hackable_diffusion_adapter.data.sudoku import sudoku_data
  from gemma.diffusion.hackable_diffusion_adapter.eval import sudoku_eval
  from gemma.diffusion.hackable_diffusion_adapter.hd import gemma_checkpointer
  from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
  from gemma.diffusion.hackable_diffusion_adapter.hd import lora
  from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model
  from hackable_diffusion import hd
  from hackable_diffusion.kdiff import core
  from hackable_diffusion.lib.training import discrete_loss
  from kauldron import kd
  import optax
  from gemma.diffusion.hackable_diffusion_adapter import safe_writer
# pylint: enable=g-import-not-at-top

CHECKPOINT_PATH = _paths.CheckpointPath.DIFFUSIONGEMMA_26B_A4B_IT


def get_config():
  """SFT config for Sudoku solving."""
  cfg = kd.train.Trainer()
  cfg.seed = 42
  cfg.aux = {}
  cfg.aux.vocab_size = 262_144  # Gemma4 vocabulary size

  cfg.aux.corruption_process = hd.corruption.CategoricalProcess.uniform_process(
      num_categories=cfg.ref.aux.vocab_size,
      schedule=hd.corruption.RFSchedule(),
  )
  cfg.aux.prompt_len = 256
  cfg.aux.num_canvases = 1
  cfg.aux.canvas_size = 256
  use_lora = False
  lora_rank = 8
  cfg.aux.use_lora = use_lora
  cfg.aux.lora_rank = lora_rank
  cfg.aux.peak_lr = 1.5e-4
  cfg.aux.end_lr = cfg.ref.aux.peak_lr / 10
  cfg.aux.checkpoint_every_n_steps = 1000
  cfg.aux.eval_num_batches = None
  cfg.aux.stop_gradient_from_denoiser_to_encoder = False
  cfg.aux.encoder_loss_weight = 1.0
  cfg.aux.decoder_loss_weight = 1.0

  cfg.aux.sudoku_prompt = (
      "<|turn>system Solve the following Sudoku puzzle. Empty cells are"
      " represented by 0. Output ONLY the solved puzzle immediately as"
      " a 9x9 grid of numbers separated by spaces. Do not include ####,"
      " explanations, or any other text.<turn|>\n<|turn>user"
      " {text}<turn|>\n<|turn>model\n"
  )

  cfg.sharding = kd.sharding.ShardingStrategy(
      params=kd.sharding.FSDPSharding(), opt_state=kd.sharding.FSDPSharding()
  )

  base_network = hd_gemma_network.WrappedDiffusionGemmaNetwork(
      gemma_model=_models.DiffusionGemma_A26B_A4B(),
  )
  if use_lora:
    gemma_network = lora.LoRA(
        rank=lora_rank,
        model=base_network,
        target_modules="all-linear",
    )
  else:
    gemma_network = base_network

  cfg.model = sft_model.SFTDiffusion(
      x0="batch.canvas",
      prompt="batch.prompt",
      canvas_id="batch.canvas_id",
      canvas_mask="batch.canvas_mask",
      encoder_target="batch.encoder_target",
      encoder_target_mask="batch.encoder_target_mask",
      corruption_process=cfg.ref.aux.corruption_process,
      time_sampler=hd.training.time_sampling.UniformTimeSampler(
          span=hd.jax_helpers.SafeSpan(safety_epsilon=1e-4)
      ),
      gemma_network=gemma_network,
      prompt_len=cfg.ref.aux.prompt_len,
      canvas_size=cfg.ref.aux.canvas_size,
      num_canvases=cfg.ref.aux.num_canvases,
      stop_gradient_from_denoiser_to_encoder=cfg.ref.aux.stop_gradient_from_denoiser_to_encoder,
  )

  cfg.train_losses = {
      "diffusion_loss": core.KauldronLossWrapper(
          loss=discrete_loss.NoWeightDiscreteLoss(
              use_mask=True,
              mask_key="target_mask",
          ),
          weight=cfg.ref.aux.decoder_loss_weight,
      ),
      "encoder_loss": sft_model.EncoderARLoss(
          encoder_logits="preds.encoder_logits",
          encoder_target="preds.encoder_target",
          encoder_target_mask="preds.encoder_target_mask",
          weight=cfg.ref.aux.encoder_loss_weight,
      ),
  }

  cfg.num_train_steps = 2_000

  cfg.schedules = {
      "learning_rate": optax.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=cfg.ref.aux.peak_lr,
          end_value=cfg.ref.aux.end_lr,
          warmup_steps=1000,
          decay_steps=cfg.ref.num_train_steps,
      ),
  }

  cfg.optimizer = kd.optim.named_chain(**{
      "clip": optax.clip_by_global_norm(max_norm=1.0),
      "adafactor": optax.scale_by_factored_rms(),
      "decay": optax.add_decayed_weights(weight_decay=1e-4),
      "lr": optax.scale_by_learning_rate(cfg.ref.schedules["learning_rate"]),
  })

  cfg.checkpointer = kd.ckpts.Checkpointer(
      fast=False,
      save_interval_steps=cfg.ref.aux.checkpoint_every_n_steps,
      max_to_keep=5,
  )

  cfg._konfig_experimental_nofreeze = True  # pylint: disable=protected-access
  cfg.rng_streams = kd.train.RngStreams([
      kd.train.RngStream("default", train=True, eval=True),
      kd.train.RngStream("sampling", train=True, eval=True),
  ])

  cfg.init_transform = gemma_checkpointer.GemmaDiffusionCheckpointLoader(
      path=CHECKPOINT_PATH,
  )

  cfg.train_ds = sudoku_data.make_sudoku_ds(
      bagz_path="gemma/diffusion/hackable_diffusion_adapter/data/sudoku/sudoku_train.bagz",
      training=True,
      batch_size=8,
      prompt_len=cfg.ref.aux.prompt_len,
      num_canvases=cfg.ref.aux.num_canvases,
      canvas_size=cfg.ref.aux.canvas_size,
      prompt_template=cfg.ref.aux.sudoku_prompt,
  )
  cfg.eval_ds = sudoku_data.make_sudoku_ds(
      bagz_path="gemma/diffusion/hackable_diffusion_adapter/data/sudoku/sudoku_eval.bagz",
      training=False,
      batch_size=8,
      prompt_len=cfg.ref.aux.prompt_len,
      num_canvases=cfg.ref.aux.num_canvases,
      canvas_size=cfg.ref.aux.canvas_size,
      slice_stop=256,
      prompt_template=cfg.ref.aux.sudoku_prompt,
  )

  eval_metrics = {
      "sudoku": sudoku_eval.SudokuAllMetrics(
          tokens="samples",
          ground_truth="batch.solution_tokens",
          puzzle="batch.puzzle_tokens",
          extraction_mode=sudoku_eval.ExtractionMode.THINKING,
      ),
  }
  eval_metrics.update({
      "processed_denoising_steps": kd.metrics.SingleDimension(
          tensor="processed_denoising_steps", index=None
      ),
      "processed_num_canvases": kd.metrics.SingleDimension(
          tensor="processed_num_canvases", index=None
      ),
      "average_denoising_steps_per_canvas": kd.metrics.SingleDimension(
          tensor="average_denoising_steps_per_canvas", index=None
      ),
  })

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
  if use_lora:
    # We only save the fused parameters.
    checkpointer_kwargs = {
        "write_original_params": False,
        "write_lora_params": False,
        "write_fused_lora_params": True,
    }
  else:
    # We only save the original parameters.
    checkpointer_kwargs = {
        "write_original_params": True,
        "write_lora_params": False,
        "write_fused_lora_params": False,
    }
  cfg.evals = {
      "gemma_checkpointer": gemma_checkpointer.GemmaCheckpointFormatter(
          run=kd.evals.StandaloneEveryCheckpoint(),
          **checkpointer_kwargs,
      ),
  }

  cfg.writer = safe_writer.SafeMetricWriter()

  return cfg
