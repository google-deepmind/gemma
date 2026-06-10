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

"""Text diffusion SFT config — PubMedQA long-answer (detailed explanation).

Flat configuration that contains all setup (model, LoRA, optimizer,
checkpointer, dataset pipeline, and evaluation).
"""

import dataclasses
from gemma.diffusion.hackable_diffusion_adapter.eval import ar_eval
from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma.diffusion import _models
  from gemma.diffusion import _paths  # pytlint: disable=unused-import
  from gemma.diffusion.hackable_diffusion_adapter.data.pubmedqa import pubmedqa_data
  from gemma.diffusion.hackable_diffusion_adapter.eval import pubmedqa_eval
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

CHECKPOINT_PATH = _paths.DIFFUSIONGEMMA_26B_A4B_IT

# Paths to the converted PubMedQA JSONL files.
PUBMEDQA_TRAIN_PATH = "gemma/diffusion/hackable_diffusion_adapter/data/pubmedqa/pubmedqa_train.jsonl"
PUBMEDQA_TEST_PATH = "gemma/diffusion/hackable_diffusion_adapter/data/pubmedqa/pubmedqa_test.jsonl"

# Reserve the first 50 training examples for overfitting tracking.
_TRAIN_EVAL_SIZE = 50

# Hyperparameters.
_LORA_RANK = 4
_PEAK_LR = 1e-4
_END_LR = 1e-5
_WARMUP_STEPS = 100


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigArgs:
  use_early_stopping: bool = True


def get_config(args: ConfigArgs = ConfigArgs()):
  """SFT config for PubMedQA long-answer (detailed explanation)."""
  use_early_stopping = args.use_early_stopping

  cfg = kd.train.Trainer()
  cfg.seed = 42
  cfg.aux = {}
  cfg.aux.vocab_size = 262_144  # Gemma4 vocabulary size

  cfg.aux.corruption_process = hd.corruption.CategoricalProcess.uniform_process(
      num_categories=cfg.ref.aux.vocab_size,
      schedule=hd.corruption.RFSchedule(),
  )
  cfg.aux.prompt_len = 1024
  cfg.aux.num_canvases = 2
  cfg.aux.canvas_size = 128
  use_lora = True
  lora_rank = _LORA_RANK
  cfg.aux.use_lora = use_lora
  cfg.aux.lora_rank = lora_rank
  cfg.aux.peak_lr = _PEAK_LR
  cfg.aux.end_lr = _END_LR
  cfg.aux.checkpoint_every_n_steps = 1_000
  cfg.aux.eval_num_batches = None
  cfg.aux.stop_gradient_from_denoiser_to_encoder = False
  cfg.aux.encoder_loss_weight = 1.0
  cfg.aux.decoder_loss_weight = 1.0

  cfg.sharding = kd.sharding.ShardingStrategy(
      params=kd.sharding.FSDPSharding(), opt_state=kd.sharding.FSDPSharding()
  )

  base_network = hd_gemma_network.WrappedDiffusionGemmaNetwork(
      gemma_model=_models.DiffusionGemma_26B_A4B(),
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
          warmup_steps=_WARMUP_STEPS,
          decay_steps=cfg.ref.num_train_steps,
      ),
  }

  _base_optimizer = kd.optim.named_chain(**{
      "clip": optax.clip_by_global_norm(max_norm=1.0),
      "adam": optax.scale_by_adam(b1=0.95, b2=0.99, eps=1e-8),
      "decay": optax.add_decayed_weights(weight_decay=1e-4),
      "lr": optax.scale_by_learning_rate(cfg.ref.schedules["learning_rate"]),
  })

  if cfg.aux.use_lora:
    cfg.optimizer = kd.optim.partial_updates(
        optimizer=_base_optimizer,
        mask=kd.optim.select("lora"),
    )
  else:
    cfg.optimizer = _base_optimizer

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

  cfg.train_ds = pubmedqa_data.make_pubmedqa_ds(
      training=True,
      batch_size=2,
      prompt_len=cfg.ref.aux.prompt_len,
      num_canvases=cfg.ref.aux.num_canvases,
      canvas_size=cfg.ref.aux.canvas_size,
      use_long_answer=True,
      train_path=PUBMEDQA_TRAIN_PATH,
      test_path=PUBMEDQA_TEST_PATH,
      slice_start=_TRAIN_EVAL_SIZE,
      num_workers=0,
  )
  cfg.eval_ds = pubmedqa_data.make_pubmedqa_ds(
      training=False,
      batch_size=2,
      prompt_len=cfg.ref.aux.prompt_len,
      num_canvases=cfg.ref.aux.num_canvases,
      canvas_size=cfg.ref.aux.canvas_size,
      use_long_answer=True,
      train_path=PUBMEDQA_TRAIN_PATH,
      test_path=PUBMEDQA_TEST_PATH,
      num_workers=0,
  )
  cfg.aux.eval_train_ds = pubmedqa_data.make_pubmedqa_ds(
      training=False,
      batch_size=4,
      prompt_len=cfg.ref.aux.prompt_len,
      num_canvases=cfg.ref.aux.num_canvases,
      canvas_size=cfg.ref.aux.canvas_size,
      use_long_answer=True,
      train_path=PUBMEDQA_TRAIN_PATH,
      test_path=PUBMEDQA_TRAIN_PATH,
      slice_start=0,
      slice_stop=_TRAIN_EVAL_SIZE,
  )

  eval_metrics = {
      "pubmedqa_accuracy": pubmedqa_eval.PubMedQAAccuracy(
          tokens="samples",
          ground_truth="batch.short_answer_tokens",
      ),
      "pubmedqa_bleu": pubmedqa_eval.BLEUScore(
          tokens="samples",
          ground_truth="batch.long_answer_tokens",
      ),
      "processed_denoising_steps": kd.metrics.SingleDimension(
          tensor="processed_denoising_steps", index=None
      ),
      "processed_num_canvases": kd.metrics.SingleDimension(
          tensor="processed_num_canvases", index=None
      ),
      "average_denoising_steps_per_canvas": kd.metrics.SingleDimension(
          tensor="average_denoising_steps_per_canvas", index=None
      ),
  }

  test_evals = {
      f"test_{k}": v
      for k, v in ar_eval.make_ar_evals(
          cfg,
          gemma_network_ref=cfg.ref.model.gemma_network,
          corruption_process_ref=cfg.ref.aux.corruption_process,
          canvas_size_ref=cfg.ref.aux.canvas_size,
          metrics=eval_metrics,
          max_num_canvases=cfg.aux.num_canvases,
          use_early_stopping=use_early_stopping,
      ).items()
  }

  train_evals = {}
  for k, v in ar_eval.make_ar_evals(
      cfg,
      gemma_network_ref=cfg.ref.model.gemma_network,
      corruption_process_ref=cfg.ref.aux.corruption_process,
      canvas_size_ref=cfg.ref.aux.canvas_size,
      metrics=eval_metrics,
      max_num_canvases=cfg.aux.num_canvases,
      use_early_stopping=use_early_stopping,
  ).items():
    v.ds = cfg.ref.aux.eval_train_ds
    train_evals[f"train_{k}"] = v

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
