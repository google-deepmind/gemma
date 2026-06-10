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

"""AR Diffusion evaluation helpers for DiffusionGemma SFT."""

from typing import Any
from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma import gm
  from gemma.diffusion.hackable_diffusion_adapter.eval import text_metric
  from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_ar_state_handler
  from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model
  from hackable_diffusion import hd
  import jax.numpy as jnp
  from kauldron import kd

# Denoising step counts for AR Diffusion evaluation.
AR_DENOISING_STEPS = [32, 64, 96]


def make_ar_evals(
    cfg,
    gemma_network_ref,
    corruption_process_ref,
    canvas_size_ref,
    metrics: dict[str, Any] | None = None,
    max_num_canvases: int = 8,
    denoising_steps: list[int] | None = None,
    use_early_stopping: bool = False,
) -> dict[str, Any]:
  """Return one GemmaSamplingEvaluator per denoising step count.

  Args:
    cfg: The Kauldron trainer config (needed for cfg.ref references).
    gemma_network_ref: Reference to the Gemma network module.
    corruption_process_ref: cfg.ref pointing to the corruption process.
    canvas_size_ref: cfg.ref pointing to the canvas size.
    metrics: Dict of metric objects.  Defaults to ``{}`` (no metrics).
    max_num_canvases: Maximum number of AR canvases to generate.
    denoising_steps: List of step counts.  Defaults to AR_DENOISING_STEPS.
    use_early_stopping: Whether to use early stopping based on entropy.

  Returns:
    Dict mapping ``"sample_ar_steps{n}"`` (or
    ``"sample_ar_steps{n}_early_stopping"``)
    to a GemmaSamplingEvaluator.
  """
  if metrics is None:
    metrics = {}
  if denoising_steps is None:
    denoising_steps = AR_DENOISING_STEPS

  def _make_one(num_steps: int):
    if use_early_stopping:
      canvas_sampler = hd.sampling.DiffusionSamplerWithEarlyStopping(
          time_schedule=hd.sampling.UniformTimeSchedule(),
          stepper=hd.sampling.DiscreteDDIMStep(
              corruption_process=corruption_process_ref,
              temperature=0.7,
              logits_dtype=jnp.bfloat16,
          ),
          update_conditioning_fn=hd_gemma_ar_state_handler.PropagateSelfConditioningFn(),
          num_steps=num_steps,
          early_stopping_fn=hd.sampling.DiffusionEntropyEarlyStopFn(),
      )
    else:
      canvas_sampler = hd.sampling.DiffusionSampler(
          time_schedule=hd.sampling.UniformTimeSchedule(),
          stepper=hd.sampling.DiscreteDDIMStep(
              corruption_process=corruption_process_ref,
              temperature=0.7,
              logits_dtype=jnp.bfloat16,
          ),
          update_conditioning_fn=hd_gemma_ar_state_handler.PropagateSelfConditioningFn(),
          num_steps=num_steps,
          store_trajectory=False,
      )

    return sft_model.GemmaSamplingEvaluator(
        run=kd.evals.StandaloneEveryCheckpoint(),
        ar_diffusion_sampler=sft_model.GemmaKDARSampler(
            state_handler=hd_gemma_ar_state_handler.GemmaARStateHandler(
                gemma_network=gemma_network_ref,
                # Loaded at runtime from the checkpoint.
                gemma_params=None,
                pad_token=gm.text.Gemma4Tokenizer.special_tokens.PAD,
                end_tokens=(
                    gm.text.Gemma4Tokenizer.special_tokens.EOS,
                    gm.text.Gemma4Tokenizer.special_tokens.END_OF_TURN,
                    gm.text.Gemma4Tokenizer.special_tokens.BEGIN_OF_TOOL_RESPONSE,
                ),
            ),
            canvas_sampler=canvas_sampler,
            diffusion_process=corruption_process_ref,
            canvas_length=canvas_size_ref,
            max_num_canvases=max_num_canvases,
            data_dtype=jnp.int32,
            data_shape=(1,),
        ),
        num_batches=cfg.ref.aux.eval_num_batches,
        metrics=metrics,
        summaries={
            "text_samples_ar": text_metric.DetokenizePromptAndResponse(
                prompt="batch.prompt",
                response="samples",
                num_texts=10,
            ),
        },
    )

  if use_early_stopping:
    return {
        f"sample_ar_steps{n}_early_stopping": _make_one(n)
        for n in denoising_steps
    }
  else:
    return {f"sample_ar_steps{n}": _make_one(n) for n in denoising_steps}
