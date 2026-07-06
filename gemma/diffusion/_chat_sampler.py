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

"""Diffusion-specific sampler and chat sampler."""

import dataclasses
import functools
from typing import override

from gemma import gm
from gemma.diffusion import _early_stopping
from gemma.diffusion import _sampler
from gemma.gm.text import _sampler_loop


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sampler(gm.text.Sampler):
  """Diffusion variant of `gm.text.Sampler`.

  This class overrides `_initialize_sampler_loop` to create a
  `DiffusionSampler`, which extends `SamplerLoop` with block-wise diffusion
  sampling.

  Attributes:
    diffusion_process: Diffusion process to use. When unset, use the default
      preset.
    logit_shaper: Temperature annealing shaper. When unset, use the default
      preset.
    sample_from_predictions: Sampling strategy for denoised predictions. When
      unset, use the default preset.
    canvas_length: Diffusion canvas length to use. If unset, the model default
      preset is used.
    max_denoising_steps: Maximum number of denoising steps per completed canvas.
      If unset, the model default preset is used.
  """

  diffusion_process: _sampler.DiffusionProcess = dataclasses.field(
      default_factory=_sampler.DiffusionProcess
  )
  logit_shaper: _sampler.AnnealingTemperatureShaper = dataclasses.field(
      default_factory=lambda: _sampler.AnnealingTemperatureShaper(
          config=_sampler.AnnealingTemperatureShaperConfig()
      )
  )
  sample_from_predictions: _sampler.SampleFromPredictions = dataclasses.field(
      default_factory=lambda: _sampler.SampleFromPredictions(
          entropy_bound=0.1,
      )
  )
  early_stop_fn: _early_stopping.EarlyStopFn = dataclasses.field(
      default_factory=lambda: _early_stopping.ChainedEarlyStop(
          early_stop_fns=(
              _early_stopping.TokenStabilityEarlyStop(),
              _early_stopping.EntropyEarlyStop(entropy_threshold=0.005),
          ),
      )
  )

  canvas_length: int = 256
  max_denoising_steps: int = 48

  @override
  def _initialize_sampler_loop(self, sampling) -> _sampler_loop.SamplerLoop:
    """Initializes the sampler loop."""
    # Ensure SampleFromPredictions gets the vocab size.
    sample_from_predictions = self.sample_from_predictions
    if sample_from_predictions.text_vocab_size == 0:
      sample_from_predictions = dataclasses.replace(
          sample_from_predictions,
          text_vocab_size=self.tokenizer.vocab_size,
      )

    return _sampler.DiffusionSampler(
        model=self.model,
        end_tokens=(
            self.tokenizer.special_tokens.EOS,
            self.tokenizer.special_tokens.END_OF_TURN,
            self.tokenizer.special_tokens.BEGIN_OF_TOOL_RESPONSE,
            *self._normalized_stop_tokens,
        ),
        forbidden_tokens=self._normalized_forbidden_tokens,
        sampling=sampling,
        cache_length=self.cache_length,
        special_tokens=self.tokenizer.special_tokens,
        diffusion_process=self.diffusion_process,
        logit_shaper=self.logit_shaper,
        sample_from_predictions=sample_from_predictions,
        canvas_length=self.canvas_length,
        max_denoising_steps=self.max_denoising_steps,
        text_vocab_size=self.tokenizer.vocab_size,
        sliding_window_size=getattr(
            self.model.config, 'sliding_window_size', None
        ),
        early_stop_fn=self.early_stop_fn,
    )


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class ChatSampler(gm.text.ChatSampler):
  """Diffusion equivalent of `gm.text.ChatSampler`.

  Check the docstring of `gm.text.ChatSampler` for usage. The only differences
  are diffusion-specific arguments in the constructor.

  Attributes:
    diffusion_process: Diffusion process to use. When unset, use the default
      preset.
    logit_shaper: Temperature annealing shaper. When unset, use the default
      preset.
    sample_from_predictions: Sampling strategy for denoised predictions. When
      unset, use the default preset.
    canvas_length: Diffusion canvas length to use. If unset, the model default
      preset is used.
    max_denoising_steps: Maximum number of denoising steps per completed canvas.
      If unset, the model default preset is used.
  """

  diffusion_process: _sampler.DiffusionProcess = dataclasses.field(
      default_factory=_sampler.DiffusionProcess
  )
  logit_shaper: _sampler.AnnealingTemperatureShaper = dataclasses.field(
      default_factory=lambda: _sampler.AnnealingTemperatureShaper(
          config=_sampler.AnnealingTemperatureShaperConfig()
      )
  )
  sample_from_predictions: _sampler.SampleFromPredictions = dataclasses.field(
      default_factory=lambda: _sampler.SampleFromPredictions(
          entropy_bound=0.1,
      )
  )
  early_stop_fn: _early_stopping.EarlyStopFn = dataclasses.field(
      default_factory=lambda: _early_stopping.ChainedEarlyStop(
          early_stop_fns=(
              _early_stopping.TokenStabilityEarlyStop(),
              _early_stopping.EntropyEarlyStop(entropy_threshold=0.005),
          ),
      )
  )

  canvas_length: int = 256
  max_denoising_steps: int = 48

  @override
  @functools.cached_property
  def sampler(self) -> Sampler:
    """Returns the underlying sampler."""

    return Sampler(
        model=self.model,
        params=self.params,
        tokenizer=self.tokenizer,
        sampling=self.sampling,
        forbidden_tokens=self.forbidden_tokens,
        stop_tokens=self.stop_tokens,
        cache_length=self.cache_length,  # pyrefly: ignore[bad-argument-type]
        max_out_length=self.max_out_length,
        pad_length=self.pad_length,
        diffusion_process=self.diffusion_process,
        logit_shaper=self.logit_shaper,
        sample_from_predictions=self.sample_from_predictions,
        canvas_length=self.canvas_length,
        max_denoising_steps=self.max_denoising_steps,
        early_stop_fn=self.early_stop_fn,
    )

  @override
  def _sample(
      self,
      prompt_text,
      *,
      images,
      audio,
      audio_lengths,
      sampling,
      max_new_tokens,
      rng,
      last_state,
      stream,
      sharding
  ):
    """Override to always use the diffusion sampler."""
    return self.sampler.sample(  # pytype: disable=wrong-arg-types
        prompt_text,
        images=images,
        sampling=sampling,
        max_new_tokens=max_new_tokens,
        rng=rng,
        return_state=True,
        last_state=last_state,
        stream=bool(stream),
        sharding=sharding,
    )
