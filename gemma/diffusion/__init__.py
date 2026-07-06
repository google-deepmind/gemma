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

"""Sampling for DiffusionGemma."""

# pylint: disable=g-importing-member,g-import-not-at-top

from etils import epy as _epy


with _epy.lazy_api_imports(globals()):
  # Models
  from gemma.diffusion._models import DiffusionGemma_26B_A4B

  # Checkpoint paths
  from gemma.diffusion._paths import CheckpointPath

  # Samplers (public interface)
  from gemma.diffusion._chat_sampler import ChatSampler
  from gemma.diffusion._chat_sampler import Sampler

  # Diffusion process components
  from gemma.diffusion._sampler import DiffusionProcess
  from gemma.diffusion._sampler import LinearSchedule
  from gemma.diffusion._sampler import SampleFromPredictions

  # Temperature shaping
  from gemma.diffusion._sampler import AnnealingTemperatureShaper
  from gemma.diffusion._sampler import AnnealingTemperatureShaperConfig

  # Transformer components
  from gemma.diffusion._transformer import DiffusionMixin
  from gemma.diffusion._transformer import SelfConditioning
  from gemma.diffusion._transformer import SelfConditioningConfig

  # Early stopping strategies
  from gemma.diffusion._early_stopping import EarlyStopFn
  from gemma.diffusion._early_stopping import NoEarlyStop
  from gemma.diffusion._early_stopping import TokenStabilityEarlyStop
  from gemma.diffusion._early_stopping import EntropyEarlyStop
  from gemma.diffusion._early_stopping import ChainedEarlyStop
