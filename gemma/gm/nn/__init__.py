# Copyright 2025 DeepMind Technologies Limited.
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

"""Gemma models."""

# pylint: disable=g-importing-member,g-import-not-at-top

from etils import epy as _epy


with _epy.lazy_api_imports(globals()):
  # ****************************************************************************
  # Gemma models
  # ****************************************************************************
  # Gemma 2
  from gemma.gm.nn._gemma import Gemma2_2B
  from gemma.gm.nn._gemma import Gemma2_9B
  from gemma.gm.nn._gemma import Gemma2_27B
  # Gemma 3
  from gemma.gm.nn._gemma import Gemma3_270M
  from gemma.gm.nn._gemma import Gemma3_1B
  from gemma.gm.nn._gemma import Gemma3_4B
  from gemma.gm.nn._gemma import Gemma3_12B
  from gemma.gm.nn._gemma import Gemma3_27B
  # Gemma 3n
  from gemma.gm.nn.gemma3n._gemma3n import Gemma3n_E2B
  from gemma.gm.nn.gemma3n._gemma3n import Gemma3n_E4B

  # ****************************************************************************
  # Wrapper (LoRA, quantization, DPO,...)
  # ****************************************************************************
  from gemma.gm.nn._lora import LoRA
  from gemma.gm.nn._quantization import QuantizationAwareWrapper
  from gemma.gm.nn._quantization import IntWrapper
  from gemma.gm.nn._policy import AnchoredPolicy
  from gemma.gm.nn._transformer import Transformer
  from gemma.gm.nn._transformer_like import TransformerLike

  # ****************************************************************************
  # Transformer building blocks
  # ****************************************************************************
  # Allow users to create their own transformers.
  # TODO(epot): Also expose the Vision encoder model as standalone.
  from gemma.gm.nn._layers import Einsum
  from gemma.gm.nn._layers import RMSNorm
  from gemma.gm.nn._modules import Embedder
  from gemma.gm.nn._modules import Attention
  from gemma.gm.nn._modules import Block
  from gemma.gm.nn._modules import FeedForward
  from gemma.gm.nn._modules import AttentionType

  # Model inputs
  from gemma.gm.nn._config import Cache

  # Model outputs
  from gemma.gm.nn._transformer import Output
  from gemma.gm.nn._policy import AnchoredPolicyOutput

  from gemma.gm.nn import config
