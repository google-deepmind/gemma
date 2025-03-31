# Copyright 2024 DeepMind Technologies Limited.
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

"""Legacy sampler for Gemma transformer.

This file is maintained for backward compatibility only. New code should use the
gemma.gm.text module instead.
"""

import dataclasses
from gemma import transformer as transformer_lib
from gemma import params as params_lib
import sentencepiece as spm


@dataclasses.dataclass
class SamplerOutput:
  """Output from a sampling operation.

  Attributes:
    text: Decoded samples from the model.
    logits: Per-step logits used during sampling.
    tokens: Tokens corresponding to the generated samples.
  """

  # Decoded samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  logits: list[list[float]]

  # Tokens corresponding to the generated samples.
  tokens: list[list[int]]


class Sampler:
  """Legacy sampler for Gemma transformer.

  This class is deprecated and should not be used. Instead, use `gm.text.Sampler` or
  `gm.text.ChatSampler` for a more robust implementation with multimodal support.
  """

  def __init__(
      self,
      transformer: transformer_lib.Transformer,
      vocab: spm.SentencePieceProcessor,
      params: params_lib.Params,
      *,
      cache_length: int | None = None,
  ):
    """Initializes a sampler for a Gemma model.

    Args:
      transformer: an instance of the Gemma transformer.
      vocab: vocabulary of the given model.
      params: weights of the model.
      cache_length: Max length of the cache.

    Raises:
      DeprecationWarning: Always raised as this class is deprecated.
    """
    raise DeprecationWarning(
        "The old sampler is deprecated and will be removed in a future release. "
        "It behaves unexpectedly and doesn't support multimodal inputs. "
        "Instead, use `gm.text.Sampler` or `gm.text.ChatSampler`. "
        "See the documentation at https://gemma-llm.readthedocs.io/ for examples."
    )
