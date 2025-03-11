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

"""Gemma models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from gemma import transformer
from gemma.gm.ckpts import _paths
from gemma.gm.nn import _transformer


class Gemma2_2B(_transformer.Transformer):  # pylint: disable=invalid-name
  """Gemma2 transformer architecture."""

  config: transformer.TransformerConfig = (
      transformer.TransformerConfig.gemma2_2b(cache_size=None)
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=2,
      default_ckpt=_paths.CheckpointPath.GEMMA2_2B_IT,
  )


class Gemma2_9B(_transformer.Transformer):  # pylint: disable=invalid-name
  """Gemma2 transformer architecture."""

  config: transformer.TransformerConfig = (
      transformer.TransformerConfig.gemma2_9b(cache_size=None)
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=2,
      default_ckpt=_paths.CheckpointPath.GEMMA2_9B_IT,
  )


class Gemma2_27B(_transformer.Transformer):  # pylint: disable=invalid-name
  """Gemma2 transformer architecture."""

  config: transformer.TransformerConfig = (
      transformer.TransformerConfig.gemma2_27b(cache_size=None)
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=2,
      default_ckpt=_paths.CheckpointPath.GEMMA2_27B_IT,
  )


class Gemma3_1B(_transformer.Transformer):  # pylint: disable=invalid-name
  """Gemma3 transformer architecture."""

  config: transformer.TransformerConfig = (
      transformer.TransformerConfig.gemma3_1b()
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=3,
      # default_ckpt=_paths.CheckpointPath.GEMMA3_1B_IT,
  )


class _Gemma3Base(_transformer.Transformer):
  """Gemma3 transformer architecture.

  Attributes:
    text_only: If True, skip the vision encoder. Saves memory and compute.
  """

  text_only: bool = False  # Whether to skip the vision encoder.
  config: transformer.TransformerConfig = None  # Auto-set in __post_init__  # pytype: disable=annotation-type-mismatch

  CONFIG_FN: ClassVar[Callable[..., transformer.TransformerConfig]]

  def __init_subclass__(cls, **kwargs: Any):
    if 'CONFIG_FN' in cls.__dict__:
      cls.CONFIG_FN = staticmethod(cls.CONFIG_FN)
    super().__init_subclass__(**kwargs)

  def __post_init__(self):
    if self.config is None:
      object.__setattr__(
          self,
          'config',
          self.CONFIG_FN(text_only=self.text_only),
      )
    super().__post_init__()


class Gemma3_4B(_Gemma3Base):  # pylint: disable=invalid-name
  """Gemma3 transformer architecture."""

  CONFIG_FN = transformer.TransformerConfig.gemma3_4b
  INFO = _transformer.ModelInfo(
      tokenizer_version=3,
      # default_ckpt=_paths.CheckpointPath.GEMMA3_4B_IT,
  )


class Gemma3_12B(_Gemma3Base):  # pylint: disable=invalid-name
  """Gemma3 transformer architecture."""

  CONFIG_FN = transformer.TransformerConfig.gemma3_12b
  INFO = _transformer.ModelInfo(
      tokenizer_version=3,
      # default_ckpt=_paths.CheckpointPath.GEMMA3_12B_IT,
  )


class Gemma3_27B(_Gemma3Base):  # pylint: disable=invalid-name
  """Gemma3 transformer architecture."""

  CONFIG_FN = transformer.TransformerConfig.gemma3_27b
  INFO = _transformer.ModelInfo(
      tokenizer_version=3,
      # default_ckpt=_paths.CheckpointPath.GEMMA3_27B_IT,
  )
