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

from __future__ import annotations

import dataclasses

from gemma.gm.ckpts import _paths
from gemma.gm.nn import _config
from gemma.gm.nn import _modules
from gemma.gm.nn import _transformer
from gemma.multimodal import vision as gemma_vision

_NUM_LAYERS_GEMMA_2B = 18
_NUM_LAYERS_GEMMA_7B = 28
_NUM_LAYERS_GEMMA2_2B = 26
_NUM_LAYERS_GEMMA2_9B = 42
_NUM_LAYERS_GEMMA2_27B = 46
_NUM_LAYERS_GEMMA3_270M = 18
_NUM_LAYERS_GEMMA3_1B = 26
_NUM_LAYERS_GEMMA3_4B = 34
_NUM_LAYERS_GEMMA3_12B = 48
_NUM_LAYERS_GEMMA3_27B = 62


GEMMA3_ATTENTION_PATTERN = (
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.GLOBAL,
)


class Gemma2_2B(_transformer.Transformer):  # pylint: disable=invalid-name
  """Gemma2 transformer architecture."""

  config: _config.TransformerConfig = _config.TransformerConfig(
      num_embed=256128,
      embed_dim=2304,
      hidden_dim=9216,
      num_heads=8,
      head_dim=256,
      num_kv_heads=4,
      final_logit_softcap=30.0,
      attention_types=(
          _modules.AttentionType.LOCAL_SLIDING,
          _modules.AttentionType.GLOBAL,
      )
      * int(_NUM_LAYERS_GEMMA2_2B / 2),
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      query_pre_attn_norm=_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
      attn_logits_soft_cap=50.0,
      sliding_window_size=4096,
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=2,
      default_ckpt=_paths.CheckpointPath.GEMMA2_2B_IT,
  )


class Gemma2_9B(_transformer.Transformer):  # pylint: disable=invalid-name
  """Gemma2 transformer architecture."""

  config: _config.TransformerConfig = _config.TransformerConfig(
      num_embed=256128,
      embed_dim=3584,
      hidden_dim=14336,
      num_heads=16,
      head_dim=256,
      num_kv_heads=8,
      final_logit_softcap=30.0,
      attention_types=(
          _modules.AttentionType.LOCAL_SLIDING,
          _modules.AttentionType.GLOBAL,
      )
      * int(_NUM_LAYERS_GEMMA2_9B / 2),
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      query_pre_attn_norm=_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
      attn_logits_soft_cap=50.0,
      sliding_window_size=4096,
      transpose_gating_einsum=True,
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=2,
      default_ckpt=_paths.CheckpointPath.GEMMA2_9B_IT,
  )


class Gemma2_27B(_transformer.Transformer):  # pylint: disable=invalid-name
  """Gemma2 transformer architecture."""

  config: _config.TransformerConfig = _config.TransformerConfig(
      num_embed=256128,
      embed_dim=4608,
      hidden_dim=36864,
      num_heads=32,
      head_dim=128,
      num_kv_heads=16,
      final_logit_softcap=30.0,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      attention_types=(
          _modules.AttentionType.LOCAL_SLIDING,
          _modules.AttentionType.GLOBAL,
      )
      * int(_NUM_LAYERS_GEMMA2_27B / 2),
      query_pre_attn_norm=_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS,
      attn_logits_soft_cap=50.0,
      sliding_window_size=4096,
      transpose_gating_einsum=True,
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=2,
      default_ckpt=_paths.CheckpointPath.GEMMA2_27B_IT,
  )


class Gemma3_270M(_transformer.Transformer):  # pylint: disable=invalid-name
  """Gemma3 transformer architecture."""

  config: _config.TransformerConfig = _config.TransformerConfig(
      final_logit_softcap=None,
      num_embed=262144,
      embed_dim=640,
      hidden_dim=2048,
      num_heads=4,
      head_dim=256,
      num_kv_heads=1,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      use_qk_norm=True,
      attention_types=_config.make_attention_layers_types(
          GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_270M
      ),
      query_pre_attn_norm=_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
      attn_logits_soft_cap=None,
      sliding_window_size=512,
      transpose_gating_einsum=True,
      local_base_frequency=10_000,
      global_base_frequency=1_000_000,
      vision_encoder=None,
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=3,
  )


class Gemma3_1B(_transformer.Transformer):  # pylint: disable=invalid-name
  """Gemma3 transformer architecture."""

  config: _config.TransformerConfig = _config.TransformerConfig(
      final_logit_softcap=None,
      num_embed=262144,
      embed_dim=1152,
      hidden_dim=6 * 1152,
      num_heads=4,
      head_dim=256,
      num_kv_heads=1,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      use_qk_norm=True,
      attention_types=_config.make_attention_layers_types(
          GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_1B
      ),
      query_pre_attn_norm=_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
      attn_logits_soft_cap=None,
      sliding_window_size=512,
      transpose_gating_einsum=True,
      local_base_frequency=10_000,
      global_base_frequency=1_000_000,
      vision_encoder=None,
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

  def __post_init__(self):
    # Remove the vision encoder if text_only is True.
    if self.text_only and self.config.vision_encoder is not None:
      object.__setattr__(
          self,
          'config',
          dataclasses.replace(self.config, vision_encoder=None),
      )
    super().__post_init__()


class Gemma3_4B(_Gemma3Base):  # pylint: disable=invalid-name
  """Gemma3 transformer architecture."""

  config: _config.TransformerConfig = _config.TransformerConfig(
      final_logit_softcap=None,
      num_embed=262_144,
      embed_dim=2560,
      hidden_dim=2560 * 8 // 2,
      num_heads=8,
      head_dim=256,
      num_kv_heads=4,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      use_qk_norm=True,
      attention_types=_config.make_attention_layers_types(
          GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_4B
      ),
      query_pre_attn_norm=_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
      attn_logits_soft_cap=None,
      sliding_window_size=1024,
      transpose_gating_einsum=True,
      local_base_frequency=10_000,
      global_base_frequency=1_000_000,
      global_scale_factor=8.0,
      vision_encoder=gemma_vision.SigLiPFromPatches(),
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=3,
      # default_ckpt=_paths.CheckpointPath.GEMMA3_4B_IT,
  )


class Gemma3_12B(_Gemma3Base):  # pylint: disable=invalid-name
  """Gemma3 transformer architecture."""

  config: _config.TransformerConfig = _config.TransformerConfig(
      final_logit_softcap=None,
      num_embed=262144,
      embed_dim=30 * 128,
      hidden_dim=8 * 30 * 128 // 2,
      num_heads=16,
      head_dim=256,
      num_kv_heads=8,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      use_qk_norm=True,
      attention_types=_config.make_attention_layers_types(
          GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_12B
      ),
      query_pre_attn_norm=_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
      attn_logits_soft_cap=None,
      sliding_window_size=1024,
      transpose_gating_einsum=True,
      local_base_frequency=10_000,
      global_base_frequency=1_000_000,
      global_scale_factor=8.0,
      vision_encoder=gemma_vision.SigLiPFromPatches(),
  )
  INFO = _transformer.ModelInfo(
      tokenizer_version=3,
      # default_ckpt=_paths.CheckpointPath.GEMMA3_12B_IT,
  )


class Gemma3_27B(_Gemma3Base):  # pylint: disable=invalid-name
  """Gemma3 transformer architecture."""

  config: _config.TransformerConfig = _config.TransformerConfig(
      final_logit_softcap=None,
      num_embed=262144,
      embed_dim=5376,
      hidden_dim=5376 * 8 // 2,
      num_heads=32,
      head_dim=128,
      num_kv_heads=16,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      use_qk_norm=True,
      attention_types=_config.make_attention_layers_types(
          GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_27B
      ),
      query_pre_attn_norm=_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS,
      attn_logits_soft_cap=None,
      sliding_window_size=1024,
      transpose_gating_einsum=True,
      local_base_frequency=10_000,
      global_base_frequency=1_000_000,
      global_scale_factor=8.0,
      vision_encoder=gemma_vision.SigLiPFromPatches(),
  )
  INFO = _transformer.ModelInfo(
      tokenizer_version=3,
      # default_ckpt=_paths.CheckpointPath.GEMMA3_27B_IT,
  )
