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

"""Gemma 4 model definitions."""

from __future__ import annotations

import dataclasses

from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _modules
from gemma.gm.nn.gemma4 import _transformer
from gemma.gm.nn.gemma4.audio import _modules as gemma4_audio
from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision


_NUM_LAYERS_GEMMA4_E2B = 35
_NUM_LAYERS_GEMMA4_E4B = 42
_NUM_LAYERS_GEMMA4_31B = 60
_NUM_LAYERS_GEMMA4_26B_A4B_MOE = 30
_DEFAULT_GLOBAL_KEY_SIZE = 512
_FFW_HIDDEN_RATIO = 4


class _Gemma4Base(_transformer.Transformer):
  """Base class for Gemma 4 models.

  Attributes:
    text_only: If True, the media encoders are excluded to save memory and
      compute.
  """

  text_only: bool = True  # Whether to exclude the media encoder.

  INFO = _transformer.ModelInfo(
      tokenizer_version=4,
  )

  def __post_init__(self):
    # Exclude the media encoders if text_only is True.
    if self.text_only and self.config.vision_encoder is not None:
      object.__setattr__(
          self,
          'config',
          dataclasses.replace(
              self.config, vision_encoder=None, audio_encoder=None
          ),
      )
    super().__post_init__()


def _gemma4_config(
    *,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    num_global_kv_heads: int | None,
    per_layer_input_dim: int,
    frac_shared_layers: float,
    sliding_window_size: int,
    attention_types: tuple[_modules.AttentionType, ...],
    k_eq_v_global: bool,
    use_post_attn_norm: bool,
    use_post_ffw_norm: bool,
    override_ffw_hidden_for_kv_cache_sharing: bool,
    vision_encoder: gemma_vision.VisionEncoder | None = None,
    use_bidirectional_attention: str | None = None,
    audio_encoder: gemma4_audio.ConformerConfig | None = None,
) -> _config.TransformerConfig:
  """Creates a TransformerConfig for a Gemma 4 model."""
  if frac_shared_layers > 0.0:
    kv_cache_sharing_config = _config.KVCacheSharingConfig(
        frac_shared_layers=frac_shared_layers,
        share_global=True,
        share_local=True,
    )
  else:
    kv_cache_sharing_config = None
  return _config.TransformerConfig(
      num_embed=262144,
      embed_dim=embed_dim,
      hidden_dim=int(embed_dim * _FFW_HIDDEN_RATIO),
      num_heads=num_heads,
      head_dim=256,
      num_kv_heads=num_kv_heads,
      final_logit_softcap=30.0,
      num_global_kv_heads=num_global_kv_heads,
      use_post_attn_norm=use_post_attn_norm,
      use_post_ffw_norm=use_post_ffw_norm,
      qk_norm_with_scale=True,
      attention_types=attention_types,
      global_key_size=_DEFAULT_GLOBAL_KEY_SIZE,
      k_eq_v_global=k_eq_v_global,
      global_rope_proportion=0.25,
      local_rope_proportion=1.0,
      attn_logits_soft_cap=None,
      sliding_window_size=sliding_window_size,
      local_base_frequency=10_000,
      global_base_frequency=1_000_000,
      vision_encoder=vision_encoder,
      audio_encoder=audio_encoder,
      per_layer_input_dim=per_layer_input_dim,
      kv_cache_sharing_config=kv_cache_sharing_config,
      override_kv_shared_ffw_hidden=int(embed_dim * _FFW_HIDDEN_RATIO * 2)
      if override_ffw_hidden_for_kv_cache_sharing
      else None,
      use_bidirectional_attention=use_bidirectional_attention,
  )


class Gemma4_E2B(_Gemma4Base):  # pylint: disable=invalid-name
  """Gemma 4 E2B model."""

  attention_pattern = (
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.GLOBAL,
  )
  global_local_pattern = _config.make_attention_layers_types(
      attention_pattern, num_layers=_NUM_LAYERS_GEMMA4_E2B
  )

  config: _config.TransformerConfig = _gemma4_config(
      embed_dim=1536,
      num_heads=8,
      num_kv_heads=1,
      num_global_kv_heads=None,
      per_layer_input_dim=256,
      frac_shared_layers=20.0 / _NUM_LAYERS_GEMMA4_E2B,
      attention_types=global_local_pattern,
      sliding_window_size=512,
      k_eq_v_global=False,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      override_ffw_hidden_for_kv_cache_sharing=True,
      vision_encoder=gemma_vision.VisionEncoder(use_clipped_linears=True),
      audio_encoder=gemma4_audio.ConformerConfig(),
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=4,
  )


class Gemma4_E4B(_Gemma4Base):  # pylint: disable=invalid-name
  """Gemma 4 E4B model."""

  attention_pattern = (
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.GLOBAL,
  )
  global_local_pattern = _config.make_attention_layers_types(
      attention_pattern, num_layers=_NUM_LAYERS_GEMMA4_E4B
  )

  config: _config.TransformerConfig = _gemma4_config(
      embed_dim=2560,
      num_heads=8,
      num_kv_heads=2,
      num_global_kv_heads=None,
      per_layer_input_dim=256,
      frac_shared_layers=18.0 / _NUM_LAYERS_GEMMA4_E4B,
      attention_types=global_local_pattern,
      sliding_window_size=512,
      k_eq_v_global=False,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      override_ffw_hidden_for_kv_cache_sharing=False,
      vision_encoder=gemma_vision.VisionEncoder(use_clipped_linears=True),
      audio_encoder=gemma4_audio.ConformerConfig(),
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=4,
  )


class Gemma4_31B(_Gemma4Base):  # pylint: disable=invalid-name
  """Gemma 4 31B model."""

  attention_pattern = (
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.GLOBAL,
  )
  global_local_pattern = _config.make_attention_layers_types(
      attention_pattern, num_layers=_NUM_LAYERS_GEMMA4_31B
  )
  config: _config.TransformerConfig = _gemma4_config(
      embed_dim=5376,
      num_heads=32,
      num_kv_heads=16,
      num_global_kv_heads=4,
      per_layer_input_dim=0,
      frac_shared_layers=0.0,
      attention_types=global_local_pattern,
      sliding_window_size=1024,
      k_eq_v_global=True,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      override_ffw_hidden_for_kv_cache_sharing=False,
      vision_encoder=gemma_vision.VisionEncoder(
          d_model=1152,
          num_layers=27,
          num_heads=16,
          ffw_hidden=4304,
          use_clipped_linears=False,
          standardize_embeddings=True,
      ),
      use_bidirectional_attention='vision',
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=4,
  )


class Gemma4_26B_A4B(_Gemma4Base):  # pylint: disable=invalid-name
  """Gemma 4 26B_A4B MoE model.

  A Mixture-of-Experts model with 128 experts per layer.
  Each layer has:
    - A MoE branch (128 experts, expert_dim=704)
    - A dense shared MLP branch (intermediate_dim=2112)
  """

  attention_pattern = (
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.LOCAL_SLIDING,
      _modules.AttentionType.GLOBAL,
  )
  global_local_pattern = _config.make_attention_layers_types(
      attention_pattern, num_layers=_NUM_LAYERS_GEMMA4_26B_A4B_MOE
  )
  config: _config.TransformerConfig = _config.TransformerConfig(
      num_embed=262144,
      embed_dim=2816,
      hidden_dim=2112,  # Dense shared MLP (mlp2) hidden dim
      num_heads=16,
      head_dim=256,
      num_kv_heads=8,
      final_logit_softcap=30.0,
      num_global_kv_heads=2,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      qk_norm_with_scale=True,
      attention_types=global_local_pattern,
      global_key_size=_DEFAULT_GLOBAL_KEY_SIZE,
      k_eq_v_global=True,
      global_rope_proportion=0.25,
      local_rope_proportion=1.0,
      attn_logits_soft_cap=None,
      sliding_window_size=1024,
      local_base_frequency=10_000,
      global_base_frequency=1_000_000,
      per_layer_input_dim=0,
      # MoE configuration
      enable_moe=True,
      num_experts=128,
      expert_dim=704,
      top_k_experts=8,
      moe_dense_hidden_dim=2112,
      vision_encoder=gemma_vision.VisionEncoder(
          d_model=1152,
          num_layers=27,
          num_heads=16,
          ffw_hidden=4304,
          output_length=280,
          use_clipped_linears=False,
          standardize_embeddings=True,
      ),
      use_bidirectional_attention='vision',
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=4,
  )
