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

from gemma.gm.nn.gemma3n import _config
from gemma.gm.nn.gemma3n import _modules
from gemma.gm.nn.gemma3n import _transformer
from gemma.multimodal import vision as gemma_vision

_HIDDEN_DIM_E4B = 2048 * 8
_NUM_LAYERS_GEMMA3N_MODEL_E4B = 35

_HIDDEN_DIM_E2B = 2048 * 4
_NUM_LAYERS_GEMMA3N_MODEL_E2B = 30

GEMMA3N_ATTENTION_PATTERN = (
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.GLOBAL,
)


class _Gemma3nBase(_transformer.Gemma3nTransformer):
  """Gemma3n transformer architecture.

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


def _get_gemma3n_config(
    hidden_dim: int,
    num_layers: int,
    num_unshared_kv_layers: int = 20,
) -> _config.TransformerConfig:
  return _config.TransformerConfig(
      final_logit_softcap=None,
      num_embed=262_144,
      embed_dim=2048,
      hidden_dim=hidden_dim,
      num_heads=8,
      head_dim=256,
      num_kv_heads=2,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      use_qk_norm=True,
      qk_norm_with_scale=True,
      use_value_norm=True,
      attention_types=_config.make_attention_layers_types(
          GEMMA3N_ATTENTION_PATTERN,
          num_layers=num_layers,
      ),
      query_pre_attn_norm=_config.QueryPreAttentionNormalisation.NONE,
      attn_logits_soft_cap=None,
      sliding_window_size=512,
      transpose_gating_einsum=True,
      local_base_frequency=10_000,
      global_base_frequency=1_000_000,
      global_scale_factor=1.0,
      vision_encoder=gemma_vision.SigLiPFromPatches(),
      activation_sparsity_pattern=(tuple([0.95] * 10 + [0.0] * (
          num_layers - 10
      ))),
      use_altup=True,
      altup_coef_clip=120.0,
      per_layer_input_dim=256,
      use_laurel=True,
      laurel_rank=64,
      kv_cache_sharing_config=_config.KVCacheSharingConfig(
          frac_shared_layers=(
              (num_layers - num_unshared_kv_layers) * 1.0 / num_layers
          ),
          share_global=True,
          share_local=True,
      ),
      scale_plus_one=False,
      guard_against_excess_precision=True,
  )


class Gemma3n_E4B(_Gemma3nBase):  # pylint: disable=invalid-name
  """Gemma3n E4B transformer architecture."""

  config: _config.TransformerConfig = _get_gemma3n_config(
      hidden_dim=_HIDDEN_DIM_E4B,
      num_layers=_NUM_LAYERS_GEMMA3N_MODEL_E4B,
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version='3n',
  )


class Gemma3n_E2B(_Gemma3nBase):  # pylint: disable=invalid-name
  """Gemma3n E2B transformer architecture."""

  config: _config.TransformerConfig = _get_gemma3n_config(
      hidden_dim=_HIDDEN_DIM_E2B,
      num_layers=_NUM_LAYERS_GEMMA3N_MODEL_E2B,
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version='3n',
  )
