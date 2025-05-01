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

import dataclasses

from gemma.gm.ckpts import _paths
from gemma.gm.nn.nano import _config
from gemma.gm.nn.nano import _modules
from gemma.gm.nn.nano import _transformer
from gemma.multimodal import vision as gemma_vision

# TODO(gmenghani): Update the Nano v3 configs.
_NUM_LAYERS_NANO3_MODEL = 24


NANO3_ATTENTION_PATTERN = (
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.GLOBAL,
)


# TODO(gmenghani): Update the architecture to match Nano3.
class _Nano3Base(_transformer.Transformer):
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


# TODO(gmenghani): Update the Nano v3 configs.
class Nano3(_Nano3Base):  # pylint: disable=invalid-name
  """Nano3 transformer architecture."""

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
          NANO3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_NANO3_MODEL,
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
      # default_ckpt=_paths.CheckpointPath.NANO3_IT,
  )

