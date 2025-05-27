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

"""Experimental models."""

from gemma.gm.nn import _config
from gemma.gm.nn import _gemma
from gemma.gm.nn import _transformer


class Gemma3_500m(_transformer.Transformer):  # pylint: disable=invalid-name
  """Gemma3 500m transformer architecture."""

  config: _config.TransformerConfig = _config.TransformerConfig(
      final_logit_softcap=None,
      num_embed=262144,
      embed_dim=896,
      hidden_dim=4 * 896,
      num_heads=4,
      head_dim=256,
      num_kv_heads=1,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      use_qk_norm=True,
      attention_types=_config.make_attention_layers_types(
          pattern=_gemma.GEMMA3_ATTENTION_PATTERN,
          num_layers=22,
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
