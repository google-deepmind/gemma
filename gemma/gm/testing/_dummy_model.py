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

"""Dummy models for testing."""

from gemma.gm.nn import _transformer
from gemma.gm.nn import config as config_lib


class DummyGemma(_transformer.Transformer):  # pylint: disable=invalid-name
  """Dummy transformer architecture, for testing."""

  config: config_lib.TransformerConfig = config_lib.TransformerConfig(
      num_embed=13,  # Vocab size matching `gm.testing.DummyTokenizer()`
      embed_dim=32,
      hidden_dim=128,
      num_heads=2,
      num_kv_heads=2,
      head_dim=128,
      final_logit_softcap=None,
      attention_types=(config_lib.AttentionType.GLOBAL,),
      use_post_attn_norm=None,
      attn_logits_soft_cap=None,
      use_post_ffw_norm=None,
  )
