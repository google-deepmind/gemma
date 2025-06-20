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

"""Config for T5 Gemma and checkpoints."""

import dataclasses
import enum

from gemma.gm import text
from gemma.research.t5gemma import modules
from gemma.research.t5gemma import t5gemma
import kagglehub


TransformerConfig = modules.TransformerConfig
BY_ONE_OVER_SQRT_HEAD_DIM = (
    modules.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
)


class CKPTType(enum.StrEnum):
  """Checkpoint type."""
  PT = "pt"
  IT = "it"


class PretrainType(enum.StrEnum):
  """Pretrain type."""
  UL2 = "ul2"
  PREFIXLM = "prefixlm"


class GemmaPreset(enum.StrEnum):
  """Decoder-only Gemma config."""

  GEMMA2_2B = enum.auto()
  GEMMA2_9B = enum.auto()

  GEMMA2_SMALL = enum.auto()
  GEMMA2_BASE = enum.auto()
  GEMMA2_LARGE = enum.auto()
  GEMMA2_ML = enum.auto()
  GEMMA2_XL = enum.auto()

  def make_config(
      self,
      num_layers: int,
      embed_dim: int,
      hidden_dim: int,
      num_heads: int,
      head_dim: int,
      num_kv_heads: int,
      enable_cross_attention: bool = False,
      bidirectional: bool = False,
      transpose_gating_einsum: bool = False,
  ):
    """Simplify the config creation."""
    return TransformerConfig(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        final_logit_softcap=30.0,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        ) * (num_layers // 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        query_pre_attn_norm=BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
        transpose_gating_einsum=transpose_gating_einsum,
        enable_cross_attention=enable_cross_attention,
        bidirectional=bidirectional,
    )

  def config(
      self,
      enable_cross_attention: bool = False,
      bidirectional: bool = False,
  ):
    """Returns the decoder-only config for building up encoder-decoder model."""
    match self:
      case self.GEMMA2_2B:
        return self.make_config(
            num_layers=26,
            embed_dim=2304,
            hidden_dim=9216,
            num_heads=8,
            head_dim=256,
            num_kv_heads=4,
            enable_cross_attention=enable_cross_attention,
            bidirectional=bidirectional,
        )
      case self.GEMMA2_9B:
        return self.make_config(
            num_layers=42,
            embed_dim=3584,
            hidden_dim=14336,
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            transpose_gating_einsum=True,
            enable_cross_attention=enable_cross_attention,
            bidirectional=bidirectional,
        )
      case self.GEMMA2_SMALL:
        return self.make_config(
            num_layers=8,
            embed_dim=512,
            hidden_dim=1024,
            num_heads=8,
            head_dim=64,
            num_kv_heads=8,
            enable_cross_attention=enable_cross_attention,
            bidirectional=bidirectional,
        )
      case self.GEMMA2_BASE:
        return self.make_config(
            num_layers=12,
            embed_dim=768,
            hidden_dim=2048,
            num_heads=12,
            head_dim=64,
            num_kv_heads=12,
            enable_cross_attention=enable_cross_attention,
            bidirectional=bidirectional,
        )
      case self.GEMMA2_LARGE:
        return self.make_config(
            num_layers=24,
            embed_dim=1024,
            hidden_dim=2816,
            num_heads=16,
            head_dim=64,
            num_kv_heads=16,
            enable_cross_attention=enable_cross_attention,
            bidirectional=bidirectional,
        )
      case self.GEMMA2_ML:
        return self.make_config(
            num_layers=26,
            embed_dim=1152,
            hidden_dim=6912,
            num_heads=4,
            head_dim=256,
            num_kv_heads=4,
            enable_cross_attention=enable_cross_attention,
            bidirectional=bidirectional,
        )
      case self.GEMMA2_XL:
        return self.make_config(
            num_layers=24,
            embed_dim=2048,
            hidden_dim=5120,
            num_heads=32,
            head_dim=64,
            num_kv_heads=32,
            enable_cross_attention=enable_cross_attention,
            bidirectional=bidirectional,
        )
      case _:
        raise ValueError(f"Unsupported decoder-only Gemma config: {self}")


class T5GemmaPreset(enum.StrEnum):
  """T5Gemma config."""

  GEMMA2_2B_2B = "2b-2b"
  GEMMA2_9B_9B = "9b-9b"
  GEMMA2_9B_2B = "9b-2b"

  GEMMA2_SMALL_SMALL = "s-s"
  GEMMA2_BASE_BASE = "b-b"
  GEMMA2_LARGE_LARGE = "l-l"
  GEMMA2_ML_ML = "ml-ml"
  GEMMA2_XL_XL = "xl-xl"

  def make_config(
      self,
      encoder: GemmaPreset,
      decoder: GemmaPreset,
  ):
    """Simplify the config creation."""
    cfg = t5gemma.T5GemmaConfig(
        encoder_config=encoder.config(
            enable_cross_attention=False,
            bidirectional=True,
        ),
        decoder_config=decoder.config(
            enable_cross_attention=True,
            bidirectional=False,
        ),
    )

    # Update decoder's kv_embed_dim to match encoder's embed_dim.
    # In imblanced encoder-decoder (e.g., 9B-2B), the decoder's kv_embed_dim
    # should be set to match the encoder's embed_dim.
    cfg = dataclasses.replace(
        cfg,
        decoder_config=dataclasses.replace(
            cfg.decoder_config,
            kv_embed_dim=cfg.encoder_config.embed_dim,
        ),
    )
    return cfg

  @property
  def tokenizer(self):
    return text.Gemma2Tokenizer()

  @property
  def config(self):
    match self:
      case self.GEMMA2_2B_2B:
        return self.make_config(
            encoder=GemmaPreset.GEMMA2_2B,
            decoder=GemmaPreset.GEMMA2_2B,
        )
      case self.GEMMA2_9B_9B:
        return self.make_config(
            encoder=GemmaPreset.GEMMA2_9B,
            decoder=GemmaPreset.GEMMA2_9B,
        )
      case self.GEMMA2_9B_2B:
        return self.make_config(
            encoder=GemmaPreset.GEMMA2_9B,
            decoder=GemmaPreset.GEMMA2_2B,
        )
      case self.GEMMA2_SMALL_SMALL:
        return self.make_config(
            encoder=GemmaPreset.GEMMA2_SMALL,
            decoder=GemmaPreset.GEMMA2_SMALL,
        )
      case self.GEMMA2_BASE_BASE:
        return self.make_config(
            encoder=GemmaPreset.GEMMA2_BASE,
            decoder=GemmaPreset.GEMMA2_BASE,
        )
      case self.GEMMA2_LARGE_LARGE:
        return self.make_config(
            encoder=GemmaPreset.GEMMA2_LARGE,
            decoder=GemmaPreset.GEMMA2_LARGE,
        )
      case self.GEMMA2_XL_XL:
        return self.make_config(
            encoder=GemmaPreset.GEMMA2_XL,
            decoder=GemmaPreset.GEMMA2_XL,
        )
      case self.GEMMA2_ML_ML:
        return self.make_config(
            encoder=GemmaPreset.GEMMA2_ML,
            decoder=GemmaPreset.GEMMA2_ML,
        )
      case _:
        raise ValueError(f"Unsupported T5Gemma config: {self}")

  def get_checkpoint_from_kaggle(
      self,
      ckpt_type: CKPTType,
      pretrain_type: PretrainType,
  ):
    """Downloads the checkpoint from Kaggle and returns the path."""
    ckpt_type_str = "-it" if ckpt_type == CKPTType.IT else ""
    kaggle_ckpt_name = (
        "google/t5gemma/flax/"
        f"t5gemma-{self}-{pretrain_type}{ckpt_type_str}"
    )
    kaggle_ckpt_path = kagglehub.model_download(kaggle_ckpt_name)
    return kaggle_ckpt_path
