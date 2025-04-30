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

"""Gemma transformer."""

import dataclasses
import enum
from typing import Iterable

from gemma import modules
from gemma.multimodal import vision as gemma_vision
import jax
import jax.numpy as jnp

Cache = dict[str, modules.LayerCache]


def make_attention_layers_types(
    pattern: tuple[modules.AttentionType, ...],
    *,
    num_layers: int,
) -> tuple[modules.AttentionType, ...]:
  """Returns the list of attention types for every layers."""

  pattern_size = len(pattern)
  out = pattern * (num_layers // pattern_size)
  if num_layers % pattern_size != 0:
    out += pattern[: num_layers % pattern_size]
  return tuple(out)


class QueryPreAttentionNormalisation(enum.Enum):
  """Initialization strategy."""

  # Whether to scale the query by 1/sqrt(head_dim)
  BY_ONE_OVER_SQRT_HEAD_DIM = enum.auto()

  # Whether to scale the query by `embed_dim // num_heads`
  BY_EMBED_DIM_DIV_NUM_HEADS = enum.auto()

  # Whether to scale the query by `1/sqrt(embed_dim // num_heads)`
  BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS = enum.auto()


_NUM_LAYERS_GEMMA_2B = 18
_NUM_LAYERS_GEMMA_7B = 28
_NUM_LAYERS_GEMMA2_2B = 26
_NUM_LAYERS_GEMMA2_9B = 42
_NUM_LAYERS_GEMMA2_27B = 46
_NUM_LAYERS_GEMMA3_1B = 26
_NUM_LAYERS_GEMMA3_4B = 34
_NUM_LAYERS_GEMMA3_12B = 48
_NUM_LAYERS_GEMMA3_27B = 62
GEMMA3_ATTENTION_PATTERN = (
    modules.AttentionType.LOCAL_SLIDING,
    modules.AttentionType.LOCAL_SLIDING,
    modules.AttentionType.LOCAL_SLIDING,
    modules.AttentionType.LOCAL_SLIDING,
    modules.AttentionType.LOCAL_SLIDING,
    modules.AttentionType.GLOBAL,
)


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
  """Configuration for the gemma transformer."""

  num_layers: int
  num_embed: int  # TODO(epot): Rename to `vocab_size` for consistency.
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  final_logit_softcap: float | None
  use_post_attn_norm: bool
  use_post_ffw_norm: bool
  attention_types: Iterable[modules.AttentionType]
  query_pre_attn_norm: QueryPreAttentionNormalisation = (
      QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
  )
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  transpose_gating_einsum: bool = False
  use_qk_norm: bool = False
  local_base_frequency: int = modules.DEFAULT_ROPE_BASE_FREQUENCY
  global_base_frequency: int = modules.DEFAULT_ROPE_BASE_FREQUENCY
  local_scale_factor: float = modules.DEFAULT_ROPE_SCALE_FACTOR
  global_scale_factor: float = modules.DEFAULT_ROPE_SCALE_FACTOR
  vision_encoder: gemma_vision.SigLiPFromPatches | None = None

  def query_pre_attn_scalar(self) -> float:
    """Returns the scalar to multiply the query by before attention."""
    match self.query_pre_attn_norm:
      case QueryPreAttentionNormalisation.BY_EMBED_DIM_DIV_NUM_HEADS:
        return self.embed_dim // self.num_heads
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS:  # pylint: disable=line-too-long
        return (self.embed_dim // self.num_heads) ** -0.5
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM | _:
        return self.head_dim**-0.5

  @classmethod
  def gemma_2b(cls):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA_2B,
        num_embed=256128,
        embed_dim=2048,
        hidden_dim=16384,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        final_logit_softcap=None,
        attention_types=(modules.AttentionType.GLOBAL,) * _NUM_LAYERS_GEMMA_2B,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )

  @classmethod
  def gemma_7b(cls):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA_7B,
        num_embed=256128,
        embed_dim=3072,
        hidden_dim=24576,
        num_heads=16,
        head_dim=256,
        num_kv_heads=16,
        final_logit_softcap=None,
        attention_types=(modules.AttentionType.GLOBAL,) * _NUM_LAYERS_GEMMA_7B,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )

  @classmethod
  def gemma2_2b(cls):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA2_2B,
        num_embed=256128,
        embed_dim=2304,
        hidden_dim=9216,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        final_logit_softcap=30.0,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(_NUM_LAYERS_GEMMA2_2B / 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  @classmethod
  def gemma2_9b(cls):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA2_9B,
        num_embed=256128,
        embed_dim=3584,
        hidden_dim=14336,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        final_logit_softcap=30.0,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(_NUM_LAYERS_GEMMA2_9B / 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
        transpose_gating_einsum=True,
    )

  @classmethod
  def gemma2_27b(cls):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA2_27B,
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
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(_NUM_LAYERS_GEMMA2_27B / 2),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
        transpose_gating_einsum=True,
    )

  @classmethod
  def gemma3_1b(cls):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA3_1B,
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
        attention_types=make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_1B
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=512,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        vision_encoder=None,
    )

  @classmethod
  def gemma3_4b(cls, *, text_only: bool = False):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA3_4B,
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
        attention_types=make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_4B
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=1024,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        vision_encoder=None if text_only else gemma_vision.SigLiPFromPatches(),
    )

  @classmethod
  def gemma3_12b(cls, *, text_only: bool = False):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA3_12B,
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
        attention_types=make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_12B
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=1024,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        vision_encoder=None if text_only else gemma_vision.SigLiPFromPatches(),
    )

  @classmethod
  def gemma3_27b(cls, *, text_only: bool = False):
    return cls(
        num_layers=_NUM_LAYERS_GEMMA3_27B,
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
        attention_types=make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_27B
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS,
        attn_logits_soft_cap=None,
        sliding_window_size=1024,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        vision_encoder=None if text_only else gemma_vision.SigLiPFromPatches(),
    )

  def init_cache(
      self,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
      *,
      cache_length: int,
  ) -> Cache:
    """Initializes a new Transformer cache."""
    if cache_length is None:
      raise ValueError(
          'Missing `cache_length=` kwarg when calling `init_cache()`.'
      )
    cache = {
        f'layer_{i}': modules.Attention.init_cache(
            cache_length,
            self.num_kv_heads,
            self.head_dim,
            batch_size,
            dtype,
        )
        for i in range(self.num_layers)
    }
    return cache


def build_positions_from_mask(input_mask: jax.Array) -> jax.Array:
  """Computes the `positions` from the `input_mask`.

  Args:
    input_mask: The tokens `input_mask`, True for non-padded tokens only.

  Returns:
    The indices to use for RoPE and absolute position encodings for the given
    input mask.
  """
  positions = jnp.cumsum(input_mask, axis=-1)
  # Subtract one for all positions from the first valid one as they are
  # 0-indexed
  return positions - (positions >= 1)
