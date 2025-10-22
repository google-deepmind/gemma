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

"""Gemma transformer config."""

from collections.abc import Sequence
import dataclasses
import enum
import functools

from gemma.gm.nn.gemma3n import _modules
from gemma.gm.text import _tokenizer
from gemma.gm.utils import _types
from gemma.multimodal import vision as gemma_vision
import jax.numpy as jnp

Cache = dict[str, _modules.LayerCache]


@dataclasses.dataclass(frozen=True)
class KVCacheSharingConfig:
  """Configuration for KV cache sharing."""

  frac_shared_layers: float = 0.0
  share_global: bool = False
  share_local: bool = False


def make_attention_layers_types(
    pattern: tuple[_modules.AttentionType, ...],
    *,
    num_layers: int,
) -> tuple[_modules.AttentionType, ...]:
  """Returns the list of attention types for every layers."""

  pattern_size = len(pattern)
  out = pattern * (num_layers // pattern_size)
  if num_layers % pattern_size != 0:
    out += pattern[: num_layers % pattern_size]
  return tuple(out)


def create_kv_cache_sharing_patterns(  # pylint: disable=invalid-name
    kv_cache_sharing_config: KVCacheSharingConfig,
    num_layers: int,
    attention_types: Sequence[_modules.AttentionType],
) -> list[int]:
  """Creates a list of layer indices for which KV cache is used."""
  if kv_cache_sharing_config is not None:
    kv_cache_sharing_patterns = []
    num_unshared_layers = int(
        num_layers - kv_cache_sharing_config.frac_shared_layers * num_layers
    )
    for i in range(num_layers):
      if i < num_unshared_layers:
        kv_cache_sharing_patterns.append(i)
      else:
        if (
            attention_types[i] == _modules.AttentionType.GLOBAL
            and kv_cache_sharing_config.share_global
        ):
          kv_cache_sharing_patterns.append(num_unshared_layers - 1)
        elif (
            attention_types[i] == _modules.AttentionType.LOCAL_SLIDING
            and kv_cache_sharing_config.share_local
        ):
          kv_cache_sharing_patterns.append(num_unshared_layers - 2)
        else:
          kv_cache_sharing_patterns.append(i)
  else:
    kv_cache_sharing_patterns = list(range(num_layers))
  return kv_cache_sharing_patterns


class QueryPreAttentionNormalisation(enum.Enum):
  """Initialization strategy."""

  # Apply no scaling.
  NONE = enum.auto()

  # Whether to scale the query by 1/sqrt(head_dim)
  BY_ONE_OVER_SQRT_HEAD_DIM = enum.auto()

  # Whether to scale the query by `embed_dim // num_heads`
  BY_EMBED_DIM_DIV_NUM_HEADS = enum.auto()

  # Whether to scale the query by `1/sqrt(embed_dim // num_heads)`
  BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS = enum.auto()


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
  """Configuration for the gemma transformer."""

  num_embed: int  # TODO(epot): Rename to `vocab_size` for consistency.
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  final_logit_softcap: float | None
  use_post_attn_norm: bool
  use_post_ffw_norm: bool
  # TODO(epot): Should be renamed `layers_types` or similar ?
  attention_types: Sequence[_modules.AttentionType]
  query_pre_attn_norm: QueryPreAttentionNormalisation = (
      QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
  )
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  transpose_gating_einsum: bool = False
  use_qk_norm: bool = False
  qk_norm_with_scale: bool = True
  use_value_norm: bool = False
  local_base_frequency: int = _modules.DEFAULT_ROPE_BASE_FREQUENCY
  global_base_frequency: int = _modules.DEFAULT_ROPE_BASE_FREQUENCY
  local_scale_factor: float = _modules.DEFAULT_ROPE_SCALE_FACTOR
  global_scale_factor: float = _modules.DEFAULT_ROPE_SCALE_FACTOR
  vision_encoder: gemma_vision.SigLiPFromPatches | None = None
  use_altup: bool = False
  num_altup_inputs: int = 4
  altup_coef_clip: float | None = None
  activation_sparsity_pattern: Sequence[float] | None = None
  per_layer_input_dim: int = 0
  use_laurel: bool = False
  laurel_rank: int = 64
  kv_cache_sharing_config: KVCacheSharingConfig | None = None
  scale_plus_one: bool = True
  guard_against_excess_precision: bool = False

  @functools.cached_property
  def num_layers(self) -> int:
    return len(self.attention_types)

  def query_pre_attn_scalar(self) -> float:
    """Returns the scalar to multiply the query by before attention."""
    match self.query_pre_attn_norm:
      case QueryPreAttentionNormalisation.NONE:
        return 1.0
      case QueryPreAttentionNormalisation.BY_EMBED_DIM_DIV_NUM_HEADS:
        return self.embed_dim // self.num_heads
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS:  # pylint: disable=line-too-long
        return (self.embed_dim // self.num_heads) ** -0.5
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM | _:
        return self.head_dim**-0.5

  @functools.cached_property
  def input_config(self) -> _types.InputConfig:
    """Returns the input config for the transformer."""
    # TODO(epot): Have the tokenizer version be part of the config
    # instead.
    special_tokens = _tokenizer.Gemma3Tokenizer.special_tokens

    if self.vision_encoder is not None:
      return _types.InputConfig(
          support_images=True,
          num_tokens_per_image=self.vision_encoder.num_mm_tokens_per_image,
          special_tokens=special_tokens,
      )
    else:
      return _types.InputConfig(
          support_images=False,
          num_tokens_per_image=0,
          special_tokens=special_tokens,
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
        f'layer_{i}': _modules.Attention.init_cache(
            cache_length,
            self.num_kv_heads,
            self.head_dim,
            batch_size,
            dtype,
        )
        for i in range(self.num_layers)
    }
    return cache
