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

"""Gemma transformer config."""

from collections.abc import Sequence
import dataclasses
import functools

from gemma.gm.nn.gemma4 import _modules
from gemma.gm.nn.gemma4.audio import _modules as gemma4_audio
from gemma.gm.nn.gemma4.vision import _encoder as gemma4_vision
from gemma.gm.text import _tokenizer
from gemma.gm.utils import _types
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
    kv_cache_sharing_config: KVCacheSharingConfig | None,
    num_layers: int,
    attention_types: Sequence[_modules.AttentionType],
) -> list[int]:
  """Creates a list of layer indices for which KV cache is used."""
  if kv_cache_sharing_config is None:
    return list(range(num_layers))

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
  return kv_cache_sharing_patterns


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
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  qk_norm_with_scale: bool = True
  num_global_kv_heads: int | None = None
  global_key_size: int | None = None
  k_eq_v_global: bool = False
  global_rope_proportion: float | None = None
  local_rope_proportion: float | None = None
  local_base_frequency: int = _modules.DEFAULT_ROPE_BASE_FREQUENCY
  global_base_frequency: int = _modules.DEFAULT_ROPE_BASE_FREQUENCY
  local_scale_factor: float = _modules.DEFAULT_ROPE_SCALE_FACTOR
  global_scale_factor: float = _modules.DEFAULT_ROPE_SCALE_FACTOR
  per_layer_input_dim: int = 0
  kv_cache_sharing_config: KVCacheSharingConfig | None = None
  override_kv_shared_ffw_hidden: int | None = None
  vision_encoder: gemma4_vision.VisionEncoder | None = None
  audio_encoder: gemma4_audio.ConformerConfig | None = None

  # MoE configuration
  enable_moe: bool = False
  num_experts: int = 0
  expert_dim: int = 0  # MoE expert hidden dim
  top_k_experts: int = 8
  moe_dense_hidden_dim: int = 0  # Dense shared MLP hidden dim (mlp2)

  # Bidirectional attention for image tokens in the text backbone.
  # None: purely causal for all layers (used by E2B, E4B).
  # 'vision': bidirectional for image tokens in sliding layers only,
  #   causal for global layers (used by 26B_A4B, 31B).
  use_bidirectional_attention: str | None = None

  @functools.cached_property
  def num_layers(self) -> int:
    return len(self.attention_types)

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
    cache: Cache = {}

    for i, attn_type in enumerate(self.attention_types):
      if (
          attn_type == _modules.AttentionType.GLOBAL
          and self.global_key_size is not None
      ):
        cache[f'layer_{i}'] = _modules.Attention.init_cache(
            cache_length,
            self.num_global_kv_heads
            if self.num_global_kv_heads
            else self.num_kv_heads,
            self.global_key_size,
            batch_size,
            dtype,
        )
      else:
        cache[f'layer_{i}'] = _modules.Attention.init_cache(
            cache_length,
            self.num_kv_heads,
            self.head_dim,
            batch_size,
            dtype,
        )
    return cache
