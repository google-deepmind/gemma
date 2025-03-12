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
import typing
from typing import Iterable
import warnings


import einops
from flax import linen as nn
from gemma import layers
from gemma import modules
from gemma import params as params_lib
from gemma.multimodal import vision as gemma_vision
import jax
import jax.numpy as jnp

Cache = dict[str, modules.LayerCache]


def _make_attention_type_from_pattern(
    pattern: tuple[modules.AttentionType, ...],
    num_layers: int,
) -> tuple[modules.AttentionType, ...]:

  pattern_size = len(pattern)
  out = pattern * (num_layers // pattern_size)
  if num_layers % pattern_size != 0:
    out += pattern[: num_layers % pattern_size]
  return out


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
_GEMMA3_ATTENTION_PATTERN = (
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
  max_cache_length: int | None = 1024
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
  mm_extra_vocab_size: int = 0
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
  def from_params(
      cls, params: params_lib.Params, cache_size: int | None = 1024
  ) -> 'TransformerConfig':
    """Creates a TransformerConfig from loaded parameters.

    Args:
      params: Model parameters
      cache_size: Number of tokens to cache

    Returns:
      TransformerConfig.
    """
    layer_names = [
        name for name in params['transformer'].keys() if 'layer' in name
    ]
    layer_names = [name.replace('layer_', '') for name in layer_names]
    num_layers = max([int(layer) for layer in layer_names]) + 1

    if num_layers == _NUM_LAYERS_GEMMA_2B:
      return cls.gemma_2b(cache_size)
    if num_layers == _NUM_LAYERS_GEMMA_7B:
      return cls.gemma_7b(cache_size)
    if num_layers == _NUM_LAYERS_GEMMA2_2B:
      return cls.gemma2_2b(cache_size)
    if num_layers == _NUM_LAYERS_GEMMA2_9B:
      return cls.gemma2_9b(cache_size)
    if num_layers == _NUM_LAYERS_GEMMA2_27B:
      return cls.gemma2_27b(cache_size)
    if num_layers == _NUM_LAYERS_GEMMA3_4B:
      return cls.gemma3_4b(text_only=False)

    raise ValueError(
        'Params are not a Gemma 2b, 7b, or Gemma 2 2b, 9b, or 27b variant.'
    )

  @classmethod
  def gemma_2b(cls, cache_size: int | None):
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
        max_cache_length=cache_size,
    )

  @classmethod
  def gemma_7b(cls, cache_size: int | None):
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
        max_cache_length=cache_size,
    )

  @classmethod
  def gemma2_2b(cls, cache_size: int | None):
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
        max_cache_length=cache_size,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  @classmethod
  def gemma2_9b(cls, cache_size: int | None):
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
        max_cache_length=cache_size,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
        transpose_gating_einsum=True,
    )

  @classmethod
  def gemma2_27b(cls, cache_size: int | None):
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
        max_cache_length=cache_size,
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
        attention_types=_make_attention_type_from_pattern(
            _GEMMA3_ATTENTION_PATTERN, _NUM_LAYERS_GEMMA3_1B
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=512,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        vision_encoder=None,
        max_cache_length=None,
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
        attention_types=_make_attention_type_from_pattern(
            _GEMMA3_ATTENTION_PATTERN, _NUM_LAYERS_GEMMA3_4B
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=1024,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        vision_encoder=None if text_only else gemma_vision.SigLiPFromPatches(),
        mm_extra_vocab_size=0 if text_only else 128,
        max_cache_length=None,
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
        attention_types=_make_attention_type_from_pattern(
            _GEMMA3_ATTENTION_PATTERN, _NUM_LAYERS_GEMMA3_12B
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=1024,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        vision_encoder=None if text_only else gemma_vision.SigLiPFromPatches(),
        mm_extra_vocab_size=0 if text_only else 128,
        max_cache_length=None,
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
        attention_types=_make_attention_type_from_pattern(
            _GEMMA3_ATTENTION_PATTERN, _NUM_LAYERS_GEMMA3_27B
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS,
        attn_logits_soft_cap=None,
        sliding_window_size=1024,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        vision_encoder=None if text_only else gemma_vision.SigLiPFromPatches(),
        mm_extra_vocab_size=0 if text_only else 128,
        max_cache_length=None,
    )

  def init_cache(
      self,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
      *,
      cache_length: int | None = None,
  ) -> Cache:
    """Initializes a new Transformer cache."""
    cache_length = cache_length or self.max_cache_length
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


class Transformer(nn.Module):
  """Gemma transformer."""

  config: TransformerConfig

  def __post_init__(self):
    if type(self) == Transformer:  # pylint: disable=unidiomatic-typecheck]
      msg = (
          'The old Transformer class is deprecated, behave unexpectedly and'
          " doesn't support multimodal."
          ' Instead, `gm.nn.GemmaXX` should be used.'
          ' See the documentation at https://gemma-llm.readthedocs.io/. '
      )
      raise DeprecationWarning(msg)
    super().__post_init__()

  def setup(self):
    self.embedder = modules.Embedder(
        vocab_size=self.config.num_embed,
        embed_dim=self.config.embed_dim,
        vision_proj_dim=self.config.vision_encoder.siglip_encoder.width
        if self.config.vision_encoder
        else None,
    )

    self.blocks = [
        modules.Block(
            name=f'layer_{i}',
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            embed_dim=self.config.embed_dim,
            head_dim=self.config.head_dim,
            hidden_dim=self.config.hidden_dim,
            sliding_window_size=self.config.sliding_window_size,
            use_post_attn_norm=self.config.use_post_attn_norm,
            use_post_ffw_norm=self.config.use_post_ffw_norm,
            attn_logits_soft_cap=self.config.attn_logits_soft_cap,
            attn_type=attn_type,
            query_pre_attn_scalar=self.config.query_pre_attn_scalar(),
            transpose_gating_einsum=self.config.transpose_gating_einsum,
            use_qk_norm=self.config.use_qk_norm,
            rope_base_frequency=self.config.local_base_frequency
            if attn_type == modules.AttentionType.LOCAL_SLIDING
            else self.config.global_base_frequency,
            rope_scale_factor=self.config.local_scale_factor
            if attn_type == modules.AttentionType.LOCAL_SLIDING
            else self.config.global_scale_factor,
        )
        for i, attn_type in zip(
            range(self.config.num_layers), self.config.attention_types
        )
    ]
    self.final_norm = layers.RMSNorm()

    self.vision_encoder = self.config.vision_encoder

  def __call__(
      self,
      last_tokens: jax.Array,  # [B, L]
      positions: jax.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jax.Array,  # [B, L, L']
      patches: jax.Array | None = None,  # [B, N, P, D']
  ) -> tuple[jax.Array, Cache | None]:
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      last_tokens: input sequence of tokens.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.
      patches: visual data.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    if patches is not None:
      _check_tokens_for_vision(last_tokens, patches)
    x = self.embedder.encode(last_tokens)
    if patches is not None:
      x = self._include_vision_embeddings(
          last_tokens=last_tokens, embeddings=x, patches=patches
      )
    for i, block in enumerate(self.blocks):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = block(
          x,
          positions,
          layer_cache,
          attention_mask,
      )
      if cache is not None:
        cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)
    logits = self.embedder.decode(x)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return logits, cache  # pytype: disable=bad-return-type

  def _include_vision_embeddings(
      self,
      last_tokens: jax.Array,  # [B, L]
      embeddings: jax.Array,  # [B, L, D]
      patches: jax.Array | None,  # [B, N, P, D']
  ) -> jax.Array:
    assert self.vision_encoder is not None
    # get soft tokens
    encoder_output: jax.Array = self.vision_encoder(  # pylint: disable=unexpected-keyword-arg
        patches=patches, is_training=False
    )
    # project soft tokens
    vision_embeddings = einops.rearrange(
        encoder_output,
        'b media num_embeds patches -> (b media) num_embeds patches',
    )
    vision_embeddings = self.embedder.encode_vision(vision_embeddings)
    embeddings = jnp.place(
        arr=embeddings,
        mask=jnp.expand_dims(last_tokens, -1).repeat(
            embeddings.shape[-1], axis=-1
        )
        == gemma_vision.TOKEN_PLACEHOLDER,
        vals=vision_embeddings,
        inplace=False,
    )
    return embeddings

  if not typing.TYPE_CHECKING:

    def __getattr__(self, name: str):
      # It's convenient to be able to access the vision encoder directly.
      # However it has to be initialized in setup, so can't use a standard
      # `@property`
      if name == 'vision_encoder':
        return self.config.vision_encoder
      return super().__getattr__(name)

  else:  # For type checking / auto-complete

    @property
    def vision_encoder(self) -> gemma_vision.SigLiPFromPatches | None:
      return self.config.vision_encoder


def _check_tokens_for_vision(
    last_tokens: jax.Array,  # [B, L]
    patches: jax.Array | None = None,  # [B, N, P, D']
) -> None:
  """Checks if the last tokens are correctly formatted for vision tokens."""
  if patches is not None:
    correct_placeholder, start_positions = jax.vmap(
        gemma_vision.check_mask, in_axes=-1, out_axes=-1
    )(input_data=last_tokens)
    if not jnp.all(correct_placeholder):
      raise ValueError(
          'The last tokens \n\n%s\n\n are not correctly formatted for vision'
          ' tokens: in your input, you do not have contiguous sets of value'
          ' %d over %d tokens.'
          % (
              last_tokens,
              gemma_vision.TOKEN_PLACEHOLDER,
              gemma_vision.NUM_PLACEHOLDER_TOKENS_PER_IMAGE,
          )
      )
    for special_token, position_offset in [
        (gemma_vision.NEW_LINE_TOKEN, -2),
        (gemma_vision.BEGIN_IMAGE_TOKEN, -1),
        (
            gemma_vision.END_IMAGE_TOKEN,
            gemma_vision.NUM_PLACEHOLDER_TOKENS_PER_IMAGE,
        ),
        (
            gemma_vision.NEW_LINE_TOKEN,
            gemma_vision.NUM_PLACEHOLDER_TOKENS_PER_IMAGE + 1,
        ),
    ]:
      correct_special_token = gemma_vision.check_special_vision_token(
          input_data=last_tokens,
          start_positions=start_positions,
          special_token=special_token,
          position_offset=position_offset,
      )
      if not jnp.all(correct_special_token):
        raise ValueError(
            'The last tokens \n\n%s\n\n are not correctly formatted for'
            ' vision tokens: in your input, you do not have the correct'
            ' special token %d at position %d w.r.t. the image'
            ' placeholders %d.'
            % (
                last_tokens,
                special_token,
                position_offset,
                gemma_vision.TOKEN_PLACEHOLDER,
            )
        )


def compute_sequence_attention_mask(  # TODO(epot): Cleanup this function.
    time_step: jax.Array,
    *,
    seq_len: int,
    input_mask: jax.Array,
    bi_directional_mask: jax.Array | None = None,
) -> jax.Array:
  """Computes sequence attention mask."""
  bsz = input_mask.shape[0]
  attention_mask = jnp.tile(
      jnp.expand_dims(jnp.tri(N=int(time_step), M=int(seq_len)), axis=0),
      (bsz, 1, 1),
  )
  if bi_directional_mask is not None:
    bi_directional_mask = jnp.expand_dims(
        jnp.concatenate([
            bi_directional_mask[0],
            jnp.zeros((seq_len - len(bi_directional_mask))),
        ]),
        axis=0,
    )
    bi_directional_mask = jnp.tile(
        jnp.expand_dims(
            jnp.outer(bi_directional_mask, bi_directional_mask)[
                :time_step, :seq_len
            ],
            axis=0,
        ),
        (bsz, 1, 1),
    ).astype(jnp.bool_)
    attention_mask = jnp.logical_or(attention_mask, bi_directional_mask).astype(
        jnp.bool_
    )
  return attention_mask


def compute_attention_masks(
    time_step: jax.Array, seq_len: int, input_mask: jax.Array
) -> jax.Array:
  """Computes causal attention mask."""
  bsz = input_mask.shape[0]
  batch_time_step = jnp.full((bsz, 1), time_step, dtype=jnp.uint32)
  causal_mask = jnp.less_equal(
      jnp.expand_dims(jnp.arange(seq_len), 0), batch_time_step
  )
  max_seq_len = min(input_mask.shape[-1], seq_len)
  input_mask = jax.lax.dynamic_slice(
      input_mask,
      (0, jnp.maximum(time_step - seq_len + 1, 0)),
      (bsz, max_seq_len),
  )
  input_mask = (
      jnp.ones((bsz, seq_len), dtype=jnp.bool_)
      .at[:, :max_seq_len]
      .set(input_mask)
  )

  causal_mask = jnp.logical_and(causal_mask, input_mask)
  attention_mask = causal_mask[:, jnp.newaxis, :].astype(jnp.bool_)

  return attention_mask


def mm_input_length(patches: jax.Array | None) -> int:
  """Returns the number of multimodal tokens in the input."""
  if patches is not None:
    return (
        patches.shape[1] * gemma_vision.NUM_TOKENS_PER_MEDIA
    )  # NOTE: 256 tokens + begin/end img + '\n
  return 0


def make_causal_attn_mask(
    input_mask: jax.Array,  # Shape [B, L]
) -> jax.Array:
  """Makes a causal attention mask.

  I.e., as in middle diagram of Figure 3 in https://arxiv.org/pdf/1910.10683.

  Args:
    input_mask: Input mask for the input. True for non-padded tokens only, else
      False.

  Returns:
    Attention mask of shape [B, L, L] (where B=batch dim and L=sequence dim).
  """
  if len(input_mask.shape) != 2:
    raise ValueError(
        f'Input mask must be 2D (shape [B, L]), but got {input_mask.shape}.'
    )
  seq_len = input_mask.shape[-1]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
  attn_mask = input_mask[..., None, :]
  attn_mask *= causal_mask[None, ...]
  return attn_mask


def make_causal_with_prefix_attn_mask(
    input_mask: jax.Array,  # Shape [B, L]
    prefix_mask: jax.Array,  # Shape [B, L]
) -> jax.Array:
  """Makes a causal with prefix attention mask.

  I.e., as in the right diagram of Figure 3 in https://arxiv.org/pdf/1910.10683.

  Args:
    input_mask: Input mask for the input. True for non-padded tokens only, else
      False.
    prefix_mask: Input mask for the prefix. True for prefix tokens only, else
      False.

  Returns:
    Attention mask of shape [B, L, L] (where B=batch dim and L=sequence dim).
  """
  if len(input_mask.shape) != 2:
    raise ValueError(
        f'Input mask must be 2D (shape [B, L]), but got {input_mask.shape}.'
    )
  if len(prefix_mask.shape) != 2:
    raise ValueError(
        f'Prefix mask must be 2D (shape [B, L]), but got {prefix_mask.shape}.'
    )
  seq_len = input_mask.shape[-1]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
  prefix_mask = jnp.tile(jnp.expand_dims(prefix_mask, axis=1), [1, seq_len, 1])
  causal_or_prefix_mask = jnp.logical_or(causal_mask, prefix_mask)
  attn_mask = input_mask[..., None, :]
  attn_mask *= causal_or_prefix_mask
  return attn_mask


def make_block_mask(
    bidirectional_mask: jax.Array,  # [B, T]
) -> jax.Array:  # [B, T]
  """Creates block mask identifying segments based on a bidirectional mask.

  Args:
    bidirectional_mask: boolean mask, e.g. [011110011010].

  Returns:
    block mask for segments, e.g. [011110022030].
  """
  # Left pad 0.
  padded_mask = jnp.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0)
  boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
  numbered_boundary = jnp.cumsum(boundary, -1)
  return bidirectional_mask * numbered_boundary


def add_bidirectional_mask(
    attn_mask: jax.Array,  # [B, #L, L']
    bidirectional_mask: jax.Array,  # [B #L L'],
) -> jax.Array:
  """Adds bidirectional mask to the attention mask."""
  q_block_indices = make_block_mask(bidirectional_mask)
  kv_block_indices = q_block_indices
  attn_mask = jnp.logical_or(
      attn_mask,
      jnp.logical_and(
          kv_block_indices[:, None, :] == q_block_indices[..., None],
          q_block_indices[..., None] > 0,
      ),
  )
  return attn_mask


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
