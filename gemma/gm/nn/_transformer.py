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

"""Model."""

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Any, ClassVar

import einops
import flax
from flax import linen as nn
from gemma.gm.nn import _config
from gemma.gm.nn import _layers
from gemma.gm.nn import _modules
from gemma.gm.utils import _dtype_params
from gemma.gm.utils import _jax_utils
from gemma.gm.utils import _types
from gemma.gm.vision import _token_utils
from gemma.multimodal import vision as gemma_vision
import jax.numpy as jnp
from kauldron import kd
from kauldron import kontext
from kauldron.typing import Bool, Float, Int, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member

_PADDING_ID = 0


@dataclasses.dataclass(kw_only=True, frozen=True)
class ModelInfo:
  """Model information.

  Used to auto-load the model tokenizer and params.
  """

  tokenizer_version: int | str | None = None
  default_ckpt: str | None = None


@flax.struct.dataclass
class Output:
  """Output of the Gemma model.

  Attributes:
    logits: Predicted logits of the model.
    cache: Updated cache if the input cache is not None, None elsewhere.
    hidden_states: The hidden states of the model.
  """

  # When `return_last_only`, `logits` is `*B V`
  logits: Float['*B L V'] | Float['*B V']
  cache: _config.Cache | None
  hidden_states: Float['*B L D'] | Float['*B D'] | None


@flax.struct.dataclass
class _Inputs:
  """Inputs of the Gemma model, after encoding.

  Attributes:
    embeddings: Encoded tokens, including MM.
    positions: Input absolute positions.
    attention_mask: Transformer input mask.
    inputs_mask: Mask of the input tokens.
  """

  embeddings: Float['B L D']
  positions: Int['B L']
  attention_mask: Bool['B L cache_length']
  inputs_mask: Bool['B L']


class Transformer(nn.Module):
  """Base transformer class.

  Attributes:
    return_last_only: If `True`, only compute and return the last token.
      Otherwise, return all logits. Default to `False`
    dtype: The parameter dtype. Default to `jnp.bfloat16`.
  """
  _: dataclasses.KW_ONLY

  return_last_only: bool | None = None

  dtype: jnp.dtype = jnp.bfloat16

  # Keys to specify in the config which inputs to pass to the `__call__`
  # function (e.g. `tokens='batch.tokens'`).
  tokens: kontext.Key = kontext.REQUIRED
  images: kontext.Key | None = None
  positions: kontext.Key | None = None
  attention_mask: kontext.Key | None = None

  config: _config.TransformerConfig
  # Model info to specifiy the tokenizer version and default checkpoint.
  INFO: ClassVar[ModelInfo] = ModelInfo()

  def __post_init__(self):
    super().__post_init__()

  def setup(self):
    self.embedder = _modules.Embedder(
        vocab_size=self.config.num_embed,
        embed_dim=self.config.embed_dim,
        vision_proj_dim=self.config.vision_encoder.siglip_encoder.width
        if self.config.vision_encoder
        else None,
    )

    self.blocks = [
        _modules.Block(
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
            if attn_type == _modules.AttentionType.LOCAL_SLIDING
            else self.config.global_base_frequency,
            rope_scale_factor=self.config.local_scale_factor
            if attn_type == _modules.AttentionType.LOCAL_SLIDING
            else self.config.global_scale_factor,
        )
        for i, attn_type in zip(
            range(self.config.num_layers), self.config.attention_types
        )
    ]
    self.final_norm = _layers.RMSNorm()

    self.vision_encoder = self.config.vision_encoder

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

  # Calling `model.apply` on Colab makes the Kernel crash unless it is jitted.
  @functools.partial(
      nn.jit,
      static_argnames=(
          'self',
          'return_last_only',
          'return_hidden_states',
      ),
  )
  # The function accepts/returns aribtrary batch shape, but inside the
  # function, the batch dimension is flattened to a single dimension.
  @_jax_utils.flatten_unflatten_batch_dim()
  @typechecked
  def __call__(  # pytype: disable=signature-mismatch
      self,
      tokens: Int['*B L'],
      *,
      images: UInt8['*B N H W C'] | UInt8['*B H W C'] | None = None,
      # TODO(epot): Cleanup and simplify the API.
      # When provided, the positions and attention_mask should include
      # the extra inserted multi-modal tokens.
      positions: Int['*B L_with_mm'] | None = None,
      cache: _config.Cache | None = None,
      # During training and pre-filling, the attention mask is `*B L L`
      # When sampling (after prefilling), tokens are decoded one by one,
      # so the attention mask is `*B 1 cache_length`
      attention_mask: Bool['*B L_with_mm cache_length'] | None = None,
      return_last_only: bool | None = None,
      return_hidden_states: bool | None = None,
  ) -> Output:  # Output['*B']
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      tokens: input sequence of tokens.
      images: Images to feed to the vision encoder.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.
      return_last_only: If `True`, only compute and return the logits of the
        last input token in sequence. Useful for decoding where we don't need to
        compute logits for the whole sequence, but only for the last token.
        Otherwise, return all logits. Default to `False`.
      return_hidden_states: If `True`, return the hidden states of the model.
        Useful for developing custom models. Otherwise, return only the logits
        and the cache. Default to `False`.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    return_last_only = self._get_return_last_only(return_last_only)

    with _dtype_params.initialize_param_with_dtype(
        self.dtype,
        exclude=[
            # The multi-modal params are kept in float32.
            'vision_encoder',
            'embedder.mm_input_projection',
            'embedder.mm_soft_embedding_norm',
            # Skip the LoRA params
            'lora',
        ],
    ):

      # Encode the text tokens, eventually including the vision embeddings.
      inputs = self._encode_and_get_inputs(
          tokens=tokens,
          images=images,
          positions=positions,
          attention_mask=attention_mask,
      )
      del positions, attention_mask

      x, new_cache = self._apply_attention(inputs, cache)

    if return_last_only:
      last_input_token_idx = jnp.sum(inputs.inputs_mask, axis=-1) - 1
      # TODO(epot): Use `jnp.take_along_axis`
      x = x[jnp.arange(len(x)), last_input_token_idx, ...]
    elif images is not None:
      # Remove the MM extra tokens inserted.
      # During fine-tuning, the prompt is always masked, and the model cannot
      # generate images tokens, so the logits are meaningless anyway.
      x = _token_utils.remove_mm_logits(
          logits=x,
          tokens=tokens,
          num_tokens_per_image=self.config.vision_encoder.num_mm_tokens_per_image,  # pytype: disable=attribute-error
      )

    logits = self.embedder.decode(x)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return Output(
        logits=logits,
        cache=None if cache is None else new_cache,
        hidden_states=x if return_hidden_states else None,
    )

  def _apply_attention(
      self, inputs: _Inputs, cache: _config.Cache | None
  ) -> tuple[Float['*B L D'], _config.Cache]:
    """Runs the transformer blocks.

    Args:
      inputs: Input containing embeddings, attention mask, and positions.
      cache: Attention KV cache or None.

    Returns:
      Transformer(inputs.embeddings).
    """
    x = inputs.embeddings
    old_cache = cache or {}
    new_cache = {}
    for i, block in enumerate(self.blocks):
      layer_name = f'layer_{i}'
      layer_cache, x = block(
          x,
          inputs.positions,
          old_cache.get(layer_name),
          inputs.attention_mask,
      )
      new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)
    return x, new_cache

  @functools.partial(
      nn.jit,
      static_argnames=(
          'self',
          'batch_size',
          'dtype',
          'cache_length',
          'sharding',
      ),
  )
  def init_cache(
      self,
      *,
      batch_size: int,
      dtype: jnp.dtype[Any],
      cache_length: int,
      sharding: kd.sharding.ShardingTree | None = None,
  ) -> _config.Cache:
    cache = self.config.init_cache(
        batch_size=batch_size,
        dtype=dtype,
        cache_length=cache_length,
    )
    return kd.sharding.with_sharding_constraint(cache, sharding)

  @typechecked
  def _encode_and_get_inputs(
      self,
      *,
      tokens: Int['B L_no_mm'],
      images: UInt8['B H W C'] | UInt8['B N H W C'] | None = None,
      attention_mask: Bool['B L_with_mm cache_length'] | None = None,
      positions: Int['B L_with_mm'] | None = None,
  ) -> _Inputs:
    """Encode the text tokens, eventually including the vision embeddings."""

    # If the model has images, we expand each `<start_of_image>` token to add
    # the image placeholder tokens.
    if images is not None:
      self._assert_support_mm()
      if len(images.shape) == 4:  # Expand optional `num_images` dimension
        images = einops.rearrange(images, 'b h w c -> b 1 h w c')

    inputs = _types.Input(
        text=tokens,
        images=images,
        config=self.config.input_config,
    )
    del tokens, images

    # Encode the text tokens
    # Could this be optimized to filter out the `SOFT_TOKEN_PLACEHOLDER` ?
    # Currently, The placeholders are required so the mask, positions are
    # correctly computed.
    x = self.embedder.encode(inputs.tokens_with_mm)

    # Encode the vision tokens and merge them with the text embeddings.
    if inputs.images is not None:
      x = self._merge_mm_embeddings(
          tokens=inputs.tokens_with_mm, embeddings=x, images=inputs.images
      )
    elif self.vision_encoder is not None and self.is_initializing():
      # During initialization, call the vision encoder to ensure that the
      # params are correctly initialized.
      _ = self._encode_vision(_make_dummy_images(self.vision_encoder))

    # Note: When `positions` and `attention_mask` are explicitly provided,
    # it's the user responsibility to correctly take into account the extra
    # tokens inserted for the images.
    # This is what the `gm.text.Sampler` implementation does.
    if positions is None:
      positions = inputs.positions

    if attention_mask is None:
      attention_mask = inputs.attention_mask

    return _Inputs(
        embeddings=x,
        positions=positions,
        attention_mask=attention_mask,
        inputs_mask=inputs.inputs_mask,
    )

  @typechecked
  def _merge_mm_embeddings(
      self,
      *,
      tokens: Int['B L'],
      embeddings: Float['B L D'],
      images: UInt8['B N H W C'],
  ) -> Float['B L D']:
    """Update the embeddings to include the vision embeddings."""
    # Encode the images
    soft_embeddings = self._encode_vision(images)

    # Merge the soft tokens back with the text embeddings.
    merged_embeddings = _token_utils.merge_embeddings(
        text_embeddings=embeddings,
        vision_embeddings=soft_embeddings,
        mask=tokens == gemma_vision.TOKEN_PLACEHOLDER,
    )

    return merged_embeddings

  def _encode_vision(self, images: UInt8['B N H W C']) -> Float['B N P D']:
    """Encode the images into the same space as the text embeddings."""
    assert self.vision_encoder is not None
    patches = self.vision_encoder.patchify_images(images)
    soft_embeddings = self.vision_encoder(patches=patches, is_training=False)
    soft_embeddings = self.embedder.encode_vision(soft_embeddings)
    return soft_embeddings

  def _get_return_last_only(self, return_last_only: bool | None = None) -> bool:
    """Merge `return_last_only` from the config and input."""
    # TODO(epot): Could add `default=False` to `nn.merge_param`
    if return_last_only is None and self.return_last_only is None:
      return_last_only = False
    else:
      return_last_only = nn.merge_param(
          'return_last_only', return_last_only, self.return_last_only
      )
    return return_last_only

  def _assert_support_mm(self) -> None:
    if self.config.vision_encoder is None:
      msg = ''
      if getattr(self, 'text_only', False):
        msg = ' The model was created with `text_only=True`.'
      raise ValueError(
          f'The model {type(self).__name__!r} does not have vision encoder,'
          ' yet images are provided.'
          + msg
      )


def _make_dummy_images(
    vision_encoder: gemma_vision.SigLiPFromPatches,
) -> Float['B L P D']:
  """Make dummy images for initializing the vision encoder."""
  return jnp.zeros(
      (1, 1, vision_encoder.image_height, vision_encoder.image_width, 3),
      dtype=jnp.uint8,
  )
