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

"""Model."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, ClassVar

import flax
from flax import linen as nn
from gemma import transformer
from gemma.gm.utils import _dtype_params
from gemma.gm.utils import _jax_utils
from gemma.gm.vision import _token_utils
from gemma.multimodal import vision as gemma_vision
import jax.numpy as jnp
from kauldron import kontext
from kauldron.typing import Float, Int, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member

_PADDING_ID = 0


@dataclasses.dataclass(kw_only=True, frozen=True)
class ModelInfo:
  """Model information.

  Used to auto-load the model tokenizer and params.
  """

  tokenizer_version: int | None = None
  default_ckpt: str | None = None


@flax.struct.dataclass
class Output:
  """Output of the Gemma model.

  Attributes:
    logits: Predicted logits of the model.
    cache: Updated cache if the input cache is not None, None elsewhere.
  """

  # When `return_last_only`, `logits` is `*B V`
  logits: Float['*B L V'] | Float['*B V']
  cache: transformer.Cache | None


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
  attention_mask: Int['B L L']
  inputs_mask: Int['B L']


# TODO(epot): Merge this class with `transformer.Transformer`
class Transformer(transformer.Transformer):
  """Base transformer class.

  Attributes:
    return_last_only: If `True`, only compute and return the last token.
      Otherwise, return all logits. Default to `False`
    dtype: The parameter dtype. Default to `jnp.bfloat16`.
  """

  return_last_only: bool | None = None

  dtype: jnp.dtype = jnp.bfloat16

  # Keys to specify in the config which inputs to pass to the `__call__`
  # function (e.g. `tokens='batch.tokens'`).
  tokens: kontext.Key = kontext.REQUIRED
  positions: kontext.Key | None = None
  cache: kontext.Key | None = None
  attention_mask: kontext.Key | None = None

  # Model info to specifiy the tokenizer version and default checkpoint.
  INFO: ClassVar[ModelInfo] = ModelInfo()

  def __post_init__(self):
    # TODO(epot): Config should not have `max_cache_length` parameter as
    # this is a sampling argument independent of the model architecture.
    # Also rather than inheriting from Transformer, could try unify the API
    # in a single class.
    if self.config.max_cache_length is not None:
      raise ValueError(
          'The config `max_cache_length` should be None. Got:'
          f' {self.config.max_cache_length}. Instead, the cache size is set'
          ' directly in the sampler.'
      )
    super().__post_init__()

  # Calling `model.apply` on Colab makes the Kernel crash unless it is jitted.
  @functools.partial(
      nn.jit,
      static_argnames=(
          'self',
          'return_last_only',
      ),
  )
  # The function accepts/returns aribtrary batch shape, but inside the
  # function, the batch dimension is flattened to a single dimension.
  @_jax_utils.flatten_unflatten_batch_dim()
  # TODO(epot): Restore the `@typechecked` annotation. Currently disable it
  # because during sampling, `attention_mask` is `[*B 1 1024]`, which is
  # not `*B L L`
  # @typechecked
  def __call__(  # pytype: disable=signature-mismatch
      self,
      tokens: Int['*B L'],
      *,
      images: UInt8['*B N H W C'] | None = None,
      positions: Int['*B L'] | None = None,
      cache: transformer.Cache | None = None,
      attention_mask: Int['*B L L'] | None = None,
      return_last_only: bool | None = None,
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

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    return_last_only = self._get_return_last_only(return_last_only)

    with _dtype_params.initialize_param_with_dtype(self.dtype):

      # Encode the text tokens, eventually including the vision embeddings.
      inputs = self._encode_and_get_inputs(
          tokens=tokens,
          images=images,
          positions=positions,
          attention_mask=attention_mask,
      )
      del tokens, images, positions, attention_mask

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

    if return_last_only:
      last_input_token_idx = jnp.sum(inputs.inputs_mask, axis=-1) - 1
      # TODO(epot): Use `jnp.take_along_axis`
      x = x[jnp.arange(len(x)), last_input_token_idx, ...]

    logits = self.embedder.decode(x)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return Output(
        logits=logits,
        cache=None if cache is None else new_cache,
    )

  def init_cache(
      self,
      *,
      batch_size: int,
      dtype: jnp.dtype[Any],
      cache_length: int,
  ) -> transformer.Cache:
    return self.config.init_cache(
        batch_size=batch_size,
        dtype=dtype,
        cache_length=cache_length,
    )

  def _encode_and_get_inputs(
      self,
      *,
      tokens: Int['B L'],
      images: UInt8['B N H W C'] | None = None,
      attention_mask: Int['*B L L'] | None = None,
      positions: Int['*B L'] | None = None,
  ) -> _Inputs:
    """Encode the text tokens, eventually including the vision embeddings."""

    # If the model has images, we expand each `<start_of_image>` token to add
    # the image placeholder tokens.
    if images is not None:
      tokens = _token_utils.add_extra_tokens_for_images(
          tokens,
          max_num_images=images.shape[1],
          num_tokens_per_image=self.vision_encoder.num_mm_tokens_per_image,
      )

    # Encode the text tokens
    # Could this be optimized to filter out the `SOFT_TOKEN_PLACEHOLDER` ?
    # Currently, The placeholders are required so the mask, positions are
    # correctly computed.
    x = self.embedder.encode(tokens)

    # Encode the vision tokens and merge them with the text embeddings.
    if images is not None:
      x = self._merge_mm_embeddings(tokens=tokens, embeddings=x, images=images)
    elif self.vision_encoder is not None and self.is_initializing():
      # During initialization, call the vision encoder to ensure that the
      # params are correctly initialized.
      dummy_patches = _make_dummy_patches(self.vision_encoder)
      _ = self.vision_encoder(patches=dummy_patches, is_training=False)

    # Compute the mask (after the extra tokens are added)
    inputs_mask = jnp.array(tokens != _PADDING_ID, dtype=jnp.int32)

    # Note: When `positions` and `attention_mask` are explicitly provided,
    # it's the user responsibility to correctly take into account the extra
    # tokens inserted for the images.
    # This is what the `gm.text.Sampler` implementation does.
    if positions is None:
      positions = transformer.build_positions_from_mask(inputs_mask)
    if attention_mask is None:
      attention_mask = transformer.make_causal_attn_mask(inputs_mask)

      # Add the bidirectional mask for images.
      if images is not None:
        bidirectional_mask = tokens == gemma_vision.TOKEN_PLACEHOLDER
        attention_mask = transformer.add_bidirectional_mask(
            attention_mask, bidirectional_mask
        )

    return _Inputs(
        embeddings=x,
        positions=positions,
        attention_mask=attention_mask,
        inputs_mask=inputs_mask,
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
    if self.vision_encoder is None:
      msg = ''
      if getattr(self, 'text_only', False):
        msg = ' The model was created with `text_only=True`.'
      raise ValueError(
          f'The model {type(self).__name__!r} does not have vision encoder,'
          ' yet images are provided.'
          + msg
      )

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


def _make_dummy_patches(
    vision_encoder: gemma_vision.SigLiPFromPatches,
) -> Float['B L P D']:
  """Make dummy patches for initializing the vision encoder."""
  patch_height, _ = vision_encoder.siglip_encoder.patch_size
  num_patches_one_side = vision_encoder.image_height // patch_height
  num_channels = 3 * patch_height**2
  num_patches = num_patches_one_side**2
  return jnp.zeros(
      shape=(1, 1, num_patches, num_channels),
      dtype=jnp.float32,
  )
