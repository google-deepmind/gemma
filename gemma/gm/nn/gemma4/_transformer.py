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

"""Model."""

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Any, ClassVar

import flax
from flax import linen as nn
from gemma.gm.math import _pos_utils
from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _layers
from gemma.gm.nn.gemma4 import _modules
from gemma.gm.nn.gemma4.audio import _model as gemma4_audio_model
from gemma.gm.nn.gemma4.vision import _encoder as gemma4_vision
from gemma.gm.utils import _attention_mask
from gemma.gm.utils import _dtype_params
from gemma.gm.utils import _jax_utils
from gemma.gm.utils import _types
from gemma.gm.vision import _token_utils
import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron import kontext
from kauldron.ktyping import Bool, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member

_PADDING_ID = 0


@dataclasses.dataclass(frozen=True)
class PreprocessedVisionInput:
  patches: Any
  positions_xy: Any
  soft_token_counts: tuple[int, ...]


jax.tree_util.register_dataclass(
    PreprocessedVisionInput,
    data_fields=['patches', 'positions_xy'],
    meta_fields=['soft_token_counts'],
)


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
    attention_mask: Transformer input mask for global attention layers (causal).
    sliding_attention_mask: Attention mask for sliding window layers, or None.
      When set, sliding layers use this mask (which may include bidirectional
      attention for image tokens). When None, sliding layers use attention_mask.
    inputs_mask: Mask of the input tokens.
    per_layer_inputs: Optional per-layer inputs.
  """

  embeddings: Float['B L D']
  positions: Int['B L']
  attention_mask: Bool['B L cache_length']
  inputs_mask: Bool['B L']
  sliding_attention_mask: Bool['B L cache_length'] | None = None
  per_layer_inputs: Float['B L P'] | None = None


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

  config: _config.TransformerConfig
  # Model info to specifiy the tokenizer version and default checkpoint.
  INFO: ClassVar[ModelInfo] = ModelInfo()

  def __post_init__(self):
    super().__post_init__()

  def setup(self):
    self.embedder = _modules.Embedder(
        vocab_size=self.config.num_embed,
        embed_dim=self.config.embed_dim,
        vision_proj_dim=self.config.vision_encoder.d_model
        if self.config.vision_encoder
        else None,
        per_layer_input_dim=self.config.per_layer_input_dim,
        num_layers=self.config.num_layers,
        audio_proj_dim=self.config.audio_encoder.lm_model_dims
        if self.config.audio_encoder
        else None,
    )

    self.kv_cache_sharing_patterns = _config.create_kv_cache_sharing_patterns(
        self.config.kv_cache_sharing_config,
        self.config.num_layers,
        self.config.attention_types,
    )

    blocks = []
    for i, attn_type in zip(
        range(self.config.num_layers), self.config.attention_types
    ):
      blocks.append(
          _modules.Block(
              name=f'layer_{i}',
              num_heads=self.config.num_heads,
              num_kv_heads=self.config.num_kv_heads,
              embed_dim=self.config.embed_dim,
              head_dim=self.config.head_dim,
              hidden_dim=self.config.moe_dense_hidden_dim
              if self.config.enable_moe
              else self._get_hidden_dim(layer_number=i),
              sliding_window_size=self.config.sliding_window_size,
              use_post_attn_norm=self.config.use_post_attn_norm,
              use_post_ffw_norm=self.config.use_post_ffw_norm,
              attn_logits_soft_cap=self.config.attn_logits_soft_cap,
              attn_type=attn_type,
              qk_norm_with_scale=self.config.qk_norm_with_scale,
              num_global_kv_heads=self.config.num_global_kv_heads,
              global_key_size=self.config.global_key_size,
              k_eq_v_global=self.config.k_eq_v_global,
              global_rope_proportion=self.config.global_rope_proportion,
              local_rope_proportion=self.config.local_rope_proportion,
              rope_base_frequency=self.config.local_base_frequency
              if attn_type == _modules.AttentionType.LOCAL_SLIDING
              else self.config.global_base_frequency,
              rope_scale_factor=self.config.local_scale_factor
              if attn_type == _modules.AttentionType.LOCAL_SLIDING
              else self.config.global_scale_factor,
              per_layer_input_dim=self.config.per_layer_input_dim,
              enable_moe=self.config.enable_moe,
              num_experts=self.config.num_experts,
              expert_dim=self.config.expert_dim,
              top_k_experts=self.config.top_k_experts,
          )
      )
    self.blocks = blocks
    self.final_norm = _layers.RMSNorm()

    self.vision_encoder = self.config.vision_encoder

    if self.config.audio_encoder:
      self.audio_encoder = gemma4_audio_model.AudioTokenizer(
          config=self.config.audio_encoder
      )

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
    def vision_encoder(self) -> gemma4_vision.VisionEncoder | None:
      return self.config.vision_encoder

  # Calling `model.apply` on Colab makes the Kernel crash unless it is jitted.
  @functools.partial(
      nn.jit,
      static_argnames=(
          'self',
          'return_last_only',
          'return_hidden_states',
          'audio_soft_token_counts',
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
      images: PreprocessedVisionInput | None = None,
      audio=None,  # raw waveform [batch, samples] or None
      audio_lengths=None,  # [batch] valid sample counts or None
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
      audio_soft_token_counts: tuple[int, ...] | None = None,
  ) -> Output:  # Output['*B']
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      tokens: input sequence of tokens.
      images: Images to feed to the vision encoder.
      audio: Raw audio waveforms of shape [batch, num_clips, samples], or None
        if no audio input.
      audio_lengths: Valid sample counts per audio clip of shape [batch,
        num_clips], or None.
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
      audio_soft_token_counts: Tuple of per-audio-clip valid token counts after
        conformer encoding. Used to truncate each clip's embeddings to the
        correct length before merging into the text sequence.

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
            'embedder.mm_pre_projection_norm',
            'audio_encoder',
            'embedder.audio_input_projection',
            'embedder.audio_soft_embedding_norm',
            # Skip the LoRA params
            'lora',
        ],
    ):

      # Encode the text tokens, eventually including the vision embeddings.
      inputs = self._encode_and_get_inputs(
          tokens=tokens,
          images=images,
          audio=audio,
          audio_lengths=audio_lengths,
          positions=positions,
          attention_mask=attention_mask,
          audio_soft_token_counts=audio_soft_token_counts,
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
    per_layer_inputs = inputs.per_layer_inputs
    old_cache = cache or {}
    new_cache = {}
    for i, block in enumerate(self.blocks):
      layer_name = f'layer_{i}'
      if self._kv_sharing_enabled(layer_number=i):
        shared_layer_name = f'layer_{self.kv_cache_sharing_patterns[i]}'
        kv_shared_cache = new_cache.get(shared_layer_name)
      else:
        kv_shared_cache = None
      # Select the appropriate attention mask for this layer type.
      attn_mask = inputs.attention_mask
      if (
          inputs.sliding_attention_mask is not None
          and block.attn_type == _modules.AttentionType.LOCAL_SLIDING
      ):
        attn_mask = inputs.sliding_attention_mask
      layer_cache, x = block(
          x,
          inputs.positions,
          old_cache.get(layer_name),
          attn_mask,
          per_layer_inputs[..., i, :]
          if self.config.per_layer_input_dim
          else None,
          kv_shared_cache=kv_shared_cache,
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

  def _encode_and_get_inputs(
      self,
      *,
      tokens,
      images: PreprocessedVisionInput | None = None,
      audio=None,
      audio_lengths=None,
      attention_mask=None,
      positions=None,
      audio_soft_token_counts=None,
      ignore_ple_tokens: bool = False,
  ) -> _Inputs:
    """Encode the text tokens, eventually including the vision embeddings."""
    if images is not None or audio is not None:
      x = self.embedder.encode(tokens)

      if images is not None:
        x = self._merge_mm_embeddings(
            tokens=tokens, embeddings=x, images=images
        )

      if audio is not None:
        audio_embeds = self._encode_audio(
            audio, audio_lengths, audio_soft_token_counts
        )
        audio_mask = tokens == _token_utils.AUDIO_SOFT_TOKEN_PLACEHOLDER
        x = _token_utils.merge_flat_embeddings(
            text_embeddings=x,
            multimodal_embeddings=audio_embeds,
            mask=audio_mask,
        )

      inputs_mask = tokens != _PADDING_ID
      if positions is None:
        positions = _pos_utils.build_positions_from_mask(inputs_mask)
      if attention_mask is None:
        attention_mask = (
            _attention_mask.make_causal_bidirectional_attention_mask(
                inputs_mask,
                bidirectional_mask=None,
            )
        )

      # For models with use_bidirectional_attention='vision' (26B_A4B/31B),
      # create a separate sliding attention mask with bidirectional attention
      # for image tokens within the same image block.
      sliding_attention_mask = None
      if self.config.use_bidirectional_attention == 'vision':
        bidirectional_mask = tokens == _token_utils.SOFT_TOKEN_PLACEHOLDER
        sliding_attention_mask = (
            _attention_mask.make_causal_bidirectional_attention_mask(
                inputs_mask,
                bidirectional_mask=bidirectional_mask,
            )
        )

      if self.config.per_layer_input_dim:
        per_layer_inputs = self.embedder.encode_per_layer_input(
            x, tokens, ignore_ple_tokens=ignore_ple_tokens
        )
      else:
        per_layer_inputs = None

      return _Inputs(
          embeddings=x,
          positions=positions,
          attention_mask=attention_mask,
          inputs_mask=inputs_mask,
          sliding_attention_mask=sliding_attention_mask,
          per_layer_inputs=per_layer_inputs,
      )

    inputs = _types.Input(
        text=tokens,
        images=None,
        config=self.config.input_config,
    )
    del tokens

    x = self.embedder.encode(inputs.tokens_with_mm)

    if self.vision_encoder is not None and self.is_initializing():
      dummy_patches, dummy_positions = _make_dummy_images(self.vision_encoder)
      _ = self.vision_encoder(dummy_patches, dummy_positions)

    if positions is None:
      positions = inputs.positions

    if attention_mask is None:
      attention_mask = inputs.attention_mask

    if self.config.per_layer_input_dim:
      per_layer_inputs = self.embedder.encode_per_layer_input(
          x, inputs.tokens_with_mm, ignore_ple_tokens=ignore_ple_tokens
      )
    else:
      per_layer_inputs = None

    return _Inputs(
        embeddings=x,
        positions=positions,
        attention_mask=attention_mask,
        inputs_mask=inputs.inputs_mask,
        per_layer_inputs=per_layer_inputs,
    )

  def _merge_mm_embeddings(
      self,
      *,
      tokens,
      embeddings,
      images: PreprocessedVisionInput,
  ):
    """Update the embeddings to include the vision embeddings."""
    soft_embeddings = self._encode_vision(images)
    mask = tokens == gemma4_vision.TOKEN_PLACEHOLDER

    merged_embeddings = _token_utils.merge_flat_embeddings(
        text_embeddings=embeddings,
        multimodal_embeddings=soft_embeddings,
        mask=mask,
    )

    return merged_embeddings

  def _encode_vision(self, vision_input: PreprocessedVisionInput):
    """Encode images into the same space as the text embeddings."""
    assert self.vision_encoder is not None

    n_images = len(vision_input.soft_token_counts)
    patches = vision_input.patches
    positions_xy = vision_input.positions_xy
    max_patches = patches.shape[1] // n_images

    patches = jnp.reshape(patches, (n_images, max_patches, patches.shape[2]))
    positions_xy = jnp.reshape(
        positions_xy, (n_images, max_patches, positions_xy.shape[2])
    )

    encoder_outputs = self.vision_encoder(patches, positions_xy)

    embeddings, mask = encoder_outputs[0]

    per_image_tokens = []
    for i in range(n_images):
      expected_count = vision_input.soft_token_counts[i]
      if mask is not None:
        valid_indices = jnp.nonzero(mask[i], size=expected_count)[0]
        real_tokens = embeddings[i][valid_indices]
      else:
        real_tokens = embeddings[i][:expected_count]
      per_image_tokens.append(real_tokens)

    all_tokens = jnp.concatenate(per_image_tokens, axis=0)
    all_tokens = self.embedder.encode_vision(all_tokens[None, None, :, :])
    all_tokens = all_tokens[:, 0, :, :]
    return all_tokens

  def _encode_audio(self, audio, audio_lengths, audio_soft_token_counts):
    """Encode audio waveforms and project to text embedding space.

    Processes audio through the conformer-based audio encoder, projects the
    resulting embeddings into the text embedding space, then truncates each
    clip to its valid token count and concatenates.

    Args:
      audio: Raw audio waveforms of shape [1, num_clips, samples]. The leading
        batch dim is squeezed so the encoder receives [num_clips, samples].
      audio_lengths: Valid sample counts of shape [1, num_clips].
      audio_soft_token_counts: Tuple of valid token counts per clip after
        conformer encoding and subsampling.

    Returns:
      Audio embeddings of shape [1, total_valid_tokens, embed_dim], ready to
      be merged into the text token sequence.
    """

    assert self.audio_encoder is not None
    audio = jnp.asarray(audio)[0]
    audio_lengths = jnp.asarray(audio_lengths)[0]

    audio_embeddings, _ = self.audio_encoder(audio, audio_lengths)

    all_tokens = self.embedder.encode_audio(audio_embeddings)

    results = []
    for i, count in enumerate(audio_soft_token_counts):
      results.append(all_tokens[i, :count])
    concatenated = jnp.concatenate(results, axis=0)
    return concatenated[None]

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

  def _kv_sharing_enabled(self, *, layer_number: int) -> bool:
    """Check if the layer has KV sharing."""
    return (
        self.config.kv_cache_sharing_config is not None
        and self.kv_cache_sharing_patterns is not None
        and layer_number != self.kv_cache_sharing_patterns[layer_number]
    )

  def _get_hidden_dim(self, *, layer_number: int) -> int:
    """Get the hidden ffw dimension for the layer."""
    if (
        self._kv_sharing_enabled(layer_number=layer_number)
        and self.config.override_kv_shared_ffw_hidden is not None
    ):
      return self.config.override_kv_shared_ffw_hidden
    else:
      return self.config.hidden_dim


def _make_dummy_images(
    vision_encoder: gemma4_vision.VisionEncoder,
):
  """Make dummy patches/positions for initializing the vision encoder."""
  max_patches = vision_encoder.max_patches
  patch_dim = vision_encoder.patch_size**2 * 3
  dummy_patches = jnp.zeros((1, max_patches, patch_dim), dtype=jnp.float32)
  dummy_positions = jnp.full((1, max_patches, 2), -1, dtype=jnp.int32)
  return dummy_patches, dummy_positions
