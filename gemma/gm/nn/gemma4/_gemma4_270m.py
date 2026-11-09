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

"""Internal Gemma 4 270M preset with product-quantized PLE and extra MM tokens."""

import dataclasses

from flax import linen as nn
from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _gemma4
from gemma.gm.nn.gemma4 import _layers
from gemma.gm.nn.gemma4 import _modules
from gemma.gm.nn.gemma4.audio import _modules as gemma4_audio
from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
import jax
import jax.numpy as jnp

_NUM_LAYERS_GEMMA4_270M = 24
_MM_EXTRA_TOKENS = 128
# SentencePiece `<image|>` token ID routed to `mm_input_embedding_extra[0]` in
# `_Tiny270MEmbedder.encode` and zeroed in
# `_Tiny270MEmbedder.encode_per_layer_input`.
_EOI_SPM_TOKEN_ID = 258882
# SentencePiece `<audio|>` token ID routed to `audio_input_embedding_extra[0]`
# in `_Tiny270MEmbedder.encode` and zeroed in
# `_Tiny270MEmbedder.encode_per_layer_input`.
_EOA_SPM_TOKEN_ID = 258883
# Number of entries per product-quantized PLE sub-codebook (`512 * 2 = 1024`
# rows) and sub-vectors across both halves (`5 * 2 = 10` forming a `5 * 5 = 25`
# layer grid).
_PQ_PLE_CODES_PER_BOOK = 512
_PQ_PLE_SUB_VECTORS = 10


class _Tiny270MEmbedder(_modules.Embedder):
  """Embedder with product-quantized PLE and extra multimodal tokens."""

  num_mm_extra_tokens: int = _MM_EXTRA_TOKENS

  def setup(self):
    self.input_embedding_table = self.param(
        'input_embedding',
        nn.initializers.normal(),
        (self.vocab_size, self.embed_dim),
    )
    if self.vision_proj_dim:
      self.mm_input_projection = _layers.Einsum(
          (self.vision_proj_dim, self.embed_dim)
      )
      self.mm_pre_projection_norm = _layers.RMSNorm(with_scale=False)
      if self.num_mm_extra_tokens:
        self.mm_input_embedding_extra = self.param(
            'mm_input_embedding_extra',
            nn.initializers.zeros,
            (self.num_mm_extra_tokens, self.vision_proj_dim),
        )
    if self.audio_proj_dim:
      self.audio_input_projection = _layers.Einsum(
          (self.audio_proj_dim, self.embed_dim)
      )
      self.audio_soft_embedding_norm = _layers.RMSNorm(with_scale=False)
      if self.num_mm_extra_tokens:
        self.audio_input_embedding_extra = self.param(
            'audio_input_embedding_extra',
            nn.initializers.zeros,
            (self.num_mm_extra_tokens, self.audio_proj_dim),
        )
    if self.per_layer_input_dim:
      ple_shape = (
          _PQ_PLE_CODES_PER_BOOK * 2,
          _PQ_PLE_SUB_VECTORS,
          self.per_layer_input_dim,
      )
      self.per_layer_input_embedding_table = self.param(
          'per_layer_embeddings',
          nn.initializers.normal(),
          ple_shape,
      )
      self.per_layer_model_projection = _layers.Einsum(
          (self.embed_dim, self.num_layers, self.per_layer_input_dim),
          w_scale=(float(self.embed_dim) ** -0.5),
      )
      self.per_layer_projection_norm = _layers.RMSNorm()

  def encode(self, x: jax.Array) -> jax.Array:
    if self.vision_proj_dim is not None and self.num_mm_extra_tokens:
      in_vocab = jnp.logical_and(x >= 0, x < self.vocab_size)
      safe_tokens = jnp.where(in_vocab, x, 0)
      embeddings = self.input_embedding_table[(safe_tokens,)]
      embeddings *= jnp.sqrt(self.embed_dim).astype(embeddings.dtype)
      is_eoi = x == _EOI_SPM_TOKEN_ID
      extra_ids = jnp.where(
          is_eoi,
          0,
          jnp.clip(x - self.vocab_size, 0, self.num_mm_extra_tokens - 1),
      )
      scaled = self.mm_input_embedding_extra[(extra_ids,)] * jnp.sqrt(
          self.vision_proj_dim
      ).astype(self.mm_input_embedding_extra.dtype)
      projected = self.mm_input_projection('...m,md->...d', scaled).astype(
          embeddings.dtype
      )
      in_extra = (x >= self.vocab_size) & (
          x < self.vocab_size + self.num_mm_extra_tokens
      )
      result = jnp.where((is_eoi | in_extra)[..., None], projected, embeddings)
      if self.audio_proj_dim is not None:
        audio_offset = self.vocab_size + self.num_mm_extra_tokens
        is_eoa = x == _EOA_SPM_TOKEN_ID
        in_audio_extra = (x >= audio_offset) & (
            x < audio_offset + self.num_mm_extra_tokens
        )
        audio_extra_ids = jnp.where(
            is_eoa,
            0,
            jnp.clip(x - audio_offset, 0, self.num_mm_extra_tokens - 1),
        )
        audio_extra_emb = self.audio_input_embedding_extra[(audio_extra_ids,)]
        audio_extra_proj = self.audio_input_projection(
            '...m,md->...d', audio_extra_emb
        ).astype(embeddings.dtype)
        result = jnp.where(
            (is_eoa | in_audio_extra)[..., None], audio_extra_proj, result
        )
      return result
    return super().encode(x)

  def encode_per_layer_input(
      self,
      x: jax.Array,
      t: jax.Array,
      ignore_ple_tokens: bool = False,
  ) -> jax.Array:
    keep_ple = jnp.logical_and(t >= 0, t < self.vocab_size)
    if self.num_mm_extra_tokens:
      keep_ple = jnp.logical_and(
          keep_ple, (t != _EOI_SPM_TOKEN_ID) & (t != _EOA_SPM_TOKEN_ID)
      )
    safe_tokens = jnp.where(keep_ple, t, jnp.zeros_like(t))
    x = self.per_layer_model_projection('...td,dnp->...tnp', x)
    x = self.per_layer_projection_norm(x)
    if ignore_ple_tokens:
      return x
    table = self.per_layer_input_embedding_table
    first_sub, second_sub = jnp.split(table, 2, axis=1)
    layer_grid = (first_sub[:, :, None, :] + second_sub[:, None, :, :]).reshape(
        table.shape[0], -1, self.per_layer_input_dim
    )[:, : self.num_layers, :]
    first_book, second_book = jnp.split(layer_grid, 2, axis=0)
    first_codes = (
        safe_tokens // _PQ_PLE_CODES_PER_BOOK
    ) % _PQ_PLE_CODES_PER_BOOK
    second_codes = safe_tokens % _PQ_PLE_CODES_PER_BOOK
    ple_tokens = first_book[(first_codes,)] + second_book[(second_codes,)]
    ple_tokens *= jnp.sqrt(self.per_layer_input_dim).astype(ple_tokens.dtype)
    return (x + ple_tokens) * jax.lax.rsqrt(2.0).astype(x.dtype)


class Gemma4_270M(_gemma4._Gemma4Base):  # pylint: disable=invalid-name,protected-access
  """Gemma 4 270M model.

  Attributes:
    text_only: If True, skips media encoders to save memory and compute.
    config: Model configuration.
  """

  text_only: bool = False

  attention_pattern = (_modules.AttentionType.LOCAL_SLIDING,) * 5 + (
      _modules.AttentionType.GLOBAL,
  )
  global_local_pattern = _config.make_attention_layers_types(
      attention_pattern, num_layers=_NUM_LAYERS_GEMMA4_270M
  )

  config: _config.TransformerConfig = dataclasses.replace(
      _gemma4._gemma4_config(  # pylint: disable=protected-access
          embed_dim=512,
          num_heads=4,
          num_kv_heads=2,
          num_global_kv_heads=1,
          per_layer_input_dim=512,
          frac_shared_layers=0.0,
          attention_types=global_local_pattern,
          sliding_window_size=1024,
          k_eq_v_global=False,
          use_post_attn_norm=True,
          use_post_ffw_norm=True,
          override_ffw_hidden_for_kv_cache_sharing=False,
          vision_encoder=gemma_vision.VisionEncoder(use_clipped_linears=True),
          audio_encoder=gemma4_audio.ConformerConfig(use_key_scale=False),
      ),
      global_rope_proportion=1.0,
      final_logit_softcap=None,
  )

  def _make_embedder(self) -> _modules.Embedder:
    return _Tiny270MEmbedder(
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
