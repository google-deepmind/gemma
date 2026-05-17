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

"""Audio Tokenizer module for Gemma models."""

import flax.linen as nn
from gemma.gm.nn.gemma4.audio import _modules as audio_modules
import jax.numpy as jnp
from kauldron import typing
from kauldron.typing import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member


typechecked = typing.typechecked


class AudioTokenizer(nn.Module):
  """Conformer-based audio encoder.

  Takes in a batch of audio waveforms and their corresponding sequence lengths,
  and returns encoded audio embeddings with a padding mask.
  """

  config: audio_modules.ConformerConfig

  @staticmethod
  def infer_mask(
      x: jnp.ndarray, sequence_lengths: jnp.ndarray, original_seq_len: int
  ) -> jnp.ndarray:
    """Infer boolean validity mask after temporal compression.

    Args:
      x: Tensor with compressed time dimension in axis 1.
      sequence_lengths: Original sequence lengths per batch element.
      original_seq_len: Original time dimension before compression.

    Returns:
      Boolean mask of shape [batch, compressed_time], True for valid positions.
    """
    compressed_seq_len = x.shape[1]
    compression_rate = original_seq_len / compressed_seq_len
    new_sequence_lengths = jnp.floor(
        sequence_lengths / compression_rate
    ).astype(jnp.int32)

    indices = jnp.arange(compressed_seq_len)[jnp.newaxis, :]
    mask = indices < new_sequence_lengths[:, jnp.newaxis]
    return mask

  @staticmethod
  def _compute_causal_valid_mask(config: audio_modules.ConformerConfig):
    """Computes the local causal validity mask for chunked attention."""
    chunk_size = audio_modules.LocalDotProductAttention.block_size
    max_future_horizon = config.atten_right_context
    max_past_horizon = max(0, config.atten_left_context - 1)
    context_size = chunk_size + max_past_horizon + max_future_horizon
    upper_diagonal = max_past_horizon + max_future_horizon

    lower_causal_mask = audio_modules.LocalDotProductAttention._ones_matrix_band_part(  # pylint: disable=protected-access
        context_size,
        chunk_size,
        num_lower=-1,
        num_upper=0,
        out_dtype=jnp.bool_,
    ).T
    upper_causal_mask = audio_modules.LocalDotProductAttention._ones_matrix_band_part(  # pylint: disable=protected-access
        chunk_size,
        context_size,
        num_lower=-1,
        num_upper=upper_diagonal,
        out_dtype=jnp.bool_,
    )
    causal_valid_mask = lower_causal_mask & upper_causal_mask
    return causal_valid_mask

  def to_float32(self, audio_data: jnp.ndarray):
    if audio_data.dtype == jnp.int16:
      return audio_data.astype(jnp.float32) / 32768.0
    elif audio_data.dtype == jnp.int32:
      return audio_data.astype(jnp.float32) / 2147483648.0
    elif audio_data.dtype == jnp.uint8:
      return (audio_data.astype(jnp.float32) - 128.0) / 128.0
    elif audio_data.dtype in [jnp.float16, jnp.float32]:
      return audio_data.astype(jnp.float32)
    else:
      raise ValueError(f'Unsupported format: {audio_data.dtype}')

  @typechecked
  @nn.compact
  def __call__(
      self,
      x: Array['batch samples'],
      sequence_lengths: Int['batch'],
  ) -> tuple[Float['batch seq_len model_dims'], jnp.ndarray]:
    """Computes audio embeddings.

    Args:
      x: Input audio waveforms.
      sequence_lengths: Length of each sequence in the batch.

    Returns:
      Tuple of (audio_embeddings, padding_mask) where padding_mask is a
      boolean array with True indicating padding positions.
    """
    x = self.to_float32(x)

    original_seq_len = x.shape[-1]

    x = audio_modules.GemaxMelFilterbank(sample_rate=16000, n_mels=128)(x)

    mask = self.infer_mask(x, sequence_lengths, original_seq_len)
    x = jnp.where(mask[:, :, jnp.newaxis], x, 0.0)
    x, mask = audio_modules.SubSamplingBlock(
        output_proj_dim=self.config.model_dims, name='feature'
    )(x, mask)
    causal_valid_mask = self._compute_causal_valid_mask(self.config)

    for i in range(self.config.num_layers):
      x = audio_modules.ConformerLayer(
          config=self.config,
          name=f'conformer/stacked_layers_{i}',
          layer_idx=i,
      )(x, mask, causal_valid_mask)

    if self.config.conf_reduction_factor > 1:
      x = x[:, :: self.config.conf_reduction_factor]
      mask = mask[:, :: self.config.conf_reduction_factor]

    x = nn.Dense(
        features=self.config.lm_model_dims,
        use_bias=True,
        name='output_projection',
    )(x)

    padding_mask = ~mask
    x = jnp.where(padding_mask[:, :, jnp.newaxis], 0.0, x)

    return x, padding_mask
