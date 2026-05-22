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

"""Contains Flax modules for the Gemma audio encoder."""

import dataclasses
from typing import Optional

import flax.linen as nn
from gemma.gm.nn.gemma4 import _layers
import jax
import jax.numpy as jnp
from kauldron import typing
from kauldron.typing import Bool, Float  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np


ClippedEinsum = _layers.ClippedEinsum
typechecked = typing.typechecked


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConformerConfig:
  """Configuration for the Conformer model."""

  num_layers: int = 12
  model_dims: int = 1024
  lm_model_dims: int = 1536
  atten_num_heads: int = 8
  atten_left_context: int = 13
  atten_right_context: int = 0
  conv_kernel_size: int = 5
  gradient_clipping: float = 10_000_000_000.0
  conf_reduction_factor: int = 1
  param_dtype: jnp.dtype = jnp.float32
  compute_dtype: Optional[jnp.dtype] = None


class SubSamplingBlock(nn.Module):
  """Subsampling block for the Conformer model."""

  filters_list: tuple[int, ...] = (128, 32)
  kernel_size_list: tuple[tuple[int, int], ...] = ((3, 3), (3, 3))
  strides_list: tuple[tuple[int, int], ...] = ((2, 2), (2, 2))
  output_proj_dim: int = 1024  # Default value
  use_dense_bias: bool = False
  use_conv_bias: bool = False
  use_norm_scale: bool = True
  compute_dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @typechecked
  @nn.compact
  def __call__(
      self, x: Float['batch time features'], mask: Bool['batch time']
  ) -> tuple[Float['batch new_time output_dim'], Bool['batch new_time']]:
    """Applies two conv+LayerNorm+ReLU subsampling stages and a projection.

    Each convolution stage reduces the time dimension by its stride factor.
    The validity mask is downsampled accordingly. Finally, the frequency and
    channel dimensions are projected to `output_proj_dim`.

    Args:
      x: Input features of shape [batch, time, features].
      mask: Boolean validity mask of shape [batch, time], True = valid.

    Returns:
      Tuple of (subsampled_features, subsampled_mask).
    """

    x = jnp.expand_dims(x, -1)

    # Two subsampling layers (Conv+LN+ReLu)
    x = nn.Conv(
        features=self.filters_list[0],
        kernel_size=self.kernel_size_list[0],
        strides=self.strides_list[0],
        padding=((1, 1), (1, 1)),
        use_bias=self.use_conv_bias,
        dtype=self.compute_dtype,
        param_dtype=self.param_dtype,
        feature_group_count=1,
        name='subsampling_0',
    )(x)
    mask = mask[:, :: self.strides_list[0][0]][:, : x.shape[1]]
    x = nn.LayerNorm(
        name='norm_0', use_bias=False, use_scale=True, use_fast_variance=False
    )(x)

    x = nn.relu(x)

    x = nn.Conv(
        features=self.filters_list[1],
        kernel_size=self.kernel_size_list[1],
        strides=self.strides_list[1],
        padding=((1, 1), (1, 1)),
        use_bias=self.use_conv_bias,
        dtype=self.compute_dtype,
        param_dtype=self.param_dtype,
        feature_group_count=1,
        name='subsampling_1',
    )(x)
    mask = mask[:, :: self.strides_list[1][0]][:, : x.shape[1]]
    x = nn.LayerNorm(
        name='norm_1', use_bias=False, use_scale=True, use_fast_variance=False
    )(x)

    x = nn.relu(x)

    # Dense layer to prepare input for the conformer stack.
    x = nn.DenseGeneral(
        features=self.output_proj_dim,
        axis=(-2, -1),
        use_bias=self.use_dense_bias,
        name='input_proj',
    )(x)
    return x, mask


class FFNBlock(nn.Module):
  """A Flax implementation of the residual FFN block."""

  # Define model dimensions and dtypes as class attributes
  config: ConformerConfig
  ffn_residual_weight: float = 0.5

  @typechecked
  @nn.compact
  def __call__(
      self, x: Float['batch seq_len model_dims']
  ) -> Float['batch seq_len model_dims']:
    """Defines the forward pass of the FFN block.

    The @nn.compact decorator allows us to define layers directly within the
    call.

    Args:
      x: The input tensor to the FFN block.

    Returns:
      The output tensor after passing through the FFN block.
    """
    # Store the original input for the residual connection
    residual = x

    x = jnp.clip(
        x, -self.config.gradient_clipping, self.config.gradient_clipping
    )

    # 1. Pre-Normalization
    y = nn.RMSNorm(
        dtype=self.config.compute_dtype,
        param_dtype=self.config.param_dtype,
        use_scale=True,
        use_fast_variance=False,
        name='pre_layer_norm',
    )(x)

    # 2. FFN Up-projection with Swish activation
    y = ClippedEinsum(
        shape=(self.config.model_dims, self.config.model_dims * 4),
        weight_name='kernel',
        dtype=self.config.param_dtype,
        name='ffn_layer1',
    )('...D,DF->...F', y)

    y = nn.swish(y)

    # 3. FFN Down-projection
    y = ClippedEinsum(
        shape=(self.config.model_dims * 4, self.config.model_dims),
        weight_name='kernel',
        dtype=self.config.param_dtype,
        name='ffn_layer2',
    )('...D,DF->...F', y)

    y = jnp.clip(
        y, -self.config.gradient_clipping, self.config.gradient_clipping
    )

    # 4. Post-Normalization
    y = nn.RMSNorm(
        dtype=self.config.compute_dtype,
        param_dtype=self.config.param_dtype,
        use_scale=True,
        use_fast_variance=False,
        name='post_layer_norm',
    )(y)

    # 5. Scaling and 6. Residual Connection
    output = residual + y * self.ffn_residual_weight

    return output


class AttentionBlock(nn.Module):
  """The final residual block wrapping the local attention mechanism."""

  config: ConformerConfig

  @typechecked
  @nn.compact
  def __call__(
      self,
      x: Float['batch seq_len model_dims'],
      mask: jnp.ndarray,
      causal_valid_mask: jnp.ndarray,
  ) -> Float['batch seq_len model_dims']:
    residual = x

    x = jnp.clip(
        x, -self.config.gradient_clipping, self.config.gradient_clipping
    )

    # 1. Pre-Normalization
    y = nn.RMSNorm(
        param_dtype=self.config.param_dtype,
        use_scale=True,
        use_fast_variance=False,
        name='pre_norm',
    )(x)

    # Local Attention
    y = LocalDotProductAttention(
        atten_num_heads=self.config.atten_num_heads,
        units_per_head=self.config.model_dims // self.config.atten_num_heads,
        atten_left_context=self.config.atten_left_context,
        atten_right_context=self.config.atten_right_context,
        model_dims=self.config.model_dims,
        param_dtype=self.config.param_dtype,
        compute_dtype=self.config.compute_dtype,
        name='self_atten',
    )(y, mask, causal_valid_mask)

    y = ClippedEinsum(
        shape=(
            self.config.atten_num_heads,
            self.config.model_dims // self.config.atten_num_heads,
            self.config.model_dims,
        ),
        weight_name='kernel',
        dtype=self.config.param_dtype,
        name='post',
    )('...NH,NHD->...D', y)

    y = jnp.clip(
        y, -self.config.gradient_clipping, self.config.gradient_clipping
    )

    # Post-Normalization (inside the residual branch)
    y = nn.RMSNorm(
        param_dtype=self.config.param_dtype,
        use_scale=True,
        use_fast_variance=False,
        name='post_norm',
    )(y)

    return residual + y


class LightweightConvBlock(nn.Module):
  """A Flax implementation of the residual lightweight convolutional block."""

  config: ConformerConfig

  @typechecked
  @nn.compact
  def __call__(
      self, x: Float['batch seq_len model_dims']
  ) -> Float['batch seq_len model_dims']:
    """Defines the forward pass of the block."""
    residual = x

    # 1. Pre-Normalization
    y = nn.RMSNorm(
        param_dtype=self.config.param_dtype,
        use_scale=True,
        use_fast_variance=False,
        name='ln',
    )(x)

    # 2. Projection and Gated Linear Unit (GLU)
    # Project to twice the model dimension
    gated_input = ClippedEinsum(
        shape=(self.config.model_dims, 2 * self.config.model_dims),
        weight_name='kernel',
        dtype=self.config.param_dtype,
        name='linear_start',
    )('...D,DF->...F', y)
    # Apply GLU. jax.nn.glu splits the last axis, applies sigmoid to the
    # second half, and multiplies it with the first half.
    y = nn.glu(gated_input)

    # 3. Causal Depthwise 1D Convolution
    # feature_group_count=model_dims makes it a depthwise convolution.
    # padding='CAUSAL' ensures no future information is used.
    y = nn.Conv(
        features=self.config.model_dims,
        kernel_size=(self.config.conv_kernel_size,),
        strides=(1,),
        padding='CAUSAL',
        feature_group_count=self.config.model_dims,
        use_bias=False,
        param_dtype=self.config.param_dtype,
        dtype=self.config.compute_dtype,
        name='depthwise_conv1d',
    )(y)

    y = jnp.clip(
        y, -self.config.gradient_clipping, self.config.gradient_clipping
    )

    # 4. Normalization and Activation
    y = nn.RMSNorm(
        param_dtype=self.config.param_dtype,
        use_scale=True,
        use_fast_variance=False,
        name='conv_norm',
    )(y)

    y = nn.swish(y)

    # 5. Final Projection
    y = ClippedEinsum(
        shape=(self.config.model_dims, self.config.model_dims),
        weight_name='kernel',
        dtype=self.config.param_dtype,
        name='linear_end',
    )('...D,DF->...F', y)

    # 6. Residual Connection
    return residual + y


class ConformerLayer(nn.Module):
  """A single layer of the Conformer model."""

  config: ConformerConfig
  layer_idx: int = -1

  @typechecked
  @nn.compact
  def __call__(
      self,
      x: Float['batch seq_len model_dims'],
      mask: jnp.ndarray,
      causal_valid_mask: jnp.ndarray,
  ) -> Float['batch seq_len model_dims']:
    """Applies one Conformer layer: FFN → Attention → LConv → FFN → Norm.

    Args:
      x: Input of shape [batch, seq_len, model_dims].
      mask: Boolean validity mask [batch, seq_len], True = valid.
      causal_valid_mask: Local causal attention mask [block_size, context_size].

    Returns:
      Output of shape [batch, seq_len, model_dims].
    """

    # First FFN Block, scaled by 0.5
    x = FFNBlock(
        config=self.config, ffn_residual_weight=0.5, name='fflayer_start'
    )(x)
    # Self-Attention Block
    x = AttentionBlock(config=self.config, name='trans_atten')(
        x, mask, causal_valid_mask
    )
    validity_mask = mask[:, :, jnp.newaxis].astype(x.dtype)
    x = x * validity_mask
    # Convolution Block
    x = LightweightConvBlock(config=self.config, name='lconv')(x)
    # Second FFN Block, scaled by 0.5
    x = FFNBlock(
        config=self.config, ffn_residual_weight=0.5, name='fflayer_end'
    )(x)
    x = jnp.clip(
        x, -self.config.gradient_clipping, self.config.gradient_clipping
    )
    # Final normalization for the layer
    x = nn.RMSNorm(
        param_dtype=self.config.param_dtype,
        name='final_ln',
        use_fast_variance=False,
    )(x)
    return x


class LocalDotProductAttention(nn.Module):
  """Local dot-product self-attention with Transformer-XL relative embeddings.

  This function uses parts of the code from LocalDotProductAttention from
  the SequenceLayers library. The latter is made for streaming attention and
  requires a specific kind of tensor manipulation. In order to match exactly
  output of the internal implementation of the AudioTokenizer,
  we perform those manipulation here even though they are not theoritically
  necessary. This results in a complexified implementation of a theoritically
  simple attention mechanism.
  """

  atten_num_heads: int
  units_per_head: int
  model_dims: int
  atten_left_context: int
  atten_right_context: int = 0
  attention_logits_soft_capping: float = 50.0
  block_size: int = 12
  param_dtype: jnp.dtype = jnp.float32
  compute_dtype: Optional[jnp.dtype] = None

  @staticmethod
  def _extract_block_context(
      x: jnp.ndarray,
      block_size: int,
      left_context: int,
      right_context: int,
      padding_val: float | jnp.bool_ = 0.0,
  ) -> jnp.ndarray:
    """Extracts temporal context for every block.

    Args:
      x: a tensor of [batch, time, ...].
      block_size: int. Number of time frames in a block.
      left_context: int. Left context size.
      right_context: int. Right context size.
      padding_val: float. value on the padded frames.

    Returns:
      A tensor of [batch, num_blocks, context_size, ...], with necessary
      paddings, where context_size = block_size + left_context + right_context
      and output[:, i, ...] are x[:, start-left_context:end+right_context, ..]
      start = i * block_size, end = (i + 1) * block_size.
    """
    if block_size < 1:
      raise ValueError(f'{block_size=} must be at least 1.')
    if left_context < 0:
      raise ValueError(f'{left_context=} must be >= 0.')
    if right_context < 0:
      raise ValueError(f'{right_context=} must be >= 0.')

    # Pad outside of signal.frame so that we get the desired left/right
    # context and padding behavior.
    paddings = [(0, 0)] * len(x.shape)
    paddings[1] = (left_context, right_context + block_size - 1)
    x = jnp.pad(x, paddings, constant_values=jnp.asarray(padding_val, x.dtype))

    frame_length = block_size + left_context + right_context
    frame_step = block_size
    num_frames = (x.shape[1] - frame_length) // frame_step + 1

    start_indices = jnp.arange(num_frames) * frame_step
    relative_indices = jnp.arange(frame_length)
    indices = start_indices[:, jnp.newaxis] + relative_indices[jnp.newaxis, :]

    return jnp.take(x, indices, axis=1)

  @staticmethod
  def _convert_to_block(
      x: jnp.ndarray, block_size: int, padding_val: float = 0.0
  ) -> jnp.ndarray:
    """Turns a sequence to non overlapping blocks.

    Args:
      x: a tensor of [batch, time, ...].
      block_size: int. Number of time frames in a block.
      padding_val: float. value on the padded frames.

    Returns:
      A tensor of [batch, num_blocks, block_size, ..], with necessary paddings
      where output[:, i, ...] are x[:, i*block_size:(i+1)*block_size, ...].
    """
    shape = x.shape
    b, t = shape[0], shape[1]
    if block_size < 1:
      raise ValueError(f'{block_size=} must be at least 1.')
    # Pad it to be a multiple of w.
    num_blocks = (t + block_size - 1) // block_size
    pad_length = num_blocks * block_size - t

    if pad_length > 0:
      paddings = [[0, 0]] * len(shape)
      paddings[1] = [0, pad_length]
      x = jnp.pad(x, paddings, constant_values=jnp.array(padding_val, x.dtype))
    reshaped = jnp.reshape(x, (b, num_blocks, block_size) + shape[2:])
    return reshaped

  @staticmethod
  def _ones_matrix_band_part(
      rows: int,
      cols: int,
      num_lower: int,
      num_upper: int,
      out_dtype: jnp.dtype = jnp.float32,
      out_shape: Optional[tuple[int, ...]] = None,
  ) -> jnp.ndarray:
    """Matrix band part of ones."""
    m = jnp.arange(rows).reshape((rows, 1))
    n = jnp.arange(cols).reshape((1, cols))

    mask_lower = True
    if num_lower >= 0:
      mask_lower = (m - n) <= num_lower

    mask_upper = True
    if num_upper >= 0:
      mask_upper = (n - m) <= num_upper

    band = jnp.logical_and(mask_lower, mask_upper).astype(out_dtype)

    if out_shape:
      band = jnp.reshape(band, out_shape)

    return band

  @typechecked
  @nn.compact
  def __call__(
      self,
      x: Float['batch seq_len model_dims'],
      mask: jnp.ndarray,
      causal_valid_mask: jnp.ndarray,
  ) -> Float['batch seq_len num_heads units_per_head']:

    batch_size, seq_len, _ = x.shape

    # --- 1. Input Projections for Q, K, V ---
    q = ClippedEinsum(
        shape=(self.model_dims, self.model_dims),
        weight_name='kernel',
        dtype=self.param_dtype,
        name='query',
    )('...D,DF->...F', x)

    k = ClippedEinsum(
        shape=(self.model_dims, self.model_dims),
        weight_name='kernel',
        dtype=self.param_dtype,
        name='key',
    )('...D,DF->...F', x)

    v = ClippedEinsum(
        shape=(self.model_dims, self.model_dims),
        weight_name='kernel',
        dtype=self.param_dtype,
        name='value',
    )('...D,DF->...F', x)

    # Reshape and transpose for multi-head computation
    q = q.reshape(
        batch_size, seq_len, self.atten_num_heads, self.units_per_head
    ).astype('float32')
    k = k.reshape(
        batch_size, seq_len, self.atten_num_heads, self.units_per_head
    ).astype('float32')
    v = v.reshape(
        batch_size, seq_len, self.atten_num_heads, self.units_per_head
    ).astype('float32')

    # Scaling queries with learned scales
    per_dim_scale = self.param(
        'per_dim_scale', nn.initializers.ones, self.units_per_head
    )
    r_softplus_0 = 1.442695041
    query_scale = jnp.array(
        r_softplus_0 / jnp.sqrt(self.units_per_head), dtype=q.dtype
    )
    q *= query_scale * jax.nn.softplus(per_dim_scale.astype(q.dtype))

    key_scale = jnp.array(
        r_softplus_0 * jax.nn.softplus(jnp.ones(())), dtype=k.dtype
    )
    k *= key_scale

    q = q.astype('float32')
    k = k.astype('float32')

    batch_size, original_query_time = q.shape[:2]
    k = self._extract_block_context(
        k,
        self.block_size,
        max(0, self.atten_left_context - 1),
        self.atten_right_context,
    )
    q = self._convert_to_block(q, block_size=self.block_size)

    # Position-based scores (bias)
    logits = TransformerXLRelativePositionEmbedding(
        atten_num_heads=self.atten_num_heads,
        units_per_head=self.units_per_head,
        model_dims=self.model_dims,
        atten_left_context=self.atten_left_context,
        param_dtype=self.param_dtype,
        name='relative_position_embedding',
    )(q, k)

    # Logits capping
    logits = self.attention_logits_soft_capping * jax.nn.tanh(
        logits / self.attention_logits_soft_capping
    )

    num_query_blocks = q.shape[1]

    # Squeeze the heads and query time out to get [b, keys_time].
    # valid_mask_blocked is [b, num_blocks, context_size]
    valid_mask_blocked = self._extract_block_context(
        mask,
        self.block_size,
        max(0, self.atten_left_context - 1),
        self.atten_right_context,
        padding_val=jnp.bool_(False),
    )

    # Reshape to [b, h=1, num_blocks, block_size=1, context_size].
    valid_mask_blocked = valid_mask_blocked[:, jnp.newaxis, :, jnp.newaxis, :]

    valid_mask_blocked = jnp.logical_and(
        valid_mask_blocked,
        causal_valid_mask[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :],
    )

    logits = jnp.where(
        valid_mask_blocked,
        logits,
        jnp.asarray(-1e9, dtype=logits.dtype),
    )
    probabilities = jax.nn.softmax(logits, axis=-1).astype('float32')

    # [B, U, C, N, H]
    values_blocks = self._extract_block_context(
        v,
        self.block_size,
        max(0, self.atten_left_context - 1),
        self.atten_right_context,
    )

    # Contract the context windows dimension (c) into per-head context
    # vectors across each local block:
    # [batch, num_query_blocks, block_size, num_heads, units_per_head].
    context_vectors = jnp.einsum(
        'BNuwc,BucNH->BuwNH',
        probabilities,
        values_blocks.astype('float32'),
        precision='float32',
    )

    context_vectors = jnp.reshape(
        context_vectors,
        [
            batch_size,
            num_query_blocks * self.block_size,
            self.atten_num_heads,
            self.units_per_head,
        ],
    )

    return context_vectors[:, :original_query_time]


class TransformerXLRelativePositionEmbedding(nn.Module):
  """Implements the relative position embedding logic from Transformer-XL.

  This module calculates the position-based logits (terms B from the paper)
  to be added to the content-based logits in the attention mechanism.

  Term (B): Content-dependent position bias (Query @ Relative_Position)
  Term (D): Position-dependent position bias
    (Global_Pos_Bias_v @ Relative_Position)

  The parent attention layer is responsible for calculating:
  Term (A): Content-based logits (Query @ Key)
  Term (C): Content-dependent position bias (Global_Pos_Bias_u @ Key)
  """

  atten_num_heads: int
  units_per_head: int
  model_dims: int
  atten_left_context: int
  atten_right_context: int = 0
  use_bias: bool = False  # True is not implemented
  param_dtype: jnp.dtype = jnp.float32
  compute_dtype: Optional[jnp.dtype] = None

  def setup(self):
    """Initializes the learnable parameters for relative attention."""

    assert (
        self.atten_right_context == 0
    ), 'Not yet implemented for right context'

    # Ensure model dimensions are consistent
    if self.model_dims != self.atten_num_heads * self.units_per_head:
      raise ValueError(
          f'model_dims ({self.model_dims}) must equal '
          f'atten_num_heads ({self.atten_num_heads}) * units_per_head '
          f'({self.units_per_head})'
      )

    # Projection matrix for relative positional embeddings
    # (W_r in the paper)
    self.pos_proj = nn.DenseGeneral(
        features=(self.atten_num_heads, self.units_per_head),
        dtype=self.param_dtype,
        use_bias=False,
        kernel_init=nn.initializers.glorot_uniform(),
        name='pos_proj',
    )

    if self.use_bias:
      raise NotImplementedError('Positional biases are not implemented yet')

  @staticmethod
  def _get_timing_signal_1d_pos(
      position: jnp.ndarray,
      channels: int,
      min_timescale: float = 1.0,
      max_timescale: float = 1.0e4,
      dtype: jnp.dtype = jnp.float32,
  ) -> jnp.ndarray:
    """Sinusoidal position embeddings with explicit positions."""
    position = jnp.asarray(position, jnp.float32)
    num_timescales = channels // 2
    log_timescale_increment = jnp.log(
        float(max_timescale) / float(min_timescale)
    ) / max(num_timescales - 1, 1)
    inv_timescales = min_timescale * jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
    )
    scaled_time = (
        position[:, :, jnp.newaxis]
        * inv_timescales[jnp.newaxis, jnp.newaxis, :]
    )
    timing_signal = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2
    )
    timing_signal = jnp.pad(timing_signal, [[0, 0], [0, 0], [0, channels % 2]])
    return timing_signal.astype(dtype)

  @typechecked
  def __call__(
      self,
      queries: Float[
          'batch num_query_blocks block_size num_heads units_per_head'
      ],
      keys: Float[
          'batch num_query_blocks context_size num_heads units_per_head'
      ],
  ) -> Float['batch num_heads num_query_blocks block_size context_size']:
    """Calculates the relative positional attention scores.

    Args:
        queries: The query tensor, shaped `(batch_size, num_query_blocks,
          block_size, num_heads, units_per_head)`.
        keys: The key tensor, shaped `(batch_size, num_query_blocks,
          context_size, num_heads, units_per_head)`.

    Returns:
        The calculated relative positional attention scores. Shape:
        `(batch_size,
        num_heads, num_query_blocks, block_size, context_size)`.
    """

    # Compute term_ac.
    term_ac = jnp.einsum(
        'BuwNH,BucNH->BNuwc',
        queries,
        keys,
        precision='highest',
    )

    b = queries.shape[0]
    u = queries.shape[1]
    w = queries.shape[2]
    c = keys.shape[2]
    n = self.atten_num_heads
    l = max(0, self.atten_left_context - 1)
    r = self.atten_right_context
    assert c == w + l + r

    pos = jnp.arange(l, -r - 1, -1)[jnp.newaxis, :]
    assert pos.shape[1] == l + r + 1

    # [1, F, position_bias_dim]
    sin_emb = self._get_timing_signal_1d_pos(
        pos,
        self.model_dims,
        min_timescale=1,
        max_timescale=10000,
        dtype=queries.dtype,
    )
    # [1, F, N, H]
    sin_emb = self.pos_proj(sin_emb)
    # [F, N, H]
    sin_emb = jnp.squeeze(sin_emb, 0)

    # [B, N, U, W, F]
    term_bd = jnp.einsum(
        'BuwNH,FNH->BNuwF',
        queries,
        sin_emb,
        precision='float32',
    )
    # Perform relative shift in order to get [B, N, U, W, C]
    # Pads the input to [B, N, U, W, C + 1]
    term_bd = jnp.pad(
        term_bd, ((0, 0), (0, 0), (0, 0), (0, 0), (0, (c + 1) - (l + r + 1)))
    )
    term_bd = jnp.reshape(term_bd, [b, n, u, w * (c + 1)])
    term_bd = term_bd[:, :, :, : w * c]
    # Reshapes to [B, N, U, W, C]. Note the output last dim is 1-smaller
    # than the input, which "pushses" one element off to the next row for each
    # row. The accumulated effect is row_i is right-shifted i steps (i>=0).
    term_bd = jnp.reshape(term_bd, [b, n, u, w, c])

    return term_ac + term_bd


class GemaxMelFilterbank(nn.Module):
  """A Flax module to compute Mel-filterbanks from a raw audio waveform.

  This module is designed to be JIT-compilable and compatible with Flax models.
  """

  # Parameters for the transformation, these cannot be changed
  sample_rate: int = 16000
  win_length: int = 320
  hop_length: int = 160
  subframe_factor: int = 160
  n_mels: int = 128
  f_min: float = 0
  f_max: float = 8000
  num_mel_bins: float = 128
  constant: float = 0.001

  def hertz_to_mel(self, freq):
    """Mel scale used in Gemma 4: htk."""
    return 2595.0 * np.log10(1.0 + (freq / 700.0))

  def mel_to_hertz(self, mels):
    """Mel scale used in Gemma 4: htk."""
    return 700.0 * (np.power(10, mels / 2595.0) - 1.0)

  def _create_triangular_filter_bank(self, fft_freqs, filter_freqs):
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(0.0, np.minimum(down_slopes, up_slopes))

  def linear_to_mel_weight_matrix(self) -> jnp.ndarray:
    """Inspired from `tf.signal.linear_to_mel_weight_matrix` but allows a custom hertz_to_mel function.

    Returns:
      An array of shape `[num_spectrogram_bins, num_mel_bins]`.
    Raises:
      ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are
        not positive, `lower_edge_hertz` is negative, frequency edges are
        incorrectly ordered, `upper_edge_hertz` is larger than the Nyquist
        frequency.
    """

    num_spectrogram_bins = int(self.n_fft / 2) + 1
    nyquist_hertz = self.sample_rate / 2.0
    linear_frequencies = np.linspace(
        0.0, nyquist_hertz, num_spectrogram_bins, dtype=np.float64
    )

    mel_min = self.hertz_to_mel(self.f_min)
    mel_max = self.hertz_to_mel(self.f_max)
    mel_freqs = np.linspace(mel_min, mel_max, int(self.num_mel_bins) + 2)
    filter_freqs = self.mel_to_hertz(mel_freqs)

    mel_weights_matrix = self._create_triangular_filter_bank(
        linear_frequencies, filter_freqs
    )

    return jnp.array(mel_weights_matrix.T.astype(np.float32))

  def hann_window(
      self, window_length: int, periodic: bool, nonzero: bool = False
  ) -> jnp.ndarray:
    """Computes a raised cosine window ported from tf.signal.

    Jax Hanning window is not periodic

    Args:
      window_length: The length of the window.
      periodic: Whether the window is periodic.
      nonzero: If True, uses a +0.5 offset so the window never touches zero at
        endpoints (matches HF's hanning_nonzero).

    Returns:
      A `jnp.ndarray` containing the Hann window.
    """

    if nonzero:
      arg = jnp.pi * 2.0 / window_length
      return 0.5 - (
          0.5
          * jnp.cos(arg * (jnp.arange(window_length, dtype=jnp.float32) + 0.5))
      )

    a = 0.5
    b = 1 - a
    even = 1 - window_length % 2
    n = jnp.asarray(window_length + int(periodic) * even - 1, dtype=jnp.float32)
    count = jnp.arange(window_length, dtype=jnp.float32)
    cos_arg = 2 * jnp.pi * count / n
    hann_values = a - b * jnp.cos(cos_arg)
    return hann_values

  def setup(self):
    """Initializes non-trainable parameters and pre-computes constants.

    This method is called once by Flax when the module is initialized.
    """
    assert self.win_length > self.hop_length

    # NFFT is next power of two of win_length
    self.n_fft = int(2 ** np.ceil(np.log2(self.win_length)))

    # 1. Custom Hanning windows because jax Hann window is not periodic
    self.window = self.hann_window(self.win_length, True, True)

    # 2. Pre-compute the Mel filter-bank matrix.
    self.mel_basis = self.linear_to_mel_weight_matrix()[
        jnp.newaxis, :, :
    ].transpose(0, 2, 1)

  def sl_signal_frame(self, x: jnp.ndarray) -> jnp.ndarray:
    # This function is a streamlined re-implementation of tf.signal.frame with
    # that only allows the SEMICAUSAL padding mode as defined in SequenceLayers
    # utils/convolution_explicit_padding.py

    axis = -1 % x.ndim
    outer_dimensions = x.shape[:axis]
    inner_dimensions = x.shape[axis + 1 :]

    frame_length = self.win_length
    frame_step = self.hop_length

    # Padding left and right for semi causality
    pad_left = frame_length - frame_step
    pad_right = frame_step - 1
    num_frames = (x.shape[axis] + frame_step - 1) // frame_step

    # Performance optimization: If frame_length and frame_step have common
    # factors, we can reduce the number of gather indices by dividing frames in
    # "subframes". This can speed up framing by up by 10-20x if frame_length is
    # divisible by frame_step.
    pad_right += (-x.shape[axis] - pad_left - pad_right) % frame_length

    paddings = [(0, 0)] * x.ndim
    paddings[axis] = (pad_left, pad_right)
    x = jnp.pad(x, paddings, constant_values=jnp.array(0, dtype=x.dtype))

    assert x.shape[axis] % self.subframe_factor == 0, (
        x.shape,
        self.subframe_factor,
    )

    x = x.reshape(
        outer_dimensions + (-1, self.subframe_factor) + inner_dimensions
    )

    # After extracting subframe_factor from the frame axis, divide the length
    # and step appropriately.
    frame_length = frame_length // self.subframe_factor
    frame_step = frame_step // self.subframe_factor

    # Build a [num_frames, frame_length] array of indices to take from x along
    # the frame axis, where selector[i] corresponds to the range [frame_step *
    # i, frame_step * i + frame_length).
    start_indices = (jnp.arange(num_frames) * frame_step)[:, jnp.newaxis]
    window_indices = jnp.arange(frame_length)[jnp.newaxis, :]
    x = jnp.reshape(
        jnp.take(x, indices=start_indices + window_indices, axis=axis),
        outer_dimensions
        + (-1, frame_length * self.subframe_factor)
        + inner_dimensions,
    )

    x = x.squeeze(axis=1)
    return x

  # @chex.chexify
  @typechecked
  def __call__(
      self, waveform: Float['batch samples']
  ) -> Float['batch frames n_mels']:
    """Takes a raw waveform and computes the Mel-filterbank spectrogram.

    Args:
        waveform: The input audio signal, shape: (batch, samples).

    Returns:
        The Mel-filterbank spectrogram, shape: (batch, frames, n_mels).
    """
    waveform = waveform.reshape(waveform.shape[0], 1, -1)
    assert len(waveform.shape) == 3, 'Must be [batch, 1, seq_len]'
    assert waveform.shape[1] == 1, 'Must be 1'

    frame_size_for_unfold = self.win_length + 1
    seq_len = waveform.shape[-1]
    num_frames = (seq_len - frame_size_for_unfold) // self.hop_length + 1

    start_indices = (jnp.arange(num_frames) * self.hop_length)[:, jnp.newaxis]
    window_indices = jnp.arange(frame_size_for_unfold)[jnp.newaxis, :]
    indices = start_indices + window_indices

    frames = waveform[:, 0, :][:, indices]
    frames = frames[..., :-1]

    # Apply the window function to each frame
    windowed_frames = frames * self.window

    # 3. Short-Time Fourier Transform (STFT)
    # We use rfft for real-valued inputs for efficiency
    stft_spectrogram = jnp.fft.rfft(windowed_frames, n=self.n_fft)

    # 4. Compute Spectrogram instead of power spectrogram
    spectrogram = jnp.abs(stft_spectrogram)

    # 5. Apply the Mel Filterbank to batched audio
    batch_size = spectrogram.shape[0]
    mel_basis = jnp.repeat(self.mel_basis, batch_size, axis=0)
    mel_spectrogram = spectrogram @ mel_basis

    # Adding a constant value
    mel_spectrogram += self.constant

    # Natural log instead of log10
    mel_spectrogram = jnp.log(mel_spectrogram)
    return mel_spectrogram
