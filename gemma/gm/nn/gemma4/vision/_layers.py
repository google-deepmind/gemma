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

"""Base layers for Gemma4 vision models."""

from typing import cast

from flax import linen as nn
from gemma.gm.nn.gemma4._layers import Einsum
from gemma.gm.nn.gemma4.vision import _images
from gemma.gm.nn.gemma4.vision import _utils
import jax
import jax.numpy as jnp
from kauldron.ktyping import Bool, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member

POSITIONS_PAD_VALUE = -1


class VisionEntry(nn.Module):
  """The vision entry layer.

  Attributes:
    d_model: The model dimension.
    patch_size: The size to patchify images.
    pos_emb_shape_yx: The shape of the positional embedding. It will be
      bilinearly resized to match the input image size.
    projectable: Whether to use a projection layer to convert the input to the
      same dimension as the positional embedding.
  """

  d_model: int
  patch_size: int
  pos_emb_shape_yx: tuple[int, int]

  def setup(self):
    self.input_projection = Einsum(
        shape=(self.patch_size * self.patch_size * 3, self.d_model)
    )
    pos_emb_init = nn.initializers.normal(stddev=0.02)

    # essential for FACTORIZED embedding
    assert self.pos_emb_shape_yx[-1] == 2, f'{self.pos_emb_shape_yx=}'

    self.pos_emb_param = self.param(
        'pos_emb',
        pos_emb_init,
        (self.pos_emb_shape_yx[0], self.pos_emb_shape_yx[1], self.d_model),
        jnp.float32,
    )

  @typechecked
  def __call__(
      self,
      images_or_patches: Float['B H W C'] | Float['B L P'],
      positions_xy: Int['B L 2'] | None = None,
  ) -> Float['B L D']:

    if images_or_patches.ndim == 4:
      # Patchify inputs. Assume constant aspect ratio.
      patches, positions_xy = _images.patchify(
          images_or_patches, self.patch_size
      )
    else:
      patches = images_or_patches
      assert patches.ndim == 3
      assert positions_xy is not None

    patches = 2 * (patches - 0.5)

    x = self.input_projection('btm,md->btd', patches)

    pos_embed = _images.factorized_posemb(
        cast(jax.Array, self.pos_emb_param), positions_xy
    ).astype(x.dtype)

    return x + pos_embed


class VisionExit(nn.Module):
  """Vision exit layer with scaling and optional spatial pooling.

  Responsible for pooling and exit scaling. This layer:
  1. Optionally downsamples via spatial pooling to target output length(s)
  2. Scales by sqrt(d_model) with learnable scale parameter

  Attributes:
    d_model: The model dimension.
    output_length: The embed will be spatially avg-pooled to this output length.
      Can be an int or tuple of ints for multiple output lengths.
  """

  d_model: int
  output_length: int | tuple[int, ...] = 256
  param_dtype: jnp.dtype = jnp.float32

  @typechecked
  def _maybe_downsample(
      self,
      x: Float['B L D'],
      *,
      positions_xy: Int['B L 2'] | None = None,
      length: int,
  ) -> tuple[Float['B l D'], Bool['B l'] | None]:
    cur_length = x.shape[1]
    if cur_length == length:
      if positions_xy is None:
        mask = jnp.ones(x.shape[:-1], dtype=jnp.bool_)
      else:
        mask = jnp.logical_not(
            (positions_xy == POSITIONS_PAD_VALUE).all(axis=-1)
        )
      return x, mask

    # Downsample via spatial pooling
    if positions_xy is not None:
      # Position-based pooling
      x_pooled, mask = _utils.avg_pool_by_positions(
          x, positions_xy=positions_xy, length=length
      )
      return x_pooled, mask

    # Grid-based pooling when positions not provided
    cur_width = int(cur_length**0.5)
    if cur_width**2 != cur_length:
      raise ValueError(f'x.shape[1]={cur_length} must be a perfect square.')

    output_width = int(length**0.5)
    if output_width**2 != length:
      raise ValueError(f'{length=} must be a perfect square.')

    if cur_width % output_width != 0:
      raise ValueError(f'{cur_width=} must be divisible by {output_width=}.')

    # Reshape to 2D grid
    x_2d = x.reshape(x.shape[0], cur_width, cur_width, x.shape[-1])

    # Apply average pooling
    window = cur_width // output_width
    window_shape = (window, window)
    x_2d = nn.avg_pool(x_2d, window_shape=window_shape, strides=window_shape)

    # Reshape back to sequence
    x_pooled = x_2d.reshape(x.shape[0], length, x.shape[-1])
    mask = jnp.ones(x_pooled.shape[:-1], dtype=jnp.bool_)
    return x_pooled, mask

  @typechecked
  def _single_call(
      self,
      x: Float['B L D'],
      *,
      positions_xy: Int['B L 2'] | None = None,
      length: int,
  ) -> tuple[Float['B l D'], Bool['B l'] | None]:
    """Apply vision exit processing for a single output length."""
    x, mask = self._maybe_downsample(
        x, positions_xy=positions_xy, length=length
    )

    x = x * jnp.sqrt(self.d_model)

    return x, mask

  @typechecked
  def __call__(
      self,
      x: Float['B L D'],
      *,
      positions_xy: Int['B L 2'] | None = None,
      output_length_overrides: tuple[int, ...] | None = None,
  ) -> tuple[tuple[Float['B l D'], Bool['B l'] | None], ...]:
    """Apply vision exit processing.

    Args:
      x: Input embeddings [batch, seq_len, d_model].
      positions_xy: Optional patch positions as (x, y) coordinates. Required
        when output_length != input seq_len.
      output_length_overrides: Optional override for output lengths. If
        provided, uses these instead of self.output_length.

    Returns:
      Tuple of (embeddings, mask) pairs, one for each output length.
    """
    lens = output_length_overrides or self.output_length
    if isinstance(lens, int):
      lens = (lens,)
    finfo = jnp.finfo(x.dtype)
    x = jax.lax.reduce_precision(x, finfo.nexp, finfo.nmant)
    return tuple(
        self._single_call(x, positions_xy=positions_xy, length=length)
        for length in lens
        if length <= x.shape[1]
    )


class Standardize(nn.Module):
  """Applies feature-wise standardization: x = (x - bias) * scale."""

  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    dim = x.shape[-1]
    scale = self.param('scale', nn.initializers.ones, (dim,), self.param_dtype)
    bias = self.param('bias', nn.initializers.zeros, (dim,), self.param_dtype)
    return (x - bias.astype(x.dtype)) * scale.astype(x.dtype)
