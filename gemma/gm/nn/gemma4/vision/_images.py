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

"""Utility functions for Gemma4 vision models."""

import einops
import jax
import jax.numpy as jnp
from kauldron.ktyping import Float, Int, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@typechecked
def preprocess_image(
    image: UInt8['H W C'],
    *,
    patch_size: int = 16,
    max_patches: int = 2520,
    pooling_kernel_size: int = 3,
) -> Float['H2 W2 C']:
  """Preprocesses image."""
  # Step 1: Calculate the target dimensions preserving aspect ratio
  height, width, _ = image.shape
  from gemma.gm.nn.gemma4.vision import _preprocessing  # pylint: disable=g-import-not-at-top

  target_height, target_width = _preprocessing.get_target_dimensions(
      height,
      width,
      patch_size=patch_size,
      max_patches=max_patches,
      pooling_kernel_size=pooling_kernel_size,
  )

  # Step 2: Normalize image pixels to [0, 1] range
  image_f = image.astype(jnp.float32) / 255.0

  # Step 3: Resize the image to the target dimensions using bicubic
  # interpolation
  if target_height != height or target_width != width:
    image_f = jax.image.resize(
        image_f,
        (target_height, target_width, 3),
        method='bicubic',
    )

  return image_f


@typechecked
def factorized_posemb(
    posemb: Float['S 2 D'], positions_xy: Int['B L 2']
) -> Float['B L D']:
  """Compute factorized position embedding from (x, y) coordinates."""
  # One-hot positions_xy to the range [0, posemb.shape[0])
  one_hot = jax.nn.one_hot(positions_xy, posemb.shape[0], dtype=posemb.dtype)
  # Create a mask of all invalid positions by finding the zero values
  # Note: jax.nn.one_hot returns zeros when the value is < 0 (usually padding)
  # or >= posemb.shape[0] (invalid)
  nan = jnp.logical_not(one_hot.any(axis=-1, keepdims=True))
  # Padding positions are valid, so remove them from the mask
  nan = jnp.logical_and(nan, positions_xy[..., None] != -1)
  # Compute the final one-hot encoding by replacing any invalid positions w/ NaN
  pos_oh = jnp.where(nan, jnp.nan, one_hot)
  # Compute the XY position embedding for each valid position
  # Note: jnp.einsum() will preserve the NaNs
  pe_seq = jnp.einsum('blis,sid->ibld', pos_oh, posemb).astype(posemb.dtype)
  # Sum over the XY-index axis to get the final embeddings
  # Note: jnp.sum() will return NaNs if either the X or Y coordinate is invalid
  return jnp.sum(pe_seq, axis=0)


@typechecked
def patchify(
    images: Float['*B H W C'], patch_size: int
) -> tuple[Float['*B L D'], Int['*B L 2']]:
  """Patchify images and return patches and (x, y) coordinates."""
  patches = einops.rearrange(
      images,
      '... (h p) (w q) c -> ... (h w) (p q c)',
      p=patch_size,
      q=patch_size,
  )
  *b, h, w, _ = images.shape
  # Positions come out of preproc as (x, y) coords. We match that here
  # x indexes into the width of the image.
  xy = jnp.meshgrid(jnp.arange(w // patch_size), jnp.arange(h // patch_size))
  positions_xy = jnp.stack(xy, axis=-1)
  # positions_xy[:, 0] is the x coordinate, indexing into image width.
  # positions_xy[:, 1] is the y coordinate, indexing into image height.
  positions_xy = einops.rearrange(positions_xy, 'y x c -> (y x) c')
  return patches, jnp.broadcast_to(positions_xy, (*b, *positions_xy.shape))
