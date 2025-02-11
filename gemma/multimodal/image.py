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

"""Gemma implementation of the vision encoders."""

from __future__ import annotations
from collections.abc import Sequence
import einops
import jax
from jax import numpy as jnp
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_MEAN = (127.5,) * 3
IMAGE_STD = (127.5,) * 3


def normalize_images(images: jax.Array) -> jax.Array:
  """Normalize the image to zero mean and unit variance.

  In order to change the image mean and std, we need to change the IMAGE_MEAN
  and IMAGE_STD global constants in this file.

  Args:
    images: The images to normalize.

  Returns:
    The normalized images.
  """
  images -= jnp.array(IMAGE_MEAN).reshape(1, 1, 3)
  images /= jnp.array(IMAGE_STD).reshape(1, 1, 3)
  return images


def pre_process_image(
    image: np.ndarray, *, image_height: int = 896, image_width: int = 896
) -> jax.Array:
  """Pre-process image.

  Performs a bi-linear resize (with anti-aliasing) and normalizes the image.

  Args:
    image: The image to pre-process.
    image_height: The height of the image.
    image_width: The width of the image.

  Returns:
    The pre-processed image.
  """
  image = jnp.array(tf.image.decode_jpeg(tf.io.encode_jpeg(image), channels=3))
  image = jax.image.resize(
      image,
      shape=(image_height, image_width, 3),
      method="bilinear",
      antialias=True,
  )
  image = normalize_images(image)
  image = jnp.clip(image, -1, 1)
  return image


def jax_patchify_images(
    images: jax.Array, patch_size: int = 14, padding: str = "VALID"
) -> jax.Array:
  """Extract patches from images.

  This function is a wrapper for jax.lax.conv
  to conform to the same interface as tf.image.extract_patches.
  The function extracts patches of shape sizes from the input images in the same
  manner as a convolution with kernel of shape sizes, stride equal to strides,
  and the given padding scheme.
  The patches are stacked in the channel dimension.

  Args:
    images: input batch of images of shape [B, H, W, C].
    patch_size: size of extracted patches. Must be [1, size_rows, size_cols, 1].
    padding: padding algorithm to use.

  Returns:
    Tensor of shape [B, patch_rows, patch_cols, size_rows * size_cols * C]
  """

  sizes = [1, patch_size, patch_size, 1]

  channels = images.shape[-1]
  kernel_size = patch_size * patch_size * channels
  kernel = jnp.reshape(
      jnp.eye(kernel_size, dtype=images.dtype),
      [patch_size, patch_size, channels, -1],
  )

  if len(sizes) != 4 or sizes[0] != 1 or sizes[3] != 1:
    raise ValueError(
        f"Shape of sizes must be [1, size_rows, size_cols, 1], got {sizes}."
    )
  if images.ndim != 4:
    raise ValueError(
        f"Rank of images must be 4 (got tensor of shape {jnp.shape(images)})"
    )
  patches = jax.lax.conv(
      images.transpose(0, 3, 1, 2),
      kernel.transpose(3, 2, 0, 1),
      [patch_size, patch_size],
      padding=padding,
      precision=jax.lax.Precision.HIGH,
  ).transpose(0, 2, 3, 1)
  return patches


def patchify_images(
    images: jax.Array, patch_size: int = 14, padding: str = "VALID"
) -> jax.Array:
  """Extract patches from images.

  This function is a wrapper for jax.lax.conv_general_dilated_patches
  to conforms to the same interface as tf.image.extract_patches.
  The function extracts patches of shape sizes from the input images in the same
  manner as a convolution with kernel of shape sizes, stride equal to strides,
  and the given padding scheme.
  The patches are stacked in the channel dimension.

  Args:
    images: input batch of images of shape.
    patch_size: size of extracted patches.
    padding: padding algorithm to use.

  Returns:
    Tensor of shape.
  """
  num_batches, num_media, num_frames, _, _, channels = images.shape
  images = einops.rearrange(
      images, "b media frames h w c -> (b media frames) h w c"
  )

  if images.ndim != 4:
    raise ValueError(
        f"Rank of images must be 4 (got tensor of shape {jnp.shape(images)})"
    )
  patches = jax_patchify_images(images, patch_size, padding)
  patches = patches.reshape(
      num_batches,
      num_media * num_frames,
      -1,
      channels * patch_size * patch_size,
  )
  return patches


def load_image_files(
    img_paths: Sequence[Sequence[str | None]], patch_size: int = 14
) -> jax.Array | None:
  """Loads image files.

  Args:
    img_paths: A list of list of image paths. Each element in the list is a list
      of image paths. We use a list of lists since we want to support batching
      (first list) and multiple images per sample (second list).
    patch_size: The size of the patches.

  Returns:
    The patches of the images.
  """
  if len(img_paths) == 1 and len(img_paths[0]) == 1 and img_paths[0][0] is None:
    return None
  patches = []
  for imgs_path in img_paths:
    tmp = []
    for img_path in imgs_path:
      if img_path is None:
        raise ValueError(
            "some img_paths are None and some are not. we only support all None"
            " or all not None for now."
        )
      img = pre_process_image(np.array(Image.open(img_path).convert("RGB")))
      height, width, channels = img.shape
      img = img.reshape((1, 1, 1, height, width, channels))
      tmp.append(patchify_images(img).reshape(-1, channels * (patch_size**2)))
    patches.append(jnp.array(tmp))
  patches = jnp.array(patches)
  return patches
