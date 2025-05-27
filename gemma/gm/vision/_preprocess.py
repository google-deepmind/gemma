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

"""Preprocess images."""

import einops
import jax
from jax import numpy as jnp
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


_IMAGE_MEAN = (127.5,) * 3
_IMAGE_STD = (127.5,) * 3


def pre_process_image(
    image: Float["H W C"],
    *,
    image_shape: tuple[int, int, int],
) -> Float["H W C"]:
  """Pre-process image.

  Performs a bi-linear resize (with anti-aliasing) and normalizes the image.

  Args:
    image: The image to pre-process.
    image_shape: The target shape (h, w, c) of the image (default to (896, 896,
      3)).

  Returns:
    The pre-processed image.
  """
  # TODO(epot): All inputs are expected to have been jpeg encoded with
  # TensorFlow.
  # tf.image.decode_jpeg(tf.io.encode_jpeg(image), channels=3)

  image = jax.image.resize(
      image,
      shape=image_shape,
      method="bilinear",
      antialias=True,
  )
  image = _normalize_images(image)
  image = jnp.clip(image, -1, 1)
  return image


@typechecked
def patchify_images(
    images: Float["B H W C"],
    *,
    patch_size: tuple[int, int],
    padding: str = "VALID",
) -> Float["B P D"]:
  """Extract patches from images.

  This function is a wrapper for jax.lax.conv_general_dilated_patches
  to conform to the same interface as tf.image.extract_patches.
  The function extracts patches of shape sizes from the input images in the same
  manner as a convolution with kernel of shape sizes, stride equal to strides,
  and the given padding scheme.
  The patches are stacked in the channel dimension.

  Args:
    images: input batch of images of shape [B, H, W, C].
    patch_size: size of extracted patches.
    padding: padding algorithm to use.

  Returns:
    Tensor of shape [batch, num patches, patch_size * patch_size * C]
  """
  channels = images.shape[-1]
  patches = jax.lax.conv_general_dilated_patches(
      lhs=images,
      filter_shape=patch_size,
      window_strides=patch_size,
      padding=padding,
      rhs_dilation=[1, 1],
      dimension_numbers=("NHWC", "OIHW", "NHWC"),
      precision=jax.lax.Precision.HIGH,
  )
  patches = einops.rearrange(
      patches, "b ph pw (c p) -> b (ph pw) (p c)", c=channels
  )
  return patches


@typechecked
def _normalize_images(
    images: Float["H W C"],
) -> Float["H W C"]:
  """Normalize the image to zero mean and unit variance.

  In order to change the image mean and std, we need to change the _IMAGE_MEAN
  and _IMAGE_STD global constants in this file.

  Args:
    images: The images to normalize.

  Returns:
    The normalized images.
  """
  images -= jnp.asarray(_IMAGE_MEAN)
  images /= jnp.asarray(_IMAGE_STD)
  return images
