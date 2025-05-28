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

"""Gemma implementation of the vision encoders."""

from __future__ import annotations
from collections.abc import Sequence
import einops
from etils import epath
import jax
from jax import numpy as jnp
from kauldron import typing
import numpy as np
from PIL import Image
import tensorflow as tf

_IMAGE_MEAN = (127.5,) * 3
_IMAGE_STD = (127.5,) * 3
_DEFAULT_IMAGE_SIZE = 896  # SigLip expected input image size
_DEFAULT_PATCH_SIZE = 14  # SigLip expected patch size


@typing.typechecked
def normalize_images(
    images: typing.Float["H W C"],
) -> typing.Float["H W C"]:
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


def pre_process_image(
    image: typing.Float["H W C"],
    *,
    image_height: int = _DEFAULT_IMAGE_SIZE,
    image_width: int = _DEFAULT_IMAGE_SIZE,
) -> typing.Float["H W C"]:
  """Pre-process image.

  Performs a bi-linear resize (with anti-aliasing) and normalizes the image.

  Args:
    image: The image to pre-process.
    image_height: The height of the image (default to 896).
    image_width: The width of the image (default to 896).

  Returns:
    The pre-processed image.
  """
  # all inputs are expected to have been jpeg compressed.
  # TODO(eyvinec): we should remove tf dependency.
  image = jnp.asarray(
      tf.image.decode_jpeg(tf.io.encode_jpeg(image), channels=3)
  )
  image = jax.image.resize(
      image,
      shape=(image_height, image_width, 3),
      method="bilinear",
      antialias=True,
  )
  image = normalize_images(image)
  image = jnp.clip(image, -1, 1)
  return image


@typing.typechecked
def patchify_images(
    images: typing.Float["B H W C"],
    patch_size: int = _DEFAULT_PATCH_SIZE,
    padding: str = "VALID",
) -> typing.Float["P D"]:
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
    Tensor of shape [num patches, patch_size * patch_size * C]
  """
  channels = images.shape[-1]
  patches = jax.lax.conv_general_dilated_patches(
      lhs=images,
      filter_shape=[patch_size, patch_size],
      window_strides=[patch_size, patch_size],
      padding=padding,
      rhs_dilation=[1, 1],
      dimension_numbers=("NHWC", "OIHW", "NHWC"),
      precision=jax.lax.Precision.HIGH,
  )
  patches = einops.rearrange(patches, "... (c p) -> (...) (p c)", c=channels)
  return patches


@typing.typechecked
def load_image_files(
    img_paths: Sequence[Sequence[str | None]],
    patch_size: int = _DEFAULT_PATCH_SIZE,
) -> typing.Float["B S P D"] | None:
  """Loads image files.

  Args:
    img_paths: A list of list of image paths. Each element in the list is a list
      of image paths. We use a list of lists since we want to support batching
      (first list) and multiple images per sample (second list).
    patch_size: The size of the patches.

  Returns:
    The patches of the images of shape [batch size, num images, num patches,
    patch size * patch size * channels]
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
      with epath.Path(img_path).open("rb") as f:
        img = pre_process_image(np.array(Image.open(f).convert("RGB")))
      tmp.append(patchify_images(img[None, ...], patch_size))
    patches.append(jnp.asarray(tmp))
  patches = jnp.asarray(patches)
  return patches
