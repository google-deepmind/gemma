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

"""Image preprocessing utilities for Gemma4 vision models.

Handles variable-aspect-ratio images: each image is resized independently
preserving its aspect ratio, with dimensions constrained to multiples of
pooling_kernel_size * patch_size and total patches ≤ max_patches.
"""

import math

import jax.numpy as jnp
import numpy as np
from PIL import Image


def get_target_dimensions(
    height: int,
    width: int,
    patch_size: int = 16,
    max_patches: int = 10080,
    pooling_kernel_size: int = 3,
) -> tuple[int, int]:
  """Calculates target height and width preserving aspect ratio."""
  total_px = height * width
  target_px = max_patches * (patch_size**2)
  if total_px == 0:
    return pooling_kernel_size * patch_size, pooling_kernel_size * patch_size

  factor = math.sqrt(target_px / total_px)
  ideal_height = factor * height
  ideal_width = factor * width
  side_mult = pooling_kernel_size * patch_size

  target_height = int(math.floor(ideal_height / side_mult)) * side_mult
  target_width = int(math.floor(ideal_width / side_mult)) * side_mult

  if target_height == 0 and target_width == 0:
    target_height = side_mult
    target_width = side_mult
  elif target_height == 0:
    target_height = side_mult
    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    target_width = min(
        max(1, int(math.floor(width / height))) * side_mult,
        max_side_length,
    )
  elif target_width == 0:
    target_width = side_mult
    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    target_height = min(
        max(1, int(math.floor(height / width))) * side_mult,
        max_side_length,
    )

  return int(target_height), int(target_width)


def aspect_ratio_preserving_resize(
    image: np.ndarray,
    patch_size: int = 16,
    max_patches: int = 10080,
    pooling_kernel_size: int = 3,
) -> np.ndarray:
  """Resizes an image while preserving its aspect ratio.

  The target dimensions are calculated to fit within a constraint of
  `max_patches`, ensuring dimensions are multiples of
  `pooling_kernel_size * patch_size`.

  Args:
    image: The input image as a numpy array.
    patch_size: The size of each patch in pixels.
    max_patches: The maximum number of patches allowed.
    pooling_kernel_size: The size of the pooling kernel.

  Returns:
    The resized image as a numpy array.
  """
  height, width = image.shape[:2]
  target_height, target_width = get_target_dimensions(
      height,
      width,
      patch_size=patch_size,
      max_patches=max_patches,
      pooling_kernel_size=pooling_kernel_size,
  )

  if target_height == height and target_width == width:
    return image

  pil_image = Image.fromarray(image)
  pil_image = pil_image.resize(
      (target_width, target_height), resample=Image.BICUBIC
  )
  return np.array(pil_image)


def _to_rgb_uint8(image: np.ndarray | Image.Image) -> np.ndarray:
  if isinstance(image, Image.Image):
    image = image.convert("RGB")
    return np.array(image)

  if image.ndim == 2:
    image = np.stack([image] * 3, axis=-1)
  elif image.shape[-1] == 4:
    image = image[..., :3]
  elif image.shape[-1] == 1:
    image = np.concatenate([image] * 3, axis=-1)
  return image


def preprocess_image(
    image: np.ndarray | Image.Image,
    patch_size: int = 16,
    max_soft_tokens: int = 1120,
    pooling_kernel_size: int = 3,
) -> jnp.ndarray:
  """Preprocesses a single image for the VisionEncoder.

  This involves converting to RGB, resizing while preserving aspect ratio
  to fit within `max_soft_tokens`, and normalizing pixel values to [0, 1].

  Args:
    image: The input image as a numpy array or PIL Image.
    patch_size: The size of each patch in pixels.
    max_soft_tokens: The maximum number of soft tokens allowed after pooling.
    pooling_kernel_size: The size of the pooling kernel.

  Returns:
    The preprocessed image as a jax numpy array with float32 dtype,
    with pixel values in the range [0, 1].
  """
  image = _to_rgb_uint8(image)
  max_patches = max_soft_tokens * pooling_kernel_size**2

  image = aspect_ratio_preserving_resize(
      image,
      patch_size=patch_size,
      max_patches=max_patches,
      pooling_kernel_size=pooling_kernel_size,
  )

  image = image.astype(np.float32) / 255.0
  return jnp.array(image)


def num_soft_tokens_for_image(
    image: jnp.ndarray,
    patch_size: int = 16,
    pooling_kernel_size: int = 3,
) -> int:
  h, w = image.shape[:2]
  num_patches = (h // patch_size) * (w // patch_size)
  return int(num_patches // (pooling_kernel_size**2))


def predict_soft_token_count(
    height: int,
    width: int,
    patch_size: int = 16,
    max_soft_tokens: int = 1120,
    pooling_kernel_size: int = 3,
) -> int:
  """Predicts the number of soft tokens for an image of given dimensions.

  This function calculates the number of soft tokens that would be produced
  by an image with the given height and width after being resized by
  `get_target_dimensions` and then patchified and pooled.

  Args:
    height: The original height of the image.
    width: The original width of the image.
    patch_size: The size of each patch in pixels.
    max_soft_tokens: The maximum number of soft tokens allowed.
    pooling_kernel_size: The size of the pooling kernel.

  Returns:
    The predicted number of soft tokens.
  """
  max_patches = max_soft_tokens * pooling_kernel_size**2
  target_height, target_width = get_target_dimensions(
      height,
      width,
      patch_size=patch_size,
      max_patches=max_patches,
      pooling_kernel_size=pooling_kernel_size,
  )

  num_patches = (target_height // patch_size) * (target_width // patch_size)
  return int(num_patches // (pooling_kernel_size**2))


def preprocess_images(
    images: list[np.ndarray | Image.Image],
    patch_size: int = 16,
    max_soft_tokens: int = 1120,
    pooling_kernel_size: int = 3,
) -> tuple[list[jnp.ndarray], list[int]]:
  """Preprocess a batch of variable-size images.

  Args:
    images: List of raw images as numpy arrays or PIL Images.
    patch_size: Patch size in pixels.
    max_soft_tokens: Maximum soft tokens per image.
    pooling_kernel_size: Pooling kernel size.

  Returns:
    processed_images: List of float32 arrays in [0,1], each [H_i, W_i, 3].
    soft_token_counts: Number of soft tokens each image will produce after
      pooling (different for each image since aspect ratios differ).
  """
  processed = []
  soft_token_counts = []
  for img in images:
    p = preprocess_image(
        img,
        patch_size=patch_size,
        max_soft_tokens=max_soft_tokens,
        pooling_kernel_size=pooling_kernel_size,
    )
    processed.append(p)
    soft_token_counts.append(
        num_soft_tokens_for_image(p, patch_size, pooling_kernel_size)
    )
  return processed, soft_token_counts


def preprocess_and_patchify(
    images: list[np.ndarray | Image.Image],
    patch_size: int = 16,
    max_soft_tokens: int = 1120,
    pooling_kernel_size: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, list[int]]:
  """Preprocess variable-size images and patchify+pad for the encoder.

  This is the single entry point for preparing images for the VisionEncoder.
  Each image is independently resized (preserving aspect ratio), rescaled to
  [0, 1], patchified, and padded to a common max_patches length.

  Args:
    images: List of raw images as numpy arrays or PIL Images.
    patch_size: Patch size in pixels.
    max_soft_tokens: Maximum soft tokens per image.
    pooling_kernel_size: Pooling kernel size.

  Returns:
    patches: Padded patches [N_images, max_patches, patch_dim].
    positions_xy: Padded positions [N_images, max_patches, 2], -1 for padding.
    soft_token_counts: Number of real soft tokens per image.
  """
  from gemma.gm.nn.gemma4.vision import _encoder  # pylint: disable=g-import-not-at-top

  processed, soft_token_counts = preprocess_images(
      images,
      patch_size=patch_size,
      max_soft_tokens=max_soft_tokens,
      pooling_kernel_size=pooling_kernel_size,
  )

  patches, positions_xy, _ = _encoder.patchify_and_pad(
      processed,
      patch_size=patch_size,
      max_soft_tokens=max_soft_tokens,
      pooling_kernel_size=pooling_kernel_size,
  )

  return patches, positions_xy, soft_token_counts
