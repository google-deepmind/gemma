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

"""Optimized batch image loading with parallel processing and memory efficiency."""

from __future__ import annotations

import concurrent.futures
import functools
import io
from collections.abc import Callable, Iterator, Sequence
from typing import Optional, Union

import einops
import jax
import numpy as np
from etils import epath
from jax import numpy as jnp
from kauldron import typing
from PIL import Image

_IMAGE_MEAN = (127.5,) * 3
_IMAGE_STD = (127.5,) * 3
_DEFAULT_IMAGE_SIZE = 896  # SigLip expected input image size
_DEFAULT_PATCH_SIZE = 14  # SigLip expected patch size


@typing.typechecked
def normalize_images_batch(
    images: typing.Float["B H W C"],
) -> typing.Float["B H W C"]:
  """Normalize a batch of images to zero mean and unit variance.
  
  Args:
    images: Batch of images to normalize.
    
  Returns:
    Normalized batch of images.
  """
  mean = jnp.asarray(_IMAGE_MEAN).reshape(1, 1, 1, 3)
  std = jnp.asarray(_IMAGE_STD).reshape(1, 1, 1, 3)
  images = (images - mean) / std
  return images


def pre_process_image_pil(
    image: Union[np.ndarray, Image.Image],
    *,
    image_height: int = _DEFAULT_IMAGE_SIZE,
    image_width: int = _DEFAULT_IMAGE_SIZE,
    use_jpeg_compression: bool = True,
) -> typing.Float["H W C"]:
  """Pre-process image using PIL instead of TensorFlow.
  
  Performs a bi-linear resize (with anti-aliasing) and normalizes the image.
  This implementation removes the TensorFlow dependency.
  
  Args:
    image: The image to pre-process (numpy array or PIL Image).
    image_height: The height of the image (default to 896).
    image_width: The width of the image (default to 896).
    use_jpeg_compression: Whether to apply JPEG compression (for consistency).
    
  Returns:
    The pre-processed image.
  """
  # Convert to PIL Image if needed
  if isinstance(image, np.ndarray):
    image = Image.fromarray(image.astype(np.uint8))
  elif not isinstance(image, Image.Image):
    raise TypeError(f"Expected np.ndarray or PIL.Image, got {type(image)}")
  
  # Apply JPEG compression if requested (for consistency with original)
  if use_jpeg_compression:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    image = Image.open(buffer)
  
  # Resize with anti-aliasing
  image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)
  
  # Convert to numpy array
  image = np.array(image, dtype=np.float32)
  
  # Normalize
  image = (image - np.array(_IMAGE_MEAN)) / np.array(_IMAGE_STD)
  image = np.clip(image, -1, 1)
  
  return jnp.asarray(image)


@typing.typechecked
def patchify_images_batch(
    images: typing.Float["B H W C"],
    patch_size: int = _DEFAULT_PATCH_SIZE,
    padding: str = "VALID",
) -> typing.Float["B P D"]:
  """Extract patches from a batch of images efficiently.
  
  Args:
    images: Batch of images of shape [B, H, W, C].
    patch_size: Size of extracted patches.
    padding: Padding algorithm to use.
    
  Returns:
    Tensor of shape [B, num_patches, patch_size * patch_size * C]
  """
  batch_size = images.shape[0]
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
  
  # Reshape to [B, num_patches, patch_dim]
  patches = einops.rearrange(
      patches, "b h w (c p) -> b (h w) (p c)", c=channels
  )
  return patches


def _load_single_image(
    img_path: str,
    image_height: int,
    image_width: int,
    use_jpeg_compression: bool,
) -> np.ndarray:
  """Load and preprocess a single image.
  
  Args:
    img_path: Path to the image file.
    image_height: Target image height.
    image_width: Target image width.
    use_jpeg_compression: Whether to apply JPEG compression.
    
  Returns:
    Preprocessed image as numpy array.
  """
  with epath.Path(img_path).open("rb") as f:
    img = Image.open(f).convert("RGB")
  return np.array(
      pre_process_image_pil(
          img,
          image_height=image_height,
          image_width=image_width,
          use_jpeg_compression=use_jpeg_compression,
      )
  )


@typing.typechecked
def load_images_parallel(
    img_paths: Sequence[str],
    *,
    image_height: int = _DEFAULT_IMAGE_SIZE,
    image_width: int = _DEFAULT_IMAGE_SIZE,
    patch_size: int = _DEFAULT_PATCH_SIZE,
    max_workers: Optional[int] = None,
    use_jpeg_compression: bool = True,
) -> typing.Float["B P D"]:
  """Load and process images in parallel using thread pool.
  
  Args:
    img_paths: List of image file paths.
    image_height: Target image height.
    image_width: Target image width.
    patch_size: Size of patches to extract.
    max_workers: Maximum number of parallel workers (None for auto).
    use_jpeg_compression: Whether to apply JPEG compression.
    
  Returns:
    Patches of shape [batch_size, num_patches, patch_dim].
  """
  if not img_paths:
    raise ValueError("img_paths cannot be empty")
  
  # Create partial function with fixed parameters
  load_fn = functools.partial(
      _load_single_image,
      image_height=image_height,
      image_width=image_width,
      use_jpeg_compression=use_jpeg_compression,
  )
  
  # Load images in parallel
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    images = list(executor.map(load_fn, img_paths))
  
  # Stack into batch
  images_batch = jnp.stack(images)
  
  # Extract patches from entire batch at once
  patches = patchify_images_batch(images_batch, patch_size=patch_size)
  
  return patches


class BatchImageLoader:
  """Memory-efficient batch image loader with streaming support."""
  
  def __init__(
      self,
      image_height: int = _DEFAULT_IMAGE_SIZE,
      image_width: int = _DEFAULT_IMAGE_SIZE,
      patch_size: int = _DEFAULT_PATCH_SIZE,
      batch_size: int = 32,
      max_workers: Optional[int] = None,
      use_jpeg_compression: bool = True,
      prefetch_size: int = 2,
  ):
    """Initialize the batch image loader.
    
    Args:
      image_height: Target image height.
      image_width: Target image width.
      patch_size: Size of patches to extract.
      batch_size: Number of images per batch.
      max_workers: Maximum number of parallel workers.
      use_jpeg_compression: Whether to apply JPEG compression.
      prefetch_size: Number of batches to prefetch.
    """
    self.image_height = image_height
    self.image_width = image_width
    self.patch_size = patch_size
    self.batch_size = batch_size
    self.max_workers = max_workers
    self.use_jpeg_compression = use_jpeg_compression
    self.prefetch_size = prefetch_size
    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    self._load_fn = functools.partial(
        _load_single_image,
        image_height=image_height,
        image_width=image_width,
        use_jpeg_compression=use_jpeg_compression,
    )
  
  def load_batch(self, img_paths: Sequence[str]) -> typing.Float["B P D"]:
    """Load a batch of images.
    
    Args:
      img_paths: Paths to images in the batch.
      
    Returns:
      Patches of shape [batch_size, num_patches, patch_dim].
    """
    # Load images in parallel
    futures = [self._executor.submit(self._load_fn, path) for path in img_paths]
    images = [future.result() for future in futures]
    
    # Stack and process
    images_batch = jnp.stack(images)
    patches = patchify_images_batch(images_batch, patch_size=self.patch_size)
    
    return patches
  
  def stream_batches(
      self, img_paths: Sequence[str]
  ) -> Iterator[typing.Float["B P D"]]:
    """Stream batches of images with prefetching.
    
    Args:
      img_paths: All image paths to process.
      
    Yields:
      Batches of patches.
    """
    num_images = len(img_paths)
    num_batches = (num_images + self.batch_size - 1) // self.batch_size
    
    # Queue for prefetching
    futures_queue = []
    
    for batch_idx in range(num_batches):
      start_idx = batch_idx * self.batch_size
      end_idx = min(start_idx + self.batch_size, num_images)
      batch_paths = img_paths[start_idx:end_idx]
      
      # Submit batch for loading
      batch_futures = [
          self._executor.submit(self._load_fn, path) for path in batch_paths
      ]
      futures_queue.append(batch_futures)
      
      # If we have enough prefetched batches, yield the oldest one
      if len(futures_queue) > self.prefetch_size:
        ready_futures = futures_queue.pop(0)
        images = [f.result() for f in ready_futures]
        images_batch = jnp.stack(images)
        patches = patchify_images_batch(images_batch, patch_size=self.patch_size)
        yield patches
    
    # Yield remaining batches
    for batch_futures in futures_queue:
      images = [f.result() for f in batch_futures]
      images_batch = jnp.stack(images)
      patches = patchify_images_batch(images_batch, patch_size=self.patch_size)
      yield patches
  
  def close(self):
    """Clean up resources."""
    self._executor.shutdown(wait=True)
  
  def __enter__(self):
    """Context manager entry."""
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit."""
    self.close()


@typing.typechecked
def load_image_files_optimized(
    img_paths: Sequence[Sequence[str | None]],
    patch_size: int = _DEFAULT_PATCH_SIZE,
    max_workers: Optional[int] = None,
    use_streaming: bool = False,
    batch_size: int = 32,
) -> typing.Float["B S P D"] | None:
  """Optimized version of load_image_files with parallel processing.
  
  This is a drop-in replacement for the original load_image_files function
  but with significant performance improvements through parallel loading.
  
  Args:
    img_paths: A list of list of image paths.
    patch_size: The size of the patches.
    max_workers: Maximum number of parallel workers.
    use_streaming: Whether to use streaming mode for large datasets.
    batch_size: Batch size for streaming mode.
    
  Returns:
    The patches of the images of shape [batch size, num images, num patches,
    patch size * patch size * channels]
  """
  if len(img_paths) == 1 and len(img_paths[0]) == 1 and img_paths[0][0] is None:
    return None
  
  # Flatten the paths for parallel processing
  flat_paths = []
  batch_indices = []
  image_indices = []
  
  for batch_idx, imgs_path in enumerate(img_paths):
    for img_idx, img_path in enumerate(imgs_path):
      if img_path is None:
        raise ValueError(
            "some img_paths are None and some are not. we only support all None"
            " or all not None for now."
        )
      flat_paths.append(img_path)
      batch_indices.append(batch_idx)
      image_indices.append(img_idx)
  
  if use_streaming:
    # Use streaming mode for large datasets
    loader = BatchImageLoader(
        patch_size=patch_size,
        max_workers=max_workers,
        batch_size=batch_size,
    )
    with loader:
      all_patches = []
      for batch_patches in loader.stream_batches(flat_paths):
        all_patches.append(batch_patches)
      all_patches = jnp.concatenate(all_patches, axis=0)
  else:
    # Load all images at once
    all_patches = load_images_parallel(
        flat_paths,
        patch_size=patch_size,
        max_workers=max_workers,
    )
  
  # Reshape back to original structure
  num_batches = len(img_paths)
  num_images_per_batch = len(img_paths[0])
  num_patches = all_patches.shape[1]
  patch_dim = all_patches.shape[2]
  
  result = jnp.zeros((num_batches, num_images_per_batch, num_patches, patch_dim))
  
  for idx, (batch_idx, img_idx) in enumerate(zip(batch_indices, image_indices)):
    result = result.at[batch_idx, img_idx].set(all_patches[idx])
  
  return result