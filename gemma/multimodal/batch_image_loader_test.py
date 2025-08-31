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

"""Tests for batch_image_loader module."""

import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image

from gemma.multimodal import batch_image_loader


class BatchImageLoaderTest(unittest.TestCase):
  """Test cases for batch image loading optimization."""
  
  def setUp(self):
    """Set up test fixtures."""
    self.temp_dir = tempfile.mkdtemp()
    self.temp_path = Path(self.temp_dir)
    
    # Create test images
    self.test_images = []
    self.image_paths = []
    
    for i in range(6):
      # Create a simple test image with different colors
      img_array = np.zeros((224, 224, 3), dtype=np.uint8)
      img_array[:, :, i % 3] = 255  # Different color for each image
      img = Image.fromarray(img_array)
      
      img_path = self.temp_path / f"test_image_{i}.jpg"
      img.save(img_path, "JPEG")
      
      self.test_images.append(img_array)
      self.image_paths.append(str(img_path))
  
  def tearDown(self):
    """Clean up test fixtures."""
    import shutil
    shutil.rmtree(self.temp_dir)
  
  def test_normalize_images_batch(self):
    """Test batch normalization."""
    # Create a batch of test images
    images = jnp.ones((2, 224, 224, 3)) * 127.5
    normalized = batch_image_loader.normalize_images_batch(images)
    
    # Check shape preservation
    self.assertEqual(normalized.shape, images.shape)
    
    # Check normalization (should be close to 0 for 127.5 input)
    np.testing.assert_allclose(normalized, jnp.zeros_like(normalized), atol=0.01)
  
  def test_pre_process_image_pil(self):
    """Test PIL-based image preprocessing."""
    # Create a test image
    img = Image.new("RGB", (100, 100), color="red")
    
    # Process with default size
    processed = batch_image_loader.pre_process_image_pil(img)
    
    # Check output shape
    self.assertEqual(processed.shape, (896, 896, 3))
    
    # Check value range
    self.assertTrue(jnp.all(processed >= -1))
    self.assertTrue(jnp.all(processed <= 1))
    
    # Test with custom size
    processed_custom = batch_image_loader.pre_process_image_pil(
        img, image_height=224, image_width=224
    )
    self.assertEqual(processed_custom.shape, (224, 224, 3))
  
  def test_patchify_images_batch(self):
    """Test batch patchification."""
    # Create batch of images
    batch_size = 4
    image_size = 224
    patch_size = 14
    images = jnp.ones((batch_size, image_size, image_size, 3))
    
    # Extract patches
    patches = batch_image_loader.patchify_images_batch(
        images, patch_size=patch_size
    )
    
    # Check output shape
    num_patches = (image_size // patch_size) ** 2
    patch_dim = patch_size * patch_size * 3
    self.assertEqual(patches.shape, (batch_size, num_patches, patch_dim))
  
  def test_load_images_parallel(self):
    """Test parallel image loading."""
    # Load first 4 images in parallel
    patches = batch_image_loader.load_images_parallel(
        self.image_paths[:4],
        image_height=224,
        image_width=224,
        patch_size=14,
        max_workers=2,
    )
    
    # Check output shape
    batch_size = 4
    num_patches = (224 // 14) ** 2
    patch_dim = 14 * 14 * 3
    self.assertEqual(patches.shape, (batch_size, num_patches, patch_dim))
  
  def test_batch_image_loader_class(self):
    """Test BatchImageLoader class."""
    loader = batch_image_loader.BatchImageLoader(
        image_height=224,
        image_width=224,
        patch_size=14,
        batch_size=2,
        max_workers=2,
    )
    
    try:
      # Load a batch
      patches = loader.load_batch(self.image_paths[:2])
      
      # Check output shape
      num_patches = (224 // 14) ** 2
      patch_dim = 14 * 14 * 3
      self.assertEqual(patches.shape, (2, num_patches, patch_dim))
    finally:
      loader.close()
  
  def test_streaming_batches(self):
    """Test streaming batch loading."""
    batch_size = 2
    loader = batch_image_loader.BatchImageLoader(
        image_height=224,
        image_width=224,
        patch_size=14,
        batch_size=batch_size,
        max_workers=2,
        prefetch_size=1,
    )
    
    with loader:
      batches = list(loader.stream_batches(self.image_paths))
      
      # Check number of batches
      expected_batches = (len(self.image_paths) + batch_size - 1) // batch_size
      self.assertEqual(len(batches), expected_batches)
      
      # Check shape of each batch
      num_patches = (224 // 14) ** 2
      patch_dim = 14 * 14 * 3
      
      for i, batch in enumerate(batches):
        if i < len(batches) - 1:
          # Full batch
          self.assertEqual(batch.shape, (batch_size, num_patches, patch_dim))
        else:
          # Last batch might be smaller
          remaining = len(self.image_paths) % batch_size
          if remaining == 0:
            remaining = batch_size
          self.assertEqual(batch.shape, (remaining, num_patches, patch_dim))
  
  def test_load_image_files_optimized(self):
    """Test optimized load_image_files function."""
    # Create nested structure like original function expects
    img_paths = [
        [self.image_paths[0], self.image_paths[1]],
        [self.image_paths[2], self.image_paths[3]],
        [self.image_paths[4], self.image_paths[5]],
    ]
    
    # Load without streaming
    patches = batch_image_loader.load_image_files_optimized(
        img_paths,
        patch_size=14,
        max_workers=2,
        use_streaming=False,
    )
    
    # Check output shape
    num_batches = 3
    num_images_per_batch = 2
    num_patches = (896 // 14) ** 2  # Default size
    patch_dim = 14 * 14 * 3
    
    self.assertEqual(
        patches.shape,
        (num_batches, num_images_per_batch, num_patches, patch_dim)
    )
    
    # Test with streaming
    patches_streaming = batch_image_loader.load_image_files_optimized(
        img_paths,
        patch_size=14,
        max_workers=2,
        use_streaming=True,
        batch_size=2,
    )
    
    self.assertEqual(patches_streaming.shape, patches.shape)
  
  def test_none_handling(self):
    """Test handling of None image paths."""
    # Test all None case
    result = batch_image_loader.load_image_files_optimized([[None]])
    self.assertIsNone(result)
    
    # Test mixed None case (should raise error)
    with self.assertRaises(ValueError):
      batch_image_loader.load_image_files_optimized(
          [[self.image_paths[0], None]]
      )
  
  def test_context_manager(self):
    """Test context manager functionality."""
    with batch_image_loader.BatchImageLoader(
        image_height=224,
        image_width=224,
        patch_size=14,
    ) as loader:
      patches = loader.load_batch(self.image_paths[:2])
      self.assertIsNotNone(patches)
    
    # Executor should be shut down after context exit
    self.assertTrue(loader._executor._shutdown)
  
  def test_performance_comparison(self):
    """Compare performance with original implementation (if available)."""
    import time
    
    # Time the optimized version
    start = time.time()
    patches_opt = batch_image_loader.load_images_parallel(
        self.image_paths,
        image_height=224,
        image_width=224,
        patch_size=14,
        max_workers=4,
    )
    time_parallel = time.time() - start
    
    # Time sequential loading for comparison
    start = time.time()
    patches_seq = []
    for path in self.image_paths:
      patches_seq.append(
          batch_image_loader.load_images_parallel(
              [path],
              image_height=224,
              image_width=224,
              patch_size=14,
              max_workers=1,
          )
      )
    patches_seq = jnp.concatenate(patches_seq, axis=0)
    time_sequential = time.time() - start
    
    # Parallel should be faster (or at least not significantly slower)
    # Note: For small test cases, overhead might make parallel slower
    print(f"Parallel time: {time_parallel:.3f}s")
    print(f"Sequential time: {time_sequential:.3f}s")
    print(f"Speedup: {time_sequential/time_parallel:.2f}x")
    
    # Check that results are the same
    np.testing.assert_allclose(patches_opt, patches_seq, rtol=1e-5)


if __name__ == "__main__":
  unittest.main()