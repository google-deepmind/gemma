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

"""Example demonstrating optimized batch image loading for Gemma multimodal models.

This example shows how to use the new batch image loading functionality
which provides significant performance improvements through:
- Parallel image loading using thread pools
- Memory-efficient streaming for large datasets
- Removal of TensorFlow dependency
- Batch processing of image transformations
"""

import time
from pathlib import Path
from typing import List, Optional

import jax
import numpy as np
from PIL import Image

from gemma.multimodal import batch_image_loader
from gemma.multimodal import image as original_image


def create_dummy_images(num_images: int, output_dir: Path) -> List[str]:
  """Create dummy images for demonstration.
  
  Args:
    num_images: Number of images to create.
    output_dir: Directory to save images.
    
  Returns:
    List of image file paths.
  """
  output_dir.mkdir(exist_ok=True)
  image_paths = []
  
  for i in range(num_images):
    # Create a simple test image with gradients
    img_array = np.zeros((512, 512, 3), dtype=np.uint8)
    img_array[:, :, 0] = np.linspace(0, 255, 512).astype(np.uint8)[:, None]
    img_array[:, :, 1] = np.linspace(0, 255, 512).astype(np.uint8)[None, :]
    img_array[:, :, 2] = (i * 255 // num_images)
    
    img = Image.fromarray(img_array)
    img_path = output_dir / f"sample_image_{i:03d}.jpg"
    img.save(img_path, "JPEG")
    image_paths.append(str(img_path))
  
  return image_paths


def example_basic_parallel_loading():
  """Example 1: Basic parallel image loading."""
  print("\n" + "="*60)
  print("Example 1: Basic Parallel Image Loading")
  print("="*60)
  
  # Create sample images
  temp_dir = Path("temp_images")
  image_paths = create_dummy_images(8, temp_dir)
  
  # Load images in parallel
  print(f"\nLoading {len(image_paths)} images in parallel...")
  start_time = time.time()
  
  patches = batch_image_loader.load_images_parallel(
      image_paths,
      image_height=224,  # Smaller size for faster processing
      image_width=224,
      patch_size=14,
      max_workers=4,  # Use 4 parallel workers
  )
  
  elapsed = time.time() - start_time
  print(f"Loaded in {elapsed:.2f} seconds")
  print(f"Output shape: {patches.shape}")
  
  # Clean up
  import shutil
  shutil.rmtree(temp_dir)


def example_streaming_large_dataset():
  """Example 2: Streaming for large datasets."""
  print("\n" + "="*60)
  print("Example 2: Memory-Efficient Streaming")
  print("="*60)
  
  # Create sample images
  temp_dir = Path("temp_images_streaming")
  num_images = 20
  image_paths = create_dummy_images(num_images, temp_dir)
  
  print(f"\nStreaming {num_images} images in batches of 4...")
  
  # Create batch loader with streaming
  loader = batch_image_loader.BatchImageLoader(
      image_height=224,
      image_width=224,
      patch_size=14,
      batch_size=4,
      max_workers=2,
      prefetch_size=2,  # Prefetch 2 batches ahead
  )
  
  with loader:
    batch_count = 0
    total_patches = 0
    
    for batch_patches in loader.stream_batches(image_paths):
      batch_count += 1
      total_patches += batch_patches.shape[0]
      print(f"  Batch {batch_count}: shape {batch_patches.shape}")
    
    print(f"\nProcessed {total_patches} images in {batch_count} batches")
  
  # Clean up
  import shutil
  shutil.rmtree(temp_dir)


def example_gemma_multimodal_integration():
  """Example 3: Integration with Gemma multimodal models."""
  print("\n" + "="*60)
  print("Example 3: Gemma Multimodal Integration")
  print("="*60)
  
  # Create sample images
  temp_dir = Path("temp_images_gemma")
  image_paths = create_dummy_images(6, temp_dir)
  
  # Organize images for multimodal input (2 batches, 3 images each)
  img_paths_nested = [
      [image_paths[0], image_paths[1], image_paths[2]],
      [image_paths[3], image_paths[4], image_paths[5]],
  ]
  
  print(f"\nLoading images for multimodal model...")
  print(f"Structure: {len(img_paths_nested)} batches, "
        f"{len(img_paths_nested[0])} images per batch")
  
  # Use optimized loader (drop-in replacement)
  patches = batch_image_loader.load_image_files_optimized(
      img_paths_nested,
      patch_size=14,
      max_workers=4,
      use_streaming=False,  # For small datasets, streaming isn't needed
  )
  
  print(f"Output shape: {patches.shape}")
  print(f"  Batches: {patches.shape[0]}")
  print(f"  Images per batch: {patches.shape[1]}")
  print(f"  Patches per image: {patches.shape[2]}")
  print(f"  Patch dimension: {patches.shape[3]}")
  
  # Clean up
  import shutil
  shutil.rmtree(temp_dir)


def example_performance_comparison():
  """Example 4: Performance comparison with sequential loading."""
  print("\n" + "="*60)
  print("Example 4: Performance Comparison")
  print("="*60)
  
  # Create sample images
  temp_dir = Path("temp_images_perf")
  num_images = 12
  image_paths = create_dummy_images(num_images, temp_dir)
  
  print(f"\nComparing loading methods for {num_images} images...")
  
  # Sequential loading (one by one)
  print("\n1. Sequential loading:")
  start_time = time.time()
  patches_seq = []
  for path in image_paths:
    patches = batch_image_loader.load_images_parallel(
        [path],
        image_height=224,
        image_width=224,
        patch_size=14,
        max_workers=1,
    )
    patches_seq.append(patches)
  patches_sequential = jax.numpy.concatenate(patches_seq, axis=0)
  time_sequential = time.time() - start_time
  print(f"   Time: {time_sequential:.2f} seconds")
  
  # Parallel loading (optimized)
  print("\n2. Parallel loading (4 workers):")
  start_time = time.time()
  patches_parallel = batch_image_loader.load_images_parallel(
      image_paths,
      image_height=224,
      image_width=224,
      patch_size=14,
      max_workers=4,
  )
  time_parallel = time.time() - start_time
  print(f"   Time: {time_parallel:.2f} seconds")
  
  # Streaming with prefetch
  print("\n3. Streaming with prefetch:")
  start_time = time.time()
  loader = batch_image_loader.BatchImageLoader(
      image_height=224,
      image_width=224,
      patch_size=14,
      batch_size=3,
      max_workers=2,
      prefetch_size=2,
  )
  with loader:
    batches = list(loader.stream_batches(image_paths))
    patches_streaming = jax.numpy.concatenate(batches, axis=0)
  time_streaming = time.time() - start_time
  print(f"   Time: {time_streaming:.2f} seconds")
  
  # Calculate speedups
  print("\nSpeedup Analysis:")
  print(f"  Parallel vs Sequential: {time_sequential/time_parallel:.2f}x faster")
  print(f"  Streaming vs Sequential: {time_sequential/time_streaming:.2f}x faster")
  
  # Verify outputs are the same
  np.testing.assert_allclose(patches_sequential, patches_parallel, rtol=1e-5)
  np.testing.assert_allclose(patches_sequential, patches_streaming, rtol=1e-5)
  print("\n✓ All methods produce identical results")
  
  # Clean up
  import shutil
  shutil.rmtree(temp_dir)


def example_custom_preprocessing():
  """Example 5: Custom preprocessing options."""
  print("\n" + "="*60)
  print("Example 5: Custom Preprocessing Options")
  print("="*60)
  
  # Create a sample image
  temp_dir = Path("temp_images_custom")
  temp_dir.mkdir(exist_ok=True)
  
  # Create a high-resolution image
  img_array = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
  img = Image.fromarray(img_array)
  img_path = temp_dir / "high_res_image.jpg"
  img.save(img_path, "JPEG")
  
  print("\nProcessing high-resolution image with different settings:")
  
  # Standard processing
  print("\n1. Standard (896x896, with JPEG compression):")
  start = time.time()
  processed_standard = batch_image_loader.pre_process_image_pil(
      img,
      image_height=896,
      image_width=896,
      use_jpeg_compression=True,
  )
  print(f"   Shape: {processed_standard.shape}")
  print(f"   Time: {time.time() - start:.3f}s")
  
  # Lower resolution for faster processing
  print("\n2. Low resolution (224x224, no compression):")
  start = time.time()
  processed_low = batch_image_loader.pre_process_image_pil(
      img,
      image_height=224,
      image_width=224,
      use_jpeg_compression=False,
  )
  print(f"   Shape: {processed_low.shape}")
  print(f"   Time: {time.time() - start:.3f}s")
  
  # Custom resolution
  print("\n3. Custom resolution (512x384):")
  start = time.time()
  processed_custom = batch_image_loader.pre_process_image_pil(
      img,
      image_height=512,
      image_width=384,
      use_jpeg_compression=False,
  )
  print(f"   Shape: {processed_custom.shape}")
  print(f"   Time: {time.time() - start:.3f}s")
  
  # Clean up
  import shutil
  shutil.rmtree(temp_dir)


def main():
  """Run all examples."""
  print("\n" + "="*60)
  print("GEMMA BATCH IMAGE LOADING EXAMPLES")
  print("="*60)
  print("\nThis demonstrates the optimized batch image loading functionality")
  print("for Gemma multimodal models, providing:")
  print("  • Parallel image loading with configurable workers")
  print("  • Memory-efficient streaming for large datasets")
  print("  • Removal of TensorFlow dependency")
  print("  • Significant performance improvements")
  
  # Run examples
  example_basic_parallel_loading()
  example_streaming_large_dataset()
  example_gemma_multimodal_integration()
  example_performance_comparison()
  example_custom_preprocessing()
  
  print("\n" + "="*60)
  print("All examples completed successfully!")
  print("="*60)


if __name__ == "__main__":
  main()