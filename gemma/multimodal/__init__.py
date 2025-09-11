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

"""Gemma multimodal module for vision and image processing."""

from gemma.multimodal import batch_image_loader
from gemma.multimodal import image
from gemma.multimodal import vision
from gemma.multimodal import vision_utils

# Export optimized batch loading functions
from gemma.multimodal.batch_image_loader import (
    BatchImageLoader,
    load_images_parallel,
    load_image_files_optimized,
    pre_process_image_pil,
)

# Export original functions for compatibility
from gemma.multimodal.image import (
    load_image_files,
    normalize_images,
    patchify_images,
    pre_process_image,
)

__all__ = [
    # Modules
    "batch_image_loader",
    "image",
    "vision",
    "vision_utils",
    # Optimized batch loading
    "BatchImageLoader",
    "load_images_parallel",
    "load_image_files_optimized",
    "pre_process_image_pil",
    # Original functions
    "load_image_files",
    "normalize_images",
    "patchify_images",
    "pre_process_image",
]