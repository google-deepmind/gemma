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
from etils import epath
import jax
from jax import numpy as jnp
from kauldron import typing
import numpy as np
from PIL import Image
import tensorflow as tf

_IMAGE_MEAN = (127.5,) * 3
_IMAGE_STD = (127.5,) * 3
_DEFAULT_IMAGE_SIZE = 896
_DEFAULT_PATCH_SIZE = 14


@typing.typechecked
def normalize_images(images: typing.Float["H W C"]) -> typing.Float["H W C"]:
    """Normalize images to zero mean and unit variance."""
    images = (images - jnp.asarray(_IMAGE_MEAN)) / jnp.asarray(_IMAGE_STD)
    return images


def pre_process_image(
    image: typing.Float["H W C"],
    *,
    image_height: int = _DEFAULT_IMAGE_SIZE,
    image_width: int = _DEFAULT_IMAGE_SIZE,
) -> typing.Float["H W C"]:
    """Resize and normalize the image."""
    image = jnp.asarray(
        tf.image.decode_jpeg(tf.io.encode_jpeg(image), channels=3)
    )
    image = jax.image.resize(
        image, shape=(image_height, image_width, 3), method="bilinear", antialias=True
    )
    return jnp.clip(normalize_images(image), -1, 1)


@typing.typechecked
def patchify_images(
    images: typing.Float["B H W C"],
    patch_size: int = _DEFAULT_PATCH_SIZE,
    padding: str = "VALID",
) -> typing.Float["P D"]:
    """Extract patches from images."""
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
    return einops.rearrange(patches, "... (c p) -> (...) (p c)", c=channels)


@typing.typechecked
def load_image_files(
    img_paths: Sequence[Sequence[str | None]], patch_size: int = _DEFAULT_PATCH_SIZE
) -> typing.Float["B S P D"] | None:
    """Load image files and extract patches."""
    if img_paths == [[None]]:
        return None

    patches = []
    for imgs_path in img_paths:
        tmp = []
        for img_path in imgs_path:
            if img_path is None:
                raise ValueError(
                    "All img_paths must be either None or valid paths, mixed values are not supported."
                )
            with epath.Path(img_path).open("rb") as f:
                img = pre_process_image(np.array(Image.open(f).convert("RGB")))
            tmp.append(patchify_images(img[None, ...], patch_size))
        patches.append(jnp.asarray(tmp))

    return jnp.asarray(patches)

