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

"""Full VisionEncoder for Gemma4 that composes entry, transformer, and exit.

Supports batches of variable-aspect-ratio images. Each image is patchified
individually, padded to a common max_patches length, batched through the
encoder, and then padding is stripped from the output soft tokens.
"""

from flax import linen as nn
from gemma.gm.nn.gemma4.vision import _images
from gemma.gm.nn.gemma4.vision import _layers
from gemma.gm.nn.gemma4.vision import _transformer
import jax.numpy as jnp
from kauldron.ktyping import Bool, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member

BEGIN_IMAGE_TOKEN = 255999
END_IMAGE_TOKEN = 262144
NEW_LINE_TOKEN = 108
TOKEN_PLACEHOLDER = -2
POSITIONS_PAD_VALUE = _layers.POSITIONS_PAD_VALUE


class VisionEncoder(nn.Module):
  """Vision encoder for Gemma4."""

  d_model: int = 768
  num_layers: int = 16
  num_heads: int = 12
  ffw_hidden: int = 3072
  patch_size: int = 16
  output_length: int | tuple[int, ...] = 280
  pos_emb_shape_yx: tuple[int, int] = (10240, 2)
  pooling_kernel_size: int = 3
  use_clipped_linears: bool = False
  standardize_embeddings: bool = False

  def setup(self):
    self.entry = _layers.VisionEntry(
        d_model=self.d_model,
        patch_size=self.patch_size,
        pos_emb_shape_yx=self.pos_emb_shape_yx,
    )
    self.transformer = _transformer.VisionTransformer(
        d_model=self.d_model,
        ffw_hidden=self.ffw_hidden,
        num_heads=self.num_heads,
        num_layers=self.num_layers,
        use_clipped_linears=self.use_clipped_linears,
    )
    self.exit = _layers.VisionExit(
        d_model=self.d_model,
        output_length=self.output_length,
    )
    if self.standardize_embeddings:
      self.standardize = _layers.Standardize(name='standardize')

  @property
  def max_patches(self) -> int:
    output_len = self.output_length
    if isinstance(output_len, tuple):
      output_len = max(output_len)
    return int(output_len * self.pooling_kernel_size**2)

  @property
  def num_mm_tokens_per_image(self) -> int:
    output_len = self.output_length
    if isinstance(output_len, tuple):
      output_len = max(output_len)
    return int(output_len)

  @property
  def image_height(self) -> int:
    side = int(self.max_patches**0.5) * self.patch_size
    return side

  @property
  def image_width(self) -> int:
    side = int(self.max_patches**0.5) * self.patch_size
    return side

  @typechecked
  def __call__(
      self,
      patches: Float['B L P'],
      positions_xy: Int['B L 2'],
  ) -> tuple[tuple[Float['B l D'], Bool['B l'] | None], ...]:
    input_mask = jnp.logical_not(
        (positions_xy == POSITIONS_PAD_VALUE).all(axis=-1)
    )

    embeddings = self.entry(patches, positions_xy=positions_xy)

    transformer_output = self.transformer(
        inputs=embeddings,
        input_mask=input_mask,
        positions_xy=positions_xy,
    )

    outputs = self.exit(
        transformer_output,
        positions_xy=positions_xy,
    )

    if self.standardize_embeddings:
      standardized_outputs = []
      for emb, mask in outputs:
        dtype = emb.dtype
        emb_std = self.standardize(emb.astype(jnp.float32)).astype(dtype)
        standardized_outputs.append((emb_std, mask))
      outputs = tuple(standardized_outputs)

    return outputs


def patchify_and_pad(
    images: list[jnp.ndarray],
    patch_size: int = 16,
    max_patches: int | None = None,
    max_soft_tokens: int = 1120,
    pooling_kernel_size: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, list[int]]:
  """Patchify variable-size images and pad to a common sequence length.

  Each image is independently patchified, then all are padded to max_patches
  so they can be batched together for the vision encoder.

  Args:
    images: List of preprocessed images as float32 arrays in [0,1] range. Each
      has shape [H_i, W_i, 3] with potentially different H_i, W_i.
    patch_size: Patch size in pixels.
    max_patches: Maximum number of patches (overrides max_soft_tokens).
    max_soft_tokens: Maximum soft tokens (used to compute max_patches if
      max_patches is not provided).
    pooling_kernel_size: Pooling kernel size for computing max_patches.

  Returns:
    patches: Padded patches [B, max_patches, patch_dim].
    positions_xy: Padded positions [B, max_patches, 2], with -1 for padding.
    num_real_patches_per_image: Number of real (non-padding) patches per image.
  """
  if max_patches is None:
    max_patches = max_soft_tokens * pooling_kernel_size**2

  all_patches = []
  all_positions = []
  num_real_patches_per_image = []

  for image in images:
    img_patches, img_positions = _images.patchify(image[None], patch_size)
    img_patches = img_patches[0]
    img_positions = img_positions[0]

    num_real = img_patches.shape[0]
    num_real_patches_per_image.append(num_real)

    num_padding = max_patches - num_real
    if num_padding > 0:
      pad_patches = jnp.zeros(
          (num_padding, img_patches.shape[-1]), dtype=img_patches.dtype
      )
      img_patches = jnp.concatenate([img_patches, pad_patches], axis=0)

      pad_positions = jnp.full(
          (num_padding, 2), POSITIONS_PAD_VALUE, dtype=img_positions.dtype
      )
      img_positions = jnp.concatenate([img_positions, pad_positions], axis=0)

    all_patches.append(img_patches)
    all_positions.append(img_positions)

  patches = jnp.stack(all_patches, axis=0)
  positions_xy = jnp.stack(all_positions, axis=0)

  return patches, positions_xy, num_real_patches_per_image


def extract_soft_tokens(
    encoder_outputs: tuple[tuple[jnp.ndarray, jnp.ndarray | None], ...],
) -> list[jnp.ndarray]:
  """Extract non-padding soft tokens from encoder outputs.

  Given the output of VisionEncoder.__call__, strips padding tokens from each
  image using the mask, returning a list of variable-length token arrays.

  Args:
    encoder_outputs: Tuple of (embeddings, mask) pairs from VisionEncoder.

  Returns:
    List of soft token arrays, one per image in the batch. Each has shape
    [num_real_tokens_i, D] where num_real_tokens_i may differ per image.
  """
  embeddings, mask = encoder_outputs[0]
  batch_size = embeddings.shape[0]

  result = []
  for i in range(batch_size):
    if mask is not None:
      real_tokens = embeddings[i][mask[i]]
    else:
      real_tokens = embeddings[i]
    result.append(real_tokens)

  return result


def pad_soft_tokens_to_max(
    soft_tokens_list: list[jnp.ndarray],
    num_mm_tokens_per_image: int,
) -> jnp.ndarray:
  """Pad variable-length soft tokens to a uniform length.

  After the VisionEncoder, each image may produce a different number of soft
  tokens. This function pads each to num_mm_tokens_per_image so they can be
  uniformly merged into the text sequence.

  Args:
    soft_tokens_list: List of arrays, each [num_real_tokens_i, D].
    num_mm_tokens_per_image: Target uniform length.

  Returns:
    Padded soft tokens [N_images, num_mm_tokens_per_image, D].
  """
  d_model = soft_tokens_list[0].shape[-1]
  padded = []
  for tokens in soft_tokens_list:
    num_real = tokens.shape[0]
    pad_len = num_mm_tokens_per_image - num_real
    if pad_len > 0:
      tokens = jnp.concatenate(
          [tokens, jnp.zeros((pad_len, d_model), dtype=tokens.dtype)], axis=0
      )
    elif pad_len < 0:
      tokens = tokens[:num_mm_tokens_per_image]
    padded.append(tokens)
  return jnp.stack(padded, axis=0)
