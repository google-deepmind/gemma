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

"""Type utils."""

from __future__ import annotations

import dataclasses
import functools

import flax
from gemma.gm.data import _functional
from gemma.gm.math import _pos_utils
from gemma.gm.text import _tokenizer
from gemma.gm.utils import _attention_mask
from gemma.gm.vision import _token_utils
import jax
import jax.numpy as jnp
from kauldron.typing import Bool, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member

_PADDING_ID = 0


@dataclasses.dataclass(kw_only=True, frozen=True)
class InputConfig:
  """Metadata about the model."""

  support_images: bool
  num_tokens_per_image: int
  # <start_of_image>,...
  special_tokens: type[_tokenizer.SpecialTokens]


@flax.struct.dataclass(kw_only=True, frozen=True)
class Input:
  """Gemma model input.

  Wrapper around the model inputs to abstract multi-modal tokens, mask
  computation,...

  Attributes:
    text: Text after tokenization. Can contain images placeholders.
    images: Images.
    config: Model config.
  """

  # Name `text` rather than `tokens` to avoid accidental usage instead of
  # `tokens_with_mm`.
  text: Int['B length_no_mm']
  images: UInt8['B N H W C'] | None

  # Model metadata.
  config: InputConfig = flax.struct.field(pytree_node=False)

  def __post_init__(self):
    if self.images is not None and not self.config.support_images:
      raise ValueError(
          'Images are provided, but the model does not support vision.'
      )

  def pad(self, length_with_mm: int) -> Input:
    old_text_len = self.text.shape[-1]
    extra_mm_tokens = self.length_with_mm - old_text_len
    new_text_len = length_with_mm - extra_mm_tokens

    return dataclasses.replace(
        self,
        text=_functional.pad(self.text, max_length=new_text_len),
    )

  @functools.cached_property
  def batch_size(self) -> int:
    """Batch size."""
    return len(self.text)

  @functools.cached_property
  def max_num_images(self) -> int:
    """Maximum number of images in one sequence."""
    if self.images is None:
      max_num_images = 0
    else:
      _, max_num_images, _, _, _ = self.images.shape
    return max_num_images

  @functools.cached_property
  def length_with_mm(self) -> int:
    """Total length, after the multi-modal soft tokens are inserted."""
    if self.config.support_images:
      num_tokens_per_image = self.config.num_tokens_per_image
    else:
      num_tokens_per_image = 0

    inserted_mm_tokens = _token_utils.get_num_mm_tokens(
        max_num_images=self.max_num_images,
        num_tokens_per_image=num_tokens_per_image,
    )
    return self.text.shape[-1] + inserted_mm_tokens

  @property
  @jax.jit
  def tokens_with_mm(self) -> Int['B length_with_mm']:
    """Tokens after inserting placeholders for images."""
    # No images, tokens are only text.
    if not self.config.support_images or self.images is None:
      return self.text

    return _token_utils.add_extra_tokens_for_images(
        self.text,
        max_num_images=self.max_num_images,
        num_tokens_per_image=self.config.num_tokens_per_image,
    )

  @property
  @jax.jit
  def inputs_mask(self) -> Bool['B length_with_mm']:
    """Mask (after the extra MM tokens are added)."""
    return self.tokens_with_mm != _PADDING_ID

  @property
  @jax.jit
  def attention_mask(self) -> Bool['B length_with_mm length_with_mm']:
    """Attention mask for the input (include MM tokens)."""

    if self.images is not None:
      bidirectional_mask = (
          self.tokens_with_mm == _token_utils.SOFT_TOKEN_PLACEHOLDER
      )
    else:
      bidirectional_mask = None

    return _attention_mask.make_causal_bidirectional_attention_mask(
        self.inputs_mask,
        bidirectional_mask=bidirectional_mask,
    )

  @property
  @jax.jit
  def positions(self) -> Int['B length_with_mm']:
    """Positions for the input (always including the MM tokens)."""
    return _pos_utils.build_positions_from_mask(self.inputs_mask)

  @property
  @jax.jit
  def last_token_pos(self) -> Int['B']:
    """Position of the last token in the sentence (after MM tokens)."""
    # Could also be `self.positions.max(axis=-1)`
    return jnp.sum(self.inputs_mask, axis=-1) - 1

  @property
  @jax.jit
  def last_token(self) -> Int['B']:
    """Last token in the sentence (after MM tokens).

    Used as the first input token of the model for the auto-regressive sampling.
    """
    x = jnp.take_along_axis(
        self.tokens_with_mm, self.last_token_pos[:, None], axis=-1
    )
    x = jnp.squeeze(x, axis=-1)
    return x
