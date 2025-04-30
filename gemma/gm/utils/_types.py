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

"""Type utils."""

import dataclasses
import functools

import flax
from gemma.gm.utils import _attention_mask
from gemma.gm.vision import _token_utils
import jax
from kauldron.typing import Bool, Int, UInt8  # pylint: disable=g-multiple-import,g-importing-member

_PADDING_ID = 0


@dataclasses.dataclass(kw_only=True, frozen=True)
class InputConfig:
  """Metadata about the model."""

  support_images: bool
  num_tokens_per_image: int


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

  text: Int['B length_no_mm']
  images: UInt8['B N H W C'] | None

  # Model metadata.
  config: InputConfig = flax.struct.field(pytree_node=False)

  def __post_init__(self):
    if self.images is not None and not self.config.support_images:
      raise ValueError(
          'Images are provided, but the model does not support vision.'
      )

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
    """Mask (after the extra tokens are added."""
    return self.tokens_with_mm != _PADDING_ID

  @property
  @jax.jit
  def attention_mask(self) -> Bool['B length_with_mm length_with_mm']:
    """Attention mask for the input."""

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
