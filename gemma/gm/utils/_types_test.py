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

from gemma import gm
from gemma.gm.utils import _types
import jax.numpy as jnp
import numpy as np


def test_types_text_only():
  config = _types.InputConfig(
      support_images=True,
      num_tokens_per_image=4,
      special_tokens=gm.text.Gemma3Tokenizer.special_tokens,
  )
  input = _types.Input(  # pylint: disable=redefined-builtin
      text=jnp.asarray([
          [1, 2, 3, 4],
          [1, 2, 0, 0],
          [1, 2, 3, 0],
      ]),
      images=None,
      config=config,
  )

  assert input.batch_size == 3
  assert input.max_num_images == 0
  assert input.length_with_mm == 4

  np.testing.assert_array_equal(
      input.tokens_with_mm,
      input.text,  # No MM tokens
  )

  np.testing.assert_array_equal(
      input.inputs_mask,
      [
          [1, 1, 1, 1],
          [1, 1, 0, 0],
          [1, 1, 1, 0],
      ],
  )

  np.testing.assert_array_equal(
      input.attention_mask,
      [
          # Sequence 0
          [
              [1, 0, 0, 0],
              [1, 1, 0, 0],
              [1, 1, 1, 0],
              [1, 1, 1, 1],
          ],
          # Sequence 1
          [
              [1, 0, 0, 0],
              [1, 1, 0, 0],
              [1, 1, 0, 0],  # Masked
              [1, 1, 0, 0],  # Masked
          ],
          # Sequence 2
          [
              [1, 0, 0, 0],
              [1, 1, 0, 0],
              [1, 1, 1, 0],
              [1, 1, 1, 0],  # Masked
          ],
      ],
  )

  np.testing.assert_array_equal(
      input.positions,
      [
          [0, 1, 2, 3],
          [0, 1, 1, 1],  # Padded tokens have position of the last value
          [0, 1, 2, 2],  # Padded tokens have position of the last value
      ],
  )

  np.testing.assert_array_equal(
      input.last_token_pos,
      [3, 1, 2],
  )

  np.testing.assert_array_equal(
      input.last_token,
      [4, 2, 3],
  )


def test_types_text_with_images():
  IMG = gm.text.Gemma3Tokenizer.special_tokens.START_OF_IMAGE  # pylint: disable=invalid-name
  IMG_TOKS = [108, 255999, -2, -2, -2, 256000, 108]  # pylint: disable=invalid-name
  PAD = [0] * 6  # pylint: disable=invalid-name
  config = _types.InputConfig(
      support_images=True,
      num_tokens_per_image=3,
      special_tokens=gm.text.Gemma3Tokenizer.special_tokens,
  )
  input = _types.Input(  # pylint: disable=redefined-builtin
      text=jnp.asarray([
          [1, 2, IMG, IMG],
          [1, 2, 0, 0],
          [1, IMG, 3, 0],
      ]),
      images=jnp.zeros((3, 2, 4, 4, 3), dtype=jnp.uint8),  # b n h w c
      config=config,
  )

  assert input.batch_size == 3
  assert input.max_num_images == 2
  assert input.length_with_mm == 2 + len(IMG_TOKS) * 2
  assert input.length_with_mm == 16

  np.testing.assert_array_equal(
      input.tokens_with_mm,
      [
          [1, 2, *IMG_TOKS, *IMG_TOKS],
          [1, 2, 0, 0, *PAD, *PAD],
          [1, *IMG_TOKS, 3, 0, *PAD],
      ],
  )

  np.testing.assert_array_equal(
      input.attention_mask,
      [
          [
              # Bidir mask:
              # [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # \n\n
              [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <start>
              [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # bidir
              [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # bidir
              [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # bidir
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # <end>
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # \n\n
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # \n\n
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # <start>
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # bidir
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # bidir
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # bidir
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # <end>
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # \n\n
          ],
          [
              # Bidir mask:
              # [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          ],
          [
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # \n\n
              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <start>
              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # bidir
              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # bidir
              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # bidir
              [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <end>
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # \n\n
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          ],
      ],
  )

  np.testing.assert_array_equal(
      input.inputs_mask,
      [
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
      ],
  )
  np.testing.assert_array_equal(
      input.positions,
      [
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8],
      ],
  )

  np.testing.assert_array_equal(
      input.last_token_pos,
      [15, 1, 8],
  )

  np.testing.assert_array_equal(
      input.last_token,
      [108, 2, 3],
  )


def test_types_text_only_pad():
  config = _types.InputConfig(
      support_images=True,
      num_tokens_per_image=4,
      special_tokens=gm.text.Gemma3Tokenizer.special_tokens,
  )
  input = _types.Input(  # pylint: disable=redefined-builtin
      text=jnp.asarray([
          [1, 2, 3, 4],
          [1, 2, 0, 0],
          [1, 2, 3, 0],
      ]),
      images=None,
      config=config,
  )
  input = input.pad(7)

  assert input.batch_size == 3
  assert input.max_num_images == 0
  assert input.length_with_mm == 7

  np.testing.assert_array_equal(
      input.tokens_with_mm,
      [
          [1, 2, 3, 4, 0, 0, 0],
          [1, 2, 0, 0, 0, 0, 0],
          [1, 2, 3, 0, 0, 0, 0],
      ],
  )

  np.testing.assert_array_equal(
      input.inputs_mask,
      [
          [1, 1, 1, 1, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0],
      ],
  )

  np.testing.assert_array_equal(
      input.positions,
      [
          [0, 1, 2, 3, 3, 3, 3],
          [0, 1, 1, 1, 1, 1, 1],
          [0, 1, 2, 2, 2, 2, 2],
      ],
  )

  np.testing.assert_array_equal(
      input.last_token_pos,
      [3, 1, 2],
  )

  np.testing.assert_array_equal(
      input.last_token,
      [4, 2, 3],
  )
