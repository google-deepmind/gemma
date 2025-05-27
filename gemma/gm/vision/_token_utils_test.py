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

from gemma.gm.vision import _token_utils
import jax.numpy as jnp
import numpy as np
import pytest

_IMAGE_TOKENS = [108, 255999, -2, -2, -2, 256000, 108]
_PAD = [0] * (len(_IMAGE_TOKENS) - 1)


@pytest.mark.parametrize(
    'max_num_images, tokens, expected_output',
    [
        (2, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, *_PAD, *_PAD]),
        (2, [1, 2, 3, 255999, 4], [1, 2, 3, *_IMAGE_TOKENS, 4, *_PAD]),
        (
            2,
            [1, 2, 3, 255999, 255999],
            [1, 2, 3, *_IMAGE_TOKENS, *_IMAGE_TOKENS],
        ),
        (
            2,
            [1, 255999, 2, 255999, 3],
            [1, *_IMAGE_TOKENS, 2, *_IMAGE_TOKENS, 3],
        ),
    ],
)
def test_single_add_extra_tokens_for_images(
    tokens,
    max_num_images,
    expected_output,
):
  tokens = jnp.array(tokens)

  tokens_with_mm = _token_utils.add_extra_tokens_for_images(
      tokens=tokens[None, ...],
      max_num_images=max_num_images,
      num_tokens_per_image=3,
  )

  assert (
      _token_utils.get_num_mm_tokens(
          max_num_images=0,
          num_tokens_per_image=3,
      )
      == 0
  )

  # assert tokens_with_mm.shape == (1, len(expected_output))
  assert tokens_with_mm[0].tolist() == expected_output


def test_batched_add_extra_tokens_for_images():
  # Batched version
  # fmt: off
  # pylint: disable=bad-whitespace
  tokens = [
      [1,     2,   3, 255999,      5,    6,   7, 8, 9],
      [10,   20,  30, 255999, 255999,   60,   0, 0, 0],
      [100, 200, 300,    400,    500,  600, 700, 0, 0],
  ]
  # pylint: enable=bad-whitespace
  # fmt: on
  tokens = jnp.array(tokens)

  tokens_with_mm = _token_utils.add_extra_tokens_for_images(
      tokens=tokens,
      max_num_images=2,
      num_tokens_per_image=3,
  )

  expected = [
      [1, 2, 3, *_IMAGE_TOKENS, 5, 6, 7, 8, 9, *_PAD],
      [10, 20, 30, *_IMAGE_TOKENS, *_IMAGE_TOKENS, 60, 0, 0, 0],
      [100, 200, 300, 400, 500, 600, 700, 0, 0, *_PAD, *_PAD],
  ]
  expected = jnp.array(expected)

  np.testing.assert_array_equal(tokens_with_mm, expected)


@pytest.mark.parametrize('num_images', [2, 3])
def test_single_merge_embeddings(num_images):
  mask = jnp.array([0, 0, 1, 1, 1, 0, 1, 1, 1]).astype(bool)
  text_embeddings = jnp.ones((9, 2))
  vision_embeddings = jnp.ones((num_images, 3, 2)) * 10

  embeddings = _token_utils.merge_embeddings(
      text_embeddings=text_embeddings[None, ...],
      vision_embeddings=vision_embeddings[None, ...],
      mask=mask[None, ...],
  )
  expected = jnp.array([
      [1.0, 1.0],
      [1.0, 1.0],
      [10.0, 10.0],
      [10.0, 10.0],
      [10.0, 10.0],
      [1.0, 1.0],
      [10.0, 10.0],
      [10.0, 10.0],
      [10.0, 10.0],
  ])
  np.testing.assert_array_equal(embeddings, expected[None, ...])


@pytest.mark.parametrize('num_images', [2, 3])
def test_batched_merge_embeddings(num_images):
  mask = jnp.array([
      [0, 0, 1, 1, 1, 0, 1, 1, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 1, 0, 0],
  ]).astype(bool)
  text_embeddings = jnp.ones((3, 9, 2))
  vision_embeddings = jnp.ones((3, num_images, 3, 2)) * 10

  text_embeddings = jnp.einsum('bld,l->bld', text_embeddings, jnp.arange(9) + 1)
  vision_embeddings = jnp.einsum(
      'bnpd,np->bnpd',
      vision_embeddings,
      jnp.arange(num_images * 3).reshape(num_images, 3) + 1,
  )

  embeddings = _token_utils.merge_embeddings(
      text_embeddings=text_embeddings,
      vision_embeddings=vision_embeddings,
      mask=mask,
  )

  expected = jnp.array([
      [
          [1, 1],
          [2, 2],
          [10, 10],
          [20, 20],
          [30, 30],
          [6, 6],
          [40, 40],
          [50, 50],
          [60, 60],
      ],
      [
          [1, 1],
          [2, 2],
          [3, 3],
          [4, 4],
          [5, 5],
          [6, 6],
          [7, 7],
          [8, 8],
          [9, 9],
      ],
      [
          [1, 1],
          [2, 2],
          [3, 3],
          [4, 4],
          [10, 10],  # Encoding at `vision_embeddings[2, 0]`
          [20, 20],
          [30, 30],
          [8, 8],
          [9, 9],
      ],
  ])
  np.testing.assert_array_equal(embeddings, expected)
