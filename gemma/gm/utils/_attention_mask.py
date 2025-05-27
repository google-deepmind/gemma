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

"""Attention mask utilities."""

from __future__ import annotations

import jax.numpy as jnp
from kauldron.typing import Bool, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@typechecked
def make_causal_bidirectional_attention_mask(
    causal_mask: Bool['B L'],
    *,
    bidirectional_mask: Bool['B L'] | None = None,
) -> Bool['B L L']:
  """Make the attention mask for the transformer.

  Gemma transformer attention mask is a little complicated, as the text
  uses causal attention, while the images use bidirectional attention.

  Examples:

  ```python
  causal_mask =        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  bidirectional_mask = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

  attention_mask = [
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
  ]
  ```

  Args:
    causal_mask: The causal mask (to mask out future and padding tokens).
    bidirectional_mask: The bidirectional mask (location of the soft images
      tokens).

  Returns:
    The attention mask.
  """

  attention_mask = _make_causal_mask(causal_mask)

  # Add the bidirectional mask for images.
  if bidirectional_mask is not None:
    attention_mask = _add_bidirectional_mask(attention_mask, bidirectional_mask)

  return attention_mask


@typechecked
def _make_causal_mask(
    input_mask: Bool['B L'],
) -> Bool['B L L']:
  """Makes a causal attention mask.

  I.e., as in middle diagram of Figure 3 in https://arxiv.org/pdf/1910.10683.

  Args:
    input_mask: Input mask for the input. True for non-padded tokens only, else
      False.

  Returns:
    Attention mask of shape [B, L, L] (where B=batch dim and L=sequence dim).
  """
  if len(input_mask.shape) != 2:
    raise ValueError(
        f'Input mask must be 2D (shape [B, L]), but got {input_mask.shape}.'
    )
  seq_len = input_mask.shape[-1]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
  attn_mask = input_mask[..., None, :]
  attn_mask *= causal_mask[None, ...]
  return attn_mask


@typechecked
def _make_block_mask_indices(
    bidirectional_mask: Bool['B L'],
) -> Int['B L']:
  """Creates block mask identifying segments based on a bidirectional mask.

  Args:
    bidirectional_mask: boolean mask, e.g. [011110011010].

  Returns:
    block mask for segments, e.g. [011110022030].
  """
  # Left pad 0.
  padded_mask = jnp.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0)
  boundary = padded_mask[..., 1:] > padded_mask[..., :-1]
  numbered_boundary = jnp.cumsum(boundary, axis=-1)
  return bidirectional_mask * numbered_boundary


@typechecked
def _add_bidirectional_mask(
    attn_mask: Bool['B L L'],
    bidirectional_mask: Bool['B L'],
) -> Bool['B L L']:
  """Adds bidirectional mask to the attention mask."""
  q_block_indices = _make_block_mask_indices(bidirectional_mask)
  kv_block_indices = q_block_indices
  attn_mask = attn_mask | (
      (kv_block_indices[:, None, :] == q_block_indices[..., None])
      & (q_block_indices[..., None] > 0)
  )
  return attn_mask
