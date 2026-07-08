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

"""Attention-mask and KV-cache utilities for DiffusionGemma.

This module centralizes all masking, positional tracking, and KV-cache
manipulation helpers required for hybrid autoregressive-diffusion models.

Core Architectural Concepts
---------------------------
Gemma Autoregressive Diffusion operates by combining standard causal cache
(AR) prefilling with localized diffusion denoising over discrete sequence chunks
("canvases"). Because Gemma uses a right-padded memory layout, special care must
be taken to ensure:
1. Pad tokens in the prompt are correctly masked out during prefill.
2. Cached KV entries for prompt tokens and previously generated canvases are
   preserved and correctly indexed across multiple denoising passes.
3. Causality is strictly maintained between canvas chunks (canvas `i` can attend
   to the prompt and canvases `j <= i`, but never to future canvases `k > i`).

Lifecycle & Function Roles
--------------------------
* **Prefill Phase (`sft_encode` / `init_ar_state`)**:
  - `build_positions_from_mask`: Computes 0-indexed sequence positions from a
    boolean prompt mask, ensuring pad tokens clamp to the last valid position.
  - `make_causal_prefill_mask`: Builds a causal lower-triangular mask combined
    with the prompt pad mask, right-padded to the full cache width.

* **Encoder functions (`update_ar_state`)**:
  - `make_full_attention_mask`: Pre-computes a static 1-D attention mask
    representing valid prompt tokens, hidden pad slots, and future decode slots.
  - `make_causal_attention_mask_right_pad`: Creates a causal attention mask for
    appending newly sampled canvas tokens into the existing KV cache while
    respecting the active write cursor (`end_index`).

* **Localized Diffusion Denoising (`sft_decode` / Sampler Denoising)**:
  - `create_decoder_attention_mask`: Generates the block-causal attention mask
    for the SFT decoder. It ensures the model attends only to valid prompt
    tokens and canvases up to the active `selected_canvas_idx`, while completely
    hiding future canvases.
  - `set_cache_end_index`: Overrides the internal `end_index` write cursor in
    the Gemma KV cache. This allows the decoder to reuse cached key/value states
    for the prompt and previous canvases without overwriting them during
    iterative diffusion passes.
"""

from typing import Any, Mapping

import jax
import jax.numpy as jnp
from kauldron.ktyping import Bool, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member

################################################################################
# MARK: Prefill helpers
################################################################################


def build_positions_from_mask(mask: Int['*B L']) -> Int['*B L']:
  """Computes 0-indexed positions from a boolean token mask.

  Matches ``gemma.gm.math._pos_utils.build_positions_from_mask``:
  positions = cumsum(mask) - 1, clamped so pad tokens get the
  position of the last real token.

  Used to prefill the KV-cache with encoder tokens.

  Example:
    mask = [1, 1, 1, 0, 0]  →  positions = [0, 1, 2, 2, 2]

  Args:
    mask: Boolean mask, True for real tokens. Shape ``[B, L]``.

  Returns:
    Position indices of shape ``[B, L]``.
  """
  positions = jnp.cumsum(jax.lax.optimization_barrier(mask), axis=-1)
  return positions - (positions >= 1)


def make_causal_prefill_mask(
    token_mask: Int['*B L'],
    cache_length: int,
) -> Bool['*B L CacheLength']:
  """Creates a causal attention mask for right-padded prefill.

  Builds a lower-triangular causal mask, multiplied by ``token_mask``
  to hide pad tokens, and right-padded to ``cache_length``.

  Used to prefill the KV-cache with encoder tokens.

  Examples:

    **No padding** — ``token_mask = [1, 1, 1]``, ``cache_length = 3``::

      [[1, 0, 0],
       [1, 1, 0],
       [1, 1, 1]]

    **Right-padded tokens** — ``token_mask = [1, 1, 0]``,
    ``cache_length = 3``.  Pad columns are zeroed out::

      [[1, 0, 0],
       [1, 1, 0],
       [1, 1, 0]]   ← row for the pad token; column 2 also masked

    **cache_length > seq_len** — ``token_mask = [1, 1]``,
    ``cache_length = 5``.  Extra columns are filled with False::

      [[1, 0, 0, 0, 0],
       [1, 1, 0, 0, 0]]

  Args:
    token_mask: Boolean mask of shape ``[B, L]``.
    cache_length: Total cache width.

  Returns:
    Attention mask of shape ``[B, L, cache_length]``.
  """
  seq_len = token_mask.shape[-1]
  causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  attn_mask = token_mask[:, None, :] & causal[None, :, :]
  pad_width = cache_length - seq_len
  if pad_width > 0:
    attn_mask = jnp.pad(
        attn_mask,
        ((0, 0), (0, 0), (0, pad_width)),
        constant_values=False,
    )
  return attn_mask


################################################################################
# MARK: Encoder helpers
################################################################################


def make_full_attention_mask(
    input_mask: jnp.ndarray,
    cache_length: int,
) -> jnp.ndarray:
  """Pre-computes a 1-D attention mask for the full cache.

  Matches Gemma's ``_prefill._make_full_attention_mask``.

  Layout (right-padded)::

    input_mask     = [1, 1, 1, 0, 0]
    full_attn_mask = [1, 1, 1, 0, 0, 1, 1, 1, ..., 1]
                      ↑ prompt ↑ pad ↑ future decode slots (all True)

  During decoding, ``full_attn_mask`` is sliced by ``used_cache_length``
  to progressively reveal decode slots while permanently hiding pad slots.

  Args:
    input_mask: Boolean mask of shape ``[B, L]``.
    cache_length: Total cache width.

  Returns:
    Mask of shape ``[B, cache_length]``.
  """
  batch_size = input_mask.shape[0]
  prompt_len = input_mask.shape[1]
  pad_width = cache_length - prompt_len

  # Future decode slots are all True.
  decode_mask = jnp.ones((batch_size, pad_width), dtype=jnp.bool_)

  return jnp.concatenate([input_mask, decode_mask], axis=-1)


@typechecked
def make_causal_attention_mask_right_pad(
    batch_size: int,
    canvas_length: int,
    cache_length: int | None,
    num_valid_cache_tokens: Int['B'] | None,
) -> Bool['B SeqLen CacheLength']:
  """Create a causal attention mask for inserting tokens into the cache.

  Args:
    batch_size: The batch size.
    canvas_length: Number of new tokens being inserted.
    cache_length: Total cache size.
    num_valid_cache_tokens: Per-batch number of valid entries in the cache
      before inserting new tokens.  If this is larger than cache_length the
      cache is assumed to be full and the oldest entries have been evicted.

  Returns:
    Attention mask of shape [batch_size, canvas_length, cache_length].
  """

  if cache_length is None:
    causal_mask = jnp.tril(
        jnp.ones((canvas_length, canvas_length), dtype=jnp.bool_)
    )
    return jnp.broadcast_to(
        causal_mask[None, :, :], (batch_size, canvas_length, canvas_length)
    )

  if num_valid_cache_tokens is None:
    raise ValueError(
        'num_valid_cache_tokens must be provided if cache_length is set.'
    )

  valid_entries = jnp.minimum(num_valid_cache_tokens, cache_length)

  # 1. Fill base mask up to the number of valid tokens in the cache.
  mask = jnp.broadcast_to(
      jnp.arange(cache_length)[None, None, :] < valid_entries[:, None, None],
      (batch_size, canvas_length, cache_length),
  )

  # 2. Append a lower triangular matrix at the (wrapped) write positions.
  write_indices = (
      num_valid_cache_tokens[:, None] + jnp.arange(canvas_length)[None, :]
  ) % cache_length

  batch_idx = jnp.arange(batch_size)[:, None, None]
  seq_idx = jnp.arange(canvas_length)[None, :, None]
  write_idx = write_indices[:, None, :]

  causal_mask = jnp.tril(
      jnp.ones((canvas_length, canvas_length), dtype=jnp.bool_)
  )

  mask = mask.at[batch_idx, seq_idx, write_idx].set(causal_mask[None, :, :])

  return mask


################################################################################
# Decoder masks
################################################################################


def create_decoder_attention_mask(
    prompt_mask: jnp.ndarray,
    canvas_mask: jnp.ndarray,
    selected_canvas_idx: jnp.ndarray,
    prompt_len: int,
    total_canvas_len: int,
    canvas_size: int,
    num_queries: int,
) -> jnp.ndarray:
  """Creates the attention mask for the SFT decoder.

  The mask is True for non-pad prompt tokens and non-pad canvas tokens up to
  and including the selected canvas index.

  It is used for both SFT training and sampling in order to apply the decoder.

  Args:
    prompt_mask: Non-pad prompt mask, shape ``[B, PromptLen]``.
    canvas_mask: Valid-canvas mask, shape ``[B, TotalCanvasLen]``.
    selected_canvas_idx: Per-example selected canvas index, shape ``[B]``.
    prompt_len: Fixed padded maximum prompt length.
    total_canvas_len: Total canvas length.
    canvas_size: Number of tokens per canvas.
    num_queries: The number of new positions that will query the KV-cache. E.g.
      for sft training this will be total_canvas_len as we train on all canvases
      in parallel and mask out in the loss. For the start of sampling this will
      be canvas_len at the start since we are only generating a single canvas.

  Returns:
    Attention mask of shape ``[B, num_queries, PromptLen +
    TotalCanvasLen]``.
  """
  batch_size = prompt_mask.shape[0]
  cache_len = prompt_len + total_canvas_len
  kv_positions = jnp.arange(cache_len)

  # Part 1: Attend to prompt region (non-PAD).
  prompt_region = kv_positions < prompt_len
  prompt_pad_mask = jnp.zeros((batch_size, cache_len), dtype=jnp.bool_)
  prompt_pad_mask = prompt_pad_mask.at[:, :prompt_len].set(prompt_mask)
  prompt_attention = prompt_region[None, None, :] & prompt_pad_mask[:, None, :]

  # Part 2: Attend to selected canvas and all prior canvases.
  in_canvas_region = kv_positions >= prompt_len
  kv_canvas_id = (kv_positions - prompt_len) // canvas_size
  # Attend to canvases <= selected_canvas_idx (the selected one + all prior).
  canvas_attention = (
      kv_canvas_id[None, None, :] <= selected_canvas_idx[:, None, None]
  ) & in_canvas_region[None, None, :]

  # Mask out invalid (PAD) canvas tokens in the KV cache.
  canvas_valid_mask = jnp.zeros((batch_size, cache_len), dtype=jnp.bool_)
  canvas_valid_mask = canvas_valid_mask.at[
      :, prompt_len : prompt_len + total_canvas_len
  ].set(canvas_mask)
  canvas_attention = canvas_attention & canvas_valid_mask[:, None, :]

  # Combined mask: prompt OR canvas attention.
  attn_mask = prompt_attention | canvas_attention

  # Explicitly broadcast to [B, num_queries, CacheLen]
  attn_mask = jnp.broadcast_to(attn_mask, (batch_size, num_queries, cache_len))
  return attn_mask


################################################################################
# KV-cache helpers
################################################################################


def set_cache_end_index(
    kv_cache: Mapping[str, Any],
    end_index: Int['*B'],
) -> dict[str, Any]:
  """Overrides ``end_index`` in every layer of the KV cache.

  In Gemma models, ``end_index`` tracks the active write cursor (the number of
  valid tokens currently stored in the cache). When performing localized
  diffusion decoding over a specific canvas chunk, this function overrides
  ``end_index`` to point to the exact boundary before the selected canvas
  (e.g., ``prompt_len + selected_canvas_idx * canvas_size``). This guarantees
  that the decoder reuses previously cached key/value pairs for the prompt and
  earlier canvases without modifying them during iterative denoising steps.

  Args:
    kv_cache: The KV cache dict (layer_name → layer_dict).
    end_index: New end_index value, shape ``[B]``.

  Returns:
    A new cache dict with updated ``end_index`` in each layer.
  """
  return {
      name: {
          **layer,
          'end_index': jnp.broadcast_to(end_index, layer['end_index'].shape),
      }
      for name, layer in kv_cache.items()
  }
