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

"""Wrapper for Gemma Diffusion model into hackable diffusion."""

from typing import Any

import flax.linen as nn
from gemma.diffusion import _models as gemma_diffusion
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
from hackable_diffusion.lib import diffusion_network
from hackable_diffusion.lib import hd_typing

import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

Conditioning = hd_typing.Conditioning
DataArray = hd_typing.DataArray
TimeArray = hd_typing.TimeArray
TargetInfo = hd_typing.TargetInfo


DiffusionNetwork = diffusion_network.DiffusionNetwork

DiffusionGemmaModel = gemma_diffusion.DiffusionGemma_26B_A4B


# pytype: disable=bad-return-type
# pytype: disable=signature-mismatch


################################################################################
# MARK: Prefill utilities
################################################################################


def prefill_kv_cache_with_encoder(
    tokens, input_mask, init_cache_fn, encoder_fn, cache_length=None
):
  """Prefills the KV cache with the encoder output.

  Args:
    tokens: Input tokens of shape [B, S].
    input_mask: Boolean mask indicating valid prompt positions.
    init_cache_fn: Function to initialize the KV cache.
    encoder_fn: Function to run the encoder forward pass.
    cache_length: Total allocated cache capacity (defaults to sequence length).

  Returns:
    A tuple of (initialized and prefilled cache, encoder logits, positions,
    attention_mask).
  """
  batch_size, full_seq_len = tokens.shape
  if cache_length is None:
    cache_length = full_seq_len
  ############################################################################
  # Initialize Gemma KV cache.
  ############################################################################
  cache = init_cache_fn(
      batch_size=batch_size,
      cache_length=cache_length,
  )
  ##########################################################################
  # Build prefill positions and mask.
  ##########################################################################
  # Positions: cumsum of input_mask, 0-indexed.
  #   e.g. input_mask=[1,1,1,0,0] → positions=[0,1,2,2,2]
  # 0 mask means it is padded / unused.
  positions = mask_helpers.build_positions_from_mask(input_mask)

  # Prefill attention mask: causal × input_mask, right-padded to cache_length.
  attention_mask = mask_helpers.make_causal_prefill_mask(
      input_mask, cache_length
  )
  ##########################################################################
  # Prefill the cache.
  ##########################################################################
  # Gemma writes max_prompt_len entries for every batch element.
  # end_index becomes max_prompt_len for all (set by the model, NOT by us).
  encoder_out = encoder_fn(
      x=tokens,
      conditioning_embeddings={
          'kv_cache': cache,
          'positions': positions,
          'attention_mask': attention_mask,
      },
  )
  kv_cache = encoder_out.cache
  encoder_logits = encoder_out.logits
  if kv_cache is None:
    raise ValueError('KV cache should not be None after encoder pass')
  return kv_cache, encoder_logits, positions, attention_mask


################################################################################
# WrappedDiffusionGemmaNetwork Wrapper
################################################################################


class WrappedDiffusionGemmaNetwork(nn.Module):
  """Wraps a Diffusion Gemma model as an HD ``DiffusionNetwork``.

  The wrapper handles the hackable diffusion interfaces and Gemma specifics. For
  self conditioning, it uses `call_with_self_conditioning` function from Gemma.

  Attributes:
    gemma_model: The Diffusion Gemma model.
  """

  gemma_model: DiffusionGemmaModel

  @property
  def num_embed(self) -> int:
    return self.gemma_model.config.num_embed

  @nn.compact
  def init_cache(
      self,
      *,
      batch_size: int,
      cache_length: int,
  ) -> Any:
    """Initializes the Gemma KV cache.

    Args:
      batch_size: The batch size.
      cache_length: Total cache capacity in tokens.

    Returns:
      The initialized cache tree.
    """
    return self.gemma_model.init_cache(
        batch_size=batch_size,
        dtype=self.gemma_model.dtype,
        cache_length=cache_length,
    )

  @nn.compact
  def encoder_call(
      self,
      *,
      x: DataArray,
      conditioning_embeddings: dict[str, Any],
  ) -> DataArray:
    """Calls the Gemma encoder.

    Args:
      x: Input tokens or array.
      conditioning_embeddings: Dictionary containing cache, positions, and
        attention_mask.

    Returns:
      The transformer output containing updated cache and logits.
    """
    if len(x.shape) == 3:
      tokens = x[..., 0]
    else:
      tokens = x  # [Batch size, Canvas Length] with values in 0,...,VOCAB - 1
    assert len(tokens.shape) == 2

    cache = conditioning_embeddings.get('kv_cache', None)
    positions = conditioning_embeddings.get('positions', None)
    attention_mask = conditioning_embeddings.get('attention_mask', None)
    return self.gemma_model(
        tokens=tokens,
        cache=cache,
        positions=positions,
        attention_mask=attention_mask,
    )

  @nn.compact
  def __call__(
      self,
      *,
      time: TimeArray,
      xt: DataArray,
      conditioning: Conditioning | None,
      is_training: bool = True,
  ) -> TargetInfo:
    """Runs the diffusion denoiser forward pass.

    Args:
      time: Diffusion time steps.
      xt: Noisy canvas tokens or array.
      conditioning: Dictionary containing cache, positions, attention_mask, and
        sc_logits.
      is_training: Whether operating in training mode.

    Returns:
      A dictionary containing predicted denoising logits.
    """
    del is_training  # Unused.
    assert conditioning is not None

    # token dimension:
    if len(xt.shape) == 3:
      tokens = xt[..., 0]
    else:
      tokens = xt  # [Batch size, Canvas Length] with values in 0,...,VOCAB - 1
    assert len(tokens.shape) == 2
    batch_size, canvas_length = tokens.shape
    vocab_size = self.gemma_model.config.num_embed
    dtype = self.gemma_model.dtype
    ############################################################################
    # Handling of all the conditioning
    ############################################################################
    # The HD pipeline passes the self-conditioning signal as raw logits
    # (shape [B, L, V]) under the key 'sc_logits' in the conditioning dict.
    sc_logits = conditioning.get('sc_logits', None)
    if sc_logits is None:
      sc_logits = jnp.zeros(
          (batch_size, canvas_length, vocab_size), dtype=dtype
      )

    positions = conditioning.get('positions', None)
    kv_cache = conditioning.get('kv_cache', None)
    attention_mask = conditioning.get('attention_mask', None)

    # We keep this call to maintain the param init behavior.
    sc_embeddings = self.gemma_model.embedder.encode_logits(sc_logits)

    transformer_output = self.gemma_model.call_with_self_conditioning(
        tokens=tokens,
        sc_embeddings=sc_embeddings,
        cache=kv_cache,
        positions=positions,
        attention_mask=attention_mask,
    )
    logits = transformer_output.logits
    return {'logits': logits}
