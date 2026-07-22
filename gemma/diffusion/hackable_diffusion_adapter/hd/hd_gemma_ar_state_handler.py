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

"""Gemma-specific AR state handler and helpers for autoregressive diffusion."""

import dataclasses
from typing import Any

from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
from hackable_diffusion import hd
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.sampling import ar_diffusion_sampler
import jax.numpy as jnp
from kauldron import kd
from kauldron.ktyping import Bool
from kauldron.ktyping import Int
from kauldron.ktyping import typechecked  # pylint: disable=g-importing-member

################################################################################
# MARK: Constants
################################################################################

PAD_TOKEN = 0
Tokens = Int["*B L"]
Conditioning = hd_typing.Conditioning
DataArray = hd_typing.DataArray
DiffusionStep = hd.sampling.DiffusionStep


################################################################################
# MARK: PropagateSelfConditioningFn
################################################################################


class PropagateSelfConditioningFn(hd.sampling.UpdateConditioningFn):
  """Propagates self-conditioning from the previous step to the current step."""

  def __call__(
      self,
      conditioning: Conditioning,
      step_carry: DiffusionStep,
  ) -> Conditioning:
    """Update conditioning based on the current diffusion step.

    Args:
      conditioning: The current conditioning dict.
      step_carry: The current diffusion step state.

    Returns:
      The updated conditioning dict.
    """
    conditioning["sc_logits"] = step_carry.aux["logits"]
    return conditioning


################################################################################
# MARK: GemmaARStateHandler (right-pad, Gemma-consistent)
################################################################################


@dataclasses.dataclass(kw_only=True)
class GemmaARStateHandler(ar_diffusion_sampler.ARStateHandler):
  """Handles the AR sampler state for Gemma models.

  Implements the AR state lifecycle using Gemma's right-pad convention,
  matching ``gemma.gm.text._prefill.prefill`` as closely as possible:

  ``init_ar_state``:
    1. Extract pre-tokenized prompt tokens and lengths from conditioning.
    2. Build positions via ``cumsum(input_mask) - 1``.
    3. Build causal prefill mask from ``input_mask``.
    4. Prefill the cache — Gemma's forward pass sets ``end_index`` to
       ``max_prompt_len`` for all batch elements.
    5. Do NOT override ``end_index``.
    6. Build the initial canvas positions and attention mask from
       ``end_index`` in the cache.
    7. Pre-compute ``full_attention_mask`` for permanent pad masking.

  ``update_ar_state``:
    1. Truncate the sampled canvas at stop tokens.
    2. Append canvas tokens to the KV cache via a forward pass.
    3. Write canvas into the output buffer.
    4. Advance positions and attention mask for the next canvas.

  ``finalize_ar_state``:
    1. Strip the prompt prefix and return only generated tokens.

  ``create_conditioning_from_state``:
    Extracts the subset of sampler state needed by the diffusion sampler
    (KV cache, positions, attention mask, prompt info).

  ``update_from_context``:
    Refreshes ``gemma_params`` from a Kauldron ``Context``, enabling
    late-binding of model weights (params are not available at config time).

  Attributes:
    gemma_network: The Gemma Flax module wrapper.
    gemma_params: Model parameters (can be set late via
      ``update_from_context``).
    end_tokens: Token IDs that signal generation should stop.
    pad_token: Token ID used for right-padding.
    network_path: Dot-separated path to extract params from a Kauldron context.
  """

  gemma_network: Any
  gemma_params: Any
  end_tokens: tuple[int, ...]
  pad_token: int = PAD_TOKEN
  network_path: str = "gemma_network.gemma_model"

  def init_cache_fn(
      self,
      batch_size: int,
      cache_length: int,
  ):
    """Initializes the Gemma KV cache for the given batch and cache sizes.

    Args:
      batch_size: The batch size.
      cache_length: Total cache capacity in tokens.

    Returns:
      The initialized cache tree.
    """
    output = self.gemma_network.apply(
        {"params": {"gemma_model": self.gemma_params}},
        batch_size=batch_size,
        cache_length=cache_length,
        method=self.gemma_network.init_cache,
    )
    return output

  def encoder_fn(
      self,
      x: Tokens,
      conditioning_embeddings: hd_typing.Conditioning,
  ):
    """Runs the Gemma encoder forward pass on the given tokens.

    Args:
      x: Input tokens of shape [B, S].
      conditioning_embeddings: Dictionary containing cache and attention masks.

    Returns:
      The encoder output containing updated cache and logits.
    """
    return self.gemma_network.apply(
        {"params": {"gemma_model": self.gemma_params}},
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        method=self.gemma_network.encoder_call,
    )

  ############################################################################
  # init_ar_state
  ############################################################################

  def init_ar_state(
      self,
      *,
      batch_size: int,
      conditioning: Conditioning,
      canvas_length: int,
      max_num_canvases: int,
  ) -> ar_diffusion_sampler.SamplerState:
    """Creates the initial sampler state.

    Args:
      batch_size: The batch size.
      conditioning: Initial conditioning dict containing prompt tokens/lengths.
      canvas_length: Number of tokens per AR canvas generation step.
      max_num_canvases: Maximum number of canvases that can be generated.

    Returns:
      The initial AR diffusion sampler state dict.
    """
    ##########################################################################
    # Extract pre-tokenized prompt tokens and lengths from conditioning.
    ##########################################################################
    prompt_tokens = conditioning["prompt_tokens"]
    prompt_lengths = conditioning["prompt_lengths"]
    max_prompt_len = prompt_tokens.shape[1]
    cache_length = max_prompt_len + max_num_canvases * canvas_length
    ##########################################################################
    # Derive batch dimensions and input mask.
    ##########################################################################
    input_mask = jnp.arange(max_prompt_len)[None, :] < prompt_lengths[:, None]

    cache, _, _, _ = hd_gemma_network.prefill_kv_cache_with_encoder(
        tokens=prompt_tokens,
        input_mask=input_mask,
        init_cache_fn=self.init_cache_fn,
        encoder_fn=self.encoder_fn,
        cache_length=cache_length,
    )

    ##########################################################################
    # Pre-compute full_attention_mask for permanent pad masking.
    ##########################################################################
    # full_attention_mask: (B, cache_length) — True for real prompt tokens
    # and all future decode slots; False for right-pad slots.
    full_attention_mask = mask_helpers.make_full_attention_mask(
        input_mask, cache_length=cache_length
    )

    ##########################################################################
    # Build canvas positions and attention mask for the first canvas.
    ##########################################################################
    # Positions: per-element, starting after each prompt's last real token.
    # (Gemma4 end_index is a write cursor = max_prompt_len for all elements,
    #  but RoPE positions must reflect actual prompt lengths.)
    canvas_positions = (
        prompt_lengths[:, None] + jnp.arange(canvas_length)[None, :]
    )

    total_canvas_len = cache_length - max_prompt_len
    canvas_attn_mask = mask_helpers.create_decoder_attention_mask(
        prompt_mask=input_mask,
        canvas_mask=jnp.ones(
            (batch_size, total_canvas_len), dtype=jnp.bool_
        ),  # currently there are no pad tokens in our canvases to be generated
        selected_canvas_idx=jnp.zeros(
            (batch_size,), dtype=jnp.int32
        ),  # we are generating the first (0-th) canvas
        prompt_len=max_prompt_len,
        total_canvas_len=total_canvas_len,
        canvas_size=canvas_length,
        num_queries=canvas_length,
    )  # (B, canvas_length, cache_length)

    ##########################################################################
    # Allocate output buffer.
    ##########################################################################
    all_canvas_tokens = jnp.zeros(
        (batch_size, max_num_canvases * canvas_length), dtype=jnp.int32
    )
    predicted_tokens = jnp.concatenate(
        [prompt_tokens, all_canvas_tokens], axis=1
    )

    ##########################################################################
    # Assemble initial state.
    ##########################################################################
    init_ar_state = {
        "prompt_tokens": prompt_tokens,
        "prompt_lengths": prompt_lengths,
        "prompt_mask": input_mask,
        "predicted_tokens": predicted_tokens,
        "step": max_prompt_len,
        "done": jnp.zeros(shape=(batch_size,), dtype=jnp.bool_),
        "kv_cache": cache,
        "positions": canvas_positions,
        "attention_mask": canvas_attn_mask,
        "full_attention_mask": full_attention_mask,
        "processed_denoising_steps": 0,
        "processed_num_canvases": 0,
    }
    return init_ar_state

  ############################################################################
  # update_ar_state
  ############################################################################

  def update_ar_state(
      self,
      canvas_last_step: DiffusionStep,
      sampler_state: ar_diffusion_sampler.SamplerState,
  ) -> ar_diffusion_sampler.SamplerState:
    """Post-processes a sampled canvas and updates the sampler state.

    Args:
      canvas_last_step: Final diffusion step of the current canvas.
      sampler_state: Current AR diffusion sampler state.

    Returns:
      The updated AR diffusion sampler state.
    """
    ############################################################################
    # Truncate canvas at stop tokens.
    ############################################################################
    canvas = canvas_last_step.xt
    done = sampler_state["done"]
    # Hackable diffusion assumes [B, L, 1]
    canvas = canvas[..., 0]
    canvas_length = canvas.shape[1]
    canvas, batch_has_stop_token = truncate_canvas_at_stop_tokens(
        canvas=canvas,
        end_tokens=self.end_tokens,
        canvas_length=canvas_length,
        pad_token=self.pad_token,
        done=done,
    )
    ############################################################################
    # Update whether we should finish.
    ############################################################################
    done = done | batch_has_stop_token
    sampler_state["done"] = done

    # Update KV-Cache (using current canvas positions for RoPE).
    kv_cache = append_tokens_to_cache(
        gemma_network=self.gemma_network,
        gemma_params=self.gemma_params,
        tokens=canvas,
        cache=sampler_state["kv_cache"],
        positions=sampler_state["positions"],
        full_attention_mask=sampler_state["full_attention_mask"],
    )
    sampler_state["kv_cache"] = kv_cache

    # Write canvas tokens into the predicted_tokens buffer.
    indices = jnp.arange(canvas_length) + sampler_state["step"]
    predicted_tokens = (
        sampler_state["predicted_tokens"].at[:, indices].set(canvas)
    )
    sampler_state["predicted_tokens"] = predicted_tokens

    sampler_state["step"] += canvas_length

    # Update positions for the NEXT canvas: advance by canvas_length.
    sampler_state["positions"] = sampler_state["positions"] + canvas_length

    # Update attention mask for the NEXT canvas using full_attention_mask.
    batch_size = sampler_state["prompt_mask"].shape[0]
    max_prompt_len = sampler_state["prompt_mask"].shape[1]
    cache_length = sampler_state["attention_mask"].shape[2]
    total_canvas_len = cache_length - max_prompt_len
    sampler_state["attention_mask"] = (
        mask_helpers.create_decoder_attention_mask(
            prompt_mask=sampler_state["prompt_mask"],
            canvas_mask=jnp.ones(
                (batch_size, total_canvas_len), dtype=jnp.bool_
            ),
            selected_canvas_idx=(sampler_state["processed_num_canvases"] + 1)
            * jnp.ones((batch_size,), dtype=jnp.int32),
            prompt_len=max_prompt_len,
            total_canvas_len=total_canvas_len,
            canvas_size=canvas_length,
            num_queries=canvas_length,
        )
    )

    sampler_state[
        "processed_denoising_steps"
    ] += canvas_last_step.step_info.step
    sampler_state["processed_num_canvases"] += 1

    return sampler_state

  ##############################################################################
  # finalize_ar_state
  ##############################################################################

  def finalize_ar_state(
      self,
      sampler_state: ar_diffusion_sampler.SamplerState,
  ) -> DataArray:
    """Extracts the generated tokens, excluding the prompt prefix.

    Args:
      sampler_state: The final AR diffusion sampler state.

    Returns:
      An array of generated tokens.
    """
    predicted_tokens = sampler_state["predicted_tokens"]
    prompt_len = sampler_state["prompt_tokens"].shape[1]
    gen_tokens = predicted_tokens[:, prompt_len:]
    return gen_tokens

  ##############################################################################
  # update_from_context
  ##############################################################################

  def update_from_context(
      self,
      context: kd.train.Context,
  ) -> None:
    """Updates the model and params from the context.

    Args:
      context: The Kauldron training/eval context.
    """
    self.gemma_params = kd.kontext.get_by_path(
        context.params, self.network_path
    )

  ##############################################################################
  # create_conditioning_from_state
  ##############################################################################

  def create_conditioning_from_state(
      self,
      sampler_state: ar_diffusion_sampler.SamplerState,
  ) -> Conditioning:
    """Creates the conditioning dict from the sampler state.

    Args:
      sampler_state: The current AR diffusion sampler state.

    Returns:
      A conditioning dictionary containing prompt tokens, lengths, cache, and
      masks.
    """
    return {
        "prompt_tokens": sampler_state["prompt_tokens"],
        "prompt_lengths": sampler_state["prompt_lengths"],
        "kv_cache": sampler_state["kv_cache"],
        "positions": sampler_state["positions"],
        "attention_mask": sampler_state["attention_mask"],
        "full_attention_mask": sampler_state["full_attention_mask"],
    }


################################################################################
# MARK: Auxiliary functions
################################################################################


@typechecked
def truncate_canvas_at_stop_tokens(
    canvas: Tokens,
    *,
    end_tokens: tuple[int, ...],
    canvas_length: int,
    done: Bool["B"],
    pad_token: int = PAD_TOKEN,
) -> tuple[Tokens, Bool["B"]]:
  """Replaces tokens after the first stop token with PAD_TOKEN.

  Args:
    canvas: Input canvas tokens of shape [B, L].
    end_tokens: Tuple of stop token IDs.
    canvas_length: Length of the canvas.
    done: Boolean array indicating which batch elements are already finished.
    pad_token: Token ID used for padding.

  Returns:
    A tuple of (truncated canvas, boolean array indicating new stop tokens).
  """
  end_tokens_arr = jnp.array(end_tokens, dtype=jnp.int32)
  is_stop_token = jnp.isin(canvas, end_tokens_arr)
  batch_has_stop_token = jnp.any(is_stop_token, axis=-1)

  first_stop_idx = jnp.argmax(is_stop_token, axis=-1)

  seq_idx = jnp.arange(canvas_length)[None, :]
  keep_mask = seq_idx <= jnp.where(
      batch_has_stop_token[:, None],
      first_stop_idx[:, None],
      canvas_length,
  )
  keep_mask = keep_mask & ~done[:, None]
  canvas = jnp.where(keep_mask, canvas, pad_token)

  return canvas, batch_has_stop_token


def append_tokens_to_cache(
    gemma_network: hd_gemma_network.WrappedDiffusionGemmaNetwork,
    gemma_params: Any,
    *,
    tokens: Tokens,
    cache,
    positions: Int["B L"],
    full_attention_mask: Bool["B cache_length"],
):
  """Inserts tokens into the cache via a transformer forward pass.

  Uses a causal attention mask so that each token can attend to all valid
  cached tokens and to preceding tokens in the input, but not to future
  tokens.  The ``full_attention_mask`` permanently hides right-pad slots.

  Args:
    gemma_network: The wrapped Gemma diffusion network.
    gemma_params: Model parameters for the Gemma network.
    tokens: Tokens to insert, shaped [batch_size, seq_len].
    cache: The current KV cache.
    positions: RoPE positions for the tokens, shaped [batch_size, seq_len].
    full_attention_mask: Permanent pad mask, shaped [batch_size, cache_length].

  Returns:
    The updated cache with the tokens inserted.
  """
  seq_len = tokens.shape[1]
  cache_layer = list(cache.values())[0]
  cache_length = cache_layer["k"].shape[1]
  # Gemma4: end_index is a write cursor (= number of entries written).
  end_index: Int["B"] = cache_layer["end_index"]
  attention_mask = mask_helpers.make_causal_attention_mask_right_pad(
      batch_size=tokens.shape[0],
      canvas_length=seq_len,
      cache_length=cache_length,
      num_valid_cache_tokens=end_index,
  )
  # Hide right-pad KV slots from prefill.
  attention_mask = attention_mask & full_attention_mask[:, None, :]

  output = gemma_network.apply(
      {"params": {"gemma_model": gemma_params}},
      x=tokens,
      conditioning_embeddings={
          "kv_cache": cache,
          "positions": positions,
          "attention_mask": attention_mask,
      },
      method=gemma_network.encoder_call,
  )

  return output.cache
