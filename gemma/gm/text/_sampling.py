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

"""Sampling functions."""

from __future__ import annotations
import dataclasses
import jax
import jax.numpy as jnp

@dataclasses.dataclass
class SamplerState:
  """State of the sampler, holding sampling parameters."""
  temperature: float = 1.0
  top_k: int = 1
  top_p: float = 1.0

def sample_from_logits(
    logits: jnp.ndarray,
    state: SamplerState,
    rng: jax.Array,
) -> jnp.ndarray:
    """Samples from the logits using temperature, top-k, and top-p.

    Args:
        logits: The raw output logits from the model.
        state: The sampler state containing temperature, top_k, and top_p.
        rng: JAX random number generator key.

    Returns:
        The sampled token IDs.
    """
    # Use a guard for pure greedy sampling
    if state.temperature == 0.0:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    # Apply temperature
    # Use a maximum to avoid division by zero
    logits = logits / jnp.maximum(state.temperature, 1e-6)

    # Apply top-k filtering
    if state.top_k > 1:
        # Get the top-k logits and their indices
        top_k_logits, top_k_indices = jax.lax.top_k(logits, k=state.top_k)
        # Create a mask of -inf, then scatter the top-k logits back
        # into their original positions.
        mask = jnp.full_like(logits, -jnp.inf)
        mask = mask.at[..., top_k_indices].set(top_k_logits)
        logits = jnp.where(mask > -jnp.inf, logits, -jnp.inf)

    # Apply top-p (nucleus) filtering
    if state.top_p < 1.0:
        # Sort logits to easily find the cumulative probability distribution
        sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

        # Create a mask for tokens to remove
        sorted_indices_to_remove = cumulative_probs > state.top_p
        # Shift the mask to the right to ensure we keep the first token
        # that exceeds the cumulative probability.
        sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = sorted_indices_to_remove.at[..., 0].set(False)

        # Get the original indices of the sorted logits
        indices = jnp.argsort(logits, axis=-1)[..., ::-1]
        # Create a boolean mask of the same shape as logits
        mask = jnp.zeros_like(logits, dtype=jnp.bool_)
        # Scatter the removal mask back to the original logit positions
        jax.lax.scatter(mask, indices, sorted_indices_to_remove, jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(1,), scatter_dims_to_operand_dims=(1,)))

        # Apply the mask, setting filtered logits to -inf
        logits = jnp.where(mask, -jnp.inf, logits)

    # Sample from the final modified logits
    return jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)
