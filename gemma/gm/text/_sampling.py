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

"""Sampling methods."""

import abc
import dataclasses

from gemma.gm.utils import _jax_utils
import jax
import jax.numpy as jnp
from kauldron.typing import Float, Int, PRNGKey, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class SamplingMethod(abc.ABC):
  """Base class for sampling methods."""

  @abc.abstractmethod
  def get_next_tokens(self, logits: Float['*B V'], rng: PRNGKey) -> Int['*B']:
    """Returns the next tokens to generate.

    Args:
      logits: Logits, as returned by the model (i.e. before softmax).
      rng: A random key.

    Returns:
      The next tokens to generate.
    """
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True, kw_only=True)
class Greedy(SamplingMethod):
  """Greedy sampling."""

  @typechecked
  def get_next_tokens(self, logits: Float['*B V'], rng: PRNGKey) -> Int['*B']:
    del rng
    return jnp.argmax(logits, axis=-1)


@dataclasses.dataclass(frozen=True, kw_only=True)
class RandomSampling(SamplingMethod):
  """Simple random sampling."""

  temperature: float = 1.0

  @typechecked
  def get_next_tokens(self, logits: Float['*B V'], rng: PRNGKey) -> Int['*B']:
    return jax.random.categorical(rng, logits / self.temperature, axis=-1)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TopkSampling(SamplingMethod):
  """Top-k sampling."""

  temperature: float = 1.0
  k: int = 1

  @_jax_utils.flatten_unflatten_batch_dim()
  @typechecked
  def get_next_tokens(self, logits: Float['*B V'], rng: PRNGKey) -> Int['*B']:
    batch_size = logits.shape[0]
    topk_values, topk_indices = jax.lax.top_k(logits, self.k)
    sampled_topk_indices = jax.random.categorical(
        rng, topk_values / self.temperature, axis=-1
    )
    batch_indices = jnp.arange(batch_size)
    return topk_indices[batch_indices, sampled_topk_indices]


@dataclasses.dataclass(frozen=True, kw_only=True)
class NucleusSampling(SamplingMethod):
  """Nucleus (top-p) sampling."""

  temperature: float = 1.0
  p: float = 0.9

  @_jax_utils.flatten_unflatten_batch_dim()
  @typechecked
  def get_next_tokens(self, logits: Float['*B V'], rng: PRNGKey) -> Int['*B']:
    batch_size, vocab_size = logits.shape
    
    # Apply temperature scaling
    scaled_logits = logits / self.temperature
    
    # Convert logits to probabilities
    probs = jax.nn.softmax(scaled_logits, axis=-1)
    
    # Create indices array with same batch dimensions as probs
    indices = jnp.broadcast_to(jnp.arange(vocab_size), (batch_size, vocab_size))
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = jax.lax.sort_key_val(
        -probs, indices, dimension=-1
    )
    sorted_probs = -sorted_probs
    
    # Compute cumulative probabilities
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
    
    # Create mask for tokens within nucleus (cumulative prob <= p)
    nucleus_mask = cumulative_probs <= self.p
    
    # Ensure at least one token is always included (the most probable one)
    nucleus_mask = nucleus_mask.at[:, 0].set(True)
    
    # Zero out probabilities outside the nucleus
    filtered_probs = jnp.where(nucleus_mask, sorted_probs, 0.0)
    
    # Renormalize the filtered probabilities
    filtered_probs = filtered_probs / jnp.sum(filtered_probs, axis=-1, keepdims=True)
    
    # Sample from the filtered distribution
    sampled_sorted_indices = jax.random.categorical(rng, jnp.log(filtered_probs), axis=-1)
    
    # Map back to original vocabulary indices
    batch_indices = jnp.arange(batch_size)
    return sorted_indices[batch_indices, sampled_sorted_indices]
