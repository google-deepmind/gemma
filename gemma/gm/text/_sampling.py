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

"""Sampling methods."""

import abc
import dataclasses

from etils import enp
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

  @typechecked
  def get_next_tokens(self, logits: Float['*B V'], rng: PRNGKey) -> Int['*B']:
    logits, batch_shape = enp.flatten(logits, '... V')

    batch_size = logits.shape[0]
    topk_values, topk_indices = jax.lax.top_k(logits, self.k)
    sampled_topk_indices = jax.random.categorical(
        rng, topk_values / self.temperature, axis=-1
    )
    batch_indices = jnp.arange(batch_size)
    topk_indices = topk_indices[batch_indices, sampled_topk_indices]
    return enp.unflatten(topk_indices, batch_shape, '...')


@dataclasses.dataclass(frozen=True, kw_only=True)
class TopPSampling(SamplingMethod):
  """Top-p (Nucleus) Sampling."""

  p: float = 0.9
  temperature: float = 1.0

  @typechecked
  def get_next_tokens(self, logits: Float['... V'], rng: PRNGKey) -> Int['...']:
    # temperature scaling
    logits = logits / self.temperature

    if self.p < 1.0:
      sorted_logits = jnp.sort(logits, axis=-1, descending=True)

      cumulative_probs = jnp.cumsum(
          jax.nn.softmax(sorted_logits, axis=-1), axis=-1
      )

      # get the index of the first token with cumulative probability >= p.
      cutoff_index = jnp.sum(cumulative_probs < self.p, axis=-1, keepdims=True)

      # get the logit value where the cutoff is.
      cutoff_logit = jnp.take_along_axis(sorted_logits, cutoff_index, axis=-1)

      # select logit values that are smaller than the cutoff logit.
      logits = jnp.where(
          logits < cutoff_logit,
          jnp.finfo(logits.dtype).min,
          logits,
      )

    return jax.random.categorical(rng, logits, axis=-1)

