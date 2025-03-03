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
