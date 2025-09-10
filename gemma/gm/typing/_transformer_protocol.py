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

"""Minimal protocol for a transformer model to be used with `gm.text.Sampler`."""

import abc
from typing import Any, ClassVar, Protocol
import flax
from flax.typing import FrozenVariableDict, VariableDict  # pylint: disable=g-multiple-import,g-importing-member
from gemma.gm.nn import _config
from gemma.gm.utils import _types
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Bool, Float, Int, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class TransformerConfig(Protocol):
  input_config: _types.InputConfig


@flax.struct.dataclass
class Output(Protocol):
  """Output of the Gemma model.

  Attributes:
    logits: Predicted logits of the model.
    cache: Updated cache if the input cache is not None, None elsewhere.
    hidden_states: The hidden states of the model.
  """

  # When `return_last_only`, `logits` is `*B V`
  logits: Float['*B L V'] | Float['*B V']
  cache: _config.Cache | None
  hidden_states: Float['*B L D'] | Float['*B D'] | None


class ModelInfo(Protocol):
  """Model information.

  Used to auto-load the model tokenizer and params.
  """

  tokenizer_version: int | str | None = None
  default_ckpt: str | None = None


class TransformerProtocol(Protocol):
  """Protocol for a transformer model to be used with a Sampler.

  A model passed to a `Sampler` must implement `apply` and `init_cache`.
  """

  config: TransformerConfig
  INFO: ClassVar[ModelInfo]

  @typechecked
  @abc.abstractmethod
  def apply(
      self,
      variables: VariableDict,
      tokens: Int['*B L'],
      *,
      images: UInt8['*B N H W C'] | UInt8['*B H W C'] | None = None,
      cache: _config.Cache | None = None,
      positions: Int['*B L_with_mm'] | None = None,
      attention_mask: Bool['*B L_with_mm cache_length'] | None = None,
  ) -> Any | tuple[Any, FrozenVariableDict | dict[str, Any]]:
    """Applies a module method to variables and returns output and modified variables."""
    ...

  @abc.abstractmethod
  def init_cache(
      self,
      *,
      batch_size: int,
      dtype: jnp.dtype[Any],
      cache_length: int,
      sharding: kd.sharding.ShardingTree | None = None,
  ) -> _config.Cache:
    """Initializes the KV cache for efficient generation."""
    ...
