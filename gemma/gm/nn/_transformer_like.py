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
from collections.abc import Callable
from typing import Any, ClassVar, Protocol

import flax
from flax.core.scope import CollectionFilter, DenyList  # pylint: disable=g-multiple-import,g-importing-member
from flax.typing import FrozenVariableDict, PRNGKey, RNGSequences, VariableDict  # pylint: disable=g-multiple-import,g-importing-member
from gemma.gm.nn import _config
from gemma.gm.utils import _types
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Bool, Float, Int, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class TransformerConfig(Protocol):
  """Configuration protocol for a transformer model.

  Attributes:
    input_config: Configuration for the model's input.
    num_embed: Vocabulary size.
  """

  input_config: _types.InputConfig
  num_embed: int

  def init_cache(
      self,
      *,
      batch_size: int,
      dtype: jnp.dtype[Any],
      cache_length: int,
      sharding: kd.sharding.ShardingTree | None = None,
  ) -> _config.Cache:
    ...


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


class TransformerLike(Protocol):
  """Protocol for a transformer model to be used with a Sampler.

  A model passed to a `Sampler` must implement `apply` and `init_cache`.
  """

  config: TransformerConfig
  INFO: ClassVar[ModelInfo]

  def init(
      self,
      rngs: PRNGKey | RNGSequences,
      *args,
      method: Callable[..., Any] | str | None = None,
      mutable: CollectionFilter = DenyList('intermediates'),
      capture_intermediates: (
          bool | Callable[[flax.linen.Module, str], bool]
      ) = False,
      **kwargs,
  ) -> FrozenVariableDict | dict[str, Any]:
    """Initializes a module method with variables and returns modified variables.

    ``init`` takes as first argument either a single ``PRNGKey``, or a
    dictionary mapping variable collections names to their ``PRNGKeys``, and
    will call ``method`` (which is the module's ``__call__`` function by
    default) passing ``*args`` and ``**kwargs``, and returns
    a dictionary of initialized variables.

    Args:
      rngs: The PRNGKey or dictionary of PRNGKeys.
      *args: Positional arguments to pass to the method.
      method: The module method to initialize. Defaults to `__call__`.
      mutable: A filter for which variable collections are mutable.
      capture_intermediates: Whether to capture intermediate values.
      **kwargs: Keyword arguments to pass to the method.
    """
    ...

  @typechecked
  @abc.abstractmethod
  def __call__(  # pytype: disable=signature-mismatch
      self,
      tokens: Int['*B L'],
      *,
      images: UInt8['*B N H W C'] | UInt8['*B H W C'] | None = None,
      positions: Int['*B L_with_mm'] | None = None,
      cache: _config.Cache | None = None,
      attention_mask: Bool['*B L_with_mm cache_length'] | None = None,
      return_last_only: bool | None = None,
      return_hidden_states: bool | None = None,
  ) -> Output:  # Output['*B']
    ...

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

  @typechecked
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
