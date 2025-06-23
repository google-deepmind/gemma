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

"""Flax linen LoRA modules."""

from collections.abc import Sequence
import dataclasses

from flax import linen as nn
from flax.typing import Array  # pylint: disable=g-importing-member
from gemma.peft import _einsum_utils
import jax
import jax.numpy as jnp
import numpy as np


class LoRADenseAdapter(nn.Module):
  """LoRA module.

  This module only do the x @ A @ B computation.
  Use `LoRADense` to wrap a `nn.Dense` layer.
  """

  _: dataclasses.KW_ONLY

  rank: int
  features: int  # Output dimension.

  dtype: jnp.dtype = jnp.float_
  a_init: nn.initializers.Initializer = nn.initializers.kaiming_uniform()
  b_init: nn.initializers.Initializer = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    a = self.param(  # pytype: disable=wrong-keyword-args
        'a', self.a_init, (inputs.shape[-1], self.rank), dtype=self.dtype
    )
    b = self.param(  # pytype: disable=wrong-keyword-args
        'b', self.b_init, (self.rank, self.features), dtype=self.dtype
    )
    return inputs @ a @ b


class LoRADense(nn.Module):
  """Wrapper around `nn.Dense` which adds a LoRA adapter."""

  _: dataclasses.KW_ONLY

  rank: int
  wrapped: nn.Dense

  dtype: jnp.dtype = jnp.float_
  a_init: nn.initializers.Initializer = nn.initializers.kaiming_uniform()
  b_init: nn.initializers.Initializer = nn.initializers.zeros_init()

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': {'wrapped': params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    # TODO(epot): Fix infinite recursion when `repr(self.wrapped)` inside
    # interceptor.
    adapter = LoRADenseAdapter(
        name='lora',
        rank=self.rank,
        features=self.wrapped.features,
        dtype=self.dtype,
        a_init=self.a_init,
        b_init=self.b_init,
    )
    return self.wrapped(inputs) + adapter(inputs)


class LoRAEinsumAdapter(nn.Module):
  """LoRA einsum module.

  This module only do the x @ A @ B computation.
  Use `LoRAEinsum` to wrap a `nn.Einsum` layer.

  Attributes:
    rank: The rank of the LoRA decomposition.
    einsum_str: The einsum string of the original einsum op. Should be
      `inputs,weights->outputs` (this will be internally rewritten as
      `inputs,a,b->outputs`)
    shape: The shape of the original weights before the low-rank adaptation.
      Should match the `weights` shape from the `einsum_str`.
    dtype: The dtype to use for the LoRA weights.
    a_init: The initializer for the A matrix.
    b_init: The initializer for the B matrix.
  """

  _: dataclasses.KW_ONLY

  rank: int
  einsum_str: str
  shape: Sequence[int]

  dtype: jnp.dtype = jnp.float_
  a_init: nn.initializers.Initializer = nn.initializers.kaiming_uniform()
  b_init: nn.initializers.Initializer = nn.initializers.zeros_init()

  def setup(self):
    # Get the einsum decomposition given the original einsum op.
    # e.g. `BTNH,NHD->BTD` becomes `BTNH,NHr,rD->BTD`
    out = _einsum_utils.get_lora_einsum_str_and_shapes(
        einsum_str=self.einsum_str,
        weights_shape=self.shape,
        rank=self.rank,
    )
    (lora_einsum_str, a_shape, b_shape) = out

    self._lora_einsum_str = lora_einsum_str
    self._a = self.param('a', self.a_init, a_shape, dtype=self.dtype)  # pytype: disable=wrong-keyword-args
    self._b = self.param('b', self.b_init, b_shape, dtype=self.dtype)  # pytype: disable=wrong-keyword-args

  def __call__(self, inputs: Array) -> Array:
    return jnp.einsum(self._lora_einsum_str, inputs, self._a, self._b)


# Flax linen has a `nn.Einsum`, but it seems users prefer
# implementing their own einsum module. This class can still serve as a
# reference implementation.
class LoRAEinsum(nn.Module):
  """Wrapper around `nn.Einsum` which adds a LoRA adapter."""

  _: dataclasses.KW_ONLY

  rank: int
  wrapped: nn.Einsum

  dtype: jnp.dtype = jnp.float_
  a_init: nn.initializers.Initializer = nn.initializers.kaiming_uniform()
  b_init: nn.initializers.Initializer = nn.initializers.zeros_init()

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': {'wrapped': params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, inputs: Array, einsum_str: str | None = None) -> Array:
    # Warning: Calling multiple times with different `einsum_str` will
    # fail as the decomposition would not be the same.
    einsum_str = nn.merge_param(
        'einsum_str', self.wrapped.einsum_str, einsum_str
    )

    adapter = LoRAEinsumAdapter(
        name='lora',
        rank=self.rank,
        einsum_str=einsum_str,
        shape=self.wrapped.shape,
        dtype=self.dtype,
        a_init=self.a_init,
        b_init=self.b_init,
    )
    return self.wrapped(inputs) + adapter(inputs)


class LoRADenseGeneralAdapter(nn.Module):
  """LoRA general dense module.

  This module only does the x @ A @ B computation.
  Use `LoRAGeneralDense` to wrap a `nn.Einsum` layer.

  Attributes:
    rank: The rank of the LoRA decomposition.
    features: The number of output features.
    axis: int or tuple with axes to apply the transformation on.
    batch_dims: tuple with batch axes.
    dtype: The dtype to use for the LoRA weights.
    a_init: The initializer for the A matrix.
    b_init: The initializer for the B matrix.
  """

  _: dataclasses.KW_ONLY

  rank: int
  features: int | Sequence[int]
  axis: int | Sequence[int]
  batch_dims: Sequence[int]

  dtype: jnp.dtype = jnp.float_
  a_init: nn.initializers.Initializer = nn.initializers.kaiming_uniform()
  b_init: nn.initializers.Initializer = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Mostly copied from `flax.nn.DenseGeneral`."""

    # Normalize inputs
    batch_dims = nn.linear._canonicalize_tuple(self.batch_dims)
    if batch_dims:
      max_dim = np.max(batch_dims)
      if set(batch_dims) != set(range(max_dim + 1)):
        raise ValueError(
            'batch_dims %s must be consecutive leading '
            'dimensions starting from 0.'
            % str(batch_dims)
        )

    n_dim = inputs.ndim
    batch_dims = nn.linear._normalize_axes(batch_dims, n_dim)
    features = nn.linear._canonicalize_tuple(self.features)
    axis = nn.linear._normalize_axes(
        nn.linear._canonicalize_tuple(self.axis),
        n_dim,
    )

    # Create LoRA params
    batch_shape = tuple(inputs.shape[i] for i in batch_dims)
    a_shape = batch_shape + tuple(inputs.shape[i] for i in axis) + (self.rank,)
    a = self.param('a', self.a_init, a_shape, self.dtype)
    b_shape = (*batch_shape, self.rank, *features)
    b = self.param('b', self.b_init, b_shape, self.dtype)

    # Contract across given axes.
    n_batch_dims, n_axis = len(batch_dims), len(axis)
    batch_ind = tuple(range(n_batch_dims))
    contract_ind = tuple(range(n_batch_dims, n_batch_dims + n_axis))
    inputs = nn.dtypes.promote_dtype(inputs, dtype=self.dtype)[0]
    # low_rank = x @ A
    low_rank_dot = ((axis, contract_ind), (batch_dims, batch_ind))
    low_rank = jax.lax.dot_general(inputs, a, low_rank_dot)
    # out = low_rank @ B
    low_rank_cind = [n_dim - n_axis]
    b_cind = [n_batch_dims]
    out_dot = ((low_rank_cind, b_cind), (batch_dims, batch_ind))
    return jax.lax.dot_general(low_rank, b, out_dot)


class LoRADenseGeneral(nn.Module):
  """Wrapper around `nn.DenseGeneral` which adds a LoRA adapter."""

  _: dataclasses.KW_ONLY

  rank: int
  wrapped: nn.DenseGeneral

  dtype: jnp.dtype = jnp.float_
  a_init: nn.initializers.Initializer = nn.initializers.kaiming_uniform()
  b_init: nn.initializers.Initializer = nn.initializers.zeros_init()

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': {'wrapped': params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    adapter = LoRADenseGeneralAdapter(
        name='lora',
        rank=self.rank,
        features=self.wrapped.features,
        axis=self.wrapped.axis,
        batch_dims=self.wrapped.batch_dims,
        dtype=self.dtype,
        a_init=self.a_init,
        b_init=self.b_init,
    )
    return self.wrapped(inputs) + adapter(inputs)
