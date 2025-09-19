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

"""Wrapper around the cache for easier resize."""

from __future__ import annotations

import dataclasses

from etils import epy
import flax
from gemma.gm.nn import _config
import jax.numpy as jnp
from kauldron.typing import Bool, Int  # pylint: disable=g-multiple-import

_Slice = slice | int
_GetItem = _Slice | tuple[_Slice, ...]


@flax.struct.dataclass
class Cache:
  """Wrapper around the cache to support easy slicing.

  Rational: During prefill, the model expects the cache to be of the same size
  as the prompt length. So we slice the cache to match the prompt length.
  During sampling, the full cache is passed and updated in place.
  """
  cache: _config.Cache

  # Getters

  @property
  def total_cache_length(self) -> int:
    layer_data = next(iter(self.cache.values()))
    _, l, _, _ = layer_data['k'].shape
    return l

  def __getitem__(self, key: _GetItem) -> Cache:
    """Get a slice of the cache.

    Used in prefill stage where the cache size needs to match the prompt length.

    Args:
      key: The key to slice the cache. Should be `cache[batch_slice, seq_slice]`

    Returns:
      A new cache with the slice applied.
    """
    if not isinstance(key, tuple) or len(key) != 2:
      raise ValueError(
          f'Unsupported slicing: {key!r}. Should be `(batch, seq_length)`.'
      )
    return Cache(cache=_map_cache_layer(self.cache, _slice_cache, key=key))

  # Setters

  @property
  def at(self) -> _CacheProxyAt:
    """To supports `cache = cache.at[:, :].set_kv(new_cache`."""
    return _CacheProxyAt(self)

  @property
  def end_index(self) -> Int['']:
    """End index of the cache."""
    layer_data = next(iter(self.cache.values()))
    return layer_data['end_index'][0]

  def set_end_index(self, value: int) -> Cache:
    """Set the end index of the cache."""
    return Cache(
        cache=_map_cache_layer(self.cache, _set_end_index, value=value)
    )

  @property
  def is_full(self) -> Bool['']:
    """Returns whether the cache is full."""
    # Maybe will lose the last token.
    return self.end_index >= self.total_cache_length - 1


@dataclasses.dataclass(frozen=True)
class _CacheProxyAt:
  """Small wrapper to support `cache.at[].set()`."""

  cache: Cache

  def __getitem__(self, key: _GetItem):
    if not isinstance(key, tuple) or len(key) != 2:
      raise ValueError(
          f'Unsupported slicing: {key!r}. Should be `(batch, seq_length)`.'
      )
    return CacheProxyAtMutable(cache=self.cache, key=key)


@dataclasses.dataclass(frozen=True)
class CacheProxyAtMutable:
  """Small wrapper to support `cache.at[].set()`."""

  cache: Cache
  key: _GetItem

  def set_kv(self, new_cache: Cache) -> Cache:
    """Only set the kv cache values. But NOT the `end_index`."""
    return Cache(
        cache=_map2_cache_layer(
            _set_cache, self.cache.cache, new_cache.cache, key=self.key
        )
    )


# TODO(epot): Could have the layers implement a cache protocol instead, to
# supports arbitrary layers types.


def _map2_cache_layer(fn, cache0, cache1, **kwargs):
  """Apply a function to each layer of the cache."""
  new_cache = {}
  for k, (layer_data0, layer_data1) in epy.zip_dict(cache0, cache1):
    new_cache[k] = fn(dict(layer_data0), dict(layer_data1), **kwargs)
  return new_cache


def _map_cache_layer(cache, fn, **kwargs):
  """Apply a function to each layer of the cache."""
  new_cache = {}
  for k, layer_data in cache.items():
    new_cache[k] = fn(dict(layer_data), **kwargs)
  return new_cache


def _slice_cache(layer_data, *, key: _GetItem):
  layer_data['k'] = layer_data['k'][*key, :, :]
  layer_data['v'] = layer_data['v'][*key, :, :]
  layer_data['positions'] = layer_data['positions'][*key]
  return layer_data


def _set_cache(layer_data0, layer_data1, *, key):
  layer_data0['k'] = layer_data0['k'].at[*key, :, :].set(layer_data1['k'])
  layer_data0['v'] = layer_data0['v'].at[*key, :, :].set(layer_data1['v'])
  layer_data0['positions'] = (
      layer_data0['positions'].at[*key].set(layer_data1['positions'])
  )
  return layer_data0


def _set_end_index(layer_data, *, value: int):
  layer_data['end_index'] = jnp.full_like(
      layer_data['end_index'], fill_value=value
  )
  return layer_data
