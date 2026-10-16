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

import copy

from flax import linen as nn
from gemma import peft
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_split_params():
  params = {
      'dense': {
          'kernel': 0,
          'bias': 1,
          'lora': {
              'a': 0,
              'b': 1,
          },
      },
      'branch_with_only_lora': {
          'lora': {
              'a': 0,
              'b': 1,
          },
      },
      'other': 0,
      # Nested branches are fully removed from the lora tree.
      'b': {'f': {'a': {}}},
  }

  original, lora = peft.split_params(params)
  assert original == {
      'dense': {
          'kernel': 0,
          'bias': 1,
      },
      'branch_with_only_lora': {},
      'other': 0,
      'b': {'f': {'a': {}}},
  }
  assert lora == {
      'dense': {
          'lora': {
              'a': 0,
              'b': 1,
          },
      },
      'branch_with_only_lora': {
          'lora': {
              'a': 0,
              'b': 1,
          },
      },
  }

  assert peft.merge_params(original, lora) == params


def _dense_to_lora(module):
  if isinstance(module, nn.Dense):
    return peft.LoRADense(rank=2, wrapped=module)
  if isinstance(module, nn.Einsum):
    return peft.LoRAEinsum(rank=2, wrapped=module)
  if isinstance(module, nn.DenseGeneral):
    return peft.LoRADenseGeneral(rank=2, wrapped=module)
  return module


class _FusionModule(nn.Module):

  @nn.compact
  def __call__(self, x):
    y_dense = nn.Dense(3, use_bias=True)(x)
    y_einsum = nn.Einsum(
        shape=(4, 2, 3),
        einsum_str='bi,ijk->bjk',
    )(x)
    y_general = nn.DenseGeneral(features=(2, 3), axis=-1)(x)
    return {
        'dense': y_dense,
        'einsum': y_einsum,
        'general': y_general,
    }


def _with_nonzero_lora_b(params, layer_name, key):
  params = copy.deepcopy(params)
  layer = params['params'][layer_name]
  layer['lora']['b'] = jax.random.normal(key, layer['lora']['b'].shape)
  return params


def test_fuse_params_matches_lora_forward():
  model = _FusionModule()
  x = jax.random.normal(jax.random.key(0), (2, 4))

  with peft.ModuleInterceptor(_dense_to_lora):
    _, params = model.init_with_output(jax.random.key(1), x)

  # Make LoRA non-trivial (b is zero-initialized by default).
  params = _with_nonzero_lora_b(params, 'Dense_0', jax.random.key(2))
  params = _with_nonzero_lora_b(params, 'Einsum_0', jax.random.key(3))
  params = _with_nonzero_lora_b(params, 'DenseGeneral_0', jax.random.key(4))

  with peft.ModuleInterceptor(_dense_to_lora):
    lora_out = model.apply(params, x)

  fused = peft.fuse_params(params)
  # Base model (no LoRA wrappers) with fused weights should match LoRA forward.
  fused_base, _ = peft.split_params(fused)
  fused_out = model.apply(fused_base, x)

  def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  assert_close(fused_out['dense'], lora_out['dense'])
  assert_close(fused_out['einsum'], lora_out['einsum'])
  assert_close(fused_out['general'], lora_out['general'])

  # Adapters remain present so unfuse can reverse the merge.
  assert 'lora' in fused['params']['Dense_0']
  assert fused['params']['Dense_0']['lora']['a'].shape == (4, 2)
  assert fused['params']['Dense_0']['lora']['b'].shape == (2, 3)


def test_fuse_unfuse_roundtrip():
  model = _FusionModule()
  x = jnp.ones((1, 4))
  with peft.ModuleInterceptor(_dense_to_lora):
    params = model.init(jax.random.key(0), x)

  params = _with_nonzero_lora_b(params, 'Dense_0', jax.random.key(5))
  params = _with_nonzero_lora_b(params, 'Einsum_0', jax.random.key(6))
  params = _with_nonzero_lora_b(params, 'DenseGeneral_0', jax.random.key(7))

  fused = peft.fuse_params(params)
  restored = peft.unfuse_params(fused)

  np.testing.assert_allclose(
      restored['params']['Dense_0']['kernel'],
      params['params']['Dense_0']['kernel'],
      rtol=1e-5,
      atol=1e-5,
  )
  np.testing.assert_allclose(
      restored['params']['Einsum_0']['kernel'],
      params['params']['Einsum_0']['kernel'],
      rtol=1e-5,
      atol=1e-5,
  )
  np.testing.assert_allclose(
      restored['params']['DenseGeneral_0']['kernel'],
      params['params']['DenseGeneral_0']['kernel'],
      rtol=1e-5,
      atol=1e-5,
  )


def test_fuse_params_noop_without_lora():
  params = {
      'dense': {
          'kernel': jnp.arange(6.0).reshape(3, 2),
          'bias': jnp.zeros((2,)),
      }
  }
  fused = peft.fuse_params(params)
  np.testing.assert_array_equal(
      fused['dense']['kernel'], params['dense']['kernel']
  )


def test_fuse_params_dense_manual():
  a = jnp.arange(8.0).reshape(4, 2)
  b = jnp.arange(6.0).reshape(2, 3)
  kernel = jnp.ones((4, 3))
  params = {
      'layer': {
          'kernel': kernel,
          'lora': {'a': a, 'b': b},
      }
  }
  fused = peft.fuse_params(params)
  np.testing.assert_allclose(fused['layer']['kernel'], kernel + a @ b)

  # Deployment path: drop adapters after fusion.
  original, _ = peft.split_params(fused)
  assert 'lora' not in original['layer']
  np.testing.assert_allclose(original['layer']['kernel'], kernel + a @ b)


def test_fuse_params_einsum_weight_named_w():
  # Mimic Gemma's custom Einsum param name (`w`) with a layout that needs a
  # transpose after contracting the LoRA factors (SNDH vs D+SNH order).
  a = jax.random.normal(jax.random.key(2), (8, 2))  # D, r
  b = jax.random.normal(jax.random.key(3), (2, 3, 4, 5))  # r, S, N, H
  delta_dsnh = jnp.tensordot(a, b, axes=([-1], [0]))  # D, S, N, H
  weight = jnp.transpose(delta_dsnh, (1, 2, 0, 3))  # S, N, D, H
  params = {
      'qkv': {
          'w': jnp.zeros_like(weight),
          'lora': {'a': a, 'b': b},
      }
  }
  fused = peft.fuse_params(params)
  np.testing.assert_allclose(fused['qkv']['w'], weight, rtol=1e-5, atol=1e-5)


def test_fuse_params_rejects_unsupported_layout():
  params = {
      'layer': {
          'kernel': jnp.ones((4, 3)),
          'lora': {
              # Rank is not leading in b (simulates batched DenseGeneral).
              'a': jnp.ones((2, 4, 2)),
              'b': jnp.ones((2, 2, 3)),
          },
      }
  }
  with pytest.raises(ValueError, match='LoRA delta shape is incompatible'):
    peft.fuse_params(params)
