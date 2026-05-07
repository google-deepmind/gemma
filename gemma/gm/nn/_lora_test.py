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

"""Tests for gm.nn.LoRA with various Einsum implementations."""

from flax import linen as nn
from gemma import peft
from gemma.gm.ckpts import _checkpoint
from gemma.gm.nn import _lora
from gemma.gm.nn.gemma3n import _layers as _gemma3n_layers
from gemma.gm.nn.gemma4 import _layers as _gemma4_layers
import jax
import jax.numpy as jnp
import numpy as np


class _ModelWithGemma4Einsum(nn.Module):
  """Test model using Gemma4 Einsum modules."""

  @nn.compact
  def __call__(self, x):
    y = _gemma4_layers.Einsum(shape=(4, 3))('bi,io->bo', x)
    return y


class _ModelWithClippedEinsum(nn.Module):
  """Test model using Gemma4 ClippedEinsum modules."""

  @nn.compact
  def __call__(self, x):
    y = _gemma4_layers.ClippedEinsum(shape=(4, 3))('bi,io->bo', x)
    return y


class _ModelWithGemma3nEinsum(nn.Module):
  """Test model using Gemma3n Einsum modules."""

  @nn.compact
  def __call__(self, x):
    y = _gemma3n_layers.Einsum(shape=(4, 3))('bi,io->bo', x)
    return y


def _make_replace_fn(rank=2, dtype=jnp.bfloat16):
  """Returns a LoRA replacement function for use with ModuleInterceptor."""
  return lambda m: _lora._replace_by_lora(
      m, rank=rank, dtype=dtype, verbose=False
  )


def _init_with_lora(model, input_shape=(1, 4)):
  """Initialize model with LoRA and return (params, lora_params)."""
  with peft.ModuleInterceptor(_make_replace_fn()):
    params = model.init(jax.random.key(0), jnp.zeros(input_shape))['params']
  _, lora_params = peft.split_params(params)
  return params, lora_params


def test_lora_gemma4_einsum():
  """LoRA wraps Gemma4 Einsum and produces lora params."""
  _, lora_params = _init_with_lora(_ModelWithGemma4Einsum())
  leaves = jax.tree.leaves(lora_params)
  assert leaves, 'Expected LoRA params for Gemma4 Einsum'


def test_lora_gemma4_clipped_einsum():
  """LoRA wraps Gemma4 ClippedEinsum and produces lora params."""
  _, lora_params = _init_with_lora(_ModelWithClippedEinsum())
  leaves = jax.tree.leaves(lora_params)
  assert leaves, 'Expected LoRA params for Gemma4 ClippedEinsum'


def test_lora_gemma3n_einsum():
  """LoRA wraps Gemma3n Einsum and produces lora params."""
  _, lora_params = _init_with_lora(_ModelWithGemma3nEinsum())
  leaves = jax.tree.leaves(lora_params)
  assert leaves, 'Expected LoRA params for Gemma3n Einsum'



def test_lora_params_have_a_and_b():
  """LoRA params contain 'a' and 'b' matrices."""
  params, _ = _init_with_lora(_ModelWithGemma4Einsum())
  # The Einsum_0 should have a '_LoRAEinsum_0' sub-module with 'lora/a' and
  # 'lora/b'. The wrapper doesn't use nn.share_scope, so the LoRA adapter
  # lives in a nested sub-dict.
  einsum_params = params['Einsum_0']
  assert '_LoRAEinsum_0' in einsum_params, (
      f'Missing _LoRAEinsum_0 key in {einsum_params.keys()}'
  )
  lora_sub = einsum_params['_LoRAEinsum_0']
  assert 'lora' in lora_sub, f'Missing lora key in {lora_sub.keys()}'
  assert 'a' in lora_sub['lora'], 'Missing lora/a matrix'
  assert 'b' in lora_sub['lora'], 'Missing lora/b matrix'


# ---------------------------------------------------------------------------
# Checkpoint tree reconciliation tests
# ---------------------------------------------------------------------------


def test_needs_reconciliation_false_for_matching_trees():
  """Gemma3-like tree — structures match, no reconciliation needed."""
  params = {'layer': {'attn': {'w': np.zeros(2)}, 'mlp': {'w': np.zeros(3)}}}
  metadata = {'layer': {'attn': {'w': None}, 'mlp': {'w': None}}}
  assert not _checkpoint._needs_reconciliation(params, metadata)


def test_needs_reconciliation_true_for_empty_stubs():
  """LoRA stubs: empty {} dicts in model tree, absent from checkpoint."""
  params = {'layer': {'attn': {'w': np.zeros(2)}, '_LoRAEinsum_0': {}}}
  metadata = {'layer': {'attn': {'w': None}}}
  assert _checkpoint._needs_reconciliation(params, metadata)


def test_needs_reconciliation_true_for_format_mismatch():
  """Gemma4 share_scope: model has ArrayImpl, checkpoint has {'w': ...}."""
  params = {'mlp': {'gating_einsum': np.zeros(4)}}
  metadata = {'mlp': {'gating_einsum': {'w': None}}}
  assert _checkpoint._needs_reconciliation(params, metadata)


def test_needs_reconciliation_false_for_non_dict_leaves():
  """Both leaves are non-dicts — no mismatch."""
  params = {'a': np.zeros(2)}
  metadata = {'a': None}
  assert not _checkpoint._needs_reconciliation(params, metadata)


def test_needs_reconciliation_nested_detection():
  """Mismatch buried deep in the tree is still detected."""
  params = {
      'layer_0': {
          'attn': {'w': np.zeros(2)},
          'mlp': {'gating_einsum': np.zeros(3)},
      }
  }
  metadata = {
      'layer_0': {
          'attn': {'w': None},
          'mlp': {'gating_einsum': {'w': None}},
      }
  }
  assert _checkpoint._needs_reconciliation(params, metadata)


def test_reconcile_drops_empty_stubs():
  """Empty {} stubs from LoRA wrappers are removed."""
  params = {
      'layer': {
          'attn': {'w': 1},
          '_LoRAEinsum_0': {},
          '_LoRAEinsum_gating_einsum': {},
      }
  }
  metadata = {'layer': {'attn': {'w': None}}}
  result = _checkpoint._reconcile_tree(params, metadata)

  assert result == {'layer': {'attn': {'w': 1}}}
  assert '_LoRAEinsum_0' not in result.get('layer', {})


def test_reconcile_wraps_leaf_to_dict():
  """ArrayImpl leaf is wrapped to match checkpoint {'w': ...} format."""
  arr = np.zeros(4)
  params = {'mlp': {'gating_einsum': arr, 'linear': arr}}
  metadata = {'mlp': {'gating_einsum': {'w': None}, 'linear': {'w': None}}}
  result = _checkpoint._reconcile_tree(params, metadata)

  assert list(result['mlp']['gating_einsum'].keys()) == ['w']
  assert result['mlp']['gating_einsum']['w'] is arr
  assert list(result['mlp']['linear'].keys()) == ['w']
  assert result['mlp']['linear']['w'] is arr


def test_reconcile_passthrough_matching():
  """No changes when trees already match (Gemma3 case)."""
  params = {'a': {'b': 1, 'c': 2}}
  metadata = {'a': {'b': None, 'c': None}}
  result = _checkpoint._reconcile_tree(params, metadata)

  assert result == {'a': {'b': 1, 'c': 2}}


def test_reconcile_full_gemma4_like_tree():
  """End-to-end test with a Gemma4-like layer structure."""
  arr = np.zeros(2)
  params = {
      'layer_0': {
          'attn': {
              'q_einsum': {'w': arr},
              'kv_einsum': {'w': arr},
              'attn_vec_einsum': {'w': arr},
              '_LoRAEinsum_0': {},
          },
          'mlp': {
              'gating_einsum': arr,
              'linear': arr,
              '_LoRAEinsum_gating_einsum': {},
              '_LoRAEinsum_linear': {},
          },
          'per_layer_input_gate': {
              'w': arr,
              '_LoRAEinsum_0': {},
          },
      },
      'embedder': {'input_embedding': arr},
  }
  metadata = {
      'layer_0': {
          'attn': {
              'q_einsum': {'w': None},
              'kv_einsum': {'w': None},
              'attn_vec_einsum': {'w': None},
          },
          'mlp': {
              'gating_einsum': {'w': None},
              'linear': {'w': None},
          },
          'per_layer_input_gate': {'w': None},
      },
      'embedder': {'input_embedding': None},
  }
  result = _checkpoint._reconcile_tree(params, metadata)

  # LoRA stubs removed
  assert '_LoRAEinsum_0' not in result['layer_0']['attn']
  assert '_LoRAEinsum_gating_einsum' not in result['layer_0']['mlp']
  assert '_LoRAEinsum_linear' not in result['layer_0']['mlp']
  assert '_LoRAEinsum_0' not in result['layer_0']['per_layer_input_gate']

  # MLP leaves wrapped to dict format
  assert result['layer_0']['mlp']['gating_einsum'] == {'w': arr}
  assert result['layer_0']['mlp']['linear'] == {'w': arr}

  # Normal params preserved
  assert result['layer_0']['attn']['q_einsum'] == {'w': arr}
  assert result['embedder'] == {'input_embedding': arr}

