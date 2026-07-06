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

"""Tests for lora.fuse_lora_params and target_modules."""


from absl.testing import absltest
from absl.testing import parameterized

from unittest import mock

from flax import linen as nn
from gemma.diffusion.hackable_diffusion_adapter.hd import lora
import jax
import jax.numpy as jnp
import numpy as np



# ---------------------------------------------------------------------------
# Helper to build a minimal param tree with one LoRA adapter.
# ---------------------------------------------------------------------------
def _make_params(
    base_path: tuple[str, ...],
    base_shape: tuple[int, ...],
    lora_parent: tuple[str, ...],
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    *,
    base_value: float = 1.0,
    a_value: float = 1.0,
    b_value: float = 1.0,
) -> dict:  # pylint: disable=g-bare-generic
  """Build a nested param dict with one base weight and one LoRA adapter.

  Args:
    base_path: Tuple path for the base weight (e.g. ('attn', 'q_einsum', 'w')).
    base_shape: Shape of the base weight array.
    lora_parent: Tuple path to the LoRA parent (above 'lora/a', 'lora/b').
    a_shape: Shape of the LoRA 'a' matrix.
    b_shape: Shape of the LoRA 'b' matrix.
    base_value: Fill value for the base weight.
    a_value: Fill value for the LoRA 'a' matrix.
    b_value: Fill value for the LoRA 'b' matrix.

  Returns:
    A nested dict representing the param tree.
  """
  import flax.traverse_util  # pylint: disable=g-import-not-at-top

  flat = {}
  flat[base_path] = np.full(base_shape, base_value, dtype=np.float32)
  flat[lora_parent + ('lora', 'a')] = np.full(
      a_shape, a_value, dtype=np.float32
  )
  flat[lora_parent + ('lora', 'b')] = np.full(
      b_shape, b_value, dtype=np.float32
  )
  return flax.traverse_util.unflatten_dict(flat)


class FindBaseWeightKeyTest(parameterized.TestCase):
  """Tests for lora._find_base_weight_key."""

  def test_lora_einsum_named_variant(self):
    """_LoRAEinsum_gating_einsum -> sibling 'gating_einsum'."""
    original_flat = {
        ('mlp', 'gating_einsum'): np.zeros((2, 6144, 1536)),
        ('mlp', 'linear'): np.zeros((6144, 1536)),
    }
    result = lora._find_base_weight_key(
        ('mlp', '_LoRAEinsum_gating_einsum'), original_flat
    )
    self.assertEqual(result, ('mlp', 'gating_einsum'))

  def test_lora_einsum_named_variant_linear(self):
    """_LoRAEinsum_linear -> sibling 'linear'."""
    original_flat = {
        ('mlp', 'gating_einsum'): np.zeros((2, 6144, 1536)),
        ('mlp', 'linear'): np.zeros((6144, 1536)),
    }
    result = lora._find_base_weight_key(
        ('mlp', '_LoRAEinsum_linear'), original_flat
    )
    self.assertEqual(result, ('mlp', 'linear'))

  def test_lora_einsum_0_default(self):
    """_LoRAEinsum_0 -> sibling 'w'."""
    original_flat = {
        ('attn', 'q_einsum', 'w'): np.zeros((8, 1536, 256)),
    }
    result = lora._find_base_weight_key(
        ('attn', 'q_einsum', '_LoRAEinsum_0'), original_flat
    )
    self.assertEqual(result, ('attn', 'q_einsum', 'w'))

  def test_shared_scope_kernel(self):
    """Shared-scope LoRADense: lora_parent + 'kernel'."""
    original_flat = {
        ('dense', 'kernel'): np.zeros((128, 64)),
        ('dense', 'bias'): np.zeros((64,)),
    }
    result = lora._find_base_weight_key(('dense',), original_flat)
    self.assertEqual(result, ('dense', 'kernel'))

  def test_shared_scope_w(self):
    """Shared-scope LoRAEinsum: lora_parent + 'w'."""
    original_flat = {
        ('einsum', 'w'): np.zeros((8, 256)),
    }
    result = lora._find_base_weight_key(('einsum',), original_flat)
    self.assertEqual(result, ('einsum', 'w'))

  def test_bare_array_fallback(self):
    """lora_parent itself is the base weight (bare array, no sub-key)."""
    original_flat = {
        ('layer', 'weight'): np.zeros((10, 20)),
    }
    result = lora._find_base_weight_key(('layer', 'weight'), original_flat)
    self.assertEqual(result, ('layer', 'weight'))

  def test_not_found(self):
    """Returns None when no matching base weight exists."""
    original_flat = {
        ('other', 'param'): np.zeros((10,)),
    }
    result = lora._find_base_weight_key(
        ('layer', '_LoRAEinsum_0'), original_flat
    )
    self.assertIsNone(result)

  def test_lora_einsum_0_prefers_w_over_kernel(self):
    """_LoRAEinsum_0 should find 'w' first (case 1 before case 2)."""
    original_flat = {
        ('attn', 'w'): np.zeros((8,)),
        ('attn', '_LoRAEinsum_0', 'kernel'): np.zeros((8,)),
    }
    result = lora._find_base_weight_key(
        ('attn', '_LoRAEinsum_0'), original_flat
    )
    self.assertEqual(result, ('attn', 'w'))


class FuseLoraParamsTest(parameterized.TestCase):
  """Tests for lora.fuse_lora_params."""

  def test_dense_2d(self):
    """Simple Dense: a=(in, r), b=(r, out), weight=(in, out)."""
    rank = 4
    in_dim, out_dim = 16, 8
    params = _make_params(
        base_path=('dense', 'kernel'),
        base_shape=(in_dim, out_dim),
        lora_parent=('dense',),
        a_shape=(in_dim, rank),
        b_shape=(rank, out_dim),
        base_value=0.0,
        a_value=1.0,
        b_value=1.0,
    )
    fused = lora.fuse_lora_params(params)

    # delta = a @ b = ones(16,4) @ ones(4,8) = full(16,8, fill=4)
    fused_w = fused['dense']['kernel']
    expected_delta = np.full((in_dim, out_dim), rank, dtype=np.float32)
    np.testing.assert_allclose(fused_w, expected_delta)

    # LoRA keys should be gone.
    self.assertNotIn('lora', fused['dense'])

  def test_mlp_linear(self):
    """MLP linear: einsum '...H,HF->...F', weight=(H, D).

    LoRA: a=(H, r), b=(r, D). a @ b = (H, D) matches weight directly.
    _LoRAEinsum_linear -> base at ('mlp', 'linear').

    """
    rank = 4
    d_model, hidden = 16, 32
    params = _make_params(
        base_path=('mlp', 'linear'),
        base_shape=(hidden, d_model),
        lora_parent=('mlp', '_LoRAEinsum_linear'),
        a_shape=(hidden, rank),
        b_shape=(rank, d_model),
        base_value=0.0,
    )
    fused = lora.fuse_lora_params(params)
    fused_w = fused['mlp']['linear']
    self.assertEqual(fused_w.shape, (hidden, d_model))

  def test_q_einsum(self):
    """q_einsum: a=(D, r), b=(r, N, H), weight=(N, D, H).

    _LoRAEinsum_0 under q_einsum -> base at ('attn', 'q_einsum', 'w').

    """
    rank = 2
    d_model, num_heads, head_dim = 12, 3, 4
    params = _make_params(
        base_path=('attn', 'q_einsum', 'w'),
        base_shape=(num_heads, d_model, head_dim),
        lora_parent=('attn', 'q_einsum', '_LoRAEinsum_0'),
        a_shape=(d_model, rank),
        b_shape=(rank, num_heads, head_dim),
        base_value=0.0,
    )
    fused = lora.fuse_lora_params(
        params,
    )
    fused_w = fused['attn']['q_einsum']['w']
    self.assertEqual(fused_w.shape, (num_heads, d_model, head_dim))

  def test_kv_einsum(self):
    """kv_einsum: a=(D, r), b=(r, C, K, H), weight=(C, K, D, H).

    _LoRAEinsum_0 under kv_einsum -> base at ('attn', 'kv_einsum', 'w').
    """
    rank = 2
    d_model, num_kv, head_dim = 12, 1, 4
    c_dim = 2  # key+value
    params = _make_params(
        base_path=('attn', 'kv_einsum', 'w'),
        base_shape=(c_dim, num_kv, d_model, head_dim),
        lora_parent=('attn', 'kv_einsum', '_LoRAEinsum_0'),
        a_shape=(d_model, rank),
        b_shape=(rank, c_dim, num_kv, head_dim),
        base_value=0.0,
    )
    fused = lora.fuse_lora_params(params)
    fused_w = fused['attn']['kv_einsum']['w']
    self.assertEqual(fused_w.shape, (c_dim, num_kv, d_model, head_dim))

  def test_attn_vec_einsum(self):
    """attn_vec_einsum: a=(N, H, r), b=(r, D), weight=(N, H, D).

    _LoRAEinsum_0 -> base at ('attn', 'attn_vec_einsum', 'w').
    """
    rank = 2
    num_heads, head_dim, d_model = 3, 4, 12
    params = _make_params(
        base_path=('attn', 'attn_vec_einsum', 'w'),
        base_shape=(num_heads, head_dim, d_model),
        lora_parent=('attn', 'attn_vec_einsum', '_LoRAEinsum_0'),
        a_shape=(num_heads, head_dim, rank),
        b_shape=(rank, d_model),
        base_value=0.0,
    )
    fused = lora.fuse_lora_params(params)
    fused_w = fused['attn']['attn_vec_einsum']['w']
    self.assertEqual(fused_w.shape, (num_heads, head_dim, d_model))

  def test_gating_einsum(self):
    """gating_einsum: a=(D, r), b=(r, 2, H), weight=(2, H, D).

    _LoRAEinsum_gating_einsum -> base at ('mlp', 'gating_einsum').
    """
    rank = 2
    d_model, hidden = 12, 24
    params = _make_params(
        base_path=('mlp', 'gating_einsum'),
        base_shape=(2, hidden, d_model),
        lora_parent=('mlp', '_LoRAEinsum_gating_einsum'),
        a_shape=(d_model, rank),
        b_shape=(rank, 2, hidden),
        base_value=0.0,
    )
    fused = lora.fuse_lora_params(params)
    fused_w = fused['mlp']['gating_einsum']
    self.assertEqual(fused_w.shape, (2, hidden, d_model))

  def test_fuse_adds_delta_to_base(self):
    """Verify the fused weight = base_weight + delta."""
    rank = 2
    in_dim, out_dim = 6, 4

    base_val = 10.0
    params = _make_params(
        base_path=('dense', 'kernel'),
        base_shape=(in_dim, out_dim),
        lora_parent=('dense',),
        a_shape=(in_dim, rank),
        b_shape=(rank, out_dim),
        base_value=base_val,
        a_value=1.0,
        b_value=1.0,
    )
    fused = lora.fuse_lora_params(params)
    fused_w = fused['dense']['kernel']

    # delta = ones(6,2) @ ones(2,4) = full(6,4, fill=2)
    expected = np.full((in_dim, out_dim), base_val + rank, dtype=np.float32)
    np.testing.assert_allclose(fused_w, expected)

  def test_no_lora_params_returns_original(self):
    """If there are no LoRA params, return the original tree unchanged."""
    import flax.traverse_util  # pylint: disable=g-import-not-at-top

    flat = {
        ('dense', 'kernel'): np.ones((4, 4), dtype=np.float32),
    }
    params = flax.traverse_util.unflatten_dict(flat)
    fused = lora.fuse_lora_params(params)
    np.testing.assert_array_equal(
        fused['dense']['kernel'], params['dense']['kernel']
    )

  def test_multiple_adapters(self):
    """Fuse multiple LoRA adapters in the same param tree."""
    import flax.traverse_util  # pylint: disable=g-import-not-at-top

    rank = 2
    flat = {
        # Adapter 1: Dense with shared-scope.
        ('dense', 'kernel'): np.zeros((8, 4), dtype=np.float32),
        ('dense', 'lora', 'a'): np.ones((8, rank), dtype=np.float32),
        ('dense', 'lora', 'b'): np.ones((rank, 4), dtype=np.float32),
        # Adapter 2: _LoRAEinsum_0 under q_einsum.
        ('attn', 'q_einsum', 'w'): np.zeros((3, 8, 4), dtype=np.float32),
        ('attn', 'q_einsum', '_LoRAEinsum_0', 'lora', 'a'): np.ones(
            (8, rank), dtype=np.float32
        ),
        ('attn', 'q_einsum', '_LoRAEinsum_0', 'lora', 'b'): np.ones(
            (rank, 3, 4), dtype=np.float32
        ),
    }
    params = flax.traverse_util.unflatten_dict(flat)
    fused = lora.fuse_lora_params(params)

    # Both base weights should have been modified.
    self.assertFalse(np.all(fused['dense']['kernel'] == 0))
    self.assertFalse(np.all(fused['attn']['q_einsum']['w'] == 0))

    # LoRA keys should be gone.
    self.assertNotIn('lora', fused['dense'])
    self.assertNotIn('_LoRAEinsum_0', fused['attn']['q_einsum'])

  def test_missing_base_weight_raises(self):
    """Raise KeyError if no base weight is found for a LoRA adapter."""
    import flax.traverse_util  # pylint: disable=g-import-not-at-top

    flat = {
        ('orphan', '_LoRAEinsum_0', 'lora', 'a'): np.ones(
            (4, 2), dtype=np.float32
        ),
        ('orphan', '_LoRAEinsum_0', 'lora', 'b'): np.ones(
            (2, 3), dtype=np.float32
        ),
    }
    params = flax.traverse_util.unflatten_dict(flat)
    with self.assertRaises(KeyError):
      lora.fuse_lora_params(params)


class FindInterleavingTest(parameterized.TestCase):
  """Tests for lora._find_interleaving."""

  def test_identity(self):
    """seq_a + seq_b == target → trivial interleaving."""
    result = lora._find_interleaving((4, 8), (4,), (8,))
    self.assertEqual(result, ['a', 'b'])

  def test_reversed(self):
    """seq_b then seq_a."""
    result = lora._find_interleaving((8, 4), (4,), (8,))
    self.assertEqual(result, ['b', 'a'])

  def test_kv_einsum_pattern(self):
    """kv_einsum: target=(C=2, K=1, D=1536, H=256), a=(D,), b=(C,K,H)."""
    result = lora._find_interleaving((2, 1, 1536, 256), (1536,), (2, 1, 256))
    self.assertEqual(result, ['b', 'b', 'a', 'b'])

  def test_q_einsum_pattern(self):
    """q_einsum: target=(N=8, D=1536, H=256), a=(D,), b=(N,H)."""
    result = lora._find_interleaving((8, 1536, 256), (1536,), (8, 256))
    self.assertEqual(result, ['b', 'a', 'b'])

  def test_gating_einsum_pattern(self):
    """gating_einsum: target=(2, H=6144, D=1536), a=(D,), b=(2,H)."""
    result = lora._find_interleaving((2, 6144, 1536), (1536,), (2, 6144))
    self.assertEqual(result, ['b', 'b', 'a'])

  def test_no_valid_interleaving_raises(self):
    """Raise ValueError when no interleaving is possible."""
    with self.assertRaises(ValueError):
      lora._find_interleaving((10, 20), (5,), (8,))

  def test_ambiguous_interleaving_raises(self):
    """Raise ValueError when axes have the same size → ambiguous."""
    # target=(64, 64), a=(64,), b=(64,) — two valid orderings.
    with self.assertRaises(ValueError):
      lora._find_interleaving((64, 64), (64,), (64,))


class ComputeLoraDeltaTest(parameterized.TestCase):
  """Tests for lora._compute_lora_delta."""

  def test_2d_fast_path(self):
    """2D case: a @ b matches target directly."""
    a = np.ones((6, 2), dtype=np.float32)
    b = np.ones((2, 4), dtype=np.float32)
    delta = lora._compute_lora_delta(a, b, target_shape=(6, 4))
    self.assertEqual(delta.shape, (6, 4))
    # ones(6,2) @ ones(2,4) = full(6,4, fill=2)
    np.testing.assert_allclose(delta, np.full((6, 4), 2.0))

  def test_3d_needs_interleaving(self):
    """q_einsum: a=(D,r), b=(r,N,H), target=(N,D,H)."""
    rank = 2
    d, n, h = 12, 3, 4
    a = np.ones((d, rank), dtype=np.float32)
    b = np.ones((rank, n, h), dtype=np.float32)
    delta = lora._compute_lora_delta(a, b, target_shape=(n, d, h))
    self.assertEqual(delta.shape, (n, d, h))

  def test_4d_kv_einsum(self):
    """kv_einsum: a=(D,r), b=(r,C,K,H), target=(C,K,D,H)."""
    rank = 2
    d, c, k, h = 12, 2, 1, 4
    a = np.ones((d, rank), dtype=np.float32)
    b = np.ones((rank, c, k, h), dtype=np.float32)
    delta = lora._compute_lora_delta(a, b, target_shape=(c, k, d, h))
    self.assertEqual(delta.shape, (c, k, d, h))

  def test_numerical_correctness(self):
    """Verify the delta equals the correct contraction numerically."""
    rng = np.random.RandomState(42)
    a = rng.randn(5, 3).astype(np.float32)  # (D=5, r=3)
    b = rng.randn(3, 2, 4).astype(np.float32)  # (r=3, N=2, H=4)
    target_shape = (2, 5, 4)  # (N, D, H)
    delta = lora._compute_lora_delta(a, b, target_shape=target_shape)
    # Manual: einsum 'ar,rbc->bac'

    expected = jnp.einsum('ar,rbc->bac', a, b)
    np.testing.assert_allclose(delta, expected, rtol=1e-5)

# ---------------------------------------------------------------------------
# Tests for _matches_target_modules and ALL_LINEAR.
# ---------------------------------------------------------------------------


def _make_mock_module(path, name=None):
  """Create a lightweight mock with ``.path`` and ``.name`` attributes."""
  m = mock.MagicMock(spec=['path', 'name'])
  m.path = path
  m.name = name or (path[-1] if path else '')
  return m


class MatchesTargetModulesTest(parameterized.TestCase):
  """Tests for lora._matches_target_modules."""

  def test_none_matches_everything(self):
    module = _make_mock_module(('layer_0', 'attn', 'q_einsum'))
    self.assertTrue(lora._matches_target_modules(module, None))

  def test_all_linear_matches_everything(self):
    module = _make_mock_module(('layer_0', 'attn', 'q_einsum'))
    self.assertTrue(lora._matches_target_modules(module, lora.ALL_LINEAR))

  def test_exact_name_match(self):
    module = _make_mock_module(('layer_0', 'attn', 'q_einsum'))
    self.assertTrue(lora._matches_target_modules(module, ['q_einsum']))

  def test_parent_scope_match(self):
    module = _make_mock_module(('layer_0', 'attn', 'q_einsum'))
    self.assertTrue(lora._matches_target_modules(module, ['attn']))

  def test_full_path_regex_match(self):
    module = _make_mock_module(('layer_0', 'attn', 'q_einsum'))
    self.assertTrue(lora._matches_target_modules(module, [r'layer_0/attn']))

  def test_no_match(self):
    module = _make_mock_module(('layer_0', 'attn', 'q_einsum'))
    self.assertFalse(lora._matches_target_modules(module, ['mlp']))

  def test_multiple_patterns_any_matches(self):
    module = _make_mock_module(('layer_0', 'mlp', 'linear'))
    self.assertTrue(lora._matches_target_modules(module, ['q_einsum', 'mlp']))

  def test_regex_pattern(self):
    module = _make_mock_module(('layer_0', 'attn', 'kv_einsum'))
    self.assertTrue(lora._matches_target_modules(module, [r'(q|kv)_einsum']))

  def test_regex_layer_range(self):
    """Regex can match a range of layers."""
    mod0 = _make_mock_module(('layer_0', 'attn', 'q_einsum'))
    mod5 = _make_mock_module(('layer_5', 'attn', 'q_einsum'))
    mod10 = _make_mock_module(('layer_10', 'attn', 'q_einsum'))
    pattern = [r'layer_[0-4]/']
    self.assertTrue(lora._matches_target_modules(mod0, pattern))
    self.assertFalse(lora._matches_target_modules(mod5, pattern))
    self.assertFalse(lora._matches_target_modules(mod10, pattern))

  def test_invalid_string_raises(self):
    module = _make_mock_module(('layer_0', 'attn', 'q_einsum'))
    with self.assertRaises(ValueError):
      lora._matches_target_modules(module, 'some-random-string')

  def test_fallback_to_name_when_no_path(self):
    """When module has no .path attribute, falls back to .name."""
    m = mock.MagicMock(spec=['name'])
    m.name = 'q_einsum'
    self.assertTrue(lora._matches_target_modules(m, ['q_einsum']))
    self.assertFalse(lora._matches_target_modules(m, ['mlp']))


# ---------------------------------------------------------------------------
# End-to-end tests for LoRA with target_modules.
# ---------------------------------------------------------------------------


class _TwoLayerModel(nn.Module):
  """Minimal model with two named Dense sublayers."""

  @nn.compact
  def __call__(self, x):
    h = nn.Dense(8, name='dense_a')(x)
    return nn.Dense(4, name='dense_b')(h)


class LoRATargetModulesE2ETest(parameterized.TestCase):
  """End-to-end tests for LoRA target_modules filtering."""

  def _init_lora(self, target_modules):
    """Init a LoRA-wrapped _TwoLayerModel and return the params tree."""
    inner = _TwoLayerModel()
    lora_model = lora.LoRA(rank=2, model=inner, target_modules=target_modules)
    params = lora_model.init(jax.random.key(0), jnp.zeros((1, 4)))['params']
    return params

  def test_none_applies_to_all(self):
    """target_modules=None wraps all Dense layers."""
    params = self._init_lora(None)
    self.assertIn('lora', params['dense_a'])
    self.assertIn('lora', params['dense_b'])

  def test_all_linear_applies_to_all(self):
    """target_modules='all-linear' wraps all Dense layers."""
    params = self._init_lora(lora.ALL_LINEAR)
    self.assertIn('lora', params['dense_a'])
    self.assertIn('lora', params['dense_b'])

  def test_single_target(self):
    """Only dense_a gets LoRA."""
    params = self._init_lora(['dense_a'])
    self.assertIn('lora', params['dense_a'])
    self.assertNotIn('lora', params['dense_b'])

  def test_single_target_other(self):
    """Only dense_b gets LoRA."""
    params = self._init_lora(['dense_b'])
    self.assertNotIn('lora', params['dense_a'])
    self.assertIn('lora', params['dense_b'])

  def test_both_targets_explicit(self):
    """Both layers targeted explicitly."""
    params = self._init_lora(['dense_a', 'dense_b'])
    self.assertIn('lora', params['dense_a'])
    self.assertIn('lora', params['dense_b'])

  def test_regex_target(self):
    """Regex pattern matches both Dense layers."""
    params = self._init_lora([r'dense_[ab]'])
    self.assertIn('lora', params['dense_a'])
    self.assertIn('lora', params['dense_b'])

  def test_no_match_no_lora_params(self):
    """If no module matches, no LoRA params are created."""
    params = self._init_lora(['nonexistent'])
    self.assertNotIn('lora', params.get('dense_a', {}))
    self.assertNotIn('lora', params.get('dense_b', {}))

  def test_lora_shapes(self):
    """Verify the LoRA param shapes when targeting one Dense."""
    params = self._init_lora(['dense_a'])
    lora_params = params['dense_a']['lora']
    # dense_a: input=4, output=8, rank=2
    self.assertEqual(lora_params['a'].shape, (4, 2))
    self.assertEqual(lora_params['b'].shape, (2, 8))


if __name__ == '__main__':
  absltest.main()
