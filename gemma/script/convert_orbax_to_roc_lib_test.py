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

"""Tests for the Gemma orbax -> ROC conversion script."""

from absl.testing import absltest
from gemma.gm.ckpts import _checkpoint
from gemma.script import convert_orbax_to_roc_lib

# Standard Gemma 3 4B-style layer (no MoE, no skip_scale).
_GEMMA3_LAYER = {
    'pre_attention_norm': {'scale': 'PRE_ATTN_SCALE'},
    'attn': {
        'attn_vec_einsum': {'w': 'ATTN_VEC_W'},
        'q_einsum': {'w': 'Q_W'},
        'kv_einsum': {'w': 'KV_W'},
        '_query_norm': {'scale': 'QN_SCALE'},
        '_key_norm': {'scale': 'KN_SCALE'},
    },
    'post_attention_norm': {'scale': 'POST_ATTN_SCALE'},
    'pre_ffw_norm': {'scale': 'PRE_FFW_SCALE'},
    'mlp': {
        'gating_einsum': {'w': 'GATING_W'},
        'linear': {'w': 'LINEAR_W'},
    },
    'post_ffw_norm': {'scale': 'POST_FFW_SCALE'},
}

# Gemma 4 dense layer with a standalone `skip_scale` array next to submodules.
_GEMMA4_DENSE_LAYER = dict(_GEMMA3_LAYER, skip_scale='SKIP_SCALE_ARRAY')
_GEMMA4_DENSE_LAYER['attn'] = {
    'attn_vec_einsum': {'w': 'ATTN_VEC_W'},
    'q_einsum': {'w': 'Q_W'},
    'kv_einsum': {'w': 'KV_W'},
    'query_norm': {'scale': 'QN_SCALE'},
    'key_norm': {'scale': 'KN_SCALE'},
}

# Gemma 4 MoE layer: the `mlp` module contains both submodule dicts
# (`router_logits`, `gating_einsum`, `linear`) AND standalone array params
# (`per_expert_scale`, `router_scale`). This is the case that triggered the
# original conversion bug.
_GEMMA4_MOE_LAYER = dict(_GEMMA4_DENSE_LAYER)
_GEMMA4_MOE_LAYER['mlp'] = {
    'router_logits': {'w': 'ROUTER_LOGITS_W'},
    'gating_einsum': {'w': 'GATING_W'},
    'linear': {'w': 'LINEAR_W'},
    'per_expert_scale': 'PER_EXPERT_SCALE',  # standalone array
    'router_scale': 'ROUTER_SCALE',  # standalone array
}


def _nested_dense_model():
  return {
      'embedder': {'input_embedding': 'INPUT_EMBED'},
      'final_norm': {'scale': 'FINAL_NORM_SCALE'},
      'layer_0': _GEMMA3_LAYER,
      'layer_1': _GEMMA3_LAYER,
  }


def _nested_gemma4_model():
  return {
      'embedder': {'input_embedding': 'INPUT_EMBED'},
      'final_norm': {'scale': 'FINAL_NORM_SCALE'},
      'layer_0': _GEMMA4_DENSE_LAYER,
      'layer_1': _GEMMA4_DENSE_LAYER,
  }


def _nested_gemma4_moe_model():
  return {
      'embedder': {'input_embedding': 'INPUT_EMBED'},
      'final_norm': {'scale': 'FINAL_NORM_SCALE'},
      'layer_0': _GEMMA4_MOE_LAYER,
  }


def _nested_mm_model():
  return {
      'embedder': {
          'input_embedding': 'INPUT_EMBED',
          'mm_input_projection': {'w': 'MM_INPUT_PROJ'},
          'mm_soft_embedding_norm': {'scale': 'MM_NORM'},
      },
      'final_norm': {'scale': 'FINAL_NORM_SCALE'},
      'layer_0': _GEMMA3_LAYER,
      'vision_encoder': {
          'siglip_encoder': {
              'embedding': {'kernel': 'EMB_KERNEL', 'bias': 'EMB_BIAS'},
              'pos_embedding': 'POS_EMBED',
              'Transformer': {
                  'encoder_norm': {
                      'scale': 'ENC_NORM_SCALE',
                      'bias': 'ENC_NORM_BIAS',
                  },
                  'encoderblock_0': {
                      'LayerNorm_0': {
                          'scale': 'LN0_SCALE',
                          'bias': 'LN0_BIAS',
                      },
                      'MlpBlock_0': {
                          'Dense_0': {'kernel': 'D0_K', 'bias': 'D0_B'},
                          'Dense_1': {'kernel': 'D1_K', 'bias': 'D1_B'},
                      },
                  },
              },
          },
      },
  }


class ConvertOrbaxToRocTest(absltest.TestCase):
  """Verifies that the conversion matches the gemma library's _nested_to_flat.

  `_nested_to_flat` is the function the gemma library uses internally to map
  `model.init()['params']` (NESTED layout) to the FLAT layout stored on disk
  and consumed by JetEngine. The conversion script must produce the same
  structure or JetEngine will throw type-mismatch errors during model init.
  """

  def _assert_matches_library(self, nested):
    """Asserts the conversion output matches `_checkpoint._nested_to_flat`."""
    flat = convert_orbax_to_roc_lib.convert_full_model_to_flat(nested)
    expected = _checkpoint._nested_to_flat(nested)  # pylint: disable=protected-access
    self.assertEqual(set(flat.keys()), set(expected.keys()))
    for k in flat:
      self.assertEqual(
          flat[k],
          expected[k],
          f'Mismatch at flat key {k!r}: {flat[k]!r} vs {expected[k]!r}',
      )

  def test_gemma3_dense(self):
    self._assert_matches_library(_nested_dense_model())

  def test_gemma4_dense_with_skip_scale(self):
    """Verifies a layer with a standalone `skip_scale` array."""
    self._assert_matches_library(_nested_gemma4_model())

  def test_gemma4_moe(self):
    """Regression test for the Gemma 4 MoE conversion bug.

    Before the fix, the heuristic-based conversion produced flat keys like
    `transformer/layer_0/mlp/per_expert_scale` whose VALUE was the raw array
    instead of a `{leaf_name: array}` dict, breaking the FLAT layout
    convention and triggering type-mismatch errors when JetEngine tried to
    load the checkpoint.
    """
    nested = _nested_gemma4_moe_model()
    flat = convert_orbax_to_roc_lib.convert_full_model_to_flat(nested)

    # MoE-specific assertions: standalone params must be grouped under the
    # `mlp` key as a dict (not promoted to their own flat keys).
    self.assertIn('transformer/layer_0/mlp', flat)
    self.assertEqual(
        flat['transformer/layer_0/mlp'],
        {
            'per_expert_scale': 'PER_EXPERT_SCALE',
            'router_scale': 'ROUTER_SCALE',
        },
    )
    self.assertNotIn('transformer/layer_0/mlp/per_expert_scale', flat)
    self.assertNotIn('transformer/layer_0/mlp/router_scale', flat)
    # Submodule params under `mlp` keep their own flat keys.
    self.assertEqual(
        flat['transformer/layer_0/mlp/gating_einsum'], {'w': 'GATING_W'}
    )
    self.assertEqual(flat['transformer/layer_0/mlp/linear'], {'w': 'LINEAR_W'})
    self.assertEqual(
        flat['transformer/layer_0/mlp/router_logits'],
        {'w': 'ROUTER_LOGITS_W'},
    )

    # Full structural match against the library.
    self._assert_matches_library(nested)

  def test_multimodal_with_vision_encoder(self):
    """Verifies vision params are stored under SigLiPFromPatches_0/."""
    nested = _nested_mm_model()
    flat = convert_orbax_to_roc_lib.convert_full_model_to_flat(nested)

    self.assertIn('SigLiPFromPatches_0/siglip_encoder/embedding', flat)
    self.assertEqual(
        flat['SigLiPFromPatches_0/siglip_encoder/embedding'],
        {'kernel': 'EMB_KERNEL', 'bias': 'EMB_BIAS'},
    )
    # The standalone `pos_embedding` should be grouped under `siglip_encoder`.
    self.assertIn('SigLiPFromPatches_0/siglip_encoder', flat)
    self.assertEqual(
        flat['SigLiPFromPatches_0/siglip_encoder'],
        {'pos_embedding': 'POS_EMBED'},
    )
    # No transformer prefix should leak into vision params.
    for k in flat:
      self.assertFalse(
          k.startswith('transformer/SigLiPFromPatches_0'),
          f'Vision params must not be prefixed with `transformer/`: {k}',
      )

    self._assert_matches_library(nested)

  def test_all_values_are_dicts_in_flat_layout(self):
    """Every value in the flat layout must be a `{leaf_name: array}` dict.

    JetEngine expects this invariant: flat keys map to dicts of leaf params.
    A raw array value would cause type/shape mismatches during init.
    """
    for nested in (
        _nested_dense_model(),
        _nested_gemma4_model(),
        _nested_gemma4_moe_model(),
        _nested_mm_model(),
    ):
      flat = convert_orbax_to_roc_lib.convert_full_model_to_flat(nested)
      for k, v in flat.items():
        self.assertIsInstance(
            v, dict, f'Flat key {k!r} maps to a non-dict value: {v!r}'
        )


if __name__ == '__main__':
  absltest.main()
