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

from unittest import mock

from absl.testing import absltest
from etils import epath

from gemma.diffusion.hackable_diffusion_adapter.hd import gemma_checkpointer

# Sentinel values to distinguish model-init vs checkpoint values.
_MODEL = 'model_value'
_CKPT = 'ckpt_value'
_LORA = 'lora_init_value'


def _make_model_flat(keys):
  """Creates a flat model dict with sentinel values."""
  return {k: _MODEL for k in keys}


def _make_ckpt_flat(keys):
  """Creates a flat checkpoint dict with sentinel values."""
  return {k: _CKPT for k in keys}


class RemapAndMatchParamsTest(absltest.TestCase):
  """Tests for _remap_and_match_params."""

  ##############################################################################
  # MARK: Test remapping: checkpoint has /w, model does not.
  ##############################################################################

  def test_remap_strips_w_when_model_has_no_w(self):
    """mlp/gating_einsum in model, mlp/gating_einsum/w in ckpt."""
    model_flat = _make_model_flat([
        'layer_0/attn/q_einsum/w',
        'layer_0/mlp/gating_einsum',
        'layer_0/mlp/linear',
    ])
    ckpt_flat = _make_ckpt_flat([
        'layer_0/attn/q_einsum/w',
        'layer_0/mlp/gating_einsum/w',
        'layer_0/mlp/linear/w',
    ])

    result = gemma_checkpointer._remap_and_match_params(model_flat, ckpt_flat)

    # All model keys should be loaded from checkpoint.
    self.assertEqual(result['layer_0/attn/q_einsum/w'], _CKPT)
    self.assertEqual(result['layer_0/mlp/gating_einsum'], _CKPT)
    self.assertEqual(result['layer_0/mlp/linear'], _CKPT)

  ##############################################################################
  # MARK: mlp keeps /w, mlp2 strips /w.
  ##############################################################################

  def test_remap_keeps_w_for_mlp_and_strips_w_for_mlp2(self):
    """mlp has /w, mlp2 doesn't (FeedForward)."""
    model_flat = _make_model_flat([
        'layer_0/attn/q_einsum/w',
        # model expects /w.
        'layer_0/mlp/gating_einsum/w',
        'layer_0/mlp/linear/w',
        # model does NOT have /w.
        'layer_0/mlp2/gating_einsum',
        'layer_0/mlp2/linear',
    ])
    ckpt_flat = _make_ckpt_flat([
        'layer_0/attn/q_einsum/w',
        'layer_0/mlp/gating_einsum/w',
        'layer_0/mlp/linear/w',
        'layer_0/mlp2/gating_einsum/w',
        'layer_0/mlp2/linear/w',
    ])

    result = gemma_checkpointer._remap_and_match_params(model_flat, ckpt_flat)

    # mlp paths should keep /w (not stripped).
    self.assertEqual(result['layer_0/mlp/gating_einsum/w'], _CKPT)
    self.assertEqual(result['layer_0/mlp/linear/w'], _CKPT)
    # mlp2 paths should have /w stripped.
    self.assertEqual(result['layer_0/mlp2/gating_einsum'], _CKPT)
    self.assertEqual(result['layer_0/mlp2/linear'], _CKPT)

  ##############################################################################
  # MARK: Exact match: no /w stripping needed.
  ##############################################################################

  def test_exact_match_no_remapping(self):
    """When all paths match exactly, no remapping occurs."""
    keys = ['layer_0/attn/w', 'layer_0/norm/scale', 'embedder/input_embedding']
    model_flat = _make_model_flat(keys)
    ckpt_flat = _make_ckpt_flat(keys)

    result = gemma_checkpointer._remap_and_match_params(model_flat, ckpt_flat)

    for key in keys:
      self.assertEqual(result[key], _CKPT)

  ##############################################################################
  # MARK: Checkpoint-only keys are silently discarded (warning logged).
  ##############################################################################

  def test_checkpoint_only_keys_discarded(self):
    """Keys in checkpoint but not in model are discarded without error."""
    model_flat = _make_model_flat(['layer_0/attn/w'])
    ckpt_flat = _make_ckpt_flat([
        'layer_0/attn/w',
        'distill_final_norm/scale',
        'embedder/audio_input_projection/w',
    ])

    # Should not raise.
    result = gemma_checkpointer._remap_and_match_params(model_flat, ckpt_flat)
    self.assertEqual(result['layer_0/attn/w'], _CKPT)
    # Checkpoint-only keys should not appear in result.
    self.assertNotIn('distill_final_norm/scale', result)
    self.assertNotIn('embedder/audio_input_projection/w', result)

  ##############################################################################
  # MARK: LoRA keys are preserved from init values.
  ##############################################################################

  def test_lora_keys_preserved(self):
    """LoRA params are kept at their initialized values, not from ckpt."""
    model_flat = _make_model_flat([
        'layer_0/attn/w',
        'layer_0/attn/lora/a',
        'layer_0/attn/lora/b',
    ])
    ckpt_flat = _make_ckpt_flat([
        'layer_0/attn/w',
        # LoRA keys are NOT in the checkpoint.
    ])
    lora_init_values = {
        'layer_0/attn/lora/a': _LORA,
        'layer_0/attn/lora/b': _LORA,
    }

    result = gemma_checkpointer._remap_and_match_params(
        model_flat, ckpt_flat, lora_init_values
    )

    # Regular param from checkpoint.
    self.assertEqual(result['layer_0/attn/w'], _CKPT)
    # LoRA params preserved from init.
    self.assertEqual(result['layer_0/attn/lora/a'], _LORA)
    self.assertEqual(result['layer_0/attn/lora/b'], _LORA)

  ##############################################################################
  # MARK: Non-LoRA model-only keys raise KeyError.
  ##############################################################################

  def test_non_lora_model_only_keys_raise(self):
    """Non-LoRA keys in model but not checkpoint should raise KeyError."""
    model_flat = _make_model_flat([
        'layer_0/attn/w',
        'layer_0/some_new_param',
    ])
    ckpt_flat = _make_ckpt_flat([
        'layer_0/attn/w',
        # 'layer_0/some_new_param' is missing from checkpoint.
    ])

    with self.assertRaises(KeyError):
      gemma_checkpointer._remap_and_match_params(model_flat, ckpt_flat)

  ##############################################################################
  # MARK: Edge case: /w suffix that is NOT a Flax weight nesting.
  ##############################################################################

  def test_w_suffix_not_stripped_when_both_paths_in_model(self):
    """If both 'foo/w' and 'foo' are in the model, don't strip."""
    model_flat = _make_model_flat([
        'layer_0/foo/w',
        'layer_0/foo',
    ])
    ckpt_flat = _make_ckpt_flat([
        'layer_0/foo/w',
        'layer_0/foo',
    ])

    result = gemma_checkpointer._remap_and_match_params(model_flat, ckpt_flat)

    self.assertEqual(result['layer_0/foo/w'], _CKPT)
    self.assertEqual(result['layer_0/foo'], _CKPT)

  ##############################################################################
  # MARK: Multi-layer: verifies remapping works across many layers.
  ##############################################################################

  def test_multi_layer_moe(self):
    """Multiple layers with MoE structure all get correctly remapped."""
    model_keys = []
    ckpt_keys = []
    for i in range(3):
      # Shared keys (exact match).
      model_keys.append(f'layer_{i}/attn/q_einsum/w')
      ckpt_keys.append(f'layer_{i}/attn/q_einsum/w')
      # MoE mlp — keeps /w.
      model_keys.append(f'layer_{i}/mlp/gating_einsum/w')
      ckpt_keys.append(f'layer_{i}/mlp/gating_einsum/w')
      # Shared mlp2 — strips /w.
      model_keys.append(f'layer_{i}/mlp2/gating_einsum')
      ckpt_keys.append(f'layer_{i}/mlp2/gating_einsum/w')

    model_flat = _make_model_flat(model_keys)
    ckpt_flat = _make_ckpt_flat(ckpt_keys)

    result = gemma_checkpointer._remap_and_match_params(model_flat, ckpt_flat)

    for i in range(3):
      self.assertEqual(result[f'layer_{i}/attn/q_einsum/w'], _CKPT)
      self.assertEqual(result[f'layer_{i}/mlp/gating_einsum/w'], _CKPT)
      self.assertEqual(result[f'layer_{i}/mlp2/gating_einsum'], _CKPT)

  ##############################################################################
  # MARK: LoRA keys + checkpoint-only keys combined.
  ##############################################################################

  def test_lora_and_checkpoint_only_combined(self):
    """LoRA + checkpoint-only keys work together without errors."""
    model_flat = _make_model_flat([
        'layer_0/attn/w',
        'layer_0/attn/lora/a',
    ])
    ckpt_flat = _make_ckpt_flat([
        'layer_0/attn/w',
        'distill_layer_0/scale',  # Checkpoint-only, will be discarded.
    ])
    lora_init_values = {
        'layer_0/attn/lora/a': _LORA,
    }

    result = gemma_checkpointer._remap_and_match_params(
        model_flat, ckpt_flat, lora_init_values
    )

    self.assertEqual(result['layer_0/attn/w'], _CKPT)
    self.assertEqual(result['layer_0/attn/lora/a'], _LORA)
    self.assertNotIn('distill_layer_0/scale', result)


class GemmaCheckpointerTest(absltest.TestCase):

  @mock.patch(
      'gemma.diffusion.hackable_diffusion_adapter.hd.gemma_checkpointer.save_params_into_original_and_lora_params'
  )
  def test_evaluator(self, mock_save_fn):
    # Create dummy state
    mock_state = mock.MagicMock()
    mock_state.step = 1000

    dummy_params = {'weight': 1.0}

    with mock.patch('kauldron.kontext.get_by_path') as mock_get_by_path:
      mock_get_by_path.return_value = dummy_params

      formatter = gemma_checkpointer.GemmaCheckpointFormatter(
          workdir='/tmp/fake_workdir',
          gemma_param_path='fake_path',
          run=mock.MagicMock(),
      )

      res = formatter.evaluate(state=mock_state, step=1000)

      self.assertEqual(res, {})  # returns empty dict
      mock_get_by_path.assert_called_with(mock_state.params, 'fake_path')
      mock_save_fn.assert_called_with(
          params=dummy_params,
          step_nr=1000,
          workdir=epath.Path('/tmp/fake_workdir') / 'gemma_like_params',
          write_original_params=True,
          write_lora_params=True,
          write_fused_lora_params=True,
      )

  @mock.patch(
      'gemma.diffusion.hackable_diffusion_adapter.hd.lora.fuse_lora_params'
  )
  @mock.patch('gemma.peft.split_params')
  @mock.patch('orbax.checkpoint.PyTreeCheckpointer')
  def test_save_params(
      self, mock_checkpointer_cls, mock_split_params, mock_fuse
  ):
    # Setup mocks
    mock_checkpointer = mock_checkpointer_cls.return_value

    # Mock split params to return some dummy split
    mock_split = mock.MagicMock()
    mock_split.original = {'orig': 1}
    mock_split.lora = {'lora_key': 2}  # len > 0
    mock_split_params.return_value = mock_split

    mock_fuse.return_value = {'fused': 3}

    temp_dir = epath.Path(__import__('tempfile').mkdtemp())

    # Mock exist checks to avoid actually writing or skipping
    # because of existing dirs
    with mock.patch.object(epath.Path, 'exists', return_value=False):
      gemma_checkpointer.save_params_into_original_and_lora_params(
          params={'some': 'params'},
          step_nr=500,
          workdir=temp_dir,
      )

    mock_split_params.assert_called_with({'some': 'params'})
    mock_fuse.assert_called_with({'some': 'params'})

    # Check save was called for original, lora and fused
    mock_checkpointer.save.assert_has_calls(
        [
            mock.call(temp_dir / 'original_params_500', {'orig': 1}),
            mock.call(temp_dir / 'lora_params_500', {'lora_key': 2}),
            mock.call(temp_dir / 'fused_lora_params_500', {'fused': 3}),
        ],
        any_order=True,
    )


if __name__ == '__main__':
  absltest.main()
