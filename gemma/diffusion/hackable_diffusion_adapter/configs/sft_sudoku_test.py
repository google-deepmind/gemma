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

"""Tests for the Sudoku SFT config.

Validates that the Kauldron config can be loaded, has correct structure,
and the data pipeline can be resolved.
"""

from absl.testing import absltest
from gemma.diffusion.hackable_diffusion_adapter.configs import sft_sudoku
from kauldron import konfig


class SFTSudokuConfigTest(absltest.TestCase):
  """Tests that the config can be loaded and has the expected structure."""

  def setUp(self):
    super().setUp()
    self.cfg = sft_sudoku.get_config()

  def test_config_loads(self):
    """Config should load without errors."""
    self.assertIsNotNone(self.cfg)

  def test_canvas_parameters(self):
    """Canvas parameters should be set for Sudoku (smaller than default)."""
    self.assertEqual(self.cfg.aux["num_canvases"], 1)
    self.assertEqual(self.cfg.aux["canvas_size"], 256)

  def test_model_canvas_size(self):
    """Model should have canvas_size matching the config."""
    model = konfig.resolve(self.cfg.model)
    self.assertEqual(model.canvas_size, 256)

  def test_model_num_canvases(self):
    """Model should have num_canvases matching the config."""
    model = konfig.resolve(self.cfg.model)
    self.assertEqual(model.num_canvases, 1)

  def test_model_prompt_len(self):
    """Prompt length should be 256."""
    model = konfig.resolve(self.cfg.model)
    self.assertEqual(model.prompt_len, 256)

  def test_train_ds_exists(self):
    """Training dataset should be configured."""
    self.assertIsNotNone(self.cfg.train_ds)

  def test_eval_ds_exists(self):
    """Eval dataset should be configured."""
    self.assertIsNotNone(self.cfg.eval_ds)

class SFTSudokuDataPipelineTest(absltest.TestCase):
  """Tests for the data pipeline structure."""

  def test_train_ds_attributes(self):
    """Training dataset should have correct attributes."""
    cfg = sft_sudoku.get_config()
    train_ds = konfig.resolve(cfg.train_ds)
    self.assertEqual(
        train_ds.bagz_path,
        "gemma/diffusion/hackable_diffusion_adapter/data/sudoku/sudoku_train.bagz",
    )
    self.assertTrue(train_ds.shuffle)
    self.assertEqual(train_ds.batch_size, 2)
    self.assertNotEmpty(train_ds.transforms)

  def test_eval_ds_attributes(self):
    """Eval dataset should have correct attributes."""
    cfg = sft_sudoku.get_config()
    eval_ds = konfig.resolve(cfg.eval_ds)
    self.assertEqual(
        eval_ds.bagz_path,
        "gemma/diffusion/hackable_diffusion_adapter/data/sudoku/sudoku_eval.bagz",
    )
    self.assertFalse(eval_ds.shuffle)
    self.assertEqual(eval_ds.batch_size, 2)
    self.assertNotEmpty(eval_ds.transforms)


if __name__ == "__main__":
  absltest.main()
