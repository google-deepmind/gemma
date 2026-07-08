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

"""Tests for the PubMedQA SFT configs."""

from absl.testing import absltest
from gemma.diffusion.hackable_diffusion_adapter.configs import sft_pubmedqa


class PubMedQAConfigsTest(absltest.TestCase):
  """Tests that PubMedQA configs can be loaded, including with ConfigArgs."""

  def test_pubmedqa_long_loads(self):
    cfg = sft_pubmedqa.get_config()
    self.assertIsNotNone(cfg)

  def test_pubmedqa_long_loads_with_args(self):
    args = sft_pubmedqa.ConfigArgs(use_early_stopping=False)
    cfg = sft_pubmedqa.get_config(args=args)
    self.assertIsNotNone(cfg)


if __name__ == "__main__":
  absltest.main()
