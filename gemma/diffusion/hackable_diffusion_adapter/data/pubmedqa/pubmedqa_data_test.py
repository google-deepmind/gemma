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


import os
import tempfile
from absl.testing import absltest
from gemma.diffusion.hackable_diffusion_adapter.data.pubmedqa import pubmedqa_data


class JsonlDataSourceTest(absltest.TestCase):

  def test_load_and_access(self):
    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
      f.write('{"id": 1, "text": "hello"}\n')
      f.write('{"id": 2, "text": "world"}\n')
      temp_path = f.name

    try:
      # Test loading
      ds = pubmedqa_data.JsonlDataSource(temp_path)
      self.assertLen(ds, 2)
      self.assertEqual(ds[0], {"id": 1, "text": "hello"})
      self.assertEqual(ds[1], {"id": 2, "text": "world"})

      # Test slicing
      ds_sliced = pubmedqa_data.JsonlDataSource(temp_path, slice_start=1)
      self.assertLen(ds_sliced, 1)
      self.assertEqual(ds_sliced[0], {"id": 2, "text": "world"})

      ds_sliced_stop = pubmedqa_data.JsonlDataSource(temp_path, slice_stop=1)
      self.assertLen(ds_sliced_stop, 1)
      self.assertEqual(ds_sliced_stop[0], {"id": 1, "text": "hello"})

    finally:
      os.unlink(temp_path)


if __name__ == "__main__":
  absltest.main()
