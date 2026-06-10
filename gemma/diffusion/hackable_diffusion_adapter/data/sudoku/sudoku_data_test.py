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
from gemma.diffusion.hackable_diffusion_adapter.data.sudoku import sudoku_data
from kauldron import random
import tensorflow as tf


class ParseSudokuExampleTest(absltest.TestCase):

  def test_parse_example(self):
    # Create a serialized tf.train.Example
    example = tf.train.Example()
    example.features.feature["puzzle"].bytes_list.value.append(b"530070000...")
    example.features.feature["solution"].bytes_list.value.append(
        b"534678912..."
    )
    record_bytes = example.SerializeToString()

    # Parse it
    transform = sudoku_data.ParseSudokuExample()
    result = transform.map(record_bytes)

    # Verify
    self.assertEqual(result["prompt"], "530070000...")
    self.assertEqual(result["response"], "534678912...")
    self.assertEqual(result["puzzle_raw"], "530070000...")


class BagzDataSourceTest(absltest.TestCase):

  @mock.patch("bagz.Reader")
  def test_bagz_datasource(self, mock_bag_ds):
    # Setup mock data source
    mock_data = [b"record1", b"record2", b"record3"]

    class MockDataSource:

      def __len__(self):
        return len(mock_data)

      def __getitem__(self, idx):
        return mock_data[idx]

    mock_bag_ds.return_value = MockDataSource()

    # Instantiate Bagz
    bagz = sudoku_data.Bagz(
        bagz_path="fake_path", shard_by_process=False, shuffle=False
    )

    # Get dataset
    rng = random.PRNGKey(42)
    ds = bagz.ds_for_current_process(rng)

    # Verify contents
    items = list(ds)
    self.assertEqual(items, [b"record1", b"record2", b"record3"])

  @mock.patch("bagz.Reader")
  def test_bagz_datasource_slicing(self, mock_bag_ds):
    # Setup mock data source
    mock_data = [b"record1", b"record2", b"record3"]

    class MockDataSource:

      def __len__(self):
        return len(mock_data)

      def __getitem__(self, idx):
        return mock_data[idx]

    mock_bag_ds.return_value = MockDataSource()

    # Instantiate Bagz with slicing
    bagz = sudoku_data.Bagz(
        bagz_path="fake_path",
        slice_start=1,
        slice_stop=3,
        shard_by_process=False,
        shuffle=False,
    )

    # Get dataset
    rng = random.PRNGKey(42)
    ds = bagz.ds_for_current_process(rng)

    # Verify contents (should be record2 and record3)
    items = list(ds)
    self.assertEqual(items, [b"record2", b"record3"])


if __name__ == "__main__":
  absltest.main()
