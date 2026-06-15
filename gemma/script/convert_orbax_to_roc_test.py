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

from absl.testing import absltest
from third_party.py.gemma.script import convert_orbax_to_roc


class ConvertOrbaxToRocTest(absltest.TestCase):

  def test_flatten_nested_tree_unwrap_value(self):
    nested_dict = {
        'layer_0': {
            'bias': {'value': 'tensor_b'},
            'scale': {'value': 'tensor_s'},
        }
    }
    expected = {
        'layer_0': {
            'bias': 'tensor_b',
            'scale': 'tensor_s',
        }
    }
    actual = convert_orbax_to_roc.flatten_nested_tree(nested_dict)
    self.assertEqual(actual, expected)

  def test_flatten_nested_tree_normal(self):
    nested_dict = {
        'layer_0': {
            'bias': 'tensor_b',
            'scale': 'tensor_s',
        }
    }
    expected = {
        'layer_0': {
            'bias': 'tensor_b',
            'scale': 'tensor_s',
        }
    }
    actual = convert_orbax_to_roc.flatten_nested_tree(nested_dict)
    self.assertEqual(actual, expected)

if __name__ == '__main__':
  absltest.main()
