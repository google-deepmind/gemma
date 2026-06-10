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

# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the orbax to roc checkpoint conversion script and its flattening logic."""

from absl.testing import absltest
from gemma.script import convert_orbax_to_roc


class ConvertOrbaxToRocTest(absltest.TestCase):

  def test_flatten_nested_tree_unwraps_single_value_dict(self):
    nested_dict = {
        'layer_0': {
            'final_norm': {
                'scale': {'value': 'tensor_1'}
            },
            'mlp': {
                'linear': {'value': {'value': 'tensor_2'}}
            }
        }
    }
    expected = {
        'layer_0/final_norm': {'scale': 'tensor_1'},
        'layer_0/mlp': {'linear': 'tensor_2'},
    }
    result = convert_orbax_to_roc.flatten_nested_tree(nested_dict)
    self.assertEqual(result, expected)

  def test_convert_full_model_to_flat_with_single_value_dict(self):
    nested_dict = {
        'layer_0': {
            'final_norm': {
                'scale': {'value': 'tensor_1'}
            },
            'mlp': {
                'linear': {'value': 'tensor_2'}
            }
        }
    }
    expected = {
        'transformer/layer_0/final_norm': {'scale': 'tensor_1'},
        'transformer/layer_0/mlp/linear': {'w': 'tensor_2'},
    }
    result = convert_orbax_to_roc.convert_full_model_to_flat(nested_dict)
    self.assertEqual(result, expected)


if __name__ == '__main__':
  absltest.main()
