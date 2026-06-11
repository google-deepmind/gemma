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

"""Tests for convert_orbax_to_roc unwrapping logic."""

from absl.testing import absltest
from gemma.script import convert_orbax_to_roc


class ConvertOrbaxToRocTest(absltest.TestCase):

  def test_flatten_nested_tree_with_value_unwrapping(self):
    nested_dict = {
        'layer_0': {
            'attention': {
                'attn_q': {'value': 'tensor_q'},
                'attn_k': {'value': 'tensor_k'},
            },
            'mlp': {
                'linear': {'value': 'tensor_linear'},
            },
        },
        'final_norm': {'value': 'tensor_fn'},
    }

    expected_flat_map = {
        'layer_0/attention': {
            'attn_q': 'tensor_q',
            'attn_k': 'tensor_k',
        },
        'layer_0/mlp': {
            'linear': 'tensor_linear',
        },
        'final_norm': {'final_norm': 'tensor_fn'},
    }

    flat_map = convert_orbax_to_roc.flatten_nested_tree(nested_dict)

    self.assertEqual(flat_map, expected_flat_map)

  def test_flatten_nested_tree_does_not_unwrap_nested_dict(self):
    # If 'value' is a dict, it should NOT be unwrapped if it contains other
    # dicts. "unwrap dictionaries that contain only a single key 'value', where
    # the value is not another dictionary"
    nested_dict = {
        'layer_0': {
            'attention': {'attn_q': {'value': {'nested_key': 'nested_val'}}}
        }
    }
    # In this case, 'value' is a dict, but it does NOT contain other dicts.
    # if 'value' is {'nested_key': 'nested_val'}, it IS another dictionary.
    # So it should NOT be unwrapped.

    flat_map = convert_orbax_to_roc.flatten_nested_tree(nested_dict)

    # If not unwrapped, 'attn_q' remains
    # {'value': {'nested_key': 'nested_val'}}
    # Since 'attn_q' is a dict, and it contains 'value' (which is a dict of
    # non-dicts), is_module_dict for 'attn_q' will be False (because sub_val is
    # {'nested_key': 'nested_val'} which is a dict, so actually is_module_dict
    # will be True!).
    #
    # Let's trace original standard flatten_nested_tree for:
    # 'attn_q': {'value': {'nested_key': 'nested_val'}}
    # k='attn_q', v={'value': {'nested_key': 'nested_val'}}
    # v is dict.
    # sub_val in v.values() -> sub_val is {'nested_key': 'nested_val'} which
    # is dict.
    # is_module_dict = True.
    # So it recurses: flatten_nested_tree(v, 'layer_0/attention/attn_q')
    # In recursion: nested_dict = {'value': {'nested_key': 'nested_val'}},
    # parent_key = 'layer_0/attention/attn_q'
    # k='value', v={'nested_key': 'nested_val'}
    # v is dict. sub_val is 'nested_val' (not dict).
    # is_module_dict = False.
    # flat_map['layer_0/attention/attn_q/value'] = {'nested_key': 'nested_val'}
    #
    # If we apply our unwrapping logic:
    # If we unwrap if v['value'] is not a dict.
    # In the nested dict:
    # {'attn_q': {'value': {'nested_key': 'nested_val'}}}
    # 'attn_q' has single key 'value', but its value is a dict. So it should
    # NOT be unwrapped.

    expected_flat_map = {
        'layer_0/attention/attn_q/value': {'nested_key': 'nested_val'}
    }
    self.assertEqual(flat_map, expected_flat_map)


if __name__ == '__main__':
  absltest.main()
