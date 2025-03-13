# Copyright 2024 DeepMind Technologies Limited.
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

from gemma import peft


def test_split_params():
  params = {
      'dense': {
          'kernel': 0,
          'bias': 1,
          'lora': {
              'a': 0,
              'b': 1,
          },
      },
      'branch_with_only_lora': {
          'lora': {
              'a': 0,
              'b': 1,
          },
      },
      'other': 0,
      # Nested branches are fully removed from the lora tree.
      'b': {'f': {'a': {}}},
  }

  original, lora = peft.split_params(params)
  assert original == {
      'dense': {
          'kernel': 0,
          'bias': 1,
      },
      'branch_with_only_lora': {},
      'other': 0,
      'b': {'f': {'a': {}}},
  }
  assert lora == {
      'dense': {
          'lora': {
              'a': 0,
              'b': 1,
          },
      },
      'branch_with_only_lora': {
          'lora': {
              'a': 0,
              'b': 1,
          },
      },
  }

  assert peft.merge_params(original, lora) == params


def test_quantize_lora_params():
    """Test the QLoRA parameter quantization function."""
    lora_params = {
        'dense': {
            'lora': {
                'a': 0.156,
                'b': -0.234,
            },
        },
        'other_layer': {
            'lora': {
                'x': 0.78,
                'y': -0.99,
            },
        },
    }

    quantized_lora = peft.quantize_lora_params(lora_params, bits=4)

    assert 0 <= quantized_lora['dense']['lora']['a'] <= 1, "Value out of range after quantization"
    assert -1 <= quantized_lora['dense']['lora']['b'] <= 0, "Value out of range after quantization"
    assert 0 <= quantized_lora['other_layer']['lora']['x'] <= 1, "Value out of range"
    assert -1 <= quantized_lora['other_layer']['lora']['y'] <= 0, "Value out of range"

    print("QLoRA parameter quantization test passed!")
