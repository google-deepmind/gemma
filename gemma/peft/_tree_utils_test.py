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
      # Nested branches are fully removed from the peft tree.
      'b': {'f': {'a': {}}},
  }

  original, peft_params = peft.split_params(params)
  assert original == {
      'dense': {
          'kernel': 0,
          'bias': 1,
      },
      'branch_with_only_lora': {},
      'other': 0,
      'b': {'f': {'a': {}}},
  }
  assert peft_params == {
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

  assert peft.merge_params(original, peft_params) == params


def test_split_params_with_prefix():
  params = {
      'dense': {
          'kernel': 0,
          'bias': 1,
      },
      'prefix_k_0': 0,
      'prefix_v_0': 1,
      'other': 0,
  }

  original, prefix = peft.split_params(params)
  assert original == {
      'dense': {
          'kernel': 0,
          'bias': 1,
      },
      'other': 0,
  }
  assert prefix == {
      'prefix_k_0': 0,
      'prefix_v_0': 1,
  }

  assert peft.merge_params(original, prefix) == params
