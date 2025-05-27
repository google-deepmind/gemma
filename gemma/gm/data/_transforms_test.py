# Copyright 2025 DeepMind Technologies Limited.
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

"""Tests for data ops."""

from gemma import gm
import numpy as np


def test_add_seq2seq():
  tr = gm.data.AddSeq2SeqFields(
      in_prompt="prompt",
      in_response="response",
      out_input="input",
      out_target="target",
      out_target_mask="target_mask",
  )

  input_ = {
      "prompt": [10, 11, 12, 13, 14],
      "response": [20, 21, 1],  # Response ends with EOS token.
  }
  expected_output = {
      # fmt: off
      # pylint: disable=bad-whitespace
      "input":       [10, 11, 12, 13, 14, 20, 21],
      "target":      [11, 12, 13, 14, 20, 21,  1],
      "target_mask": [ 0,  0,  0,  0,  1,  1,  1],
      # pylint: enable=bad-whitespace
      # fmt: on
  }
  output = tr.map(input_)

  np.testing.assert_array_equal(output["input"], expected_output["input"])
  np.testing.assert_array_equal(output["target"], expected_output["target"])
  np.testing.assert_array_equal(
      output["target_mask"], expected_output["target_mask"]
  )
