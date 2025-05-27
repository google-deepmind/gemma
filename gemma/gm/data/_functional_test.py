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

from gemma import gm
import numpy as np
import pytest


def test_pad():
  np.testing.assert_array_equal(
      gm.data.pad([1, 2, 3], max_length=6),
      [1, 2, 3, 0, 0, 0],
  )
  np.testing.assert_array_equal(
      gm.data.pad([1, 2, 3, 4, 5, 6], max_length=6),
      [1, 2, 3, 4, 5, 6],
  )
  with pytest.raises(ValueError, match="Cannot pad sequence"):
    gm.data.pad([1, 2, 3, 4, 5, 6, 7], max_length=6)

  np.testing.assert_array_equal(
      gm.data.pad([1, 2, 3, 4, 5, 6, 7], max_length=6, truncate=True),
      [1, 2, 3, 4, 5, 6],
  )

  arr = np.arange(6).reshape((2, 3)) + 1
  np.testing.assert_array_equal(
      gm.data.pad(arr, max_length=6),
      [
          [1, 2, 3, 0, 0, 0],
          [4, 5, 6, 0, 0, 0],
      ],
  )


def test_seq2seq():

  out = gm.data.make_seq2seq_fields(
      prompt=[10, 11, 12, 13, 14],
      response=[20, 21, 1],  # Response ends with EOS token.
  )

  expected_output = {
      # fmt: off
      # pylint: disable=bad-whitespace
      "input":       [10, 11, 12, 13, 14, 20, 21],
      "target":      [11, 12, 13, 14, 20, 21,  1],
      "target_mask": [ 0,  0,  0,  0,  1,  1,  1],
      # pylint: enable=bad-whitespace
      # fmt: on
  }
  np.testing.assert_array_equal(out.input, expected_output["input"])
  np.testing.assert_array_equal(out.target, expected_output["target"])
  np.testing.assert_array_equal(out.target_mask, expected_output["target_mask"])
