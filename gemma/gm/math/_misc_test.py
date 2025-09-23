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

from gemma.gm.math import _misc
import numpy as np
import pytest


@pytest.mark.parametrize(
    "input_list, expected",
    [
        ([], ()),
        ([1], ((1, 1),)),
        ([1, 2, 3], ((1, 1), (2, 1), (3, 1))),
        (
            [1, 1, 2, 2, 2, 3, 1, 1, 1, 2, 2],
            ((1, 2), (2, 3), (3, 1), (1, 3), (2, 2)),
        ),
        (
            np.array([1, 1, 2, 2, 2, 3]),
            ((1, 2), (2, 3), (3, 1)),
        ),
    ],
)
def test_count_consecutive(input_list, expected):
  assert _misc.count_consecutive(input_list) == expected
