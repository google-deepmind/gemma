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

"""Utils."""

import itertools

from kauldron.typing import Array
import numpy as np


def count_consecutive(values: Array['L']) -> tuple[tuple[int | bool, int], ...]:
  """Counts consecutive identical elements in a list.

  Useful to debug masks with padding.

  Args:
    values: A list of elements.

  Returns:
    A tuple of tuples, where each inner tuple contains the element
    and its consecutive count.
  """
  return tuple(
      (np.asarray(key).item(), len(list(group)))
      for key, group in itertools.groupby(values)
  )
