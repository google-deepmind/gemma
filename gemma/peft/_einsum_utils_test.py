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

from gemma.peft import _einsum_utils
import pytest


def test_einsum():
  # Test with Ellipsis
  assert _einsum_utils.get_lora_einsum_str_and_shapes(
      einsum_str='...ij,jk->...ik',
      weights_shape=(3, 4),
      rank=2,
  ) == ('...ij,jr,rk->...ik', (3, 2), (2, 4))

  # Test with static batch dim
  # pylint: disable=invalid-name
  B = 1
  I = 2  # pylint: disable=unused-variable
  J = 3
  K = 4
  N = 5
  M = 6
  R = 7
  # pylint: enable=invalid-name
  assert _einsum_utils.get_lora_einsum_str_and_shapes(
      einsum_str='bijk,bnmjk->bimn',
      weights_shape=(B, N, M, J, K),
      rank=R,
  ) == ('bijk,bjkr,rnm->bimn', (B, J, K, R), (R, N, M))

  # When r is already in use in the letters, another letter is choosen
  # (here `D`)
  assert _einsum_utils.get_lora_einsum_str_and_shapes(
      einsum_str='...abcABCrR,Rgk->...ABCabcrgk',
      weights_shape=(2, 3, 4),
      rank=1,
  ) == ('...abcABCrR,RD,Dgk->...ABCabcrgk', (2, 1), (1, 3, 4))

  with pytest.raises(ValueError, match='`einsum_str` should be'):
    _einsum_utils.get_lora_einsum_str_and_shapes(
        einsum_str='bijk,bnm,jk->bimn',  # Too many inputs
        weights_shape=(2, 3, 4),
        rank=1,
    )

  with pytest.raises(ValueError, match='should not contain ellipsis'):
    assert _einsum_utils.get_lora_einsum_str_and_shapes(
        einsum_str='...ij,...jk->...ik',
        weights_shape=(3, 4),
        rank=2,
    )
