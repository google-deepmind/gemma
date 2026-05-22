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

from typing import Optional

from gemma.gm.utils import _jax_utils
import jax.numpy as jnp
from kauldron import kd
import pytest


@pytest.mark.parametrize('kt', [kd.typing, kd.ktyping])
def test_flatten_batch_dim(kt):

  @_jax_utils.flatten_unflatten_batch_dim()
  def f(
      arr0: kt.Float['*b h n'],
      arr1: kt.Float['*b'],
      arr_multi: kt.Float['*b n'] | kt.Float['*b n w'],
      arr2: Optional[kd.typing.Float['*b h']] = None,
      arr3: None | kd.typing.Float['*b h'] = None,
      *,
      expected_batch_dim_size: int = 3,
      other1=4,  # No annotations.
      expected_multi_shape: tuple[int, ...],
  ):
    del other1
    assert len(arr0.shape) == 3
    assert len(arr1.shape) == 1
    assert arr0.shape[0] == expected_batch_dim_size
    assert arr1.shape[0] == expected_batch_dim_size
    assert arr_multi.shape[0] == expected_batch_dim_size
    assert arr_multi.shape[1:] == expected_multi_shape
    if arr2 is not None:
      assert len(arr2.shape) == 2
      assert arr2.shape[0] == expected_batch_dim_size
    if arr3 is not None:
      assert len(arr3.shape) == 2
      assert arr3.shape[0] == expected_batch_dim_size
    return arr0

  assert _jax_utils._get_argname_to_non_batch_dim_size(f) == {
      'arr0': 2,
      'arr1': 0,
      'arr2': 1,
      'arr3': 1,
  }

  x = f(
      arr0=jnp.ones((2, 3, 1, 4)),  # *b == (2, 3)
      arr1=jnp.ones((2, 3)),
      arr_multi=jnp.ones((2, 3, 2)),
      arr2=jnp.ones((2, 3, 2)),
      arr3=jnp.ones((2, 3, 3)),
      expected_batch_dim_size=6,
      other1=4,
      expected_multi_shape=(2,),
  )
  assert x.shape == (2, 3, 1, 4)
  x = f(
      arr0=jnp.ones((2, 3, 4)),  # *b == (2,)
      arr1=jnp.ones((2,)),
      arr_multi=jnp.ones((2, 3, 2)),
      arr2=jnp.ones((2, 2)),
      expected_batch_dim_size=2,
      other1=4,
      expected_multi_shape=(3, 2),
  )
  assert x.shape == (2, 3, 4)
  x = f(
      arr0=jnp.ones((3, 4)),  # *b == ()
      arr1=jnp.ones(()),
      arr_multi=jnp.ones((3, 2)),
      expected_batch_dim_size=1,
      other1=4,
      expected_multi_shape=(3, 2),
  )
  assert x.shape == (3, 4)
