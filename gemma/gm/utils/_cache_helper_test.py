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
from gemma.gm.text import _prefill
from gemma.gm.text import _turn_utils
from gemma.gm.utils import _cache_helper
from gemma.gm.utils import _types
import jax
import jax.numpy as jnp
import numpy as np

# Activate the fixture
use_hermetic_tokenizer = gm.testing.use_hermetic_tokenizer


def test_cache_helper():
  tokenizer = gm.testing.DummyTokenizer()
  model = gm.testing.DummyGemma()

  prompt = 'hello world'
  text = tokenizer.encode(prompt)
  text = jnp.asarray(text)[None, ...]

  params = model.init(
      jax.random.PRNGKey(0),
      tokens=text,
  )

  input = _types.Input(  # pylint: disable=redefined-builtin
      text=text,
      images=None,
      config=_types.InputConfig(
          support_images=False,
          num_tokens_per_image=100,
          special_tokens=tokenizer.special_tokens,
      ),
  )

  cache = _prefill._get_or_init_cache(
      inputs=input,
      prev_turns=_turn_utils.PrevTurns(last_state=None),
      model=model,
      params=params,
      cache_length=64,
      sharding=None,
  )

  assert cache.total_cache_length == 64

  np.testing.assert_array_equal(
      cache.cache['layer_0']['end_index'],
      [0],
  )
  np.testing.assert_array_equal(
      cache.cache['layer_0']['k'],
      np.zeros((1, 64, 2, 128)),  # b, cache_length, heads, emb_dim
  )
  np.testing.assert_array_equal(
      cache.cache['layer_0']['v'],
      np.zeros((1, 64, 2, 128)),  # b, cache_length, heads, emb_dim
  )

  cache_ones = {
      'layer_0': {
          'end_index': jnp.asarray([4]),
          'k': jnp.ones((1, 16, 2, 128)),
          'v': jnp.ones((1, 16, 2, 128)),
      },
  }
  cache_ones = _cache_helper.Cache(cache_ones)

  # Test the setter / getter.

  new_cache = cache.at[:, :16].set_kv(cache_ones)

  sub_cache_ones = new_cache[:, :16]
  sub_cache_rest = new_cache[:, 16:]

  np.testing.assert_array_equal(
      sub_cache_ones.cache['layer_0']['end_index'],
      [0],  # No changes to the end index.
  )
  np.testing.assert_array_equal(
      sub_cache_ones.cache['layer_0']['k'],
      np.ones((1, 16, 2, 128)),
  )
  np.testing.assert_array_equal(
      sub_cache_ones.cache['layer_0']['v'],
      np.ones((1, 16, 2, 128)),
  )

  np.testing.assert_array_equal(
      sub_cache_rest.cache['layer_0']['end_index'],
      [0],  # No changes to the end index.
  )
  np.testing.assert_array_equal(
      sub_cache_rest.cache['layer_0']['k'],
      np.zeros((1, 48, 2, 128)),
  )
  np.testing.assert_array_equal(
      sub_cache_rest.cache['layer_0']['v'],
      np.zeros((1, 48, 2, 128)),
  )

  # Test `set_end_index`
  new_cache = new_cache.set_end_index(5)
  np.testing.assert_array_equal(
      new_cache.cache['layer_0']['end_index'],
      [5],
  )
