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

from gemma import gm
from gemma.gm.text import _sampler_call
import jax
import jax.numpy as jnp
import numpy as np


def test_get_last_token_pos():

  tokens = jnp.array([
      [3, 4, 2, 0, 0],
      [3, 4, 0, 0, 0],
      [3, 4, 5, 7, 0],
  ])

  last_token_pos = _sampler_call._get_last_token_pos_before_mm(tokens)
  last_token = _sampler_call._get_last_token(tokens)

  np.testing.assert_array_equal(last_token_pos, np.array([2, 1, 3]))
  np.testing.assert_array_equal(last_token, np.array([2, 4, 7]))


def test_end_tokens_mask():
  tokens = jnp.array([
      [1, 2, 3, 10, 4, 5],
      [1, 2, 3, 3, 4, 5],
      [1, 2, 11, 3, 10, 5],
  ])
  expected = jnp.array(
      [
          [1, 2, 3, 10, 0, 0],
          [1, 2, 3, 3, 4, 5],
          [1, 2, 11, 0, 0, 0],
      ],
  )
  out = _sampler_call._mask_tokens_after_end_tokens(tokens, end_tokens=(10, 11))
  np.testing.assert_array_equal(out, expected)


def test_sampler():
  model = gm.testing.DummyGemma()
  params = model.init(
      jax.random.PRNGKey(0),
      jnp.zeros((5,), dtype=jnp.int32),
  )
  params = params['params']
  tokenizer = gm.testing.DummyTokenizer()

  sampler = gm.text.Sampler(
      model=model,
      params=params,
      tokenizer=tokenizer,
      cache_length=128,
      max_out_length=128,
  )
  sampler.sample('Hello world')


# TODO(epot):
# def test_slice_cache():
#   _sampler_call._slice_cache(cache=)
