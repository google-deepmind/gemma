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
from gemma.gm.text import _prefill
from gemma.gm.utils import _types
import jax
import jax.numpy as jnp
import numpy as np


def test_prefill():

  tokenizer = gm.testing.DummyTokenizer()
  model = gm.testing.DummyGemma()

  prompt = 'hello world'
  text = tokenizer.encode(prompt)
  text = jnp.asarray(text)[None, ...]

  params = model.init(
      jax.random.PRNGKey(0),
      tokens=text,
  )
  params = params['params']

  input = _types.Input(  # pylint: disable=redefined-builtin
      text=text,
      images=None,
      config=_types.InputConfig(
          support_images=False,
          num_tokens_per_image=100,
          special_tokens=tokenizer.special_tokens,
      ),
  )

  init_state = _prefill.prefill(
      model=model,
      params=params,
      input=input,
      last_state=None,
      cache_length=64,
      max_out_length=12,
      pad_length=(8,),
      rng=jax.random.PRNGKey(0),
      sharding=None,
  )
  np.testing.assert_array_equal(init_state.step, 0)
  np.testing.assert_array_equal(init_state.done, [0])
  np.testing.assert_array_equal(init_state.last_token, input.last_token)
  np.testing.assert_array_equal(init_state.last_token_pos, input.last_token_pos)
