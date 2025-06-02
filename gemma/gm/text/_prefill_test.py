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

import dataclasses

from gemma import gm
from gemma.gm.text import _prefill
from gemma.gm.text import _sampler_loop
from gemma.gm.text import _turn_utils
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
  assert input.length_with_mm == 2

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
  np.testing.assert_array_equal(init_state.init_cache_length, 1)
  np.testing.assert_array_equal(init_state.used_cache_length, 1)
  np.testing.assert_array_equal(init_state.done, [0])
  np.testing.assert_array_equal(init_state.last_token, input.last_token)
  np.testing.assert_array_equal(init_state.last_token_pos, input.last_token_pos)
  np.testing.assert_array_equal(
      init_state.attention_mask_for_step,
      [[1, 1] + ([0] * 62)],  # Attention mask for the last token.
  )


def test_full_attention_mask():
  input = _types.Input(  # pylint: disable=redefined-builtin
      text=jnp.asarray([
          [1, 2, 3, 4],
          [1, 2, 3, 4],
          [1, 2, 0, 0],
          [1, 2, 0, 0],
      ]),
      images=None,
      config=_types.InputConfig(
          support_images=False,
          num_tokens_per_image=100,
          special_tokens=gm.text.Gemma3Tokenizer.special_tokens,
      ),
  )
  assert input.length_with_mm == 4

  first_turn_mask = _prefill._make_full_attention_mask(
      input=input,
      prev_turns=_turn_utils.PrevTurns(last_state=None),
      cache_length=20,
  )
  np.testing.assert_array_equal(
      first_turn_mask,
      [
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      ],
  )

  last_state = _sampler_loop.SamplingState(
      step=jnp.asarray(5),
      done=jnp.ones((4,), dtype=jnp.bool_),
      last_token=jnp.asarray([1, 2, 3, 4]),
      last_token_pos=jnp.asarray([1, 2, 3, 4]),
      predicted_tokens=jnp.asarray(
          [
              [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
              [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
              [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
              [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
          ],
          dtype=jnp.int32,
      ),
      cache={},
      rng=jax.random.PRNGKey(0),
      full_attention_mask=first_turn_mask,
      init_cache_length=jnp.asarray(input.length_with_mm - 1),
  )
  masked_full_attention_mask = (
      _sampler_loop._mask_full_attention_mask_prefix_for_next_turn(
          full_attention_mask=last_state.full_attention_mask,
          predicted_tokens=last_state.predicted_tokens,
          init_cache_length=last_state.init_cache_length,
      )
  )
  np.testing.assert_array_equal(
      masked_full_attention_mask,
      [
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      ],
  )
  last_state = dataclasses.replace(
      last_state, full_attention_mask=masked_full_attention_mask
  )

  second_turn_mask = _prefill._make_full_attention_mask(
      input=input,
      prev_turns=_turn_utils.PrevTurns(last_state=last_state),
      cache_length=20,
  )
  np.testing.assert_array_equal(
      second_turn_mask,
      [
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
      ],
  )
