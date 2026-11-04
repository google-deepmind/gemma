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
      model=model,  # pyrefly: ignore[bad-argument-type]
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
          [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
      ],
  )


def test_prefill_multi_turn():
  tokenizer = gm.testing.DummyTokenizer()
  model = gm.testing.DummyGemma()

  text = jnp.asarray(tokenizer.encode('hello world'))[None, ...]
  params = model.init(jax.random.PRNGKey(0), tokens=text)['params']

  input_config = _types.InputConfig(
      support_images=False,
      num_tokens_per_image=100,
      special_tokens=tokenizer.special_tokens,
  )
  turn1_input = _types.Input(text=text, images=None, config=input_config)
  assert turn1_input.length_with_mm == 2

  init_state1 = _prefill.prefill(
      model=model,  # pyrefly: ignore[bad-argument-type]
      params=params,
      input=turn1_input,
      last_state=None,
      cache_length=64,
      max_out_length=12,
      pad_length=(8,),
      rng=jax.random.PRNGKey(0),
      sharding=None,
  )
  loop = _sampler_loop.SamplerLoop(
      model=model,  # pyrefly: ignore[bad-argument-type]
      end_tokens=(),
      forbidden_tokens=None,
      sampling=gm.text.Greedy(),
      cache_length=64,
      special_tokens=tokenizer.special_tokens,
  )
  state1 = loop.sample(
      params=params,
      init_state=init_state1,
      max_new_tokens=jnp.asarray(3),
  )
  assert isinstance(state1, _sampler_loop.SamplingState)
  # 2-token prompt + 3 decode steps (the first re-writes the last prompt token) = 4 slots.
  np.testing.assert_array_equal(state1.used_cache_length, 4)
  np.testing.assert_array_equal(state1.cache['layer_0']['end_index'], [4])

  turn2_input = _types.Input(text=text, images=None, config=input_config)
  prev_turns = _turn_utils.PrevTurns(last_state=state1)
  np.testing.assert_equal(prev_turns.used_cache_length, 4)

  full_cache = _prefill._get_or_init_cache(
      inputs=turn2_input,
      prev_turns=prev_turns,
      model=model,  # pyrefly: ignore[bad-argument-type]
      params=params,
      cache_length=64,
      sharding=None,
  )
  prefill_input2 = _prefill._make_prefill_input(
      input=turn2_input,
      cache=full_cache,
      prev_turns=prev_turns,
      pad_lengths=(8,),
  )
  # Turn 2 prompt tokens are written at cache indices 4 and 5, so row 0
  # attends to indices 0..4 (5 slots) and row 1 attends to indices 0..5 (6 slots).
  np.testing.assert_array_equal(
      prefill_input2.attention_mask[0, :2, :8],
      [
          [1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0],
      ],
  )

  init_state2 = _prefill.prefill(
      model=model,  # pyrefly: ignore[bad-argument-type]
      params=params,
      input=turn2_input,
      last_state=state1,
      cache_length=64,
      max_out_length=12,
      pad_length=(8,),
      rng=jax.random.PRNGKey(0),
      sharding=None,
  )
  # Turn 2 prefill wrote indices 4 and 5; sampling restarts from index 5.
  np.testing.assert_array_equal(init_state2.init_cache_length, 5)
  np.testing.assert_array_equal(init_state2.used_cache_length, 5)
  np.testing.assert_array_equal(init_state2.cache['layer_0']['end_index'], [5])
  np.testing.assert_array_equal(
      init_state2.attention_mask_for_step,
      [[1] * 6 + [0] * 58],
  )
