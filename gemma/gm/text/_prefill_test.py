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


def test_local_window_compaction_uses_valid_tokens_and_positions():
  batch_size = 2
  window = 4
  prefill_len = 8
  persistent_layer = {
      'k': jnp.zeros((batch_size, window, 1, 1), dtype=jnp.float32),
      'v': jnp.zeros((batch_size, window, 1, 1), dtype=jnp.float32),
      'positions': jnp.full((batch_size, window), -(10**9), dtype=jnp.int32),
      'logical_index': jnp.full((batch_size, window), -1, dtype=jnp.int32),
      'valid': jnp.zeros((batch_size, window), dtype=jnp.bool_),
      'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
  }
  values = jnp.broadcast_to(
      jnp.arange(prefill_len, dtype=jnp.float32)[None, :, None, None],
      (batch_size, prefill_len, 1, 1),
  )
  prefill_layer = {
      'k': values,
      'v': values + 100,
      # Row 1 models a short padded prompt sharing a long bucket with row 0:
      # only logical slots 0, 1, 2 are real, and their positions are still
      # live local context.
      'positions': jnp.asarray(
          [
              [0, 1, 2, 3, 4, 5, 6, 7],
              [0, 1, 2, 2, 2, 2, 2, 2],
          ],
          dtype=jnp.int32,
      ),
      'end_index': jnp.asarray([prefill_len, prefill_len], dtype=jnp.int32),
  }
  logical_valid_mask = jnp.asarray(
      [
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 0, 0, 0, 0, 0],
      ],
      dtype=jnp.bool_,
  )

  compact = _prefill._compact_local_window_layer(
      persistent_layer=persistent_layer,
      prefill_layer=prefill_layer,
      logical_valid_mask=logical_valid_mask,
  )

  np.testing.assert_array_equal(
      compact['logical_index'],
      [
          [4, 5, 6, 7],
          [0, 1, 2, -1],
      ],
  )
  np.testing.assert_array_equal(
      compact['positions'],
      [
          [4, 5, 6, 7],
          [0, 1, 2, -(10**9)],
      ],
  )
  np.testing.assert_array_equal(
      compact['valid'],
      [
          [1, 1, 1, 1],
          [1, 1, 1, 0],
      ],
  )
  np.testing.assert_array_equal(
      compact['k'][..., 0, 0],
      [
          [4, 5, 6, 7],
          [0, 1, 2, 0],
      ],
  )


def test_local_window_prefill_scratch_restages_previous_cache():
  persistent_layer = {
      'k': jnp.asarray([[[[10]], [[11]], [[12]], [[13]]]], dtype=jnp.float32),
      'v': jnp.asarray([[[[20]], [[21]], [[22]], [[23]]]], dtype=jnp.float32),
      'positions': jnp.asarray([[2, 3, -(10**9), 1]], dtype=jnp.int32),
      'logical_index': jnp.asarray([[6, 7, -1, 5]], dtype=jnp.int32),
      'valid': jnp.asarray([[1, 1, 0, 1]], dtype=jnp.bool_),
      'end_index': jnp.asarray([8], dtype=jnp.int32),
  }

  scratch = _prefill._make_local_window_prefill_scratch_layer(
      persistent_layer=persistent_layer,
      prefill_cache_length=10,
  )

  assert 'logical_index' not in scratch
  assert 'valid' not in scratch
  np.testing.assert_array_equal(scratch['end_index'], [8])
  np.testing.assert_array_equal(
      scratch['k'][0, :, 0, 0],
      [0, 0, 0, 0, 0, 13, 10, 11, 0, 0],
  )
  np.testing.assert_array_equal(
      scratch['v'][0, :, 0, 0],
      [0, 0, 0, 0, 0, 23, 20, 21, 0, 0],
  )
  np.testing.assert_array_equal(
      scratch['positions'][0],
      [-(10**9), -(10**9), -(10**9), -(10**9), -(10**9),
       1, 2, 3, -(10**9), -(10**9)],
  )
