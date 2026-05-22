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

from unittest import mock
from gemma import gm
from gemma.gm.text import _sampler
from gemma.gm.text import _sampler_loop
import jax
import jax.numpy as jnp
import numpy as np


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
  out = _sampler_loop._mask_tokens_after_end_tokens(tokens, end_tokens=(10, 11))
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
      pad_length=None,
  )
  sampler.sample('Hello world')


def test_chat_sampler_gemma4_dispatch():
  """Tests that _sample() dispatches to gemma4_sampler when _is_gemma4 is True.

  Uses mocks to verify the dispatch logic without requiring a full Gemma4
  model. This catches regressions in the _sample() method that could break
  the Gemma4 path.
  """
  model = gm.testing.DummyGemma()
  params = model.init(
      jax.random.PRNGKey(0),
      jnp.zeros((5,), dtype=jnp.int32),
  )
  params = params['params']
  tokenizer = gm.testing.DummyTokenizer()
  chat_sampler = gm.text.ChatSampler(
      model=model,
      params=params,
      tokenizer=tokenizer,
      cache_length=128,
      max_out_length=128,
  )

  # Force the Gemma4 dispatch path.
  mock_sample = mock.MagicMock(
      return_value=_sampler.SamplerOutput(
          text='mock output',
          state=mock.MagicMock(),
      )
  )
  with mock.patch.object(
      type(chat_sampler),
      '_is_gemma4',
      new_callable=lambda: property(lambda self: True),
  ):
    with mock.patch.object(
        type(chat_sampler),
        'gemma4_sampler',
        new_callable=lambda: property(
            lambda self: mock.MagicMock(sample=mock_sample)
        ),
    ):
      output = chat_sampler.chat('Hello world')
      assert isinstance(output, str)
  # Verify gemma4_sampler.sample was called (not sampler.sample).
  mock_sample.assert_called_once()


def test_chat_sampler_non_gemma4_dispatch():
  """Tests that _sample() dispatches to sampler when _is_gemma4 is False.

  Uses mocks to verify the dispatch logic without exercising the full sampling
  pipeline (which is already covered by test_sampler). This catches regressions
  in _sample() that could break the non-Gemma4 dispatch path.
  """
  model = gm.testing.DummyGemma()
  params = model.init(
      jax.random.PRNGKey(0),
      jnp.zeros((5,), dtype=jnp.int32),
  )
  params = params['params']
  tokenizer = gm.testing.DummyTokenizer()
  chat_sampler = gm.text.ChatSampler(
      model=model,
      params=params,
      tokenizer=tokenizer,
      cache_length=128,
      max_out_length=128,
  )

  assert not chat_sampler._is_gemma4  # Confirm non-Gemma4 dispatch path.

  mock_sample = mock.MagicMock(
      return_value=_sampler.SamplerOutput(
          text='mock output',
          state=mock.MagicMock(),
      )
  )
  with mock.patch.object(
      type(chat_sampler),
      'sampler',
      new_callable=lambda: property(
          lambda self: mock.MagicMock(sample=mock_sample)
      ),
  ):
    output = chat_sampler.chat('Hello world')
    assert isinstance(output, str)
  # Verify sampler.sample was called (not gemma4_sampler.sample).
  mock_sample.assert_called_once()
