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

from gemma import gm
from gemma.gm.text import _sampler
from gemma.gm.text import _sampler_loop
from gemma.gm.text import _tokenizer
import jax
import jax.numpy as jnp
import numpy as np
import pytest


class DummyGemmaWrapper(gm.testing.DummyGemma):

  def __call__(
      self,
      tokens,
      positions=None,
      cache=None,
      attention_mask=None,
      images=None,
      audio=None,
      audio_lengths=None,
      audio_soft_token_counts=None,
      return_last_only=None,
  ):
    return super().__call__(
        tokens,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
        images=images,
    )


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
  model = DummyGemmaWrapper()
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


def test_normalize_token_error_message_contains_token_value():
  """_normalize_token should interpolate the token value in the error message.

  Regression test for a missing f-string prefix that caused the error message
  to show the literal string '{token!r}' instead of the actual token value.
  """
  tokenizer = gm.testing.DummyTokenizer()
  # 'Hello world' encodes to two tokens (the dummy tokenizer splits on spaces),
  # so _normalize_token should raise ValueError for it.
  with pytest.raises(ValueError, match=r'Hello world'):
    _sampler._normalize_token(tokenizer, 'Hello world')


def test_sampler_gemma2_tokenizer_no_begin_of_tool_response():
  """Sampler with Gemma2 tokenizer must not crash on BEGIN_OF_TOOL_RESPONSE.

  Gemma2's _Gemma2SpecialTokens does not define BEGIN_OF_TOOL_RESPONSE (that
  attribute was introduced in Gemma3). The Sampler.sample() method previously
  accessed it unconditionally, raising AttributeError for any Gemma2 model.
  This test verifies the hasattr() guard prevents the crash.
  """
  # DummyTokenizer uses _Gemma3SpecialTokens which has the attribute.
  # We verify the guard logic directly: _Gemma2SpecialTokens must NOT have it.
  assert not hasattr(
      _tokenizer._Gemma2SpecialTokens, 'BEGIN_OF_TOOL_RESPONSE'
  ), (
      '_Gemma2SpecialTokens should not define BEGIN_OF_TOOL_RESPONSE'
  )
  # And _Gemma3SpecialTokens MUST have it.
  assert hasattr(
      _tokenizer._Gemma3SpecialTokens, 'BEGIN_OF_TOOL_RESPONSE'
  ), (
      '_Gemma3SpecialTokens should define BEGIN_OF_TOOL_RESPONSE'
  )
