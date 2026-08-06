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

import pickle

from gemma import gm

# Activate the fixture
use_hermetic_tokenizer = gm.testing.use_hermetic_tokenizer


def test_pickle():
  tokenizer = gm.text.Gemma3Tokenizer()
  tokenizer.encode('Hello world!')  # Trigger the lazy-loading of the tokenizer.

  pickle.dumps(tokenizer)


def test_gemma4_tokenizer_forbids_multimodal_placeholder_tokens():
  """Regression test for https://github.com/google-deepmind/gemma/issues/613.

  Gemma 4 introduced distinct token ids for image and audio multimodal
  placeholders. The tokenizer must mark all six as forbidden so the sampler
  cannot generate raw placeholder tokens during text-only inference (which
  would corrupt the output).
  """
  forbidden = gm.text.Gemma4Tokenizer.FORBIDDEN_TOKENS
  st = gm.text.Gemma4Tokenizer.special_tokens
  for token in (
      st.IMAGE_PLACEHOLDER,
      st.START_OF_IMAGE,
      st.END_OF_IMAGE,
      st.AUDIO_PLACEHOLDER,
      st.START_OF_AUDIO,
      st.END_OF_AUDIO,
  ):
    assert token in forbidden, (
        f'Token {token!r} ({st(token).name}) must be in Gemma4Tokenizer'
        f'.FORBIDDEN_TOKENS, but FORBIDDEN_TOKENS is {forbidden!r}'
    )
