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

import pickle

from gemma import gm

# Activate the fixture
use_hermetic_tokenizer = gm.testing.use_hermetic_tokenizer


def test_pickle():
  tokenizer = gm.text.Gemma3Tokenizer()
  tokenizer.encode("Hello world!")  # Trigger the lazy-loading of the tokenizer.

  pickle.dumps(tokenizer)


def test_overwrite_tokens():
  fake_bos = "<fake-bos-token>"
  fake_eos = "<fake-eos-token>"
  tokenizer = gm.text.Gemma3Tokenizer(
      overwrite_tokens={
          gm.text.Gemma3Tokenizer.special_tokens.BOS: fake_bos,
          gm.text.Gemma3Tokenizer.special_tokens.EOS: fake_eos,
      }
  )
  token_ids = tokenizer.encode(
      "This should have a weird bos/eos token.",
      add_bos=True,
      add_eos=True,
  )

  assert token_ids[0] == gm.text.Gemma3Tokenizer.special_tokens.BOS
  assert token_ids[-1] == gm.text.Gemma3Tokenizer.special_tokens.EOS

  # Make sure they map to the weird ones
  assert tokenizer.tokens[token_ids[0]] == fake_bos
  assert tokenizer.tokens[token_ids[-1]] == fake_eos
