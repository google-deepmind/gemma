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


def test_gemma3_custom_tokens():
  tokenizer = gm.text.Gemma3Tokenizer(
      custom_tokens={0: '<custom_a>', 1: '<custom_b>'}
  )
  assert tokenizer.tokens[6] == '<custom_a>'
  assert tokenizer.tokens[7] == '<custom_b>'
  assert tokenizer.encode('<custom_b>') == [7]


def test_gemma3n_custom_tokens():
  # In the Gemma3n vocab, ids 7-104 are `<ctrlXX>` and `<unused1>` onwards
  # start at 256001.
  tokenizer = gm.text.Gemma3nTokenizer(
      custom_tokens={0: '<custom_a>', 1: '<custom_b>', 98: '<custom_c>'}
  )
  assert tokenizer.tokens[6] == '<custom_a>'
  assert tokenizer.tokens[256001] == '<custom_b>'
  assert tokenizer.tokens[256098] == '<custom_c>'
  assert tokenizer.encode('<custom_b>') == [256001]


def test_gemma4_custom_tokens():
  # In the Gemma4 vocab, `<unused98>` is at 256010 (ids 50-104 have gaps).
  tokenizer = gm.text.Gemma4Tokenizer(
      custom_tokens={1: '<custom_a>', 98: '<custom_c>'}
  )
  assert tokenizer.tokens[7] == '<custom_a>'
  assert tokenizer.tokens[256010] == '<custom_c>'
  assert tokenizer.encode('<custom_c>') == [256010]
